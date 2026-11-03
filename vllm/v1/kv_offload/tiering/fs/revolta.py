# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReVolta file-system secondary tier (``type: "revolta_fs"``).

A stricter variant of ``fs`` for using the disk tier as a persistent cold
store across engine restarts. It shares the ``fs`` directory layout, hash
seed, budget and thread pool, and only takes effect when configured: plain
``fs`` tiers are unchanged.

Integrity: every page carries a CRC32C of its payload in the extended
attribute ``user.revolta.kv_crc32c`` (the approach of the upstream FS-tier
checksum proposal, which keeps the payload byte-exact and O_DIRECT aligned).
The stored value is XORed with a tag of the page's offload key and the tier's
full layout identity, so an intact page stored for another prefix, group or
layout fails the check like a corrupted one. A page is published by writing a
temporary file, setting the attribute, fsync, rename and a directory fsync.
On load the payload is verified before the load is reported successful; a
page that is short, has no or a malformed attribute, or fails the check is
deleted and reported as a failed load, so the scheduler recomputes it.
"""

import contextlib
import ctypes
import errno
import functools
import glob
import hashlib
import json
import os
import struct
import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.content_manifest import (
    ManifestError,
    load_manifest,
    rehash_and_verify,
    verify_manifest,
)
from vllm.v1.kv_offload.tiering.fs.io import O_DIRECT, _ensure_dirs, _get_tmp_suffix
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

CHECKSUM_XATTR = "user.revolta.kv_crc32c"
# Attribute value: little-endian (format version, tagged CRC32C).
_XATTR_FORMAT = struct.Struct("<II")
_XATTR_VERSION = 1
# On-disk format of this tier, bound into the directory hash.
PAGE_FORMAT = "revolta_fs/page-v1+crc32c-xattr"


# ---------------------------------------------------------------------------
# CRC32C
# ---------------------------------------------------------------------------


def _load_crc32c_extend() -> Callable[[int, memoryview], int]:
    """Return ``extend(crc, view)`` computing CRC32C over a buffer in place.

    google_crc32c only accepts read-only bytes, which would copy every page;
    call its bundled libcrc32c directly (ctypes releases the GIL), and fall
    back to a copy when the library is not found.
    """
    import google_crc32c

    package_dir = os.path.dirname(google_crc32c.__file__)
    candidates = sorted(
        glob.glob(os.path.join(package_dir, "..", "google_crc32c.libs", "libcrc32c*"))
    ) + sorted(glob.glob(os.path.join(package_dir, "libcrc32c*")))
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            fn = lib.crc32c_extend
        except (OSError, AttributeError):
            continue
        fn.restype = ctypes.c_uint32
        fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]

        def extend(crc: int, view: memoryview, fn=fn) -> int:
            n = len(view)
            if n == 0:
                return crc
            if view.readonly:
                return google_crc32c.extend(crc, bytes(view))
            buf = (ctypes.c_char * n).from_buffer(view)
            try:
                return int(fn(crc, ctypes.addressof(buf), n))
            finally:
                del buf

        return extend

    logger.warning("libcrc32c not found; revolta_fs checksums copy each page")
    return lambda crc, view: google_crc32c.extend(crc, bytes(view))


_crc32c_extend: Callable[[int, memoryview], int] | None = None


def crc32c(view: memoryview | bytes, crc: int = 0) -> int:
    global _crc32c_extend
    if isinstance(view, bytes):
        import google_crc32c

        return google_crc32c.extend(crc, view)
    if _crc32c_extend is None:
        _crc32c_extend = _load_crc32c_extend()
    return _crc32c_extend(crc, view)


# ---------------------------------------------------------------------------
# Page I/O
# ---------------------------------------------------------------------------


class ChecksumError(OSError):
    """A page failed integrity verification (missing, malformed or wrong
    checksum). Reported as EBADMSG."""

    def __init__(self, path: str, reason: str):
        super().__init__(errno.EBADMSG, f"KV page failed verification: {reason}", path)
        self.reason = reason


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_tagged_crc(path_or_fd: str | int) -> int | None:
    """Return the tagged CRC stored for a page, or None if it has no valid
    attribute."""
    try:
        raw = os.getxattr(path_or_fd, CHECKSUM_XATTR)
    except OSError as exc:
        if exc.errno in (errno.ENODATA, errno.ENOTSUP, errno.ENOENT):
            return None
        raise
    if len(raw) != _XATTR_FORMAT.size:
        return None
    version, value = _XATTR_FORMAT.unpack(raw)
    if version != _XATTR_VERSION:
        return None
    return value


def store_verified_block(
    dest_path: str, view: memoryview, offset: int, block_size: int, tag: int
) -> bool:
    """Durably publish one page with its tagged checksum.

    Returns False if a page with a valid attribute already exists (it is not
    rewritten; its payload is verified on load). A file without one is a
    partial or foreign page and is replaced.
    """
    if os.path.exists(dest_path) and read_tagged_crc(dest_path) is not None:
        return False
    payload = view.cast("B")[offset : offset + block_size]
    value = crc32c(payload) ^ tag
    tmp_path = dest_path + _get_tmp_suffix()
    _ensure_dirs(dest_path)
    try:
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT,
            0o644,
        )
        try:
            written = os.write(fd, payload)
            if written < len(payload):
                raise OSError(
                    errno.EIO,
                    f"Short write: expected {len(payload)} bytes, wrote {written}",
                )
            os.setxattr(fd, CHECKSUM_XATTR, _XATTR_FORMAT.pack(_XATTR_VERSION, value))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
        _fsync_dir(os.path.dirname(dest_path))
    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        raise
    return True


def load_verified_block(
    source_path: str, view: memoryview, offset: int, block_size: int, tag: int
) -> None:
    """Read one page into ``view`` and verify it; delete it on any failure.

    The destination slice may hold partial data after a failure; the caller
    reports the load as failed so the slot is never marked ready.
    """
    fd: int | None = None
    payload = view.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(source_path, os.O_RDONLY | O_DIRECT)
        file_size = os.fstat(fd).st_size
        if file_size != block_size:
            raise ChecksumError(source_path, f"size {file_size}, expected {block_size}")
        stored = read_tagged_crc(fd)
        if stored is None:
            raise ChecksumError(source_path, "missing or malformed checksum")
        bytes_read = os.readv(fd, [payload])
        if bytes_read < block_size:
            raise ChecksumError(source_path, f"short read {bytes_read}")
        if crc32c(payload) ^ tag != stored:
            raise ChecksumError(source_path, "checksum mismatch")
    except FileNotFoundError:
        raise
    except Exception:
        try:
            os.remove(source_path)
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            logger.warning(
                "Failed to remove unverifiable file %s: %s", source_path, cleanup_exc
            )
        raise
    finally:
        if fd is not None:
            os.close(fd)


# ---------------------------------------------------------------------------
# Tier
# ---------------------------------------------------------------------------


def expected_content_roots(vllm_config) -> dict[str, str]:
    """Role -> path of the checkpoint files the engine loads."""
    roots = {"target": str(vllm_config.model_config.model)}
    spec = getattr(vllm_config, "speculative_config", None)
    draft = getattr(spec, "model", None) if spec is not None else None
    if (
        isinstance(draft, str)
        and draft
        and os.path.exists(draft)
        and os.path.realpath(draft) != os.path.realpath(roots["target"])
    ):
        roots["draft"] = draft
    return roots


class RevoltaFileSystemTierManager(FileSystemTierManager):
    """File-system secondary tier for persistent, verified KV reuse.

    Identity: the directory hash binds the page format, the engine's config
    hash and the digest of a content manifest of the loaded checkpoint files
    (see content_manifest.py). The tier fails closed: without a manifest
    (``require_content_manifest``), with a manifest that does not match the
    loaded files, or when the namespace's recorded run config differs, it
    disables itself: lookups miss, stores complete without writing, nothing
    on disk is read or changed. Inference and the CPU tier are unaffected.

    Extra tier options:
        content_manifest: path of the manifest file.
        require_content_manifest: disable the tier without one (default True).
        verify_content: "stat" (default) checks size/mtime/inode of every
            file; "full" also recomputes every content hash at startup.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        content_manifest: str | None = None,
        require_content_manifest: bool = True,
        verify_content: str = "stat",
        **kwargs,
    ):
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "pages_written": 0,
            "pages_existing": 0,
            "pages_verified": 0,
            "checksum_failures": 0,
        }
        self.disabled_reason: str | None = None
        self._disabled_finished: list[int] = []
        if verify_content not in ("stat", "full"):
            raise ValueError("verify_content must be 'stat' or 'full'")
        content_digest = self._content_identity(
            offloading_spec,
            content_manifest,
            require_content_manifest,
            verify_content,
        )
        if content_digest is None:
            self._init_disabled(offloading_spec, primary_kv_view, tier_type)
            return
        try:
            config_hash = str(offloading_spec.vllm_config.compute_hash())
        except Exception as exc:  # noqa: BLE001 - identity must not guess
            self._init_disabled(
                offloading_spec,
                primary_kv_view,
                tier_type,
                reason=f"cannot hash the engine config: {exc}",
            )
            return
        self._extra_layout = {
            "revolta_fs_format": PAGE_FORMAT,
            "content_digest": content_digest,
            "vllm_config_hash": config_hash,
        }
        mapper = self._make_file_mapper(os.path.realpath(root_dir), offloading_spec)
        conflict = self._run_config_conflict(mapper)
        if conflict is not None:
            self._init_disabled(
                offloading_spec, primary_kv_view, tier_type, reason=conflict
            )
            return
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, root_dir, **kwargs
        )
        self._check_xattr_support()
        identity = json.dumps(
            self.file_mapper.get_run_config(), sort_keys=True, separators=(",", ":")
        )
        self._identity_digest = hashlib.sha256(identity.encode()).digest()
        logger.info(
            "revolta_fs enabled at %s (content %s)",
            os.path.dirname(self.file_mapper.get_config_file_path()),
            content_digest[:16],
        )

    # --- identity ---------------------------------------------------------

    @staticmethod
    def _content_identity(
        offloading_spec: "OffloadingSpec",
        manifest_path: str | None,
        required: bool,
        verify_content: str,
    ) -> str | None:
        """Digest of the verified content manifest; None disables the tier."""
        if manifest_path is None:
            if required:
                logger.error(
                    "revolta_fs disabled: no content_manifest configured "
                    "(require_content_manifest=true)"
                )
                return None
            logger.warning(
                "revolta_fs: no content manifest; persisted KV is bound to "
                "paths and config only, not to the checkpoint contents"
            )
            return "unverified"
        try:
            manifest = load_manifest(manifest_path)
            roots = expected_content_roots(offloading_spec.vllm_config)
            digest = verify_manifest(manifest, roots)
            if verify_content == "full":
                digest = rehash_and_verify(manifest)
        except ManifestError as exc:
            logger.error("revolta_fs disabled: %s", exc)
            return None
        return digest

    def _init_disabled(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        reason: str | None = None,
    ) -> None:
        SecondaryTierManager.__init__(self, offloading_spec, primary_kv_view, tier_type)
        self.disabled_reason = reason or "content identity not verified"
        self._closed = True
        if reason is not None:
            logger.error("revolta_fs disabled: %s", reason)

    @property
    def disabled(self) -> bool:
        return self.disabled_reason is not None

    def _make_file_mapper(self, root_dir: str, offloading_spec: "OffloadingSpec"):
        from vllm.v1.kv_offload.file_mapper import FileMapper

        return FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
            extra_layout=self._extra_layout,
        )

    @staticmethod
    def _run_config_conflict(mapper) -> str | None:
        """Reason the namespace directory belongs to another identity, if so.

        The directory name holds only a hash prefix; the recorded run config
        must match the full identity."""
        config_path = mapper.get_config_file_path()
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path) as f:
                recorded = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            return f"unreadable {config_path}: {exc}"
        if recorded != json.loads(json.dumps(mapper.get_run_config())):
            return (
                f"{config_path} records another identity under the same "
                "directory hash prefix"
            )
        return None

    def _write_run_config(self) -> None:
        """Write the namespace's run config atomically if it is missing."""
        config_path = self.file_mapper.get_config_file_path()
        expected = self.file_mapper.get_run_config()
        if os.path.exists(config_path):
            return
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        tmp = config_path + _get_tmp_suffix()
        with open(tmp, "w") as f:
            json.dump(expected, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, config_path)
        _fsync_dir(os.path.dirname(config_path))

    def _check_xattr_support(self) -> None:
        """Fail at startup if the root cannot hold checksum attributes: a
        silent fallback would leave pages unverified."""
        base = os.path.dirname(self.file_mapper.get_config_file_path())
        probe = os.path.join(base, f".xattr_probe{_get_tmp_suffix()}")
        try:
            fd = os.open(probe, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            try:
                os.setxattr(fd, CHECKSUM_XATTR, b"\0" * _XATTR_FORMAT.size)
            finally:
                os.close(fd)
        except OSError as exc:
            self.shutdown()
            raise RuntimeError(
                f"revolta_fs needs user extended attributes on {base}: {exc}"
            ) from exc
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(probe)

    def _tag(self, key: OffloadKey) -> int:
        return crc32c(self._identity_digest + bytes(key))

    def _count(self, name: str, n: int = 1) -> None:
        with self._stats_lock:
            self._stats[name] += n

    def get_revolta_stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def _store_key(self, key: OffloadKey, path: str, block_id: int) -> None:
        with self._cache.write(path, self._block_size):
            written = store_verified_block(
                path,
                self._primary_kv_view,
                block_id * self._block_size,
                self._block_size,
                self._tag(key),
            )
        self._count("pages_written" if written else "pages_existing")

    def _load_key(self, key: OffloadKey, path: str, block_id: int) -> None:
        with self._cache.read(path):
            try:
                load_verified_block(
                    path,
                    self._primary_kv_view,
                    block_id * self._block_size,
                    self._block_size,
                    self._tag(key),
                )
            except ChecksumError as exc:
                self._count("checksum_failures")
                logger.warning("revolta_fs: %s (%s)", exc.reason, path)
                raise
        self._count("pages_verified")

    def lookup(self, key: OffloadKey, req_context=None) -> bool | None:
        if self.disabled:
            return False
        return super().lookup(key, req_context)

    def get_finished_jobs(self) -> Iterable[JobResult]:
        if self.disabled:
            finished, self._disabled_finished = self._disabled_finished, []
            return (JobResult(job_id=j, success=True) for j in finished)
        return super().get_finished_jobs()

    def submit_store(self, job_metadata: JobMetadata) -> None:
        if self.disabled:
            # Nothing is written; completing the job releases the CPU pins.
            self._disabled_finished.append(job_metadata.job_id)
            return
        tasks = (
            functools.partial(
                self._store_key, key, self.file_mapper.get_file_name(key), int(bid)
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        if self.disabled:
            raise RuntimeError("revolta_fs is disabled; lookups never hit it")
        tasks = (
            functools.partial(
                self._load_key, key, self.file_mapper.get_file_name(key), int(bid)
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    def shutdown(self) -> None:
        if self.disabled:
            return
        if not getattr(self, "_closed", True):
            logger.info("revolta_fs stats: %s", self.get_revolta_stats())
        super().shutdown()
