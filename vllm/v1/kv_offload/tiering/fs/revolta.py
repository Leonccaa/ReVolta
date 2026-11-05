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
The attribute separately records the full offload key and SHA-256 namespace
identity, compared exactly on load. CRC32C covers only the payload. A page is
published by writing a
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
import queue
import struct
import threading
import time
import weakref
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, get_offload_group_idx
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.identity import (
    PAGE_FORMAT,
    canonical_digest,
    disable_revolta_root,
    get_revolta_identity,
)
from vllm.v1.kv_offload.tiering.fs.io import O_DIRECT, _ensure_dirs, _get_tmp_suffix
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

CHECKSUM_XATTR = "user.revolta.kv_crc32c"
# Version, payload CRC32C, full SHA-256 namespace digest, key length; then key.
_XATTR_FORMAT = struct.Struct("<II32sI")
_XATTR_VERSION = 2


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


def read_page_identity(path_or_fd: str | int) -> tuple[int, bytes, bytes] | None:
    """Return (payload CRC32C, full namespace digest, full OffloadKey)."""
    try:
        raw = os.getxattr(path_or_fd, CHECKSUM_XATTR)
    except OSError as exc:
        if exc.errno in (errno.ENODATA, errno.ENOTSUP, errno.ENOENT):
            return None
        raise
    if len(raw) < _XATTR_FORMAT.size:
        return None
    version, crc, identity, key_len = _XATTR_FORMAT.unpack_from(raw)
    key = raw[_XATTR_FORMAT.size :]
    if version != _XATTR_VERSION or key_len != len(key) or key_len <= 4:
        return None
    return crc, identity, key


def store_verified_block(
    dest_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    key: OffloadKey,
    identity: bytes,
) -> bool:
    """Publish one page with its payload checksum and full identity.

    Returns False if a page with a valid attribute already exists (it is not
    rewritten; its payload is verified on load). A file without one is a
    partial or foreign page and is replaced.
    """
    existing = read_page_identity(dest_path)
    if existing is not None and existing[1:] == (identity, bytes(key)):
        return False
    payload = view.cast("B")[offset : offset + block_size]
    value = crc32c(payload)
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
            os.setxattr(
                fd,
                CHECKSUM_XATTR,
                _XATTR_FORMAT.pack(_XATTR_VERSION, value, identity, len(key))
                + bytes(key),
            )
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
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    key: OffloadKey,
    identity: bytes,
) -> None:
    """Read one page into ``view`` and delete confirmed verification failures.

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
        stored = read_page_identity(fd)
        if stored is None:
            raise ChecksumError(source_path, "missing or malformed checksum")
        if stored[1:] != (identity, bytes(key)):
            raise ChecksumError(source_path, "page identity mismatch")
        bytes_read = os.readv(fd, [payload])
        if bytes_read < block_size:
            raise ChecksumError(source_path, f"short read {bytes_read}")
        if crc32c(payload) != stored[0]:
            raise ChecksumError(source_path, "checksum mismatch")
    except ChecksumError:
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


class RevoltaFileSystemTierManager(FileSystemTierManager):
    """File-system secondary tier for persistent, verified KV reuse.

    Identity: the directory hash binds the page format, the engine's config
    identity and the digest of a content manifest of the loaded checkpoint files
    (see content_manifest.py). The tier fails closed: without a manifest,
    with a manifest that does not match the loaded files, or when the namespace's
    recorded run config differs, it
    disables itself: lookups miss, stores complete without writing, nothing
    on disk is read or changed. Inference and the CPU tier are unaffected.

    Extra tier options (documented as ``ssd.*`` in docs/kv-offload.md):
        content_manifest_path: path of the manifest file.
        content_verify: "stat" (default) checks size/mtime/inode of every
            file; "full" also recomputes every content hash at startup.
        mtime_update_interval_s: minimum seconds between mtime updates of a
            page (default 60; 0 = every use).

    Recency: CPU/GPU-tier hits never read the disk tier, so without help its
    LRU would only see writes, and a hot shared chain head (written first)
    would be evicted first, orphaning every chain behind it. ``touch`` moves
    the pages of a used prefix to the recent end in reverse order, so the
    head is the most recent, and refreshes their mtimes on a background
    thread (a due page refreshes the whole touched chain). The cache rebuilds its order
    after a restart. Within one store job, once a page of a group fails, the
    later pages of that group are skipped, so the persisted chain stays
    contiguous.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        content_manifest_path: str | None = None,
        content_verify: str = "stat",
        mtime_update_interval_s: float = 60.0,
        **kwargs,
    ):
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "pages_written": 0,
            "pages_existing": 0,
            "pages_verified": 0,
            "checksum_failures": 0,
            "skipped_after_chain_failure": 0,
            "mtime_updates": 0,
        }
        self.disabled_reason: str | None = None
        self._root_dir = os.path.realpath(root_dir)
        if mtime_update_interval_s < 0:
            raise ValueError("mtime_update_interval_s must be >= 0")
        self._mtime_interval = mtime_update_interval_s
        self._mtime_lock = threading.Lock()
        self._mtime_at: dict[str, float] = {}
        self._mtime_queue: queue.SimpleQueue[str | threading.Event | None] = (
            queue.SimpleQueue()
        )
        self._mtime_thread: threading.Thread | None = None
        self._disabled_finished: list[int] = []
        if content_verify not in ("stat", "full"):
            raise ValueError("content_verify must be 'stat' or 'full'")
        identity = get_revolta_identity(
            offloading_spec.vllm_config, root_dir, content_manifest_path, content_verify
        )
        if identity is None:
            self._init_disabled(offloading_spec, primary_kv_view, tier_type)
            return
        # Include the actual physical page description, not just compile hashes.
        assert primary_kv_view.shape is not None
        try:
            self._extra_layout = {
                "revolta_fs_format": PAGE_FORMAT,
                "content_digest": identity.content_digest,
                "persistent_identity_digest": identity.persistent_digest,
                "page_layout": canonical_digest(
                    {
                        "groups": [
                            {
                                "layers": list(group.layer_names),
                                "spec": group.kv_cache_spec,
                            }
                            for group in offloading_spec.kv_cache_config.kv_cache_groups
                        ],
                        "tensors": offloading_spec.kv_cache_config.kv_cache_tensors,
                        "primary_shape": primary_kv_view.shape[1:],
                        "primary_strides": primary_kv_view.strides,
                        "primary_format": primary_kv_view.format,
                        "block_size_factor": offloading_spec.block_size_factor,
                        "layout": getattr(offloading_spec, "persistent_layout", None),
                    }
                ),
            }
            self._extra_layout["persistent_identity_digest"] = canonical_digest(
                self._extra_layout
            )
        except (ValueError, TypeError) as exc:
            self._init_disabled(offloading_spec, primary_kv_view, tier_type, str(exc))
            return
        mapper = self._make_file_mapper(os.path.realpath(root_dir), offloading_spec)
        conflict = self._run_config_conflict(mapper)
        if conflict is not None:
            self._init_disabled(
                offloading_spec, primary_kv_view, tier_type, reason=conflict
            )
            return
        try:
            super().__init__(
                offloading_spec, primary_kv_view, tier_type, root_dir, **kwargs
            )
            self._check_xattr_support()
        except OSError as exc:
            self.shutdown()
            self._init_disabled(offloading_spec, primary_kv_view, tier_type, str(exc))
            return
        self._mtime_thread = threading.Thread(
            target=self._mtime_worker, name="revolta_fs_mtime", daemon=True
        )
        self._mtime_thread.start()
        namespace = json.dumps(
            self.file_mapper.get_run_config(), sort_keys=True, separators=(",", ":")
        )
        self._identity_digest = hashlib.sha256(namespace.encode()).digest()
        config = offloading_spec.vllm_config
        managers = getattr(config, "_revolta_fs_managers", None)
        if managers is None:
            managers = weakref.WeakSet()
            config.__dict__["_revolta_fs_managers"] = managers
        managers.add(self)
        logger.info(
            "revolta_fs enabled at %s (content %s)",
            os.path.dirname(self.file_mapper.get_config_file_path()),
            identity.content_digest[:16],
        )

    # --- identity ---------------------------------------------------------

    def _init_disabled(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        reason: str | None = None,
    ) -> None:
        SecondaryTierManager.__init__(self, offloading_spec, primary_kv_view, tier_type)
        self.disabled_reason = reason or "content identity not verified"
        disable_revolta_root(
            offloading_spec.vllm_config, self._root_dir, self.disabled_reason
        )
        self._closed = True
        if reason is not None:
            logger.error("revolta_fs disabled: %s", reason)

    @property
    def disabled(self) -> bool:
        return self.disabled_reason is not None

    def _make_file_mapper(self, root_dir: str, offloading_spec: "OffloadingSpec"):
        from vllm.v1.kv_offload.file_mapper import FileMapper

        mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
            extra_layout=self._extra_layout,
        )

        # Use a 128-bit directory prefix; the xattr still binds the full digest.
        digest = canonical_digest(mapper.get_run_config())
        model = os.path.basename(offloading_spec.vllm_config.model_config.model)
        mapper.base_path = f"{root_dir}/{model}_{digest[:32]}"
        return mapper

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
        except (OSError, ValueError) as exc:
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
            raise OSError(
                exc.errno, f"revolta_fs needs user extended attributes on {base}: {exc}"
            ) from exc
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(probe)

    def _count(self, name: str, n: int = 1) -> None:
        with self._stats_lock:
            self._stats[name] += n

    def get_revolta_stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def _store_key(self, key: OffloadKey, path: str, block_id: int) -> None:
        self._write_page(key, path, block_id)

    def _store_group(self, pages: list[tuple[OffloadKey, int]]) -> None:
        # Input order is prefix-chain order, not lexicographic hash order.
        for index, (key, block_id) in enumerate(pages):
            try:
                self._store_key(key, self.file_mapper.get_file_name(key), block_id)
            except Exception:
                self._count("skipped_after_chain_failure", len(pages) - index - 1)
                raise

    def _write_page(self, key: OffloadKey, path: str, block_id: int) -> None:
        with self._cache.write(path, self._block_size):
            written = store_verified_block(
                path,
                self._primary_kv_view,
                block_id * self._block_size,
                self._block_size,
                key,
                self._identity_digest,
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
                    key,
                    self._identity_digest,
                )
            except ChecksumError as exc:
                self._count("checksum_failures")
                logger.warning("revolta_fs: %s (%s)", exc.reason, path)
                raise
        self._count("pages_verified")
        self._schedule_mtime([path])

    # --- recency ------------------------------------------------------------

    def touch(self, keys, req_context=None) -> None:
        if self.disabled:
            return
        paths = [self.file_mapper.get_file_name(k) for k in reversed(list(keys))]
        self._cache.touch(paths)
        self._schedule_mtime(paths)

    def _schedule_mtime(self, paths: list[str]) -> None:
        now = time.monotonic()
        # One lock covers all producers: scheduler touches and pool loads.
        # A due page refreshes the entire batch, already ordered tail to head.
        with self._mtime_lock:
            if any(
                now - self._mtime_at.get(p, -float("inf")) >= self._mtime_interval
                for p in paths
            ):
                for path in paths:
                    self._mtime_at[path] = now
                    self._mtime_queue.put(path)
            if len(self._mtime_at) > 4 * max(self._cache.max_files, 1):
                cutoff = now - self._mtime_interval
                self._mtime_at = {
                    p: t for p, t in self._mtime_at.items() if t >= cutoff
                }

    def _mtime_worker(self) -> None:
        # File timestamps come from a coarse clock, so pages touched together
        # would tie and restart order would fall back to path order. Stamp
        # explicit, strictly increasing times in queue order instead.
        last_ns = 0
        while True:
            path = self._mtime_queue.get()
            if path is None:
                return
            if isinstance(path, threading.Event):
                path.set()
                continue
            last_ns = max(time.time_ns(), last_ns + 1)
            try:
                os.utime(path, ns=(last_ns, last_ns))
                self._count("mtime_updates")
            except FileNotFoundError:
                pass
            except OSError as exc:  # never fails a load or store
                logger.debug("revolta_fs: mtime update of %s failed: %s", path, exc)

    def flush_mtime_updates(self, timeout: float = 5.0) -> None:
        """Wait until queued mtime updates are applied (tests, shutdown)."""
        if self._mtime_thread is None:
            return
        barrier = threading.Event()
        with self._mtime_lock:
            self._mtime_queue.put(barrier)
        if not barrier.wait(timeout):
            raise TimeoutError("mtime worker did not reach the flush barrier")

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
        groups: dict[int, list[tuple[OffloadKey, int]]] = {}
        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            groups.setdefault(get_offload_group_idx(key), []).append((key, int(bid)))
        tasks = [
            functools.partial(self._store_group, pages) for pages in groups.values()
        ]
        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)

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
        if not getattr(self, "_closed", True):
            self._pool.shutdown(wait=True)
            if self._mtime_thread is not None:
                self._mtime_queue.put(None)
                self._mtime_thread.join()
                self._mtime_thread = None
            self._cache.release()
            self._closed = True
