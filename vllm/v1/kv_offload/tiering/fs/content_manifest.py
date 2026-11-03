# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content manifests: the identity of the checkpoint files a KV cache was
computed with.

A path, a config or a weight index does not identify weights: a shard can be
replaced in place under the same name. A manifest records, per role (target
model, draft model, KV scale overlays, ...), every file under the role's root
with its size, content hash and the stat fields that change when the file is
rewritten. It is produced once, when a checkpoint is prepared
(``tools/kv_offload/make_content_manifest.py``). At startup the engine checks
the stat fields against the files it loads; any difference means the content
may have changed, and the caller must not reuse persisted KV.

The manifest digest covers role, relative path, size and content hash only, so
moving or re-mounting identical files keeps the identity.
"""

import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor

MANIFEST_VERSION = 1


class ManifestError(ValueError):
    """The manifest is missing, malformed, or does not match the files."""


def _iter_files(root: str) -> Iterator[str]:
    """Relative paths of all regular files under ``root`` (following symlinks),
    sorted. A single file root yields its own base name."""
    if os.path.isfile(root):
        yield os.path.basename(root)
        return
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.isfile(path):
                found.append(os.path.relpath(path, root))
    yield from sorted(found)


def _file_path(root: str, rel: str) -> str:
    return root if os.path.isfile(root) else os.path.join(root, rel)


def _stat_fields(path: str) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino}


def _hasher() -> tuple[str, Callable[[], "hashlib._Hash"]]:
    try:
        import blake3

        return "blake3", lambda: blake3.blake3(max_threads=blake3.blake3.AUTO)
    except ImportError:
        return "sha256", hashlib.sha256


def _hash_file(path: str, factory, chunk: int = 64 << 20) -> str:
    h = factory()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def manifest_digest(roles: Mapping[str, Mapping]) -> str:
    """Identity digest over (role, path, size, content hash)."""
    canonical = [
        [role, rel, entry["size"], entry["hash"]]
        for role in sorted(roles)
        for rel, entry in sorted(roles[role]["files"].items())
    ]
    data = json.dumps(canonical, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def build_manifest(roles: Mapping[str, str], workers: int = 4) -> dict:
    """Hash every file under each role root. ``roles`` maps role -> path."""
    algorithm, factory = _hasher()
    out: dict = {}
    for role, root in sorted(roles.items()):
        if not os.path.exists(root):
            raise ManifestError(f"{role}: {root} does not exist")
        rels = list(_iter_files(root))
        paths = [_file_path(root, rel) for rel in rels]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            hashes = list(pool.map(lambda p: _hash_file(p, factory), paths))
        files = {}
        for rel, path, digest in zip(rels, paths, hashes):
            files[rel] = {**_stat_fields(path), "hash": digest}
        out[role] = {"root": os.path.abspath(root), "files": files}
    return {
        "version": MANIFEST_VERSION,
        "algorithm": algorithm,
        "roles": out,
        "digest": manifest_digest(out),
    }


def load_manifest(path: str) -> dict:
    try:
        with open(path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        raise ManifestError(f"unsupported manifest version in {path}")
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise ManifestError(f"manifest {path} has no roles")
    try:
        digest = manifest_digest(roles)
    except (KeyError, TypeError) as exc:
        raise ManifestError(f"malformed manifest {path}: {exc}") from exc
    if digest != manifest.get("digest"):
        raise ManifestError(f"manifest {path} digest does not match its entries")
    return manifest


def verify_manifest(manifest: dict, expected_roots: Mapping[str, str]) -> str:
    """Check the loaded files against the manifest; return its digest.

    Every role in ``expected_roots`` (role -> path the engine loads) must be
    in the manifest with the same resolved root and exactly the listed files,
    each with unchanged size, mtime and inode. Other roles in the manifest are
    checked file by file. Raises ManifestError on any difference.
    """
    roles = manifest["roles"]
    for role, path in expected_roots.items():
        entry = roles.get(role)
        if entry is None:
            raise ManifestError(f"manifest has no role {role!r} for {path}")
        if os.path.realpath(entry["root"]) != os.path.realpath(path):
            raise ManifestError(
                f"{role}: manifest root {entry['root']} is not the loaded {path}"
            )
        actual = set(_iter_files(path))
        listed = set(entry["files"])
        if actual != listed:
            extra = sorted(actual - listed)[:5]
            missing = sorted(listed - actual)[:5]
            raise ManifestError(
                f"{role}: files differ from manifest (unlisted {extra}, "
                f"missing {missing})"
            )
    for role, entry in roles.items():
        root = entry["root"]
        for rel, record in entry["files"].items():
            path = _file_path(root, rel)
            try:
                stat = _stat_fields(path)
            except OSError as exc:
                raise ManifestError(f"{role}/{rel}: {exc}") from exc
            for field in ("size", "mtime_ns", "ino"):
                if stat[field] != record[field]:
                    raise ManifestError(
                        f"{role}/{rel}: {field} changed since the manifest "
                        f"was made ({record[field]} -> {stat[field]})"
                    )
    return manifest["digest"]


def rehash_and_verify(manifest: dict) -> str:
    """Recompute every content hash (slow, optional full verification)."""
    algorithm, factory = _hasher()
    if algorithm != manifest["algorithm"]:
        raise ManifestError(
            f"manifest uses {manifest['algorithm']}, only {algorithm} available"
        )
    for role, entry in manifest["roles"].items():
        for rel, listed in entry["files"].items():
            digest = _hash_file(_file_path(entry["root"], rel), factory)
            if digest != listed["hash"]:
                raise ManifestError(f"{role}/{rel}: content hash differs")
    return manifest["digest"]
