# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-filesystem capacity, concurrency and restart tests without CUDA."""

import errno
import fcntl
import os
from types import SimpleNamespace

import pytest

FileSystemCache = pytest.importorskip(
    "vllm.v1.kv_offload.tiering.fs.cache", reason="bounded FS cache is new"
).FileSystemCache


def put(cache, path, data=b"a" * 4096):
    with cache.write(str(path), len(data)):
        path.write_bytes(data)


def test_lru_budget_and_restart(tmp_path):
    cache = FileSystemCache.acquire(str(tmp_path), 8192, 0)
    a, b, c = [tmp_path / f"{name}.bin" for name in "abc"]
    try:
        put(cache, a)
        put(cache, b)
        assert cache.lookup(str(a))
        put(cache, c)
        assert a.exists() and c.exists() and not b.exists()
    finally:
        cache.release()
    # A crash may leave a temp; only complete .bin pages survive reopen.
    orphan = tmp_path / "interrupted.tmp"
    orphan.write_bytes(b"incomplete")
    cache = FileSystemCache.acquire(str(tmp_path), 4096, 0)
    try:
        assert not orphan.exists()
        assert sum(p.stat().st_size for p in tmp_path.glob("*.bin")) == 4096
        put(cache, b)
        assert b.exists() and not a.exists() and not c.exists()
    finally:
        cache.release()


def test_shared_group_budget_and_inflight_protection(tmp_path):
    first = FileSystemCache.acquire(str(tmp_path), 8192, 0)
    second = FileSystemCache.acquire(str(tmp_path), 8192, 0)
    a, b, c = [tmp_path / f"{name}.bin" for name in "abc"]
    try:
        put(first, a)
        with first.read(str(a)), second.write(str(b), 4096):
            # a is pinned, b has a full reservation before its file exists.
            with pytest.raises(OSError) as exc:
                put(first, c)
            assert exc.value.errno == errno.ENOSPC
            assert a.read_bytes() == b"a" * 4096
            b.write_bytes(b"b" * 4096)
        put(first, c)
        assert not a.exists() and b.exists() and c.exists()
        first.release()
        assert second.lookup(str(c))
    finally:
        second.release()


def test_free_space_reserve_and_oversized_page(tmp_path, monkeypatch):
    cache = FileSystemCache.acquire(str(tmp_path), 8192, 4096)
    try:
        monkeypatch.setattr(
            "vllm.v1.kv_offload.tiering.fs.cache.os.statvfs",
            lambda _: SimpleNamespace(f_bavail=1, f_frsize=4096),
        )
        with pytest.raises(OSError) as exc:
            put(cache, tmp_path / "full.bin")
        assert exc.value.errno == errno.ENOSPC
        with pytest.raises(OSError):
            put(cache, tmp_path / "large.bin", b"x" * 12288)
        assert not list(tmp_path.glob("*.bin"))
    finally:
        cache.release()


def test_failed_atomic_write_releases_reservation(tmp_path):
    cache = FileSystemCache.acquire(str(tmp_path), 4096, 0)
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    try:
        with pytest.raises(OSError), cache.write(str(a), 4096):
            raise OSError(errno.ENOSPC, "disk full")
        put(cache, b)
        assert b.read_bytes() == b"a" * 4096
        # Readers pinning one key may coexist, but duplicate stores cannot race.
        with cache.read(str(b)), cache.write(str(b), 4096):
            with pytest.raises(OSError) as exc:
                put(cache, b)
            assert exc.value.errno == errno.EBUSY
    finally:
        cache.release()


@pytest.mark.parametrize("budget,reserve", [(0, 0), (-1, 0), (4096, -1)])
def test_invalid_budget(tmp_path, budget, reserve):
    with pytest.raises(ValueError):
        FileSystemCache.acquire(str(tmp_path), budget, reserve)


def test_conflicting_group_budgets(tmp_path):
    cache = FileSystemCache.acquire(str(tmp_path), 4096, 0)
    try:
        with pytest.raises(ValueError, match="same budget"):
            FileSystemCache.acquire(str(tmp_path), 8192, 0)
    finally:
        cache.release()


def test_root_lock_excludes_other_engines_and_releases(tmp_path):
    cache = FileSystemCache.acquire(str(tmp_path), 4096, 0)
    fd = os.open(tmp_path / ".lock", os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache.release()
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_external_disk_full_does_not_prevent_open(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.cache.os.statvfs",
        lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096),
    )
    cache = FileSystemCache.acquire(str(tmp_path), 4096, 4096)
    try:
        with pytest.raises(OSError) as exc:
            put(cache, tmp_path / "full.bin")
        assert exc.value.errno == errno.ENOSPC
    finally:
        cache.release()


def test_file_count_bounds_small_pages_and_restart_index(tmp_path):
    cache = FileSystemCache.acquire(str(tmp_path), 12288, 0, max_files=1)
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    try:
        put(cache, a)
        put(cache, b)
        assert not a.exists() and b.exists()
    finally:
        cache.release()
    # An older unbounded run may leave more files than the new index limit.
    a.write_bytes(b"a" * 4096)
    cache = FileSystemCache.acquire(str(tmp_path), 12288, 0, max_files=1)
    try:
        assert len(list(tmp_path.glob("*.bin"))) == 1
        assert len(cache._pages) == 1
    finally:
        cache.release()
