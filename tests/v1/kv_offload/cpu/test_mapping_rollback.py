# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construction failures close descriptors and unlink locally owned mappings."""

import errno
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from vllm.v1.kv_offload.cpu import shared_offload_region as shared


@pytest.mark.parametrize("phase", ["truncate", "mmap", "tensor"])
def test_failed_mapping_constructor_rolls_back(monkeypatch, phase):
    name = "rollback-" + uuid4().hex
    path = Path(f"/dev/shm/vllm_offload_{name}.mmap")
    closed = []
    original_close = shared.os.close

    def close(fd):
        closed.append(fd)
        original_close(fd)

    def fail(*args, **kwargs):
        raise RuntimeError("injected mapping initialization failure")

    monkeypatch.setattr(shared.os, "close", close)
    if phase == "truncate":
        monkeypatch.setattr(shared.os, "ftruncate", fail)
    elif phase == "mmap":
        monkeypatch.setattr(shared.mmap, "mmap", fail)
    else:
        monkeypatch.setattr(shared.torch, "frombuffer", fail)
    try:
        with pytest.raises(RuntimeError, match="initialization failure"):
            shared.SharedOffloadRegion(name, 4096, 1, None, 1, 4096)
        assert len(closed) == 1
        assert not path.exists()
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("error", "free", "report"),
    [(errno.ENOSPC, 0, True), (errno.EFAULT, 1024, True), (errno.EFAULT, 8192, False)],
)
def test_mapping_failure_reports_dev_shm_capacity(monkeypatch, error, free, report):
    name = "capacity-" + uuid4().hex
    path = Path(f"/dev/shm/vllm_offload_{name}.mmap")

    def fail(*args):
        raise OSError(error, "injected prefault failure")

    class Mapping:
        madvise = staticmethod(fail)
        close = staticmethod(lambda: None)

    monkeypatch.setattr(shared.mmap, "mmap", lambda *args, **kwargs: Mapping())
    monkeypatch.setattr(
        shared.shutil, "disk_usage", lambda _: SimpleNamespace(free=free)
    )
    with pytest.raises(OSError) as exc_info:
        shared.SharedOffloadRegion(name, 4096, 1, None, 1, 4096)
    assert exc_info.value.errno == error
    assert not path.exists()
    message = str(exc_info.value)
    if report:
        assert f"needs 4,096 bytes in /dev/shm. {free:,} bytes free" in message
        assert "vllm_offload_*.mmap" in message
        assert "positive --shutdown-timeout" in message
    else:
        assert message == f"[Errno {error}] injected prefault failure"
