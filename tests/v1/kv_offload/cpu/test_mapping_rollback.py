# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construction failures close descriptors and unlink locally owned mappings."""

from pathlib import Path
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
