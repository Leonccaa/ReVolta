# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ReVolta file-system secondary tier (``revolta_fs``)."""

import mmap
import os
import shutil
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_offload.test_fs_tier import (
    _BLOCK_ELEMENTS,
    _DTYPE,
    _MOCK_OFFLOADING_SPEC,
    drain,
    key,
    make_job,
)
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.config import fs_tier_configs
from vllm.v1.kv_offload.tiering.fs.revolta import (
    CHECKSUM_XATTR,
    RevoltaFileSystemTierManager,
    read_tagged_crc,
)


def _view(num_blocks: int = 4):
    backing = mmap.mmap(-1, num_blocks * _BLOCK_ELEMENTS * 4)
    tensor = torch.frombuffer(backing, dtype=_DTYPE).reshape(
        num_blocks, _BLOCK_ELEMENTS
    )
    return tensor, memoryview(tensor.numpy())


def make_tier(root, view, spec=_MOCK_OFFLOADING_SPEC, **kwargs):
    kwargs.setdefault("max_size_bytes", 64 << 20)
    kwargs.setdefault("n_read_threads", 2)
    kwargs.setdefault("n_write_threads", 2)
    return RevoltaFileSystemTierManager(
        offloading_spec=spec,
        primary_kv_view=view,
        tier_type="revolta_fs",
        root_dir=str(root),
        **kwargs,
    )


@pytest.fixture
def revolta_tier(tmp_path):
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    yield tier, tensor
    tier.shutdown()


def test_registered_and_shares_fs_seed_rules(tmp_path):
    tensor, view = _view()
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "revolta_fs",
            "root_dir": str(tmp_path),
            "max_size_bytes": 64 << 20,
            "n_read_threads": 1,
            "n_write_threads": 1,
        },
        view,
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, RevoltaFileSystemTierManager)
        assert tier.tier_type == "revolta_fs"
    finally:
        tier.shutdown()
    config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_extra_config={
            "secondary_tiers": [
                {"type": "revolta_fs", "root_dir": str(tmp_path), "max_size_bytes": 1}
            ]
        },
    )
    assert [t["type"] for t in fs_tier_configs(config)] == ["revolta_fs"]


def test_store_then_load_roundtrip(revolta_tier):
    tier, tensor = revolta_tier
    tensor[0].fill_(1.5)
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert [r.success for r in drain(tier)] == [True]
    assert tier.lookup(key(1)) is True
    tensor[1].zero_()
    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    assert [r.success for r in drain(tier)] == [True]
    assert torch.equal(tensor[1], tensor[0])


# ---------------------------------------------------------------------------
# Integrity (CRC32C attribute, atomic publication)
# ---------------------------------------------------------------------------


def _store(tier, tensor, k, slot=0, value=1.0, job_id=1):
    tensor[slot].fill_(value)
    tier.submit_store(make_job(job_id, [k], [slot]))
    assert [r.success for r in drain(tier)] == [True]
    return tier.file_mapper.get_file_name(k)


def _load(tier, k, slot=1, job_id=2):
    tier.submit_load(make_job(job_id, [k], [slot], is_promotion=True))
    return [r.success for r in drain(tier)]


def test_page_has_checksum_and_no_temp_files(revolta_tier):
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1))
    assert read_tagged_crc(path) is not None
    leftovers = [
        f
        for _, _, files in os.walk(os.path.dirname(path))
        for f in files
        if f.endswith(".tmp")
    ]
    assert leftovers == []
    assert tier.get_revolta_stats()["pages_written"] == 1


def test_equal_length_corruption_is_a_miss(revolta_tier):
    """Counterexample 3: an equal-length page with flipped bytes must not load."""
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1), value=2.0)
    with open(path, "r+b") as f:
        f.seek(4096)
        byte = f.read(1)
        f.seek(4096)
        f.write(bytes([byte[0] ^ 0xFF]))
    assert os.path.getsize(path) == _BLOCK_ELEMENTS * 4
    assert _load(tier, key(1)) == [False]
    assert not os.path.exists(path)
    assert tier.lookup(key(1)) is False
    assert tier.get_revolta_stats()["checksum_failures"] == 1


def test_page_without_checksum_is_rejected_and_replaced(revolta_tier):
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1), value=3.0)
    os.removexattr(path, CHECKSUM_XATTR)
    assert _load(tier, key(1)) == [False]
    assert not os.path.exists(path)
    # A bare file (e.g. written by plain fs) is replaced, not trusted.
    with open(path, "wb") as f:
        f.write(b"\0" * (_BLOCK_ELEMENTS * 4))
    _store(tier, tensor, key(1), value=3.0, job_id=3)
    assert read_tagged_crc(path) is not None
    assert _load(tier, key(1), job_id=4) == [True]
    assert torch.equal(tensor[1], tensor[0])


def test_intact_page_of_another_key_is_rejected(revolta_tier):
    """A valid page copied to another key's path fails the tag check."""
    tier, tensor = revolta_tier
    src = _store(tier, tensor, key(1), value=4.0)
    dst = tier.file_mapper.get_file_name(key(2))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    os.setxattr(dst, CHECKSUM_XATTR, os.getxattr(src, CHECKSUM_XATTR))
    tier._cache.lookup(dst)
    assert _load(tier, key(2)) == [False]
    assert _load(tier, key(1), job_id=3) == [True]


def test_wrong_size_page_is_rejected(revolta_tier):
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1))
    with open(path, "ab") as f:
        f.write(b"\0" * 4096)
    assert _load(tier, key(1)) == [False]
    assert not os.path.exists(path)


def test_missing_xattr_support_fails_at_startup(tmp_path, monkeypatch):
    tensor, view = _view()

    def no_xattr(*args, **kwargs):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(os, "setxattr", no_xattr)
    with pytest.raises(RuntimeError, match="extended attributes"):
        make_tier(tmp_path, view)
