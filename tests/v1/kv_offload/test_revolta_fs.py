# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ReVolta file-system secondary tier (``revolta_fs``)."""

import mmap
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
from vllm.v1.kv_offload.tiering.fs.revolta import RevoltaFileSystemTierManager


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
