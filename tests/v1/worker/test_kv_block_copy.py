# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-page KV block copies for partial prefix-cache entries.

The copy reuses the zeroer's 6-field segment table. A DFlash drafter placed in
a main-KV owner page is copied with that page; a layer in its own tensor is
reported as uncovered; the QSA key ring is scratch and never copied.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.worker.test_kv_block_zeroer import (
    _SMALL,
    _BlockFirstAttentionBackend,
    _packed_members,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, SlidingWindowSpec
from vllm.v1.worker.utils import KVBlockZeroer

NUM_BLOCKS = 3


def _packed_views(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """The two packed main-K/V members, as ``_packed_members`` lays them out."""
    spec, kernel_bs = _SMALL
    ratio = spec.block_size // kernel_bs
    shape = (NUM_BLOCKS * ratio, 2, kernel_bs, spec.num_kv_heads, spec.head_size)
    strides = list(torch.empty(shape, device="meta").stride())
    kernel_block_numel = strides[0]
    strides[0] = kernel_block_numel * 2
    return {
        f"layer{j}": torch.as_strided(
            raw.view(spec.dtype),
            size=shape,
            stride=tuple(strides),
            storage_offset=kernel_block_numel * j,
        )
        for j in range(2)
    }


@pytest.mark.skip_global_cleanup
def test_drafter_in_an_owner_page_is_covered_and_the_ring_is_ignored():
    device = torch.device("cpu")
    spec, kernel_bs = _SMALL
    raw, _, _ = _packed_members(device, NUM_BLOCKS)
    views = _packed_views(raw)
    drafter_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        sliding_window=64,
    )
    ring_spec = CircularBufferSpec(
        block_size=16, num_kv_heads=1, head_size=8, head_size_v=0, dtype=torch.float16
    )
    context = {name: SimpleNamespace(kv_cache=kv) for name, kv in views.items()}
    # One drafter layer shares the owner's storage, one has its own tensor.
    context["drafter.shared"] = SimpleNamespace(kv_cache=raw.view(torch.float16))
    context["drafter.standalone"] = SimpleNamespace(
        kv_cache=torch.zeros(NUM_BLOCKS * 256, dtype=torch.float16)
    )
    context["ring"] = SimpleNamespace(
        kv_cache=torch.zeros(NUM_BLOCKS * 128, dtype=torch.float16)
    )
    groups = [
        SimpleNamespace(
            kv_cache_spec=spec,
            kv_cache_group_id=0,
            layer_names=list(views),
            backend=_BlockFirstAttentionBackend,
        ),
        SimpleNamespace(
            kv_cache_spec=drafter_spec,
            kv_cache_group_id=1,
            layer_names=["drafter.shared", "drafter.standalone"],
            backend=None,
        ),
        SimpleNamespace(
            kv_cache_spec=ring_spec,
            kv_cache_group_id=2,
            layer_names=["ring"],
            backend=None,
        ),
    ]
    zeroer = KVBlockZeroer(device, pin_memory=False)
    zeroer.init_meta(
        groups,
        kernel_block_sizes=[kernel_bs, 16, 16],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=context,
    )
    # The segment table keeps the 6-field layout the zero kernel uses.
    assert zeroer._meta is not None and len(zeroer._meta) == 6
    assert zeroer.uncovered_layers == ["drafter.standalone"]


@pytest.mark.skip_global_cleanup
def test_copy_rejects_out_of_range_blocks_before_launch():
    _, _, zeroer = _packed_members(torch.device("cpu"), NUM_BLOCKS)
    zeroer.num_blocks = NUM_BLOCKS
    zeroer.copy_block_ids([])  # no-op
    with pytest.raises(ValueError, match="out of range"):
        zeroer.copy_block_ids([(1, NUM_BLOCKS)])
    with pytest.raises(ValueError, match="out of range"):
        zeroer.copy_block_ids([(-1, 2)])
    empty = KVBlockZeroer(torch.device("cpu"), pin_memory=False)
    with pytest.raises(RuntimeError, match="initialized"):
        empty.copy_block_ids([(0, 1)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_packed_page_copies_both_members_only():
    num_blocks = 4
    raw, block_bytes, zeroer = _packed_members(torch.device("cuda"), num_blocks)
    zeroer.num_blocks = num_blocks
    pages = raw.view(num_blocks, block_bytes)
    pages.copy_(
        torch.arange(num_blocks * block_bytes, device="cuda")
        .remainder(127)
        .to(torch.int8)
        .view(num_blocks, block_bytes)
    )
    before = pages.clone()
    zeroer.copy_block_ids([(1, 3)])
    torch.accelerator.synchronize()
    assert torch.equal(pages[3], before[1])
    for block in (0, 1, 2):
        assert torch.equal(pages[block], before[block]), block
