# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU allocation regressions using real managers and a shared block pool."""

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator as KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
    KpoolTailManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KpoolTailSpec,
    MambaSpec,
)


def mamba(pool, group=0, block_size=1632):
    return MambaManager(
        MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=4,
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=group,
    )


def test_mamba_shrink_and_regrow():
    pool = BlockPool(64, True, 1632)
    m = mamba(pool)
    m.allocate_new_blocks("r", 1633, 1633)
    before = list(m.req_to_blocks["r"])
    assert len(before) == 6
    assert m.allocate_new_blocks("r", 1631, 1631) == []
    assert m.req_to_blocks["r"] == before
    assert m.allocate_new_blocks("r", 1634, 1634) == []
    m.allocate_new_blocks("r", 3265, 3265)
    assert len(m.req_to_blocks["r"]) == 7


@pytest.mark.parametrize("tail", [False, True])
def test_ring_cannot_steal_mamba_hit(tail):
    pool = BlockPool(32, True, 16)
    fa = FullAttentionManager(
        FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )
    spec_cls = KpoolTailSpec if tail else CircularBufferSpec
    manager_cls = KpoolTailManager if tail else CircularBufferManager
    extra = {"sliding_window": 16} if tail else {}
    ring = manager_cls(
        spec_cls(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32, **extra
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=1,
    )
    m = mamba(pool, 2, 16)
    hit = pool.get_new_blocks(1)[0]
    key = make_block_hash_with_group_id(BlockHash(b"mamba-hit"), 2)
    hit.block_hash = key
    pool.cached_block_hash_to_block.insert(key, hit)
    # Reproduce cache pressure: every free block is cached, without any
    # uncached scratch that the pool would normally place ahead of this hit.
    others = pool.get_new_blocks(pool.get_num_free_blocks())
    for index, block in enumerate(others):
        other_key = make_block_hash_with_group_id(
            BlockHash(f"cached-{index}".encode()), 0
        )
        block.block_hash = other_key
        pool.cached_block_hash_to_block.insert(other_key, block)
    pool.free_blocks(others)
    pool.free_blocks([hit])
    pool.free_block_queue.remove(hit)
    pool.free_block_queue.prepend_n([hit])
    coordinator = object.__new__(KVCacheCoordinator)
    coordinator.single_type_managers = [fa, ring, m]
    coordinator.allocate_new_computed_blocks("r", ([], [], [hit]), 16, 0)
    coordinator.allocate_new_blocks("r", 17, 17)
    assert ring.req_to_blocks["r"][0].block_id != hit.block_id
    assert hit.ref_cnt == 1
    assert hit.block_id not in ring.take_new_block_ids()
    assert pool.cached_block_hash_to_block.get_one_block(key) is hit
    assert len(ring.req_to_blocks["r"]) == 1
    assert ring.allocate_new_blocks("r", 33, 33) == []
    coordinator.free("r")
    assert hit.ref_cnt == 0
