# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU allocation regressions using real managers and a shared block pool."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator as KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
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
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadingManager
from vllm.v1.request import Request


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


def _hybrid_offload_layout(drafter: bool):
    """Prefix-cacheable groups of the deployment, in planner order.

    Full attention first, then an ``align`` state group standing in for the
    GDN/PLE groups and, with DFlash2, the drafter's sliding window last
    (832-token pages, 2880-token window). Without the drafter this is the
    MTP4 layout, whose only hit-holding group comes first.
    """
    page = 1664 if drafter else 1632
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=page, num_kv_heads=1, head_size=1, dtype=torch.float16
            ),
        ),
        KVCacheGroupSpec(
            ["gdn"],
            MambaSpec(
                block_size=page,
                shapes=((1,),),
                dtypes=(torch.float16,),
                mamba_cache_mode="align",
            ),
        ),
    ]
    if drafter:
        groups.append(
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=page // 2,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                    sliding_window=2880,
                ),
                is_eagle_group=True,
            )
        )
    kv_config = KVCacheConfig(64, [], groups)
    core = KVCacheManager(kv_config, 16 * page, 16, enable_caching=True)
    spec = SimpleNamespace(
        kv_cache_config=kv_config,
        gpu_block_size=tuple(g.kv_cache_spec.block_size for g in groups),
        block_size_factor=1,
        hash_block_size=16,
        offload_prompt_only=False,
        get_manager=lambda: MagicMock(spec=OffloadingManager),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=1),
            cache_config=SimpleNamespace(
                prefix_cache_retention_interval=0, enable_prefix_caching=True
            ),
            speculative_config=None,
        ),
    )
    return core, OffloadingConnectorScheduler(spec), page


@pytest.mark.parametrize("drafter", [True, False], ids=["dflash2", "mtp4"])
def test_external_alloc_cannot_evict_later_group_hit(drafter):
    """A local prefix hit plus one external block (vllm-project/vllm#44409).

    The request hits 3 pages on the GPU and one more from the CPU tier. Its
    cached hit blocks are the oldest evictable blocks, i.e. at the head of the
    free queue, as under cache pressure. If an earlier group took its external
    block before a later group touched its hits, it would evict one of them;
    the later group would then hold a non-null block without a hash below the
    hit, which also backs another group's load (the offloading connector
    asserts on it). Only DFlash2 has such a later hit-holding group: the
    ``align`` state group drops its hit when external tokens follow.
    """
    init_none_hash(sha256)
    core, connector, page = _hybrid_offload_layout(drafter)
    pool = core.block_pool
    local, external = 3 * page, page

    def cached(group, n, tag):
        blocks = pool.get_new_blocks(n)
        for i, block in enumerate(blocks):
            key = make_block_hash_with_group_id(BlockHash(f"{tag}{i}".encode()), group)
            block.block_hash = key
            pool.cached_block_hash_to_block.insert(key, block)
        return blocks

    attn, state = cached(0, 3, "attn"), cached(1, 1, "gdn")
    hits = [attn, [pool.null_block] * 2 + state]
    # Hit blocks the request keeps; the state hit is dropped (external
    # tokens follow), so taking it for another group is legitimate.
    kept = list(attn)
    oldest = list(state)
    if drafter:
        # The drafter keeps the pages its window needs below the hit.
        pages = cached(2, 2, "draft")
        hits.append([pool.null_block] * 4 + pages)
        kept += pages
        oldest = pages + oldest
    pool.free_blocks(attn + oldest)
    for block in oldest:
        pool.free_block_queue.remove(block)
    pool.free_block_queue.prepend_n(oldest)

    request = Request(
        request_id="r",
        prompt_token_ids=list(range(local + external + 1)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    connector.on_new_request(request)
    req_status = connector._req_status["r"]
    req_status.update_offload_keys()
    req_status.num_locally_computed_tokens = local
    assert (
        core.allocate_slots(
            request,
            0,
            num_new_computed_tokens=local,
            new_computed_blocks=KVCacheBlocks(tuple(hits)),
            num_external_computed_tokens=external,
            delay_cache_blocks=True,
        )
        is not None
    )
    blocks = core.get_blocks("r")
    connector.update_state_after_alloc(request, blocks, external)

    block_ids = [b for group in blocks.get_block_ids() for b in group]
    real = [b for b in block_ids if b != pool.null_block.block_id]
    assert len(set(real)) == len(real), "a block is shared by two groups"
    assert all(b.block_hash is not None for b in kept)
    [load_job] = connector._current_batch_load_jobs.values()
    dst_spec = load_job.transfer_spec[1]
    assert isinstance(dst_spec, GPULoadStoreSpec)
    assert set(dst_spec.block_ids).isdisjoint(b.block_id for b in kept)
    if drafter:
        assert list(dst_spec.block_indices) == [3, 3, 6]
        assert list(dst_spec.group_sizes) == [1, 1, 2]
    else:
        assert list(dst_spec.block_indices) == [3, 3]
        assert list(dst_spec.group_sizes) == [1, 1]
