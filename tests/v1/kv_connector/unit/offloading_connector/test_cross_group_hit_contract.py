# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-group hit contract on the offloading connector
(``kernel_config.qsa_cross_group_hit_check``).

Groups: full attention and a Mamba ``align`` state on 32-token pages, plus a
drafter's sliding window (45 tokens) on 16-token blocks. An external hit is
reported only when every group is complete at it, the load plan restores
exactly that prefix, and probed stores (e.g. a host history pool the
connector does not manage) still hold it.
"""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    MockOffloadingSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_hit_contract import HitNeed
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import get_offload_block_hash

pytestmark = pytest.mark.cpu_test

FULL, MAMBA, DRAFT = 0, 1, 2
PAGE = 32
DRAFT_BLOCK = 16
DRAFT_WINDOW = 45
# Blocks the drafter's GPU manager keeps: cdiv(45 - 1, 16).
DRAFT_NEED = 3


def _scheduler(num_tokens: int, check: bool, *, eagle: bool = False):
    kv_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=PAGE, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=PAGE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=DRAFT_BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=DRAFT_WINDOW,
                ),
            ),
        ],
    )
    config = MagicMock()
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.world_size = 1
    config.cache_config.enable_prefix_caching = True
    config.cache_config.block_size = PAGE
    config.cache_config.hash_block_size = None
    config.cache_config.prefix_cache_retention_interval = 0
    config.kv_transfer_config.kv_connector_extra_config = {}
    config.speculative_config = SimpleNamespace(use_eagle_kv_cache=lambda: eagle)
    if check:
        config.kernel_config.qsa_cross_group_hit_check = True
    # With check False the attribute is a MagicMock, which must not enable it.
    scheduler = OffloadingConnectorScheduler(MockOffloadingSpec(config, kv_config))
    req = MagicMock()
    req.request_id = "req"
    req.skip_reading_prefix_cache = False
    req.kv_transfer_params = None
    req.block_hashes = [
        BlockHash(f"h{i}".encode()) for i in range(num_tokens // DRAFT_BLOCK)
    ]
    req.num_computed_tokens = 0
    req.num_prompt_tokens = req.num_tokens = num_tokens
    req.shared_prefix_boundary = 0
    req.is_finished.return_value = False
    scheduler.on_new_request(req)
    state = scheduler._req_status["req"]
    state.update_offload_keys()
    return scheduler, req, state.group_states


def _keys(groups, present_blocks):
    return {
        groups[g].offload_keys[b]
        for g, blocks in present_blocks.items()
        for b in blocks
    }


def _set_present(scheduler, present):
    scheduler.manager.lookup.side_effect = lambda key, ctx: key in present


def test_contract_is_off_by_default():
    scheduler, _, _ = _scheduler(320, check=False)
    assert scheduler.hit_contract is None
    scheduler, _, _ = _scheduler(320, check=True)
    assert scheduler.hit_contract is not None
    rules = scheduler.hit_contract.rules
    assert (rules[FULL].need, rules[FULL].block_size) == (HitNeed.PREFIX, PAGE)
    assert (rules[MAMBA].need, rules[MAMBA].block_size) == (HitNeed.STATE, PAGE)
    assert (rules[DRAFT].need, rules[DRAFT].window_tokens) == (
        HitNeed.WINDOW,
        DRAFT_WINDOW - 1,
    )


@pytest.mark.parametrize("missing", [None, 1, 2, 3])
def test_window_off_grid_hit_and_missing_payload(missing):
    kv = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=12, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["window"],
                SlidingWindowSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=45,
                ),
            ),
        ],
    )
    config = MagicMock()
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.world_size = 1
    config.cache_config.block_size = 4
    config.cache_config.hash_block_size = None
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_cache_retention_interval = 0
    config.kv_transfer_config.kv_connector_extra_config = {}
    config.speculative_config = None
    config.kernel_config.qsa_cross_group_hit_check = True
    scheduler = OffloadingConnectorScheduler(MockOffloadingSpec(config, kv))
    req = MagicMock()
    req.request_id = "off-grid"
    req.skip_reading_prefix_cache = False
    req.kv_transfer_params = None
    req.block_hashes = [BlockHash(f"h{i}".encode()) for i in range(16)]
    req.num_computed_tokens = 0
    req.num_prompt_tokens = req.num_tokens = 65
    req.shared_prefix_boundary = 0
    req.is_finished.return_value = False
    scheduler.on_new_request(req)
    state = scheduler._req_status[req.request_id]
    state.update_offload_keys()
    present = set(state.group_states[0].offload_keys[:5])
    present.update(
        state.group_states[1].offload_keys[i] for i in (1, 2, 3) if i != missing
    )
    _set_present(scheduler, present)
    assert list(scheduler.hit_contract.rules[1].required_blocks(60)) == [1, 2, 3]
    hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    assert hit == (60 if missing is None else 0)


@pytest.mark.parametrize("missing_state", [False, True])
def test_prefix_probe_cap_preserves_eagle_lookbehind(missing_state):
    scheduler, req, groups = _scheduler(128, check=True, eagle=True)
    present = _keys(
        groups, {FULL: {0, 1, 2}, MAMBA: set() if missing_state else {0}, DRAFT: {0, 1}}
    )
    _set_present(scheduler, present)
    scheduler.hit_contract.register_probe(FULL, lambda g, i, h, bid: i == 0)
    hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    assert hit == (0 if missing_state else PAGE)


@pytest.mark.parametrize("local_blocks", [0, 2, 4])
@pytest.mark.parametrize("seed", range(60))
def test_contract_agrees_with_lookup(seed: int, local_blocks: int):
    """Random host-tier contents, with and without a local GPU hit: the
    contract reports exactly the existing lookup result (no false rejection),
    and that result is complete in every group."""
    local = local_blocks * PAGE
    results = []
    for check in (False, True):
        scheduler, req, groups = _scheduler(320, check)
        rng_state = random.Random(seed)
        present = set()
        for group_idx in (FULL, MAMBA, DRAFT):
            for key in groups[group_idx].offload_keys:
                if rng_state.random() < (0.9 if group_idx == FULL else 0.6):
                    present.add(key)
        _set_present(scheduler, present)
        num_hit, _ = scheduler.get_num_new_matched_tokens(req, local)
        results.append(num_hit)
        if check:
            assert scheduler.hit_contract.stats.rejected_hits == 0
            if num_hit:
                end = local + num_hit
                full = groups[FULL].offload_keys[local // PAGE : end // PAGE]
                assert all(k in present for k in full)
                assert groups[MAMBA].offload_keys[end // PAGE - 1] in present
                last = end // DRAFT_BLOCK
                first = max(local // DRAFT_BLOCK, last - DRAFT_NEED)
                assert all(k in present for k in groups[DRAFT].offload_keys[first:last])
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "present_blocks",
    [
        # Only full attention and the drafter are offloaded (no state).
        {FULL: range(10), DRAFT: range(20)},
        # Full attention and states, but the drafter window is missing.
        {FULL: range(10), MAMBA: range(10)},
        # States and drafter only.
        {MAMBA: range(10), DRAFT: range(20)},
    ],
)
@pytest.mark.parametrize("check", [False, True])
def test_partial_group_offload_is_not_a_hit(present_blocks, check: bool):
    scheduler, req, groups = _scheduler(320, check)
    _set_present(scheduler, _keys(groups, present_blocks))
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 0


@pytest.mark.parametrize("missing", ["full", "state", "draft", "off_grid"])
def test_incomplete_lookup_answer_is_a_miss(missing: str):
    """A lookup answer that some group cannot serve is reported as a miss
    (the existing ``_lookup`` is replaced to inject the inconsistency)."""
    present_blocks = {FULL: range(10), MAMBA: [5], DRAFT: range(20)}
    answer = 192
    if missing == "full":
        present_blocks[FULL] = [b for b in range(10) if b != 2]
    elif missing == "state":
        present_blocks[MAMBA] = [4]
    elif missing == "draft":
        present_blocks[DRAFT] = [b for b in range(20) if b != 10]
    else:
        answer = 176  # inside a 32-token Mamba page
    for check in (False, True):
        scheduler, req, groups = _scheduler(320, check)
        _set_present(scheduler, _keys(groups, present_blocks))
        scheduler._lookup = lambda *args, **kwargs: answer  # type: ignore[method-assign]
        num_hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
        if check:
            assert num_hit == 0
            assert scheduler.hit_contract.stats.rejected_hits == 1
        else:
            assert num_hit == answer


def test_probe_only_group_caps_the_external_hit():
    """A group the connector neither stores nor loads (e.g. main KV in a host
    history pool) bounds the hit through its probe."""
    scheduler, req, groups = _scheduler(320, check=True)
    _set_present(
        scheduler,
        _keys(groups, {FULL: range(10), MAMBA: [3, 5, 8], DRAFT: range(20)}),
    )
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 9 * PAGE

    host_group = 7
    held = {bytes(h) for h in req.block_hashes[: 2 * 7]}  # pages 0..6

    def probe(group_id, index, block_hash, block_id):
        assert group_id == host_group and block_id is None
        return bytes(block_hash) in held

    contract = scheduler.hit_contract
    contract.register_probe(host_group, probe, block_size=PAGE)
    # Pages 0..6 are held: the hit falls back to the state ending at page 5.
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 6 * PAGE
    assert contract.stats.probe_capped >= 1
    assert contract.stats.rejected_hits == 0

    held.clear()
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 0


def test_probe_identity_matches_offload_keys():
    """Probes see the same chain hash the connector keys its blocks with."""
    scheduler, req, groups = _scheduler(320, check=True)
    _set_present(
        scheduler, _keys(groups, {FULL: range(10), MAMBA: [4], DRAFT: range(20)})
    )
    seen = {}

    def probe(group_id, index, block_hash, block_id):
        seen[index] = block_hash
        return True

    scheduler.hit_contract.register_probe(FULL, probe)
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 5 * PAGE
    for index, block_hash in seen.items():
        assert block_hash == get_offload_block_hash(groups[FULL].offload_keys[index])


def test_hit_inside_a_block_of_a_mixed_size_full_attention_group():
    """Full-attention groups of 12 and 16 tokens, no recurrent state: the
    lookup may end inside a 12-token block (it then loads that whole block),
    which is complete for full attention and must not be rejected."""
    kv_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [f"g{size}"],
                FullAttentionSpec(
                    block_size=size, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
            for size in (12, 16)
        ],
    )
    config = MagicMock()
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.world_size = 1
    config.cache_config.enable_prefix_caching = True
    config.cache_config.block_size = 4
    config.cache_config.hash_block_size = None
    config.cache_config.prefix_cache_retention_interval = None
    config.kv_transfer_config.kv_connector_extra_config = {}
    config.speculative_config = None
    config.kernel_config.qsa_cross_group_hit_check = True
    scheduler = OffloadingConnectorScheduler(MockOffloadingSpec(config, kv_config))
    req = MagicMock()
    req.request_id = "req"
    req.skip_reading_prefix_cache = False
    req.kv_transfer_params = None
    req.block_hashes = [BlockHash(f"h{i}".encode()) for i in range(85 // 4)]
    req.num_computed_tokens = 0
    req.num_prompt_tokens = req.num_tokens = 85
    req.shared_prefix_boundary = 0
    req.is_finished.return_value = False
    scheduler.on_new_request(req)
    scheduler.manager.lookup.return_value = True
    # 48 local tokens; 7 blocks of 12 reach 84, 5 blocks of 16 reach 80.
    assert scheduler.get_num_new_matched_tokens(req, 48)[0] == 32
    assert scheduler.hit_contract.stats.rejected_hits == 0


# --- load plan ------------------------------------------------------------------


def _alloc(num_cached: int, local: int = 0) -> KVCacheBlocks:
    """Blocks as the GPU managers hand them out for a hit of ``num_cached``
    tokens with ``local`` of them from the GPU prefix cache."""
    null = KVCacheBlock(block_id=0, is_null=True)
    next_id = iter(range(1, 1000))

    def fresh() -> KVCacheBlock:
        return KVCacheBlock(block_id=next(next_id))

    def hashed() -> KVCacheBlock:
        block = fresh()
        block.block_hash = BlockHash(b"local")  # type: ignore[assignment]
        return block

    full = [hashed() for _ in range(local // PAGE)]
    full += [fresh() for _ in range(num_cached // PAGE - len(full))]
    state = [null] * (num_cached // PAGE - 1) + [fresh()]
    skipped = max(0, num_cached - DRAFT_WINDOW + 1) // DRAFT_BLOCK
    draft = [null] * skipped
    draft += [fresh() for _ in range(num_cached // DRAFT_BLOCK - skipped)]
    return KVCacheBlocks((full, state, draft))


def test_load_plan_restores_exactly_the_hit():
    scheduler, req, groups = _scheduler(320, check=True)
    present = _keys(groups, {FULL: range(10), MAMBA: [5], DRAFT: range(20)})
    _set_present(scheduler, present)
    num_hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    assert num_hit == 192
    scheduler.update_state_after_alloc(req, _alloc(192), num_hit)
    assert scheduler.hit_contract.stats.load_plan_errors == 0
    (job,) = scheduler._current_batch_load_jobs.values()
    src, dst = job.transfer_spec
    assert dst.group_sizes == [6, 1, 3]


@pytest.mark.parametrize("check", [False, True])
def test_load_plan_with_evicted_state_is_abandoned(check: bool):
    """The state is gone between lookup and allocation (an internal error:
    nothing evicts in between today). With the contract the plan is refused
    before any transfer is issued and the load is abandoned for a recompute
    (take_failed_loads; the next lookup offers no external hit); without it
    the mock tier would load."""
    scheduler, req, groups = _scheduler(320, check)
    present = _keys(groups, {FULL: range(10), MAMBA: [5], DRAFT: range(20)})
    _set_present(scheduler, present)
    num_hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    assert num_hit == 192
    present.discard(groups[MAMBA].offload_keys[5])
    scheduler.update_state_after_alloc(req, _alloc(192), num_hit)
    if check:
        assert not scheduler._current_batch_load_jobs
        assert scheduler.hit_contract.stats.load_plan_errors == 1
        assert scheduler.take_failed_loads() == {req.request_id}
        assert scheduler.take_failed_loads() == set()
        present.add(groups[MAMBA].offload_keys[5])
        assert scheduler.get_num_new_matched_tokens(req, 0) == (0, False)
        # Repeated lookup after deferral must still skip the tier, until an
        # allocation actually admits a recompute.
        assert scheduler.get_num_new_matched_tokens(req, 0) == (0, False)
        scheduler.update_state_after_alloc(req, _alloc(0), 0)
        assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 192
    else:
        assert len(scheduler._current_batch_load_jobs) == 1
        assert scheduler.take_failed_loads() == set()


def test_load_plan_into_null_block_is_abandoned():
    scheduler, req, groups = _scheduler(320, check=True)
    _set_present(
        scheduler, _keys(groups, {FULL: range(10), MAMBA: [5], DRAFT: range(20)})
    )
    num_hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    blocks = _alloc(192)
    blocks.blocks[FULL][3] = KVCacheBlock(block_id=0, is_null=True)  # type: ignore[index]
    scheduler.update_state_after_alloc(req, blocks, num_hit)
    assert not scheduler._current_batch_load_jobs
    assert scheduler.take_failed_loads() == {req.request_id}


def test_load_plan_after_local_hit():
    scheduler, req, groups = _scheduler(320, check=True)
    present = _keys(groups, {FULL: range(2, 10), MAMBA: [5], DRAFT: range(20)})
    _set_present(scheduler, present)
    num_hit, _ = scheduler.get_num_new_matched_tokens(req, 2 * PAGE)
    assert num_hit == 192 - 2 * PAGE
    scheduler.update_state_after_alloc(req, _alloc(192, local=2 * PAGE), num_hit)
    (job,) = scheduler._current_batch_load_jobs.values()
    src, dst = job.transfer_spec
    assert dst.group_sizes == [4, 1, 3]
    assert dst.block_indices == [2, 5, 9]


# --- readiness without side effects on real tiers ------------------------------


def test_key_ready_reads_real_tiers_without_touching_recency():
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key
    from vllm.v1.kv_offload.cpu.manager import (
        CPUOffloadingManager,
        GroupedCPUOffloadingManager,
    )
    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

    scheduler, _, _ = _scheduler(320, check=True)
    pools = {g: CPUOffloadingManager(num_blocks=4) for g in (FULL, MAMBA, DRAFT)}
    ctx = ReqContext(req_id="req")
    ready = [make_offload_key(bytes([i]), FULL) for i in range(3)]
    in_flight = make_offload_key(b"w", FULL)
    absent = make_offload_key(b"x", MAMBA)
    pools[FULL].prepare_store(ready, ctx)
    pools[FULL].complete_store(ready, ctx)
    pools[FULL].prepare_store([in_flight], ctx)
    order = list(pools[FULL]._policy.blocks)  # type: ignore[attr-defined]

    grouped = GroupedCPUOffloadingManager(pools)
    tiering = object.__new__(TieringOffloadingManager)
    tiering.primary_tier = grouped  # type: ignore[assignment]
    for manager in (grouped, tiering):
        scheduler.manager = manager
        assert all(scheduler._key_ready(key, ctx) for key in ready)
        assert not scheduler._key_ready(in_flight, ctx)
        assert not scheduler._key_ready(absent, ctx)
        assert not scheduler._key_ready(make_offload_key(b"y", 9), ctx)
    assert list(pools[FULL]._policy.blocks) == order  # type: ignore[attr-defined]
