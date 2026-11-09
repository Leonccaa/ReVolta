# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Joint admission of the long-context (Active Host KV) mode in the scheduler.

The scheduler runs the deployed compressed-sparse + GDN layout (TP4/DCP2,
1,664-token blocks, no speculation) with a capacity plan from the startup
path. Requests that do not fit a domain must wait or be preempted, never fail,
and the ledger must match the blocks the managers actually hold.
"""

from dataclasses import replace

import pytest

from vllm.config.kernel import LongContextKVConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.long_context_kv import GPU_HISTORY, HOST, HOT, STATE, GiB
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .long_context_kv_utils import POOL_BLOCK_BYTES, SPAN, production_specs
from .test_scheduler_kv_capacity import make_config, step

pytestmark = pytest.mark.cpu_test

MAX_MODEL_LEN = 10 * SPAN  # one request needs 10 history blocks
PHYSICAL_BLOCKS = 1500  # the unified pool never limits these workloads


def build(
    settings: LongContextKVConfig | None,
    *,
    max_num_seqs: int = 8,
    prefix_caching: bool = True,
    reserve_full_isl: bool = False,
):
    config = make_config(max_model_len=MAX_MODEL_LEN, speculative_tokens=0)
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.scheduler_reserve_full_isl = reserve_full_isl
    config.cache_config.enable_prefix_caching = prefix_caching
    if settings is not None:
        config.kernel_config.long_context_kv = settings
    specs = production_specs(speculative_blocks=0, drafter=False)
    (worker,) = get_kv_cache_configs(
        config, [specs], [PHYSICAL_BLOCKS * POOL_BLOCK_BYTES]
    )
    return Scheduler(
        vllm_config=config,
        kv_cache_config=generate_scheduler_kv_cache_config([worker]),
        structured_output_manager=StructuredOutputManager(config),
        block_size=16,
        hash_block_size=16,
    )


def settings(**kw) -> LongContextKVConfig:
    kw.setdefault("host_pool_tokens", 40 * SPAN)
    kw.setdefault("hot_pool_gib", 0.2)
    return LongContextKVConfig(enabled=True, **kw)


_hash_ready = False


def make_requests(n, prompt_len, max_tokens, *, same_prompt=False, tag="r"):
    global _hash_ready
    if not _hash_ready:
        init_none_hash(sha256)
        _hash_ready = True
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 0)
    return [
        Request(
            request_id=f"{tag}{i}",
            prompt_token_ids=[7 if same_prompt else 1000 + i] * prompt_len,
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(16, sha256),
        )
        for i in range(n)
    ]


def recount(scheduler) -> list[int]:
    """Blocks each domain really holds, from the managers' block tables."""
    manager = scheduler.kv_cache_manager
    ledger = manager.admission_ledger
    held: list[set[int]] = [set(), set(), set()]
    for group_id, single in enumerate(manager.coordinator.single_type_managers):
        for blocks in single.req_to_blocks.values():
            for block in blocks:
                if not block.is_null and block.ref_cnt > 0:
                    for domain in ledger.group_domains[group_id]:
                        held[domain].add(block.block_id)
    return [len(ids) for ids in held]


def run(scheduler, requests, *, check=True, max_steps=20000):
    for request in requests:
        scheduler.add_request(request)
    ledger = scheduler.kv_cache_manager.admission_ledger
    trace = []
    peak_running = 0
    for _ in range(max_steps):
        if not scheduler.has_unfinished_requests():
            break
        output = step(scheduler)
        trace.append(
            (
                dict(output.num_scheduled_tokens),
                {req.req_id: req.block_ids for req in output.scheduled_new_reqs},
                sorted(output.preempted_req_ids or ()),
            )
        )
        peak_running = max(peak_running, len(scheduler.running))
        if check and ledger is not None:
            for domain in (STATE, GPU_HISTORY, HOST):
                assert 0 <= ledger.in_use[domain] <= ledger.capacity[domain]
            assert ledger.in_use == recount(scheduler)
            assert ledger.hot_used <= ledger.plan.hot_pool_groups
            assert ledger.untracked_allocations == 0
    assert not scheduler.has_unfinished_requests(), "workload did not finish"
    for request in requests:
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    if ledger is not None:
        # Everything returned: only evictable prefix-cache blocks remain.
        assert ledger.in_use == [0, 0, 0]
        assert ledger.hot_used == 0
    return trace, peak_running


@pytest.mark.parametrize("reserve_full_isl", [False, True])
def test_host_pool_full_queues_new_requests_without_errors(reserve_full_isl):
    scheduler = build(
        settings(host_pool_tokens=MAX_MODEL_LEN), reserve_full_isl=reserve_full_isl
    )
    ledger = scheduler.kv_cache_manager.admission_ledger
    assert ledger.capacity[HOST] == 10
    # Four 4-block requests: the host pool holds two whole prompts at a time.
    requests = make_requests(4, 4 * SPAN - 100, 4)
    _, peak = run(scheduler, requests)
    assert ledger.blocked[HOST] > 0
    # Admission checks what a step allocates (or, with the full-prompt gate,
    # the candidate's whole prompt) against what is free now, as for the GPU
    # pool; requests already running may still grow and preempt later.
    assert peak >= 2
    # The unified pool always had room: the host pool alone held them back.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() > 1000


def test_growth_past_the_host_pool_preempts_and_finishes():
    scheduler = build(settings(host_pool_tokens=MAX_MODEL_LEN))
    # Each request grows from 1 to 3 host blocks while decoding.
    requests = make_requests(4, SPAN - 64, 2 * SPAN)
    trace, _ = run(scheduler, requests)
    assert any(preempted for _, _, preempted in trace)
    assert sum(r.num_preemptions for r in requests) > 0
    assert scheduler.kv_cache_manager.admission_ledger.blocked[HOST] > 0


def test_hot_pool_limits_concurrency_without_preemption():
    scheduler = build(settings())
    plan = scheduler.kv_cache_manager.admission_ledger.plan
    per_request = plan.geometry.hot_reserve_groups(MAX_MODEL_LEN)
    assert per_request == 513 + 9  # one decode row, plus the pinned tail
    hot_gib = 2 * plan.geometry.hot_bytes_for_groups(per_request) / GiB + 1e-6
    scheduler = build(settings(hot_pool_gib=hot_gib))
    ledger = scheduler.kv_cache_manager.admission_ledger
    assert ledger.plan.hot_pool_groups // per_request == 2
    requests = make_requests(5, 2 * SPAN, 8)
    _, peak = run(scheduler, requests)
    assert peak == 2
    assert ledger.blocked[HOT] > 0
    # Reservations are made at admission, so a running request never needs
    # more hot-pool room and is never preempted for it.
    assert sum(r.num_preemptions for r in requests) == 0


def test_state_segment_limits_concurrency():
    probe = build(settings())
    per_request = probe.kv_cache_manager.admission_ledger.plan.geometry
    assert per_request.state_blocks_per_request == 6 * 2 + 2
    # One request peaks at all 14 state blocks (GDN boundary states and PLE),
    # but holds fewer when admitted; the segment is never overcommitted.
    scheduler = build(settings(state_blocks=2 * 14))
    ledger = scheduler.kv_cache_manager.admission_ledger
    requests = make_requests(6, 2 * SPAN, 8)
    run(scheduler, requests)
    assert ledger.blocked[STATE] > 0


def test_shared_prefix_counts_once():
    scheduler = build(settings(host_pool_tokens=4 * MAX_MODEL_LEN))
    ledger = scheduler.kv_cache_manager.admission_ledger
    main = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    requests = make_requests(3, 4 * SPAN + 10, 64, same_prompt=True)
    scheduler.add_request(requests[0])
    for _ in range(4):
        step(scheduler)
    for request in requests[1:]:
        scheduler.add_request(request)
    shared_seen = False
    while scheduler.has_unfinished_requests():
        step(scheduler)
        assert ledger.in_use == recount(scheduler)
        per_request = sum(
            sum(not block.is_null for block in blocks)
            for blocks in main.req_to_blocks.values()
        )
        shared_seen |= per_request > ledger.in_use[HOST]
    assert shared_seen  # prefix hits were charged once, not per request
    assert ledger.in_use == [0, 0, 0]


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_unbinding_plan_schedules_exactly_like_the_disabled_mode(prefix_caching):
    def workload():
        return make_requests(6, 3 * SPAN + 77, 300) + make_requests(
            3, 2 * SPAN, 50, same_prompt=True, tag="s"
        )

    disabled = build(None, prefix_caching=prefix_caching)
    assert disabled.kv_cache_manager.admission_ledger is None
    assert disabled.kv_cache_config.long_context_plan is None
    roomy = build(
        settings(host_pool_tokens=10_000 * SPAN, hot_pool_gib=0.6, state_blocks=500),
        prefix_caching=prefix_caching,
    )
    assert roomy.kv_cache_manager.admission_ledger is not None
    # Same steps, same block IDs, same preemptions when no domain binds.
    assert run(disabled, workload()) == run(roomy, workload())
    assert roomy.kv_cache_manager.admission_ledger.blocked == [0, 0, 0, 0]


def test_physical_pool_untouched_by_the_plan():
    disabled = build(None)
    enabled = build(settings())
    assert disabled.kv_cache_config.num_blocks == enabled.kv_cache_config.num_blocks
    assert replace(enabled.kv_cache_config, long_context_plan=None) == (
        disabled.kv_cache_config
    )
