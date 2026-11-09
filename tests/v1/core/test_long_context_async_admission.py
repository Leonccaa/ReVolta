# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domain reservations must preserve progress after asynchronous KV loads."""

import pytest

from tests.v1.core.test_async_kv_admission import HybridMockConnector, complete
from tests.v1.core.test_long_context_admission import (
    MAX_MODEL_LEN,
    SPAN,
    build,
    make_requests,
    recount,
    settings,
)
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def scheduler(
    *,
    enabled,
    full_isl,
    async_scheduling=False,
    matched=5 * SPAN,
    speculative_tokens=0,
):
    s = build(
        settings(
            host_pool_tokens=MAX_MODEL_LEN,
            hot_pool_gib=0.8 if speculative_tokens else 0.2,
        )
        if enabled
        else None,
        reserve_full_isl=full_isl,
        speculative_tokens=speculative_tokens,
    )
    if async_scheduling:
        s.vllm_config.scheduler_config.async_scheduling = True
        s = AsyncScheduler(
            vllm_config=s.vllm_config,
            kv_cache_config=s.kv_cache_config,
            structured_output_manager=s.structured_output_manager,
            block_size=16,
            hash_block_size=16,
        )
    cfg = s.vllm_config
    cfg.kv_transfer_config = KVTransferConfig(
        kv_connector="MockKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"matched_tokens": matched, "is_async": matched > 0},
    )
    s.connector = HybridMockConnector(cfg, KVConnectorRole.SCHEDULER, s.kv_cache_config)
    return s


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("full_isl", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("matched", [0, 5 * SPAN, 9 * SPAN])
@pytest.mark.parametrize("slack", [1, 8])
def test_dflash_single_request_fits_without_double_reservation(
    enabled, full_isl, async_scheduling, matched, slack
):
    """prefix regression. One legal request must fit, even with allocated draft
    lookahead.

    The host domain holds exactly ten pages. Both the full-ISL gate and the
    load's allocation can already cover the speculative successor. Clear the
    cache each wave so that later successes still exercise external loads.
    """
    s = scheduler(
        enabled=enabled,
        full_isl=full_isl,
        async_scheduling=async_scheduling,
        matched=matched,
        speculative_tokens=7,
    )
    ledger = s.kv_cache_manager.admission_ledger
    if ledger is not None:
        assert not ledger.plan.startup_problems()
    for wave in range(256):
        assert s.kv_cache_manager.reset_prefix_cache()
        request = make_requests(1, MAX_MODEL_LEN - slack, 1, tag=f"spec{wave}-")[0]
        s.add_request(request)
        for tick in range(128):
            output = s.schedule()
            received = {request.request_id} if tick % 4 == 3 else set()
            complete(s, output, received=received)
            if ledger is not None:
                assert ledger.in_use == recount(s)
                assert ledger.hot_used <= ledger.plan.hot_pool_groups
            if request.is_finished():
                break
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        check_drained(s)


@pytest.mark.parametrize("full_isl", [False, True])
def test_dflash_load_fits_minimal_state_pool(full_isl):
    s = build(
        settings(host_pool_tokens=40 * SPAN, hot_pool_gib=0.8),
        max_num_seqs=1,
        reserve_full_isl=full_isl,
        speculative_tokens=7,
    )
    cfg = s.vllm_config
    cfg.kv_transfer_config = KVTransferConfig(
        kv_connector="MockKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"matched_tokens": 5 * SPAN, "is_async": True},
    )
    s.connector = HybridMockConnector(cfg, KVConnectorRole.SCHEDULER, s.kv_cache_config)
    assert not s.kv_cache_manager.admission_ledger.plan.startup_problems()
    request = make_requests(1, 8 * SPAN, 1)[0]
    s.add_request(request)
    for _ in range(128):
        output = s.schedule()
        complete(s, output, received={request.request_id})
        if request.is_finished():
            break
    assert request.is_finished()
    check_drained(s)


def check_drained(s):
    for pool in s.kv_cache_manager.coordinator.block_pools:
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    ledger = s.kv_cache_manager.admission_ledger
    if ledger is not None:
        assert ledger.in_use == [0, 0, 0]
        assert ledger.hot_used == 0
        assert not ledger._hot
    assert not s._inflight_prefills
    assert not s.finished_recving_kv_req_ids


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("full_isl", [False, True])
def test_async_load_preserves_inflight_chunked_prefill(enabled, full_isl):
    s = scheduler(enabled=enabled, full_isl=full_isl)
    s.scheduler_config.long_prefill_token_threshold = SPAN
    requests = make_requests(2, 8 * SPAN, 1)
    s.connector.get_num_new_matched_tokens = lambda req, _: (
        (0, False) if req is requests[0] else (5 * SPAN, True)
    )
    for req in requests:
        s.add_request(req)
    for _ in range(128):
        output = s.schedule()
        if enabled and not requests[0].is_finished():
            assert requests[1].status == RequestStatus.WAITING
        complete(s, output, received={requests[1].request_id})
        if all(r.is_finished() for r in requests):
            break
    assert all(r.is_finished() for r in requests)
    assert requests[0].num_preemptions == 0
    check_drained(s)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("full_isl", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("matched", [5 * SPAN, 8 * SPAN])
@pytest.mark.parametrize("speculative_tokens", [0, 7])
def test_two_loads_keep_progress(
    enabled, full_isl, async_scheduling, matched, speculative_tokens
):
    """The review's two-load counterexample over 512 waves with caches cleared.

    Unlike a fixed number of idle schedule calls, every wave must finish.
    Completion order alternates and caching cannot hide a later wave's load.
    """
    s = scheduler(
        enabled=enabled,
        full_isl=full_isl,
        async_scheduling=async_scheduling,
        matched=matched,
        speculative_tokens=speculative_tokens,
    )
    for wave in range(512):
        assert s.kv_cache_manager.reset_prefix_cache()
        requests = make_requests(2, 8 * SPAN, 1, tag=f"wave{wave}-")
        for req in requests:
            s.add_request(req)
        active_steps = 0
        for tick in range(128):
            output = s.schedule()
            active_steps += bool(output.num_scheduled_tokens)
            parked = [
                r.request_id
                for r in requests
                if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and r.request_id not in s.finished_recving_kv_req_ids
            ]
            # Delay one completion while permitting reversed completion order.
            received = parked[wave % len(parked) :: 2] if parked else []
            if tick % 3 == 2:
                received = parked
            complete(s, output, received=received)
            ledger = s.kv_cache_manager.admission_ledger
            if ledger is not None:
                assert ledger.in_use == recount(s)
                assert ledger.hot_used <= ledger.plan.hot_pool_groups
            if all(r.is_finished() for r in requests):
                break
        assert all(r.status == RequestStatus.FINISHED_LENGTH_CAPPED for r in requests)
        assert active_steps >= 2
        check_drained(s)


@pytest.mark.parametrize("action", ["cancel", "fail"])
def test_parked_load_releases_domain_and_hot_reservations(action):
    s = scheduler(enabled=True, full_isl=False)
    requests = make_requests(2, 8 * SPAN, 1)
    for req in requests:
        s.add_request(req)
    output = s.schedule()
    assert requests[0].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert requests[1].status == RequestStatus.WAITING
    if action == "cancel":
        s.finish_requests(requests[0].request_id, RequestStatus.FINISHED_ABORTED)
    else:
        requests[0].num_computed_tokens = 0
        s.failed_recving_kv_req_ids.add(requests[0].request_id)
        s.connector.get_num_new_matched_tokens = lambda req, _: (0, False)
    complete(s, output, received={requests[0].request_id})
    for _ in range(128):
        output = s.schedule()
        complete(s, output, received={r.request_id for r in requests})
        if all(r.is_finished() for r in requests):
            break
    assert all(r.is_finished() for r in requests)
    check_drained(s)
