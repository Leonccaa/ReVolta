# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""load regression. Retained streaming sessions must allow each other's hot quota to
grow."""

import pytest

from tests.v1.core.test_long_context_admission import (
    MAX_MODEL_LEN,
    build,
    make_requests,
    recount,
    settings,
)
from tests.v1.core.test_long_context_async_admission import check_drained
from tests.v1.core.test_scheduler_kv_capacity import step
from vllm.v1.core.long_context_kv import GiB
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _scheduler(enabled, full_isl, async_scheduling):
    probe = build(settings())
    geometry = probe.kv_cache_manager.admission_ledger.plan.geometry
    slots = geometry.hot_reserve_groups(MAX_MODEL_LEN)
    cost = geometry.hot_bytes_for_groups(slots, 8)
    s = build(
        settings(hot_pool_gib=cost / GiB + 1e-9) if enabled else None,
        reserve_full_isl=full_isl,
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
    if enabled:
        ledger = s.kv_cache_manager.admission_ledger
        assert ledger.plan.hot_pool_groups == slots == 522
        assert not ledger.plan.startup_problems()
    return s


def _append(s, request, tokens):
    update = make_requests(1, tokens, 1, tag="update")[0]
    update.request_id = request.request_id
    update.resumable = True
    s.add_request(update)


def _end(s, request):
    sentinel = make_requests(1, 1, 1, tag="end")[0]
    sentinel.request_id = request.request_id
    s.add_request(sentinel)


def _until(s, predicate):
    for _ in range(128):
        step(s)
        ledger = s.kv_cache_manager.admission_ledger
        if ledger is not None:
            assert ledger.in_use == recount(s)
            assert 0 <= ledger.hot_used <= ledger.plan.hot_pool_groups
        if predicate():
            return
    pytest.fail("streaming workload made no bounded progress")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("full_isl", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("initial", [1000, 2000])
def test_two_streams_grow_and_finish(enabled, full_isl, async_scheduling, initial):
    s = _scheduler(enabled, full_isl, async_scheduling)
    for wave in range(256):
        assert s.kv_cache_manager.reset_prefix_cache()
        requests = make_requests(2, initial, 1, tag=f"stream{wave}-")
        for request in requests:
            request.resumable = True
            s.add_request(request)
        _until(s, lambda: s.num_waiting_for_streaming_input == 2)
        assert all(
            r.status == RequestStatus.WAITING_FOR_STREAMING_REQ for r in requests
        )
        for request in requests:
            _append(s, request, 5000 - initial)
            _end(s, request)
        _until(s, lambda requests=requests: all(r.is_finished() for r in requests))
        assert all(r.status == RequestStatus.FINISHED_LENGTH_CAPPED for r in requests)
        if enabled:
            assert sum(r.num_preemptions for r in requests) >= 1
        assert s.num_waiting_for_streaming_input == 0
        assert not s.requests and not s.waiting and not s.skipped_waiting
        check_drained(s)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("max_tokens", [1, 3])
def test_idle_peer_can_be_recomputed_after_later_input(async_scheduling, max_tokens):
    s = _scheduler(True, False, async_scheduling)
    first, peer = make_requests(2, 1000, max_tokens)
    for request in (first, peer):
        request.resumable = True
        s.add_request(request)
    _until(s, lambda: s.num_waiting_for_streaming_input == 2)
    _append(s, first, 4000)
    _end(s, first)
    _until(s, first.is_finished)
    assert peer.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert peer.num_computed_tokens == 0 and peer.num_preemptions == 1
    assert s.num_waiting_for_streaming_input == 1
    retained = peer.all_token_ids[:-1]
    _append(s, peer, 4000)
    assert list(peer.all_token_ids) == retained + [1000] * 4000
    assert peer.num_prompt_tokens == 5000 + max_tokens - 1
    _end(s, peer)
    _until(s, peer.is_finished)
    assert s.num_waiting_for_streaming_input == 0
    check_drained(s)
