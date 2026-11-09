# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal reproductions of findings from the long-context scheduler fuzzer.

Each test states the expected behavior and is marked ``xfail(strict=True)``
while the defect exists; a fix turns it into an unexpected pass. See
``long_context_sched_fuzz.KNOWN_ISSUES`` for the driver-side avoidance.
"""

import pytest

from tests.v1.core.test_async_kv_admission import HybridMockConnector, complete
from tests.v1.core.test_long_context_admission import (
    MAX_MODEL_LEN,
    SPAN,
    build,
    make_requests,
    settings,
)
from tests.v1.core.test_scheduler_kv_capacity import step
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.long_context_kv import GiB
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _update(request_id: str, tokens: int, max_tokens: int, *, resumable=True):
    update = make_requests(1, tokens, max_tokens, tag=f"u{request_id}")[0]
    update.request_id = request_id
    update.resumable = resumable
    return update


@pytest.mark.parametrize(
    "domain,max_num_seqs,target_pages,full_isl",
    [
        # Default state segment sized for one request (max_num_seqs=1).
        ("state", 1, 8, False),
        ("state", 1, 8, True),
    ],
)
def test_k2_requeued_session_must_not_block_the_queue_head(
    domain, max_num_seqs, target_pages, full_isl
):
    """A streaming session sends its next input while its turn is running.

    When the turn stops, the queued input re-enters the waiting queue with
    the session's KV still held and behind a request that arrived meanwhile.
    The head needs blocks of the domain the session holds; nothing runs, so
    nothing is preempted, and the idle-session reclamation (load regression) only
    handles a
    HOT refusal. Both must finish without any client action.
    """
    s = build(
        settings(host_pool_tokens=10 * SPAN),
        max_num_seqs=max_num_seqs,
        reserve_full_isl=full_isl,
    )
    session = make_requests(1, 3 * SPAN - 8, 2, tag="session")[0]
    session.resumable = True
    s.add_request(session)
    while session.num_output_tokens == 0:  # decode its last token next
        step(s)
    assert session.status == RequestStatus.RUNNING
    s.add_request(_update(session.request_id, 16, 1))  # queued while running
    target = make_requests(1, target_pages * SPAN - 8, 1, tag="target")[0]
    s.add_request(target)
    ended = False
    for _ in range(256):
        step(s)
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ and not ended:
            s.add_request(_update(session.request_id, 1, 1, resumable=False))
            ended = True
        if target.is_finished() and session.is_finished():
            break
    assert target.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert session.is_finished()


def _async_spec_scheduler(enabled: bool):
    lc = settings(host_pool_tokens=40 * SPAN, hot_pool_gib=0.8) if enabled else None
    s = build(lc, speculative_tokens=7)
    s.vllm_config.scheduler_config.async_scheduling = True
    return AsyncScheduler(
        vllm_config=s.vllm_config,
        kv_cache_config=s.kv_cache_config,
        structured_output_manager=s.structured_output_manager,
        block_size=16,
        hash_block_size=16,
    )


def _output(s, output, accepted: int, sampling=None):
    ids = list(output.num_scheduled_tokens)
    sampled: list[list[int]] = []
    for rid in ids:
        if sampling is not None and rid not in sampling:
            sampled.append([])  # a partial prefill chunk samples nothing
            continue
        spec = len(output.scheduled_spec_decode_tokens.get(rid, ()))
        sampled.append([1] * (1 + min(accepted, spec)))
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _decode_once(s, request):
    """Prefill, then one decode step with seven draft placeholders."""
    while request.num_output_tokens == 0:
        output = s.schedule()
        sampling = {
            r for r in output.num_scheduled_tokens if not s.requests[r].is_prefill_chunk
        }
        s.update_from_output(output, _output(s, output, 0, sampling))
    assert request.num_output_tokens == 1
    output = s.schedule()
    assert len(output.scheduled_spec_decode_tokens[request.request_id]) == 7
    return output


@pytest.mark.parametrize("enabled", [False, True])
def test_k1_stale_async_frame_after_a_session_stops(enabled):
    """Async + DFlash + streaming input: two frames in flight at the stop.

    Not specific to the long-context mode (``enabled=False`` fails too).
    The first frame accepts enough drafts to reach max_tokens; the second was
    scheduled before that was known. The stopped session must be queued once
    and keep only the tokens of its turn.
    """
    s = _async_spec_scheduler(enabled)
    session = make_requests(1, 2 * SPAN, 4, tag="k1")[0]
    session.resumable = True
    s.add_request(session)
    first = _decode_once(s, session)
    second = s.schedule()  # scheduled before the first frame returns
    assert session.request_id in second.num_scheduled_tokens
    s.update_from_output(first, _output(s, first, 7))
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    turn = list(session.output_token_ids)
    # The turn keeps the KV up to its final sampled token, nothing beyond.
    assert session.num_computed_tokens == session.num_tokens - 1
    assert session.num_output_placeholders == 0
    s.update_from_output(second, _output(s, second, 7))
    queued = [r for r in (*s.waiting, *s.skipped_waiting) if r is session]
    assert len(queued) == 1
    assert list(session.output_token_ids) == turn
    assert session.num_in_flight_tokens == 0
    assert session.num_stale_in_flight_tokens == 0
    assert session.num_computed_tokens == session.num_tokens - 1
    # The next turn runs to completion.
    s.add_request(_update(session.request_id, 64, 3))
    s.add_request(_update(session.request_id, 1, 1, resumable=False))
    for _ in range(64):
        output = s.schedule()
        complete(s, output)
        if session.is_finished():
            break
    assert session.is_finished()


@pytest.mark.parametrize("queued_input", [False, True])
def test_preempted_session_stopped_by_its_in_flight_frame_stays_queued(
    queued_input,
):
    """Found by the fuzzer on the merged shared hot region tree (seed 185, async depth
    2).

    The scheduler preempts a running streaming session while the frame with
    its final token is still in flight. When that frame stops the session,
    the continuing session was enqueued again and then the batch removal of
    stopped preempted requests dropped every entry: it was in no queue.
    """
    s = _async_spec_scheduler(False)
    session = make_requests(1, 2 * SPAN, 4, tag="k7")[0]
    session.resumable = True
    s.add_request(session)
    first = _decode_once(s, session)
    if queued_input:
        s.add_request(_update(session.request_id, 64, 3))  # queued while running
    # Preempt as the running loop does, before the final frame returns.
    s.running.remove(session)
    s._preempt_request(session, 0.0)
    assert session.status == RequestStatus.PREEMPTED
    s.update_from_output(first, _output(s, first, 7))
    queued = [r for r in (*s.waiting, *s.skipped_waiting) if r is session]
    assert len(queued) == 1
    if not queued_input:
        s.add_request(_update(session.request_id, 64, 3))
    s.add_request(_update(session.request_id, 1, 1, resumable=False))
    for _ in range(64):
        output = s.schedule()
        complete(s, output)
        if session.is_finished():
            break
    assert session.is_finished()


def test_stale_async_frame_after_a_session_stops_with_queued_input():
    """Like K1, but the next input is already queued at the stop.

    The session re-enters the waiting queue at once and can be scheduled
    for its next turn while the stale frame is still in flight; that frame's
    output must still be dropped and the new turn must finish.
    """
    s = _async_spec_scheduler(False)
    session = make_requests(1, 2 * SPAN, 4, tag="k1q")[0]
    session.resumable = True
    s.add_request(session)
    first = _decode_once(s, session)
    second = s.schedule()
    s.add_request(_update(session.request_id, 64, 3))  # queued while running
    s.update_from_output(first, _output(s, first, 7))
    assert session.status == RequestStatus.WAITING
    prompt = list(session.prompt_token_ids)
    third = s.schedule()  # the next turn starts before the stale frame returns
    assert session.request_id in third.num_scheduled_tokens
    s.update_from_output(second, _output(s, second, 7))
    assert list(session.prompt_token_ids) == prompt
    assert session.num_output_tokens == 0
    assert session.num_stale_in_flight_tokens == 0
    s.update_from_output(third, _output(s, third, 0))
    s.add_request(_update(session.request_id, 1, 1, resumable=False))
    for _ in range(64):
        output = s.schedule()
        complete(s, output)
        if session.is_finished():
            break
    assert session.is_finished()


@pytest.mark.parametrize("enabled", [False, True])
def test_k3_session_turn_ends_without_stale_placeholders(enabled):
    """Async + DFlash + streaming input, one frame in flight.

    Not specific to the long-context mode (``enabled=False`` fails too).
    The decode frame accepts all seven drafts but max_tokens trims the turn.
    The placeholders of the trimmed tokens must not survive into the next
    turn, which must then run to completion.
    """
    s = _async_spec_scheduler(enabled)
    session = make_requests(1, 2 * SPAN, 3, tag="k3")[0]
    session.resumable = True
    s.add_request(session)
    output = _decode_once(s, session)
    s.update_from_output(output, _output(s, output, 7))
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert session.num_output_placeholders == 0
    s.add_request(_update(session.request_id, 64, 3))
    s.add_request(_update(session.request_id, 1, 1, resumable=False))
    for _ in range(64):
        output = s.schedule()
        complete(s, output)
        if session.is_finished():
            break
    assert session.is_finished()


def _drive_frames(s, requests, steps: int, frames: int) -> int:
    """frames=2 keeps one output in flight, as the async engine does."""
    for request in requests:
        s.add_request(request)
    pending = None
    preemptions = 0
    for _ in range(steps):
        output = s.schedule()
        preemptions += len(output.preempted_req_ids or ())
        sampling = {
            rid
            for rid in output.num_scheduled_tokens
            if not s.requests[rid].is_prefill_chunk
        }
        if pending is not None:
            s.update_from_output(*pending)
        pending = (output, _output(s, output, 0, sampling))
        if frames == 1:
            s.update_from_output(*pending)
            pending = None
        if all(r.is_finished() for r in requests):
            break
    return preemptions


@pytest.mark.parametrize(
    "frames",
    [
        1,
        2,
    ],
)
def test_k5_one_request_finishes_in_a_one_request_state_segment(frames):
    """Async (two frames in flight) + DFlash, state segment for one request.

    Startup used to accept ``state_blocks == state_blocks_per_request``; while
    two chunks are in flight the state demand exceeds that bound and a lone
    long prefill preempted itself forever. Startup now requires the in-flight
    peak, and with that size the request finishes.
    """
    probe = build(
        settings(host_pool_tokens=40 * SPAN, hot_pool_gib=0.8),
        speculative_tokens=7,
        async_scheduling=True,
    )
    per_request = probe.kv_cache_manager.admission_ledger.plan.geometry
    # Startup now requires the in-flight peak, refusing the livelocking size.
    assert per_request.state_inflight_blocks > 0
    with pytest.raises(ValueError, match="asynchronous frame"):
        build(
            settings(
                host_pool_tokens=40 * SPAN,
                hot_pool_gib=0.8,
                state_blocks=per_request.state_blocks_per_request,
            ),
            speculative_tokens=7,
            async_scheduling=True,
        )
    s = build(
        settings(
            host_pool_tokens=40 * SPAN,
            hot_pool_gib=0.8,
            state_blocks=per_request.state_blocks_per_request
            + per_request.state_inflight_blocks,
        ),
        speculative_tokens=7,
        async_scheduling=True,
    )
    s.vllm_config.scheduler_config.async_scheduling = True
    s = AsyncScheduler(
        vllm_config=s.vllm_config,
        kv_cache_config=s.kv_cache_config,
        structured_output_manager=s.structured_output_manager,
        block_size=16,
        hash_block_size=16,
    )
    assert not s.kv_cache_manager.admission_ledger.plan.startup_problems()
    request = make_requests(1, 9 * SPAN - 217, 16, tag="k5")[0]
    preemptions = _drive_frames(s, [request], 256, frames)
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED, preemptions


@pytest.mark.parametrize("spec", [0, 7])
@pytest.mark.parametrize(
    "policy",
    [
        "fcfs",
        "priority",
    ],
)
def test_k4_completed_loads_are_promoted_under_priority(policy, spec):
    """Two lower-priority async loads complete; a higher-priority request
    arrives before they are promoted and needs the state blocks they hold.

    FCFS visits the skipped (parked) queue first. PRIORITY compares queue
    heads, picks the newcomer, cannot allocate it and stops; parked loads
    cannot be preempted, so the loads are never promoted.
    """
    probe = build(
        settings(host_pool_tokens=40 * SPAN, hot_pool_gib=0.8),
        speculative_tokens=spec,
    )
    geo = probe.kv_cache_manager.admission_ledger.plan.geometry
    base = build(
        settings(
            host_pool_tokens=40 * SPAN,
            hot_pool_gib=0.8,
            state_blocks=2 * geo.state_blocks_per_request,
        ),
        max_num_seqs=4,
        speculative_tokens=spec,
    )
    config = base.vllm_config
    config.scheduler_config.policy = policy
    config.kv_transfer_config = None
    s = Scheduler(
        vllm_config=config,
        kv_cache_config=base.kv_cache_config,
        structured_output_manager=base.structured_output_manager,
        block_size=16,
        hash_block_size=16,
    )
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MockKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"matched_tokens": 0, "is_async": True},
    )
    s.connector = HybridMockConnector(
        config, KVConnectorRole.SCHEDULER, s.kv_cache_config
    )
    loads = make_requests(2, 8 * SPAN, 1, tag="load")
    for request in loads:
        request.priority = 2
        s.add_request(request)

    def matched(req, local):
        # Like the real connector, report only tokens beyond the local hit;
        # once promoted, a load can be preempted and looked up again.
        external = max(5 * SPAN - local, 0) if req in loads else 0
        return external, external > 0

    s.connector.get_num_new_matched_tokens = matched
    output = s.schedule()
    assert all(r.status == RequestStatus.WAITING_FOR_REMOTE_KVS for r in loads)
    complete(s, output, received={r.request_id for r in loads})
    urgent = make_requests(1, 8 * SPAN, 1, tag="urgent")[0]
    urgent.priority = 0
    s.add_request(urgent)
    for _ in range(128):
        complete(s, s.schedule())
        if urgent.is_finished() and all(r.is_finished() for r in loads):
            break
    assert urgent.is_finished() and all(r.is_finished() for r in loads)


def test_k4_completed_load_behind_a_skipped_request_under_fcfs():
    """FCFS form of K4, with a hot pool for one max-length request.

    The connector cannot answer the first request's lookup twice (the real
    connector returns None while a preempted request's stores are pending),
    so it sits in skipped_waiting. A later request's async load completes
    behind it and holds the hot quota. The earlier request needs that quota:
    HOT refuses, load regression skips parked loads, the scan stops, and the completed
    load is never promoted.
    """
    probe = build(settings())
    geo = probe.kv_cache_manager.admission_ledger.plan.geometry
    cost = geo.hot_bytes_for_groups(geo.hot_reserve_groups(MAX_MODEL_LEN), 8)
    s = build(settings(host_pool_tokens=40 * SPAN, hot_pool_gib=cost / GiB + 1e-9))
    config = s.vllm_config
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MockKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"matched_tokens": 0, "is_async": True},
    )
    s.connector = HybridMockConnector(
        config, KVConnectorRole.SCHEDULER, s.kv_cache_config
    )
    early, load = make_requests(2, 8 * SPAN, 1, tag="k4f")
    unanswered = [2]

    def matched(req, _):
        if req is load:
            return 5 * SPAN, True
        if unanswered[0]:
            unanswered[0] -= 1
            return None, False
        return 0, False

    s.connector.get_num_new_matched_tokens = matched
    s.add_request(early)
    complete(s, s.schedule())
    s.add_request(load)
    output = s.schedule()
    assert load.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert [r.request_id for r in s.skipped_waiting] == ["k4f0", "k4f1"]
    complete(s, output, received={load.request_id})
    for _ in range(64):
        complete(s, s.schedule())
        if early.is_finished() and load.is_finished():
            break
    assert early.is_finished() and load.is_finished()


@pytest.mark.parametrize("enabled", [False, True])
def test_k6_queued_input_after_a_trimmed_spec_step_does_not_crash(enabled):
    """Sync scheduler + DFlash + streaming input queued while running.

    The decode step accepts all seven drafts, max_tokens keeps one of them,
    and the queued input resumes the session. num_computed_tokens must not
    count the trimmed drafts; otherwise the resumed prompt can be fully
    "computed" and ``schedule()`` asserts ``num_new_tokens > 0``. Not specific
    to the long-context mode.
    """
    lc = settings(host_pool_tokens=40 * SPAN, hot_pool_gib=0.8) if enabled else None
    s = build(lc, speculative_tokens=7)
    session = make_requests(1, 2 * SPAN, 2, tag="k6")[0]
    session.resumable = True
    s.add_request(session)
    while session.num_output_tokens == 0:
        output = s.schedule()
        sampling = {
            rid
            for rid in output.num_scheduled_tokens
            if not s.requests[rid].is_prefill_chunk
        }
        s.update_from_output(output, _output(s, output, 0, sampling))
    s.update_draft_token_ids(DraftTokenIds([session.request_id], [[1] * 7]))
    s.add_request(_update(session.request_id, 3, 1))  # queued while running
    output = s.schedule()
    assert len(output.scheduled_spec_decode_tokens[session.request_id]) == 7
    s.update_from_output(output, _output(s, output, 7))
    assert session.status == RequestStatus.WAITING
    # The resumed prompt keeps the turn's computed tokens, not the drafts.
    assert session.num_computed_tokens == session.num_prompt_tokens - 3
    s.add_request(_update(session.request_id, 1, 1, resumable=False))
    for _ in range(32):
        output = s.schedule()
        complete(s, output)
        if session.is_finished():
            break
    assert session.is_finished()
