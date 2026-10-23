# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""reset_prefix_cache(reset_running_requests=True) preempts running requests
outside schedule(). Their IDs reach the connector in the next schedule()
through preempted_req_ids. If such a request has no transfer left in flight,
the same schedule() can re-admit it from the waiting queue with a CPU hit and
issue a fresh load into newly allocated blocks before the connector processes
the preemption."""

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_reset_lifecycle import (
    BS,
    LAYOUTS,
    NONE,
    TP4,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.v1.request import RequestStatus


@pytest.fixture(autouse=True)
def _gpu_platform_hma(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)


def _run_until_idle_transfers(h: TP4, rid: str, max_steps: int = 8) -> None:
    """Decode until the request runs with no transfer job in flight."""
    req = h.sched.requests[rid]
    for _ in range(max_steps):
        if (
            req.status == RequestStatus.RUNNING
            and not h.cs._req_status[rid].transfer_jobs
        ):
            return
        h.step(token=0)
    raise AssertionError(
        f"request {rid} never ran without transfers: "
        f"{req.status} {h.cs._req_status[rid].transfer_jobs}"
    )


def _finish(h: TP4) -> None:
    for _ in range(3):
        h.step(token=0)
    for _ in range(10):
        if all(r.is_finished() for r in h.sched.requests.values()):
            break
        h.step(token=EOS_TOKEN_ID)
    h.drain()
    h.assert_clean()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("loaded_first", [True, False])
def test_reset_running_reloads_request_without_inflight_jobs(
    request_runner, layout, loaded_first
):
    h = TP4(request_runner, layout)
    if loaded_first:
        # Observed shape: the prefix came from the CPU tier, the load
        # finished and the request is running when the reset arrives.
        h.hit(1)
    else:
        h.miss()
    rid = h.add(h.obs * 2, 0)
    _run_until_idle_transfers(h, rid)
    req = h.sched.requests[rid]
    recv_before = h.recv_count(rid)
    assert recv_before == int(loaded_first)

    # GPU prefix cache only: the CPU tier still holds the prefix.
    assert h.sched.reset_prefix_cache(reset_running_requests=True)
    assert req.status == RequestStatus.PREEMPTED
    h.hit(1)

    so = h.step(complete=NONE)
    meta = so.kv_connector_metadata
    assert rid in so.preempted_req_ids
    # Re-admitted in the same step with a fresh load into new blocks.
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert {job.req_id for job in meta.load_jobs.values()} == {rid}
    # The new load is not a transfer of the freed blocks: do not flush it.
    assert meta.load_jobs.keys().isdisjoint(meta.jobs_to_flush)
    assert h.cs._req_status[rid].transfer_jobs == set(meta.load_jobs)

    h.step()  # every rank completes the load
    assert h.recv_count(rid) == recv_before + 1
    _finish(h)
    assert h.recv_count(rid) == recv_before + 1


def test_reset_running_mixed_pending_stores_and_reload(request_runner):
    """One request still has stores in flight (flushed, re-admission
    deferred); the other has none (reloaded in the same step)."""
    h = TP4(request_runner, "single")
    h.miss()
    idle = h.add(h.obs * 2, 1)
    _run_until_idle_transfers(h, idle)
    busy = h.add(h.obs * 2, 2)
    skew = (True, False, False, False)
    h.step(complete=skew)
    for _ in range(2 * h.obs - BS):
        h.step(complete=skew, token=0)
        if (
            not h.cs._req_status[idle].transfer_jobs
            and h.cs._req_status[busy].transfer_jobs
        ):
            break
    pending = set(h.cs._req_status[busy].transfer_jobs)
    assert pending, "expected pending stores"
    assert not h.cs._req_status[idle].transfer_jobs

    assert h.sched.reset_prefix_cache(reset_running_requests=True)
    h.hit(1)
    so = h.step(complete=NONE)
    meta = so.kv_connector_metadata
    assert {idle, busy} <= so.preempted_req_ids
    assert pending <= meta.jobs_to_flush
    assert {job.req_id for job in meta.load_jobs.values()} == {idle}
    assert meta.load_jobs.keys().isdisjoint(meta.jobs_to_flush)
    assert h.sched.requests[idle].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert h.sched.requests[busy].status == RequestStatus.PREEMPTED
    _finish(h)
