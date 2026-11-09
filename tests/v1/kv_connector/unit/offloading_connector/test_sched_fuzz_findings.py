# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connector-side finding of the long-context scheduler fuzz review.

``reset_prefix_cache(reset_connector=True)`` while an external load is in
flight marks the load job stale. When that load then fails, the failure must
still void the external hit: the request is recomputed from token 0, exactly
as without the reset, and is never promoted as if the bytes had arrived.
"""

from dataclasses import replace

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_reset_lifecycle import (
    NONE,
    TP4,
    _gpu_platform_hma,  # noqa: F401  (autouse fixture)
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize(
    "reset",
    ["none", "gpu_only", "connector"],
)
def test_failed_inflight_load_is_voided_across_reset(request_runner, reset):
    h = TP4(request_runner, "two_group")
    h.cs._recompute_failed_loads = True
    h.sched.recompute_failed_loads = True
    for worker in h.workers:
        worker.connector_worker._report_failures = True
    h.hit(1)
    rid = h.add(h.obs * 2, 0)
    h.step(complete=NONE)
    request = h.sched.requests[rid]
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert h.cs.num_abandoned_loads == 0
    voided: list[str] = []
    void = h.sched._void_failed_loads

    def record(req_ids):
        voided.extend(sorted(req_ids))
        return void(req_ids)

    h.sched._void_failed_loads = record
    for worker in h.workers:
        handler = worker.connector_worker.spec.handler
        complete = handler.complete_jobs

        def fail(job_ids, handler=handler, complete=complete):
            complete(job_ids)
            handler.completed_transfers = [
                replace(result, success=False) for result in handler.completed_transfers
            ]

        handler.complete_jobs = fail
    if reset != "none":
        # A busy GPU pool refuses its own reset; the connector reset proceeds.
        assert not h.sched.reset_prefix_cache(reset_connector=reset == "connector")
    h.miss()
    h.step()  # the failed load completes on every rank
    assert voided == [rid]
    assert request.num_computed_tokens == 0
    so = h.step()
    # Recomputed from token 0, not resumed after the unloaded external hit.
    assert so.num_scheduled_tokens[rid] == h.obs * 2


def _inflight_load_then_reset(request_runner, recompute: bool):
    h = TP4(request_runner, "two_group")
    h.cs._recompute_failed_loads = recompute
    h.sched.recompute_failed_loads = recompute
    for worker in h.workers:
        worker.connector_worker._report_failures = recompute
    h.hit(1)
    rid = h.add(h.obs * 2, 0)
    h.step(complete=NONE)
    request = h.sched.requests[rid]
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    (job_id,) = h.cs._jobs
    hit = request.num_computed_tokens
    assert hit > 0
    assert not h.sched.reset_prefix_cache(reset_connector=True)
    return h, rid, job_id, hit


@pytest.mark.parametrize("recompute", [False, True])
def test_successful_inflight_load_keeps_its_hit_across_reset(request_runner, recompute):
    h, rid, job_id, hit = _inflight_load_then_reset(request_runner, recompute)
    assert (job_id in h.cs._stale_load_jobs) == recompute
    h.miss()
    h.step()  # the load succeeds on every rank
    request = h.sched.requests[rid]
    assert h.cs.num_abandoned_loads == 0
    assert not h.cs._stale_load_jobs
    so = h.step()
    # Resumed after the loaded external hit.
    assert request.num_computed_tokens >= hit
    assert so.num_scheduled_tokens[rid] == h.obs * 2 - hit


def test_stale_load_failure_on_one_rank_is_voided_when_all_ranks_report(
    request_runner,
):
    h, rid, job_id, _ = _inflight_load_then_reset(request_runner, True)
    num_workers = len(h.workers)
    # One rank reports the failure first; the job is done only at num_workers.
    h.cs.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: 1}, failed_jobs={job_id: 1}
            )
        )
    )
    assert h.cs.num_abandoned_loads == 0
    assert job_id in h.cs._stale_load_jobs
    h.cs.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: num_workers - 1}
            )
        )
    )
    assert h.cs.num_abandoned_loads == 1
    assert not h.cs._stale_load_jobs
    assert h.cs.take_failed_loads() == {rid}
