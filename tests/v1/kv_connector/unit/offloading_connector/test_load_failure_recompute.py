# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A broken external KV load is recomputed, never crashes the engine.

With ``load_failure_recompute_enabled`` (the cross-group hit contract or the
long-context mode), three failures void a request's external hit instead of
raising or asserting:

* a key is evicted after the hit was committed (between lookup and the load
  plan),
* only some KV cache groups are still readable when the load is planned,
* a worker reports the load transfer as failed.

The scheduler then releases the request's blocks (none of the loaded blocks
was published to the prefix cache), schedules it again from its local hit
with the external store skipped once, and it finishes normally. Other
requests are not touched.
"""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import tests.v1.kv_connector.unit.offloading_connector.utils as runner_utils
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.config.kernel import LongContextKVConfig
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import get_offload_group_idx
from vllm.v1.kv_offload.worker.worker import TransferResult
from vllm.v1.request import RequestStatus

BLOCK_SIZE = 4
NUM_GPU_BLOCKS = 100


def _groups() -> list[KVCacheGroupSpec]:
    return [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=2 * BLOCK_SIZE,
            ),
        ),
    ]


def _runner(request_runner, monkeypatch, gate: str, async_scheduling: bool):
    """A two-group runner with the recompute path switched on by ``gate``."""
    create = runner_utils.create_vllm_config

    def create_with_gate(*args: Any, **kwargs: Any):
        config = create(*args, **kwargs)
        if gate == "contract":
            config.kernel_config.qsa_cross_group_hit_check = True
        elif gate == "long_context":
            config.kernel_config.long_context_kv = LongContextKVConfig(
                enabled=True, host_pool_gib=1.0
            )
        return config

    monkeypatch.setattr(runner_utils, "create_vllm_config", create_with_gate)
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        async_scheduling=async_scheduling,
        kv_cache_groups=_groups(),
    )
    assert runner.scheduler.recompute_failed_loads is (gate != "off")
    # The tier holds exactly what was stored (runner.tier).
    runner.tier = set()

    def store(keys, req_context):
        keys = list(keys)
        runner.tier.update(keys)
        return generate_store_output(keys)

    runner.manager.prepare_store.side_effect = store
    runner.manager.lookup.side_effect = lambda key, req_context: key in runner.tier
    return runner


class _Recorder:
    """Watches a runner's scheduler for voided loads and released blocks."""

    def __init__(self, runner):
        self.runner = runner
        scheduler = runner.scheduler
        self.voided: list[str] = []
        self.unpublished_release: list[bool] = []
        void = scheduler._void_failed_loads

        def record_void(req_ids):
            req_ids = set(req_ids)
            self.voided.extend(sorted(req_ids))
            return void(req_ids)

        scheduler._void_failed_loads = record_void
        manager = scheduler.kv_cache_manager
        free = manager.free

        def record_free(request):
            if request.request_id in scheduler.failed_recving_kv_req_ids:
                # The failed load's blocks must leave without a prefix hash:
                # nothing restored by it may serve another request.
                self.unpublished_release.append(
                    all(
                        block.block_hash is None
                        for blocks in manager.coordinator.get_blocks(request.request_id)
                        for block in blocks
                    )
                )
            return free(request)

        manager.free = record_free

        handler = runner.offloading_spec.handler
        self.load_jobs: list[int] = []
        transfer = handler.transfer_async

        def record_transfer(job_id, spec):
            src, _ = spec
            if isinstance(src, runner_utils.MockLoadStoreSpec):
                self.load_jobs.append(job_id)
            return transfer(job_id, spec)

        handler.transfer_async = record_transfer


def _store_prefix(runner, num_blocks: int) -> list[int]:
    """Run one request to completion so its prompt is offloaded, then drop
    the GPU prefix cache so the next identical prompt hits only the tier."""
    prompt = [0] * (BLOCK_SIZE * num_blocks)
    runner.new_request(token_ids=prompt)
    runner._run(decoded_tokens=[0, 0, EOS_TOKEN_ID], complete_transfers=True)
    runner.completed_loads.clear()
    runner.completed_stores.clear()
    runner.flushed_gpu_blocks.clear()
    assert not runner.scheduler.requests
    # Both groups of every prompt block reached the tier.
    assert {get_offload_group_idx(k) for k in runner.tier} == {0, 1}
    assert runner.scheduler.reset_prefix_cache()
    return prompt + [0]


def _check_pool_drained(runner) -> None:
    for pool in runner.scheduler.kv_cache_manager.coordinator.block_pools:
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


def _finish(runner, recorder: _Recorder) -> None:
    runner._run(decoded_tokens=[0, 0, EOS_TOKEN_ID], complete_transfers=True)
    runner.completed_loads.clear()
    runner.completed_stores.clear()
    runner.flushed_gpu_blocks.clear()
    assert not runner.scheduler.requests
    _check_pool_drained(runner)


@pytest.mark.parametrize("gate", ["contract", "long_context"])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_key_evicted_after_commit_recomputes(
    request_runner, monkeypatch, gate: str, async_scheduling: bool
):
    runner = _runner(request_runner, monkeypatch, gate, async_scheduling)
    recorder = _Recorder(runner)
    prompt = _store_prefix(runner, 3)

    cs = runner.connector_scheduler
    plan = cs.update_state_after_alloc

    def evict_then_plan(request, blocks, num_external_tokens):
        if num_external_tokens:
            runner.tier.clear()  # the tier lost every key after the lookup
        return plan(request, blocks, num_external_tokens)

    cs.update_state_after_alloc = evict_then_plan
    runner.new_request(token_ids=prompt)
    request = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, recorder)

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert recorder.load_jobs == []  # nothing was read back
    # Abandoned at plan time: parked as a finished failed receive directly.
    assert cs.num_abandoned_loads == 1
    assert recorder.voided == []
    assert recorder.unpublished_release == [True]
    assert not cs._skip_external_once  # the retry consumed it


@pytest.mark.parametrize("gate", ["contract", "long_context"])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_one_group_missing_at_plan_time_recomputes(
    request_runner, monkeypatch, gate: str, async_scheduling: bool
):
    runner = _runner(request_runner, monkeypatch, gate, async_scheduling)
    recorder = _Recorder(runner)
    prompt = _store_prefix(runner, 3)

    cs = runner.connector_scheduler
    plan = cs.update_state_after_alloc

    def lose_window_group(request, blocks, num_external_tokens):
        if num_external_tokens:
            # Full attention is still there; the sliding-window group is not.
            runner.tier = {k for k in runner.tier if get_offload_group_idx(k) == 0}
        return plan(request, blocks, num_external_tokens)

    cs.update_state_after_alloc = lose_window_group
    runner.new_request(token_ids=prompt)
    request = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, recorder)

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert recorder.load_jobs == []
    assert cs.num_abandoned_loads == 1
    assert recorder.voided == []
    assert recorder.unpublished_release == [True]


@pytest.mark.parametrize("gate", ["contract", "long_context"])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_failed_load_transfer_recomputes_only_that_request(
    request_runner, monkeypatch, gate: str, async_scheduling: bool
):
    runner = _runner(request_runner, monkeypatch, gate, async_scheduling)
    recorder = _Recorder(runner)
    prompt = _store_prefix(runner, 3)

    handler = runner.offloading_spec.handler
    complete = handler.complete_jobs

    def fail_loads(job_ids):
        complete(job_ids)
        handler.completed_transfers = [
            TransferResult(
                job_id=r.job_id,
                success=r.job_id not in recorder.load_jobs,
                transfer_size=None,
                transfer_time=None,
                transfer_type=None,
            )
            for r in handler.completed_transfers
        ]

    handler.complete_jobs = fail_loads
    runner.new_request(token_ids=prompt)
    failed = runner.scheduler.requests[str(runner.req_id)]
    # A request with another prompt runs in the same steps and misses.
    runner.new_request(token_ids=[7] * (BLOCK_SIZE * 2 + 1))
    other = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, recorder)

    assert len(recorder.load_jobs) == 1  # one load, and it failed
    assert recorder.voided == [failed.request_id]
    assert recorder.unpublished_release == [True]
    assert failed.status == other.status == RequestStatus.FINISHED_STOPPED
    assert other.num_preemptions == 0
    cs = runner.connector_scheduler
    assert cs.num_abandoned_loads == 1
    # The failed job's bookkeeping was released like a completed one.
    assert not cs._jobs


def test_disabled_gate_keeps_the_assert(request_runner, monkeypatch):
    runner = _runner(request_runner, monkeypatch, "off", async_scheduling=False)
    prompt = _store_prefix(runner, 2)
    handler = runner.offloading_spec.handler
    complete = handler.complete_jobs

    def fail_all(job_ids):
        complete(job_ids)
        handler.completed_transfers = [
            TransferResult(
                job_id=r.job_id,
                success=False,
                transfer_size=None,
                transfer_time=None,
                transfer_type=None,
            )
            for r in handler.completed_transfers
        ]

    handler.complete_jobs = fail_all
    runner.new_request(token_ids=prompt)
    with pytest.raises(AssertionError):
        runner._run(decoded_tokens=[0, EOS_TOKEN_ID], complete_transfers=True)


@pytest.mark.parametrize(
    "failure", ["short_table", "local_alignment", "short_keys", "prepare"]
)
@pytest.mark.parametrize("gate", ["contract", "long_context"])
def test_preparation_errors_recompute_before_creating_job(
    request_runner, monkeypatch, failure, gate
):
    runner = _runner(request_runner, monkeypatch, gate, async_scheduling=False)
    prompt = _store_prefix(runner, 3)
    recorder = _Recorder(runner)
    cs = runner.connector_scheduler
    prepare = cs.update_state_after_alloc

    def fail_plan(request, blocks, external):
        if external:
            state = cs._req_status[request.request_id]
            if failure == "short_table":
                blocks = KVCacheBlocks(
                    tuple(
                        [] if i == 0 else group for i, group in enumerate(blocks.blocks)
                    )
                )
            elif failure == "local_alignment":
                state.num_locally_computed_tokens = 1
            elif failure == "short_keys":
                state.group_states[0].offload_keys.clear()
            else:

                def fail_prepare(*args):
                    raise RuntimeError("prepare failed")

                runner.manager.prepare_load = fail_prepare
        return prepare(request, blocks, external)

    cs.update_state_after_alloc = fail_plan
    runner.new_request(token_ids=prompt)
    request = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, recorder)
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert cs.num_abandoned_loads == 1
    assert recorder.load_jobs == []
    assert recorder.unpublished_release == [True]
    assert not cs._jobs
    assert not cs._current_batch_load_jobs


@pytest.mark.parametrize("is_load", [False, True])
@pytest.mark.parametrize("failure", ["false", "exception"])
def test_rejected_worker_submission_reports_failure_after_drain(
    request_runner, monkeypatch, is_load, failure
):
    runner = _runner(request_runner, monkeypatch, "contract", async_scheduling=False)
    worker = runner.worker_connector.connector_worker
    calls = []

    def reject(job_id, spec):
        if failure == "exception":
            raise OSError("injected submission failure")
        return False

    worker.worker.transfer_async = reject
    worker.worker.wait = lambda job_ids: calls.append(set(job_ids))
    worker.worker.get_finished = lambda: []
    if is_load:
        worker._load_jobs[42] = "failed"
    worker._submit_transfer(42, (None, None))
    assert calls == [{42}]
    assert worker.get_finished(set()) == (set(), {"failed"} if is_load else set())
    meta = worker.build_connector_worker_meta()
    assert meta.failed_jobs == {42: 1}
    assert meta.completed_jobs == {42: 1}
    assert worker.get_finished(set()) == (set(), set())
    assert worker.build_connector_worker_meta() is None


@pytest.mark.parametrize("preempted", [False, True])
@pytest.mark.parametrize("failure", ["false", "exception"])
def test_deferred_store_submission_reports_failure(
    request_runner, monkeypatch, preempted, failure
):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
        TransferJob,
    )

    runner = _runner(request_runner, monkeypatch, "contract", False)
    worker = runner.worker_connector.connector_worker
    calls = []

    def reject(job_id, spec):
        calls.append("submitted")
        if failure == "exception":
            raise OSError("injected deferred store submission error")
        return False

    worker.worker.transfer_async = reject
    worker.worker.wait = lambda job_ids: calls.append("drained")
    worker.worker.get_finished = lambda: []
    metadata = OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={
            42: TransferJob(req_id="store", transfer_spec=(None, None)),
        },
    )
    worker.prepare_store_kv(metadata)
    assert not calls
    if preempted:
        worker.handle_preemptions(
            OffloadingConnectorMetadata(load_jobs={}, store_jobs={}, jobs_to_flush={42})
        )
    else:
        worker.start_kv_transfers(
            OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
        )
    assert calls[:2] == ["submitted", "drained"]
    assert not worker._unsubmitted_store_jobs
    assert worker.get_finished(set()) == (set(), set())
    result = worker.build_connector_worker_meta()
    assert result.failed_jobs == result.completed_jobs == {42: 1}
    assert worker.get_finished(set()) == (set(), set())
    assert worker.build_connector_worker_meta() is None


@pytest.mark.parametrize("failure", ["copy", "end_event"])
def test_partial_submission_drains_before_failure_is_returned(monkeypatch, failure):
    from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec
    from vllm.v1.kv_offload.cpu import gpu_worker
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    calls = []
    stream = SimpleNamespace(
        wait_stream=lambda other: None, synchronize=lambda: calls.append("drained")
    )
    start = SimpleNamespace(record=lambda other: calls.append("start"))

    def record_end(other):
        calls.append("end")
        raise RuntimeError("event failed")

    end = SimpleNamespace(record=record_end)

    def copy(*args, **kwargs):
        calls.append("copies enqueued")
        if failure == "copy":
            raise RuntimeError("batch failed after partial submission")

    handler = gpu_worker.SingleDirectionOffloadingHandler.__new__(
        gpu_worker.SingleDirectionOffloadingHandler
    )
    handler.gpu_to_cpu = False
    handler.src_tensors = [torch.empty((1, 8), dtype=torch.int8)]
    handler.dst_tensors = [torch.empty((1, 8), dtype=torch.int8)]
    handler.src_block_size_factor = handler.dst_block_size_factor = 1
    handler.kv_cache_groups_data_refs = [[CanonicalKVCacheRef(0, 8)]]
    handler._stream_pool = [stream]
    handler._event_pool = [end, start]
    handler._transfers = deque()
    handler._transfer_events = {}
    handler._failed_transfers = {}
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda other: nullcontext())
    monkeypatch.setattr(gpu_worker.ops, "swap_blocks_batch", copy)
    with pytest.raises(RuntimeError):
        handler.transfer_async(
            42, (CPULoadStoreSpec([0]), GPULoadStoreSpec([0], [1], [0]))
        )
    assert calls[-1] == "drained"
    assert "copies enqueued" in calls
    assert not handler._transfer_events
    assert not handler._transfers


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("failure", ["false", "exception"])
def test_rejected_load_submission_recomputes_through_scheduler(
    request_runner, monkeypatch, async_scheduling, failure
):
    runner = _runner(request_runner, monkeypatch, "contract", async_scheduling)
    prompt = _store_prefix(runner, 3)
    recorder = _Recorder(runner)
    handler = runner.offloading_spec.handler
    transfer = handler.transfer_async

    def reject_load(job_id, spec):
        if isinstance(spec[0], runner_utils.MockLoadStoreSpec):
            # The fixture's fence recorder requires the rejected descriptor
            # even though this handler never queues or completes its I/O.
            handler.transfer_specs[job_id] = spec
            if failure == "exception":
                raise OSError("injected submission error")
            return False
        return transfer(job_id, spec)

    handler.transfer_async = reject_load
    runner.new_request(token_ids=prompt)
    request = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, recorder)
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert recorder.voided == [request.request_id]
    assert recorder.unpublished_release == [True]
    assert runner.connector_scheduler.num_abandoned_loads == 1


def test_lookup_error_is_a_miss_before_admission(request_runner, monkeypatch):
    runner = _runner(request_runner, monkeypatch, "contract", False)

    def fail_lookup(*args):
        raise AssertionError("offload keys incomplete")

    runner.manager.lookup.side_effect = fail_lookup
    runner.new_request(token_ids=[0] * 9)
    request = runner.scheduler.requests[str(runner.req_id)]
    _finish(runner, _Recorder(runner))
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert runner.connector_scheduler.num_abandoned_loads == 0


@pytest.mark.parametrize("failed_rank", [0, 3])
def test_cancelled_load_late_rank_failure_cleans_markers(
    request_runner, monkeypatch, failed_rank
):
    from dataclasses import replace

    from tests.v1.kv_connector.unit.offloading_connector.test_reset_lifecycle import (
        NONE,
        TP4,
    )

    h = TP4(request_runner, "two_group")
    h.cs._recompute_failed_loads = True
    for worker in h.workers:
        worker.connector_worker._report_failures = True
    h.hit(1)
    rid = h.add(h.obs * 2, 0)
    h.step(complete=NONE)
    request = h.sched.requests[rid]
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    handler = h.workers[failed_rank].connector_worker.spec.handler
    complete = handler.complete_jobs

    def fail(job_ids):
        complete(job_ids)
        handler.completed_transfers = [
            replace(result, success=False) for result in handler.completed_transfers
        ]

    handler.complete_jobs = fail
    h.sched.finish_requests((rid,), RequestStatus.FINISHED_ABORTED)
    assert rid in h.sched.requests  # active destinations survive until rank acks
    h.step(complete=NONE)  # terminal flush reports the late failed receive
    h.drain()
    h.assert_clean()
    assert h.recv_count(rid) == 1
    assert not h.cs._skip_external_once
    assert not h.cs.take_failed_loads()
    _check_pool_drained(h.r)
