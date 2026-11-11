# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Terminal shutdown through real engine/executor and FutureWrapper entries."""

import gc
import threading
import time
import weakref
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from tests.v1.spec_decode.test_dflash2_lookup_stats import REG, message
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    WorkerProc,
)
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.spec_decode import lookup_quarantine
from vllm.v1.spec_decode.lookup_shutdown import LookupStatsShutdown
from vllm.v1.spec_decode.lookup_stats import (
    LookupStatsAccumulator,
    LookupStatsDrainReply,
)

pytest_plugins = ("tests.v1.spec_decode.test_dflash2_lookup_stats",)
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(lookup_quarantine, "_retained", {})
    monkeypatch.setattr(lookup_quarantine, "_closing", {})
    monkeypatch.setattr(lookup_quarantine, "_reason", None)


def engine(executor, timeout=15.0):
    result = object.__new__(EngineCore)
    result.model_executor = executor
    result.vllm_config = NS(shutdown_timeout=timeout, parallel_config=NS(world_size=1))
    result.lookup_stats = LookupStatsAccumulator(REG)
    result.lookup_stats_shutdown = LookupStatsShutdown(result)
    result._shutdown_requested_at = None
    result.scheduler = NS(shutdown=Mock(), update_from_output=Mock())
    result.structured_output_manager = NS(clear_backend=Mock())
    result.batch_queue = deque()
    return result


def multiproc(reply=None):
    executor = object.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = Mock()
    status = WorkerProc.ResponseStatus.SUCCESS
    executor.response_mqs = [
        NS(dequeue=Mock(return_value=(status, reply)), shutdown=Mock())
    ]
    return executor


def _tagged_reply(rank, payload, quiescent, error=None):
    return LookupStatsDrainReply(
        rank, payload, quiescent, error, REG.owner, REG.generation
    )


def final_reply():
    return LookupStatsDrainReply(
        0, message(100, 7, 100), True, owner=REG.owner, generation=REG.generation
    )


def test_real_futurewrapper_fifo_and_last_rpc(monkeypatch):
    executor = multiproc(final_reply())
    core = engine(executor)
    order = []

    def queued_output(label, producer):
        order.append(label)
        return NS(lookup_stats=message(producer, None))

    first = FutureWrapper(
        executor.futures_queue,
        lambda: queued_output("deferred-1", 98),
    )
    second = FutureWrapper(
        executor.futures_queue,
        lambda: queued_output("deferred-2", 99),
    )
    core.batch_queue.extendleft([(first, "first", first), (second, "second", second)])
    # done() deliberately does not consume the response.
    assert not first.done() and not second.done()
    broadcast = executor.rpc_broadcast_mq
    original = executor.collective_rpc

    def rpc(*args, **kwargs):
        assert order == ["deferred-1", "deferred-2"]
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "collective_rpc", rpc)
    monkeypatch.setattr("vllm.v1.engine.core.cleanup_dist_env_and_memory", lambda: None)
    core.shutdown()
    receipt = core.lookup_stats_shutdown.receipt
    assert order == ["deferred-1", "deferred-2"], "F3_FUTURE_FIFO"
    assert (
        receipt.complete
        and receipt.sequence == 7
        and receipt.producer == receipt.covered == 100
    )
    send = broadcast.enqueue.call_args.args[0]
    assert send[0] == "drain_lookup_stats"
    assert send[2] == {
        "stats_deadline_monotonic": receipt.deadline,
        "expected_generation": REG.generation,
        "expected_owner": REG.owner,
    }
    assert executor.response_mqs == []  # actual transport retirement path ran
    assert not executor.futures_queue
    core.shutdown()
    assert core.lookup_stats_shutdown.receipt is receipt


@pytest.mark.parametrize(
    "replies",
    [
        [],
        [_tagged_reply(1, message(), True)],
        [_tagged_reply(0, message(101, 7, 100), True)],
        [_tagged_reply(0, message(), False)],
        [_tagged_reply(0, message(), True, "worker error")],
    ],
)
def test_partial_error_or_unfinished_is_terminal_and_no_more_rpcs(replies):
    executor = multiproc()
    executor.collective_rpc = Mock(return_value=replies)
    core = engine(executor)
    core.request_shutdown()
    receipt = core.lookup_stats_shutdown.finish()
    assert not receipt.complete and receipt.transport_retired and executor.is_failed, (
        "F3_TRANSPORT_RETIRED"
    )
    executor.collective_rpc = MultiprocExecutor.collective_rpc.__get__(executor)
    with pytest.raises(RuntimeError, match="failed"):
        executor.check_health()
    assert core.lookup_stats_shutdown.finish() is receipt, "F3_TERMINAL_RECEIPT"


def test_timeout_retains_sole_reader_and_late_reply_cannot_revise(monkeypatch):
    executor = multiproc()
    entered, release = threading.Event(), threading.Event()

    def dequeue(timeout):
        entered.set()
        release.wait(2)
        return WorkerProc.ResponseStatus.SUCCESS, final_reply()

    executor.response_mqs[0].dequeue = dequeue
    core = engine(executor)
    core.request_shutdown()
    core.lookup_stats_shutdown.deadline = time.monotonic() + 0.02
    receipt = core.lookup_stats_shutdown.finish()
    assert entered.is_set() and not receipt.complete
    assert not receipt.callback_finished and receipt.transport_retired
    with pytest.raises(RuntimeError, match="failed"):
        executor.check_health()
    release.set()
    assert core.lookup_stats_shutdown.callback_finished.wait(1)
    assert core.lookup_stats_shutdown.finish() is receipt, "F3_TERMINAL_RECEIPT"
    assert core.lookup_stats.sequence == 0
    executor.shutdown()


@pytest.mark.parametrize("timeout", [0.0, 4.0, 5.0])
def test_direct_no_host_kv_expired_deadline_never_enqueues(monkeypatch, timeout):
    executor = multiproc()
    core = engine(executor, timeout)
    monkeypatch.setattr("vllm.v1.engine.core.cleanup_dist_env_and_memory", lambda: None)
    core.shutdown()
    receipt = core.lookup_stats_shutdown.receipt
    assert receipt.reason == "deadline_exhausted" and not receipt.complete
    assert executor.rpc_broadcast_mq is None
    assert receipt.deadline <= time.monotonic()
    assert core._shutdown_requested_at == receipt.requested_at


@pytest.mark.parametrize("quiescent", [False, True])
def test_uniproc_cpu_only_proof_and_all_cleanup_guards(monkeypatch, quiescent):
    from vllm.distributed import parallel_state
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    exporter = NS(cpu_receipt=Mock(return_value=(message(), quiescent)))
    runner = object.__new__(GPUModelRunner)
    runner.speculator = NS(_native_lookup=NS(stats=exporter))
    worker = object.__new__(Worker)
    worker.model_runner = runner
    worker.profiler = Mock()
    worker._ple_offload_worker_handle = None
    runner._ple_offload_connector = None
    runner.host_history = None
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = worker
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = wrapper
    core = engine(executor)
    core.request_shutdown()
    receipt = core.lookup_stats_shutdown.finish()
    assert receipt.complete is quiescent
    assert receipt.isolated is not quiescent
    if quiescent:
        assert not lookup_quarantine.context_isolated()
        return
    reference = weakref.ref(core)
    exporter.cpu_receipt.assert_called_once()
    # External owners, finally blocks and duplicate shutdowns cannot sync/free.
    monkeypatch.setattr(
        "torch.accelerator.synchronize",
        Mock(side_effect=AssertionError("F3_FORBIDDEN_SYNCHRONIZE")),
    )
    monkeypatch.setattr(
        "torch.accelerator.empty_cache",
        Mock(side_effect=AssertionError("F3_FORBIDDEN_FREE")),
    )
    try:
        executor.shutdown()
        wrapper.shutdown()
        worker.shutdown()
        runner.shutdown()
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
        parallel_state.cleanup_dist_env_and_memory()
    worker.profiler.shutdown.assert_not_called()
    for callback in (
        executor.check_health,
        executor._init_executor,
        lambda: GPUModelRunner(None, None),
        lambda: Worker(None, 0, 0, "fixture"),
    ):
        with pytest.raises(RuntimeError, match="process restart"):
            callback()
    del core
    gc.collect()
    assert reference() is not None


@pytest.mark.parametrize("variant", ["ray-v1", "ray-v2"])
def test_ray_interface_stub_receives_same_absolute_deadline(variant):
    class RayContract:
        def collective_rpc(self, method, timeout, kwargs):
            assert method == "drain_lookup_stats"
            assert 0 < timeout <= 5
            self.kwargs = kwargs
            return [final_reply()]

    executor = RayContract()
    core = engine(executor)
    core.request_shutdown()
    result = core.lookup_stats_shutdown.finish()
    assert result.complete
    assert executor.kwargs["stats_deadline_monotonic"] == result.deadline


def test_send_blocked_before_future_creation_retired_without_retry():
    executor = multiproc(final_reply())
    entered, release = threading.Event(), threading.Event()

    def enqueue(packet):
        entered.set()
        release.wait(1)

    executor.rpc_broadcast_mq.enqueue = enqueue
    core = engine(executor)
    core.request_shutdown()
    core.lookup_stats_shutdown.deadline = time.monotonic() + 0.02
    receipt = core.lookup_stats_shutdown.finish()
    assert entered.is_set() and not receipt.callback_finished
    assert not receipt.complete and receipt.transport_retired
    with pytest.raises(RuntimeError, match="failed"):
        executor.check_health()
    release.set()
    assert core.lookup_stats_shutdown.callback_finished.wait(1)
    assert core.lookup_stats_shutdown.finish() is receipt, "F3_TERMINAL_RECEIPT"
    executor.shutdown()


def test_normal_and_idle_real_rpc_responses_are_one_to_one():
    import contextlib

    from vllm.v1.outputs import ModelRunnerOutput

    executor = multiproc()
    executor.output_rank = 0
    executor.kv_output_aggregator = None
    core = engine(executor)
    core.log_error_detail = lambda output: contextlib.nullcontext()
    core.log_iteration_details = lambda output: contextlib.nullcontext()
    core._process_aborts_queue = lambda: None
    core.scheduler.has_requests = lambda: True
    core.scheduler.schedule = lambda: NS(total_num_scheduled_tokens=8)
    core.scheduler.get_grammar_bitmask = lambda scheduled: None
    for producer in range(1, 9):
        result = ModelRunnerOutput([], {}, lookup_stats=message(producer, None))
        executor.response_mqs[0].dequeue.return_value = (
            WorkerProc.ResponseStatus.SUCCESS,
            result,
        )
        core.step()
    assert executor.response_mqs[0].dequeue.call_count == 8
    assert executor.rpc_broadcast_mq.enqueue.call_count == 8, "F3_NO_IDLE_STATS_RPC"
    assert all(
        call.args[0][0] == "execute_model"
        for call in executor.rpc_broadcast_mq.enqueue.call_args_list
    )
    assert core.lookup_stats.producer == 8 and core.lookup_stats.pending
    core.scheduler.has_requests = lambda: False
    for _ in range(10):
        assert core.step() == ({}, False)
    assert executor.rpc_broadcast_mq.enqueue.call_count == 8, "F3_NO_IDLE_STATS_RPC"
    executor.shutdown()


def test_host_kv_mandatory_drain_precedes_final_rpc_and_uses_original_t0(monkeypatch):
    executor = multiproc(final_reply())
    core = engine(executor)
    core.scheduler.host_history_pool = NS()
    order = []

    def host_drain(scheduler, pool, deadline):
        order.append("host-kv")
        core._host_kv_drained = True

    core._drain_host_kv = host_drain
    original = executor.collective_rpc

    def rpc(*args, **kwargs):
        assert order == ["host-kv"]
        order.append("statistics")
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "collective_rpc", rpc)
    monkeypatch.setattr("vllm.v1.engine.core.cleanup_dist_env_and_memory", lambda: None)
    core.request_shutdown()
    original_t0 = core._shutdown_requested_at
    core.shutdown()
    assert order == ["host-kv", "statistics"]
    receipt = core.lookup_stats_shutdown.receipt
    assert receipt.requested_at == original_t0
    assert receipt.deadline <= core.shutdown_deadline()
    core.shutdown()
    assert core.lookup_stats_shutdown.receipt is receipt


def test_uniproc_known_quiescence_missing_vector_allows_normal_cleanup(monkeypatch):
    exporter = NS(cpu_receipt=lambda: (message(100, None), True))
    runner = NS(speculator=NS(_native_lookup=NS(stats=exporter)))
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = NS(worker=NS(model_runner=runner), shutdown=Mock())
    core = engine(executor)
    monkeypatch.setattr("vllm.v1.engine.core.cleanup_dist_env_and_memory", lambda: None)
    core.shutdown()
    receipt = core.lookup_stats_shutdown.receipt
    assert not receipt.complete and not receipt.isolated
    assert receipt.reason == "cpu_record_incomplete"
    executor.driver_worker.shutdown.assert_called_once()
    assert not lookup_quarantine.context_isolated()


def test_timeout_transport_shutdown_wakes_and_retires_sole_reader():
    executor = multiproc()
    entered, closed = threading.Event(), threading.Event()

    class Queue:
        def dequeue(self, timeout):
            entered.set()
            assert closed.wait(2)
            raise RuntimeError("retired response channel")

        def shutdown(self):
            closed.set()

    executor.response_mqs = [Queue()]
    core = engine(executor)
    core.request_shutdown()
    core.lookup_stats_shutdown.deadline = time.monotonic() + 0.02
    receipt = core.lookup_stats_shutdown.finish()
    assert entered.is_set() and not receipt.callback_finished
    executor.shutdown()
    assert closed.is_set() and core.lookup_stats_shutdown.callback_finished.wait(1)
    assert core.lookup_stats_shutdown.finish() is receipt, "F3_TERMINAL_RECEIPT"
    assert executor.rpc_broadcast_mq is None and executor.response_mqs == []


def test_pending_uniproc_finally_destructor_and_rebuild_are_guarded(monkeypatch):
    from concurrent.futures import Future

    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    exporter = NS(cpu_receipt=lambda: (message(1, None), False))
    runner = object.__new__(GPUModelRunner)
    runner.speculator = NS(_native_lookup=NS(stats=exporter))
    worker = object.__new__(Worker)
    worker.model_runner = runner
    worker.profiler = Mock()
    worker._ple_offload_worker_handle = None
    runner._ple_offload_connector = None
    runner.host_history = None
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = worker
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = wrapper
    core = engine(executor)
    pending: Future[ModelRunnerOutput] = Future()
    core.batch_queue.append((pending, "scheduled", pending))
    closed = threading.Event()

    def close_engine():
        core.shutdown()
        closed.set()

    thread = threading.Thread(target=close_engine, daemon=True)
    thread.start()
    deadline = time.monotonic() + 1
    while not lookup_quarantine.isolated(worker):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert not closed.is_set()
    try:
        executor.shutdown()
        wrapper.shutdown()
        worker.shutdown()
        runner.shutdown()
        llm = object.__new__(AsyncLLM)
        llm.engine_core = NS(shutdown=lambda **kwargs: core.shutdown())
        llm.renderer = NS(shutdown=Mock())
        del llm
        gc.collect()
        with pytest.raises(RuntimeError, match="shutdown is in progress"):
            GPUModelRunner(None, None)
        config = NS(parallel_config=NS(data_parallel_rank_local=1))
        with pytest.raises(RuntimeError, match="shutdown is in progress"):
            EngineCore(config, Mock(), False)
    finally:
        pending.set_result(ModelRunnerOutput([], {}))
        thread.join(2)
    assert closed.is_set() and core.lookup_stats_shutdown.receipt.isolated


@pytest.mark.parametrize(
    "scenario", ["zero", "no_forward", "inflight_activity", "busy"]
)
def test_multiproc_final_checkpoint_through_real_rpc(exporter, monkeypatch, scenario):
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.worker.gpu.spec_decode.dflash2.stats import (
        attach_empty_output,
        drain_worker,
    )

    if scenario != "zero":
        exporter.submitted()
        exporter.service(force=True)
        if scenario == "no_forward":
            exporter.export_done.ready = True
            exporter.service()
            assert exporter.quiescent
        exporter.begin_activity()
        runner = NS(speculator=NS(_native_lookup=NS(stats=exporter)))
        attach_empty_output(runner, EMPTY_MODEL_RUNNER_OUTPUT)
    exporter.export_done.ready = scenario != "busy"
    before = exporter.snapshot_ready.records
    producer = exporter.producer
    worker = NS(
        rank=0,
        vllm_config=NS(_lookup_stats_registration=REG),
        model_runner=NS(speculator=NS(_native_lookup=NS(stats=exporter))),
    )
    executor = multiproc()

    def dequeue(timeout):
        packet = executor.rpc_broadcast_mq.enqueue.call_args.args[0]
        assert packet[0] == "drain_lookup_stats"
        return WorkerProc.ResponseStatus.SUCCESS, drain_worker(worker, **packet[2])

    executor.response_mqs[0].dequeue = dequeue
    core = engine(executor)
    monkeypatch.setattr("vllm.v1.engine.core.cleanup_dist_env_and_memory", lambda: None)
    core.request_shutdown()
    if scenario == "busy":
        core.lookup_stats_shutdown.deadline = time.monotonic() + 0.1
        receipt = core.lookup_stats_shutdown.finish()
        assert not receipt.complete, "F2_BUSY_CANNOT_PROVE_QUIET"
        assert exporter.snapshot_ready.records == before, "F2_BUSY_SLOT_NOT_REUSED"
        assert not exporter.quiescent
    else:
        core.shutdown()
        receipt = core.lookup_stats_shutdown.receipt
        assert receipt.complete, "F2_FINAL_CHECKPOINT_COMPLETE"
        assert receipt.producer == receipt.covered == producer
        assert exporter.snapshot_ready.records == before + 1
        assert exporter.completed_activity == exporter.activity
        assert not exporter.busy and exporter.quiescent
        assert receipt.sequence == exporter.sequence
        assert exporter.snapshot_ready.records == exporter.export_done.records
    assert exporter.frozen
    assert exporter.producer == producer
