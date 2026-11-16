# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real transfer handlers and receipts; only CUDA events/copies use CPU doubles."""

from collections import deque
from concurrent.futures import Future
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.v1.core.test_history_block_pool import pool
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import CanonicalKVCacheRef
from vllm.v1.kv_offload.cpu.gpu_worker import (
    SingleDirectionOffloadingHandler,
    Transfer,
)


class CPUEvent:
    def __init__(self):
        self.done = False
        self.failure = False
        self.gate = None

    def synchronize(self):
        if self.gate is not None:
            self.gate.wait()
        if self.failure:
            raise RuntimeError("injected fence failure")
        self.done = True

    def query(self):
        return self.done

    def elapsed_time(self, _):
        return 1.0


def real_worker():
    tensor = torch.zeros((2, 8), dtype=torch.int8)
    handler = SingleDirectionOffloadingHandler(
        [tensor], [tensor.clone()], 1, [[CanonicalKVCacheRef(0, 8)]], True
    )
    worker = OffloadingConnectorWorker(
        SimpleNamespace(
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(qsa_cross_group_hit_check=True)
            )
        )
    )
    worker.worker.handlers = {handler}
    return worker, handler


def enqueue(handler, job_id):
    start, end = CPUEvent(), CPUEvent()
    handler._transfers.append(Transfer(job_id, None, start, end, 8))
    handler._transfer_events[job_id] = end
    return end


@pytest.mark.parametrize("kind", ["load", "deferred_store"])
@pytest.mark.parametrize("retry", [False, True])
def test_real_drain_preserves_receipt_until_parent_io_retires(kind, retry):
    worker, handler = real_worker()
    p = pool()
    block = p.get_new_blocks(1)[0]
    handle = p.bindings[block.block_id]
    p.track_job(1, [block.block_id])
    event = None
    if kind == "load":
        event = enqueue(handler, 1)
        worker._load_jobs[1] = "request"
    else:
        worker._unsubmitted_store_jobs.append((1, (None, None)))

        def submit(job, _):
            nonlocal event
            event = enqueue(handler, job)
            event.failure = retry

        worker._submit_transfer = submit
    if retry:
        if event is not None:
            event.failure = True
        with pytest.raises(RuntimeError, match="injected fence failure"):
            worker.drain()
        assert p.host_pages.get(handle).io == {1}
        assert handler._transfers and worker.build_connector_worker_meta() is None
        assert event is not None
        event.failure = False
    _, receiving, meta = worker.drain()
    assert meta is not None and meta.completed_jobs == {1: 1}
    assert receiving == ({"request"} if kind == "load" else set())
    # Ordinary transfer receipts retain the existing per-worker count contract.
    p.complete_jobs(meta.completed_jobs)
    assert p.host_pages.get(handle).io == {1}
    p.complete_jobs(meta.completed_jobs)
    assert not p._jobs and not p.host_pages.get(handle).io
    assert worker.drain() == (set(), set(), None)
    assert not handler._transfers and not handler._transfer_events
    worker.shutdown()


@pytest.mark.parametrize(
    "blocked", ["batch", "wrapped-batch", "rank", "multiproc-rank"]
)
def test_shutdown_deadline_quarantines_a_wait_that_does_not_return(
    monkeypatch, blocked
):
    from vllm.v1.engine import core
    from vllm.v1.executor.multiproc_executor import (
        FutureWrapper,
        MultiprocExecutor,
        WorkerProc,
    )
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    worker, handler = real_worker()
    gate = Event()
    entered = Event()
    finished = Event()
    errors = []
    events = []
    p = pool()
    block = p.get_new_blocks(1)[0]
    handle = p.bindings[block.block_id]
    p.track_job(1, [block.block_id])
    end = enqueue(handler, 1)
    end.gate = gate
    worker._load_jobs[1] = "request"

    def rank_drain():
        entered.set()
        worker.drain()  # Actual handler blocks in the physical fence.
        return (0, None, None)

    executor: Any = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = SimpleNamespace(drain_host_kv=rank_drain)
    if blocked == "multiproc-rank":
        executor = MultiprocExecutor.__new__(MultiprocExecutor)
        executor.is_failed = False
        executor.futures_queue = deque()
        executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda _: None)

        def dequeue(timeout):
            assert 0 < timeout <= 0.1
            # Simulate a rank whose physical access never retires. The actual
            # RPC/Future wrapper must not make the control thread wait forever.
            return WorkerProc.ResponseStatus.SUCCESS, rank_drain()

        executor.response_mqs = [SimpleNamespace(dequeue=dequeue)]
    executor.shutdown = lambda: events.append("executor-release")
    engine = core.EngineCore.__new__(core.EngineCore)
    engine.vllm_config = SimpleNamespace(shutdown_timeout=5.1)
    engine.scheduler = SimpleNamespace(
        host_history_pool=p,
        current_step=0,
        update_from_output=lambda *_: None,
        shutdown=lambda **_: events.append("unlink"),
    )
    engine.model_executor = executor
    engine.structured_output_manager = SimpleNamespace(clear_backend=lambda: None)
    batch: Future[object] = Future()
    if blocked == "wrapped-batch":

        def batch_result():
            entered.set()
            gate.wait()
            return object()

        batch = FutureWrapper(deque(), get_response=batch_result)
    engine.batch_queue = (
        [(batch, object(), Future())] if blocked in ("batch", "wrapped-batch") else []
    )
    monkeypatch.setattr(core, "cleanup_dist_env_and_memory", lambda: None)

    def shutdown():
        try:
            engine.shutdown()
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    thread = Thread(target=shutdown, daemon=True)
    thread.start()
    try:
        if blocked != "batch":
            assert entered.wait(1)
        assert finished.wait(1), "shutdown hung beyond the shared deadline"
        assert errors and isinstance(errors[0], TimeoutError)
        assert getattr(engine, "_host_kv_quarantined", False)
        assert not events
        assert p.host_pages.get(handle).io == {1}
        with pytest.raises(RuntimeError, match="quarantin"):
            engine.shutdown()
    finally:
        gate.set()
        if blocked != "wrapped-batch" and not batch.done():
            batch.set_result(object())
        thread.join(2)
        worker.shutdown()


@pytest.mark.parametrize("retry", [False, True])
def test_all_real_rank_receipts_reach_core_once_before_release(monkeypatch, retry):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import core
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    p = pool()
    block = p.get_new_blocks(1)[0]
    handle = p.bindings[block.block_id]
    p.track_job(1, [block.block_id])
    connections = []
    workers = []
    events = []
    released = []
    for rank in range(2):
        worker, handler = real_worker()
        workers.append(worker)
        event = enqueue(handler, 1)
        event.failure = retry and rank == 1
        events.append(event)
        worker._load_jobs[1] = "request"
        connection = ActiveKVConnector.__new__(ActiveKVConnector)
        connection._host_step = None
        connection.kv_connector = SimpleNamespace(connector_worker=worker)
        connections.append(connection)

    def rpc(method, **_):
        assert method == "drain_host_kv"
        receipts: list[tuple[int, Any, str | None]] = []
        for rank, connection in enumerate(connections):
            try:
                receipts.append((rank, connection.drain_host_kv(), None))
            except RuntimeError as error:
                receipts.append((rank, None, repr(error)))
        return receipts

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.host_history_pool = p
    scheduler.current_step = 0
    scheduler.connector = SimpleNamespace(update_connector_output=lambda _: None)
    scheduler.shutdown = lambda **_: released.append("unlink")
    engine = core.EngineCore.__new__(core.EngineCore)
    engine.vllm_config = SimpleNamespace(shutdown_timeout=30)
    engine.scheduler = scheduler
    engine.model_executor = SimpleNamespace(
        collective_rpc=rpc, shutdown=lambda: released.append("executor")
    )
    engine.structured_output_manager = SimpleNamespace(clear_backend=lambda: None)
    engine.batch_queue = []
    monkeypatch.setattr(core, "cleanup_dist_env_and_memory", lambda: None)
    if retry:
        with pytest.raises(RuntimeError, match="retaining resources"):
            engine.shutdown()
        assert p._jobs[1][1] == 0 and not released
        events[1].failure = False
    engine.shutdown()
    assert not p._jobs and not p.host_pages.get(handle).io
    assert released == ["executor", "unlink"]
    for worker in workers:
        assert worker.drain() == (set(), set(), None)
        worker.shutdown()
