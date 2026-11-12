# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU receipts and kernel-exit doubles; no CUDA or NCCL runtime claims."""

import ctypes
import multiprocessing
import os
import platform
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed import parallel_state as ps
from vllm.distributed import shutdown as teardown
from vllm.distributed.device_communicators import pynccl
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.utils import process_reaping as reaping
from vllm.v1 import utils as engine_utils
from vllm.v1.executor import multiproc_executor as mp


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Process:
    """Leader has exited; is_alive remains true until its last thread exits."""

    pid = 123

    def __init__(self, clock, exits_at=float("inf")):
        self.clock = clock
        self.exits_at = exits_at
        self.signals = []
        self.joins = []

    def is_alive(self):
        return self.clock.now < self.exits_at

    def join(self, timeout=None):
        self.joins.append(timeout)

    def terminate(self):
        self.signals.append(("term", self.clock.now))

    def kill(self):
        self.signals.append(("kill", self.clock.now))


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(reaping, "time", clock)
    monkeypatch.setattr(mp, "time", clock)
    return clock


def receipt(clock, *, isolated=False):
    reader, writer = multiprocessing.Pipe(duplex=False)
    reaping.notify_process_retired(writer, isolated=isolated)
    writer.close()
    result = reaping.ProcessRetirement(reader)
    assert result.observe()
    assert result.isolated is isolated
    return result


@pytest.mark.parametrize("isolated", [False, True])
def test_retired_leader_waits_for_thread_group_without_signals(clock, isolated):
    process = Process(clock, exits_at=114)
    retired = receipt(clock, isolated=isolated)
    reaping.reap_processes([process], {process: retired}, reap_timeout=30)
    assert process.signals == []
    assert process.joins == []
    assert not process.is_alive()
    assert 114 <= clock.now < 114.1


def test_unreported_process_keeps_original_four_plus_four(clock):
    process = Process(clock)
    reaping.reap_processes([process])
    assert [s for s, _ in process.signals] == ["term", "kill"]
    assert 104 <= process.signals[0][1] < 104.1
    assert 108 <= process.signals[1][1] < 108.2


def test_retired_budget_expires_then_escalates(clock):
    process = Process(clock)
    reaping.reap_processes([process], {process: receipt(clock)}, reap_timeout=20)
    assert [s for s, _ in process.signals] == ["term", "kill"]
    assert 120 <= process.signals[0][1] < 120.1
    assert 124 <= process.signals[1][1] < 124.2


def test_unreported_peer_does_not_consume_retired_budget(clock):
    retired, stuck = Process(clock, 114), Process(clock)
    reaping.reap_processes([retired, stuck], {retired: receipt(clock)}, reap_timeout=20)
    assert retired.signals == []
    assert [s for s, _ in stuck.signals] == ["term", "kill"]
    assert clock.now < 114.1


def test_deferred_reap_preserves_absolute_receipt_deadline(clock):
    process = Process(clock)
    state = {process: receipt(clock)}
    reaping.reap_processes([process], state, reap_timeout=10, defer_retired=True)
    assert clock.now == 100
    # Persistence and resource teardown consume time, never reset the receipt.
    clock.now = 108
    reaping.reap_processes([process], state, reap_timeout=10, graceful_timeout=0)
    assert 110 <= process.signals[0][1] < 110.1


@pytest.mark.parametrize("payload", [b"", b"bad", reaping._RECEIPT.pack(9, 100)])
def test_eof_and_invalid_receipts_are_not_retirement(clock, payload):
    reader, writer = multiprocessing.Pipe(duplex=False)
    if payload:
        os.write(writer.fileno(), payload)
    writer.close()
    state = reaping.ProcessRetirement(reader)
    assert not state.observe()
    process = Process(clock)
    reaping.reap_processes([process], {process: state}, reap_timeout=30)
    assert process.signals[0][1] < 104.1
    assert state.closed


def test_full_retirement_pipe_never_blocks_or_changes_flags_on_stderr():
    reader, writer = multiprocessing.Pipe(duplex=False)
    try:
        os.set_blocking(writer.fileno(), False)
        with pytest.raises(BlockingIOError):
            while True:
                os.write(writer.fileno(), b"x" * 4096)
        started = time.monotonic()
        reaping.notify_process_retired(writer, isolated=True)
        assert time.monotonic() - started < 0.1
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize(
    "requested,limit,expected",
    [(None, 30, 30), (60, 30, 30), (10, 30, 10), (0, 30, 0), (None, 0, 0)],
)
def test_reap_configuration_is_capped(requested, limit, expected):
    config = SimpleNamespace(shutdown_timeout=limit, shutdown_reap_timeout=requested)
    assert reaping.process_reap_timeout(config) == expected


@pytest.mark.parametrize(
    "retired,exits_at,forced",
    [(True, 114, False), (False, 114, True), (True, 150, True)],
)
def test_engine_parent_honors_receipt_and_bounded_tree_kill(
    clock, monkeypatch, retired, exits_at, forced
):
    process = Process(clock, exits_at)
    states = {process: receipt(clock)} if retired else {}
    kills = []
    monkeypatch.setattr(
        engine_utils, "kill_process_tree", lambda pid: kills.append((pid, clock.now))
    )
    engine_utils.shutdown([process], timeout=5, retirements=states, reap_timeout=20)
    assert bool(kills) is forced
    assert process.signals == ([] if retired else [("term", 100.0)])
    if kills:
        assert (120 if retired else 105) <= kills[0][1] < (120.1 if retired else 105.1)


@pytest.mark.parametrize("hang", [False, True])
def test_six_half_second_aborts_share_one_deadline(monkeypatch, hang):
    order: list[str] = []
    comms = []
    release = threading.Event()
    monkeypatch.setattr(teardown, "SHUTDOWN_TIMEOUT_S", 1.2)
    monkeypatch.setattr(
        pynccl.torch.accelerator, "device_index", lambda *a: nullcontext()
    )
    monkeypatch.setattr(ps, "context_isolated", lambda: False)
    monkeypatch.setattr(ps.torch.distributed, "destroy_process_group", order.append)
    names = ("_TP", "_DCP", "_PCP", "_PP", "_DP", "_EP", "_EPLB")
    for index, name in enumerate(names):
        monkeypatch.setattr(ps, name, None)
        if index == 6:
            continue
        comm = pynccl.PyNcclCommunicator.__new__(pynccl.PyNcclCommunicator)
        comm.available, comm.disabled = True, False
        comm.device = SimpleNamespace(index=0)
        comm.comm = object()
        comm.nccl = SimpleNamespace(
            ncclCommAbort=lambda *a, index=index: release.wait(
                None if hang and index == 0 else 0.5
            )
        )
        comms.append(comm)
        custom = CudaCommunicator.__new__(CudaCommunicator)
        custom.pynccl_comm = comm
        custom.ca_comm = custom.fi_ar_comm = custom.all2all_manager = None
        group = ps.GroupCoordinator.__new__(ps.GroupCoordinator)
        group.device_communicator = custom
        group.device_group, group.cpu_group = name + ".nccl", name + ".gloo"
        group.mq_broadcaster = None
        monkeypatch.setattr(ps, name, group)
    started = time.monotonic()
    try:
        if hang:
            with pytest.raises(
                teardown.ShutdownIncompleteError, match="abort incomplete"
            ):
                ps.destroy_model_parallel()
            assert 1.15 <= time.monotonic() - started < 1.6
            assert order == []
            assert comms[0].available and comms[0].disabled
        else:
            ps.destroy_model_parallel()
            assert 0.45 <= time.monotonic() - started < 1.1
            assert len(order) == 12
            assert all(not comm.available for comm in comms)
    finally:
        release.set()
        for comm in comms:
            comm._abort_thread.join(1)
            assert not comm._abort_thread.is_alive()


def _leader_exits_before_thread(writer, isolated):
    def finish():
        time.sleep(0.7)
        os._exit(1 if isolated else 0)

    threading.Thread(target=finish, daemon=True).start()
    reaping.notify_process_retired(writer, isolated=isolated)
    # Linux x86_64 exit(2) terminates this thread, not the thread group.
    ctypes.CDLL(None).syscall(60, 0)


@pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64",
    reason="Linux x86_64 thread-exit syscall",
)
@pytest.mark.parametrize("isolated", [False, True])
def test_real_unreapable_leader_waits_without_parent_signals(isolated, tmp_path):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    proc = context.Process(target=_leader_exits_before_thread, args=(writer, isolated))
    proc.start()
    (tmp_path / "pid.txt").write_text(str(proc.pid))
    writer.close()
    state = reaping.ProcessRetirement(reader)
    try:
        assert reader.poll(20)
        assert state.observe()
        assert proc.is_alive()
        assert proc.exitcode is None
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                proc, "terminate", Mock(side_effect=AssertionError("unexpected TERM"))
            )
            patch.setattr(
                proc, "kill", Mock(side_effect=AssertionError("unexpected KILL"))
            )
            reaping.reap_processes([proc], {proc: state}, reap_timeout=3)
        assert proc.exitcode == (1 if isolated else 0)
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)
        state.close()


def _receipt_worker_entry(ready, writer, mode):
    from unittest.mock import patch

    entry = mp.WorkerProc.worker_main

    class Worker:
        READY_STR = "READY"

        def __init__(self, **kwargs):
            if mode == "constructor":
                raise RuntimeError("partial initialization")
            self.rpc_broadcast_mq = None
            self.worker_response_mq = SimpleNamespace(
                export_handle=lambda: None, wait_until_ready=lambda: None
            )
            self.peer_response_handles = []

        def monitor_death_pipe(self, pipe, requested):
            pass

        def worker_busy_loop(self):
            pass

        def shutdown(self):
            if mode == "hung":
                threading.Event().wait(30)
            if mode == "failed":
                raise RuntimeError("release incomplete")

    with (
        patch.object(mp, "WorkerProc", Worker),
        patch.object(mp, "set_worker_net_device", lambda *a: None),
        patch.object(mp, "start_watch_dog", lambda *a: None),
        patch.object(mp, "get_watch_dog", lambda: SimpleNamespace(stop=lambda: None)),
        patch.object(teardown, "PROCESS_SHUTDOWN_TIMEOUT_S", 0.3),
    ):
        entry(
            ready_pipe=ready,
            retirement_pipe=writer,
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(numa_bind=False), watchdog_config=None
            ),
        )


@pytest.mark.parametrize("mode", ["success", "failed", "hung", "constructor"])
def test_real_worker_entry_reports_success_or_isolation(mode, tmp_path):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    ready, ready_writer = context.Pipe(duplex=False)
    proc = context.Process(
        target=_receipt_worker_entry, args=(ready_writer, writer, mode)
    )
    proc.start()
    (tmp_path / "pid.txt").write_text(str(proc.pid))
    writer.close()
    ready_writer.close()
    state = reaping.ProcessRetirement(reader)
    try:
        assert reader.poll(20)
        assert state.observe()
        assert state.isolated is (mode != "success")
        proc.join(3)
        assert proc.exitcode == (0 if mode == "success" else 1)
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(3)
        state.close()
        ready.close()


@pytest.mark.parametrize("complete", [True, False])
def test_engine_terminal_flush_precedes_deferred_reaping(clock, monkeypatch, complete):
    from vllm.v1.engine.core import EngineCoreProc

    process = Process(clock, exits_at=114)
    worker_retirement = receipt(clock)
    executor = mp.MultiprocExecutor.__new__(mp.MultiprocExecutor)
    executor.vllm_config = SimpleNamespace(shutdown_timeout=30)
    executor.workers = [
        SimpleNamespace(
            proc=process,
            retirement=worker_retirement,
            death_writer=None,
            worker_response_mq=None,
        )
    ]
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.model_executor = executor
    order = []

    def shutdown():
        executor.shutdown()
        assert process.is_alive()
        assert executor._defer_worker_reap
        order.append("flush")
        clock.now += 2
        core._resources_shutdown_complete = complete

    core.shutdown = shutdown
    reader, writer = multiprocessing.Pipe(duplex=False)
    state = reaping.ProcessRetirement(reader)
    real_reap = executor.reap_workers

    def reap():
        assert order == ["flush"]
        assert state.observe() is complete
        order.append("reap")
        real_reap()

    monkeypatch.setattr(executor, "reap_workers", reap)
    try:
        core._shutdown_for_process_exit(writer)
        assert order == ["flush", "reap"]
        assert not process.signals
        assert clock.now < 114.1
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize("value", [None, 0, 10, 60])
def test_cli_exposes_reap_budget_without_changing_shutdown_default(value):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    argv = [] if value is None else ["--shutdown-reap-timeout", str(value)]
    args = EngineArgs.from_cli_args(parser.parse_args(argv))
    assert args.shutdown_timeout == 30
    assert args.shutdown_reap_timeout == value
    assert reaping.process_reap_timeout(args) == (
        30 if value is None else min(30, value)
    )


def test_isolation_receipt_grants_time_but_cannot_defer_io_quiescence(clock):
    process = Process(clock, exits_at=114)
    reaping.reap_processes(
        [process],
        {process: receipt(clock, isolated=True)},
        reap_timeout=30,
        defer_retired=True,
    )
    assert not process.is_alive()
    assert clock.now >= 114
    assert process.signals == []


@pytest.mark.parametrize("reported", [False, True])
@pytest.mark.parametrize("budget", [0, 10])
def test_unreaped_after_kill_blocks_executor_success(clock, reported, budget):
    process = Process(clock)
    retirement = receipt(clock, isolated=True) if reported else None
    executor = mp.MultiprocExecutor.__new__(mp.MultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        shutdown_timeout=30, shutdown_reap_timeout=budget
    )
    executor._defer_worker_reap = True
    executor.workers = [
        SimpleNamespace(
            proc=process,
            retirement=retirement,
            death_writer=None,
            worker_response_mq=None,
        )
    ]
    with pytest.raises(teardown.ShutdownIncompleteError, match="exit unconfirmed"):
        executor.shutdown()
    assert process.is_alive()
    assert [name for name, _ in process.signals] == ["term", "kill"]
    assert clock.now < 125
    with pytest.raises(teardown.ShutdownIncompleteError, match="remains unconfirmed"):
        executor.shutdown()


def test_concurrent_executor_shutdown_cannot_return_before_retirement(monkeypatch):
    executor = mp.MultiprocExecutor.__new__(mp.MultiprocExecutor)
    executor._shutdown_lock = threading.Lock()
    entered, release, second_done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    calls = []

    def shutdown():
        calls.append("begin")
        entered.set()
        assert release.wait(2)
        executor._worker_shutdown_incomplete = True
        raise teardown.ShutdownIncompleteError("retirement unknown")

    monkeypatch.setattr(executor, "_shutdown_workers_and_queues", shutdown)
    errors = []

    def run(done=None):
        try:
            executor.shutdown()
        except teardown.ShutdownIncompleteError:
            errors.append("failed")
        finally:
            if done:
                done.set()

    first = threading.Thread(target=run)
    second = threading.Thread(target=run, args=(second_done,))
    first.start()
    assert entered.wait(1)
    second.start()
    try:
        assert not second_done.wait(0.1)
    finally:
        release.set()
        first.join(2)
        second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert errors == ["failed", "failed"]
    assert len(calls) == 2


def test_worker_monitor_notifies_engine_even_when_reaping_fails(monkeypatch):
    executor = mp.MultiprocExecutor.__new__(mp.MultiprocExecutor)
    executor.workers = [
        SimpleNamespace(proc=SimpleNamespace(sentinel=12, name="worker"))
    ]
    events = []
    executor.failure_callback = lambda: events.append("failure-notified")
    monkeypatch.setattr(mp.multiprocessing.connection, "wait", lambda *a: [12])
    monkeypatch.setattr(mp.logger, "error", lambda *a: events.append("diagnostic"))

    def fail():
        raise teardown.ShutdownIncompleteError("exit unconfirmed")

    monkeypatch.setattr(executor, "shutdown", fail)
    with pytest.raises(teardown.ShutdownIncompleteError):
        executor.start_worker_monitor(inline=True)
    assert events == ["failure-notified", "diagnostic"]
    assert executor.failure_callback is None


def test_zero_extra_budget_keeps_original_grace_across_deferred_reap(clock):
    process = Process(clock, exits_at=106)
    state = receipt(clock)
    executor = mp.MultiprocExecutor.__new__(mp.MultiprocExecutor)
    executor.vllm_config = SimpleNamespace(shutdown_timeout=30, shutdown_reap_timeout=0)
    executor._defer_worker_reap = True
    executor.workers = [
        SimpleNamespace(
            proc=process,
            retirement=state,
            death_writer=None,
            worker_response_mq=None,
        )
    ]
    executor.shutdown()
    assert clock.now == 100
    clock.now += 2  # Scheduler persistence does not reset the original grace.
    executor.reap_workers()
    assert [name for name, _ in process.signals] == ["term"]
    assert 104 <= process.signals[0][1] < 104.1
    assert not process.is_alive()
