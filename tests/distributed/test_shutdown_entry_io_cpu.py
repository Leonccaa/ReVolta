# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real terminal entrypoints under log backpressure, without CUDA resources."""

import json
import logging
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import zmq

from vllm.distributed import shutdown as teardown
from vllm.distributed.device_communicators import shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import (
    MessageQueue,
    SpinCondition,
)
from vllm.v1.executor import multiproc_executor as mp


def _io_child(ready, mode, full_pipe, events):
    entry = mp.WorkerProc.worker_main
    real_shutdown = mp.WorkerProc.shutdown
    real_monitor = mp.WorkerProc.monitor_death_pipe
    real_guard = teardown.process_shutdown_guard
    death = Event()

    def record(event):
        with open(events, "a") as stream:
            stream.write(event + "\n")

    # Imports are complete before redirecting the actual logging streams.
    read_fd, write_fd = os.pipe()
    if full_pipe:
        os.set_blocking(write_fd, False)
        try:
            while True:
                os.write(write_fd, b"x" * 4096)
        except BlockingIOError:
            pass
        os.set_blocking(write_fd, True)
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
    os.close(write_fd)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    for logger in (mp.logger, shm_broadcast.logger):
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.disabled = False

    # Mark the attempted log before the real handler can block in write/flush.
    emit = handler.emit

    def observed_emit(log_record):
        record("log:" + log_record.getMessage())
        emit(log_record)

    @contextmanager
    def guard():
        with real_guard():
            record("guard.enter")
            yield
        record("guard.leave")

    def failure(reason):
        record("watchdog:" + reason)
        if mode.endswith("watchdog"):
            Event().wait(30)

    class SlowError(RuntimeError):
        def __str__(self):
            record("error.str")
            Event().wait(30)
            return "cancelled"

    class DeathPipe:
        def recv(self):
            death.wait()
            raise EOFError

        def close(self):
            record("death.close")

    class Worker:
        READY_STR = "READY"

        def __init__(self, *args, **kwargs):
            if mode == "constructor":
                ready.send({"status": "CONSTRUCTING"})
                raise RuntimeError("partial construction")
            self.rpc_broadcast_mq = None
            if mode.startswith("death"):
                self.context = zmq.Context()
                condition = SpinCondition(
                    True, self.context, "inproc://terminal-notify", busy_loop_s=0
                )
                self.rpc_broadcast_mq = MessageQueue.__new__(MessageQueue)
                self.rpc_broadcast_mq._spin_condition = condition
                self.rpc_broadcast_mq.shutting_down = False
                self.rpc_broadcast_mq.wait_until_ready = lambda: None
            response_calls = 0

            def response_shutdown():
                nonlocal response_calls
                response_calls += 1
                if mode == "death_late_error" and response_calls == 1:
                    time.sleep(0.2)
                    record("response.failed")
                    raise RuntimeError("late response cancellation failed")
                record("response.close")

            self.worker_response_mq = SimpleNamespace(
                export_handle=lambda: None,
                wait_until_ready=lambda: None,
                shutdown=response_shutdown,
            )
            self.peer_response_handles = []
            self.worker = SimpleNamespace(shutdown=lambda: record("runner.retired"))

        def monitor_death_pipe(self, pipe, requested):
            self.requested = requested
            if mode.startswith("death"):
                real_monitor(self, pipe, requested)
            elif mode in ("cancelled", "format", "inherited"):
                requested.set()

        def worker_busy_loop(self):
            record("serving")
            if mode.startswith("signal"):
                signal.raise_signal(
                    signal.SIGINT if mode == "signal_int" else signal.SIGTERM
                )
            if mode.startswith("death"):
                assert self.rpc_broadcast_mq is not None
                self.rpc_broadcast_mq._spin_condition.wait()
                assert self.rpc_broadcast_mq.shutting_down
                assert self.requested.is_set()
            if mode == "format":
                raise SlowError()
            raise RuntimeError("peer lost" if mode == "runtime" else "cancelled")

        shutdown = real_shutdown

    with (
        patch.object(handler, "emit", observed_emit),
        patch.object(mp, "WorkerProc", Worker),
        patch.object(mp, "set_worker_net_device", lambda *a: None),
        patch.object(mp, "maybe_init_worker_tracer", lambda **kw: None),
        patch.object(mp, "start_watch_dog", lambda *a: death.set()),
        patch.object(
            mp,
            "get_watch_dog",
            lambda: SimpleNamespace(
                stop=lambda: record("watchdog.stop"), handle_failure=failure
            ),
        ),
        patch.object(mp, "destroy_model_parallel", lambda: record("groups.destroy")),
        patch.object(mp, "destroy_distributed_environment", lambda: None),
        patch.object(teardown, "process_shutdown_guard", guard),
    ):
        entry(
            ready_pipe=ready,
            death_pipe=DeathPipe(),
            inherited_fds=[-1] if mode == "inherited" else [],
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(numa_bind=False), watchdog_config=None
            ),
        )
    record("entry.returned")
    # Avoid interpreter flushing redirected output after the terminal guard.
    os.close(read_fd)
    os._exit(0)


@pytest.mark.parametrize(
    "mode,full_pipe,expected",
    [
        ("cancelled", False, 0),
        ("death", False, 0),
        ("runtime", False, 1),
        ("constructor", False, 1),
        ("signal_int", False, 0),
        ("signal_term", False, 0),
        ("cancelled", True, 1),
        ("runtime", True, 1),
        ("constructor", True, 1),
        ("signal_int", True, 1),
        ("signal_term", True, 1),
        ("death", True, 1),
        ("death_watchdog", False, 1),
        ("signal_watchdog", False, 1),
        ("format", False, 1),
        ("inherited", False, 0),
        ("inherited", True, 1),
        ("death_late_error", False, 1),
    ],
)
def test_terminal_io_is_guarded(mode, full_pipe, expected, tmp_path):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    events = tmp_path / "terminal.events"
    child = context.Process(
        target=_io_child, args=(writer, mode, full_pipe, str(events))
    )
    child.start()
    (tmp_path / "child.pid").write_text(str(child.pid))
    writer.close()
    signals = []

    def signal_child(name, action):
        signals.append(name)
        action()

    observed = SimpleNamespace(
        is_alive=child.is_alive,
        terminate=lambda: signal_child("TERM", child.terminate),
        kill=lambda: signal_child("KILL", child.kill),
    )
    try:
        assert reader.poll(20), "CPU child did not finish imports"
        assert reader.recv()["status"] in ("READY", "CONSTRUCTING")
        started = time.monotonic()
        mp.MultiprocExecutor._ensure_worker_termination([observed])
        child.join(1)
        order = events.read_text().splitlines()
        (tmp_path / "terminal.json").write_text(
            json.dumps(
                dict(
                    mode=mode,
                    full_pipe=full_pipe,
                    pid=child.pid,
                    signals=signals,
                    exitcode=child.exitcode,
                    elapsed_s=time.monotonic() - started,
                    events=order,
                ),
                indent=2,
            )
        )
        assert not signals
        assert child.exitcode == expected
        assert time.monotonic() - started < 4
        guard_index = order.index("guard.enter")
        for index, event in enumerate(order):
            if event.startswith(("log:", "watchdog:")) or event == "error.str":
                assert index > guard_index
        if full_pipe or mode.endswith("watchdog") or mode == "format":
            assert "guard.leave" not in order
        if mode.startswith("death"):
            assert "serving" in order
    finally:
        if child.is_alive():
            child.kill()
            child.join(2)
        reader.close()
        child.close()


def test_cancel_wakeup_is_nonblocking_and_survives_full_queue(monkeypatch):
    def unexpected_log(*args, **kwargs):
        pytest.fail("cancellation must not log before the terminal guard")

    monkeypatch.setattr(shm_broadcast.logger, "debug", unexpected_log)
    context = zmq.Context()
    condition = SpinCondition(True, context, "inproc://cancel-test", busy_loop_s=0)
    try:
        # Exercise the real in-process socket, including saturation. A pending
        # cancellation remains readable when later NOBLOCK sends report EAGAIN.
        for _ in range(5000):
            condition.cancel()
        assert condition.read_cancel_socket.poll(1000)
        condition.wait(timeout_ms=1000)
    finally:
        condition.local_notify_socket.close(linger=0)
        condition.write_cancel_socket.close(linger=0)
        condition.read_cancel_socket.close(linger=0)
        context.term()


def test_parent_shutdown_notifications_do_not_log(monkeypatch):
    def unexpected_log(*args, **kwargs):
        pytest.fail("parent shutdown control path must not wait on logging")

    monkeypatch.setattr(mp.logger, "debug", unexpected_log)
    actions = []
    worker = SimpleNamespace(
        death_writer=SimpleNamespace(close=lambda: actions.append("notify")),
        proc=SimpleNamespace(is_alive=lambda: False),
        worker_response_mq=None,
    )
    executor = mp.MultiprocExecutor.__new__(mp.MultiprocExecutor)
    executor.workers = [worker]
    executor.shutdown()
    assert actions == ["notify"]


def test_queue_shutdown_cancels_only_once_across_threads():
    entered, released = Event(), Event()

    def cancel():
        entered.set()
        assert released.wait(1)

    queue = MessageQueue.__new__(MessageQueue)
    queue.shutting_down = False
    queue._spin_condition = SimpleNamespace(cancel=cancel)
    thread = Thread(target=queue.shutdown)
    thread.start()
    try:
        assert entered.wait(1)
        # A main-thread cleanup must not touch the monitor's send socket while
        # the first cancellation is still in progress in that monitor thread.
        queue.shutdown()
    finally:
        released.set()
        thread.join(1)
    assert not thread.is_alive()
