# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real busy-loop methods with CPU queues and controlled stalls."""

import queue
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skip_global_cleanup


def _wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "watchdog did not make progress"
        time.sleep(0.005)


def _watchdog(tmp_path):
    from vllm.config import WatchdogConfig
    from vllm.utils.watch_dog import ThreadedWatchDog

    wd = ThreadedWatchDog(
        "worker_cpu", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock()
    )
    wd.timeout = 0.08
    wd.check_interval = 0.005
    return wd


class _Done(BaseException):
    pass


class _RPCQueue:
    def __init__(self):
        self.queue: queue.Queue[Any] = queue.Queue()
        self.idle = threading.Event()

    def dequeue(self, indefinite):
        assert indefinite is True
        self.idle.set()
        item = self.queue.get()
        self.idle.clear()
        if item is None:
            raise _Done from None
        return item


def test_worker_idle_blocked_rpc_and_recovery(tmp_path, monkeypatch):
    from vllm.v1.executor import multiproc_executor as module

    wd = _watchdog(tmp_path)
    monkeypatch.setattr(module, "get_watch_dog", lambda: wd)
    rpc_queue = _RPCQueue()
    release = threading.Event()
    outputs: list[str] = []

    def blocked_rpc():
        assert release.wait(5)
        return "recovered"

    worker = SimpleNamespace(
        rpc_broadcast_mq=rpc_queue,
        rank=0,
        worker=SimpleNamespace(blocked_rpc=blocked_rpc),
        handle_output=outputs.append,
    )

    def run():
        with suppress(_Done):
            module.WorkerProc.worker_busy_loop(worker)

    thread = threading.Thread(target=run, daemon=True)
    wd.start()
    thread.start()
    try:
        assert rpc_queue.idle.wait(2)
        time.sleep(0.2)
        assert wd.dump_seq == 1  # idle dequeue is unmonitored
        for expected in (2, 3):
            rpc_queue.queue.put(("blocked_rpc", (), {}, None))
            _wait_until(lambda expected=expected: wd.dump_seq == expected)
            time.sleep(0.2)
            assert wd.dump_seq == expected  # only one dump for this stall
            release.set()
            _wait_until(lambda expected=expected: len(outputs) == expected - 1)
            assert rpc_queue.idle.wait(2)
            release.clear()
            time.sleep(0.2)
            assert wd.dump_seq == expected
        content = Path(wd.dump_file).read_text()
        assert content.count("due to feed timeout") == 2
        assert "blocked_rpc" in content
    finally:
        release.set()
        rpc_queue.queue.put(None)
        thread.join(2)
        wd.stop()
    assert not thread.is_alive()
    assert outputs == ["recovered", "recovered"]


def test_default_worker_rpc_trace():
    from vllm.v1.executor.multiproc_executor import WorkerProc

    trace = []
    calls: Any = iter([("compute", (7,), {}, None), ("compute", (9,), {}, 1)])

    def dequeue(indefinite):
        trace.append(("dequeue", indefinite))
        try:
            return next(calls)
        except StopIteration:
            raise _Done from None

    def compute(value):
        trace.append(("compute", value))
        return value * 3

    worker = SimpleNamespace(
        rpc_broadcast_mq=SimpleNamespace(dequeue=dequeue),
        rank=0,
        worker=SimpleNamespace(compute=compute),
        handle_output=lambda result: trace.append(("output", result)),
    )
    with pytest.raises(_Done):
        WorkerProc.worker_busy_loop(worker)
    assert trace == [
        ("dequeue", True),
        ("compute", 7),
        ("output", 21),
        ("dequeue", True),
        ("compute", 9),
        ("dequeue", True),
    ]


def test_default_engine_step_trace():
    from vllm.v1.engine.core import EngineCoreProc

    trace = []
    outputs: queue.Queue[Any] = queue.Queue()
    engine = SimpleNamespace(
        step_fn=lambda: (dict(client="token"), True),
        output_queue=outputs,
        post_step=lambda executed: trace.append(("post_step", executed)),
    )
    assert EngineCoreProc._process_engine_step(engine) is True
    assert outputs.get_nowait() == ("client", "token")
    assert outputs.empty()
    assert trace == [("post_step", True)]


def test_engine_idle_queue_and_step_timeout(tmp_path, monkeypatch):
    from vllm.v1.engine import core as module

    wd = _watchdog(tmp_path)
    monkeypatch.setattr(module, "get_watch_dog", lambda: wd)
    inputs: queue.Queue[Any] = queue.Queue()
    consumed = threading.Event()
    release = threading.Event()
    engine = SimpleNamespace(
        has_work=consumed.is_set,
        is_running=lambda: True,
        _notify_idle_state_callbacks=lambda: None,
        aborts_queue=queue.Queue(),
        input_queue=inputs,
        process_input_queue_block=True,
        _handle_client_request=lambda *_: consumed.set(),
        output_queue=queue.Queue(),
        post_step=lambda _: None,
    )

    def step():
        assert release.wait(5)
        return {}, True

    engine.step_fn = step

    def run():
        module.EngineCoreProc._process_input_queue(engine)
        module.EngineCoreProc._process_engine_step(engine)

    thread = threading.Thread(target=run, daemon=True)
    wd.start()
    thread.start()
    try:
        _wait_until(lambda: not wd.active)
        time.sleep(0.2)
        assert wd.dump_seq == 1
        inputs.put(("request", None))
        _wait_until(lambda: wd.dump_seq == 2)
        assert "step" in Path(wd.dump_file).read_text()
    finally:
        inputs.put(("request", None))
        release.set()
        thread.join(2)
        wd.stop()
    assert not thread.is_alive()


@pytest.mark.parametrize("kind", ["engine", "worker"])
def test_signal_reentry_and_process_cleanup(tmp_path, monkeypatch, kind):
    import signal
    from unittest.mock import patch

    from vllm.config import VllmConfig
    from vllm.v1.engine import core
    from vllm.v1.executor import multiproc_executor

    module = core if kind == "engine" else multiproc_executor
    handlers: dict[int, Any] = {}
    monkeypatch.setattr(signal, "signal", lambda sig, fn: handlers.update({sig: fn}))
    wd = _watchdog(tmp_path)
    monkeypatch.setattr(module, "get_watch_dog", lambda: wd)
    start = MagicMock(return_value=wd)
    monkeypatch.setattr(module, "start_watch_dog", start)
    stop = MagicMock(wraps=wd.stop)
    monkeypatch.setattr(wd, "stop", stop)
    original_dump = wd.dump_stack

    def reentrant_dump(reason):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        original_dump(reason)

    monkeypatch.setattr(wd, "dump_stack", reentrant_dump)
    fake = MagicMock()
    config = VllmConfig()

    def busy_loop():
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        raise SystemExit

    monkeypatch.setattr(module, "maybe_init_worker_tracer", lambda *a, **kw: None)
    if kind == "engine":
        entry = core.EngineCoreProc.run_engine_core
        callback = MagicMock()
        fake.run_busy_loop.side_effect = busy_loop
        with (
            patch.object(core, "maybe_register_config_serialize_by_value"),
            patch.object(core, "set_process_title"),
            patch.object(core, "decorate_logs"),
            patch.object(core, "SignalCallback", return_value=callback),
            patch.object(core, "EngineCoreProc", return_value=fake),
            pytest.raises(SystemExit),
        ):
            entry(vllm_config=config)
        assert fake.shutdown_state == core.EngineShutdownState.REQUESTED
        callback.stop.assert_called_once()
    else:
        entry = multiproc_executor.WorkerProc.worker_main
        fake.worker_busy_loop.side_effect = busy_loop
        with (
            patch.object(multiproc_executor, "set_worker_net_device"),
            patch.object(multiproc_executor, "WorkerProc", return_value=fake),
            pytest.raises(SystemExit),
        ):
            entry(vllm_config=config, ready_pipe=MagicMock())
    start.assert_called_once()
    stop.assert_called_once()
    fake.shutdown.assert_called_once()
    content = Path(wd.dump_file).read_text()
    assert content.count("due to SIGTERM") == 1
