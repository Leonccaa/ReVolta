# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import sys
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.worker.worker_base import WorkerBase


@pytest.mark.parametrize("failure", [None, "cudaDeviceReset", "allocator unavailable"])
def test_gpu_preflight_reports_allocator_failure_as_data(monkeypatch, failure):
    # Importing GPUWorker also imports device-specific kernels. Execute its
    # unmodified preflight method alone, with only the allocator mocked.
    path = Path(__file__).resolve().parents[3] / "vllm/v1/worker/gpu_worker.py"
    tree = ast.parse(path.read_text())
    worker = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker"
    )
    method = next(
        (
            n
            for n in worker.body
            if isinstance(n, ast.FunctionDef) and n.name == "check_sleep"
        ),
        None,
    )
    assert method is not None, "GPU worker needs a non-mutating sleep preflight"
    allocator = Mock()
    if failure is not None:
        allocator.get_instance.side_effect = RuntimeError(failure)
    module = ModuleType("vllm.device_allocator.cumem")
    monkeypatch.setattr(module, "CuMemAllocator", allocator, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    result = namespace["check_sleep"](object())
    assert result == (None if failure is None else f"RuntimeError: {failure}")


def test_other_worker_preflight_is_noop():
    assert WorkerBase.check_sleep(object()) is None


@pytest.mark.parametrize("level", [1, 2])
def test_failed_sleep_check_preserves_serving_state(level):
    events: list[Any] = []
    executor = Mock()
    executor.check_sleep.side_effect = RuntimeError("missing cudaDeviceReset")
    executor.sleep.side_effect = lambda level: events.append("sleep")
    core = object.__new__(EngineCore)
    core.model_executor = executor
    core.pause_scheduler = Mock(side_effect=lambda **kw: events.append("pause"))
    # This models the next successful inference without mutating worker state.
    executor.execute_model.return_value = [42]
    with pytest.raises(RuntimeError, match="missing cudaDeviceReset"):
        core.sleep(level)
    assert not events
    core.pause_scheduler.assert_not_called()
    assert executor.execute_model(object()) == [42]


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("async_pause", [False, True])
def test_sleep_check_precedes_pause_and_memory_changes(level, async_pause):
    events: list[Any] = []
    pause_future: Future[None] | None = Future() if async_pause else None
    executor = Mock()
    executor.check_sleep.side_effect = lambda: events.append("check")
    executor.sleep.side_effect = lambda level: events.append("sleep")
    core = object.__new__(EngineCore)
    core.model_executor = executor

    def pause(**kw):
        events.append(("pause", kw))
        return pause_future

    core.pause_scheduler = pause
    result = core.sleep(level, mode="wait")
    expected: list[Any] = [("pause", {"mode": "wait", "clear_cache": level >= 1})]
    if level >= 1:
        expected.insert(0, "check")
        if not async_pause:
            expected.append("sleep")
    assert events == expected
    if async_pause:
        assert not executor.sleep.called
        assert pause_future is not None
        pause_future.set_result(None)
        if level >= 1:
            assert events == [*expected, "sleep"]
        assert result.done()


def test_executor_collects_all_preflight_statuses():
    """Worker failures are ordinary RPC results, consumed before raising."""
    statuses = ["AttributeError: cudaDeviceReset", None, "unavailable", None]
    executor = SimpleNamespace(collective_rpc=Mock(return_value=statuses))
    with pytest.raises(RuntimeError, match="cudaDeviceReset"):
        Executor.check_sleep(executor)
    executor.collective_rpc.assert_called_once_with("check_sleep")


def test_executor_accepts_successful_preflight():
    executor = SimpleNamespace(collective_rpc=Mock(return_value=[None] * 4))
    Executor.check_sleep(executor)
    executor.collective_rpc.assert_called_once_with("check_sleep")


def test_rejected_preflight_does_not_contaminate_next_rpc():
    """Exercise real collective_rpc/FutureWrapper with four CPU-only queues."""
    queues: list[deque[Any]] = [deque() for _ in range(4)]
    statuses = ["AttributeError: cudaDeviceReset", None, "unavailable", None]

    def dispatch(message):
        method, _, _, _ = message
        for rank, queue in enumerate(queues):
            output = statuses[rank] if method == "check_sleep" else [42, rank]
            queue.append((WorkerProc.ResponseStatus.SUCCESS, output))

    executor = SimpleNamespace(
        rpc_broadcast_mq=SimpleNamespace(enqueue=dispatch),
        is_failed=False,
        futures_queue=deque(),
        response_mqs=[
            SimpleNamespace(dequeue=lambda timeout=None, q=q: q.popleft())
            for q in queues
        ],
    )
    executor.collective_rpc = MethodType(MultiprocExecutor.collective_rpc, executor)
    with pytest.raises(RuntimeError, match="cudaDeviceReset"):
        Executor.check_sleep(executor)
    assert all(not q for q in queues)
    assert not executor.futures_queue
    assert executor.collective_rpc("execute_model") == [[42, rank] for rank in range(4)]
