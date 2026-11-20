# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import signal
import threading
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.third_party.pynvml import NVML_ERROR_NO_PERMISSION, NVMLError
from vllm.v1.engine import EngineCoreRequestType, FinishReason, idle_gpu_clocks
from vllm.v1.engine import core as engine_core
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.idle_gpu_clocks import IdleClockEngineCoreProc, IdleGpuClocks
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


@pytest.fixture
def core_type():
    return IdleClockEngineCoreProc


@pytest.fixture
def engine(monkeypatch, request, core_type):
    tp = getattr(request, "param", 4)
    state = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        idle_gpu_clocks, "time", SimpleNamespace(monotonic=lambda: state.now)
    )
    monkeypatch.setattr(queue, "time", lambda: state.now)
    monkeypatch.setattr(engine_core, "get_watch_dog", lambda: Mock(disable=nullcontext))
    events = []
    locked = set()
    workers = [SimpleNamespace(device=torch.device("cuda", rank)) for rank in range(tp)]
    uuids = [f"GPU-card{rank}" for rank in range(tp)]
    parallel = SimpleNamespace(
        nnodes=1,
        tensor_parallel_size=tp,
        world_size=tp,
        distributed_executor_backend="uni" if tp == 1 else "mp",
        data_parallel_size=1,
        numa_bind=False,
        enable_eplb=False,
        eplb_config=SimpleNamespace(use_async=False),
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        kv_transfer_config=None,
        ec_transfer_config=None,
        watchdog_config=None,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            major=7, minor=0, name="Tesla V100", uuid=f"card{device.index}"
        ),
    )
    sync = Mock(side_effect=lambda device: events.append(("sync", device.index)))
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)

    def lock(uuid, low, high):
        assert low == high == 405
        events.append(("lock", uuid))
        locked.add(uuid)

    def reset(uuid):
        events.append(("reset", uuid))
        locked.discard(uuid)

    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(side_effect=lambda: events.append(("nvml_shutdown", None))),
        nvmlDeviceGetHandleByUUID=Mock(side_effect=lambda uuid: uuid),
        nvmlDeviceSetGpuLockedClocks=Mock(side_effect=lock),
        nvmlDeviceResetGpuLockedClocks=Mock(side_effect=reset),
        NVMLError=NVMLError,
    )
    monkeypatch.setattr(idle_gpu_clocks, "pynvml", nvml)

    def wait_for_input(timeout=None):
        if timeout is None:
            raise queue.Empty
        state.now += timeout

    core = object.__new__(core_type)
    core.vllm_config = config
    core.model_executor = SimpleNamespace(
        parallel_config=parallel,
        vllm_config=config,
        is_sleeping=False,
        collective_rpc=Mock(
            side_effect=lambda method, timeout: [method(w) for w in workers]
        ),
    )
    core.scheduler = Mock()
    core.scheduler.has_requests.return_value = False
    core.scheduler.get_idle_stats.return_value = (None, None)
    core.scheduler.get_kv_connector.return_value = None
    core.engines_running = False
    core.batch_queue = deque()
    core.shutdown_state = EngineShutdownState.RUNNING
    core.process_input_queue_block = True
    core.input_queue = core_type._input_queue_cls()
    core.input_queue.not_empty.wait = wait_for_input
    core.aborts_queue = queue.Queue()
    core.output_queue = queue.Queue()
    core._idle_state_callbacks = []
    core.add_request = Mock(
        side_effect=lambda *_: setattr(core, "engines_running", True)
    )

    def step():
        events.append(("step", None))
        return {}, True

    core.step_fn = Mock(side_effect=step)
    core.post_step = Mock()
    return SimpleNamespace(
        core=core,
        state=state,
        nvml=nvml,
        sync=sync,
        events=events,
        locked=locked,
        uuids=uuids,
        lock=lock,
        reset=reset,
    )


def enqueue_request(core):
    request = SimpleNamespace(request_id="request", client_index=0)
    core.input_queue.put((EngineCoreRequestType.ADD, (request, 0)))


def enter_idle(engine):
    core = engine.core
    assert core.start_idle_clocks(2) is not None
    engine.events.clear()
    core._process_input_queue()
    assert engine.state.now == 2
    assert not engine.locked
    core._process_input_queue()


def prepare_engine_runner(monkeypatch, core, idle_seconds, busy_loop):
    monkeypatch.setattr(engine_core.envs, "VLLM_SM70_IDLE_CLOCK_SECONDS", idle_seconds)
    run_engine = EngineCoreProc.run_engine_core
    monkeypatch.setattr(EngineCoreProc, "run_busy_loop", lambda _: busy_loop())
    monkeypatch.setattr(engine_core, "EngineCoreProc", lambda **_: core)

    def create_idle_core(*, idle_clock_seconds, **_):
        core.idle_clock_seconds = idle_clock_seconds
        return core

    monkeypatch.setattr(idle_gpu_clocks, "IdleClockEngineCoreProc", create_idle_core)
    monkeypatch.setattr(engine_core.signal, "signal", Mock())
    monkeypatch.setattr(engine_core, "SignalCallback", Mock())
    monkeypatch.setattr(engine_core, "start_watch_dog", Mock())
    monkeypatch.setattr(engine_core, "maybe_register_config_serialize_by_value", Mock())
    monkeypatch.setattr(engine_core, "maybe_init_worker_tracer", Mock())
    monkeypatch.setattr(engine_core, "decorate_logs", Mock())
    monkeypatch.setattr(engine_core, "set_process_title", Mock())
    return lambda: run_engine(vllm_config=core.vllm_config)


@pytest.mark.parametrize("value", [None, "0"])
@pytest.mark.parametrize("core_type", [EngineCoreProc])
def test_default_off_skips_nvml(monkeypatch, engine, value):
    core = engine.core
    name = "VLLM_SM70_IDLE_CLOCK_SECONDS"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)

    def run_busy_loop():
        core._process_input_queue()
        assert core._process_engine_step()

    enqueue_request(core)
    run_engine = prepare_engine_runner(
        monkeypatch, core, engine_core.envs.environment_variables[name](), run_busy_loop
    )
    idle_constructor = Mock()
    monkeypatch.setattr(idle_gpu_clocks, "IdleClockEngineCoreProc", idle_constructor)
    core._shutdown_for_process_exit = Mock()

    run_engine()

    idle_constructor.assert_not_called()
    core.add_request.assert_called_once()
    core.model_executor.collective_rpc.assert_not_called()
    engine.nvml.nvmlInit.assert_not_called()
    engine.nvml.nvmlDeviceSetGpuLockedClocks.assert_not_called()
    engine.nvml.nvmlDeviceResetGpuLockedClocks.assert_not_called()


def test_explicit_zero_preserves_legacy_compile_factor(monkeypatch):
    name = "VLLM_SM70_IDLE_CLOCK_SECONDS"
    monkeypatch.setenv(name, "0")
    assert engine_core.envs.compile_factors()[name] == "0"
    monkeypatch.delenv(name)
    assert name not in engine_core.envs.compile_factors()
    monkeypatch.setenv(name, "2")
    assert name not in engine_core.envs.compile_factors()


@pytest.mark.parametrize("startup", [True, False], ids=["uuid", "drain"])
def test_missing_rank_receipt_is_fatal(monkeypatch, engine, startup):
    core = engine.core
    executor = object.__new__(MultiprocExecutor)
    executor.parallel_config = core.vllm_config.parallel_config
    executor.vllm_config = core.vllm_config
    executor.is_sleeping = False
    executor.is_failed = False
    executor.rpc_broadcast_mq = Mock()
    executor.futures_queue = deque()

    def missing_reply(timeout):
        assert timeout is not None, "missing rank would wait indefinitely"
        assert 0 < timeout <= idle_gpu_clocks.IDLE_CLOCK_RPC_TIMEOUT
        assert not threading.Event().wait(timeout)
        raise TimeoutError

    success = WorkerProc.ResponseStatus.SUCCESS
    response_mqs = executor.response_mqs = [
        *(Mock(dequeue=Mock(return_value=(success, True))) for _ in range(3)),
        Mock(dequeue=Mock(side_effect=missing_reply)),
    ]
    if not startup:
        startup_rpc = core.model_executor.collective_rpc
        real_rpc = executor.collective_rpc
        executor.collective_rpc = Mock(
            side_effect=lambda method, **kwargs: (
                startup_rpc if method is idle_gpu_clocks._worker_v100_uuid else real_rpc
            )(method, **kwargs)
        )
    core.model_executor = executor

    def run_busy_loop():
        core._process_input_queue()
        # Inject the request only after the RPC is sent, when rank 3
        # withholds its receipt.
        executor.rpc_broadcast_mq.enqueue.side_effect = (
            lambda *_args, **_kwargs: enqueue_request(core)
        )
        core._process_input_queue()

    core._send_engine_dead = Mock()
    core.shutdown = Mock(side_effect=executor.shutdown)
    run_engine = prepare_engine_runner(monkeypatch, core, 2, run_busy_loop)
    monkeypatch.setattr(idle_gpu_clocks, "IDLE_CLOCK_RPC_TIMEOUT", 0.02, raising=False)

    with pytest.raises(RuntimeError, match="Executor failed.*idle clocks"):
        run_engine()

    core._send_engine_dead.assert_called_once()
    core.shutdown.assert_called_once()
    assert executor.is_failed
    assert executor.rpc_broadcast_mq is None
    assert not executor.response_mqs
    for response in response_mqs:
        response.dequeue.assert_called_once()
        response.shutdown.assert_called_once()
    core.add_request.assert_not_called()
    engine.nvml.nvmlDeviceSetGpuLockedClocks.assert_not_called()


def test_async_eplb_rejected_before_engine_init(monkeypatch, engine):
    core = engine.core
    parallel = core.model_executor.parallel_config
    parallel.enable_eplb = parallel.eplb_config.use_async = True
    run_engine = prepare_engine_runner(monkeypatch, core, 2, Mock())
    constructor = Mock()
    monkeypatch.setattr(idle_gpu_clocks, "IdleClockEngineCoreProc", constructor)
    with pytest.raises(ValueError, match="asynchronous EPLB"):
        run_engine()
    constructor.assert_not_called()
    core.model_executor.collective_rpc.assert_not_called()
    engine.nvml.nvmlInit.assert_not_called()


@pytest.mark.parametrize("rank", [0, 3], ids=["first", "last"])
def test_request_arriving_during_lock_waits_for_restore(engine, rank):
    core = engine.core
    add_request = core.add_request.side_effect

    def lock(uuid, low, high):
        engine.lock(uuid, low, high)
        if uuid == engine.uuids[rank]:
            enqueue_request(core)

    def add(*args):
        assert not engine.locked
        add_request(*args)

    core.add_request.side_effect = add
    engine.nvml.nvmlDeviceSetGpuLockedClocks.side_effect = lock
    enter_idle(engine)
    assert core._process_engine_step()
    affected = engine.uuids[: rank + 1]
    assert engine.events[4:] == [
        *(("lock", uuid) for uuid in affected),
        *(("reset", uuid) for uuid in affected),
        ("step", None),
    ]
    assert not engine.locked


@pytest.mark.parametrize("signals", [0, 1, 2])
def test_sigterm_before_reset_preserves_cleanup(monkeypatch, engine, signals):
    core = engine.core
    register_signal = signal.signal
    original_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    first = engine.uuids[0]

    def default_handler(*_):
        # Model default termination without killing the whole pytest process.
        raise SystemExit("signal interrupted clock restoration")

    def register(signum, handler):
        if handler == signal.SIG_DFL:
            handler = default_handler
        register_signal(signum, handler)

    def reset(uuid):
        if signals and uuid == first and uuid in engine.locked:
            signal.raise_signal(signal.SIGTERM)
        engine.reset(uuid)

    def run_busy_loop():
        core._process_input_queue()
        core._process_input_queue()
        assert engine.locked == set(engine.uuids)
        if signals == 2:
            signal.raise_signal(signal.SIGTERM)

    core.shutdown = Mock(side_effect=lambda: engine.events.append(("shutdown", None)))
    run_engine = prepare_engine_runner(monkeypatch, core, 2, run_busy_loop)
    monkeypatch.setattr(signal, "signal", register)
    engine.nvml.nvmlDeviceResetGpuLockedClocks.side_effect = reset

    try:
        run_engine()
    finally:
        for signum, handler in original_handlers.items():
            register_signal(signum, handler)

    assert engine.events[-6:] == [
        *(("reset", uuid) for uuid in engine.uuids),
        ("nvml_shutdown", None),
        ("shutdown", None),
    ]
    assert not engine.locked


@pytest.mark.parametrize("engine", [1, 4], indirect=True)
def test_idle_drains_ranks_and_restores_before_request(engine):
    core = engine.core
    enter_idle(engine)
    assert engine.events == [
        *(("sync", rank) for rank in range(len(engine.uuids))),
        *(("lock", uuid) for uuid in engine.uuids),
    ]
    enqueue_request(core)
    core._process_input_queue()
    assert core._process_engine_step()
    assert engine.events[-(len(engine.uuids) + 1) :] == [
        *(("reset", uuid) for uuid in engine.uuids),
        ("step", None),
    ]
    core.engines_running = False
    core._process_input_queue()
    assert not engine.locked
    core._process_input_queue()
    assert engine.locked == set(engine.uuids)


def test_inflight_batch_prevents_lock(engine):
    enter_idle(engine)
    enqueue_request(engine.core)
    engine.core._process_input_queue()
    engine.core.engines_running = False
    engine.core.batch_queue.append(object())
    engine.state.now += 10
    engine.core._process_input_queue()
    assert not engine.locked
    assert engine.sync.call_count == 4


def test_failed_gpu_drain_prevents_lock(engine):
    engine.sync.side_effect = RuntimeError("stream error")
    enter_idle(engine)
    assert engine.sync.call_count == 4
    engine.nvml.nvmlDeviceSetGpuLockedClocks.assert_not_called()


def test_request_arriving_during_gpu_drain_skips_lock(engine):
    core = engine.core

    def synchronize(device):
        if device.index == 0:
            enqueue_request(core)

    engine.sync.side_effect = synchronize
    enter_idle(engine)
    engine.nvml.nvmlDeviceSetGpuLockedClocks.assert_not_called()
    assert core._process_engine_step()


def test_reset_permission_denial_disables_without_lock(engine, caplog):
    engine.nvml.nvmlDeviceResetGpuLockedClocks.side_effect = NVMLError(
        NVML_ERROR_NO_PERMISSION
    )
    assert IdleGpuClocks.start(2, engine.core.model_executor) is None
    engine.nvml.nvmlDeviceSetGpuLockedClocks.assert_not_called()
    engine.nvml.nvmlShutdown.assert_called_once()
    assert "V100 idle clocks disabled" in caplog.text


def test_partial_lock_rolls_back(engine):
    def fail_second(uuid, low, high):
        if uuid == engine.uuids[1]:
            raise RuntimeError("lock refused")
        engine.lock(uuid, low, high)

    engine.nvml.nvmlDeviceSetGpuLockedClocks.side_effect = fail_second
    enter_idle(engine)
    assert engine.events[-2:] == [("lock", engine.uuids[0]), ("reset", engine.uuids[0])]
    assert not engine.locked
    core = engine.core
    core._process_input_queue()
    assert engine.nvml.nvmlDeviceSetGpuLockedClocks.call_count == 2
    enqueue_request(core)
    core._process_input_queue()
    core.add_request.assert_called_once()


def test_reset_failure_rejects_gpu_dispatch(engine, caplog):
    def fail_first(uuid):
        if uuid == engine.uuids[0]:
            raise RuntimeError("reset refused")
        engine.reset(uuid)

    core = engine.core
    enter_idle(engine)
    engine.nvml.nvmlDeviceResetGpuLockedClocks.side_effect = fail_first
    enqueue_request(core)
    core._process_input_queue()
    assert not core._process_engine_step()
    core.add_request.assert_not_called()
    core.step_fn.assert_not_called()
    _, result = core.output_queue.get_nowait()
    assert result.outputs[0].finish_reason == FinishReason.ERROR
    assert engine.locked == {engine.uuids[0]}
    assert "Failed to restore default V100 clocks on GPU GPU-card0" in caplog.text

    core.execute_dummy_batch = Mock()
    core._handle_client_request(
        EngineCoreRequestType.UTILITY, (0, 1, "execute_dummy_batch", ())
    )
    core.execute_dummy_batch.assert_not_called()
    _, result = core.output_queue.get_nowait()
    assert (
        result.utility_output.failure_message == "Failed to restore default V100 clocks"
    )

    engine.nvml.nvmlDeviceResetGpuLockedClocks.side_effect = engine.reset
    enqueue_request(core)
    core._process_input_queue()
    core.add_request.assert_called_once()
    assert not engine.locked
