# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
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
from vllm.v1.engine.idle_gpu_clocks import IdleGpuClocks


@pytest.fixture
def engine(monkeypatch, request):
    tp = getattr(request, "param", 4)
    state = SimpleNamespace(now=0.0)
    monkeypatch.setattr(idle_gpu_clocks.time, "monotonic", lambda: state.now)
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

    class TimedQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            if block and self.empty():
                if timeout is not None:
                    state.now += timeout
                raise queue.Empty
            return super().get(block, timeout)

    core = object.__new__(EngineCoreProc)
    core.vllm_config = config
    core.model_executor = SimpleNamespace(
        parallel_config=parallel,
        vllm_config=config,
        is_sleeping=False,
        collective_rpc=Mock(side_effect=lambda method: [method(w) for w in workers]),
    )
    core.scheduler = Mock()
    core.scheduler.has_requests.return_value = False
    core.scheduler.get_idle_stats.return_value = (None, None)
    core.scheduler.get_kv_connector.return_value = None
    core.engines_running = False
    core.batch_queue = deque()
    core.shutdown_state = EngineShutdownState.RUNNING
    core.process_input_queue_block = True
    core.input_queue = TimedQueue()
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
    core._idle_gpu_clocks = IdleGpuClocks.start(2, core.model_executor)
    assert core._idle_gpu_clocks is not None
    engine.events.clear()
    core._process_input_queue()
    assert engine.state.now == 2
    assert not engine.locked
    core._process_input_queue()


def test_default_off_skips_nvml(monkeypatch, engine):
    core = engine.core
    monkeypatch.delenv("VLLM_SM70_IDLE_CLOCK_SECONDS", raising=False)
    monkeypatch.setattr(
        engine_core.envs,
        "VLLM_SM70_IDLE_CLOCK_SECONDS",
        engine_core.envs.environment_variables["VLLM_SM70_IDLE_CLOCK_SECONDS"](),
    )
    run_engine = EngineCoreProc.run_engine_core
    monkeypatch.setattr(engine_core, "EngineCoreProc", lambda **_: core)
    monkeypatch.setattr(engine_core.signal, "signal", Mock())
    monkeypatch.setattr(engine_core, "SignalCallback", Mock())
    monkeypatch.setattr(engine_core, "start_watch_dog", Mock())
    monkeypatch.setattr(engine_core, "maybe_register_config_serialize_by_value", Mock())
    monkeypatch.setattr(engine_core, "maybe_init_worker_tracer", Mock())
    monkeypatch.setattr(engine_core, "decorate_logs", Mock())
    monkeypatch.setattr(engine_core, "set_process_title", Mock())
    core.run_busy_loop = core._process_input_queue
    core._shutdown_for_process_exit = Mock()

    run_engine(vllm_config=core.vllm_config)

    core.model_executor.collective_rpc.assert_not_called()
    engine.nvml.nvmlInit.assert_not_called()
    engine.nvml.nvmlDeviceSetGpuLockedClocks.assert_not_called()
    engine.nvml.nvmlDeviceResetGpuLockedClocks.assert_not_called()


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


def test_normal_exit_restores_before_shutdown(engine):
    enter_idle(engine)
    engine.core.shutdown = Mock(
        side_effect=lambda: engine.events.append(("shutdown", None))
    )
    engine.core._shutdown_for_process_exit(None)
    assert engine.events[-6:] == [
        *(("reset", uuid) for uuid in engine.uuids),
        ("nvml_shutdown", None),
        ("shutdown", None),
    ]
    assert not engine.locked
