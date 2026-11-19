# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.third_party import pynvml

if TYPE_CHECKING:
    from vllm.v1.executor import Executor
    from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)


def _worker_v100_uuid(worker: "WorkerBase") -> str | None:
    import torch

    device = worker.device
    if device is None or device.type != "cuda":
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) == (7, 0) and "V100" in props.name:
            uuid = str(props.uuid)
            return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    except Exception:
        logger.exception("Unable to identify the worker's V100")
    return None


def _synchronize_worker_gpu(worker: "WorkerBase") -> bool:
    import torch

    try:
        torch.accelerator.synchronize(worker.device)
    except Exception:
        # Return a receipt so the executor consumes the replies from every rank.
        logger.exception("Unable to synchronize the worker GPU for idle clocks")
        return False
    return True


class IdleGpuClocks:
    """Own the instance's SM clock locks on the EngineCore thread."""

    def __init__(
        self, seconds: int, executor: "Executor", handles: dict[str, Any]
    ) -> None:
        self.seconds = seconds
        self.executor = executor
        self.handles = handles
        self.locked: dict[str, Any] = {}
        self.enabled = True
        self.deadline: float | None = None

    @classmethod
    def start(cls, seconds: int, executor: "Executor") -> "IdleGpuClocks | None":
        parallel = executor.parallel_config
        tp = parallel.tensor_parallel_size
        if (
            parallel.nnodes != 1
            or tp not in (1, 4)
            or parallel.world_size != tp
            or parallel.distributed_executor_backend not in ("uni", "mp")
        ):
            logger.warning("V100 idle clocks require single-host TP1 or TP4 (uni/mp)")
            return None

        # The built-in CPU offload handlers submit GPU work on the worker RPC
        # thread. External connectors may submit from independent producers.
        kv = executor.vllm_config.kv_transfer_config
        if executor.vllm_config.ec_transfer_config is not None or (
            kv is not None
            and kv.is_kv_transfer_instance
            and (
                kv.kv_connector != "OffloadingConnector"
                or kv.kv_connector_module_path is not None
                or kv.get_from_extra_config("spec_name", "CPUOffloadingSpec")
                != "CPUOffloadingSpec"
            )
        ):
            logger.warning("V100 idle clocks disabled for this cache connector")
            return None

        try:
            uuids = executor.collective_rpc(_worker_v100_uuid)
            if len(uuids) != tp or len(set(uuids)) != tp or None in uuids:
                raise RuntimeError("expected a distinct V100 UUID from every rank")
            pynvml.nvmlInit()
        except Exception as exc:
            logger.warning("V100 idle clocks disabled: %s", exc)
            return None

        handles: dict[str, Any] = {}
        clocks = cls(seconds, executor, handles)
        try:
            for uuid in uuids:
                assert uuid is not None
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                # Prove reset is permitted on every GPU before applying any lock.
                pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                handles[uuid] = handle
        except Exception as exc:
            logger.warning("V100 idle clocks disabled for GPU %s: %s", uuid, exc)
            clocks.close()
            return None
        logger.info("V100 idle clocks enabled after %ds on %s", seconds, uuids)
        return clocks

    def wait_timeout(
        self, timeout: float | None, still_idle: Callable[[], bool]
    ) -> float | None:
        if not self.enabled or self.locked:
            return timeout
        now = time.monotonic()
        if self.deadline is None:
            self.deadline = now + self.seconds
        remaining = self.deadline - now
        if remaining > 0:
            return remaining if timeout is None else min(remaining, timeout)

        # The engine has no requests or queued batches. This FIFO RPC also
        # fences work on side streams, including already submitted KV copies.
        receipts = self.executor.collective_rpc(_synchronize_worker_gpu)
        if receipts != [True] * len(self.handles):
            self.enabled = False
            logger.warning("V100 idle clocks disabled: GPU work did not drain")
        elif still_idle():
            self._lock()
        return timeout

    def _lock(self) -> None:
        for uuid, handle in self.handles.items():
            try:
                pynvml.nvmlDeviceSetGpuLockedClocks(handle, 405, 405)
            except Exception as exc:
                self.enabled = False
                logger.warning("V100 idle clocks disabled for GPU %s: %s", uuid, exc)
                self.restore()
                return
            self.locked[uuid] = handle
        logger.debug("Locked V100 SM clocks at 405 MHz on %s", list(self.locked))

    def restore(self) -> bool:
        self.deadline = None
        for uuid, handle in list(self.locked.items()):
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            except Exception:
                self.enabled = False
                logger.exception(
                    "Failed to restore default V100 clocks on GPU %s", uuid
                )
            else:
                del self.locked[uuid]
                logger.debug("Restored default V100 clocks on GPU %s", uuid)
        return not self.locked

    def close(self) -> None:
        if not self.restore():
            logger.error("Host must reset V100 GPU clocks on %s", list(self.locked))
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            logger.exception("Failed to close NVML for V100 idle clocks")
