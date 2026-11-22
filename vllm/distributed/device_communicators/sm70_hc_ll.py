# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-Next LL scratch and SM70 physical NVLink admission (1Cat #992/#1521)."""

from itertools import combinations
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm import _custom_ops as ops
from vllm.distributed.parallel_state import (
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import import_pynvml

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.sm70_hcx import Sm70HcxRuntime

logger = init_logger(__name__)

MAX_ROWS = 64
_DOWN_BYTES = 2 * MAX_ROWS * 336 * 4
_UP_BYTES = 2 * MAX_ROWS * 2560 * 4
_AR_BYTES = 2 * 4 * MAX_ROWS * 2560 * 4


def _full_nvlink_mesh(uuids: list[str]) -> bool:
    pynvml = import_pynvml()
    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByUUID(uuid) for uuid in uuids]
        return all(
            pynvml.nvmlDeviceGetP2PStatus(a, b, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
            == pynvml.NVML_P2P_STATUS_OK
            for a, b in combinations(handles, 2)
        )
    finally:
        pynvml.nvmlShutdown()


class Sm70HcLLCommunicator:
    """One graph-stable, single-stream mailbox per TP group."""

    @classmethod
    def create(
        cls, group: ProcessGroup, device: torch.device, rank: int
    ) -> "Sm70HcLLCommunicator | None":
        uuids = [""] * 4
        dist.all_gather_object(
            uuids, current_platform.get_device_uuid(device.index), group=group
        )
        visible = {
            current_platform.get_device_uuid(i): i
            for i in range(torch.accelerator.device_count())
        }
        # All six pairs must be direct NVLink with native peer atomics.
        if not _full_nvlink_mesh(uuids) or not all(
            ops.sm70_ring_native_peer_atomics(device.index, visible[uuid])
            for index, uuid in enumerate(uuids)
            if index != rank
        ):
            logger.info("SM70 HC LL disabled: physical NVLink mesh not confirmed.")
            return None
        logger.info("SM70 HC LL enabled: six direct NVLink pairs and peer atomics.")
        return cls(group, device, rank)

    def __init__(self, group: ProcessGroup, device: torch.device, rank: int) -> None:
        from .custom_all_reduce import CustomAllreduce

        self.rank = rank
        self.device = device
        self.hcx: Sm70HcxRuntime | None = None
        with torch.accelerator.device_index(device.index):
            self.pointers = CustomAllreduce.create_shared_buffer(
                _DOWN_BYTES + _UP_BYTES + _AR_BYTES, group=group
            )
            self.down_ptrs = self.pointers
            self.up_ptrs = [ptr + _DOWN_BYTES for ptr in self.pointers]
            self.ar_ptrs = [ptr + _DOWN_BYTES + _UP_BYTES for ptr in self.pointers]
            self.partial = torch.empty(
                40 * MAX_ROWS * 96, dtype=torch.float32, device=device
            )
            self.down_counter = torch.zeros(16, dtype=torch.int32, device=device)
            self.down_counter[4:6] = MAX_ROWS
            self.up_counter = torch.zeros(3, dtype=torch.int32, device=device)
            self.up_counter[1:3] = MAX_ROWS
            self.down_seq = torch.zeros(1, dtype=torch.int32, device=device)
            self.up_seq = torch.zeros(1, dtype=torch.int32, device=device)
            self.ar_counter = torch.zeros(1, dtype=torch.int32, device=device)
            self.ar_dirty = torch.zeros(2, dtype=torch.int32, device=device)
            self.ar_seq = torch.zeros(1, dtype=torch.int32, device=device)
            self.empty = torch.empty(0, dtype=torch.float16, device=device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def project(
        self, x: torch.Tensor, down: torch.Tensor, up: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        out = x.new_empty((rows, 2560))
        injection = x.new_empty((rows, 4))
        # 1Cat v1.5.4 (1fa9c438bc) uses the fused down+up kernel by default.
        ops.sm70_hc_ll_fused_down_up_out(
            x,
            down,
            up,
            self.partial,
            self.down_counter,
            self.up_counter,
            self.down_ptrs,
            self.up_ptrs,
            self.down_seq,
            self.up_seq,
            self.rank,
            out,
            self.empty,
            injection,
        )
        return out, injection

    def combine_norm(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        injection: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(residual, memory_format=torch.contiguous_format)
        norm = torch.empty_like(out)
        ops.sm70_hc_ll_allreduce_combine_norm_out(
            out,
            norm,
            _aligned(partial),
            self.empty,
            _aligned(residual),
            injection.contiguous(),
            _aligned(weight),
            eps,
            self.ar_counter,
            self.ar_dirty,
            self.ar_seq,
            self.ar_ptrs,
            self.rank,
        )
        return out, norm

    def close(self) -> None:
        from .custom_all_reduce import CustomAllreduce

        if self.hcx is not None:
            self.hcx.close()
            self.hcx = None
        with torch.accelerator.device_index(self.device.index):
            for rank, pointer in enumerate(self.pointers):
                if rank != self.rank:
                    ops.sm70_ring_close_mem_handle(pointer)
            CustomAllreduce.free_shared_buffer(self.pointers, rank=self.rank)


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def get_tp_hc_ll() -> Sm70HcLLCommunicator | None:
    if model_parallel_is_initialized():
        return getattr(get_tp_group().device_communicator, "sm70_hc_ll", None)
    return None
