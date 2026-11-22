# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-Next QSA DCP2 peer exchange adapted from 1Cat #1570."""

from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm import _custom_ops as ops
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)
MAX_ROWS = 64
LOCAL_HEADS = 6
HEAD_DIM = 256

# Two epochs of 8-byte (payload, tag) entries. Q packs two FP16 values
# per entry. Combine sends one FP32 value per entry, plus one entry for LSE.
Q_BYTES = 2 * MAX_ROWS * LOCAL_HEADS * (HEAD_DIM // 2) * 8
O_BYTES = 2 * MAX_ROWS * LOCAL_HEADS * (HEAD_DIM + 1) * 8


class Sm70Dcp2Runtime:
    """One mailbox per DCP pair, shared by serial QSA calls and graph replay."""

    @classmethod
    def create(
        cls,
        tp: "GroupCoordinator",
        group: ProcessGroup,
        ranks: list[int],
        device: torch.device,
        rank: int,
    ) -> "Sm70Dcp2Runtime | None":
        # DCP pairs are TP subgroups; reuse the TP-wide physical mesh decision.
        if getattr(tp.device_communicator, "sm70_hc_ll", None) is None:
            return None
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        with torch.accelerator.device_index(device.index):
            pointers = CustomAllreduce.create_shared_buffer(
                Q_BYTES + O_BYTES, group=group
            )
            runtime = cls(device, rank, pointers)
            torch.accelerator.synchronize()
        dist.barrier(group=group)
        logger.info(
            "SM70 NVLink QSA DCP2 Q exchange and LSE combine enabled (pair=%s).",
            ranks,
        )
        return runtime

    def __init__(self, device: torch.device, rank: int, pointers: list[int]) -> None:
        self.rank = rank
        self.device = device
        self.pointers = pointers
        self.q_seq = torch.zeros(2, dtype=torch.int32, device=device)
        self.o_seq = torch.zeros_like(self.q_seq)
        self.gathered = torch.empty(
            (MAX_ROWS, 2 * LOCAL_HEADS, HEAD_DIM),
            dtype=torch.float16,
            device=device,
        )
        # The imported Q kernel also remaps metadata for 1Cat Host KV.
        # ReVolta already localizes its selections. These operands are
        # independent of the query exchange and their outputs are unused.
        self.positions = torch.zeros(MAX_ROWS, dtype=torch.int64, device=device)
        self.lengths = torch.zeros(1, dtype=torch.int32, device=device)
        self.local_positions = torch.empty_like(self.positions)
        self.local_lengths = torch.empty_like(self.lengths)

    def gather_query(self, query: torch.Tensor) -> torch.Tensor:
        query = query.contiguous()
        if query.storage_offset() % 2:
            query = query.clone()  # The native kernel loads pairs of half values.
        rows = query.shape[0]
        gathered = self.gathered[:rows]
        ops.sm70_qsa_device_dcp2_q_prep(
            query,
            gathered,
            self.positions,
            self.lengths,
            self.local_positions,
            self.local_lengths,
            self.pointers[self.rank],
            self.pointers[1 - self.rank],
            self.q_seq,
            self.rank,
            1,
        )
        return gathered

    def combine_gate(
        self,
        partial: torch.Tensor,
        lse: torch.Tensor,
        gate: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        ops.sm70_qsa_device_dcp2_combine_gate(
            partial,
            lse,
            gate,
            output,
            self.pointers[self.rank] + Q_BYTES,
            self.pointers[1 - self.rank] + Q_BYTES,
            self.o_seq,
            self.rank,
        )

    def close(self) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        with torch.accelerator.device_index(self.device.index):
            ops.sm70_ring_close_mem_handle(self.pointers[1 - self.rank])
            CustomAllreduce.free_shared_buffer(self.pointers, rank=self.rank)


def get_sm70_dcp2_for_batch(
    group: "GroupCoordinator",
    query: torch.Tensor,
    partial: torch.Tensor,
    lse: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
) -> Sm70Dcp2Runtime | None:
    runtime = cast(
        Sm70Dcp2Runtime | None, getattr(group.device_communicator, "sm70_dcp2", None)
    )
    if runtime is None:
        return None
    rows = query.shape[0]
    local_shape = (rows, LOCAL_HEADS, HEAD_DIM)
    if (
        0 < rows <= MAX_ROWS
        and query.shape == gate.shape == output.shape == local_shape
        and query.dtype == gate.dtype == output.dtype == torch.float16
        and gate.stride(-1) == output.stride(-1) == 1
        and partial.shape == (rows, 2 * LOCAL_HEADS, HEAD_DIM)
        and lse.shape == (rows, 2 * LOCAL_HEADS)
        and partial.dtype == lse.dtype == torch.float32
        and partial.is_contiguous()
        and lse.is_contiguous()
    ):
        return runtime
    return None
