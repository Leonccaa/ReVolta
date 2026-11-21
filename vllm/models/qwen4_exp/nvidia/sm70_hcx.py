# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M, separate-projection HCX (1CatAI/1Cat-vLLM#1007).

The existing HC LL communicator has already admitted a direct TP4 NVLink
mesh. HCX uses its own mailboxes and epochs, not the HC LL or PEX buffers.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm import _custom_ops as ops

HD, KD, LORA, INJ = 2560, 10240, 320, 4
MAX_ROWS = 8
# Only group 0 is used by the native full-mesh local schedule for M <= 8.
_AR_BYTES = 4 * 80 * 256 * 8
_LORA_BYTES = 8 * 336 * 8
_HB_BYTES = 8 * HD * 4
_LANE_R = [(L & 3) + (4 if L & 16 else 0) for L in range(32)]
_LANE_Q = [(L >> 2) & 3 for L in range(32)]


def pack_down(w: torch.Tensor, rank: int) -> torch.Tensor:
    """[336, 10240] FP16 -> [80 cta, 6 warp, 4 kstep, 2, 32 lane, 8]."""
    assert w.shape == (336, KD) and w.dtype == torch.float16
    dev = w.device
    lane_r = torch.tensor(_LANE_R, device=dev)
    lane_q = torch.tensor(_LANE_Q, device=dev)
    n = torch.arange(96, device=dev)
    g = torch.where(n < 80, rank * 80 + n, 320 + n - 80)
    ok = (n < 80) | ((rank == 3) & (n < 84))
    cols = torch.where(
        ok[:, None], w[g.clamp(max=335)], torch.zeros((), dtype=w.dtype, device=dev)
    )
    i = torch.arange(80, device=dev).view(80, 1, 1, 1, 1, 1)
    wp = torch.arange(6, device=dev).view(1, 6, 1, 1, 1, 1)
    s = torch.arange(4, device=dev).view(1, 1, 4, 1, 1, 1)
    hl = torch.arange(2, device=dev).view(1, 1, 1, 2, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 1, 1, 32, 1)
    j = torch.arange(8, device=dev).view(1, 1, 1, 1, 1, 8)
    tile, kh = wp >> 1, wp & 1
    nn = tile * 32 + (lane_q[lane] * 8 + lane_r[lane])
    kl = (kh * 4 + s) * 16 + hl * 8 + j
    k = (kl // 32) * HD + 32 * i + kl % 32
    return cols[nn, k].contiguous()


def pack_up(w: torch.Tensor, rank: int) -> torch.Tensor:
    """[10240, 320] FP16 -> [80 cta, 5 warp, 4 kstep, 2, 32 lane, 8]."""
    assert w.shape == (KD, LORA) and w.dtype == torch.float16
    dev = w.device
    lane_r = torch.tensor(_LANE_R, device=dev)
    lane_q = torch.tensor(_LANE_Q, device=dev)
    i = torch.arange(80, device=dev).view(80, 1, 1, 1, 1, 1)
    uw = torch.arange(5, device=dev).view(1, 5, 1, 1, 1, 1)
    s = torch.arange(4, device=dev).view(1, 1, 4, 1, 1, 1)
    hl = torch.arange(2, device=dev).view(1, 1, 1, 2, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 1, 1, 32, 1)
    j = torch.arange(8, device=dev).view(1, 1, 1, 1, 1, 8)
    row = lane_q[lane] * HD + 640 * rank + 8 * i + lane_r[lane]
    k = (uw * 4 + s) * 16 + hl * 8 + j
    return w[row, k].contiguous()


class Sm70HcxRuntime:
    """Graph-stable scratch for the admitted HC LL owner's single stream."""

    def __init__(self, group: ProcessGroup, device: torch.device, rank: int) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.device, self.rank = device, rank
        with torch.accelerator.device_index(device.index):
            self.pointers = CustomAllreduce.create_shared_buffer(
                _AR_BYTES + _LORA_BYTES + _HB_BYTES, group=group
            )
            self.ar = self.pointers
            self.lora = [p + _AR_BYTES for p in self.pointers]
            self.hb = [p + _AR_BYTES + _LORA_BYTES for p in self.pointers]
            self.xn = torch.empty((MAX_ROWS, KD), dtype=torch.float16, device=device)
            self.sq = torch.empty(80 * 8 * 4, dtype=torch.float32, device=device)
            self.dpart = torch.empty(
                80 * 8 * 96 * 4, dtype=torch.float32, device=device
            )
            self.bar = torch.zeros(4, dtype=torch.int32, device=device)
            self.seq = torch.zeros(1, dtype=torch.int32, device=device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def run(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        injection: torch.Tensor,
        norm: torch.Tensor,
        eps: float,
        down: torch.Tensor,
        up: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = partial.shape[0]
        assert 1 <= rows <= MAX_ROWS
        hidden = residual.new_empty((rows, KD))
        block = partial.new_empty((rows, HD))
        inject = injection.new_empty((rows, INJ))
        # Use ReVolta's shipped ABI: no deferred projection or gated GDN args.
        ops.sm70_hcx_out(
            partial.contiguous(),
            None,
            residual.contiguous(),
            injection.contiguous(),
            norm.contiguous(),
            eps,
            down,
            up,
            hidden,
            block,
            inject,
            self.xn,
            self.sq,
            self.dpart,
            self.bar,
            self.seq,
            self.ar,
            self.lora,
            self.hb,
            self.rank,
            None,
            1,
            True,
        )
        return hidden, block, inject

    def close(self) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        with torch.accelerator.device_index(self.device.index):
            for rank, pointer in enumerate(self.pointers):
                if rank != self.rank:
                    ops.sm70_ring_close_mem_handle(pointer)
            CustomAllreduce.free_shared_buffer(self.pointers, rank=self.rank)
