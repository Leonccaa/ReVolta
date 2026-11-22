# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor


class ReasoningEOSState:
    def __init__(self, max_num_reqs: int, device: torch.device):
        # Slot layout: EOS + 1 (zero means inactive), EOS, think end, tool start.
        self.slots = StagedWriteTensor((max_num_reqs, 4), torch.int32, device)
        self.requests: set[int] = set()

    def add_request(
        self,
        slot: int,
        config: tuple[tuple[int, ...], int, int] | None,
        confirmed_output: list[int],
    ) -> None:
        self.requests.discard(slot)
        if config is not None:
            eos, end, tool = config
            if end not in confirmed_output and tool not in confirmed_output:
                self.requests.add(slot)
                self.slots.stage_write(slot, 0, [eos[0] + 1, eos[-1], end, tool])
                return
        self.slots.stage_write_elem(slot, 0)

    def active_requests(self, indices: np.ndarray) -> list[int]:
        if not self.requests:
            return []
        return [i for i, slot in enumerate(indices) if slot in self.requests]

    def mask(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        input_ids: torch.Tensor,
        local_pos: torch.Tensor,
        masked_rows: torch.Tensor | None = None,
    ) -> None:
        _mask_reasoning_eos[(logits.shape[0],)](
            logits,
            logits.stride(0),
            self.slots.gpu,
            expanded_idx_mapping,
            input_ids,
            local_pos,
            masked_rows,
            num_warps=1,
        )


@triton.jit
def _mask_reasoning_eos(
    logits_ptr,
    logits_stride,
    slots_ptr,
    expanded_idx_mapping_ptr,
    input_ids_ptr,
    local_pos_ptr,
    masked_rows_ptr,
):
    row = tl.program_id(0)
    slot = tl.load(expanded_idx_mapping_ptr + row)
    first_eos_plus_one = tl.load(slots_ptr + 4 * slot)
    active = first_eos_plus_one > 0
    if active:
        local_pos = tl.load(local_pos_ptr + row)
        end = tl.load(slots_ptr + 4 * slot + 2)
        tool = tl.load(slots_ptr + 4 * slot + 3)
        # Packed row zero holds the last confirmed input. On verifier row i,
        # input IDs at offsets 1..i are the first i proposals.
        for offset in range(1, local_pos + 1):
            token = tl.load(input_ids_ptr + row - local_pos + offset)
            active &= (token != end) & (token != tool)
        if active:
            second_eos = tl.load(slots_ptr + 4 * slot + 1)
            tl.store(
                logits_ptr + row * logits_stride + first_eos_plus_one - 1, -float("inf")
            )
            tl.store(logits_ptr + row * logits_stride + second_eos, -float("inf"))
    if masked_rows_ptr is not None:
        tl.store(masked_rows_ptr + row, active)
