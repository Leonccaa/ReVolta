# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native seven-token lookup with per-step single-request admission."""

from typing import Any

import torch

from vllm.config.dflash2_lookup import ResolvedDFlash2Lookup
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
    suffix_lookup,
)


@triton.jit
def _native_eligibility_kernel(
    mapping,
    sampled,
    total_len,
    control,
    eligible,
    mapping_stride: tl.constexpr,
    STATES: tl.constexpr,
    HISTORY: tl.constexpr,
    MAX_LEN: tl.constexpr,
    NMIN: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(mapping + row * mapping_stride)
    valid_slot = (slot >= 0) & (slot < STATES)
    length = tl.load(total_len + slot, mask=valid_slot, other=0)
    allowed = (
        (row == 0)
        & (tl.load(control) != 0)
        & (tl.load(control + 1) != 0)
        & valid_slot
        & (tl.load(sampled + row) > 0)
        & (length >= NMIN + 1)
        & (length <= HISTORY)
        & (length < MAX_LEN)
    )
    tl.store(eligible + row, allowed.to(tl.int32))


@triton.jit
def _native_fuse_kernel(
    draft,
    draft_stride,
    lookup,
    lookup_stride,
    match,
    valid,
    eligible,
    use,
    NMIN: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, 8)
    mask = offsets < 7
    tl.store(use + row * 7 + offsets, 0, mask=mask)
    if tl.load(eligible + row) == 0:
        return
    first = tl.load(draft + row * draft_stride)
    first_lookup = tl.load(lookup + row * lookup_stride)
    take = (
        (tl.load(match + row) >= NMIN)
        & (tl.load(valid + row) > 0)
        & (first == first_lookup)
    )
    replace = take & (offsets > 0) & (offsets < tl.load(valid + row)) & mask
    tokens = tl.load(lookup + row * lookup_stride + offsets, mask=mask, other=0)
    tl.store(draft + row * draft_stride + offsets, tokens, mask=replace)
    tl.store(use + row * 7 + offsets, replace.to(tl.int32), mask=mask)


class NativeDFlash2Lookup:
    """Persistent graph inputs and scratch. Sampling remains owned by the selector."""

    def __init__(
        self,
        config: ResolvedDFlash2Lookup,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
    ):
        if (
            config.mode != "gpu_head"
            or config.policy != "longest"
            or config.draft_tokens != 7
            or config.verify_tokens != 7
        ):
            raise ValueError("Native lookup requires longest and K=D=7")
        assert config.nmin is not None and config.nmax is not None
        assert config.search_window is not None
        self.nmin, self.nmax = config.nmin, config.nmax
        self.search_window = config.search_window
        self.max_model_len = max_model_len
        self.tokens = torch.zeros((max_num_reqs, 7), dtype=torch.int32, device=device)
        self.match = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.valid = torch.zeros_like(self.match)
        self.eligible = torch.zeros_like(self.match)
        self.use = torch.zeros_like(self.tokens)
        self.sampled = torch.zeros_like(self.match)
        self.control = torch.zeros(2, dtype=torch.int32, device=device)

    def reset_for_capture(self) -> None:
        self.control.zero_()
        self.sampled.zero_()

    def prepare(self, batch: Any, sampled: torch.Tensor, *, real_run: bool) -> None:
        single = (
            batch.num_reqs == 1
            and not batch.has_structured_output_reqs
            and not batch.is_prefilling_np[0]
        )
        self.control[0].fill_(int(single))
        self.control[1].fill_(int(real_run))
        self.sampled.zero_()
        self.sampled[: batch.num_reqs].copy_(
            sampled[: batch.num_reqs], non_blocking=True
        )

    def apply(
        self,
        states: Any,
        mapping: torch.Tensor,
        draft: torch.Tensor,
        num_reqs: int,
        logits: torch.Tensor | None,
        cached_ids: torch.Tensor,
        cached_scores: torch.Tensor | None,
    ) -> None:
        if states is None:
            raise RuntimeError("Native lookup history must be bound before capture")
        history, lengths = states.all_token_ids.gpu, states.total_len.gpu
        _native_eligibility_kernel[(num_reqs,)](
            mapping,
            self.sampled,
            lengths,
            self.control,
            self.eligible,
            mapping_stride=7,
            STATES=lengths.numel(),
            HISTORY=history.shape[1],
            MAX_LEN=self.max_model_len,
            NMIN=self.nmin,
            num_warps=1,
        )
        suffix_lookup(
            history,
            lengths,
            mapping,
            self.eligible,
            num_reqs,
            7,
            idx_mapping_stride=7,
            nmin=self.nmin,
            nmax=self.nmax,
            search_max=self.search_window,
            out_tokens=self.tokens,
            out_len=self.match,
            out_valid=self.valid,
            native=True,
        )
        _native_fuse_kernel[(num_reqs,)](
            draft,
            draft.stride(0),
            self.tokens,
            self.tokens.stride(0),
            self.match,
            self.valid,
            self.eligible,
            self.use,
            NMIN=self.nmin,
            num_warps=1,
        )
        if logits is not None:
            width = cached_ids.shape[-1]
            _point_mass_draft_logits_kernel[(num_reqs * 7,)](
                logits,
                cached_ids,
                cached_ids if cached_scores is None else cached_scores,
                draft,
                draft.stride(0),
                self.use,
                mapping,
                7,
                cached_ids.stride(0),
                cached_ids.stride(1),
                logits.stride(0),
                logits.stride(1),
                num_steps=7,
                top_k=width,
                BLOCK_K=triton.next_power_of_2(width),
                CACHE_SCORES=cached_scores is not None,
                num_warps=1,
            )
