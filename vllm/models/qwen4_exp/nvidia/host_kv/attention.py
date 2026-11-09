# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preallocated adapter to the unchanged QSA Triton split/merge kernels.

This deliberately binds the Triton computation plan, not the production
grouped/XQA dispatch. Both GPU-only and host-backed arms must use that same
plan. Selection width, column order, split boundaries, gating and the point
where V scale is applied are unchanged. CUDA tests also compare the public
GPU-only Triton route to this adapter.
"""

from __future__ import annotations

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    _qsa_merge_splitk_kernel,
    _qsa_sparse_launch_profile,
    _qsa_sparse_paged_gqa_splitk_kernel,
    _use_sm70_qsa_two_warp_partial,
)
from vllm.triton_utils import triton

from .pool import LayerScales


class SparseAttentionPlan:
    def __init__(
        self,
        *,
        rows: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        selection_width: int,
        device: torch.device | str,
        e4m3: bool,
        scales: LayerScales | None = None,
        pre_ampere: bool = True,
        output_dtype: torch.dtype = torch.float16,
    ) -> None:
        if (
            min(rows, query_heads, kv_heads, head_dim, selection_width) <= 0
            or query_heads % kv_heads
            or head_dim < 16
            or head_dim & (head_dim - 1)
            or output_dtype not in (torch.float16, torch.float32)
        ):
            raise ValueError("Invalid QSA attention plan geometry or output dtype")
        group = query_heads // kv_heads
        self.block_n, target, self.warps = _qsa_sparse_launch_profile(
            rows * kv_heads,
            triton.next_power_of_2(group),
            pre_ampere,
        )
        if pre_ampere and _use_sm70_qsa_two_warp_partial(rows, group, head_dim):
            self.warps = 2
        self.tiles = triton.cdiv(selection_width, self.block_n)
        self.splits = min(1 << (self.tiles.bit_length() - 1), target)
        self.rows, self.heads, self.kv_heads = rows, query_heads, kv_heads
        self.dim, self.width = head_dim, selection_width
        self.e4m3 = e4m3
        self.scales = LayerScales() if scales is None else scales
        self.output = torch.empty(
            (rows, query_heads, head_dim), dtype=output_dtype, device=device
        )
        self.lse = torch.empty((rows, query_heads), dtype=torch.float32, device=device)
        if self.splits == 1:
            self.partial_output, self.partial_lse = self.output, self.lse
        else:
            self.partial_output = torch.empty(
                (self.splits, rows, query_heads, head_dim),
                dtype=torch.float32,
                device=device,
            )
            self.partial_lse = torch.empty(
                (self.splits, rows, query_heads), dtype=torch.float32, device=device
            )

    def run(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor,
        block_table: torch.Tensor,
        requests: torch.Tensor,
        *,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run with logical IDs/table, or remapped IDs/identity hot table.

        Inputs must already be causally masked, as on the existing DCP-local
        Triton path. No new arithmetic, allocation or device-to-host read.
        Output and LSE remain valid until the next run on this plan.
        """
        if (
            query.shape != self.output.shape
            or query.dtype != torch.float16
            or query.stride(-1) != 1
            or key.ndim != 4
            or value.shape != key.shape
            or not all(key.shape)
            or key.shape[2:] != (self.kv_heads, self.dim)
            or key.dtype != (torch.uint8 if self.e4m3 else torch.float16)
            or value.dtype != key.dtype
            or key.stride(-1) != 1
            or value.stride(-1) != 1
            or indices.shape != (self.rows, self.width)
            or indices.dtype != torch.int32
            or not indices.is_contiguous()
            or block_table.ndim != 2
            or not all(block_table.shape)
            or block_table.dtype != torch.int32
            or block_table.stride(1) != 1
            or requests.shape != (self.rows,)
            or requests.dtype != torch.int32
            or not requests.is_contiguous()
        ):
            raise ValueError("QSA tensors disagree with the bound attention plan")
        tensors = (query, key, value, indices, block_table, requests)
        if any(x.device != self.output.device for x in tensors):
            raise ValueError("QSA plan inputs must share its device")
        if output_gate is not None and (
            output_gate.shape != query.shape
            or output_gate.dtype != query.dtype
            or output_gate.device != query.device
            or output_gate.stride(-1) != 1
        ):
            raise ValueError("QSA output gate must match query")
        gate_s0 = output_gate.stride(0) if output_gate is not None else 0
        gate_s1 = output_gate.stride(1) if output_gate is not None else 0
        _qsa_sparse_paged_gqa_splitk_kernel[(self.rows, self.kv_heads, self.splits)](
            query,
            key,
            value,
            indices,
            block_table,
            requests,
            self.partial_output,
            self.partial_lse,
            self.output,
            self.lse,
            output_gate,
            query.stride(0),
            query.stride(1),
            *key.stride()[:3],
            *value.stride()[:3],
            indices.stride(0),
            block_table.stride(0),
            self.output.stride(0),
            self.output.stride(1),
            gate_s0,
            gate_s1,
            self.rows,
            key.shape[0],
            block_table.shape[0],
            self.scales.key,
            self.scales.value,
            TOPK=self.width,
            PAGE_SIZE=key.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=self.heads // self.kv_heads,
            HEAD_DIM=self.dim,
            NUM_QUERY_HEADS=self.heads,
            NUM_SPLITS=self.splits,
            NUM_TILES=self.tiles,
            BLOCK_M=triton.next_power_of_2(self.heads // self.kv_heads),
            BLOCK_N=self.block_n,
            KV_E4M3=self.e4m3,
            RESOLVED_INDICES=False,
            num_warps=self.warps,
            num_stages=2,
        )
        if self.splits > 1:
            _qsa_merge_splitk_kernel[(self.rows, self.heads)](
                self.partial_output,
                self.partial_lse,
                self.output,
                self.lse,
                output_gate,
                self.output.stride(0),
                self.output.stride(1),
                gate_s0,
                gate_s1,
                self.rows,
                self.scales.value,
                HEAD_DIM=self.dim,
                NUM_QUERY_HEADS=self.heads,
                NUM_SPLITS=self.splits,
                BLOCK_SPLITS=triton.next_power_of_2(self.splits),
                KV_E4M3=self.e4m3,
                num_warps=2,
                num_stages=1,
            )
        return self.output
