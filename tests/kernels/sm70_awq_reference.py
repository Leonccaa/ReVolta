# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent AWQ reference helpers for SM70 kernel tests.

Copied verbatim from benchmarks/benchmark_sm70_turbomind_exactness.py at
69ff71dc82, before benchmarks/ was removed from ReVolta.
"""

import torch

from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _align_awq_input_dim,
    _align_awq_output_dim,
)


def _unpack_int4_cols(
    packed: torch.Tensor, order: list[int] | None = None
) -> torch.Tensor:
    vals = [((packed >> (4 * i)) & 0xF).to(torch.float16) for i in range(8)]
    if order is not None:
        vals = [vals[i] for i in order]
    return torch.stack(vals, dim=-1).reshape(packed.shape[0], packed.shape[1] * 8)


def _unpack_awq_gemm_qweight(qweight: torch.Tensor) -> torch.Tensor:
    k = qweight.shape[0]
    n = qweight.shape[1] * 8
    if k % 8 != 0:
        raise ValueError("AWQ qweight K must be divisible by 8.")
    vals = [((qweight >> (4 * i)) & 0xF).to(torch.float16) for i in range(8)]
    flat = torch.stack(vals, dim=-1).reshape(-1)
    return flat.reshape(n, k // 8, 2, 4).permute(0, 1, 3, 2).contiguous().reshape(k, n)


def _awq_reference_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    # Matches the AWQ unpacking used by awq_sm70_prepare. The upstream
    # awq_dequantize CUDA reference uses SM75-only inline PTX and is not a
    # valid SM70 quality oracle.
    zero_order = [0, 4, 1, 5, 2, 6, 3, 7]
    w = _unpack_awq_gemm_qweight(qweight)
    zeros = _unpack_int4_cols(qzeros, zero_order)
    row_groups = (
        torch.arange(qweight.shape[0], device=qweight.device) // group_size
    ).long()
    return ((w - zeros[row_groups]) * scales[row_groups]).to(torch.float16)


def _prepare_awq_moe_checkpoint_for_tp(
    tensors: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tp_size: int,
    tp_rank: int,
    checkpoint_group_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid AWQ MoE TP size or rank.")
    w13_qweight, w13_scales, w13_qzeros, w2_qweight, w2_scales, w2_qzeros = tensors
    hidden_size = int(w13_qweight.shape[-2])
    intermediate_size = int(w13_scales.shape[-1]) // 2
    if intermediate_size % tp_size:
        raise ValueError("AWQ intermediate size cannot be evenly sharded for TP.")
    intermediate_size_per_partition = intermediate_size // tp_size
    group_size = checkpoint_group_size
    if hidden_size % group_size or intermediate_size_per_partition % group_size:
        raise ValueError("AWQ dimensions must be divisible by the checkpoint group.")

    w13_q_half = w13_qweight.shape[-1] // 2
    w13_n_half = w13_scales.shape[-1] // 2
    if w13_q_half % tp_size or w13_n_half % tp_size:
        raise ValueError("AWQ W13 output cannot be evenly sharded for TP.")
    q_start = tp_rank * (w13_q_half // tp_size)
    q_end = (tp_rank + 1) * (w13_q_half // tp_size)
    n_start = tp_rank * (w13_n_half // tp_size)
    n_end = (tp_rank + 1) * (w13_n_half // tp_size)
    w13_qweight = torch.cat(
        (
            w13_qweight[..., q_start:q_end],
            w13_qweight[..., w13_q_half + q_start : w13_q_half + q_end],
        ),
        dim=-1,
    ).contiguous()
    w13_qzeros = torch.cat(
        (
            w13_qzeros[..., q_start:q_end],
            w13_qzeros[..., w13_q_half + q_start : w13_q_half + q_end],
        ),
        dim=-1,
    ).contiguous()
    w13_scales = torch.cat(
        (
            w13_scales[..., n_start:n_end],
            w13_scales[..., w13_n_half + n_start : w13_n_half + n_end],
        ),
        dim=-1,
    ).contiguous()

    w2_k = w2_qweight.shape[-2]
    if w2_k % tp_size or (w2_k // tp_size) % group_size:
        raise ValueError("AWQ W2 input cannot be evenly sharded by groups for TP.")
    k_start = tp_rank * (w2_k // tp_size)
    k_end = (tp_rank + 1) * (w2_k // tp_size)
    group_start = k_start // group_size
    group_end = k_end // group_size
    w13_qweight, w13_scales, w13_qzeros, _ = _align_awq_output_dim(
        w13_qweight,
        w13_scales,
        w13_qzeros,
        8,
        group_size * 2,
    )
    w2_qweight, w2_scales, w2_qzeros, _ = _align_awq_input_dim(
        w2_qweight[..., k_start:k_end, :].contiguous(),
        w2_scales[..., group_start:group_end, :].contiguous(),
        w2_qzeros[..., group_start:group_end, :].contiguous(),
        group_size,
        group_size,
    )
    w2_qweight, w2_scales, w2_qzeros, _ = _align_awq_output_dim(
        w2_qweight,
        w2_scales,
        w2_qzeros,
        8,
        group_size,
    )
    return (
        w13_qweight,
        w13_scales,
        w13_qzeros,
        w2_qweight,
        w2_scales,
        w2_qzeros,
        group_size,
    )
