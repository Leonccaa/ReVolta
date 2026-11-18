# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Split-K decode for Hy4 Flash's paged MLA cache."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

from .kernels import load_main_kv

# Two 16-key tiles keep the SM70 shared-memory footprint within a CTA limit.
_KEYS_PER_SPLIT = 32


def decode_workspace_specs(
    rows: int, heads: int, width: int, latent: int
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    splits = cdiv(width, _KEYS_PER_SPLIT)
    return (
        ((rows, heads, splits, 2), torch.float32),
        ((rows, heads, splits, latent), torch.float32),
    )


@triton.jit
def _decode_split(
    Q,
    KV,
    Indices,
    Counts,
    State,
    Acc,
    stride_qt,
    stride_qh,
    stride_kvp,
    stride_kvr,
    stride_indices,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
):
    # Adapted from DeepSeek V4 SM70 _sm70_sparse_paged_fp8_splitk_kernel.
    row = tl.program_id(0)
    split = tl.program_id(2)
    start = split * SPLIT_SIZE
    length = tl.load(Counts + row)
    if start < length:
        heads = tl.program_id(1) * H + tl.arange(0, H)
        d = tl.arange(0, LATENT)
        r = tl.arange(0, ROPE)
        qb = Q + row * stride_qt + heads[:, None] * stride_qh
        q = tl.load(qb + d[None, :], heads[:, None] < HEADS, other=0)
        qr = tl.load(qb + LATENT + r[None, :], heads[:, None] < HEADS, other=0)

        maximum = tl.full((H,), float("-inf"), tl.float32)
        total = tl.full((H,), 0.0, tl.float32)
        acc = tl.full((H, LATENT), 0.0, tl.float32)
        end = tl.minimum(start + SPLIT_SIZE, length)
        for tile in range(start, end, N):
            n = tile + tl.arange(0, N)
            live = n < end
            slot = tl.load(Indices + row * stride_indices + n, live, other=0)
            page = (slot // PAGE).to(tl.int64)
            kb = KV + page * stride_kvp + (slot % PAGE) * stride_kvr
            k, kr = load_main_kv(kb, d, r, live, LATENT)
            logits = (tl.dot(q, tl.trans(k)) + tl.dot(qr, tl.trans(kr))) * SCALE
            logits = tl.where(live[None, :], logits, float("-inf"))
            new_max = tl.maximum(maximum, tl.max(logits, axis=1))
            rescale = tl.exp(maximum - new_max)
            p = tl.exp(logits - new_max[:, None])
            # Triton lowers both variants to FP32 FMAs on SM70.
            acc = tl.dot(
                p, k.to(tl.float32), acc * rescale[:, None], input_precision="ieee"
            )
            total = total * rescale + tl.sum(p, axis=1)
            maximum = new_max

        state = (row * HEADS + heads) * SPLITS + split
        tl.store(State + state * 2, maximum, heads < HEADS)
        tl.store(State + state * 2 + 1, total, heads < HEADS)
        tl.store(
            Acc + state[:, None] * LATENT + d[None, :], acc, heads[:, None] < HEADS
        )


@triton.jit
def _decode_merge(
    State,
    Acc,
    Counts,
    Sinks,
    Out,
    LSE,
    HEADS: tl.constexpr,
    LATENT: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    # Empty splits are never read. A padding row has only the sink term.
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.program_id(2) * D + tl.arange(0, D)
    split = tl.arange(0, SPLIT_BLOCK)
    live = (split * SPLIT_SIZE < tl.load(Counts + row)) & (split < SPLITS)
    state = (row * HEADS + head) * SPLITS + split
    m = tl.load(State + state * 2, live, other=float("-inf"))
    total = tl.load(State + state * 2 + 1, live, other=0)
    final_m = tl.max(m, axis=0)
    if LSE is None:
        sink = tl.load(Sinks + head).to(tl.float32)
        final_m = tl.maximum(final_m, sink)
    else:
        final_m = tl.where(final_m == -float("inf"), 0.0, final_m)
    w = tl.exp(m - final_m)
    denominator = tl.sum(total * w, axis=0)
    if LSE is None:
        denominator += tl.exp(sink - final_m)
    else:
        if tl.program_id(2) == 0:
            tl.store(LSE + row * HEADS + head, final_m + tl.log(denominator))
        denominator = tl.where(denominator > 0, denominator, 1.0)
    acc = tl.load(Acc + state[:, None] * LATENT + d[None, :], live[:, None], other=0)
    numerator = tl.sum(acc * w[:, None], axis=0)
    tl.store(Out + (row * HEADS + head) * LATENT + d, numerator / denominator)


def paged_attention_splitk(
    q: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    sinks: torch.Tensor | None,
    scale: float,
    partial_state: torch.Tensor,
    partial_acc: torch.Tensor,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    rows, heads, dim = q.shape
    splits, latent = partial_acc.shape[-2:]
    out = q.new_empty((rows, heads, latent))
    _decode_split[(rows, triton.cdiv(heads, 8), splits)](
        q,
        cache,
        slots,
        counts,
        partial_state,
        partial_acc,
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        cache.stride(1),
        slots.stride(0),
        heads,
        cache.shape[1],
        latent,
        dim - latent,
        scale,
        splits,
        _KEYS_PER_SPLIT,
        8,
        16,
        num_warps=4,
    )
    reduce_dim = min(latent, 64)
    _decode_merge[(rows, heads, latent // reduce_dim)](
        partial_state,
        partial_acc,
        counts,
        sinks,
        out,
        lse,
        heads,
        latent,
        splits,
        triton.next_power_of_2(splits),
        _KEYS_PER_SPLIT,
        reduce_dim,
        num_warps=4,
    )
    return out
