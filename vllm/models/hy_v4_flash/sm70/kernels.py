# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
    fp32_to_fp8_e4m3fn_bits,
)
from vllm.triton_utils import tl, triton


@triton.jit
def load_main_kv(kb, d, r, live, LATENT: tl.constexpr):
    k = tl.load(kb[:, None] + d[None, :], live[:, None], other=0)
    if kb.dtype.element_ty == tl.uint8:
        # vLLM fp8_ds_mla layout with an FP16 RoPE tail on SM70.
        groups: tl.constexpr = LATENT // 128
        scales = (kb + LATENT).to(tl.pointer_type(tl.float32))
        scale = tl.load(scales[:, None] + tl.arange(0, groups), live[:, None], other=0)
        decoded = fp8_e4m3fn_bits_to_fp32_bitcast(k).reshape((kb.shape[0], groups, 128))
        k = (decoded * scale[:, :, None]).reshape(k.shape).to(tl.float16)
        rope = (kb + LATENT + groups * 4).to(tl.pointer_type(tl.float16))
    else:
        rope = kb + LATENT
    kr = tl.load(rope[:, None] + r[None, :], live[:, None], other=0)
    return k, kr


def pack_fp8_ds_mla_torch(kv_c: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """Torch reference for fp8_ds_mla with an FP16 RoPE tail."""
    groups = kv_c.float().unflatten(-1, (-1, 128))
    scales = (groups.abs().amax(-1) / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    codes = (groups / scales[..., None]).to(torch.float8_e4m3fn).flatten(-2)
    return torch.cat(
        (codes.view(torch.uint8), scales.view(torch.uint8), k_pe.view(torch.uint8)),
        dim=-1,
    )


@triton.jit
def _cache_fp8_ds_mla(
    Latent,
    Rope,
    Cache,
    Slots,
    stride_latent,
    stride_rope,
    stride_page,
    stride_row,
    PAGE: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(Slots + token)
    if slot < 0:
        return

    row = Cache + (slot // PAGE) * stride_page + (slot % PAGE) * stride_row
    group = tl.arange(0, 4)
    d = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    values = tl.load(Latent + token * stride_latent + d).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=1)
    scale = tl.maximum(tl.div_rn(amax, 448.0), 2.0**-126)
    codes = fp32_to_fp8_e4m3fn_bits(tl.div_rn(values, scale[:, None]))
    tl.store(row + d, codes)
    tl.store((row + 512).to(tl.pointer_type(tl.float32)) + group, scale)

    r = tl.arange(0, 64)
    rope = tl.load(Rope + token * stride_rope + r)
    tl.store((row + 528).to(tl.pointer_type(tl.float16)) + r, rope)


def cache_fp8_ds_mla(
    kv_c: torch.Tensor, k_pe: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor
) -> None:
    _cache_fp8_ds_mla[(slots.numel(),)](
        kv_c,
        k_pe,
        cache,
        slots,
        kv_c.stride(0),
        k_pe.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.shape[1],
        num_warps=4,
    )


def _unpack_fp8_ds_mla_torch(kv: torch.Tensor, latent: int) -> torch.Tensor:
    rope_offset = latent + latent // 128 * 4
    scales = kv[..., latent:rope_offset].view(torch.float32)
    groups = kv[..., :latent].view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 128))
    kv_c = (groups * scales[..., None]).flatten(-2).to(torch.float16)
    return torch.cat((kv_c, kv[..., rope_offset:].view(torch.float16)), dim=-1)


@triton.jit
def _paged_attention(
    Q,
    KV,
    Indices,
    Counts,
    Sinks,
    Out,
    stride_qt,
    stride_qh,
    stride_kvp,
    stride_kvr,
    stride_indices,
    stride_out,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
):
    # Adapted from DeepSeek V4 SM70 _sm70_sparse_gathered_kernel. The
    # extra rotary channels enter QK only; V is the shared latent.
    row = tl.program_id(0)
    heads = tl.program_id(1) * H + tl.arange(0, H)
    d = tl.arange(0, LATENT)
    r = tl.arange(0, ROPE)
    qb = Q + row * stride_qt + heads[:, None] * stride_qh
    q = tl.load(qb + d[None, :], heads[:, None] < HEADS, other=0)
    qr = tl.load(qb + LATENT + r[None, :], heads[:, None] < HEADS, other=0)
    # Initialize with the sink's single denominator term and zero value.
    m = tl.load(Sinks + heads, heads < HEADS, other=0).to(tl.float32)
    z = tl.full((H,), 1.0, tl.float32)
    acc = tl.full((H, LATENT), 0.0, tl.float32)
    length = tl.load(Counts + row)
    for start in range(0, length, N):
        n = start + tl.arange(0, N)
        live = n < length
        slot = tl.load(Indices + row * stride_indices + n, live, other=0)
        kb = KV + (slot // PAGE).to(tl.int64) * stride_kvp + (slot % PAGE) * stride_kvr
        k, kr = load_main_kv(kb, d, r, live, LATENT)
        logits = (tl.dot(q, tl.trans(k)) + tl.dot(qr, tl.trans(kr))) * SCALE
        logits = tl.where(live[None, :], logits, float("-inf"))
        new_m = tl.maximum(m, tl.max(logits, axis=1))
        rescale = tl.exp(m - new_m)
        p = tl.exp(logits - new_m[:, None])
        acc = tl.dot(p.to(tl.float16), k, acc * rescale[:, None])
        z = z * rescale + tl.sum(p, axis=1)
        m = new_m
    tl.store(
        Out + row * stride_out + heads[:, None] * LATENT + d[None, :],
        acc / z[:, None],
        heads[:, None] < HEADS,
    )


def paged_attention(
    q: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    latent: int,
) -> torch.Tensor:
    """Packed physical slots per query with FP32 softmax."""
    tokens, heads, dim = q.shape
    out = q.new_empty((tokens, heads, latent))
    _paged_attention[(tokens, triton.cdiv(heads, 16))](
        q,
        cache,
        slots,
        counts,
        sinks,
        out,
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        cache.stride(1),
        slots.stride(0),
        out.stride(0),
        heads,
        cache.shape[1],
        latent,
        dim - latent,
        scale,
        16,
        16,
        num_warps=4,
    )
    return out


def paged_attention_torch(
    q: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    latent: int,
) -> torch.Tensor:
    """Small CPU counterpart of the paged latent aggregation."""
    out = q.new_empty((*q.shape[:2], latent))
    page = cache.shape[1]
    for start in range(0, q.shape[0], 32):
        end = start + 32
        indices = slots[start:end]
        live = torch.arange(indices.shape[1], device=q.device) < counts[start:end, None]
        indices = indices.masked_fill(~live, 0)
        kv = cache[indices // page, indices % page]
        if cache.dtype == torch.uint8:
            kv = _unpack_fp8_ds_mla_torch(kv, latent)
        kv = kv.float().masked_fill(~live[..., None], 0)
        scores = ((q[start:end].float() @ kv.transpose(1, 2)) * scale).masked_fill(
            ~live[:, None, :], -torch.inf
        )
        sink = sinks.float()[None, :, None].expand(scores.shape[0], -1, -1)
        probs = torch.cat((scores, sink), dim=-1).softmax(-1)[..., :-1]
        out[start:end] = probs @ kv[..., :latent]
    return out


@triton.jit
def _index_scores(
    Q,
    W,
    Pool,
    Table,
    Logits,
    Lengths,
    Slots,
    stride_qt,
    stride_qh,
    stride_pool_p,
    stride_pool_g,
    stride_table,
    FIRST,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    GROUPS,
    RATIO: tl.constexpr,
    GROUP_TOPK: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    DECODE: tl.constexpr,
):
    # Weighted ReLU head-sum as in the DeepSeek V4 SM70 indexer, with
    # plain FP16 pages in place of software-decoded FP8 keys.
    row = tl.program_id(0)
    head = tl.arange(0, H)
    d = tl.arange(0, DIM)
    start = tl.program_id(1) * N
    g = start + tl.arange(0, N)
    if DECODE:
        length = tl.load(Lengths + row)
        active = (tl.load(Slots + row) >= 0) & (length > GROUP_TOPK * RATIO)
        upper = tl.where(active, (length - 1) // RATIO, 0)
    else:
        upper = (FIRST + row) // RATIO
    logits = tl.full((N,), float("-inf"), tl.float32)
    if not DECODE or start < upper:
        logical_page = g // PAGE
        page = tl.load(
            Table + row * stride_table + logical_page // PAGES_PER_BLOCK,
            g < upper,
            other=0,
        )
        page = page * PAGES_PER_BLOCK + logical_page % PAGES_PER_BLOCK
        k = tl.load(
            Pool
            + page[:, None].to(tl.int64) * stride_pool_p
            + (g % PAGE)[:, None] * stride_pool_g
            + d[None, :],
            g[:, None] < upper,
            other=0,
        )
        q = tl.load(
            Q + row * stride_qt + head[:, None] * stride_qh + d[None, :],
            head[:, None] < HEADS,
            other=0,
        )
        w = tl.load(W + row * HEADS + head, head < HEADS, other=0)
        scores = tl.maximum(tl.dot(q, tl.trans(k)), 0.0)
        logits = tl.where(g < upper, tl.sum(scores * w[:, None], axis=0), float("-inf"))
    tl.store(
        Logits + row * GROUPS + g,
        logits,
        g < GROUPS,
    )


def index_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    pool: torch.Tensor,
    table: torch.Tensor,
    first: int,
    groups: int,
    ratio: int,
    pages_per_block: int,
    *,
    logits: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    slots: torch.Tensor | None = None,
    group_topk: int = 0,
) -> torch.Tensor:
    if logits is None:
        logits = q.new_empty((q.shape[0], groups), dtype=torch.float32)
    _index_scores[(q.shape[0], triton.cdiv(groups, 64))](
        q,
        weights,
        pool,
        table,
        logits,
        lengths,
        slots,
        q.stride(0),
        q.stride(1),
        pool.stride(0),
        pool.stride(1),
        table.stride(0) if lengths is not None else 0,
        first,
        q.shape[1],
        q.shape[2],
        pool.shape[1],
        pages_per_block,
        groups,
        ratio,
        group_topk,
        max(16, triton.next_power_of_2(q.shape[1])),
        64,
        lengths is not None,
        num_warps=4,
    )
    return logits
