# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import gcd

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
    fp32_to_fp8_e4m3fn_bits,
)
from vllm.models.deepseek_v4.sm70.indexer import (
    _EPILOGUE_BLOCK_H,
    _EPILOGUE_BLOCK_K,
    _PREFILL_TILE_MB,
    _RELU_BLOCK_N,
    _relu_weight_headsum,
)
from vllm.triton_utils import tl, triton

_INDEX_PREFILL_GROUP_TILE = 65536
_INDEX_PREFILL_GEMM_BUDGET = _PREFILL_TILE_MB * 2**18


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
    LSE,
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
    if LSE is None:
        # The sink contributes one denominator term and has zero value.
        m = tl.load(Sinks + heads, heads < HEADS, other=0).to(tl.float32)
        z = tl.full((H,), 1.0, tl.float32)
    else:
        m = tl.full((H,), -float("inf"), tl.float32)
        z = tl.full((H,), 0.0, tl.float32)
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
    if LSE is not None:
        tl.store(LSE + row * HEADS + heads, m + tl.log(z), heads < HEADS)
        z = tl.where(length > 0, z, 1.0)
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
    sinks: torch.Tensor | None,
    scale: float,
    latent: int,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed physical slots per query with FP32 softmax."""
    tokens, heads, dim = q.shape
    out = q.new_empty(
        (tokens, heads, latent), dtype=q.dtype if lse is None else torch.float32
    )
    _paged_attention[(tokens, triton.cdiv(heads, 16))](
        q,
        cache,
        slots,
        counts,
        sinks,
        out,
        lse,
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
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
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
    if WORLD > 1:
        upper = (upper + WORLD - 1 - RANK) // WORLD
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
    # DCP selection reads only the live prefix, so unused tiles need no fill.
    if not DECODE or WORLD == 1 or start < upper:
        tl.store(Logits + row * GROUPS + g, logits, g < GROUPS)


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
    world: int = 1,
    rank: int = 0,
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
        world,
        rank,
        num_warps=4,
    )
    return logits


def index_prefill_key_rows(groups: int) -> int:
    return min(groups, _INDEX_PREFILL_GROUP_TILE)


def index_prefill_gemm_elements(rows: int, key_rows: int) -> int:
    """Bound the FP32 GEMM output for rows of flattened query heads."""
    return min(rows * key_rows, max(rows, _INDEX_PREFILL_GEMM_BUDGET))


@triton.jit
def _gather_index_prefill_keys(
    Pool,
    Table,
    Keys,
    stride_pool_p,
    stride_pool_g,
    start,
    live,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    N: tl.constexpr,
):
    # Adapted from the DeepSeek V4 SM70 paged index key gather.
    col = tl.program_id(0) * N + tl.arange(0, N)
    group = start + col
    logical_page = group // PAGE
    page = tl.load(Table + logical_page // PAGES_PER_BLOCK, col < live, other=0)
    page = page * PAGES_PER_BLOCK + logical_page % PAGES_PER_BLOCK
    d = tl.arange(0, DIM)
    keys = tl.load(
        Pool
        + page[:, None].to(tl.int64) * stride_pool_p
        + (group % PAGE)[:, None] * stride_pool_g
        + d[None, :],
        col[:, None] < live,
        other=0,
    )
    tl.store(Keys + col[:, None] * DIM + d[None, :], keys, col[:, None] < live)


@triton.jit
def _index_prefill_epilogue(
    Scores,
    W,
    Logits,
    first,
    groups,
    start,
    width,
    HEADS: tl.constexpr,
    RATIO: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    upper = ((first + row) // RATIO + WORLD - 1 - RANK) // WORLD
    visible = start + col < upper
    acc = _relu_weight_headsum(
        Scores,
        W,
        row,
        col,
        (col < width) & visible,
        HEADS * width,
        width,
        HEADS,
        HEADS,
        BLOCK_K,
        BLOCK_H,
    )
    tl.store(
        Logits + row * groups + start + col,
        tl.where(visible, acc, float("-inf")),
        col < width,
    )


def index_prefill_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    pool: torch.Tensor,
    table: torch.Tensor,
    first: int,
    groups: int,
    ratio: int,
    pages_per_block: int,
    *,
    world: int = 1,
    rank: int = 0,
) -> torch.Tensor:
    rows, heads, dim = q.shape
    q2 = q.flatten(0, 1)
    key_rows = index_prefill_key_rows(groups)
    score_cols = index_prefill_gemm_elements(rows * heads, key_rows) // (rows * heads)
    if score_cols < key_rows:
        score_cols = score_cols // _RELU_BLOCK_N * _RELU_BLOCK_N or score_cols
    block_h = gcd(heads, _EPILOGUE_BLOCK_H)
    keys = pool.new_empty((key_rows, dim))
    logits = q.new_empty((rows, groups), dtype=torch.float32)
    # The final group in the output can still be open, so only gather keys
    # visible to at least one row. Convert the bound to rank-local group IDs.
    upper = ((first + rows - 1) // ratio + world - 1 - rank) // world
    for start in range(0, upper, key_rows):
        live = min(key_rows, upper - start)
        _gather_index_prefill_keys[(triton.cdiv(live, 16),)](
            pool,
            table,
            keys,
            pool.stride(0),
            pool.stride(1),
            start,
            live,
            dim,
            pool.shape[1],
            pages_per_block,
            16,
        )
        for col in range(0, live, score_cols):
            key_tile = keys[col : min(col + score_cols, live)]
            # As in the DeepSeek V4 SM70 cuBLAS scorer, fold heads into M.
            # Keep the GEMM output in FP32 for the top-k score comparison.
            scores = torch.mm(q2, key_tile.t(), out_dtype=torch.float32)
            width = scores.shape[1]
            _index_prefill_epilogue[(rows, triton.cdiv(width, _EPILOGUE_BLOCK_K))](
                scores,
                weights,
                logits,
                first,
                groups,
                start + col,
                width,
                heads,
                ratio,
                world,
                rank,
                _EPILOGUE_BLOCK_K,
                block_h,
                num_warps=4,
            )
            del scores
    logits[:, upper:].fill_(-torch.inf)
    return logits
