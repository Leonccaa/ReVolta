# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


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
        k = tl.load(kb[:, None] + d[None, :], live[:, None], other=0)
        kr = tl.load(kb[:, None] + LATENT + r[None, :], live[:, None], other=0)
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
    """Packed physical slots per query; FP16 main pages with FP32 softmax."""
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
        kv = (
            cache[indices // page, indices % page]
            .float()
            .masked_fill(~live[..., None], 0)
        )
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
    stride_qt,
    stride_qh,
    stride_pool_p,
    stride_pool_g,
    FIRST,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    GROUPS,
    RATIO: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
):
    # Weighted ReLU head-sum as in the DeepSeek V4 SM70 indexer, with
    # plain FP16 pages in place of software-decoded FP8 keys.
    row = tl.program_id(0)
    head = tl.arange(0, H)
    d = tl.arange(0, DIM)
    g = tl.program_id(1) * N + tl.arange(0, N)
    upper = (FIRST + row) // RATIO
    logical_page = g // PAGE
    page = tl.load(Table + logical_page // PAGES_PER_BLOCK, g < upper, other=0)
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
    logits = tl.sum(scores * w[:, None], axis=0)
    tl.store(
        Logits + row * GROUPS + g,
        tl.where(g < upper, logits, float("-inf")),
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
) -> torch.Tensor:
    logits = q.new_empty((q.shape[0], groups), dtype=torch.float32)
    _index_scores[(q.shape[0], triton.cdiv(groups, 64))](
        q,
        weights,
        pool,
        table,
        logits,
        q.stride(0),
        q.stride(1),
        pool.stride(0),
        pool.stride(1),
        first,
        q.shape[1],
        q.shape[2],
        pool.shape[1],
        pages_per_block,
        groups,
        ratio,
        max(16, triton.next_power_of_2(q.shape[1])),
        64,
        num_warps=4,
    )
    return logits
