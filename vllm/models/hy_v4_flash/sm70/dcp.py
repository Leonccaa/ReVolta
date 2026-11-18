# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hy4 Flash DCP selection and deferred attention sink."""

import torch

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.triton_utils import tl, triton

# Candidate pack and stable merge adapted from vllm-project/vllm#57239.
TOPK = 512


@triton.jit
def pack_dcp_topk_candidates(
    Logits,
    Indices,
    Packed,
    stride,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, K)
    local = tl.load(Indices + row * K + i)
    valid = local >= 0
    score = tl.load(Logits + row * stride + local, valid, other=-float("inf"))
    gid = tl.where(valid, local * WORLD + RANK, -1)
    base = Packed + (row * K + i) * 2
    tl.store(base, score)
    tl.store(base + 1, gid.to(tl.float32))


@triton.jit
def _stable_key(base, columns, ID_MASK: tl.constexpr, VALID: tl.constexpr):
    score = tl.load(base + columns * 2)
    gid = tl.load(base + columns * 2 + 1).to(tl.int32)
    # Give signed zeros the same key and order scores before complemented IDs.
    bits = score.to(tl.uint32, bitcast=True)
    bits = tl.where(bits == 0x80000000, 0, bits)
    ordered = bits ^ tl.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    key = VALID | (ordered.to(tl.int64) << 30) | (ID_MASK - gid).to(tl.int64)
    return tl.where(gid >= 0, key, 0)


@triton.jit
def stable_topk_from_gathered_candidates_triton(
    Gathered, Out, WORLD: tl.constexpr, K: tl.constexpr
):
    ID_MASK: tl.constexpr = (1 << 30) - 1
    VALID: tl.constexpr = 1 << 62
    row = tl.program_id(0)
    base = Gathered + row * WORLD * K * 2
    offsets = tl.arange(0, K)
    winners = tl.topk(_stable_key(base, offsets, ID_MASK, VALID), K)
    for start in tl.range(K, WORLD * K, K):
        tile = tl.topk(_stable_key(base, start + offsets, ID_MASK, VALID), K)
        winners = tl.topk(tl.cat(winners, tile, can_reorder=True), K)
    gids = (ID_MASK - (winners & ID_MASK)).to(tl.int32)
    tl.store(Out + row * K + offsets, tl.where(winners >= VALID, gids, -1))


def select_history(
    logits: torch.Tensor, lengths: torch.Tensor, group: GroupCoordinator
) -> torch.Tensor:
    rows, stride = logits.shape
    indices = logits.new_empty((rows, TOPK), dtype=torch.int32)
    torch.ops._C.qsa_lexicographic_topk(logits, lengths, indices, TOPK)
    packed = logits.new_empty((rows, TOPK, 2))
    pack_dcp_topk_candidates[(rows,)](
        logits,
        indices,
        packed,
        stride,
        group.world_size,
        group.rank_in_group,
        TOPK,
        num_warps=8,
    )
    gathered = group.all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_triton[(rows,)](
        gathered, indices, group.world_size, TOPK, num_warps=8
    )
    # Keep the 511 score winners before sorting into attention read order.
    return indices[:, : TOPK - 1].sort(dim=-1).values


def apply_attn_sink(
    out: torch.Tensor, lse: torch.Tensor, attn_sink: torch.Tensor
) -> torch.Tensor:
    scale = torch.exp(lse - torch.logaddexp(lse, attn_sink.to(lse.dtype)))
    return out * scale.to(out.dtype).unsqueeze(-1)
