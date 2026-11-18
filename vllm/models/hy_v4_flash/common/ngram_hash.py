# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash n-gram hashes. Lookback rows are ordered oldest to newest."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _hash_ids_kernel(
    input_ids,
    query_start_loc,
    lookback,
    token_maps,
    map_sizes,
    multipliers,
    primes,
    offsets,
    output,
    num_tokens,
    num_actual_tokens,
    num_queries,
    pad_token_id,
    map_stride: tl.constexpr,
    num_heads: tl.constexpr,
    max_ngram: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Adapted from deepseek_v41/common/engram.py's stateless hash kernel.
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    valid = token < num_tokens
    live_token = valid & (
        token < tl.minimum(num_actual_tokens, tl.load(query_start_loc + num_queries))
    )
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.where(live_token, num_queries, 0)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + mid + 1, lo < hi, other=0)
        active = lo < hi
        right = token >= end
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    req = tl.minimum(lo, num_queries - 1).to(tl.int64)
    start = tl.load(query_start_loc + req, live_token, other=0)
    pad = tl.load(token_maps + layer * map_stride + pad_token_id).to(tl.int64)
    size = tl.load(map_sizes + layer)
    current = tl.load(input_ids + token, live_token, other=pad_token_id)
    head = tl.arange(0, BLOCK_H)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(max_ngram):
        from_batch = token - shift >= start
        previous = tl.load(
            input_ids + tl.maximum(token - shift, 0),
            live_token & from_batch,
            other=pad_token_id,
        )
        col = max_ngram - 1 + token - start - shift
        from_history = live_token & ~from_batch & (col >= 0) & (col < max_ngram - 1)
        previous_history = tl.load(
            lookback + req * (max_ngram - 1) + tl.maximum(col, 0),
            from_history,
            other=pad_token_id,
        )
        original = tl.where(from_batch, previous, previous_history)
        in_map = (original >= 0) & (original < size)
        mapped = tl.load(
            token_maps + layer * map_stride + tl.where(in_map, original, 0),
            live_token & in_map,
            other=0,
        ).to(tl.int64)
        source = tl.where(in_map, mapped, pad)
        multiplier = tl.load(multipliers + layer * max_ngram + shift)
        rolling ^= source * multiplier
        if shift > 0:
            column = (shift - 1) * num_heads + head
            total_heads = (max_ngram - 1) * num_heads
            prime = tl.load(primes + layer * total_heads + column, head < num_heads, 1)
            offset = tl.load(
                offsets + layer * total_heads + column, head < num_heads, 0
            )
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            live = live_token & (current != pad_token_id)
            hashed = tl.where(live[:, None], hashed, -1)
            target = (layer * num_tokens + token.to(tl.int64))[:, None] * total_heads
            tl.store(
                output + target + column[None, :],
                hashed,
                valid[:, None] & (head < num_heads),
            )


def hash_ngrams_torch(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    lookback: torch.Tensor,
    token_maps: torch.Tensor,
    map_sizes: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_token_id: int,
    num_actual_tokens: int,
) -> torch.Tensor:
    """CPU reference returning [layers, tokens, hash heads], with pad rows -1."""
    num_layers, max_ngram = multipliers.shape
    num_tokens = input_ids.numel()
    heads_per_order = primes.shape[1] // (max_ngram - 1)
    result = input_ids.new_empty(
        (num_layers, num_tokens, primes.shape[1]), dtype=torch.int32
    )
    if not num_tokens:
        return result
    rows = torch.arange(num_tokens, device=input_ids.device)
    requests = (
        torch.searchsorted(query_start_loc[1:], rows, right=True)
        .clamp_max(query_start_loc.numel() - 2)
        .long()
    )
    starts = query_start_loc[requests].long()
    for layer in range(num_layers):
        mapping = token_maps[layer]
        pad = mapping[pad_token_id].long()
        rolling = torch.zeros_like(rows, dtype=torch.int64)
        for shift in range(max_ngram):
            in_batch = rows - shift >= starts
            batch_ids = input_ids[(rows - shift).clamp_min(0)].long()
            col = max_ngram - 1 + rows - starts - shift
            history_ids = lookback[requests, col.clamp(0, max_ngram - 2)].long()
            ids = torch.where(in_batch, batch_ids, history_ids)
            mapped = mapping[ids.clamp(0, mapping.numel() - 1)].long()
            mapped = torch.where((ids >= 0) & (ids < map_sizes[layer]), mapped, pad)
            rolling = torch.bitwise_xor(rolling, mapped * multipliers[layer, shift])
            if shift:
                columns = slice((shift - 1) * heads_per_order, shift * heads_per_order)
                result[layer, :, columns] = (
                    torch.remainder(rolling[:, None], primes[layer, columns])
                    + offsets[layer, columns]
                ).int()
    padded = (rows >= num_actual_tokens) | (rows >= query_start_loc[-1])
    result[:, padded | (input_ids == pad_token_id), :] = -1
    return result


def hash_ngrams_triton(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    lookback: torch.Tensor,
    token_maps: torch.Tensor,
    map_sizes: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_token_id: int,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Launch the stateless hash for a packed batch on CUDA."""
    num_layers, max_ngram = multipliers.shape
    num_tokens = input_ids.numel()
    heads = primes.shape[1] // (max_ngram - 1)
    output = input_ids.new_empty(
        (num_layers, num_tokens, primes.shape[1]), dtype=torch.int32
    )
    if num_tokens:
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            query_start_loc,
            lookback,
            token_maps,
            map_sizes,
            multipliers,
            primes,
            offsets,
            output,
            num_tokens,
            num_actual_tokens,
            query_start_loc.numel() - 1,
            pad_token_id,
            map_stride=token_maps.shape[1],
            num_heads=heads,
            max_ngram=max_ngram,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(heads),
            num_warps=4,
        )
    return output
