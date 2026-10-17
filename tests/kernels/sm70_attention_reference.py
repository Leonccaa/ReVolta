# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention-V100 varlen layout diagnostics used by the SM70 kernel tests.

Copied verbatim from benchmarks/benchmark_sm70_attention_exactness.py at
69ff71dc82, before benchmarks/ was removed from ReVolta.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual - expected).abs()
    finite = diff[torch.isfinite(diff)]
    if finite.numel() == 0:
        return float("nan"), float("nan")
    return float(finite.max().item()), float(finite.mean().item())


def _mismatch_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    mismatch = torch.ne(actual, expected)
    num_different = int(mismatch.sum().item())
    first_mismatch = None
    if num_different:
        first_mismatch = int(torch.nonzero(mismatch.flatten(), as_tuple=False)[0])

    diff = (actual - expected).abs().flatten()
    if diff.numel() == 0:
        return {
            "num_different": num_different,
            "first_mismatch_flat_index": first_mismatch,
            "max_diff_flat_index": None,
            "actual_at_max_diff": None,
            "expected_at_max_diff": None,
        }
    max_idx = int(torch.argmax(diff).item())
    return {
        "num_different": num_different,
        "first_mismatch_flat_index": first_mismatch,
        "max_diff_flat_index": max_idx,
        "actual_at_max_diff": float(actual.flatten()[max_idx].item()),
        "expected_at_max_diff": float(expected.flatten()[max_idx].item()),
    }


def _make_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = k.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    key_cache = torch.zeros(
        (num_blocks, block_size, k.shape[2], k.shape[3]),
        device=k.device,
        dtype=k.dtype,
    )
    value_cache = torch.zeros_like(key_cache)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        key_cache[block_idx, : end - start] = k[0, start:end]
        value_cache[block_idx, : end - start] = v[0, start:end]
    block_table = torch.arange(num_blocks, device=k.device, dtype=torch.int32).view(
        1, num_blocks
    )
    seq_lens = torch.tensor([seq_len], device=k.device, dtype=torch.int32)
    return key_cache, value_cache, block_table, seq_lens


def _make_batched_paged_cache(
    k_values: list[torch.Tensor],
    v_values: list[torch.Tensor],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens_host = [int(k.shape[1]) for k in k_values]
    blocks_per_seq = [
        (seq_len + block_size - 1) // block_size for seq_len in seq_lens_host
    ]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)
    device = k_values[0].device
    dtype = k_values[0].dtype
    kv_heads = k_values[0].shape[2]
    head_dim = k_values[0].shape[3]
    key_cache = torch.zeros(
        (total_blocks, block_size, kv_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros(
        (len(k_values), max_blocks),
        device=device,
        dtype=torch.int32,
    )

    physical_block = 0
    for batch_idx, (k, v, num_blocks) in enumerate(
        zip(k_values, v_values, blocks_per_seq)
    ):
        for logical_block in range(num_blocks):
            start = logical_block * block_size
            end = min(start + block_size, k.shape[1])
            key_cache[physical_block, : end - start] = k[0, start:end]
            value_cache[physical_block, : end - start] = v[0, start:end]
            block_table[batch_idx, logical_block] = physical_block
            physical_block += 1

    seq_lens = torch.tensor(seq_lens_host, device=device, dtype=torch.int32)
    return key_cache, value_cache, block_table, seq_lens


def _triton_unified_attention_varlen(
    q_flat: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: list[int],
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    q_flat = q_flat.contiguous()
    out = torch.empty_like(q_flat)
    cu_seqlens = [0]
    for query_len in query_lens:
        cu_seqlens.append(cu_seqlens[-1] + query_len)
    query_start_loc = torch.tensor(
        cu_seqlens,
        device=q_flat.device,
        dtype=torch.int32,
    )
    unified_attention(
        q=q_flat,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max(query_lens),
        seqused_k=seq_lens,
        max_seqlen_k=int(seq_lens.max().item()),
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.NONE,
    )
    return out


def _flash_paged_loop_flat(
    flash_attn_v100: Any,
    q_values: list[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    chunks = []
    for idx, q in enumerate(q_values):
        out = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table[idx : idx + 1],
            seq_lens[idx : idx + 1],
            softmax_scale=softmax_scale,
            causal=causal,
        )
        chunks.append(out.squeeze(0))
    return torch.cat(chunks, dim=0)


def _flash_paged_standalone_loop_flat(
    flash_attn_v100: Any,
    q_values: list[torch.Tensor],
    k_values: list[torch.Tensor],
    v_values: list[torch.Tensor],
    block_size: int,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    chunks = []
    for q, k, v in zip(q_values, k_values, v_values):
        key_cache, value_cache, block_table, seq_lens = _make_paged_cache(
            k, v, block_size
        )
        out = flash_attn_v100.flash_attn_prefill_paged(
            q,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        chunks.append(out.squeeze(0))
    return torch.cat(chunks, dim=0)


def _flash_dense_loop_flat(
    flash_attn_v100: Any,
    q_values: list[torch.Tensor],
    k_values: list[torch.Tensor],
    v_values: list[torch.Tensor],
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    chunks = []
    for q, k, v in zip(q_values, k_values, v_values):
        out = flash_attn_v100.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        chunks.append(out.squeeze(0))
    return torch.cat(chunks, dim=0)


def _triton_loop_flat(
    q_values: list[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    chunks = []
    for idx, q in enumerate(q_values):
        out = _triton_unified_attention(
            q,
            key_cache,
            value_cache,
            block_table[idx : idx + 1],
            seq_lens[idx : idx + 1],
            softmax_scale,
            causal=causal,
        )
        chunks.append(out.squeeze(0))
    return torch.cat(chunks, dim=0)


def _make_qkv(
    *,
    seq_len: int,
    query_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.randn(
        (1, query_len, q_heads, head_dim),
        device=device,
        dtype=torch.float16,
    )
    k = torch.randn(
        (1, seq_len, kv_heads, head_dim),
        device=device,
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    return q, k, v


def _triton_unified_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool = True,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    q_flat = q.squeeze(0).contiguous()
    out = torch.empty_like(q_flat)
    query_len = q_flat.shape[0]
    query_start_loc = torch.tensor(
        [0, query_len],
        device=q.device,
        dtype=torch.int32,
    )
    unified_attention(
        q=q_flat,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=query_len,
        seqused_k=seq_lens,
        max_seqlen_k=int(seq_lens.max().item()),
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.NONE,
    )
    return out.unsqueeze(0)


def _compare_payload(
    name: str,
    reference: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    max_diff, mean_diff = _diff_stats(actual, expected)
    mismatch = _mismatch_stats(actual, expected)
    return {
        "name": name,
        "reference": reference,
        "equal": bool(torch.equal(actual, expected)),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "num_different": mismatch["num_different"],
        "first_mismatch_flat_index": mismatch["first_mismatch_flat_index"],
        "max_diff_flat_index": mismatch["max_diff_flat_index"],
        "actual_at_max_diff": mismatch["actual_at_max_diff"],
        "expected_at_max_diff": mismatch["expected_at_max_diff"],
    }


def run_varlen_layout_diagnostic_cases(
    flash_attn_v100: Any,
) -> list[dict[str, Any]]:
    """Diagnose whether varlen diffs come from layout/indexing or arithmetic.

    This is intentionally separate from run_varlen_edge_cases. It compares
    same-kernel layout changes first, so a nonzero diff here is a Type-A bug.
    """
    q_heads = 6
    kv_heads = 1
    head_dim = 256
    block_size = 16
    softmax_scale = 1.0 / math.sqrt(head_dim)
    payloads: list[dict[str, Any]] = []

    cases = [
        ("uniform", [(69, 8), (181, 8), (4096, 8)], 7000),
        ("ragged", [(69, 1), (181, 3), (4096, 8)], 9000),
    ]
    for case_name, specs, seed_base in cases:
        q_values = []
        k_values = []
        v_values = []
        for idx, (seq_len, query_len) in enumerate(specs):
            q, k, v = _make_qkv(
                seq_len=seq_len,
                query_len=query_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                seed=seed_base + idx,
            )
            q_values.append(q)
            k_values.append(k)
            v_values.append(v)

        key_cache, value_cache, block_table, seq_lens = _make_batched_paged_cache(
            k_values, v_values, block_size
        )
        query_lens = [query_len for _, query_len in specs]
        q_flat = torch.cat([q.squeeze(0) for q in q_values], dim=0)
        triton_varlen = _triton_unified_attention_varlen(
            q_flat,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            query_lens,
            softmax_scale,
            causal=True,
        )
        flash_full_cache_loop = _flash_paged_loop_flat(
            flash_attn_v100,
            q_values,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale,
            causal=True,
        )
        flash_standalone_loop = _flash_paged_standalone_loop_flat(
            flash_attn_v100,
            q_values,
            k_values,
            v_values,
            block_size,
            softmax_scale,
            causal=True,
        )
        flash_dense_loop = _flash_dense_loop_flat(
            flash_attn_v100,
            q_values,
            k_values,
            v_values,
            softmax_scale,
            causal=True,
        )
        triton_loop = _triton_loop_flat(
            q_values,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale,
            causal=True,
        )
        comparisons = [
            _compare_payload(
                "flash_v100_full_cache_looped_paged_prefill",
                "flash_v100_standalone_looped_paged_prefill",
                flash_full_cache_loop,
                flash_standalone_loop,
            ),
            _compare_payload(
                "flash_v100_full_cache_looped_paged_prefill",
                "flash_v100_dense_looped_prefill",
                flash_full_cache_loop,
                flash_dense_loop,
            ),
            _compare_payload(
                "triton_unified_attention_varlen",
                "triton_unified_attention_looped",
                triton_varlen,
                triton_loop,
            ),
            _compare_payload(
                "flash_v100_full_cache_looped_paged_prefill",
                "triton_unified_attention_varlen",
                flash_full_cache_loop,
                triton_varlen,
            ),
        ]
        if case_name == "uniform":
            q_batch = torch.cat(q_values, dim=0)
            flash_batched = flash_attn_v100.flash_attn_prefill_paged(
                q_batch,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                softmax_scale=softmax_scale,
                causal=True,
            ).reshape(-1, q_heads, head_dim)
            comparisons.insert(
                0,
                _compare_payload(
                    "flash_v100_batched_paged_prefill",
                    "flash_v100_full_cache_looped_paged_prefill",
                    flash_batched,
                    flash_full_cache_loop,
                ),
            )
            comparisons.append(
                _compare_payload(
                    "flash_v100_batched_paged_prefill",
                    "triton_unified_attention_varlen",
                    flash_batched,
                    triton_varlen,
                )
            )

        type_a_must_equal = [
            "flash_v100_full_cache_looped_paged_prefill vs "
            "flash_v100_standalone_looped_paged_prefill",
            "triton_unified_attention_varlen vs triton_unified_attention_looped",
        ]
        if case_name == "uniform":
            type_a_must_equal.insert(
                0,
                "flash_v100_batched_paged_prefill vs "
                "flash_v100_full_cache_looped_paged_prefill",
            )
        payloads.append(
            {
                "name": f"varlen_{case_name}_layout_diagnostic",
                "seq_lens": [seq for seq, _ in specs],
                "query_lens": query_lens,
                "block_size": block_size,
                "type_a_must_equal": type_a_must_equal,
                "comparisons": comparisons,
            }
        )
    return payloads
