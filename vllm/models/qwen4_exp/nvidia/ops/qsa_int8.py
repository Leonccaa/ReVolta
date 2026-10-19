# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planar INT8 readers. Legacy QSA kernels remain in qsa.py unchanged."""

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.kv_cache_view import KVPageMapping


@triton.jit(do_not_specialize=["num_requests"])
def _int8_sparse_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    final_lse_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    k_scale,
    v_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_E4M3: tl.constexpr,
    RESOLVED_INDICES: tl.constexpr = False,
    K_SCALES=None,
    V_SCALES=None,
    KS_BLOCK: tl.constexpr = 0,
    KS_TOKEN: tl.constexpr = 0,
    KS_HEAD: tl.constexpr = 0,
    VS_BLOCK: tl.constexpr = 0,
    VS_TOKEN: tl.constexpr = 0,
    VS_HEAD: tl.constexpr = 0,
    DENSE: tl.constexpr = False,
    POSITIONS=None,
    SEQLENS=None,
    CAUSAL: tl.constexpr = True,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )
    max_value = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = HEAD_DIM ** (-0.5) * 1.4426950408889634
    qk_scale_log2 = softmax_scale_log2 * k_scale
    if DENSE:
        live_length = tl.load(SEQLENS + safe_request)
        if CAUSAL:
            live_length = tl.minimum(live_length, tl.load(POSITIONS + row) + 1)
        live_tiles = tl.cdiv(tl.maximum(live_length, 0), BLOCK_N)
        split_tile_start = split_id * live_tiles // NUM_SPLITS
        split_tile_end = (split_id + 1) * live_tiles // NUM_SPLITS
    else:
        split_tile_start = split_id * NUM_TILES // NUM_SPLITS
        split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        if DENSE:
            seq_len = tl.load(SEQLENS + safe_request)
            limit = tl.load(POSITIONS + row) + 1 if CAUSAL else seq_len
            logical_token = tl.where(columns < tl.minimum(limit, seq_len), columns, -1)
        else:
            logical_token = tl.load(
                indices_ptr + row * stride_indices_row + columns,
                mask=columns < TOPK,
                other=-1,
            )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (request >= 0) & (request < num_requests) & (logical_token >= 0)
        if RESOLVED_INDICES:
            physical_page = logical_page
        else:
            valid &= logical_page < PAGE_TABLE_WIDTH
            physical_page = tl.load(
                block_table_ptr
                + safe_request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=valid,
                other=-1,
            )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        keys = keys.to(query.dtype)
        values = values.to(query.dtype)
        key_scales = tl.load(
            K_SCALES
            + safe_page * KS_BLOCK
            + page_offset * KS_TOKEN
            + kv_head * KS_HEAD,
            mask=valid,
            other=0.0,
        )
        value_scales = tl.load(
            V_SCALES
            + safe_page * VS_BLOCK
            + page_offset * VS_TOKEN
            + kv_head * VS_HEAD,
            mask=valid,
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores *= key_scales[None, :]
        scores *= qk_scale_log2
        scores = tl.where(valid[None, :], scores, -1e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        weighted = tl.where(valid[None, :], probabilities * value_scales[None, :], 0.0)
        high = weighted.to(values.dtype)
        low = ((weighted - high.to(tl.float32)) * 2048.0).to(values.dtype)
        pv = tl.dot(low, values) * (1.0 / 2048.0)
        pv = tl.dot(high, values, acc=pv)
        accumulator = accumulator * alpha[:, None] + pv
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max
    has_values = (normalizer > 0) | (normalizer != normalizer)
    normalized_output = tl.where(
        has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1e-20), 0.0
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        if final_lse_ptr is not None:
            lse = tl.where(
                has_values,
                max_value + tl.math.log2(tl.maximum(normalizer, 1e-20)),
                -float("inf"),
            )
            tl.store(
                final_lse_ptr + row * NUM_QUERY_HEADS + first_head + head_offsets,
                lse,
                mask=head_offsets < GROUP_SIZE,
            )
        if output_gate_ptr is not None:
            normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
            output_gate = tl.load(
                output_gate_ptr
                + row * stride_output_gate_row
                + (first_head + head_offsets[:, None]) * stride_output_gate_head
                + dim_offsets[None, :],
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
            normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(
                output_gate
            )
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _int8_merge_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    final_lse_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    v_scale,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    KV_E4M3: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = ((lse_max > -float("inf")) | (lse_max != lse_max)).to(tl.int1)
    shifted = tl.where(
        split_mask, tl.where(has_values, lse - lse_max, -float("inf")), -float("inf")
    )
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    if final_lse_ptr is not None:
        merged_lse = tl.where(
            has_values,
            lse_max + tl.math.log2(tl.maximum(denominator, 1.0e-20)),
            -float("inf"),
        )
        tl.store(final_lse_ptr + row * NUM_QUERY_HEADS + head, merged_lse)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(
        (denominator > 0) | (denominator != denominator), merged / denominator, 0.0
    )
    if KV_E4M3:
        # Apply the V scale once after combining all independently normalized
        # splits. Scaling partials earlier would repeat this work per split.
        merged *= v_scale
    if output_gate_ptr is not None:
        merged = merged.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + head * stride_output_gate_head
            + dim_offsets
        ).to(tl.float32)
        merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def require_int8_abi():
    """Reject an incomplete installation before admitting a serving backend."""
    from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda as native

    version = getattr(native, "int8_page4_abi_version", None)
    if not callable(version) or version() != 1:
        raise RuntimeError(
            "INT8 KV requires Flash-V100 page4 ABI v1; rebuild the wheel"
        )
    for name in (
        "decode_int8_page4_fwd",
        "grouped_sparse_int8_page4_fwd",
        "grouped_sparse_page4_plan_fwd",
    ):
        if not callable(getattr(native, name, None)):
            raise RuntimeError(f"INT8 KV native reader missing: {name}")
    if not hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash_int8"):
        raise RuntimeError("INT8 KV native writer ABI missing; rebuild the vLLM wheel")
    return native


def validate_planes(q, k, v, sk, sv):
    if k.dtype != torch.int8 or v.dtype != torch.int8 or k.shape != v.shape:
        raise ValueError("INT8 reader requires signed int8 K/V payloads")
    if q.ndim != 3 or k.ndim != 4 or q.shape[2] != k.shape[3]:
        raise ValueError("INT8 reader Q/K/V shape mismatch")
    if q.shape[1] % k.shape[2] or q.dtype != torch.float16:
        raise ValueError("INT8 reader requires FP16 grouped-query heads")
    for plane in (sk, sv):
        if (
            plane is None
            or plane.dtype != torch.float32
            or plane.shape != k.shape[:3]
            or plane.device != k.device
        ):
            raise ValueError(
                "INT8 reader requires explicit FP32 token/head scale planes"
            )
    if q.device != k.device or k.device != v.device:
        raise ValueError("INT8 reader device mismatch")
    return KVPageMapping.bind_payload(k, v, 4)


def triton_attention(
    q,
    k,
    v,
    indices,
    table,
    requests,
    out,
    sk,
    sv,
    lse=None,
    gate=None,
    *,
    dense=False,
    positions=None,
    seq_lens=None,
    max_seq_len=0,
    causal=True,
    softmax_scale=None,
):
    """FP32 late-scale reference/fallback; never expands the KV cache to Half."""
    validate_planes(q, k, v, sk, sv)
    rows, heads, dim = q.shape
    width = max_seq_len if dense else indices.shape[1]
    tiles = triton.cdiv(width, 32)
    splits = min(32, max(1, tiles))
    partial = torch.empty((splits, *q.shape), device=q.device, dtype=torch.float32)
    partial_lse = torch.empty(
        (splits, rows, heads), device=q.device, dtype=torch.float32
    )
    group = heads // k.shape[2]
    # The legacy kernel's scale is fixed by D; adjust it explicitly for dense users.
    scale_ratio = (softmax_scale * dim**0.5) if softmax_scale is not None else 1.0
    _int8_sparse_kernel[(rows, k.shape[2], splits)](
        q,
        k,
        v,
        indices,
        table,
        requests,
        partial,
        partial_lse,
        out,
        lse,
        gate,
        q.stride(0),
        q.stride(1),
        *k.stride()[:3],
        *v.stride()[:3],
        indices.stride(0),
        table.stride(0),
        out.stride(0),
        out.stride(1),
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        rows,
        k.shape[0],
        table.shape[0],
        scale_ratio,
        1.0,
        TOPK=width,
        PAGE_SIZE=k.shape[1],
        PAGE_TABLE_WIDTH=table.shape[1],
        GROUP_SIZE=group,
        HEAD_DIM=dim,
        NUM_QUERY_HEADS=heads,
        NUM_SPLITS=splits,
        NUM_TILES=tiles,
        BLOCK_M=max(16, triton.next_power_of_2(group)),
        BLOCK_N=32,
        KV_E4M3=False,
        K_SCALES=sk,
        V_SCALES=sv,
        KS_BLOCK=sk.stride(0),
        KS_TOKEN=sk.stride(1),
        KS_HEAD=sk.stride(2),
        VS_BLOCK=sv.stride(0),
        VS_TOKEN=sv.stride(1),
        VS_HEAD=sv.stride(2),
        DENSE=dense,
        POSITIONS=positions,
        SEQLENS=seq_lens,
        CAUSAL=causal,
        num_warps=4,
    )
    if splits == 1:
        return out
    _int8_merge_kernel[(rows, heads)](
        partial,
        partial_lse,
        out,
        lse,
        gate,
        out.stride(0),
        out.stride(1),
        gate.stride(0) if gate is not None else 0,
        gate.stride(1) if gate is not None else 0,
        rows,
        1.0,
        HEAD_DIM=dim,
        NUM_QUERY_HEADS=heads,
        NUM_SPLITS=splits,
        BLOCK_SPLITS=triton.next_power_of_2(splits),
        KV_E4M3=False,
        num_warps=4,
    )
    return out


def native_decode(q, k, v, indices, table, requests, positions, seq_lens, out, sk, sv):
    from .qsa import _qsa_xqa_page4_block_table, _qsa_xqa_page4_workspace

    native = require_int8_abi()
    mapping = validate_planes(q, k, v, sk, sv)
    pages, lengths = _qsa_xqa_page4_block_table(
        indices,
        table,
        requests,
        positions,
        seq_lens,
        k.shape[0],
        mapping.owner_tokens,
        mapping.owner_page_span,
    )
    parts = triton.cdiv(2051, 256)
    partial, maxima, sums, active = _qsa_xqa_page4_workspace(q, parts, "fp8_e4m3")
    native.decode_int8_page4_fwd(
        q.contiguous(),
        k,
        v,
        out,
        pages,
        lengths,
        partial,
        maxima,
        sums,
        active,
        sk,
        sv,
        mapping.owner_page_span,
        q.shape[2] ** -0.5,
    )
    return out


def native_grouped(
    q, k, v, indices, table, requests, positions, seq_lens, out, sk, sv, lse=None
):
    from .qsa import _qsa_grouped_page4_workspace

    native = require_int8_abi()
    mapping = validate_planes(q, k, v, sk, sv)
    rows = q.shape[0]
    padded = triton.cdiv(rows, 8) * 8
    if padded != rows:
        qp = torch.zeros((padded, *q.shape[1:]), device=q.device, dtype=q.dtype)
        qp[:rows].copy_(q)
        ip = torch.full((padded, 2051), -1, device=q.device, dtype=torch.int32)
        ip[:rows].copy_(indices)
        rp = torch.full((padded,), -1, device=q.device, dtype=torch.int32)
        rp[:rows].copy_(requests)
        pp = torch.full((padded,), -1, device=q.device, dtype=torch.int64)
        pp[:rows].copy_(positions)
    else:
        qp, ip, rp, pp = q, indices, requests, positions
    pages, masks, lengths, slice_lse = _qsa_grouped_page4_workspace(qp[:, :6])
    native.grouped_sparse_page4_plan_fwd(
        ip,
        table,
        rp,
        pp,
        seq_lens,
        pages,
        masks,
        lengths,
        mapping.owner_tokens,
        mapping.owner_page_span,
        k.shape[0],
    )
    slice_out = torch.empty((padded, 6, q.shape[2]), device=q.device, dtype=out.dtype)
    for head in range(0, q.shape[1], 6):
        native.grouped_sparse_int8_page4_fwd(
            qp[:, head : head + 6].contiguous(),
            k,
            v,
            slice_out,
            pages,
            masks,
            lengths,
            slice_lse,
            sk,
            sv,
            mapping.owner_page_span,
            q.shape[2] ** -0.5,
        )
        out[:, head : head + 6].copy_(slice_out[:rows])
        if lse is not None:
            sl = slice_lse[:rows]
            lse[:, head : head + 6].copy_(
                (sl * math.log2(math.e)).masked_fill(sl <= -1e29, -math.inf)
            )
    return out


def sparse_attention(
    q, k, v, indices, table, requests, out, gate, positions, seq_lens, sk, sv, lse=None
):
    from .qsa import _qsa_output_gate

    require_int8_abi()
    validate_planes(q, k, v, sk, sv)
    if out is None:
        out = torch.empty_like(q)
    if q.shape[0] == 0:
        return out
    native_shape = (
        q.shape[1] % 6 == 0
        and q.shape[2] == 256
        and indices.shape[1] == 2051
        and k.shape[1:] == (16, 1, 256)
        and positions is not None
        and seq_lens is not None
    )
    if native_shape and lse is None:
        if q.shape[0] < 8 and q.shape[1] == 6:
            native_decode(
                q, k, v, indices, table, requests, positions, seq_lens, out, sk, sv
            )
        else:
            native_grouped(
                q, k, v, indices, table, requests, positions, seq_lens, out, sk, sv
            )
        if gate is not None:
            _qsa_output_gate(out, gate.view_as(q))
        return out
    return triton_attention(
        q,
        k,
        v,
        indices,
        table,
        requests,
        out,
        sk,
        sv,
        lse,
        gate.view_as(q) if gate is not None else None,
    )


def dcp_grouped(q, k, v, indices, table, requests, out, lse, sk, sv):
    """INT8 DCP uses the same owner mapping on each rank, with separate scales."""
    require_int8_abi()
    if q.shape[0] < 8 or q.shape[1] % 6 or indices.shape[1] != 2051:
        return 0
    capacity = table.shape[1] * k.shape[1]
    positions = torch.full(
        (q.shape[0],), capacity - 1, device=q.device, dtype=torch.int64
    )
    lengths = torch.full(
        (table.shape[0],), capacity, device=q.device, dtype=torch.int32
    )
    native_grouped(
        q, k, v, indices, table, requests, positions, lengths, out, sk, sv, lse
    )
    return q.shape[0]
