# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit late-scale planar INT8 dense fallback for Flash-V100 draft KV."""

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa_int8 import (
    require_int8_abi,
    triton_attention,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import INT8


@triton.jit
def _dense_rows(starts, lengths, requests, positions, rows, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    lane = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(starts + request)
    end = tl.load(starts + request + 1)
    length = tl.load(lengths + request)
    valid = (start + lane < end) & (start + lane < rows)
    tl.store(requests + start + lane, request, valid)
    tl.store(positions + start + lane, length - (end - start) + lane, valid)


def forward_int8_dense(
    impl, query, kv_cache, metadata, output, output_scale=None, output_block_scale=None
):
    from vllm.logger import init_logger
    from vllm.v1.attention.backend import AttentionType

    require_int8_abi()
    if (
        impl.attn_type != AttentionType.DECODER
        or impl.alibi_slopes is not None
        or impl.logits_soft_cap
        or impl.sinks is not None
        or impl.sliding_window != (-1, -1)
        or output_scale is not None
        or output_block_scale is not None
    ):
        raise NotImplementedError("Flash-V100 planar INT8 dense feature is unsupported")
    init_logger(__name__).info_once(
        "FLASH_ATTN_V100 INT8 dense/draft uses the explicit Triton late-scale route."
    )
    if output is None:
        output = torch.empty_like(query)
    if metadata is None:
        return output.zero_()
    rows = metadata.num_actual_tokens
    output.zero_()
    if not rows:
        return output
    view = KVCacheView.bind(kv_cache, INT8, impl.head_size)
    requests = torch.full((rows,), -1, dtype=torch.int32, device=query.device)
    positions = torch.full((rows,), -1, dtype=torch.int64, device=query.device)
    _dense_rows[
        (metadata.query_start_loc.numel() - 1, triton.cdiv(metadata.max_query_len, 256))
    ](
        metadata.query_start_loc,
        metadata.seq_lens,
        requests,
        positions,
        rows,
        BLOCK=256,
    )
    # The dense specialization does not dereference this placeholder.
    dummy_indices = requests[:, None]
    # Bound FP32 partials during long draft prefill; no full-cache Half bridge.
    splits = min(32, max(1, triton.cdiv(metadata.max_seq_len, 32)))
    rows_per_chunk = max(
        1, (64 * 1024**2) // (splits * query.shape[1] * query.shape[2] * 4)
    )
    for start in range(0, rows, rows_per_chunk):
        end = min(rows, start + rows_per_chunk)
        triton_attention(
            query[start:end],
            view.key,
            view.value,
            dummy_indices[start:end],
            metadata.block_table,
            requests[start:end],
            output[start:end],
            view.key_scale,
            view.value_scale,
            dense=True,
            positions=positions[start:end],
            seq_lens=metadata.seq_lens,
            max_seq_len=metadata.max_seq_len,
            causal=metadata.causal,
            softmax_scale=impl.scale,
        )
    return output


def write_int8_dense(impl, layer, key, value, kv_cache, slots):
    from vllm import _custom_ops as ops

    require_int8_abi()
    view = KVCacheView.bind(kv_cache, INT8, impl.head_size)
    ops.reshape_and_cache_flash(
        key,
        value,
        view.key,
        view.value,
        slots,
        "int8",
        layer._k_scale,
        layer._v_scale,
        **view.scale_kwargs,
    )
