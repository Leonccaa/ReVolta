# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Undispatched request-segmented QSA candidate, pending SM70 qualification.

Retain the paged scorer's 64x128 by 128x16 dot for each row. Keys are
loaded once per segment/tile and reused across those row dots. This candidate
is not selected by serving code. Wider query dots are deferred to preserve
the original arithmetic shape.
"""

import math
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _minimum(a, b):
    return tl.minimum(a, b)


@triton.jit
def _build_segments(
    requests,
    positions,
    lengths,
    segments,
    visible,
    stride_request,
    stride_position,
    stride_length,
    num_rows,
    num_requests,
    compress,
    MAX_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.arange(0, BLOCK_M)
    req = tl.load(requests + row * stride_request, row < num_rows, other=-1)
    prev = tl.load(
        requests + (row - 1) * stride_request, (row > 0) & (row < num_rows), other=-1
    )
    nxt = tl.load(requests + (row + 1) * stride_request, row + 1 < num_rows, other=-1)
    start = tl.associative_scan(
        tl.where((row == 0) | (req != prev), row, 0), 0, _maximum
    )
    end = tl.associative_scan(
        tl.where((row + 1 >= num_rows) | (req != nxt), row + 1, num_rows),
        0,
        _minimum,
        reverse=True,
    )
    leader = ((row - start) % MAX_ROWS == 0) & (row < num_rows)
    tl.store(segments + row * 3, tl.where(leader, req, -1), row < num_rows)
    tl.store(segments + row * 3 + 1, tl.where(leader, row, -1), row < num_rows)
    tl.store(
        segments + row * 3 + 2,
        tl.where(leader, tl.minimum(row + MAX_ROWS, end), -1),
        row < num_rows,
    )
    valid = (req >= 0) & (req < num_requests) & (row < num_rows)
    safe = tl.minimum(tl.maximum(req, 0), num_requests - 1)
    length = tl.load(lengths + safe * stride_length, valid, other=0)
    pos = tl.load(positions + row * stride_position, row < num_rows, other=-1)
    # Match paged Triton's signed division, including padded negative positions.
    vis = tl.minimum((pos + 1) // compress, length // compress)
    tl.store(visible + row, vis, row < num_rows)


@triton.jit
def _segment_scores(
    q,
    cache,
    table,
    segments,
    visible,
    output,
    sq0,
    sq1,
    sq2,
    sk0,
    sk1,
    sk3,
    st0,
    st1,
    num_columns,
    num_pages,
    num_requests,
    divisor,
    PAGE_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    slot = tl.program_id(0)
    req = tl.load(segments + slot * 3)
    begin = tl.load(segments + slot * 3 + 1)
    end = tl.load(segments + slot * 3 + 2)
    if begin < 0:
        return
    rows = begin + tl.arange(0, MAX_ROWS)
    limits = tl.load(visible + rows, rows < end, other=0)
    columns = tl.program_id(1) * 64 + tl.arange(0, 64)
    live = columns < tl.max(limits, 0)
    if tl.program_id(1) * 64 >= tl.max(limits, 0):
        return
    safe_req = tl.minimum(tl.maximum(req, 0), num_requests - 1)
    page = tl.load(
        table
        + safe_req * st0
        + tl.minimum(columns // PAGE_SIZE, TABLE_WIDTH - 1) * st1,
        live & (columns < num_columns),
        other=-1,
    )
    valid = live & (page >= 0) & (page < num_pages)
    dims = tl.arange(0, 128)
    # Keep the same key tile, FP16 input, FP32 dot and divisor as paged QSA.
    # One source load per segment/tile; physical traffic still needs measurement.
    keys = tl.load(
        cache
        + tl.maximum(page, 0).to(tl.int64)[:, None] * sk0
        + (columns % PAGE_SIZE)[:, None] * sk1
        + dims[None, :] * sk3,
        valid[:, None] & (columns[:, None] < num_columns),
        other=0.0,
        eviction_policy="evict_first",
    )
    heads = tl.arange(0, 16)
    # Keep one dot body. Hoisting its expanded dot operands across the row
    # loop creates a large live register set on SM70; retain conversions inside.
    for row in tl.range(begin, end, disable_licm=True):
        limit = tl.load(visible + row)
        query = tl.load(
            q + row * sq0 + heads[None, :] * sq1 + dims[:, None] * sq2,
            heads[None, :] < 4,
            other=0.0,
        )
        raw = tl.dot(keys, query, out_dtype=tl.float32)
        raw = tl.where(heads[None, :] < 4, tl.maximum(raw, 0.0), 0.0)
        score = tl.sum(raw, 1) / divisor
        tl.store(
            output + row * num_columns + columns,
            tl.where(valid, score, -float("inf")),
            (columns < num_columns) & (columns < limit),
        )


@dataclass
class SegmentWorkspace:
    """Fixed M slots: unused slots launch and exit, without a host count readback."""

    descriptors: torch.Tensor
    visible: torch.Tensor
    scores: torch.Tensor

    @classmethod
    def allocate(cls, q: torch.Tensor, columns: int) -> "SegmentWorkspace":
        return cls(
            torch.empty((q.shape[0], 3), dtype=torch.int32, device=q.device),
            torch.empty(q.shape[0], dtype=torch.int32, device=q.device),
            torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device),
        )


def qsa_mqa_request_segments(
    q: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    compress: int,
    workspace: SegmentWorkspace,
    *,
    max_rows: int = 8,
    score_divisor: float = math.sqrt(128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild descriptors and score in caller-owned buffers; never auto-dispatched.

    Finite FP16 Q/K, H4/D128, positive strides and integer metadata are the
    research input contract. CPU tensors require Triton's interpreter. No input
    data or dynamic segment count is copied to the host. Allocate before capture.
    GPU bit patterns, graph replay, memory safety and speed remain unqualified.
    """
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        raise ValueError("requires Q[rows,4,128]")
    if cache.ndim != 4 or cache.shape[2:] != (1, 128):
        raise ValueError("requires cache[pages,page_size,1,128]")
    if q.dtype != torch.float16 or cache.dtype != torch.float16:
        raise ValueError("requires FP16 queries and cache")
    if table.ndim != 2 or lengths.shape != (table.shape[0],):
        raise ValueError("requires one sequence length per request")
    if any(x.shape != (q.shape[0],) for x in (requests, positions)):
        raise ValueError("row metadata shape mismatch")
    if any(x.dtype != torch.int32 for x in (table, requests, lengths)):
        raise ValueError("request metadata must be int32")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be int32 or int64")
    if max_rows not in (1, 2, 4, 8) or q.shape[0] > 256:
        raise ValueError("requires max_rows 1/2/4/8 and at most 256 rows")
    if compress <= 0 or not math.isfinite(score_divisor) or score_divisor <= 0:
        raise ValueError("compression and divisor must be positive")
    m = q.shape[0]
    columns = workspace.scores.shape[1] if workspace.scores.ndim == 2 else -1
    buffers = (workspace.descriptors, workspace.visible, workspace.scores)
    if (
        workspace.descriptors.shape != (m, 3)
        or workspace.visible.shape != (m,)
        or workspace.scores.shape != (m, columns)
        or workspace.descriptors.dtype != torch.int32
        or workspace.visible.dtype != torch.int32
        or workspace.scores.dtype != torch.float32
        or not all(x.is_contiguous() for x in buffers)
    ):
        raise ValueError("invalid workspace")
    inputs = (q, cache, table, requests, positions, lengths)
    if not all(x.device == q.device for x in (*inputs, *buffers)):
        raise ValueError("all inputs and workspace must use one device")
    if any(s <= 0 for x in inputs for s in x.stride()):
        raise ValueError("input strides must be positive")
    if columns < 0 or columns > table.shape[1] * cache.shape[1]:
        raise ValueError("score width must fit the page-table capacity")
    if m and not (cache.shape[0] and cache.shape[1] and all(table.shape)):
        raise ValueError("nonempty queries require paged storage")
    if m:
        _build_segments[(1,)](
            requests,
            positions,
            lengths,
            workspace.descriptors,
            workspace.visible,
            requests.stride(0),
            positions.stride(0),
            lengths.stride(0),
            m,
            table.shape[0],
            compress,
            MAX_ROWS=max_rows,
            BLOCK_M=triton.next_power_of_2(m),
            num_warps=4,
        )
        if columns:
            _segment_scores[(m, triton.cdiv(columns, 64))](
                q,
                cache,
                table,
                workspace.descriptors,
                workspace.visible,
                workspace.scores,
                *q.stride(),
                cache.stride(0),
                cache.stride(1),
                cache.stride(3),
                *table.stride(),
                columns,
                cache.shape[0],
                table.shape[0],
                score_divisor,
                PAGE_SIZE=cache.shape[1],
                TABLE_WIDTH=table.shape[1],
                MAX_ROWS=max_rows,
                num_warps=2,
            )
    return workspace.scores, workspace.visible
