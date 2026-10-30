# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-segmented paged QSA scorer for multi-request SM70 decode batches.

The paged scorer (`qsa.qsa_mqa_paged`) loads every 64-key tile once per query row. In a
decode batch the rows of one request share their page table, so this scorer loads each
tile once per segment of up to four rows of one request and multiplies it by those rows
together: four rows x four heads form one 128x16 query block, the same 64x128 by 128x16
dot shape, the same 2 warps and the same software-pipelined tile loop as the paged
kernel, and each row's score is reduced over the same 16 lanes as there, with the other
rows' lanes zeroed. On SM70 this gives bitwise the same scores and visible lengths as
the paged scorer (measured over the decode shapes of DFlash2 and MTP4 at 4 and 8
requests and 32K to 256K tokens, eager and through CUDA graphs) while reading the keys a
quarter as often. The query blocks are loaded once before the tile loop; loading them
inside the loop, a data-dependent branch in the loop body, 8-row segments or more warps
either spill the key tile at 2 warps or change the reduction order (not bitwise).

Segments are built on the device into a fixed [M, 3] descriptor buffer (no host
readback), so the call can be captured in a CUDA graph; unused slots launch and exit.
The runtime keeps the paged scorer for single-request batches (shared-key and cuBLAS
paths) and for batches with fewer than two rows per request, where one-row segments
gain nothing.
"""

import math
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

# 64-key tiles per program. More than one keeps the pipelined loop (one constant
# iteration spills); 4 also cuts the programs that start past a request's visible
# length.
_TILES_PER_PROG = 4
# Rows per segment: four rows fill the 16 dot lanes; eight (two query blocks) spill on
# SM70.
SEGMENT_ROWS = (1, 2, 4)


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
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
):
    slot = tl.program_id(0)
    req = tl.load(segments + slot * 3)
    begin = tl.load(segments + slot * 3 + 1)
    end = tl.load(segments + slot * 3 + 2)
    if begin < 0:
        return
    rows = begin + tl.arange(0, MAX_ROWS)
    limits = tl.load(visible + rows, rows < end, other=0)
    seg_visible = tl.max(limits, 0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    if tile_start * 64 >= seg_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(seg_visible, 64))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, 64))
    safe_req = tl.minimum(tl.maximum(req, 0), num_requests - 1)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    lane_row = lanes // 4
    lane_head = lanes % 4
    column_offsets = tl.arange(0, 64)
    # Loop-invariant query blocks, loaded before the pipelined tile loop (see the module
    # docstring).
    grow0 = begin + lane_row
    query0 = tl.load(
        q + grow0[None, :] * sq0 + lane_head[None, :] * sq1 + dims[:, None] * sq2,
        ((lane_row < MAX_ROWS) & (grow0 < end))[None, :],
        other=0.0,
    )
    if MAX_ROWS > 4:
        grow1 = begin + 4 + lane_row
        query1 = tl.load(
            q + grow1[None, :] * sq0 + lane_head[None, :] * sq1 + dims[:, None] * sq2,
            ((4 + lane_row < MAX_ROWS) & (grow1 < end))[None, :],
            other=0.0,
        )
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * 64 + column_offsets
        live = columns < seg_visible
        page = tl.load(
            table
            + safe_req * st0
            + tl.minimum(columns // PAGE_SIZE, TABLE_WIDTH - 1) * st1,
            live & (columns < num_columns),
            other=-1,
        )
        valid = live & (page >= 0) & (page < num_pages)
        keys = tl.load(
            cache
            + tl.maximum(page, 0).to(tl.int64)[:, None] * sk0
            + (columns % PAGE_SIZE)[:, None] * sk1
            + dims[None, :] * sk3,
            valid[:, None] & (columns[:, None] < num_columns),
            other=0.0,
            eviction_policy="evict_first",
        )
        raw0 = tl.maximum(tl.dot(keys, query0, out_dtype=tl.float32), 0.0)
        for j in tl.static_range(4):
            if j < MAX_ROWS:
                row = begin + j
                score = (
                    tl.sum(tl.where((lane_row == j)[None, :], raw0, 0.0), 1) / divisor
                )
                limit = tl.load(visible + row, row < end, other=0)
                tl.store(
                    output + row * num_columns + columns,
                    tl.where(valid, score, -float("inf")),
                    (row < end) & (columns < num_columns) & (columns < limit),
                )
        if MAX_ROWS > 4:
            raw1 = tl.maximum(tl.dot(keys, query1, out_dtype=tl.float32), 0.0)
            for j in tl.static_range(4):
                if 4 + j < MAX_ROWS:
                    row = begin + 4 + j
                    score = (
                        tl.sum(tl.where((lane_row == j)[None, :], raw1, 0.0), 1)
                        / divisor
                    )
                    limit = tl.load(visible + row, row < end, other=0)
                    tl.store(
                        output + row * num_columns + columns,
                        tl.where(valid, score, -float("inf")),
                        (row < end) & (columns < num_columns) & (columns < limit),
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
    max_rows: int = 4,
    score_divisor: float = math.sqrt(128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the segment descriptors and score into caller-owned buffers.

    Finite FP16 Q/K, H4/D128, positive strides and integer metadata. CPU tensors
    need Triton's interpreter. No input data or segment count is copied to the
    host, so the call can be captured in a CUDA graph.
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
    if max_rows not in SEGMENT_ROWS or q.shape[0] > 256:
        raise ValueError("requires max_rows 1/2/4 and at most 256 rows")
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
            _segment_scores[(m, triton.cdiv(columns, 64 * _TILES_PER_PROG))](
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
                TILES_PER_PROG=_TILES_PER_PROG,
                STAGES=2,
                num_warps=2,
            )
    return workspace.scores, workspace.visible


def request_segments_unsupported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> str | None:
    """Why the runtime keeps the paged scorer for this batch, or None."""

    if page_table.ndim != 2 or page_table.shape[0] < 2:
        return "single_request"
    if not 0 < q.shape[0] <= 256:
        return "rows"
    # Fewer than two rows per request on average (MTP drafting steps, including
    # CUDA-graph padding): one-row segments read the keys as often as the paged
    # scorer with a quarter of its programs, which measured 0-15% slower.
    if q.shape[0] < 2 * page_table.shape[0]:
        return "one_row_per_request"
    if q.dtype != torch.float16 or k_cache.dtype != torch.float16:
        return "dtype"
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        return "query_shape"
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        return "cache_shape"
    if any(
        x.dtype != torch.int32 for x in (page_table, token_to_req, sequence_lengths)
    ):
        return "metadata_dtype"
    if query_positions.dtype not in (torch.int32, torch.int64):
        return "position_dtype"
    if token_to_req.shape != (q.shape[0],) or query_positions.shape != (q.shape[0],):
        return "row_metadata_shape"
    if sequence_lengths.shape != (page_table.shape[0],):
        return "lengths_shape"
    tensors = (q, k_cache, page_table, token_to_req, query_positions, sequence_lengths)
    if any(s <= 0 for x in tensors for s in x.stride()):
        return "strides"
    return None


def qsa_mqa_request_segments_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for `qsa.qsa_mqa_paged` on supported batches.

    Same logits width (the page-table capacity), scores and visible lengths.
    Buffers are allocated per call like the paged scorer (from the graph pool
    when captured).
    """

    columns = page_table.shape[1] * k_cache.shape[1]
    workspace = SegmentWorkspace.allocate(q, columns)
    return qsa_mqa_request_segments(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        workspace,
        max_rows=4,
        score_divisor=math.sqrt(q.shape[2]),
    )
