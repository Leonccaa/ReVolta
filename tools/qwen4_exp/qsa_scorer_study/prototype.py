# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent row reference, grouped tiles, and request-segment prototypes.

The explicit FP32 CPU reduction is a comparison contract, not an emulator of
Triton or SM70 MMA. No function imports the runtime, CUDA, or a model.
"""

from dataclasses import dataclass
from math import ceil, isfinite, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
POISON = np.uint32(0x7FC00042)


@dataclass(frozen=True)
class Inputs:
    q: Array
    cache: Array
    table: Array
    requests: Array
    positions: Array
    lengths: Array
    columns: int
    compress: int = 4
    divisor: float = sqrt(128)

    def validate(self) -> None:
        if self.q.ndim != 3 or self.q.shape[1:] != (4, 128):
            raise ValueError("queries must be [rows, 4, 128]")
        if self.cache.ndim != 4 or self.cache.shape[2:] != (1, 128):
            raise ValueError("cache must be [pages, page_size, 1, 128]")
        if self.q.dtype != np.float16 or self.cache.dtype != np.float16:
            raise ValueError("queries and keys must use FP16")
        if not np.isfinite(self.q).all() or not np.isfinite(self.cache).all():
            raise ValueError("this CPU arithmetic contract requires finite Q/K")
        if self.table.ndim != 2 or self.lengths.shape != (len(self.table),):
            raise ValueError("one length is required per page-table request")
        if any(x.shape != (len(self.q),) for x in (self.requests, self.positions)):
            raise ValueError("row metadata must match query rows")
        if any(
            x.dtype.kind not in "iu"
            for x in (self.table, self.requests, self.positions, self.lengths)
        ):
            raise ValueError("metadata must use integer arrays")
        if self.compress <= 0 or not isfinite(self.divisor) or self.divisor <= 0:
            raise ValueError("compression and score divisor must be positive")
        if self.columns < 0 or self.columns > self.table.shape[1] * self.cache.shape[1]:
            raise ValueError("score columns must fit the page-table capacity")
        if len(self.q) and (
            not len(self.table)
            or not all(self.cache.shape[:2])
            or not self.table.shape[1]
        ):
            raise ValueError("nonempty queries need nonempty paged storage")


@dataclass(frozen=True)
class Segment:
    request: int
    begin: int
    end: int


@dataclass
class Result:
    scores: Array
    visible: Array
    writes: Array
    logical_key_fetches: int = 0
    launched_programs: int = 0
    active_programs: int = 0


def trunc_div(value: int, divisor: int) -> int:
    return value // divisor if value >= 0 else -((-value) // divisor)


def visible_lengths(x: Inputs) -> Array:
    """Match signed division, including negative padded positions."""
    values = []
    for request, position in zip(x.requests, x.positions):
        request = int(request)
        length = int(x.lengths[request]) if 0 <= request < len(x.lengths) else 0
        values.append(
            min(trunc_div(int(position) + 1, x.compress), trunc_div(length, x.compress))
        )
    return np.asarray(values, dtype=np.int64)


def make_segments(
    requests: Array, num_requests: int, max_rows: int = 8
) -> list[Segment]:
    """CPU oracle for a future device builder; never a GPU-to-host readback.

    Preserve row order, split long runs, and keep separate runs of one request
    separate. At most one descriptor per row is needed, including padding.
    """
    if max_rows < 1 or max_rows > 8:
        raise ValueError("shared-key segments support one to eight rows")
    if num_requests < 0 or requests.ndim != 1 or requests.dtype.kind not in "iu":
        raise ValueError("invalid request metadata")
    segments = []
    begin = 0
    while begin < len(requests):
        request = int(requests[begin])
        if not 0 <= request < num_requests:
            begin += 1
            continue
        end = begin + 1
        while (
            end < len(requests)
            and end - begin < max_rows
            and int(requests[end]) == request
        ):
            end += 1
        segments.append(Segment(request, begin, end))
        begin = end
    return segments


def segment_descriptors(
    requests: Array, num_requests: int, capacity: int | None = None
) -> Array:
    """Fixed-capacity [request, begin, end] buffer; -1 marks an unused slot.

    The conservative capacity is the row count, so a future device builder can
    change the segment count without a host readback or replay allocation.
    A smaller capacity requires a bound from existing CPU scheduler metadata.
    """
    capacity = len(requests) if capacity is None else capacity
    segments = make_segments(requests, num_requests)
    if capacity < len(segments) or capacity < 0:
        raise ValueError("descriptor capacity is too small")
    out = np.full((capacity, 3), -1, dtype=np.int32)
    for slot, segment in enumerate(segments):
        out[slot] = (segment.request, segment.begin, segment.end)
    return out


def _empty(x: Inputs) -> Result:
    x.validate()
    scores = np.full((len(x.q), x.columns), POISON, dtype=np.uint32).view(np.float32)
    return Result(scores, visible_lengths(x), np.zeros(scores.shape, dtype=np.uint8))


def _keys(x: Inputs, request: int, columns: Array) -> tuple[Array, Array]:
    """Candidate gather; ordinary indexing preserves all input strides."""
    pages = x.table[request, columns // x.cache.shape[1]].astype(np.int64)
    valid = (pages >= 0) & (pages < len(x.cache))
    keys = np.zeros((len(columns), 128), dtype=np.float16)
    keys[valid] = x.cache[pages[valid], columns[valid] % x.cache.shape[1], 0]
    return keys, valid


def _score_queries(queries: Array, keys: Array, divisor: float) -> Array:
    # Vectorize independent rows/heads/columns, never the ordered K reduction.
    accum = np.zeros((len(queries), 4, len(keys)), dtype=np.float32)
    for dim in range(128):
        product = (
            queries[:, :, dim].astype(np.float32)[:, :, None]
            * keys[:, dim].astype(np.float32)[None, None, :]
        )
        accum += product
    np.maximum(accum, np.float32(0), out=accum)
    return (((accum[:, 0] + accum[:, 1]) + accum[:, 2]) + accum[:, 3]) / np.float32(
        divisor
    )


def row_reference(x: Inputs) -> Result:
    """Independent gather and scalar head/K loops; vectorize columns only."""
    result = _empty(x)
    # Intentionally do not call the candidate gather or score helpers.
    for row in range(len(x.q)):
        request = int(x.requests[row])
        if not 0 <= request < len(x.lengths):
            continue
        length = min(int(x.positions[row]) + 1, int(x.lengths[request]))
        end = max(0, min(x.columns, trunc_div(length, x.compress)))
        columns = np.arange(end)
        page_ids = x.table[request][columns // x.cache.shape[1]]
        valid = (page_ids >= 0) & (page_ids < x.cache.shape[0])
        vectors = np.zeros((end, 128), dtype=np.float32)
        vectors[valid] = x.cache[page_ids[valid], columns[valid] % x.cache.shape[1], 0]
        heads = []
        for head in range(4):
            value = np.zeros(end, dtype=np.float32)
            for dim in range(128):
                value = value + np.float32(x.q[row, head, dim]) * vectors[:, dim]
            heads.append(np.maximum(value, np.float32(0)))
        scores = ((heads[0] + heads[1]) + heads[2]) + heads[3]
        scores = scores / np.float32(x.divisor)
        result.scores[row, :end] = np.where(valid, scores, -np.inf)
        result.writes[row, :end] = 1
        result.logical_key_fetches += int(valid.sum())
    return result


def tile_grouped(x: Inputs, tiles_per_program: int) -> Result:
    """Keep 64-column score tiles; group 1, 4, or 8 tiles per row/program.

    CPU arithmetic across independent tiles is vectorized. This checks the
    resulting column ownership, not the GPU compiler's instruction schedule.
    """
    if tiles_per_program not in (1, 4, 8):
        raise ValueError("expected one, four, or eight tiles per program")
    result = _empty(x)
    width = 64 * tiles_per_program
    grid = ceil(x.columns / width)
    result.launched_programs = len(x.q) * grid
    for row in range(len(x.q)):
        end = max(0, min(x.columns, int(result.visible[row])))
        for program in range(grid):
            begin = program * width
            if begin >= end:
                continue
            columns = np.arange(begin, min(begin + width, end))
            keys, valid = _keys(x, int(x.requests[row]), columns)
            values = _score_queries(x.q[row : row + 1], keys, x.divisor)[0]
            result.scores[row, columns] = np.where(valid, values, -np.inf)
            result.writes[row, columns] += 1
            result.logical_key_fetches += int(valid.sum())
            result.active_programs += 1
    return result


def request_segmented(x: Inputs) -> Result:
    """Share paged keys within each request run, using 32-column native tiles."""
    result = _empty(x)
    descriptors = segment_descriptors(x.requests, len(x.lengths))
    grid = ceil(x.columns / 32)
    result.launched_programs = len(descriptors) * grid
    for request, begin_row, end_row in descriptors:
        if request < 0:
            continue
        segment = Segment(int(request), int(begin_row), int(end_row))
        visible = result.visible[segment.begin : segment.end]
        end = max(0, min(x.columns, int(visible.max())))
        for program in range(grid):
            begin = program * 32
            if begin >= end:
                continue
            columns = np.arange(begin, min(begin + 32, end))
            keys, valid = _keys(x, segment.request, columns)
            values = _score_queries(x.q[segment.begin : segment.end], keys, x.divisor)
            for local, row in enumerate(range(segment.begin, segment.end)):
                live = columns < result.visible[row]
                result.scores[row, columns[live]] = np.where(
                    valid, values[local], -np.inf
                )[live]
                result.writes[row, columns[live]] += 1
            result.logical_key_fetches += int(valid.sum())
            result.active_programs += 1
    return result


def select_topk(result: Result, k: int = 512) -> Array:
    """Score descending/index ascending selection, final indices ascending.

    Invalid physical pages still have a defined -inf score in the visible
    prefix. They are not silently removed from the reference selection.
    """
    if k < 1:
        raise ValueError("top-k must be positive")
    selected = np.full((len(result.visible), k), -1, dtype=np.int32)
    for row, visible in enumerate(result.visible):
        end = max(0, min(result.scores.shape[1], int(visible)))
        indices = np.arange(end, dtype=np.int32)
        order = np.lexsort((indices, -result.scores[row, :end]))
        ids = np.sort(indices[order[:k]])
        selected[row, : len(ids)] = ids
    return selected


def assert_equivalent(reference: Result, actual: Result) -> dict[str, int]:
    """Check values, writes, untouched padding, and exact top-512 IDs."""
    np.testing.assert_array_equal(actual.visible, reference.visible)
    np.testing.assert_array_equal(actual.writes, reference.writes)
    np.testing.assert_array_equal(
        actual.scores.view(np.uint32), reference.scores.view(np.uint32)
    )
    np.testing.assert_array_equal(select_topk(actual), select_topk(reference))
    assert np.all(actual.writes <= 1), "duplicate score write"
    return {
        "score_words": int(actual.scores.size),
        "visible_score_words": int(actual.writes.sum()),
        "top512_rows": len(actual.visible),
    }


def launch_geometry(
    requests: Array, visible: Array, columns: int, num_requests: int
) -> dict[str, dict[str, int]]:
    """Schedule arithmetic only; counts are not GPU timings or HBM traffic."""
    if columns < 0 or visible.shape != requests.shape:
        raise ValueError("invalid geometry")
    ends = np.clip(visible, 0, columns).astype(np.int64)
    plans = {}
    for group in (1, 4, 8):
        launched = len(requests) * ceil(columns / (64 * group))
        active = sum(ceil(int(end) / (64 * group)) for end in ends)
        plans[f"tile{group}"] = {
            "launched": launched,
            "active": active,
            "empty": launched - active,
        }
    segments = make_segments(requests, num_requests)
    launched = len(requests) * ceil(columns / 32)
    active = sum(ceil(int(ends[s.begin : s.end].max()) / 32) for s in segments)
    plans["segmented"] = {
        "launched": launched,
        "active": active,
        "empty": launched - active,
        "segments": len(segments),
        "compact_ideal_launched": len(segments) * ceil(columns / 32),
        "descriptor_capacity": len(requests),
        "descriptor_bytes": len(requests) * 3 * 4,
    }
    return plans
