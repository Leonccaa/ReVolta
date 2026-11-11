# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU plans for same-stream, page/subpage-coalesced prefill write-through."""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Copy2D:
    source: int
    destination: int
    width: int
    height: int = 1
    source_pitch: int = 0
    destination_pitch: int = 0


@dataclass(frozen=True)
class WritePlan:
    targets: tuple[tuple[int, int], ...] | NDArray[np.int64]
    copies: tuple[Copy2D, ...]
    payload_bytes: int


def prefill_slots(
    tables: Mapping[int, Sequence[int]],
    indices: Sequence[int],
    counts: Sequence[int],
    starts: Sequence[int],
    prefilling: Sequence[bool],
    padded_rows: int,
    kbs: int,
    world: int,
    rank: int,
    interleave: int,
) -> list[int]:
    """Mirror the slot kernel using known prefill positions. Decode stays UVA.

    Speculative decode positions can be GPU-owned, so they are deliberately
    absent from this plan even in a mixed batch.
    """
    slots = [-1] * padded_rows
    cursor = 0
    for index, count, start, prefill in zip(indices, counts, starts, prefilling):
        count, start = int(count), int(start)
        if prefill:
            table = tables[int(index)]
            for row in range(count):
                position = start + row
                block, offset = divmod(position, kbs * world)
                if offset // interleave % world == rank:
                    local = offset // (world * interleave) * interleave
                    local += offset % interleave
                    slots[cursor + row] = int(table[block]) * kbs + local
        cursor += count
    if cursor > padded_rows:
        raise ValueError("Write batch exceeds its padded row count")
    return slots


def plan_write(
    slots: Sequence[int],
    pages: Sequence[int],
    ratio: int,
    kbs: int,
    row_bytes: int,
    page_stride: int,
    subpage_stride: int,
    plane_stride: int,
) -> WritePlan:
    """Pack contiguous token runs as K-run, V-run. Merge full subpages in 2D.

    Offsets are bytes relative to a layer view. Its data pointer supplies the
    layer offset at submission, so all layers share one CPU plan.
    """
    active = sorted((int(slot), row) for row, slot in enumerate(slots) if slot >= 0)
    if len({slot for slot, _ in active}) != len(active):
        raise ValueError("Duplicate write slot")
    targets = [(-1, -1)] * len(slots)
    runs: list[tuple[int, int, list[int]]] = []
    for slot, row in active:
        kb, offset = divmod(slot, kbs)
        hid, sub = divmod(kb, ratio)
        if hid >= len(pages):
            raise ValueError("Write slot exceeds page table")
        page = int(pages[hid])
        if page <= 0:
            continue
        destination = page * page_stride + sub * subpage_stride + offset * row_bytes
        # A run never crosses a subpage or a gap within it.
        if (
            runs
            and offset
            and destination == runs[-1][0] + len(runs[-1][2]) * row_bytes
        ):
            runs[-1][2].append(row)
        else:
            runs.append((destination, offset, [row]))
    copies: list[Copy2D] = []
    packed = 0
    for destination, offset, rows in runs:
        count = len(rows)
        for position, row in enumerate(rows):
            targets[row] = (packed + position, packed + count + position)
        size = count * row_bytes
        if offset == 0 and count == kbs:
            copy = Copy2D(packed * row_bytes, destination, 2 * size)
            # Coalesce the fixed subpage pitch to keep every 2D pitch bounded.
            if copies and copies[-1].height >= 1 and copies[-1].width == copy.width:
                prev = copies[-1]
                sp = (
                    copy.source - prev.source if prev.height == 1 else prev.source_pitch
                )
                dp = (
                    copy.destination - prev.destination
                    if prev.height == 1
                    else prev.destination_pitch
                )
                if (
                    sp == copy.width
                    and dp == subpage_stride
                    and copy.source == prev.source + prev.height * sp
                    and copy.destination == prev.destination + prev.height * dp
                ):
                    copies[-1] = replace(
                        prev,
                        height=prev.height + 1,
                        source_pitch=sp,
                        destination_pitch=dp,
                    )
                else:
                    copies.append(copy)
            else:
                copies.append(copy)
        else:
            copies.append(
                Copy2D(packed * row_bytes, destination, size, 2, size, plane_stride)
            )
        packed += 2 * count
    return WritePlan(tuple(targets), tuple(copies), packed * row_bytes)


def plan_prefill_write(
    tables: Mapping[int, Sequence[int]],
    indices: Sequence[int],
    counts: Sequence[int],
    starts: Sequence[int],
    prefilling: Sequence[bool],
    targets: NDArray[np.int64],
    ramp: NDArray[np.int64],
    pages: Sequence[int],
    ratio: int,
    kbs: int,
    row_bytes: int,
    page_stride: int,
    subpage_stride: int,
    plane_stride: int,
    world: int,
    rank: int,
    interleave: int,
) -> WritePlan | None:
    """Single contiguous prefill, planned in subpages rather than tokens.

    Return None for multi-request/mixed batches, unsupported CP geometry,
    missing/unbound pages or overlapping slots; the caller uses plan_write.
    Sort only subpage runs to preserve the reference's physical-slot packing.
    ``targets`` and ``ramp`` are reusable contiguous int64 CPU arrays owned by
    the caller. Their lifetime must extend through the target HtoD operation.
    """
    if not (len(indices) == len(counts) == len(starts) == len(prefilling) == 1):
        return None
    count, start = int(counts[0]), int(starts[0])
    if (
        not prefilling[0]
        or count <= 0
        or start < 0
        or count > len(targets)
        or world < 1
        or not 0 <= rank < world
        or interleave < 1
        or kbs < 1
        or kbs % interleave
        or ratio < 1
        or len(ramp) < kbs
    ):
        return None
    table = tables.get(int(indices[0]))
    if table is None:
        return None
    span = kbs * world
    cycle = world * interleave

    def local_before(offset: int) -> int:
        full, tail = divmod(offset, cycle)
        return full * interleave + min(max(tail - rank * interleave, 0), interleave)

    # (physical slot, logical block, local offset, count). A rank's local
    # offsets are contiguous even when its global rows are interleaved.
    runs: list[tuple[int, int, int, int]] = []
    for block in range(start // span, (start + count - 1) // span + 1):
        if block >= len(table):
            return None
        lo = local_before(max(start - block * span, 0))
        hi = local_before(min(start + count - block * span, span))
        if lo == hi:
            continue
        physical = int(table[block])
        hid = physical // ratio
        if physical < 0 or hid >= len(pages) or int(pages[hid]) <= 0:
            return None
        runs.append((physical * kbs + lo, block, lo, hi - lo))
    runs.sort()
    # Aliased table entries can also produce disjoint slices of one subpage.
    # The reference joins those slices in slot order; leave that case to it.
    if any(a[0] // kbs == b[0] // kbs for a, b in zip(runs, runs[1:])):
        return None
    targets.fill(-1)
    # Mutable descriptor fields allow one allocation per final copy, without
    # dataclasses.replace for every full subpage joined to a 2D copy.
    copies: list[list[int]] = []
    packed = 0
    for slot, block, offset, take in runs:
        physical = slot // kbs
        hid, sub = divmod(physical, ratio)
        destination = (
            int(pages[hid]) * page_stride + sub * subpage_stride + offset * row_bytes
        )
        values = ramp[:take]
        if world == 1:
            row = block * span + offset - start
            rows: slice | NDArray[np.int64] = slice(row, row + take)
        elif interleave == 1:
            row = block * span + offset * world + rank - start
            rows = slice(row, row + take * world, world)
        else:
            local = values + offset
            rows = (
                block * span
                + local // interleave * cycle
                + rank * interleave
                + local % interleave
                - start
            )
        targets[rows, 0] = values + packed
        targets[rows, 1] = values + packed + take
        size = take * row_bytes
        if offset == 0 and take == kbs:
            source, width = packed * row_bytes, 2 * size
            prev = copies[-1] if copies else None
            if (
                prev is not None
                and prev[2] == width
                and source == prev[0] + prev[3] * width
                and destination == prev[1] + prev[3] * subpage_stride
                and (prev[3] == 1 or prev[4:] == [width, subpage_stride])
            ):
                prev[3] += 1
                prev[4:] = [width, subpage_stride]
            else:
                copies.append([source, destination, width, 1, 0, 0])
        else:
            copies.append(
                [packed * row_bytes, destination, size, 2, size, plane_stride]
            )
        packed += 2 * take
    return WritePlan(
        targets, tuple(Copy2D(*copy) for copy in copies), packed * row_bytes
    )


@lru_cache(maxsize=1)
def _memcpy2d():
    names = [ctypes.util.find_library("cudart"), "libcudart.so.12", "libcudart.so"]
    for root in sys.path:
        names.extend(
            str(p)
            for p in (Path(root) / "nvidia/cuda_runtime/lib").glob("libcudart.so*")
        )
    for name in filter(None, names):
        try:
            library = ctypes.CDLL(name)
            function = library.cudaMemcpy2DAsync
        except (OSError, AttributeError):
            continue
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return library, function
    raise RuntimeError("Cannot load cudaMemcpy2DAsync for staged write-through")


def submit(plan: WritePlan, source: int, destination: int, stream: int) -> None:
    """Enqueue D2H after the producer on the caller's stream, then commit there."""
    _, memcpy = _memcpy2d()
    for copy in plan.copies:
        code = memcpy(
            destination + copy.destination,
            copy.destination_pitch or copy.width,
            source + copy.source,
            copy.source_pitch or copy.width,
            copy.width,
            copy.height,
            2,
            stream,
        )
        if code:
            raise RuntimeError(
                f"Staged write-through cudaMemcpy2DAsync failed ({code})"
            )
