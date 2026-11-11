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
    first, last = start // span, (start + count - 1) // span
    if last >= len(table):
        return None
    blocks = np.arange(first, last + 1, dtype=np.int64)
    left = np.maximum(start - blocks * span, 0)
    right = np.minimum(start + count - blocks * span, span)
    if world == 1:
        lo, hi = left, right
    elif interleave == 1:
        lo = (left + world - 1 - rank) // world
        hi = (right + world - 1 - rank) // world
    else:
        lo = left // cycle * interleave + np.clip(
            left % cycle - rank * interleave, 0, interleave
        )
        hi = right // cycle * interleave + np.clip(
            right % cycle - rank * interleave, 0, interleave
        )
    active = hi > lo
    physical = np.asarray(table[first : last + 1], dtype=np.int64)[active]
    blocks, lo, take = blocks[active], lo[active], (hi - lo)[active]
    if len(physical) == 0:
        targets.fill(-1)
        return WritePlan(targets, (), 0)
    hid = physical // ratio
    if np.any(physical < 0) or np.any(hid >= len(pages)):
        return None
    page = np.asarray(pages)[hid].astype(np.int64, copy=False)
    if np.any(page <= 0):
        return None
    order = np.argsort(physical)
    physical, blocks, lo, take, page = (
        array[order] for array in (physical, blocks, lo, take, page)
    )
    # Disjoint aliases of a subpage need the reference's run joining too.
    if np.any(physical[1:] == physical[:-1]):
        return None
    ends = np.cumsum(take)
    prefix = ends - take
    local_rows = int(ends[-1])
    # The store supplies a chunk-sized ramp. Small standalone callers can
    # still supply a subpage ramp without changing the planning contract.
    linear = (
        ramp[:local_rows]
        if len(ramp) >= local_rows
        else np.arange(local_rows, dtype=np.int64)
    )
    token_prefix = np.repeat(prefix, take)
    if world == 1:
        rows = linear + np.repeat(blocks * span + lo - prefix - start, take)
    elif interleave == 1:
        rows = linear * world + np.repeat(
            blocks * span + rank + (lo - prefix) * world - start, take
        )
    else:
        local = linear + np.repeat(lo - prefix, take)
        rows = (
            np.repeat(blocks * span + rank * interleave - start, take)
            + local // interleave * cycle
            + local % interleave
        )
    targets.fill(-1)
    k_target = linear + token_prefix
    targets[rows, 0] = k_target
    targets[rows, 1] = k_target + np.repeat(take, take)

    destination = (
        page * page_stride + physical % ratio * subpage_stride + lo * row_bytes
    )
    source = 2 * prefix * row_bytes
    size = take * row_bytes
    full = (lo == 0) & (take == kbs)
    # Only full, consecutive subpages may join. Source packing already makes
    # their source pitch exactly 2*kbs*row_bytes. Partial planes stay separate.
    join = full[1:] & full[:-1] & (destination[1:] - destination[:-1] == subpage_stride)
    starts_of_copies = np.concatenate(([0], np.flatnonzero(~join) + 1))
    copy_heights = np.diff(np.concatenate((starts_of_copies, [len(take)])))
    copies = []
    for begin, height in zip(starts_of_copies, copy_heights):
        nbytes = int(size[begin])
        if full[begin]:
            width = 2 * nbytes
            h = int(height)
            copy = Copy2D(
                int(source[begin]),
                int(destination[begin]),
                width,
                h,
                width if h > 1 else 0,
                subpage_stride if h > 1 else 0,
            )
        else:
            copy = Copy2D(
                int(source[begin]),
                int(destination[begin]),
                nbytes,
                2,
                nbytes,
                plane_stride,
            )
        copies.append(copy)
    return WritePlan(targets, tuple(copies), 2 * local_rows * row_bytes)


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
