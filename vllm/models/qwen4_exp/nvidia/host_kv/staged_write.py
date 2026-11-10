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
    targets: tuple[tuple[int, int], ...]
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
            # Merge arithmetic destination strides, including unordered page bindings.
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
                    and dp >= copy.width
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
