# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-shape SM70 decode kernels for Hy4 Flash."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def gather_decode_group(
    Keys,
    Lengths,
    Slots,
    RingTable,
    Ring,
    Raw,
    GroupPositions,
    ring_table_stride,
    ring_page_stride,
    ring_row_stride,
    RATIO: tl.constexpr,
    DIM: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(Lengths + row)
    active = tl.load(Slots + row) >= 0
    close = active & (length % RATIO == 0)
    member = tl.arange(0, RATIO)
    d = tl.arange(0, DIM)
    ring_id = tl.load(RingTable + row * ring_table_stride, close, other=0)
    base = Ring + ring_id.to(tl.int64) * ring_page_stride
    current = tl.load(Keys + row * DIM + d, close, other=0).to(tl.float32)
    saved = tl.load(
        base + member[:, None] * ring_row_stride + d[None, :],
        close & (member[:, None] < RATIO - 1),
        other=0,
    )
    raw = tl.where(member[:, None] == RATIO - 1, current[None, :], saved)
    tl.store(Raw + (row * RATIO + member[:, None]) * DIM + d[None, :], raw)
    first_position = tl.load(base + DIM, close, other=0).to(tl.int64)
    tl.store(GroupPositions + row, first_position)


@triton.jit
def store_decode_group(
    Keys,
    Positions,
    Pooled,
    Lengths,
    Slots,
    PoolTable,
    RingTable,
    Pool,
    Ring,
    pool_table_stride,
    ring_table_stride,
    pool_page_stride,
    pool_row_stride,
    ring_page_stride,
    ring_row_stride,
    RATIO: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(Slots + row) >= 0:
        logical = tl.load(Lengths + row) - 1
        member = logical % RATIO
        d = tl.arange(0, DIM)
        if member == RATIO - 1:
            group = logical // RATIO
            page = group // PAGE
            block = tl.load(
                PoolTable + row * pool_table_stride + page // PAGES_PER_BLOCK
            )
            physical = block * PAGES_PER_BLOCK + page % PAGES_PER_BLOCK
            pool_dst = Pool + physical.to(tl.int64) * pool_page_stride
            pool_dst += (group % PAGE) * pool_row_stride
            tl.store(pool_dst + d, tl.load(Pooled + row * DIM + d))
        else:
            ring_id = tl.load(RingTable + row * ring_table_stride)
            ring_dst = Ring + ring_id.to(tl.int64) * ring_page_stride
            ring_dst += member * ring_row_stride
            tl.store(ring_dst + d, tl.load(Keys + row * DIM + d).to(tl.float32))
            tl.store(ring_dst + DIM, tl.load(Positions + row).to(tl.float32))


@triton.jit
def _decode_slots(
    Lengths,
    Slots,
    Table,
    Groups,
    Indices,
    Counts,
    table_stride,
    indices_stride,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(Lengths + row)
    active = tl.load(Slots + row) >= 0
    columns = tl.arange(0, BLOCK)
    count = tl.minimum(length, WIDTH)
    if RATIO > 1:
        logical = columns
        if active & (length > WIDTH):
            history = WIDTH // RATIO - 1
            group = tl.load(
                Groups + row * history + columns // RATIO,
                columns < history * RATIO,
                other=0,
            )
            current = (length - 1) // RATIO
            group = tl.where(columns < history * RATIO, group, current)
            logical = group.to(tl.int32) * RATIO + columns % RATIO
            count = history * RATIO + (length - 1) % RATIO + 1
    else:
        logical = tl.maximum(length - WIDTH, 0) + columns
    live = active & (columns < count)
    page = tl.load(Table + row * table_stride + logical // PAGE, live, other=0)
    physical = page * PAGE + logical % PAGE
    tl.store(
        Indices + row * indices_stride + columns,
        tl.where(live, physical, -1),
        columns < WIDTH,
    )
    tl.store(Counts + row, tl.where(active, count, 0))


def decode_slots(
    lengths: torch.Tensor,
    slots: torch.Tensor,
    table: torch.Tensor,
    groups: torch.Tensor | None,
    indices: torch.Tensor,
    counts: torch.Tensor,
    page: int,
    ratio: int,
) -> None:
    rows, width = indices.shape
    _decode_slots[(rows,)](
        lengths,
        slots,
        table,
        groups,
        indices,
        counts,
        table.stride(0),
        indices.stride(0),
        page,
        width,
        ratio if groups is not None else 1,
        triton.next_power_of_2(width),
        num_warps=4,
    )
