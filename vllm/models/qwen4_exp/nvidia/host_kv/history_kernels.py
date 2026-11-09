# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-ID-keyed kernels for the engine's host history store.

These mirror the host KV pool single-layer kernels in ``kernels.py`` but address the
host pages and the hot groups by the scheduler's history block IDs instead of
a request slot. A hot group ID is ``kernel_block * (kernel_block_tokens // 4)
+ offset // 4`` in the rank-local token space, so shared prefix pages are one
identity and a batch reorder never moves residency. Victim selection and the
mapping update are identity-agnostic and reuse ``kernels.py``.

All grids are fixed by host-side shapes; counts stay on the device.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def local_length(length, WORLD: tl.constexpr, RANK: tl.constexpr, IL: tl.constexpr):
    """Positions in [0, length) this DCP rank owns (interleave IL)."""
    span: tl.constexpr = WORLD * IL
    rem = length % span
    return length // span * IL + tl.minimum(tl.maximum(rem - RANK * IL, 0), IL)


@triton.jit
def store_offsets(
    kernel_block,
    plane,
    offset,
    head,
    dim,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
):
    # Widen every independent product before adding it to the block term.
    return (
        tl.cast(kernel_block, tl.int64) * HS0
        + tl.cast(plane, tl.int64) * HS1
        + tl.cast(offset, tl.int64) * HS2
        + tl.cast(head, tl.int64) * HS3
        + dim
    )


@triton.jit
def write_slots(
    K,
    V,
    Slot,
    Host,
    Hot,
    Map,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    KBS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    """Copy one encoded row to its host page and, if resident, its hot group."""
    row = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, B).to(tl.int64)
    slot = tl.load(Slot + row).to(tl.int64)
    valid = slot >= 0
    safe = tl.maximum(slot, 0)
    kernel_block, offset = safe // KBS, safe % KBS
    lane = valid & (x < H * D)
    key = tl.load(K + row * KS0 + (x // D) * KS1 + x % D, lane, 0)
    value = tl.load(V + row * VS0 + (x // D) * VS1 + x % D, lane, 0)
    host = store_offsets(kernel_block, 0, offset, x // D, x % D, HS0, HS1, HS2, HS3)
    tl.store(Host + host, key, lane)
    tl.store(Host + host + HS1, value, lane)
    hot_slot = tl.load(Map + kernel_block * (KBS // 4) + offset // 4, valid, -1)
    hot = hot_slot.to(tl.int64) * (8 * H * D) + (offset % 4) * H * D + x
    tl.store(Hot + hot, key, lane & (hot_slot >= 0))
    tl.store(Hot + hot + 4 * H * D, value, lane & (hot_slot >= 0))


@triton.jit
def commit_slots(Slot, Ready, N: tl.constexpr, B: tl.constexpr):
    """Publish rows after a kernel boundary that follows both copies."""
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    slot = tl.load(Slot + i, i < N, -1).to(tl.int64)
    tl.store(Ready + tl.maximum(slot, 0), 1, (i < N) & (slot >= 0))


@triton.jit
def set_block_ready(
    Blocks,
    Ready,
    Generation,
    RATIO: tl.constexpr,
    KBS: tl.constexpr,
    B: tl.constexpr,
    VALUE: tl.constexpr,
    BUMP: tl.constexpr,
):
    """Set every token's readiness of the listed blocks; optionally bump
    their generation (a new allocation)."""
    i = tl.program_id(0)
    chunk = tl.program_id(1)
    block = tl.load(Blocks + i).to(tl.int64)
    x = chunk.to(tl.int64) * B + tl.arange(0, B)
    tl.store(Ready + block * (RATIO * KBS) + x, VALUE, x < RATIO * KBS)
    if BUMP and chunk == 0:
        tl.store(Generation + block, tl.load(Generation + block) + 1)


@triton.jit
def retire_blocks(
    Blocks,
    Map,
    Owner,
    Age,
    G: tl.constexpr,
    S: tl.constexpr,
    GPB: tl.constexpr,
    B: tl.constexpr,
):
    """Drop every layer's hot residency of newly allocated blocks."""
    i = tl.program_id(0)
    chunk = tl.program_id(1)
    layer = tl.program_id(2).to(tl.int64)
    block = tl.load(Blocks + i).to(tl.int64)
    g = chunk.to(tl.int64) * B + tl.arange(0, B)
    gid = block * GPB + g
    active = g < GPB
    slot = tl.load(Map + layer * G + gid, active, -1)
    resident = active & (slot >= 0)
    tl.store(Owner + layer * S + tl.maximum(slot, 0), -1, resident)
    tl.store(Age + layer * S + tl.maximum(slot, 0), 0, resident)
    tl.store(Map + layer * G + gid, -1, active)


@triton.jit
def append_wanted(gid, valid, Wanted, Touched, Count, M: tl.constexpr):
    old = tl.atomic_xchg(Wanted + gid, 1, mask=valid, sem="relaxed")
    first = valid & (old == 0)
    rank = tl.cumsum(first.to(tl.int32)) - 1
    count = tl.sum(first.to(tl.int32), 0)
    if count > 0:
        start = tl.atomic_add(Count, count, sem="relaxed")
        tl.device_assert(start + count <= M, "host KV wanted bound exceeded")
        tl.store(Touched + start + rank, gid, first & (start + rank < M))


@triton.jit
def mark_tail(
    Table,
    SeqLens,
    Ready,
    Wanted,
    Touched,
    Count,
    TS0,
    KBS: tl.constexpr,
    M: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    IL: tl.constexpr,
):
    """Keep each request's newest 32 local tokens resident (nine groups)."""
    req = tl.program_id(0).to(tl.int64)
    length = local_length(tl.load(SeqLens + req), WORLD, RANK, IL)
    token = length - 32 + tl.arange(0, 32)
    valid = token >= 0
    safe = tl.maximum(token, 0)
    kernel_block = tl.load(Table + req * TS0 + safe // KBS, valid, 0).to(tl.int64)
    ready = tl.load(Ready + kernel_block * KBS + safe % KBS, valid, 1)
    tl.device_assert((~valid) | (ready != 0), "tail KV is not committed")
    gid = kernel_block * (KBS // 4) + (safe % KBS) // 4
    append_wanted(gid, valid, Wanted, Touched, Count, M)


@triton.jit
def mark_selection(
    Indices,
    Req,
    Table,
    SeqLens,
    Ready,
    Wanted,
    Touched,
    Count,
    IS0,
    TS0,
    W: tl.constexpr,
    KBS: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    IL: tl.constexpr,
    QSA_COLUMNS: tl.constexpr,
):
    """Claim each selected group once; attention keeps every column."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    req = tl.load(Req + row).to(tl.int64)
    length = local_length(tl.load(SeqLens + req), WORLD, RANK, IL)
    token = tl.load(Indices + row * IS0 + col, col < W, -1)
    tl.device_assert((token < 0) | (token < length), "host KV selection past length")
    valid = (col < W) & (token >= 0) & (token < length)
    safe = tl.maximum(token, 0)
    kernel_block = tl.load(Table + req * TS0 + safe // KBS, valid, 0).to(tl.int64)
    ready = tl.load(Ready + kernel_block * KBS + safe % KBS, valid, 1)
    tl.device_assert((~valid) | (ready != 0), "selected KV is not committed")
    gid = tl.where(valid, kernel_block * (KBS // 4) + (safe % KBS) // 4, -1)
    if QSA_COLUMNS:
        # QSA emits whole groups (pairs after DCP2 localization): one claim
        # per chunk, after checking every column falls in the same group.
        first = tl.load(
            Indices + row * IS0 + col // QSA_COLUMNS * QSA_COLUMNS, col < W, -1
        )
        tl.device_assert(
            (~valid) | ((first >= 0) & (first // 4 == token // 4)),
            "host KV QSA group layout exceeded",
        )
        group = tl.max(tl.reshape(gid, (B // QSA_COLUMNS, QSA_COLUMNS)), 1)
        append_wanted(tl.maximum(group, 0), group >= 0, Wanted, Touched, Count, M)
    else:
        append_wanted(tl.maximum(gid, 0), valid, Wanted, Touched, Count, M)


@triton.jit
def gather_groups(
    Host,
    Hot,
    Ready,
    Misses,
    MissSlots,
    Count,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    KBS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    """Copy every committed row of each missed group (unready rows read 0)."""
    x = tl.arange(0, B).to(tl.int64)
    plane = x // (4 * H * D)
    within = x // (H * D) % 4
    head, dim = x // D % H, x % D
    for i in range(tl.program_id(0), tl.load(Count), tl.num_programs(0)):
        gid = tl.load(Misses + i).to(tl.int64)
        slot = tl.load(MissSlots + i).to(tl.int64)
        kernel_block = gid // (KBS // 4)
        offset = gid % (KBS // 4) * 4 + within
        ready = tl.load(Ready + kernel_block * KBS + offset)
        host = store_offsets(kernel_block, plane, offset, head, dim, HS0, HS1, HS2, HS3)
        payload = tl.load(Host + host, (x < 8 * H * D) & (ready != 0), 0)
        tl.store(Hot + slot * (8 * H * D) + x, payload, x < 8 * H * D)


@triton.jit
def remap_selection(
    Indices,
    Req,
    Table,
    SeqLens,
    Map,
    Out,
    IS0,
    TS0,
    OS0,
    W: tl.constexpr,
    KBS: tl.constexpr,
    B: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    IL: tl.constexpr,
):
    """Rewrite local token IDs to hot-pool token IDs, column for column."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    req = tl.load(Req + row).to(tl.int64)
    length = local_length(tl.load(SeqLens + req), WORLD, RANK, IL)
    token = tl.load(Indices + row * IS0 + col, col < W, -1)
    valid = (col < W) & (token >= 0) & (token < length)
    safe = tl.maximum(token, 0)
    kernel_block = tl.load(Table + req * TS0 + safe // KBS, valid, 0).to(tl.int64)
    slot = tl.load(Map + kernel_block * (KBS // 4) + (safe % KBS) // 4, valid, -1)
    tl.device_assert((~valid) | (slot >= 0), "selected KV has no hot slot")
    out = tl.where(valid & (slot >= 0), slot * 4 + safe % 4, -1)
    tl.store(Out + row * OS0 + col, out, col < W)
