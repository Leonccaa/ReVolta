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
    PageOf,
    plane,
    offset,
    head,
    dim,
    HS0: tl.constexpr,
    HSS: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    RATIO: tl.constexpr,
):
    # Widen every independent product before adding it to the block term.
    return (
        tl.load(PageOf + kernel_block // RATIO).to(tl.int64) * HS0
        + (kernel_block % RATIO).to(tl.int64) * HSS
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
    PageOf,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    HS0: tl.constexpr,
    HSS: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    RATIO: tl.constexpr,
    KBS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    Ready=None,
    COMMIT: tl.constexpr = False,
):
    """Copy one encoded row to its host page and, if resident, its hot group."""
    row = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, B).to(tl.int64)
    slot = tl.load(Slot + row).to(tl.int64)
    valid = slot >= 0
    safe = tl.maximum(slot, 0)
    kernel_block, offset = safe // KBS, safe % KBS
    page = tl.load(PageOf + kernel_block // RATIO)
    valid = valid & (page > 0)
    lane = valid & (x < H * D)
    key = tl.load(K + row * KS0 + (x // D) * KS1 + x % D, lane, 0)
    value = tl.load(V + row * VS0 + (x // D) * VS1 + x % D, lane, 0)
    host = store_offsets(
        kernel_block, PageOf, 0, offset, x // D, x % D, HS0, HSS, HS1, HS2, HS3, RATIO
    )
    tl.store(Host + host, key, lane)
    tl.store(Host + host + HS1, value, lane)
    hot_slot = tl.load(Map + kernel_block * (KBS // 4) + offset // 4, valid, -1)
    hot = hot_slot.to(tl.int64) * (8 * H * D) + (offset % 4) * H * D + x
    tl.store(Hot + hot, key, lane & (hot_slot >= 0))
    tl.store(Hot + hot + 4 * H * D, value, lane & (hot_slot >= 0))
    if COMMIT:
        # Every consumer is on the same stream, after this kernel boundary.
        # The connector's retirement event still fences all host writes.
        tl.store(Ready + safe, 1, valid)


@triton.jit
def stage_slots(
    K,
    V,
    Slot,
    Host,
    Hot,
    Stage,
    Targets,
    Map,
    PageOf,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    HS0: tl.constexpr,
    HSS: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    RATIO: tl.constexpr,
    KBS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    """Write planned rows to compact GPU storage, refresh hot, retain decode UVA."""
    row = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, B).to(tl.int64)
    slot = tl.load(Slot + row).to(tl.int64)
    valid = slot >= 0
    safe = tl.maximum(slot, 0)
    kernel_block, offset = safe // KBS, safe % KBS
    page = tl.load(PageOf + kernel_block // RATIO)
    valid = valid & (page > 0)
    lane = valid & (x < H * D)
    key = tl.load(K + row * KS0 + (x // D) * KS1 + x % D, lane, 0)
    value = tl.load(V + row * VS0 + (x // D) * VS1 + x % D, lane, 0)
    host = store_offsets(
        kernel_block, PageOf, 0, offset, x // D, x % D, HS0, HSS, HS1, HS2, HS3, RATIO
    )
    k_target = tl.load(Targets + row * 2).to(tl.int64)
    v_target = tl.load(Targets + row * 2 + 1).to(tl.int64)
    staged = k_target >= 0
    tl.store(Stage + k_target * (H * D) + x, key, lane & staged)
    tl.store(Stage + v_target * (H * D) + x, value, lane & staged)
    tl.store(Host + host, key, lane & ~staged)
    tl.store(Host + host + HS1, value, lane & ~staged)
    hot_slot = tl.load(Map + kernel_block * (KBS // 4) + offset // 4, valid, -1)
    hot = hot_slot.to(tl.int64) * (8 * H * D) + (offset % 4) * H * D + x
    tl.store(Hot + hot, key, lane & (hot_slot >= 0))
    tl.store(Hot + hot + 4 * H * D, value, lane & (hot_slot >= 0))


@triton.jit
def write_slots_vector(
    K,
    V,
    Slot,
    Host,
    Hot,
    Stage,
    Targets,
    Map,
    PageOf,
    N,
    KS0: tl.constexpr,
    VS0: tl.constexpr,
    HS0: tl.constexpr,
    HSS: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    RATIO: tl.constexpr,
    KBS: tl.constexpr,
    W: tl.constexpr,
    WB: tl.constexpr,
    ROWS: tl.constexpr,
    STAGED: tl.constexpr,
):
    """All strides and W are in 8-byte words. One program = ROWS rows."""
    r = (tl.program_id(0) * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    rmask = r < N
    x = tl.max_contiguous(tl.arange(0, WB), WB).to(tl.int64)
    slot = tl.load(Slot + r, rmask, -1).to(tl.int64)
    valid = slot >= 0
    safe = tl.maximum(slot, 0)
    kb, off = safe // KBS, safe % KBS
    page = tl.load(PageOf + kb // RATIO, valid, 0).to(tl.int64)
    valid = valid & (page > 0)
    lane = valid[:, None] & (x < W)[None, :]
    src_k = tl.multiple_of(r * KS0, 2)
    src_v = tl.multiple_of(r * VS0, 2)
    key = tl.load(K + src_k[:, None] + x[None, :], lane, 0)
    value = tl.load(V + src_v[:, None] + x[None, :], lane, 0)
    base = page * HS0 + (kb % RATIO) * HSS + off * HS2
    base = tl.multiple_of(base, 2)
    host_lane = lane
    if STAGED:
        k_target = tl.load(Targets + r * 2, rmask, -1).to(tl.int64)
        v_target = tl.load(Targets + r * 2 + 1, rmask, -1).to(tl.int64)
        staged = k_target >= 0
        stage_k = tl.multiple_of(k_target * W, 2)
        stage_v = tl.multiple_of(v_target * W, 2)
        tl.store(Stage + stage_k[:, None] + x[None, :], key, lane & staged[:, None])
        tl.store(Stage + stage_v[:, None] + x[None, :], value, lane & staged[:, None])
        host_lane = lane & ~staged[:, None]
    tl.store(Host + base[:, None] + x[None, :], key, host_lane)
    tl.store(Host + (base + HS1)[:, None] + x[None, :], value, host_lane)
    hot_slot = tl.load(Map + kb * (KBS // 4) + off // 4, valid, -1).to(tl.int64)
    hot = tl.multiple_of(hot_slot * (8 * W) + (off % 4) * W, 2)
    hmask = lane & (hot_slot >= 0)[:, None]
    tl.store(Hot + hot[:, None] + x[None, :], key, hmask)
    tl.store(Hot + (hot + 4 * W)[:, None] + x[None, :], value, hmask)


@triton.jit
def commit_slots(
    Slot, Ready, PageOf, SPAN: tl.constexpr, N: tl.constexpr, B: tl.constexpr
):
    """Publish rows after a kernel boundary that follows both copies."""
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    slot = tl.load(Slot + i, i < N, -1).to(tl.int64)
    page = tl.load(PageOf + tl.maximum(slot, 0) // SPAN)
    tl.store(Ready + tl.maximum(slot, 0), 1, (i < N) & (slot >= 0) & (page > 0))


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
    # Owner guard: a slot another group or a reserved ID page owns is kept.
    owner = tl.load(Owner + layer * S + tl.maximum(slot, 0), resident, -3)
    resident = resident & (owner == gid)
    tl.store(Owner + layer * S + tl.maximum(slot, 0), -1, resident)
    tl.store(Age + layer * S + tl.maximum(slot, 0), 0, resident)
    tl.store(Map + layer * G + gid, -1, active)


@triton.jit
def set_block_slots(
    Blocks,
    Map,
    Owner,
    Age,
    G: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    CLAIM: tl.constexpr,
):
    """Shared region roles of whole ID pages, every layer (grid: ids, layers).

    Claim: unmap any group resident in the page's k slots, then reserve them
    (owner -2, age 0). Donate: the caller first retired the ID's own groups;
    the reserved slots become free hot slots (owner -1)."""
    i = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    block = tl.load(Blocks + i).to(tl.int64)
    x = tl.arange(0, KP)
    valid = x < K
    slot = block * K + x
    owner = tl.load(Owner + layer * S + slot, valid, -2)
    if CLAIM:
        tl.store(Map + layer * G + tl.maximum(owner, 0), -1, valid & (owner >= 0))
        tl.store(Owner + layer * S + slot, -2, valid)
        tl.store(Age + layer * S + slot, 0, valid)
    else:
        tl.store(Owner + layer * S + slot, -1, valid & (owner == -2))
        tl.store(Age + layer * S + slot, 0, valid & (owner == -2))


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
def append_wanted_bit(
    gid, valid, Wanted, Touched, Count, M: tl.constexpr, BIT: tl.constexpr
):
    """``append_wanted`` for one mark phase of the ordered fused kernel.

    Each phase owns one bit of ``Wanted`` and appends its first claims to its
    own list, so a group the tail wants is listed ahead of selection-only
    groups no matter which CTA runs first. ``collect_misses_ordered`` drops
    the selection list's copy of a group both phases want.
    """
    old = tl.atomic_or(Wanted + gid, BIT, mask=valid, sem="relaxed")
    first = valid & ((old & BIT) == 0)
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
    ROW_OFFSET: tl.constexpr = 0,
    ORDERED: tl.constexpr = False,
):
    """Keep each request's newest 32 local tokens resident (nine groups)."""
    req = tl.program_id(0).to(tl.int64) - ROW_OFFSET
    length = local_length(tl.load(SeqLens + req), WORLD, RANK, IL)
    token = length - 32 + tl.arange(0, 32)
    valid = token >= 0
    safe = tl.maximum(token, 0)
    kernel_block = tl.load(Table + req * TS0 + safe // KBS, valid, 0).to(tl.int64)
    ready = tl.load(Ready + kernel_block * KBS + safe % KBS, valid, 1)
    tl.device_assert((~valid) | (ready != 0), "tail KV is not committed")
    gid = kernel_block * (KBS // 4) + (safe % KBS) // 4
    if ORDERED:
        append_wanted_bit(gid, valid, Wanted, Touched, Count, M, 1)
    else:
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
    ORDERED: tl.constexpr = False,
    Sel=None,
    SelCount=None,
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
        if ORDERED:
            append_wanted_bit(
                tl.maximum(group, 0),
                group >= 0,
                Wanted,
                Sel,
                SelCount,
                M,
                2,
            )
        else:
            append_wanted(tl.maximum(group, 0), group >= 0, Wanted, Touched, Count, M)
    else:
        if ORDERED:
            append_wanted_bit(tl.maximum(gid, 0), valid, Wanted, Sel, SelCount, M, 2)
        else:
            append_wanted(tl.maximum(gid, 0), valid, Wanted, Touched, Count, M)


@triton.jit
def mark_decode(
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
    ROWS: tl.constexpr,
    Sel=None,
    SelCount=None,
    ORDERED: tl.constexpr = False,
):
    """Independent tail and selection claims in one grid, then collect.

    ORDERED (the ``REVOLTA_RESEARCH_HOST_DECODE_FAST_ORDERED`` sub-switch)
    keeps the two phases' first claims in separate lists, the tail list
    ahead of the selection list, so refill order and the Misses/victims
    pairing match the two-kernel baseline. The phases stay fully parallel:
    ordering comes from per-phase bits in ``Wanted``, not from scheduling.
    """
    if tl.program_id(0) < ROWS:
        mark_selection(
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
            W,
            KBS,
            M,
            B,
            WORLD,
            RANK,
            IL,
            QSA_COLUMNS,
            ORDERED,
            Sel,
            SelCount,
        )
    elif tl.program_id(1) == 0:
        mark_tail(
            Table,
            SeqLens,
            Ready,
            Wanted,
            Touched,
            Count,
            TS0,
            KBS,
            M,
            WORLD,
            RANK,
            IL,
            ROW_OFFSET=ROWS,
            ORDERED=ORDERED,
        )


@triton.jit
def gather_groups(
    Host,
    Hot,
    Ready,
    Misses,
    MissSlots,
    Count,
    PageOf,
    HS0: tl.constexpr,
    HSS: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    RATIO: tl.constexpr,
    KBS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    Map=None,
    Owner=None,
    Age=None,
    Clock=None,
    Victims=None,
    Hand=None,
    S: tl.constexpr = 0,
    M: tl.constexpr = 0,
    ALLOCATE: tl.constexpr = False,
):
    """Copy every committed row of each missed group (unready rows read 0)."""
    x = tl.arange(0, B).to(tl.int64)
    plane = x // (4 * H * D)
    within = x // (H * D) % 4
    head, dim = x // D % H, x % D
    count = tl.load(Count)
    if ALLOCATE:
        tl.device_assert((count >= 0) & (count <= M), "host KV miss bound exceeded")
    for i in range(tl.program_id(0), count, tl.num_programs(0)):
        gid = tl.load(Misses + i).to(tl.int64)
        if ALLOCATE:
            # Victims are unique and not wanted; missed gids are unique and
            # wanted. Each CTA therefore owns both mapping writes and its
            # payload destination. Remap waits for this whole kernel.
            slot = tl.load(Victims + i).to(tl.int64)
            old = tl.load(Owner + slot)
            tl.store(Map + tl.maximum(old, 0), -1, old >= 0)
            tl.store(Map + gid, slot.to(tl.int32))
            tl.store(Owner + slot, gid.to(tl.int32))
            tl.store(MissSlots + i, slot.to(tl.int32))
            tl.store(Age + slot, tl.load(Clock))
        else:
            slot = tl.load(MissSlots + i).to(tl.int64)
        kernel_block = gid // (KBS // 4)
        offset = gid % (KBS // 4) * 4 + within
        ready = tl.load(Ready + kernel_block * KBS + offset)
        page = tl.load(PageOf + kernel_block // RATIO)
        tl.device_assert(page >= 0, "unbound host page read")
        host = store_offsets(
            kernel_block,
            PageOf,
            plane,
            offset,
            head,
            dim,
            HS0,
            HSS,
            HS1,
            HS2,
            HS3,
            RATIO,
        )
        payload = tl.load(Host + host, (x < 8 * H * D) & (ready != 0) & (page >= 0), 0)
        tl.store(Hot + slot * (8 * H * D) + x, payload, x < 8 * H * D)
    # All selectors finished in the preceding kernel. Other gather CTAs
    # never read Hand, so advancing it here requires no grid barrier.
    if ALLOCATE and ((tl.program_id(0) == 0) & (count > 0)):
        last = tl.load(Victims + count - 1)
        tl.store(Hand, tl.where(last + 1 == S, 0, last + 1))


@triton.jit
def begin_gather_ordered(WantedCount, SelCount, MissCount, Clock):
    """``begin_gather`` plus the selection segment of the ordered marks."""
    tl.store(WantedCount, 0)
    tl.store(SelCount, 0)
    tl.store(MissCount, 0)
    tl.store(Clock, tl.load(Clock) + 1)


@triton.jit
def clear_wanted_ordered(Wanted, Touched, Count, Sel, SelCount, B: tl.constexpr):
    """``clear_wanted`` over both lists of the ordered fused mark kernel."""
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    active = i < tl.load(Count)
    gid = tl.load(Touched + i, active, 0)
    tl.store(Wanted + gid, 0, active)
    active = i < tl.load(SelCount)
    gid = tl.load(Sel + i, active, 0)
    tl.store(Wanted + gid, 0, active)


@triton.jit
def collect_misses_ordered(
    Touched,
    Count,
    Sel,
    SelCount,
    Wanted,
    Map,
    Age,
    Clock,
    Misses,
    MissCount,
    M: tl.constexpr,
    B: tl.constexpr,
):
    """``collect_misses`` over the ordered mark kernel's two Touched lists.

    The logical order is the baseline's: every tail claim first, then the
    selection claims whose group the tail does not also want (bit 1 of
    ``Wanted`` marks tail-owned groups; the selection list can still hold a
    copy because its CTA may have claimed first).
    """
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tail = tl.load(Count)
    active = i < tail + tl.load(SelCount)
    in_sel = i >= tail
    safe = tl.where(in_sel, i - tail, i)
    gid = tl.where(
        in_sel,
        tl.load(Sel + safe, active & in_sel, 0),
        tl.load(Touched + safe, active & ~in_sel, 0),
    )
    tail_owned = (tl.load(Wanted + gid, active & in_sel, 0) & 1) != 0
    active &= ~(in_sel & tail_owned)
    slot = tl.load(Map + gid, active, -1)
    missing = active & (slot < 0)
    tl.store(Age + tl.maximum(slot, 0), tl.load(Clock), active & (slot >= 0))
    rank = tl.cumsum(missing.to(tl.int32)) - 1
    count = tl.sum(missing.to(tl.int32), 0)
    if count > 0:
        start = tl.atomic_add(MissCount, count, sem="relaxed")
        tl.device_assert(start + count <= M, "host KV miss bound exceeded")
        tl.store(Misses + start + rank, gid, missing & (start + rank < M))


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


# ----- selection-union gather: union gather of a mixed step (selection-union gather)
# ------------------------------
#
# One scan kernel per input kind (request tails, selection rows) runs in four
# modes over the same gid computation:
#   MARK   (0): atomic_or the class bits; count each first claim and append it
#               to Touched only while it fits M. The count saturates nothing,
#               so a caller can compare it with any capacity; Touched is
#               complete exactly when the count is at most M.
#   FIRST  (1): atomic_max a first-reference code (larger = earlier row group,
#               the tail code exceeds every row group's code).
#   COUNT  (2): compare-and-swap the code of the row group that references a
#               gid first with a sentinel, so each gid is counted once, in
#               ``Hist[0]`` (tail) or ``Hist[1 + row_group]``.
#   RESET  (3): store zero at every gid the same inputs reference.
# RESET after MARK or COUNT restores the all-zero state for these gids
# without relying on a bounded Touched list.

UNION_MARK = 0
UNION_FIRST = 1
UNION_COUNT = 2
UNION_RESET = 3


@triton.jit
def _union_update(
    gid,
    valid,
    code,
    Wanted,
    Touched,
    Count,
    Hist,
    hist_slot,
    sentinel,
    M: tl.constexpr,
    MODE: tl.constexpr,
):
    # One code per lane (the interpreter does not broadcast atomic values).
    code = code + tl.zeros_like(valid.to(tl.int32))
    if MODE == 0:
        old = tl.atomic_or(Wanted + gid, code, mask=valid, sem="relaxed")
        first = valid & (old == 0)
        rank = tl.cumsum(first.to(tl.int32)) - 1
        count = tl.sum(first.to(tl.int32), 0)
        if count > 0:
            start = tl.atomic_add(Count, count, sem="relaxed")
            tl.store(Touched + start + rank, gid, first & (start + rank < M))
    elif MODE == 1:
        tl.atomic_max(Wanted + gid, code, mask=valid, sem="relaxed")
    elif MODE == 2:
        # Invalid lanes compare against -1, which no code equals.
        expected = tl.where(valid, code, -1)
        new = sentinel + tl.zeros_like(code)
        old = tl.atomic_cas(Wanted + gid, expected, new, sem="relaxed")
        won = valid & (old == expected)
        count = tl.sum(won.to(tl.int32), 0)
        if count > 0:
            tl.atomic_add(Hist + hist_slot, count, sem="relaxed")
    else:
        tl.store(Wanted + gid, 0, valid)


@triton.jit
def union_tail(
    Table,
    SeqLens,
    Ready,
    Class,
    Wanted,
    Touched,
    Count,
    Hist,
    tail_code,
    sentinel,
    TS0,
    KBS: tl.constexpr,
    M: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    IL: tl.constexpr,
    MODE: tl.constexpr,
):
    """Every request's newest 32 local tokens (nine groups), as mark_tail."""
    req = tl.program_id(0).to(tl.int64)
    length = local_length(tl.load(SeqLens + req), WORLD, RANK, IL)
    token = length - 32 + tl.arange(0, 32)
    valid = token >= 0
    safe = tl.maximum(token, 0)
    kernel_block = tl.load(Table + req * TS0 + safe // KBS, valid, 0).to(tl.int64)
    if MODE == 0:
        ready = tl.load(Ready + kernel_block * KBS + safe % KBS, valid, 1)
        tl.device_assert((~valid) | (ready != 0), "tail KV is not committed")
        code = tl.load(Class + req)
    else:
        code = tail_code
    gid = kernel_block * (KBS // 4) + (safe % KBS) // 4
    _union_update(gid, valid, code, Wanted, Touched, Count, Hist, 0, sentinel, M, MODE)


@triton.jit
def union_selection(
    Indices,
    Req,
    Table,
    SeqLens,
    Ready,
    Class,
    Wanted,
    Touched,
    Count,
    Hist,
    num_row_groups,
    sentinel,
    IS0,
    TS0,
    W: tl.constexpr,
    KBS: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    IL: tl.constexpr,
    MODE: tl.constexpr,
    ROW_GROUP: tl.constexpr = 8,
):
    """Selected groups of rows given relative to an eight-row-aligned start.

    Every column claims its own group, so no selection layout is assumed."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    req = tl.load(Req + row).to(tl.int64)
    length = local_length(tl.load(SeqLens + req), WORLD, RANK, IL)
    token = tl.load(Indices + row * IS0 + col, col < W, -1)
    valid = (col < W) & (token >= 0) & (token < length)
    safe = tl.maximum(token, 0)
    kernel_block = tl.load(Table + req * TS0 + safe // KBS, valid, 0).to(tl.int64)
    row_group = (row // ROW_GROUP).to(tl.int32)
    if MODE == 0:
        tl.device_assert(
            (token < 0) | (token < length), "host KV selection past length"
        )
        ready = tl.load(Ready + kernel_block * KBS + safe % KBS, valid, 1)
        tl.device_assert((~valid) | (ready != 0), "selected KV is not committed")
        code = tl.load(Class + req)
    else:
        code = num_row_groups - row_group
    gid = kernel_block * (KBS // 4) + (safe % KBS) // 4
    _union_update(
        gid, valid, code, Wanted, Touched, Count, Hist, 1 + row_group, sentinel,
        M, MODE,
    )  # fmt: skip


@triton.jit
def begin_fill(MissCount, Clock):
    """Start filling marked groups: the clock moves only when a fill runs."""
    tl.store(MissCount, 0)
    tl.store(Clock, tl.load(Clock) + 1)


@triton.jit
def collect_misses_classed(
    Touched,
    WantedCount,
    Wanted,
    Map,
    Age,
    Clock,
    Misses,
    Count,
    M: tl.constexpr,
    B: tl.constexpr,
):
    """``collect_misses``, except a hit only prefill rows want (class bits 1)
    keeps ``max(age, clock - 9)``: older ages drop to the old bucket, a page a
    recent decode row touched keeps that age."""
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    active = i < tl.load(WantedCount)
    gid = tl.load(Touched + i, active, 0)
    slot = tl.load(Map + gid, active, -1)
    missing = active & (slot < 0)
    clock = tl.load(Clock)
    bits = tl.load(Wanted + gid, active, 0)
    age = tl.load(Age + tl.maximum(slot, 0), active & (slot >= 0), 0)
    hit_age = tl.where(bits == 1, tl.maximum(age, clock - 9), clock)
    tl.store(Age + tl.maximum(slot, 0), hit_age, active & (slot >= 0))
    rank = tl.cumsum(missing.to(tl.int32)) - 1
    count = tl.sum(missing.to(tl.int32), 0)
    if count > 0:
        start = tl.atomic_add(Count, count, sem="relaxed")
        tl.device_assert(start + count <= M, "host KV miss bound exceeded")
        tl.store(Misses + start + rank, gid, missing & (start + rank < M))


@triton.jit
def age_prefill_misses(
    Misses,
    MissSlots,
    Count,
    Wanted,
    Age,
    Clock,
    Counters,
    B: tl.constexpr,
):
    """Insert groups only prefill rows want as old (``clock - 9``)."""
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    active = i < tl.load(Count)
    gid = tl.load(Misses + i, active, 0)
    slot = tl.load(MissSlots + i, active, 0)
    prefill = active & (tl.load(Wanted + gid, active, 0) == 1)
    tl.store(Age + slot, tl.load(Clock) - 9, prefill)
    count = tl.sum(prefill.to(tl.int64), 0)
    if count > 0:
        tl.atomic_add(Counters + 2, count, sem="relaxed")


@triton.jit
def accumulate_union(WantedCount, MissCount, Counters):
    """Device counters: union groups and missed groups of every fill."""
    tl.store(Counters, tl.load(Counters) + tl.load(WantedCount).to(tl.int64))
    tl.store(Counters + 1, tl.load(Counters + 1) + tl.load(MissCount).to(tl.int64))


@triton.jit
def build_microblock_table(
    Table,
    SeqLens,
    Map,
    Out,
    requests,
    TS0,
    WMB: tl.constexpr,
    KBS: tl.constexpr,
    B: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    IL: tl.constexpr,
):
    """Page-size-4 table: ``Out[q, m]`` is the hot slot of the group that
    holds local tokens ``[4m, 4m + 4)`` of request q, the same lookup
    ``remap_selection`` applies per column. Groups past a request's length,
    rows past ``requests`` and non-resident groups read -1."""
    req = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    seq = tl.load(SeqLens + tl.minimum(req, requests - 1))
    length = tl.where(req < requests, local_length(seq, WORLD, RANK, IL), 0)
    valid = (m < WMB) & (m * 4 < length)
    gpk: tl.constexpr = KBS // 4
    kernel_block = tl.load(Table + req * TS0 + m // gpk, valid, 0).to(tl.int64)
    slot = tl.load(Map + kernel_block * gpk + m % gpk, valid, -1)
    tl.store(Out + req * WMB + m, tl.where(valid, slot, -1), m < WMB)
