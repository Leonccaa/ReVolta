# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-grid metadata and byte-preserving UVA transfers.

All launches use one stream. Debug assertions are deliberately enabled by the
caller: invalid metadata must fail before a payload or residency mutation.
Host loads use ordinary vector tl.load, without a cache modifier.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def host_offsets(
    page,
    plane,
    token,
    head,
    dim,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    P: tl.constexpr,
):
    # Each independent product must be wide before addition to the page term.
    return (
        tl.cast(page, tl.int64) * HS0
        + tl.cast(plane, tl.int64) * HS1
        + (tl.cast(token, tl.int64) % P) * HS2
        + tl.cast(head, tl.int64) * HS3
        + dim
    )


@triton.jit
def claim_writes(
    Req, Pos, Claims, N: tl.constexpr, R: tl.constexpr, T: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    req = tl.load(Req + i, i < N, 0)
    pos = tl.load(Pos + i, i < N, -1)
    active = (i < N) & (pos != -1)
    valid = (req >= 0) & (req < R) & (pos >= 0) & (pos < T)
    tl.device_assert((~active) | valid, "invalid host KV write coordinate")
    old = tl.atomic_xchg(
        Claims + req * T + pos, i + 1, sem="relaxed", mask=active & valid
    )
    tl.device_assert((~active) | (old == 0), "duplicate host KV write coordinate")


@triton.jit
def write_rows(
    K,
    V,
    Req,
    Pos,
    Table,
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
    P: tl.constexpr,
    C: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, B).to(tl.int64)
    req = tl.load(Req + row).to(tl.int64)
    pos = tl.load(Pos + row).to(tl.int64)
    valid = pos >= 0
    page = tl.load(Table + req * C + tl.maximum(pos, 0) // P, valid, 0).to(tl.int64)
    host_offset = host_offsets(page, 0, pos, x // D, x % D, HS0, HS1, HS2, HS3, P)
    key = tl.load(K + row * KS0 + (x // D) * KS1 + x % D, valid & (x < H * D), 0)
    value = tl.load(V + row * VS0 + (x // D) * VS1 + x % D, valid & (x < H * D), 0)
    tl.store(Host + host_offset, key, valid & (x < H * D))
    tl.store(Host + host_offset + HS1, value, valid & (x < H * D))
    slot = tl.load(Map + req * G + tl.maximum(pos, 0) // 4, valid, -1)
    hot_offset = slot.to(tl.int64) * (8 * H * D) + (pos % 4) * H * D + x
    tl.store(Hot + hot_offset, key, valid & (slot >= 0) & (x < H * D))
    tl.store(Hot + hot_offset + 4 * H * D, value, valid & (slot >= 0) & (x < H * D))


@triton.jit
def commit_writes(
    Req, Pos, Ready, Claims, N: tl.constexpr, T: tl.constexpr, B: tl.constexpr
):
    x = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    req = tl.load(Req + x, x < N, 0)
    pos = tl.load(Pos + x, x < N, -1)
    tl.store(Ready + req * T + pos, 1, (x < N) & (pos >= 0))
    # Only addresses claimed by this batch need clearing for the next write.
    tl.store(Claims + req * T + pos, 0, (x < N) & (pos >= 0))


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
def clear_wanted(Wanted, Touched, Count, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    active = i < tl.load(Count)
    gid = tl.load(Touched + i, active, 0)
    tl.store(Wanted + gid, 0, active)


@triton.jit
def begin_gather(WantedCount, MissCount, Clock):
    tl.store(WantedCount, 0)
    tl.store(MissCount, 0)
    tl.store(Clock, tl.load(Clock) + 1)


@triton.jit
def mark_selection(
    Indices,
    Req,
    Length,
    QueryPos,
    Ready,
    Wanted,
    Touched,
    Count,
    R: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    W: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    QSA_COLUMNS: tl.constexpr,
    QSA_WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    req = tl.load(Req + row)
    tl.device_assert((req >= 0) & (req < R), "invalid host KV request")
    safe_req = tl.minimum(tl.maximum(req, 0), R - 1)
    length = tl.load(Length + safe_req)
    qpos = tl.load(QueryPos + row)
    token = tl.load(Indices + row * W + col, col < W, -1)
    tl.device_assert((token >= -1) & (token < T), "invalid host KV selection")
    # Causal and padding masks are applied in logical coordinates, before
    # remapping. Attention still receives every original selection column.
    valid = (col < W) & (token >= 0) & (token < T) & (token < length) & (token <= qpos)
    ready = tl.load(Ready + safe_req * T + tl.maximum(token, 0), valid, 1)
    tl.device_assert((~valid) | (ready == 1), "selected KV is not committed")
    if QSA_COLUMNS:
        # QSA emits quartets (pairs after DCP2 interleave 1/2 localization),
        # followed by one open-group suffix and padding. Verify the contract
        # before using the tighter capacity bound.
        first_col = col // QSA_COLUMNS * QSA_COLUMNS
        first = tl.load(Indices + row * W + first_col, col < W, -1)
        tl.device_assert(
            (col >= W) | (token < 0) | ((first >= 0) & (token // 4 == first // 4)),
            "host KV QSA group layout exceeded",
        )
        tl.device_assert(
            (col >= W) | (col < QSA_WIDTH) | (token < 0),
            "host KV QSA group layout exceeded",
        )
        # Claim once per group while checking ready/causality for every
        # column. Remapping still preserves every original attention column.
        groups = tl.reshape(
            tl.where(valid, token // 4, -1), (B // QSA_COLUMNS, QSA_COLUMNS)
        )
        group = tl.max(groups, 1)
        append_wanted(
            safe_req * G + tl.maximum(group, 0), group >= 0, Wanted, Touched, Count, M
        )
    else:
        append_wanted(
            safe_req * G + tl.maximum(token, 0) // 4, valid, Wanted, Touched, Count, M
        )


@triton.jit
def mark_tail(
    Length,
    Ready,
    Wanted,
    Touched,
    Count,
    R: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    length = tl.load(Length + req)
    tl.device_assert((length >= 0) & (length <= T), "invalid host KV length")
    # A sliding 32-token suffix intersects nine groups when unaligned.
    x = tl.arange(0, 32)
    token = length - 32 + x
    valid = (token >= 0) & (token < T)
    ready = tl.load(Ready + req * T + token, valid, 1)
    tl.device_assert((~valid) | (ready == 1), "tail KV is not committed")
    group = tl.maximum(token, 0) // 4
    append_wanted(req * G + group, valid, Wanted, Touched, Count, M)


@triton.jit
def collect_misses(
    Touched,
    WantedCount,
    Map,
    Age,
    Clock,
    Misses,
    Count,
    M: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    active = i < tl.load(WantedCount)
    gid = tl.load(Touched + i, active, 0)
    slot = tl.load(Map + gid, active, -1)
    missing = active & (slot < 0)
    tl.store(Age + tl.maximum(slot, 0), tl.load(Clock), active & (slot >= 0))
    rank = tl.cumsum(missing.to(tl.int32)) - 1
    count = tl.sum(missing.to(tl.int32), 0)
    if count > 0:
        start = tl.atomic_add(Count, count, sem="relaxed")
        tl.device_assert(start + count <= M, "host KV miss bound exceeded")
        tl.store(Misses + start + rank, gid, missing & (start + rank < M))


@triton.jit
def victim_bucket(x, Wanted, Owner, Age, Clock, S: tl.constexpr):
    owner = tl.load(Owner + x, x < S, -1)
    age = tl.load(Age + x, x < S, 0)
    needed = tl.load(Wanted + tl.maximum(owner, 0), (x < S) & (owner >= 0), 0)
    # Free, older than eight calls, recently touched resident. Wanted
    # always wins over age, including current hits whose age was just updated.
    # Owner -2 is a reserved shared-region slot (an in-use ID page): never.
    bucket = tl.where(owner < 0, 0, tl.where(age < tl.load(Clock) - 8, 1, 2))
    return tl.where((x < S) & (needed == 0) & (owner != -2), bucket, 3)


@triton.jit
def count_victim_buckets(
    Wanted,
    Owner,
    Age,
    Clock,
    Count,
    Counts,
    Hand,
    S: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
):
    if tl.load(Count) > 0:
        tile = tl.program_id(0)
        chunk: tl.constexpr = triton.cdiv(S, C * B) * B
        free, old, recent = 0, 0, 0
        for base in range(
            tile.to(tl.int64) * chunk, (tile + 1).to(tl.int64) * chunk, B
        ):
            offset = base + tl.arange(0, B)
            x = offset + tl.load(Hand)
            x = tl.where(x >= S, x - S, x)
            bucket = victim_bucket(x, Wanted, Owner, Age, Clock, S)
            bucket = tl.where(offset < S, bucket, 3)
            free += tl.sum((bucket == 0).to(tl.int32), 0)
            old += tl.sum((bucket == 1).to(tl.int32), 0)
            recent += tl.sum((bucket == 2).to(tl.int32), 0)
        tl.store(Counts + tile, free)
        tl.store(Counts + C + tile, old)
        tl.store(Counts + 2 * C + tile, recent)


@triton.jit
def select_bucket_victims(
    Wanted,
    Owner,
    Age,
    Clock,
    Count,
    Counts,
    Victims,
    Hand,
    S: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    CB: tl.constexpr,
):
    count = tl.load(Count)
    if count > 0:
        tile = tl.program_id(0)
        tiles = tl.arange(0, CB)
        free = tl.load(Counts + tiles, tiles < C, 0)
        old = tl.load(Counts + C + tiles, tiles < C, 0)
        recent = tl.load(Counts + 2 * C + tiles, tiles < C, 0)
        free_start = tl.sum(tl.where(tiles < tile, free, 0), 0)
        old_start = tl.sum(free, 0) + tl.sum(tl.where(tiles < tile, old, 0), 0)
        recent_start = tl.sum(free + old, 0) + tl.sum(
            tl.where(tiles < tile, recent, 0), 0
        )
        if tl.sum(free + old + recent, 0) < count:
            # No CTA writes victims in this case. Clear the downstream count
            # even when assertions are disabled, so stale scratch is unusable.
            if tile == 0:
                tl.store(Count, 0)
            tl.device_assert(False, "host KV capacity exceeded")
        else:
            chunk: tl.constexpr = triton.cdiv(S, C * B) * B
            for base in range(
                tile.to(tl.int64) * chunk, (tile + 1).to(tl.int64) * chunk, B
            ):
                offset = base + tl.arange(0, B)
                x = offset + tl.load(Hand)
                x = tl.where(x >= S, x - S, x)
                bucket = victim_bucket(x, Wanted, Owner, Age, Clock, S)
                bucket = tl.where(offset < S, bucket, 3)
                for b in tl.static_range(3):
                    start = tl.where(
                        b == 0, free_start, tl.where(b == 1, old_start, recent_start)
                    )
                    eligible = bucket == b
                    rank = start + tl.cumsum(eligible.to(tl.int32)) - 1
                    tl.store(Victims + rank, x.to(tl.int32), eligible & (rank < count))
                free_start += tl.sum((bucket == 0).to(tl.int32), 0)
                old_start += tl.sum((bucket == 1).to(tl.int32), 0)
                recent_start += tl.sum((bucket == 2).to(tl.int32), 0)


@triton.jit
def prepare_slot_priorities(
    Wanted,
    Owner,
    Age,
    Clock,
    Priority,
    S: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    owner = tl.load(Owner + x, x < S, -1)
    age = tl.load(Age + x, x < S, 0)
    needed = tl.load(Wanted + tl.maximum(owner, 0), (x < S) & (owner >= 0), 0)
    clock = tl.load(Clock)
    priority = tl.where(needed == 0, age, 0x7FFFFFFFFFFFFFFF)
    tl.store(Priority + x, priority, x < S)
    tl.store(Age + x, clock, (x < S) & (needed != 0))


@triton.jit
def allocate_slots(
    Map,
    Owner,
    Age,
    Clock,
    Misses,
    MissSlots,
    Count,
    Victims,
    SortedPriority,
    Hand,
    M: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    EXACT: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    count = tl.load(Count)
    tl.device_assert((count >= 0) & (count <= M), "host KV miss bound exceeded")
    active = i < count
    slot = tl.load(Victims + i, active, 0)
    if EXACT:
        priority = tl.load(SortedPriority + i, active, 0)
        tl.device_assert(
            (~active) | (priority != 0x7FFFFFFFFFFFFFFF), "host KV capacity exceeded"
        )
    old = tl.load(Owner + slot, active, -1)
    gid = tl.load(Misses + i, active, 0)
    tl.store(Map + old, -1, active & (old >= 0))
    tl.store(Map + gid, slot.to(tl.int32), active)
    tl.store(Owner + slot, gid, active)
    tl.store(MissSlots + i, slot.to(tl.int32), active)
    tl.store(Age + slot, tl.load(Clock), active)
    # Advance only after every selector has consumed the old hand.
    if not EXACT and ((tl.program_id(0) == 0) & (count > 0)):
        last = tl.load(Victims + count - 1)
        tl.store(Hand, tl.where(last + 1 == S, 0, last + 1))


@triton.jit
def gather_groups(
    Host,
    Hot,
    Table,
    Ready,
    Misses,
    MissSlots,
    Count,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.arange(0, B).to(tl.int64)
    plane = x // (4 * H * D)
    within = x // (H * D) % 4
    head, dim = x // D % H, x % D
    # Fixed worker grid, with no per-capacity empty CTA. Count stays on device.
    for i in range(tl.program_id(0), tl.load(Count), tl.num_programs(0)):
        gid = tl.load(Misses + i).to(tl.int64)
        slot = tl.load(MissSlots + i).to(tl.int64)
        req, group = gid // G, gid % G
        token = group * 4 + within
        page = tl.load(Table + req * C + token // P).to(tl.int64)
        ready = tl.load(Ready + req * T + token, token < T, 0)
        offset = host_offsets(page, plane, token, head, dim, HS0, HS1, HS2, HS3, P)
        # Copy every committed row, independently of current visibility.
        payload = tl.load(
            Host + offset, (x < 8 * H * D) & (token < T) & (ready != 0), 0
        )
        tl.store(Hot + slot * (8 * H * D) + x, payload, x < 8 * H * D)


@triton.jit
def remap_selection(
    Indices,
    Req,
    Length,
    QueryPos,
    Map,
    Out,
    T: tl.constexpr,
    G: tl.constexpr,
    W: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    req = tl.load(Req + row)
    length = tl.load(Length + req)
    qpos = tl.load(QueryPos + row)
    token = tl.load(Indices + row * W + col, col < W, -1)
    valid = (col < W) & (token >= 0) & (token < length) & (token <= qpos)
    slot = tl.load(Map + req * G + tl.maximum(token, 0) // 4, valid, -1)
    tl.device_assert((~valid) | (slot >= 0), "selected KV has no hot slot")
    tl.store(Out + row * W + col, tl.where(valid, slot * 4 + token % 4, -1), col < W)


@triton.jit
def retire_slots(
    Length, Map, Owner, Age, G: tl.constexpr, S: tl.constexpr, B: tl.constexpr
):
    x = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    gid = tl.load(Owner + x, x < S, -1)
    length = tl.load(Length + tl.maximum(gid, 0) // G, (x < S) & (gid >= 0), 0)
    retire = (x < S) & (gid >= 0) & ((gid.to(tl.int64) % G + 1) * 4 > length)
    tl.store(Map + gid, -1, retire)
    tl.store(Owner + x, -1, retire)
    tl.store(Age + x, 0, retire)


@triton.jit
def truncate_ready(Length, Ready, R: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    req, pos = x // T, x % T
    length = tl.load(Length + req, req < R, 0)
    tl.device_assert((length >= 0) & (length <= T), "invalid truncate length")
    tl.store(Ready + x, 0, (req < R) & (pos >= length))
