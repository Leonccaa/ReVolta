# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-grid metadata and byte-preserving UVA transfers.

All launches use one stream. Debug assertions are deliberately enabled by the
caller: invalid metadata must fail before a payload or residency mutation.
Host loads use ordinary vector tl.load, without a cache modifier.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def claim_writes(
    Req, Pos, Claims, N: tl.constexpr, R: tl.constexpr, T: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
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
    row = tl.program_id(0)
    x = tl.arange(0, B)
    req = tl.load(Req + row)
    pos = tl.load(Pos + row)
    valid = pos >= 0
    page = tl.load(Table + req * C + tl.maximum(pos, 0) // P, valid, 0).to(tl.int64)
    host_offset = page * HS0 + (pos % P) * HS2 + (x // D) * HS3 + x % D
    key = tl.load(K + row * KS0 + (x // D) * KS1 + x % D, valid & (x < H * D), 0)
    value = tl.load(V + row * VS0 + (x // D) * VS1 + x % D, valid & (x < H * D), 0)
    tl.store(Host + host_offset, key, valid & (x < H * D))
    tl.store(Host + host_offset + HS1, value, valid & (x < H * D))
    slot = tl.load(Map + req * G + tl.maximum(pos, 0) // 4, valid, -1)
    hot_offset = slot.to(tl.int64) * (8 * H * D) + (pos % 4) * H * D + x
    tl.store(Hot + hot_offset, key, valid & (slot >= 0) & (x < H * D))
    tl.store(Hot + hot_offset + 4 * H * D, value, valid & (slot >= 0) & (x < H * D))


@triton.jit
def commit_writes(Req, Pos, Ready, N: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    req = tl.load(Req + x, x < N, 0)
    pos = tl.load(Pos + x, x < N, -1)
    tl.store(Ready + req * T + pos, 1, (x < N) & (pos >= 0))


@triton.jit
def mark_selection(
    Indices,
    Req,
    Length,
    QueryPos,
    Ready,
    Wanted,
    R: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    W: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
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
    tl.atomic_or(
        Wanted + safe_req * G + tl.maximum(token, 0) // 4, 1, valid, sem="relaxed"
    )


@triton.jit
def mark_tail(Length, Ready, Wanted, R: tl.constexpr, T: tl.constexpr, G: tl.constexpr):
    req = tl.program_id(0)
    length = tl.load(Length + req)
    tl.device_assert((length >= 0) & (length <= T), "invalid host KV length")
    # A sliding 32-token suffix intersects nine groups when unaligned.
    x = tl.arange(0, 32)
    token = length - 32 + x
    valid = (token >= 0) & (token < T)
    ready = tl.load(Ready + req * T + token, valid, 1)
    tl.device_assert((~valid) | (ready == 1), "tail KV is not committed")
    group = tl.maximum(token, 0) // 4
    tl.atomic_or(Wanted + req * G + group, 1, valid, sem="relaxed")


@triton.jit
def collect_misses(
    Wanted, Map, Misses, Count, N: tl.constexpr, M: tl.constexpr, B: tl.constexpr
):
    gid = tl.program_id(0) * B + tl.arange(0, B)
    wanted = tl.load(Wanted + gid, gid < N, 0)
    slot = tl.load(Map + gid, gid < N, -1)
    missing = (gid < N) & (wanted != 0) & (slot < 0)
    rank = tl.cumsum(missing.to(tl.int32)) - 1
    count = tl.sum(missing.to(tl.int32), 0)
    start = tl.atomic_add(Count, count, sem="relaxed")
    tl.device_assert(start + count <= M, "host KV miss bound exceeded")
    tl.store(Misses + start + rank, gid, missing & (start + rank < M))


@triton.jit
def allocate_slots(
    Wanted,
    Map,
    Owner,
    Age,
    Clock,
    Misses,
    MissSlots,
    Count,
    S: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.arange(0, B)
    owner = tl.load(Owner + x, x < S, -1)
    age = tl.load(Age + x, x < S, 0)
    needed = tl.load(Wanted + tl.maximum(owner, 0), (x < S) & (owner >= 0), 0)
    clock = tl.load(Clock) + 1
    count = tl.load(Count)
    tl.device_assert((count >= 0) & (count <= M), "host KV miss bound exceeded")
    eligible = (x < S) & (needed == 0)
    tl.device_assert(
        tl.sum(eligible.to(tl.int32), 0) >= count, "host KV capacity exceeded"
    )
    # Empty slots first, then least recently selected step, then slot index.
    priority = tl.where(eligible, age * S + x, 0x7FFFFFFFFFFFFFFF)
    # A bounded device loop avoids a giant bitonic sort/shared-memory frame
    # at long-context capacities. This is a correctness prototype, not a
    # throughput-optimized allocator: O(misses * slots), one CTA per layer.
    for i in range(count):
        slot = (tl.min(priority, 0) % S).to(tl.int32)
        old = tl.load(Owner + slot)
        gid = tl.load(Misses + i)
        tl.store(Map + old, -1, old >= 0)
        tl.store(Map + gid, slot)
        tl.store(Owner + slot, gid)
        tl.store(MissSlots + i, slot)
        tl.store(Age + slot, clock)
        priority = tl.where(x == slot, 0x7FFFFFFFFFFFFFFF, priority)
    # Hits and misses are disjoint; hits were protected from eviction.
    tl.store(Age + x, clock, (x < S) & (needed != 0))
    tl.store(Clock, clock)


@triton.jit
def gather_groups(
    Host,
    Hot,
    Table,
    Length,
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
    i = tl.program_id(0)
    active = i < tl.load(Count)
    gid = tl.load(Misses + i, active, 0)
    slot = tl.load(MissSlots + i, active, 0)
    req, group = gid // G, gid % G
    length = tl.load(Length + req, active, 0)
    x = tl.arange(0, B)
    plane = x // (4 * H * D)
    within = x // (H * D) % 4
    head, dim = x // D % H, x % D
    token = group * 4 + within
    page = tl.load(Table + req * C + token // P, active, 0).to(tl.int64)
    ready = tl.load(Ready + req * T + token, active & (token < T), 0)
    offset = page * HS0 + plane * HS1 + token % P * HS2 + head * HS3 + dim
    # Byte loads for E4M3, typed loads for FP16. No conversion or .cv.
    payload = tl.load(
        Host + offset, active & (x < 8 * H * D) & (token < length) & (ready != 0), 0
    )
    tl.store(
        Hot + slot.to(tl.int64) * (8 * H * D) + x, payload, active & (x < 8 * H * D)
    )


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
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
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
    x = tl.program_id(0) * B + tl.arange(0, B)
    gid = tl.load(Owner + x, x < S, -1)
    length = tl.load(Length + tl.maximum(gid, 0) // G, (x < S) & (gid >= 0), 0)
    retire = (x < S) & (gid >= 0) & ((gid % G + 1) * 4 > length)
    tl.store(Map + gid, -1, retire)
    tl.store(Owner + x, -1, retire)
    tl.store(Age + x, 0, retire)


@triton.jit
def truncate_ready(Length, Ready, R: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    req, pos = x // T, x % T
    length = tl.load(Length + req, req < R, 0)
    tl.device_assert((length >= 0) & (length <= T), "invalid truncate length")
    tl.store(Ready + x, 0, (req < R) & (pos >= length))
