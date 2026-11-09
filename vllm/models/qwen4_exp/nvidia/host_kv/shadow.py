# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-0 hot shadow records and arbitration counters .

One byte per history group, as billed in ``budget.shared_history_memory_bytes``:
-1 is invalid, otherwise the window epoch modulo ``EPOCH_MOD`` of a forced
eviction of a recently used slot. Only the current and the previous window
are ever valid: advancing by one window clears the older epoch, advancing by
more clears every record. Matching therefore never relies on wrap-around.

The counters are device int64 values with fixed addresses (FULL graphs may
capture the gather kernels); the window epoch is a device scalar for the
same reason. Values are read back asynchronously, one step late.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.core.value_arbiter import (
    HOT_CLAIM_EVICTED,
    HOT_FORCED_RECENT,
    HOT_GATHERS,
    HOT_MISSES,
    HOT_REFETCH,
    NUM_HOT_COUNTERS,
)

EPOCH_MOD = 127


@triton.jit
def shadow_check_misses(
    Misses,
    MissCount,
    Shadow,
    Window,
    Counters,
    MOD: tl.constexpr,
    REFETCH: tl.constexpr,
    MISSES: tl.constexpr,
    GATHERS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    count = tl.load(MissCount)
    active = i < count
    gid = tl.load(Misses + i, active, 0).to(tl.int64)
    rec = tl.load(Shadow + gid, active, -1).to(tl.int32)
    cur = tl.load(Window)
    prev = (cur + MOD - 1) % MOD
    hit = active & (rec >= 0) & ((rec == cur) | (rec == prev))
    # Any miss consumes its record; only a fresh one counts as a refetch.
    tl.store(Shadow + gid, -1, active & (rec >= 0))
    hits = tl.sum(hit.to(tl.int64), 0)
    if hits > 0:
        # Most programs of the fixed grid see no record: skip their atomics.
        tl.atomic_add(Counters + REFETCH, hits, sem="relaxed")
    if tl.program_id(0) == 0:
        tl.atomic_add(Counters + MISSES, count.to(tl.int64), sem="relaxed")
        tl.atomic_add(Counters + GATHERS, 1, sem="relaxed")


@triton.jit
def shadow_record_victims(
    Victims,
    Owner,
    MissCount,
    VictimCounts,
    Shadow,
    Window,
    Counters,
    C: tl.constexpr,
    CB: tl.constexpr,
    FORCED: tl.constexpr,
    B: tl.constexpr,
):
    """Before allocation: victims hold free, then old, then recent slots."""
    count = tl.load(MissCount)
    if count > 0:
        tiles = tl.arange(0, CB)
        free = tl.sum(tl.load(VictimCounts + tiles, tiles < C, 0), 0)
        old = tl.sum(tl.load(VictimCounts + C + tiles, tiles < C, 0), 0)
        i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
        recent = (i < count) & (i >= free + old)
        slot = tl.load(Victims + i, recent, 0).to(tl.int64)
        owner = tl.load(Owner + slot, recent, -1).to(tl.int64)
        cur = tl.load(Window).to(tl.int8)
        tl.store(Shadow + tl.maximum(owner, 0), cur, recent & (owner >= 0))
        if tl.program_id(0) == 0:
            forced = tl.maximum(count - free - old, 0).to(tl.int64)
            tl.atomic_add(Counters + FORCED, forced, sem="relaxed")


@triton.jit
def shadow_expire(Shadow, Window, ALL: tl.constexpr, N: tl.constexpr, MOD: tl.constexpr,
                  B: tl.constexpr):  # fmt: skip
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    rec = tl.load(Shadow + i, i < N, -1).to(tl.int32)
    cur = tl.load(Window)
    keep = (rec == cur) | (rec == (cur + MOD - 1) % MOD)
    if ALL:
        keep = rec < -1
    tl.store(Shadow + i, -1, (i < N) & (rec >= 0) & ~keep)


@triton.jit
def shadow_clear_blocks(Blocks, Shadow, GPB: tl.constexpr, B: tl.constexpr):
    block = tl.load(Blocks + tl.program_id(0)).to(tl.int64)
    g = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    tl.store(Shadow + block * GPB + g, -1, g < GPB)


@triton.jit
def shadow_count_claims(
    Blocks, Owner, Counters, K: tl.constexpr, KB: tl.constexpr, CLAIMED: tl.constexpr
):
    """Hot slots of claimed IDs that still hold a group: forced evictions."""
    block = tl.load(Blocks + tl.program_id(0)).to(tl.int64)
    j = tl.arange(0, KB)
    owner = tl.load(Owner + block * K + j, j < K, -1)
    evicted = tl.sum((owner >= 0).to(tl.int64), 0)
    if evicted > 0:
        tl.atomic_add(Counters + CLAIMED, evicted, sem="relaxed")


class HotShadowTable:
    """Per-store shadow records, counters and their lagged readback."""

    def __init__(
        self, num_groups: int, groups_per_block: int, device: torch.device
    ) -> None:
        self.num_groups, self.groups_per_block = num_groups, groups_per_block
        self.device = device
        self.epoch = torch.full((num_groups,), -1, dtype=torch.int8, device=device)
        self.window = torch.zeros((), dtype=torch.int32, device=device)
        self.window_value = 0
        self.counters = torch.zeros(NUM_HOT_COUNTERS, dtype=torch.int64, device=device)
        pin = device.type == "cuda"
        self._host = [
            torch.zeros(NUM_HOT_COUNTERS, dtype=torch.int64, pin_memory=pin)
            for _ in range(2)
        ]
        self._events: list[torch.cuda.Event | None] = [None, None]
        self._steps: list[int | None] = [None, None]
        self._turn = 0
        if pin:
            # Compile both expiry variants now (no-ops on the empty table):
            # the first window advance otherwise JIT-compiles inside a decode
            # step (about 250 ms on a cold Triton cache).
            for expire_all in (False, True):
                shadow_expire[(triton.cdiv(num_groups, 1024),)](
                    self.epoch, self.window, expire_all, num_groups, EPOCH_MOD, 1024
                )

    def device_bytes(self) -> dict[str, int]:
        return {
            "shadow": self.epoch.numel(),
            "v2_counters": self.counters.numel() * 8 + 4,
        }

    def reset(self) -> None:
        """After profiling/capture dummy runs: no record, no count."""
        self.epoch.fill_(-1)
        self.counters.zero_()

    def set_window(self, window: int) -> None:
        """Advance on the forward stream, outside graphs, before the gathers."""
        if window < self.window_value:
            raise ValueError("Hot shadow windows only move forward")
        if window == self.window_value:
            return
        jump = window - self.window_value
        self.window_value = window
        self.window.fill_(window % EPOCH_MOD)
        n = self.num_groups
        shadow_expire[(triton.cdiv(n, 1024),)](
            self.epoch, self.window, jump > 1, n, EPOCH_MOD, 1024
        )

    def clear_blocks(self, ids: torch.Tensor) -> None:
        """begin/claim reuse of history IDs."""
        if ids.numel():
            gpb = self.groups_per_block
            shadow_clear_blocks[(ids.numel(), triton.cdiv(gpb, 256))](
                ids, self.epoch, gpb, 256
            )

    def count_claims(self, ids: torch.Tensor, slot_owner: torch.Tensor, k: int) -> None:
        """Call with layer 0's owners before the claim kernel unmaps them."""
        if ids.numel():
            shadow_count_claims[(ids.numel(),)](
                ids, slot_owner, self.counters, k, triton.next_power_of_2(k),
                HOT_CLAIM_EVICTED,
            )  # fmt: skip

    def check_misses(self, store, grid: int) -> None:
        shadow_check_misses[(grid,)](
            store.miss_groups, store.miss_count, self.epoch, self.window,
            self.counters, EPOCH_MOD, HOT_REFETCH, HOT_MISSES, HOT_GATHERS, 256,
        )  # fmt: skip

    def record_victims(self, store, slot_owner: torch.Tensor, grid: int) -> None:
        shadow_record_victims[(grid,)](
            store.victim_slots, slot_owner, store.miss_count, store.victim_counts,
            self.epoch, self.window, self.counters, store.victim_tiles,
            triton.next_power_of_2(store.victim_tiles), HOT_FORCED_RECENT, 256,
        )  # fmt: skip

    def harvest(self, step: int) -> tuple[int, list[int]] | None:
        """Queue this step's counters and return the previous step's.

        Call once per host-KV step after its forward work was queued. The
        copy and the reset are stream ordered; reading waits only for the
        previous step's copy (normally already complete).
        """
        turn = self._turn
        self._host[turn].copy_(self.counters, non_blocking=True)
        self.counters.zero_()
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record()
            self._events[turn] = event
        self._steps[turn] = step
        self._turn = other = 1 - turn
        prev = self._steps[other]
        if prev is None:
            return None
        event = self._events[other]
        if event is not None:
            event.synchronize()
        self._steps[other] = None
        return prev, self._host[other].tolist()
