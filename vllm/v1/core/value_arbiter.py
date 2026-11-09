# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-feedback value arbitration between hot pages and the GPU index
cache (Active Host KV v2, the documented ownership rules).

The arbiter only moves T_idx, the boundary between unreferenced cached
history IDs (the prefix index cache) and the DONATED IDs that lend their
pages to the hot four-token slots. It never takes part in admission or in
any correctness check: the hot floor F, the growth order (evict cached
index -> shrink hot to F -> preempt) and the host-page reclaim of the ownership rules
stay in ``HistoryBlockPool`` and are counted here as forced adjustments.

Signals, evaluated once per window of ``window_steps`` host-KV steps:

- hot side: forced evictions of recently used slots
  (``max(0, miss - free - old)``, from the worker's device counters, layer
  0) and re-misses of those slots within the current or previous window
  (shadow records, :class:`HotShadowRef` / the device table).
- index side: cached IDs that lost their prefix (any reason) and later
  restorations that were finally admitted, completed, and did an index H2D
  for an ID whose hash was given up in the current or previous window
  (:class:`IndexShadow`).

Density is extra seconds paid afterwards per evicted byte. A side whose
evicted bytes are zero has no observation; with no observation on both
sides, missing counters, or arbitration disabled, T_idx does not move.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)

GIB = 1 << 30

# Worker device counters, one int64 each, summed over ranks. The class split
# (decode/prefill) follows the selection-union gather addendum, section 4.3; until
# selection-union gather lands only
# the class-free totals are filled.
HOT_MISSES = 0
HOT_FORCED_RECENT = 1
HOT_REFETCH = 2
HOT_CLAIM_EVICTED = 3
HOT_GATHERS = 4
HOT_MISSES_DECODE = 5
HOT_MISSES_PREFILL = 6
HOT_REFETCH_DECODE = 7
HOT_REFETCH_PREFILL = 8
HOT_FORCED_DECODE = 9
HOT_FORCED_PREFILL = 10
NUM_HOT_COUNTERS = 11


@dataclass(frozen=True)
class ArbiterConfig:
    """Arbitration settings. The four weights are placeholders until the value
    arbitration
    GPU measurement freezes them ."""

    enabled: bool = True
    """False pins T_idx at the shared hot region default split; nothing else changes."""
    window_steps: int = 32
    delta: float = 0.25
    """Dead band: one side must be (1 + delta) times denser to move T_idx."""
    q_fraction: float = 0.01
    """Q = max(1, floor(q_fraction * A)) IDs per window."""
    reverse_gap_windows: int = 2
    """Windows that must pass between two adjustments in opposite directions."""
    one_sided: Literal["follow", "hold"] = "follow"
    """One side observed, the other not: move toward the observed side when
    its density is positive (``follow``), or hold (``hold``)."""
    lag_steps: int = 2
    """Steps after a window's end to wait for its lagged worker counters."""
    # PLACEHOLDER weights, to be frozen by the value arbitration GPU measurement.
    bw_uva: float = 12.3 * GIB
    """Bytes/s of the UVA hot-slot gather, summed over ranks (placeholder:
    a reference four-GPU shared uplink assumption)."""
    bw_h2d: float = 12.3 * GIB
    """Bytes/s of the NEW_LOAD index H2D copy, summed over ranks (placeholder)."""
    w_async: float = 0.5
    """Share of index H2D time on the critical path (placeholder: copies
    overlap the forward pass)."""
    w_ttft: float = 1.0
    """Step times of TTFT each index refetch transaction costs (placeholder)."""

    def __post_init__(self) -> None:
        if (
            self.window_steps < 1
            or self.delta < 0
            or not 0 < self.q_fraction <= 1
            or self.reverse_gap_windows < 0
            or self.lag_steps < 0
            or self.one_sided not in ("follow", "hold")
            or min(self.bw_uva, self.bw_h2d) <= 0
            or min(self.w_async, self.w_ttft) < 0
        ):
            raise ValueError(f"Invalid value arbitration settings: {self}")

    @classmethod
    def from_settings(
        cls, enabled: bool, settings: Mapping[str, float | int | str] | None = None
    ) -> ArbiterConfig:
        """From ``long_context_kv.value_arbitration`` and
        ``value_arbiter_settings`` (measurement overrides)."""
        kinds = {
            "window_steps": int,
            "delta": float,
            "q_fraction": float,
            "reverse_gap_windows": int,
            "lag_steps": int,
            "bw_uva": float,
            "bw_h2d": float,
            "w_async": float,
            "w_ttft": float,
            "one_sided": str,
        }
        settings = dict(settings or {})
        unknown = set(settings) - set(kinds)
        if unknown:
            raise ValueError(f"Unknown value arbiter settings: {sorted(unknown)}")
        kwargs = {name: kinds[name](value) for name, value in settings.items()}
        return cls(enabled=enabled, **kwargs)

    def quantum(self, num_ids: int) -> int:
        return max(1, math.floor(self.q_fraction * num_ids))


@dataclass(frozen=True)
class PoolSnapshot:
    """The shared region as the scheduler sees it after this step's roles."""

    num_ids: int
    """A: usable history IDs (the null ID excluded)."""
    inuse: int
    """U: non-reclaimable IDs (referenced, ring, parked, held, draining)."""
    floor_blocks: int
    """ceil(F / k)."""
    cached: int
    """Unreferenced IDs that keep a prefix hash (the index cache)."""

    @property
    def upper(self) -> int:
        return max(0, self.num_ids - self.inuse - self.floor_blocks)


class ArbiterPool(Protocol):
    """What the arbiter needs from ``HistoryBlockPool`` (shared hot region)."""

    def arbiter_snapshot(self) -> PoolSnapshot: ...

    def set_index_cache_target(self, target: int) -> None: ...

    def donate_cached_for_value(self, num_blocks: int) -> int:
        """Give up to ``num_blocks`` LRU cached IDs to the hot region;
        return how many were given (each reported via ``on_index_evicted``)."""
        ...


class IndexShadow:
    """Hashes of index pages given up in the current and previous window.

    Entries older than the previous window are dropped when the window
    advances, so the table holds at most two windows of evictions and an old
    record never matches (no epoch wrap: Python integers).
    """

    def __init__(self) -> None:
        self._epoch: dict[Hashable, int] = {}
        self.window = 0

    def __len__(self) -> int:
        return len(self._epoch)

    def record(self, block_hash: Hashable) -> None:
        self._epoch[block_hash] = self.window

    def forget(self, block_hash: Hashable) -> None:
        self._epoch.pop(block_hash, None)

    def consume(self, block_hashes: Iterable[Hashable]) -> int:
        """Count and clear records within the current or previous window."""
        hits = 0
        for block_hash in block_hashes:
            epoch = self._epoch.pop(block_hash, None)
            if epoch is not None and epoch >= self.window - 1:
                hits += 1
        return hits

    def advance(self, window: int) -> None:
        if window < self.window:
            raise ValueError("Index shadow windows only move forward")
        self.window = window
        stale = [h for h, epoch in self._epoch.items() if epoch < window - 1]
        for block_hash in stale:
            del self._epoch[block_hash]


class HotShadowRef:
    """CPU reference of the layer-0 hot shadow table (the device table in
    ``host_kv/shadow.py`` must agree with it).

    One record per history group: invalid (-1) or the window epoch of the
    forced eviction. A miss on a valid record of the current or previous
    window counts one refetch; any miss clears its record. ``expire`` clears
    older records actively. Epochs are bounded below 2**31 - 1 and refuse to
    wrap.
    """

    MAX_EPOCH = (1 << 31) - 2

    def __init__(self, num_groups: int) -> None:
        self.epoch = [-1] * num_groups
        self.window = 0

    def set_window(self, window: int) -> None:
        if not self.window <= window <= self.MAX_EPOCH:
            raise ValueError("Hot shadow epoch must grow and stay below 2**31 - 1")
        self.window = window
        self.expire()

    def record(self, groups: Iterable[int]) -> None:
        for group in groups:
            self.epoch[group] = self.window

    def check(self, groups: Iterable[int]) -> int:
        hits = 0
        for group in groups:
            epoch = self.epoch[group]
            if epoch >= 0 and epoch >= self.window - 1:
                hits += 1
            self.epoch[group] = -1
        return hits

    def expire(self) -> None:
        for group, epoch in enumerate(self.epoch):
            if 0 <= epoch < self.window - 1:
                self.epoch[group] = -1

    def clear_groups(self, groups: Iterable[int]) -> None:
        for group in groups:
            self.epoch[group] = -1


@dataclass
class WindowStats:
    """One window's sums. Hot counts are layer-0 slots summed over ranks."""

    steps: set[int] = field(default_factory=set)
    """Host-KV steps the scheduler emitted in this window."""
    reported: set[int] = field(default_factory=set)
    """Steps whose worker counters arrived from every rank."""
    hot: list[int] = field(default_factory=lambda: [0] * NUM_HOT_COUNTERS)
    idx_evicted: int = 0
    idx_refetch_blocks: int = 0
    idx_refetch_txns: int = 0
    step_seconds: float = 0.0
    timed_steps: int = 0

    @property
    def complete(self) -> bool:
        return bool(self.steps) and self.steps <= self.reported


@dataclass(frozen=True)
class Decision:
    window: int
    action: Literal["raise", "lower", "hold"]
    reason: str
    t_idx: int
    hot_density: float | None
    idx_density: float | None
    value_donated: int = 0


class ValueArbiter:
    """The scheduler-thread controller. Call order per host-KV step:
    ``begin_step`` -> index events -> ``add_hot_counters`` (lagged) ->
    ``end_step``."""

    def __init__(
        self,
        config: ArbiterConfig,
        pool: ArbiterPool,
        *,
        default_target: int,
        slot_bytes: int,
        index_block_bytes: int,
        num_ranks: int = 1,
        hot_layers: int = 1,
        forced_counts: Callable[[], Mapping[str, int]] | None = None,
    ) -> None:
        snap = pool.arbiter_snapshot()
        self.config = config
        self.pool = pool
        self.q = config.quantum(snap.num_ids)
        self.default_target = default_target
        self.target = default_target
        """Value target; the applied T_idx is min(target, upper)."""
        self.t_idx = min(default_target, snap.upper)
        self.slot_bytes = slot_bytes * hot_layers
        """Bytes of one hot slot across layers (layer-0 counts scale up)."""
        self.index_block_bytes = index_block_bytes * num_ranks
        self.index_shadow = IndexShadow()
        self.window = 0
        self.windows: dict[int, WindowStats] = {0: WindowStats()}
        self.last_dir = 0
        self.last_move_window = -(1 << 30)
        self.value_moves = 0
        self.value_donated = 0
        self.forced_clamps = 0
        self.decisions: list[Decision] = []
        self.late_reports = 0
        self._step_window: dict[int, int] = {}
        self._partial: dict[int, tuple[int, list[int]]] = {}
        self.forced_counts = forced_counts
        """Pool-side forced cache evictions (floor, growth, host reclaim)."""
        pool.set_index_cache_target(self.t_idx)

    # ----- inputs ------------------------------------------------------------

    def begin_step(self, step: int, step_seconds: float | None = None) -> int:
        """Register host-KV step ``step``; return its window epoch (the
        worker's shadow epoch)."""
        window = self.windows[self.window]
        if len(window.steps) >= self.config.window_steps:
            self.window += 1
            self.windows[self.window] = window = WindowStats()
            self.index_shadow.advance(self.window)
        window.steps.add(step)
        self._step_window[step] = self.window
        if step_seconds is not None:
            window.step_seconds += step_seconds
            window.timed_steps += 1
        return self.window

    def add_hot_counters(self, step: int, counters: Iterable[int] | None) -> None:
        """Counters of ``step`` complete over all ranks; None marks them lost."""
        window = self._step_window.pop(step, None)
        stats = self.windows.get(window) if window is not None else None
        if stats is None:
            self.late_reports += 1
            return
        if counters is None:
            return
        values = list(counters)
        if len(values) != NUM_HOT_COUNTERS:
            return
        stats.reported.add(step)
        for i, value in enumerate(values):
            stats.hot[i] += int(value)

    def add_rank_counters(
        self, step: int, ranks: int, values: Iterable[int], all_ranks: int
    ) -> None:
        """Accumulate per-output partial reports until every rank reported."""
        old_ranks, old_values = self._partial.get(step, (0, [0] * NUM_HOT_COUNTERS))
        values = list(values)
        if ranks & old_ranks or len(values) != NUM_HOT_COUNTERS:
            # A duplicate or malformed report cannot stand in for another rank.
            self._partial.pop(step, None)
            self.add_hot_counters(step, None)
            return
        merged = (old_ranks | ranks, [a + b for a, b in zip(old_values, values)])
        if merged[0] == all_ranks:
            self._partial.pop(step, None)
            self.add_hot_counters(step, merged[1])
        else:
            self._partial[step] = merged

    def on_index_evicted(self, block_hash: Hashable | None) -> None:
        """A cached ID gave up its prefix (value, allocation, floor or host
        reclaim)."""
        self.windows[self.window].idx_evicted += 1
        if block_hash is not None:
            self.index_shadow.record(block_hash)

    def on_index_restored(self, h2d_hashes: Iterable[Hashable]) -> None:
        """A transaction was finally admitted, completed, and copied these
        IDs' index pages host -> GPU."""
        hits = self.index_shadow.consume(h2d_hashes)
        if hits:
            stats = self.windows[self.window]
            stats.idx_refetch_blocks += hits
            stats.idx_refetch_txns += 1

    # ----- control -----------------------------------------------------------

    def end_step(self, step: int) -> Decision | None:
        """Apply the forced clamp, then evaluate windows whose lagged
        counters had their chance to arrive."""
        snap = self.pool.arbiter_snapshot()
        decision = None
        for window in sorted(self.windows):
            stats = self.windows[window]
            if window >= self.window or (
                stats.steps and step < max(stats.steps) + self.config.lag_steps
            ):
                continue
            decision = self._evaluate(window, stats, snap)
            del self.windows[window]
            for s in stats.steps - stats.reported:
                self._step_window.pop(s, None)
                self._partial.pop(s, None)
        self._apply(snap)
        return decision

    def _density(self, stats: WindowStats) -> tuple[float | None, float | None]:
        c = self.config
        hot_evicted = stats.hot[HOT_FORCED_RECENT] * self.slot_bytes
        idx_evicted = stats.idx_evicted * self.index_block_bytes
        hot = idx = None
        if hot_evicted > 0:
            hot = stats.hot[HOT_REFETCH] * self.slot_bytes / c.bw_uva / hot_evicted
        if idx_evicted > 0:
            step_time = stats.step_seconds / max(stats.timed_steps, 1)
            cost = (
                stats.idx_refetch_blocks * self.index_block_bytes / c.bw_h2d * c.w_async
                + stats.idx_refetch_txns * step_time * c.w_ttft
            )
            idx = cost / idx_evicted
        return hot, idx

    def _evaluate(
        self, window: int, stats: WindowStats, snap: PoolSnapshot
    ) -> Decision:
        c = self.config
        hot, idx = self._density(stats)
        direction, reason = 0, "dead_band"
        if not c.enabled:
            reason = "disabled"
        elif not stats.complete:
            reason = "counters_missing" if stats.steps else "no_steps"
        elif hot is None and idx is None:
            reason = "no_observation"
        elif hot is None or idx is None:
            observed = hot if hot is not None else idx
            if c.one_sided == "hold":
                reason = "one_sided_hold"
            elif observed is not None and observed > 0:
                direction = -1 if idx is None else 1
                reason = "one_sided"
            else:
                reason = "no_cost"
        elif hot > (1 + c.delta) * idx:
            direction, reason = -1, "hot_denser"
        elif idx > (1 + c.delta) * hot:
            direction, reason = 1, "index_denser"
        if (
            direction
            and direction == -self.last_dir
            and window - self.last_move_window <= c.reverse_gap_windows
        ):
            direction, reason = 0, "reverse_gap"
        if direction:
            self.target = min(max(self.target + direction * self.q, 0), snap.upper)
            self.last_dir, self.last_move_window = direction, window
            self.value_moves += 1
        action = _ACTIONS[direction]
        decision = Decision(window, action, reason, self.target, hot, idx)
        self.decisions.append(decision)
        logger.info(
            "host KV arbitration window %d: hot density %s, index density %s "
            "-> %s (%s), T_idx %d, Q %d, forced clamps %d, hot refetch %d/%d, "
            "claim evicted %d, index refetch %d (%d txns)/%d, value donated %d, "
            "pool %s",
            window,
            _fmt(hot),
            _fmt(idx),
            action,
            reason,
            min(self.target, snap.upper),
            self.q,
            self.forced_clamps,
            stats.hot[HOT_REFETCH],
            stats.hot[HOT_FORCED_RECENT],
            stats.hot[HOT_CLAIM_EVICTED],
            stats.idx_refetch_blocks,
            stats.idx_refetch_txns,
            stats.idx_evicted,
            self.value_donated,
            dict(self.forced_counts()) if self.forced_counts else {},
        )
        # Value adjustment gives up at most Q cached IDs per evaluated window,
        # and only when this window moved T_idx toward the hot side.
        excess = snap.cached - min(self.target, snap.upper)
        if direction < 0 and excess > 0:
            given = self.pool.donate_cached_for_value(min(self.q, excess))
            self.value_donated += given
            self.decisions[-1] = Decision(
                window, action, reason, self.target, hot, idx, given
            )
        return self.decisions[-1]

    def _apply(self, snap: PoolSnapshot) -> None:
        target = self.target if self.config.enabled else self.default_target
        t_idx = min(target, snap.upper)
        if t_idx < target and t_idx < self.t_idx:
            # Growth (U or F) pushed the bound below the value target: forced,
            # never limited by Q.
            self.forced_clamps += 1
        if t_idx != self.t_idx:
            self.t_idx = t_idx
            self.pool.set_index_cache_target(t_idx)


_ACTIONS: dict[int, Literal["raise", "lower", "hold"]] = {
    1: "raise",
    -1: "lower",
    0: "hold",
}


def _fmt(value: float | None) -> str:
    return "none" if value is None else f"{value:.3e}"
