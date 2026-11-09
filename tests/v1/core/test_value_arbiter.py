# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""value arbitration bounded-feedback value arbitration , CPU only."""

import itertools
from typing import Any

import pytest

from vllm.v1.core.value_arbiter import (
    HOT_FORCED_RECENT,
    HOT_REFETCH,
    NUM_HOT_COUNTERS,
    ArbiterConfig,
    HotShadowRef,
    IndexShadow,
    PoolSnapshot,
    ValueArbiter,
)

W = 32


class FakePool:
    """Cached/donated/in-use accounting of the shared region."""

    def __init__(self, num_ids=1000, inuse=100, floor=50, cached=300):
        self.num_ids, self.inuse, self.floor, self.cached = (
            num_ids,
            inuse,
            floor,
            cached,
        )
        self.target = None
        self.arbiter: Any = None
        self.donations: list[int] = []
        self.next_hash = itertools.count()

    def arbiter_snapshot(self):
        return PoolSnapshot(self.num_ids, self.inuse, self.floor, self.cached)

    def set_index_cache_target(self, target):
        self.target = target

    def donate_cached_for_value(self, num_blocks):
        given = min(num_blocks, self.cached)
        self.cached -= given
        self.donations.append(given)
        for _ in range(given):
            self.arbiter.on_index_evicted(("value", next(self.next_hash)))
        return given


def make(config=None, pool=None, default=200, **kwargs):
    pool = pool or FakePool(**kwargs)
    config = config or ArbiterConfig(bw_uva=1.0, bw_h2d=1.0, w_async=1.0, w_ttft=0.0)
    arb = ValueArbiter(
        config, pool, default_target=default, slot_bytes=1, index_block_bytes=1
    )
    pool.arbiter = arb
    return arb, pool


class Feeder:
    """Drives one arbiter with lagged hot counters and index events."""

    def __init__(self, arb, pool, lag=1):
        self.arb, self.pool, self.lag = arb, pool, lag
        self.step = 0
        self.pending: list[tuple[int, list[int] | None]] = []
        self.decisions = []

    def window(
        self,
        hot_forced=0,
        hot_refetch=0,
        idx_evicted=0,
        idx_refetch=0,
        drop_step=None,
    ):
        """One window; totals are spread over its first step."""
        for i in range(W):
            self.step += 1
            self.arb.begin_step(self.step, 0.01)
            counters = [0] * NUM_HOT_COUNTERS
            hashes = []
            if i == 0:
                counters[HOT_FORCED_RECENT] = hot_forced
                counters[HOT_REFETCH] = hot_refetch
                for _ in range(idx_evicted):
                    h = ("evict", next(self.pool.next_hash))
                    hashes.append(h)
                    self.arb.on_index_evicted(h)
                if idx_refetch:
                    self.arb.on_index_restored(hashes[:idx_refetch])
            self.pending.append((self.step, None if i == drop_step else counters))
            while self.pending and self.pending[0][0] <= self.step - self.lag:
                step, values = self.pending.pop(0)
                if values is not None:
                    self.arb.add_hot_counters(step, values)
            d = self.arb.end_step(self.step)
            if d is not None:
                self.decisions.append(d)
        return self.decisions[-1] if self.decisions else None

    def flush(self):
        # Opening the next window closes this one; lag steps then evaluate.
        return self.window()


def test_quantum_and_bounds():
    arb, pool = make(num_ids=1000)
    assert arb.q == 10
    arb2, _ = make(num_ids=50)
    assert arb2.q == 1
    # upper = A - U - ceil(F/k) = 1000 - 100 - 50
    assert pool.arbiter_snapshot().upper == 850
    assert arb.t_idx == 200


def test_zero_denominators_hold():
    arb, pool = make()
    f = Feeder(arb, pool)
    f.window()  # nothing evicted on either side
    f.window(hot_refetch=7, idx_evicted=0)
    f.window()
    reasons = [d.reason for d in f.decisions]
    assert reasons[:2] == ["no_observation", "no_observation"]
    assert pool.target == 200 and arb.value_moves == 0


def test_lingering_refetch_without_eviction_is_not_a_density():
    # Hot refetches from the previous window but no forced eviction now: the
    # hot side has no observation; the index side has a positive density.
    arb, pool = make()
    f = Feeder(arb, pool)
    f.window(hot_forced=0, hot_refetch=50, idx_evicted=10, idx_refetch=5)
    f.window()
    d = f.decisions[0]
    assert d.hot_density is None and d.action == "raise"


def test_missing_counters_hold():
    arb, pool = make()
    f = Feeder(arb, pool)
    f.window(hot_forced=10, hot_refetch=10, idx_evicted=10, drop_step=5)
    f.window()
    assert f.decisions[0].reason == "counters_missing"
    assert arb.value_moves == 0 and pool.target == 200


def test_dead_band_and_direction():
    arb, pool = make()
    f = Feeder(arb, pool)
    # Hot refetches 10/100, index 9/100: inside (1 + delta).
    f.window(hot_forced=100, hot_refetch=10, idx_evicted=100, idx_refetch=9)
    f.window(hot_forced=100, hot_refetch=50, idx_evicted=100, idx_refetch=10)
    f.flush()
    assert [d.action for d in f.decisions[:2]] == ["hold", "lower"]
    assert pool.donations == [10]
    assert arb.target == 190


def test_value_migration_per_window_is_at_most_q():
    # Large hot pressure for many windows, a big cache above the target.
    arb, pool = make(cached=600, default=600)
    f = Feeder(arb, pool)
    for _ in range(20):
        f.window(hot_forced=100, hot_refetch=100, idx_evicted=1, idx_refetch=0)
    assert pool.donations and max(pool.donations) <= arb.q
    assert all(d.value_donated <= arb.q for d in f.decisions)
    assert arb.target == 600 - arb.q * len(
        [d for d in f.decisions if d.action == "lower"]
    )


def test_forced_clamp_is_separate_and_unbounded():
    arb, pool = make(default=800)
    f = Feeder(arb, pool)
    f.window()
    pool.inuse = 900  # growth: upper drops to 50
    f.window()
    assert arb.t_idx == 50 and pool.target == 50
    assert arb.forced_clamps >= 1 and arb.value_moves == 0
    assert arb.q < 800 - 50  # moved by more than Q, counted as forced


def test_reverse_gap():
    arb, pool = make()
    f = Feeder(arb, pool)
    hot = dict(hot_forced=100, hot_refetch=80, idx_evicted=100, idx_refetch=1)
    idx = dict(hot_forced=100, hot_refetch=1, idx_evicted=100, idx_refetch=80)
    f.window(**hot)
    f.window(**idx)
    f.window(**idx)
    f.window(**idx)
    f.flush()
    acts = [d.action for d in f.decisions]
    assert acts[0] == "lower"
    assert acts[1:3] == ["hold", "hold"]
    assert [d.reason for d in f.decisions[1:3]] == ["reverse_gap"] * 2
    assert acts[3] == "raise"


def test_disabled_keeps_default_split():
    config = ArbiterConfig(enabled=False, bw_uva=1, bw_h2d=1, w_ttft=0)
    arb, pool = make(config=config)
    f = Feeder(arb, pool)
    for _ in range(4):
        f.window(hot_forced=100, hot_refetch=100, idx_evicted=1)
    assert pool.target == 200 and not pool.donations
    assert {d.reason for d in f.decisions} == {"disabled"}


def test_one_sided_hold_option():
    config = ArbiterConfig(one_sided="hold", bw_uva=1, bw_h2d=1, w_ttft=0)
    arb, pool = make(config=config)
    f = Feeder(arb, pool)
    f.window(hot_forced=10, hot_refetch=10)
    f.flush()
    assert f.decisions[0].reason == "one_sided_hold"


def test_index_density_uses_ttft_weight():
    config = ArbiterConfig(bw_uva=1, bw_h2d=1e30, w_async=1.0, w_ttft=1.0)
    arb, pool = make(config=config)
    f = Feeder(arb, pool)
    f.window(hot_forced=100, hot_refetch=0, idx_evicted=10, idx_refetch=3)
    f.flush()
    d = f.decisions[0]
    # One transaction x 10 ms step time over 10 evicted bytes.
    assert d.idx_density == pytest.approx(0.001)
    assert d.action == "raise"


# ----- convergence and oscillation under synthetic loads --------------------


def run_load(density_fn, windows=60, **kwargs):
    """``density_fn(w, t_idx) -> (hot_refetch, idx_refetch)`` out of 100."""
    arb, pool = make(**kwargs)
    pool.cached = 10**6  # value donations never run dry
    f = Feeder(arb, pool)
    trace = []
    for w in range(windows):
        hot, idx = density_fn(w, arb.t_idx)
        f.window(hot_forced=100, hot_refetch=hot, idx_evicted=100, idx_refetch=idx)
        trace.append(arb.t_idx)
    return arb, f, trace


def reversals(decisions):
    moves = [d.action for d in decisions if d.action != "hold"]
    return sum(a != b for a, b in zip(moves, moves[1:]))


def stable(w, t):
    # Hot value falls as the hot region grows (T_idx down), index value falls
    # as the cache grows; equilibrium near T_idx = 400.
    hot = max(0, min(100, 50 + (t - 400) // 10))
    idx = max(0, min(100, 50 - (t - 400) // 10))
    return hot, idx


def test_stable_load_converges_without_oscillation():
    arb, f, trace = run_load(stable, windows=80, default=200)
    tail = trace[-20:]
    assert max(tail) - min(tail) <= 2 * arb.q
    assert 330 <= tail[-1] <= 470
    # Approach is monotone; at the dead band it may step at most a few times.
    assert reversals(f.decisions) <= 4


def test_alternating_load_reversals_are_spaced():
    def alternating(w, t):
        return (90, 5) if w % 2 == 0 else (5, 90)

    arb, f, trace = run_load(alternating, windows=60)
    moves = [(d.window, d.action) for d in f.decisions if d.action != "hold"]
    for (w0, a0), (w1, a1) in zip(moves, moves[1:]):
        if a0 != a1:
            assert w1 - w0 > arb.config.reverse_gap_windows
    assert max(abs(a - b) for a, b in zip(trace, trace[1:])) <= arb.q


def test_burst_load_is_bounded_and_recovers():
    def burst(w, t):
        if 20 <= w < 25:
            return 95, 5
        return stable(w, t)

    arb, f, trace = run_load(burst, windows=90, default=400)
    # Five burst windows move T_idx by at most 5 Q (plus lag of one window).
    assert min(trace[18:30]) >= 400 - 7 * arb.q
    tail = trace[-15:]
    assert 330 <= tail[-1] <= 470


# ----- shadow records -------------------------------------------------------


def test_hot_shadow_counts_current_and_previous_window_only():
    ref = HotShadowRef(16)
    ref.set_window(3)
    ref.record([1, 2, 3])
    ref.set_window(4)
    assert ref.check([1]) == 1  # previous window
    assert ref.check([1]) == 0  # cleared on refetch
    ref.set_window(5)
    assert ref.epoch[2] == -1 and ref.epoch[3] == -1  # actively expired
    assert ref.check([2, 3]) == 0


def test_hot_shadow_never_wraps():
    ref = HotShadowRef(4)
    ref.set_window(7)
    ref.record([0])
    # A huge jump never aliases back to the record's epoch.
    ref.window = 7 + (1 << 31)  # bypass expiry: comparison alone must reject
    assert ref.check([0]) == 0
    ref = HotShadowRef(4)
    with pytest.raises(ValueError):
        ref.set_window(HotShadowRef.MAX_EPOCH + 1)
    with pytest.raises(ValueError):
        ref.set_window(-1)


def test_hot_shadow_clear_on_reuse():
    ref = HotShadowRef(8)
    ref.set_window(1)
    ref.record([4, 5])
    ref.clear_groups([4])
    assert ref.check([4, 5]) == 1


def test_index_shadow_windows():
    shadow = IndexShadow()
    shadow.record("a")
    shadow.advance(1)
    shadow.record("b")
    assert shadow.consume(["a"]) == 1
    shadow.record("a")
    shadow.advance(3)
    assert len(shadow) == 0
    assert shadow.consume(["a", "b"]) == 0
    with pytest.raises(ValueError):
        shadow.advance(2)


def test_index_restore_counts_only_shadowed_hashes():
    arb, pool = make()
    arb.begin_step(1)
    arb.on_index_evicted("x")
    arb.on_index_restored(["y"])
    assert arb.windows[0].idx_refetch_txns == 0
    arb.on_index_restored(["x", "x"])
    assert arb.windows[0].idx_refetch_blocks == 1
    assert arb.windows[0].idx_refetch_txns == 1


def test_late_counters_are_dropped():
    arb, pool = make()
    f = Feeder(arb, pool, lag=5)  # beyond lag_steps=2
    f.window(hot_forced=100, hot_refetch=100, idx_evicted=10)
    f.window()
    assert f.decisions[0].reason == "counters_missing"
    assert arb.late_reports > 0


def test_config_from_settings():
    config = ArbiterConfig.from_settings(False, {"w_ttft": "2.5", "window_steps": 8})
    assert not config.enabled and config.w_ttft == 2.5 and config.window_steps == 8
    with pytest.raises(ValueError):
        ArbiterConfig.from_settings(True, {"bogus": 1})
    with pytest.raises(ValueError):
        ArbiterConfig.from_settings(True, {"delta": -1})


def test_arbitration_knobs_keep_the_compile_hash():
    from vllm.config.kernel import KernelConfig, LongContextKVConfig

    def digest(**kwargs):
        lc = LongContextKVConfig(enabled=True, host_pool_gib=1.0, **kwargs)
        return KernelConfig(long_context_kv=lc).compute_hash()

    base = digest()
    assert digest(value_arbitration=False) == base
    assert digest(value_arbiter_settings={"w_ttft": 2.0}) == base
    assert digest(prewarm_hot_pool=False) != base


def test_rank_counters_need_every_rank_once():
    arb, pool = make()
    arb.begin_step(1)
    arb.begin_step(2)
    one = [1] * NUM_HOT_COUNTERS
    arb.add_rank_counters(1, 0b01, one, 0b11)
    assert 1 not in arb.windows[0].reported
    arb.add_rank_counters(1, 0b10, one, 0b11)
    assert 1 in arb.windows[0].reported
    assert arb.windows[0].hot[HOT_REFETCH] == 2
    # The same rank twice never completes the step.
    arb.add_rank_counters(2, 0b01, one, 0b11)
    arb.add_rank_counters(2, 0b01, one, 0b11)
    arb.add_rank_counters(2, 0b10, one, 0b11)
    assert 2 not in arb.windows[0].reported


def test_worker_metadata_aggregates_counters():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )

    a = OffloadingWorkerMetadata(value_counters={7: (1, (1, 2))})
    b = OffloadingWorkerMetadata(value_counters={7: (2, (10, 20)), 8: (2, (3, 4))})
    merged = a.aggregate(b)
    assert merged.value_counters == {7: (3, (11, 22)), 8: (2, (3, 4))}
