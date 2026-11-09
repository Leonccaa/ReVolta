# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""value arbitration value arbitration inside the random-schedule fuzzer.

The simulated workers report synthetic hot-side counters (stable, alternating
or burst pressure) so that T_idx really moves and cached IDs are donated
while the scheduler runs with aborts, resets, failed loads and preemptions.
Per step: T_idx stays in [0, A - U - ceil(F/k)] and matches the pool, and no
window donates more than Q cached IDs. Each run reports, per scheduling
policy, the longest wait to first progress and the longest no-progress gap,
with arbitration on and off on the same seed .

Opt-in sweep (writes JSON lines):

    REVOLTA_ARBITER_FUZZ_SEEDS=0:40 REVOLTA_ARBITER_FUZZ_OUT=out.jsonl \
        pytest -s tests/v1/core/test_value_arbiter_fuzz.py -k sweep
"""

import json
import os
import random

import pytest

from vllm.v1.core.value_arbiter import (
    HOT_FORCED_RECENT,
    HOT_REFETCH,
    NUM_HOT_COUNTERS,
)

from .long_context_sched_fuzz import Driver, Extension, make_profile

pytestmark = pytest.mark.cpu_test

MODES = ("stable", "alternating", "burst")


class ArbiterProbe(Extension):
    name = "value-arbiter"

    def __init__(self, mode: str, enabled: bool = True):
        self.mode, self.enabled = mode, enabled
        self.first_seen: dict[str, int] = {}
        self.max_wait = 0
        self.last_checked_window = -1
        self.max_cached = 0

    def attach(self, driver: Driver) -> None:
        s = driver.scheduler
        self.pool = s.host_history_pool
        self.arb = s.value_arbiter
        if not self.enabled and self.arb is not None:
            # Arbitration off: T_idx stays at the shared hot region default split.
            self.pool.index_cache_target = self.arb.default_target
            self.pool.value_observer = None
            s.value_arbiter = self.arb = None
        self.rng = random.Random(f"arbiter-{driver.profile.seed}-{self.mode}")
        if self.arb is not None:
            # Start below the default split so value donations really run.
            self.arb.target = self.rng.choice((0, 1, self.arb.target // 4))

    def _pressure(self, window: int) -> tuple[int, int]:
        if self.mode == "stable":
            return 8, 4
        if self.mode == "alternating":
            return (8, 8) if window % 2 == 0 else (8, 0)
        return (8, 8) if 2 <= window % 8 < 5 else (8, 1)

    def on_worker_step(self, driver: Driver, output) -> None:
        if self.arb is None:
            return
        meta = output.kv_connector_worker_meta
        if meta is None:
            return
        all_ranks = self.pool.host_pages.all_ranks
        for step in list(meta.completed_steps):
            forced, refetch = self._pressure(self.arb.window)
            values = [0] * NUM_HOT_COUNTERS
            values[HOT_FORCED_RECENT] = forced
            values[HOT_REFETCH] = min(refetch, forced) if self.rng.random() < 0.9 else 0
            if self.rng.random() < 0.003:
                continue  # a lost report: that window must hold
            meta.value_counters[step] = (all_ranks, tuple(values))

    def check(self, driver: Driver) -> None:
        s = driver.scheduler
        for rid, request in s.requests.items():
            start = self.first_seen.setdefault(rid, driver.step)
            if request.num_computed_tokens == 0 and not request.is_finished():
                self.max_wait = max(self.max_wait, driver.step - start)
        arb = self.arb
        if arb is None:
            return
        snap = self.pool.arbiter_snapshot()
        self.max_cached = max(self.max_cached, snap.cached)
        if not 0 <= arb.t_idx <= max(snap.upper, arb.t_idx):
            driver.fail(f"T_idx {arb.t_idx} outside [0, {snap.upper}]")
        if self.pool.index_cache_target != arb.t_idx:
            driver.fail("pool T_idx differs from the arbiter's")
        for decision in arb.decisions:
            if decision.window <= self.last_checked_window:
                continue
            if decision.value_donated > arb.q:
                driver.fail(f"window {decision.window} donated more than Q")
            if decision.reason == "counters_missing" and decision.action != "hold":
                driver.fail("moved without complete counters")
        if arb.decisions:
            self.last_checked_window = arb.decisions[-1].window


def run(seed: int, policy: str, mode: str, enabled: bool, steps: int) -> dict:
    profile = make_profile(
        seed,
        placement="host",
        priority_policy=policy == "priority",
        avoid=(),
    )
    probe = ArbiterProbe(mode, enabled)
    result = Driver(profile, steps, extensions=[probe]).run()
    arb = probe.arb
    return {
        "seed": seed,
        "policy": policy,
        "mode": mode,
        "arbitration": enabled,
        "finished": result.finished,
        "max_idle_wait": result.max_idle_wait,
        "max_wait_first_progress": probe.max_wait,
        "moves": arb.value_moves if arb else 0,
        "windows": sum(arb.decision_counts.values()) if arb else 0,
        "value_donated": probe.arb.value_donated if probe.arb else 0,
        "forced_clamps": probe.arb.forced_clamps if probe.arb else 0,
        "max_cached": probe.max_cached,
        "reasons": sorted(arb.decision_counts) if arb else [],
    }


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
@pytest.mark.parametrize("mode", MODES)
def test_arbiter_under_fuzz(policy, mode):
    seed = MODES.index(mode) + (10 if policy == "priority" else 0)
    on = run(seed, policy, mode, True, 160)
    off = run(seed, policy, mode, False, 160)
    assert on["windows"] > 0
    if mode == "stable":
        assert on["moves"] > 0
    print(json.dumps(on), json.dumps(off))


def _seeds() -> range:
    spec = os.environ.get("REVOLTA_ARBITER_FUZZ_SEEDS")
    if not spec:
        return range(0)
    start, stop = (int(x) for x in spec.split(":"))
    return range(start, stop)


@pytest.mark.skipif(not _seeds(), reason="opt-in sweep")
def test_arbiter_fuzz_sweep():
    out = os.environ.get("REVOLTA_ARBITER_FUZZ_OUT")
    steps = int(os.environ.get("REVOLTA_ARBITER_FUZZ_STEPS", "300"))
    failures = []
    for seed in _seeds():
        for policy in ("fcfs", "priority"):
            mode = MODES[seed % len(MODES)]
            for enabled in (True, False):
                try:
                    row = run(seed, policy, mode, enabled, steps)
                except AssertionError as error:
                    row = {
                        "seed": seed,
                        "policy": policy,
                        "mode": mode,
                        "arbitration": enabled,
                        "failure": str(error)[:400],
                    }
                    failures.append(row)
                if out:
                    with open(out, "a") as f:
                        f.write(json.dumps(row) + "\n")
    assert not failures, failures[:3]
