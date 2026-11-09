# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Random-schedule fuzzing of the long-context (Active Host KV) mode.

The quick mode runs a few seeds in the normal suite. The long mode is opt-in:

    REVOLTA_SCHED_FUZZ_SEEDS=0:200 REVOLTA_SCHED_FUZZ_STEPS=300 pytest -s \
        tests/v1/core/test_long_context_sched_fuzz.py -k long

``REVOLTA_SCHED_FUZZ_OUT`` names a JSON-lines file for per-seed results and
``REVOLTA_SCHED_FUZZ_AVOID`` (comma list, empty for none) selects which known
findings the driver steps around (default: the open ones in ``KNOWN_ISSUES``).
See ``long_context_sched_fuzz.py`` for the events, invariants and the
extension points of the v2 phases.
"""

import dataclasses
import json
import os

import pytest

from vllm.v1.core.long_context_kv import HOST

from .long_context_sched_fuzz import (
    Driver,
    Extension,
    FuzzFailure,
    make_profile,
    run_seed,
)

pytestmark = pytest.mark.cpu_test

QUICK_SEEDS = 6
QUICK_STEPS = 120


@pytest.mark.parametrize("seed", range(QUICK_SEEDS))
def test_quick_fuzz(seed):
    run_seed(seed, QUICK_STEPS)


def test_same_seed_same_schedule():
    first = run_seed(3, 80)
    second = run_seed(3, 80)
    assert first.digest == second.digest
    assert first.stats == second.stats


class _Plant(Extension):
    """Corrupt one piece of state once, at the first step from ``at`` on where
    the fault applies; the checks must notice."""

    def __init__(self, fault: str, at: int = 40):
        self.fault = fault
        self.at = at
        self.planted = False

    def on_events(self, driver):
        if self.planted or driver.step < self.at:
            return
        s = driver.scheduler
        ledger = driver.ledger
        if self.fault == "ledger":
            ledger.in_use[HOST] += 1
        elif self.fault == "hot":
            ledger._hot["ghost"] = 1
            ledger.hot_used += 1
        elif self.fault == "queue":
            if not s.waiting:
                return
            s.waiting.add_request(next(iter(s.waiting)))  # queued twice
        elif self.fault == "stall":
            from vllm.v1.core.sched.interface import PauseState

            # Nothing is scheduled any more, whatever is running at the time.
            s._pause_state = PauseState.PAUSED_ALL
        elif self.fault == "pin":
            driver.conn.pins["ghost"] += 1
        self.planted = True


@pytest.mark.parametrize(
    "fault,message",
    [
        ("ledger", "ledger in_use"),
        ("hot", "hot reservation of freed request"),
        # A duplicate queue entry is caught when counted or when the request
        # finishes while still queued, whichever the trajectory reaches first.
        ("queue", "queued 2 times|still queued"),
        ("stall", "no progress"),
        ("pin", "connector pins"),
    ],
)
def test_checks_catch_planted_faults(fault, message):
    seed = 1
    driver = Driver(
        make_profile(seed, p_arrive=0.6, avoid=()),
        steps=200,
        extensions=[_Plant(fault)],
    )
    with pytest.raises(FuzzFailure, match=message):
        driver.run()


def _seed_range() -> range:
    spec = os.environ.get("REVOLTA_SCHED_FUZZ_SEEDS", "0:200")
    start, _, stop = spec.partition(":")
    return range(int(start), int(stop))


@pytest.mark.skipif(
    "REVOLTA_SCHED_FUZZ_SEEDS" not in os.environ,
    reason="long fuzz mode: set REVOLTA_SCHED_FUZZ_SEEDS=start:stop",
)
def test_long_fuzz():
    steps = int(os.environ.get("REVOLTA_SCHED_FUZZ_STEPS", "300"))
    avoid = os.environ.get("REVOLTA_SCHED_FUZZ_AVOID")
    overrides = {}
    if avoid is not None:
        overrides["avoid"] = tuple(k for k in avoid.split(",") if k)
    out = os.environ.get("REVOLTA_SCHED_FUZZ_OUT")
    failures = []
    passed = 0
    for seed in _seed_range():
        record: dict = {"seed": seed, "steps": steps}
        try:
            result = run_seed(seed, steps, **overrides)
            record.update(dataclasses.asdict(result), status="pass")
            passed += 1
        except FuzzFailure as failure:
            text = str(failure)
            record.update(
                status="fail",
                profile=make_profile(seed, **overrides).describe(),
                error=text.splitlines()[0],
                detail=text,
            )
            failures.append((seed, text.splitlines()[0]))
        print(
            f"seed={seed} {record['status']} "
            + (record.get("error", "") or f"digest={record.get('digest')}")
        )
        if out:
            with open(out, "a") as handle:
                handle.write(json.dumps(record) + "\n")
    print(f"SUMMARY passed={passed} failed={len(failures)} seeds={_seed_range()}")
    assert not failures, failures


__all__ = ["Driver"]
