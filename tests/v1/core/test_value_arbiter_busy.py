# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A6 behavior regression, independently of observability switches."""

import pytest

from tests.v1.core.test_value_arbiter_pool import cached_pool
from vllm.v1.core.value_arbiter import (
    HOT_FORCED_RECENT,
    HOT_REFETCH,
    NUM_HOT_COUNTERS,
    ArbiterConfig,
    BusyTime,
    ValueArbiter,
)


def test_busy_union_overlap_idle_duplicate_and_reverse_completion():
    busy = BusyTime()
    busy.start(1, 10.0)
    busy.start(2, 11.0)
    assert busy.finish(2, 13.0) == 3.0
    assert busy.finish(2, 14.0) is None
    assert busy.finish(1, 15.0) == 2.0
    busy.start(3, 1000.0)
    assert busy.finish(3, 1000.25) == 0.25
    assert busy.steps == set() and busy.since is None


def decision_trace(busy_time):
    pool, blocks = cached_pool(n=6, blocks=9, target=4)
    arb = ValueArbiter(
        ArbiterConfig(
            window_steps=1,
            busy_time=busy_time,
            bw_uva=1.0,
            bw_h2d=1.0,
            w_async=0.0,
            w_ttft=1.0,
        ),
        pool,
        default_target=4,
        slot_bytes=1,
        index_block_bytes=1,
    )
    pool.value_observer = arb.on_index_evicted
    arb.begin_step(2, None if busy_time else 10.0)
    if busy_time:
        clock = BusyTime()
        clock.start(2, 1000.0)
        arb.record_step_seconds(2, clock.finish(2, 1000.01))
    arb.on_index_evicted("reload")
    arb.on_index_restored(["reload"])
    counters = [0] * NUM_HOT_COUNTERS
    counters[HOT_FORCED_RECENT], counters[HOT_REFETCH] = 100, 50
    arb.add_hot_counters(2, counters)
    arb.begin_step(3)
    arb.end_step(3, completed_step=2)
    d = arb.decisions[-1]
    roles = pool.finish_roles()
    return d, roles, pool


def test_busy_time_changes_p4_decision_and_shared_region_role_trace():
    old, old_roles, old_pool = decision_trace(False)
    busy, busy_roles, busy_pool = decision_trace(True)
    assert old.action == "raise" and old.t_idx == 5
    assert busy.action == "lower" and busy.t_idx == 3
    assert old.idx_density > old.hot_density
    assert busy.idx_density < busy.hot_density
    assert old_roles != busy_roles
    assert old_pool.num_cached_free() == 6
    assert busy_pool.num_cached_free() == 5


def test_busy_time_config_is_independent_and_strict():
    from vllm.config.kernel import LongContextKVConfig

    settings = LongContextKVConfig(value_arbiter_settings={"busy_time": False})
    assert not ArbiterConfig.from_settings(
        True, settings.value_arbiter_settings
    ).busy_time
    assert ArbiterConfig.from_settings(True).busy_time
    assert not ArbiterConfig.from_settings(True, {"busy_time": False}).busy_time
    with pytest.raises(ValueError, match="boolean"):
        ArbiterConfig.from_settings(True, {"busy_time": "false"})


def test_hold_decisions_are_a_bounded_ring_with_cumulative_summary():
    from tests.v1.core.test_value_arbiter import make

    arb, _ = make(config=ArbiterConfig(window_steps=1))
    for step in range(1, 601):
        arb.begin_step(step, 0.01)
        arb.add_hot_counters(step, [0] * NUM_HOT_COUNTERS)
        arb.end_step(step)
    assert len(arb.decisions) == arb.decisions.maxlen == 256
    assert sum(arb.decision_counts.values()) == 599
    assert len(arb.windows) <= 2


def test_transfer_seal_and_role_only_batches_are_busy_but_empty_poll_is_idle():
    from types import SimpleNamespace

    host = SimpleNamespace(bindings=[], seals=[], donate=[], claim=[])
    meta = SimpleNamespace(load_jobs={}, store_jobs={}, jobs_to_flush=set())
    output = SimpleNamespace(
        total_num_scheduled_tokens=0, host_kv=host, kv_connector_metadata=meta
    )
    assert not BusyTime.has_work(output)
    for name in ("bindings", "seals", "donate", "claim"):
        getattr(host, name).append(1)
        assert BusyTime.has_work(output)
        getattr(host, name).clear()
    for name in ("load_jobs", "store_jobs"):
        getattr(meta, name)[1] = None
        assert BusyTime.has_work(output)
        getattr(meta, name).clear()
    meta.jobs_to_flush.add(1)
    assert BusyTime.has_work(output)


def test_feedback_before_completion_keeps_the_busy_window_open():
    from tests.v1.core.test_value_arbiter import make

    arb, _ = make(
        config=ArbiterConfig(
            window_steps=1, bw_uva=1.0, bw_h2d=1.0, w_async=0.0, w_ttft=1.0
        )
    )
    arb.begin_step(1)
    arb.mark_busy(1)
    arb.on_index_evicted("x")
    arb.on_index_restored(["x"])
    arb.add_hot_counters(1, [0] * NUM_HOT_COUNTERS)
    arb.begin_step(2)
    assert arb.end_step(2, completed_step=2) is None
    assert 0 in arb.windows
    arb.record_step_seconds(1, 0.01)
    decision = arb.end_step(2, completed_step=2)
    assert decision is not None and decision.idx_density == pytest.approx(0.01)
    assert not arb._busy_window
