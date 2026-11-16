# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-tier evictions counted by reason (read-only observability).

- capacity + superseded equals the existing ``num_evicted_blocks`` total and
  the keys carried by the removal events;
- reset counts blocks dropped by ``reset_cache``;
- the bookkeeping never changes admission or eviction: a randomized workload
  gives identical results with the bookkeeping disabled;
- the scheduler reports deltas that add up to the cumulative totals, and the
  connector stats / Prometheus counter carry them by reason.
"""

import random
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    CPU_EVICTIONS_KEY,
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)

pytestmark = pytest.mark.cpu_test

CTX = ReqContext("test")


def key(i, group=0):
    return make_offload_key(i.to_bytes(8, "big"), group)


def store(m, keys):
    out = m.prepare_store(keys, CTX)
    if out is not None:
        m.complete_store(out.keys_to_store, CTX)
    return out


def removed_event_keys(m):
    return sum(len(e.keys) for e in m.take_events() if e.removed)


def test_lru_capacity_and_superseded_reasons():
    m = CPUOffloadingManager(3, cache_policy="lru", enable_events=True)
    a, b, c, d, e = (key(i) for i in range(5))
    store(m, [a, b, c])
    m.demote([c], CTX)
    assert store(m, [d]).evicted_keys == [c]  # demoted -> superseded
    assert store(m, [e]).evicted_keys == [a]  # plain LRU -> capacity
    assert m.evicted_blocks_by_reason == {"superseded": 1, "capacity": 1}
    assert m.num_evicted_blocks == 2
    assert removed_event_keys(m) == 2


def test_touch_after_demote_is_not_superseded():
    m = CPUOffloadingManager(2, cache_policy="lru")
    a, b, c = (key(i) for i in range(3))
    store(m, [a, b])
    m.demote([a], CTX)
    m.touch([a], CTX)  # used again: back at the MRU end, no longer demoted
    assert store(m, [c]).evicted_keys == [b]
    store(m, [key(9)])  # now evicts a, as capacity
    assert m.evicted_blocks_by_reason == {"capacity": 2}


def test_reset_counts_cached_blocks():
    m = CPUOffloadingManager(4, cache_policy="lru")
    store(m, [key(i) for i in range(3)])
    m.reset_cache()
    assert m.evicted_blocks_by_reason == {"reset": 3}
    assert m.num_evicted_blocks == 0  # unchanged meaning: policy evictions


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_reasons_reconcile_with_total_and_events(policy):
    rng = random.Random(0)
    m = CPUOffloadingManager(16, cache_policy=policy, enable_events=True)
    events = 0
    for _ in range(400):
        keys = [key(rng.randrange(64)) for _ in range(rng.randint(1, 6))]
        op = rng.random()
        if op < 0.6:
            store(m, keys)
        elif op < 0.8:
            m.touch(keys, CTX)
        else:
            m.demote(keys, CTX)
        events += removed_event_keys(m)
    by_reason = m.evicted_blocks_by_reason
    assert by_reason["capacity"] + by_reason["superseded"] == m.num_evicted_blocks
    assert m.num_evicted_blocks == events > 0
    if policy == "arc":  # ARC does not implement demote
        assert by_reason["superseded"] == 0


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_bookkeeping_does_not_change_decisions(policy, monkeypatch):
    def run():
        rng = random.Random(1)
        m = CPUOffloadingManager(12, cache_policy=policy, enable_events=True)
        trace: list[object] = []
        for _ in range(500):
            keys = [key(rng.randrange(48)) for _ in range(rng.randint(1, 5))]
            op = rng.random()
            if op < 0.55:
                out = store(m, keys)
                trace.append(
                    None
                    if out is None
                    else (
                        out.keys_to_store,
                        list(out.store_spec.block_ids),
                        out.evicted_keys,
                    )
                )
            elif op < 0.75:
                m.touch(keys, CTX)
            elif op < 0.9:
                m.demote(keys, CTX)
            else:
                trace.append([m.lookup(k, CTX) for k in keys])
            trace.append([(e.keys, e.removed) for e in m.take_events()])
        return trace

    with_bookkeeping = run()
    monkeypatch.setattr(CPUOffloadingManager, "_record_evictions", lambda *_: None)
    monkeypatch.setattr(
        CPUOffloadingManager,
        "demote",
        lambda self, keys, ctx: self._policy.demote(keys),
    )
    assert run() == with_bookkeeping


def _scheduler_over(manager):
    sched = object.__new__(OffloadingConnectorScheduler)
    sched.manager = manager
    return sched


def test_scheduler_deltas_sum_to_totals_across_group_pools():
    pools = {g: CPUOffloadingManager(2, cache_policy="lru") for g in (0, 1)}
    sched = _scheduler_over(GroupedCPUOffloadingManager(pools))
    reported: dict[str, int] = {}
    for step in range(6):
        for g, m in pools.items():
            store(m, [key(10 * step + i, g) for i in range(2)])
        if step == 3:
            pools[0].demote([key(31, 0)], CTX)
        for r, n in sched.cpu_tier_eviction_deltas().items():
            assert n > 0
            reported[r] = reported.get(r, 0) + n
    totals: dict[str, int] = {}
    for m in pools.values():
        for r, n in m.evicted_blocks_by_reason.items():
            totals[r] = totals.get(r, 0) + n
    assert reported == totals
    assert sum(totals.values()) == sum(m.num_evicted_blocks for m in pools.values())
    assert sched.cpu_tier_eviction_deltas() == {}


def test_connector_reports_scheduler_side_evictions():
    m = CPUOffloadingManager(2, cache_policy="lru")
    connector = object.__new__(OffloadingConnector)
    connector.connector_worker = None
    connector.connector_scheduler = _scheduler_over(m)
    assert connector.get_kv_connector_stats() is None
    store(m, [key(0), key(1)])
    store(m, [key(2)])
    stats = connector.get_kv_connector_stats()
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.data[CPU_EVICTIONS_KEY] == [{"reason": "capacity", "blocks": 1}]
    assert stats.reduce() == {f"{CPU_EVICTIONS_KEY}_capacity": 1}
    assert connector.get_kv_connector_stats() is None


def test_prometheus_counter_by_reason():
    registry = CollectorRegistry()

    def bind(cls):
        return lambda *a, **k: cls(*a, registry=registry, **k)

    prom = OffloadPromMetrics(
        SimpleNamespace(kv_transfer_config=None),
        {Gauge: bind(Gauge), Counter: bind(Counter), Histogram: bind(Histogram)},
        ["model_name", "engine"],
        {0: ["m", "0"]},
    )
    # Worker stats arrive serialized (dicts), as from the worker process.
    worker = OffloadingConnectorStats(
        data={"GPU_to_CPU": [{"op_size": 1024, "op_time": 0.5}]}
    )
    sched = OffloadingConnectorStats()
    sched.record_evictions({"capacity": 3, "superseded": 2})
    merged = worker.aggregate(sched)
    prom.observe(merged.data, engine_idx=0)
    sched2 = OffloadingConnectorStats()
    sched2.record_evictions({"capacity": 1})
    prom.observe(sched2.data, engine_idx=0)

    def value(reason):
        return registry.get_sample_value(
            "vllm:kv_offload_cpu_evicted_blocks_total",
            {"model_name": "m", "engine": "0", "reason": reason},
        )

    assert value("capacity") == 4
    assert value("superseded") == 2
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_total_bytes_total",
            {"model_name": "m", "engine": "0", "transfer_type": "GPU_to_CPU"},
        )
        == 1024
    )
