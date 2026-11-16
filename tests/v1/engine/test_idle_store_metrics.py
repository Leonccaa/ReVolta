# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Idle polling uses the same delta exporter as active scheduler steps."""

import queue
from collections import Counter
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prometheus_client import CollectorRegistry, Gauge, Histogram
from prometheus_client import Counter as PromCounter

from tests.v1.core.utils import create_scheduler
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.metrics.store import StoreMetricsProm
from vllm.v1.spec_decode.lookup_stats import (
    ZERO,
    LookupStatsAccumulator,
    LookupStatsMessage,
    LookupStatsRegistration,
    LookupStatsSnapshot,
)


@pytest.mark.parametrize("persistence_pending", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_idle_polls_export_deltas_and_resume_without_double_counting(
    monkeypatch, persistence_pending, block
):
    scheduler = create_scheduler(enable_prefix_caching=True)
    metrics = scheduler.store_metrics
    assert metrics is not None
    log = Mock()
    monkeypatch.setattr("vllm.v1.metrics.store.logger.info", log)
    clock = SimpleNamespace(now=100.0, unix=1000.0)
    monkeypatch.setattr("vllm.v1.engine.core.time.monotonic", lambda: clock.now)
    monkeypatch.setattr("vllm.v1.metrics.store.time.time", lambda: clock.unix)
    dog = SimpleNamespace(disable=nullcontext, feed=lambda: None)
    monkeypatch.setattr("vllm.v1.engine.core.get_watch_dog", lambda: dog)
    totals = Counter(writes=3)
    scheduler.value_arbiter = SimpleNamespace(decision_counts=totals)
    connector = SimpleNamespace(
        has_pending_persistence=lambda: persistence_pending,
        persistence_step=Mock(),
    )
    monkeypatch.setattr(scheduler, "get_kv_connector", lambda: connector)
    # Idle export must not drain prefix statistics belonging to real steps.
    scheduler.kv_cache_manager.prefix_cache_stats.record(100, 20, False)
    metrics.counters["admission_retry_receive"] += 2
    metrics.observe("receive", "host", 0.5)
    timeouts = []

    class IdleQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            timeouts.append(timeout)
            elapsed = timeout if block else 1.0
            assert elapsed is not None
            clock.now += elapsed
            clock.unix += elapsed
            totals["writes"] += 1
            raise queue.Empty

    engine = object.__new__(EngineCoreProc)
    engine.scheduler = scheduler
    registration = LookupStatsRegistration(0, "output", "worker-generation", 0)
    engine.lookup_stats = LookupStatsAccumulator(registration)
    lookup_totals = (1,) + ZERO[1:]
    engine.lookup_stats.merge(
        LookupStatsMessage(
            registration.owner,
            registration.generation,
            2,
            LookupStatsSnapshot(1, 1, lookup_totals, 999.0),
        )
    )
    engine.model_executor = Mock()
    engine.model_executor.collective_rpc.side_effect = AssertionError(
        "IDLE_MUST_NOT_ISSUE_LOOKUP_RPC"
    )
    engine.has_work = lambda: False
    engine.is_running = lambda: True
    engine._notify_idle_state_callbacks = Mock()
    engine.input_queue = IdleQueue()
    engine.aborts_queue = queue.Queue()
    engine.output_queue = queue.Queue()
    engine.process_input_queue_block = block
    # Pending persistence keeps its one-second poll; otherwise sleep to export.
    polls = 121 if persistence_pending or not block else 3
    for _ in range(polls):
        engine._process_input_queue()
    outputs = list(engine.output_queue.queue)
    assert len(outputs) == 3
    assert log.call_count == 3
    assert all(client == 0 for client, _ in outputs)
    assert all(out.scheduler_stats.lookup_stats is None for _, out in outputs)
    state = engine.lookup_stats
    assert (state.producer, state.covered, state.sequence) == (2, 1, 1), (
        "IDLE_LOOKUP_WATERMARK_UNCHANGED"
    )
    assert state.totals == lookup_totals, "IDLE_LOOKUP_COUNTS_UNCHANGED"
    assert state.timestamp == 999.0 and state.pending == 1
    engine.model_executor.collective_rpc.assert_not_called()
    payloads = [out.scheduler_stats.store_stats for _, out in outputs]
    assert [p["gauges"]["store_snapshot_unix_s"] for p in payloads] == [
        1000.0,
        1060.0,
        1120.0,
    ]
    assert all(p["gauges"]["store_snapshot_idle"] == 1 for p in payloads)
    if persistence_pending:
        assert timeouts == [1.0] * polls
    else:
        assert max(timeouts) == 60.0 and min(timeouts) > 0
    assert connector.persistence_step.call_count == (
        polls if persistence_pending else 0
    )
    assert scheduler.kv_cache_manager.prefix_cache_stats.queries == 100
    # Resume: export remaining events once, then suppress another due check.
    clock.now, clock.unix = 280.0, 1180.0
    active = scheduler.make_stats().store_stats
    assert active["gauges"]["store_snapshot_idle"] == 0
    assert scheduler.get_idle_stats()[0] is None
    assert log.call_count == 4
    payloads.append(active)
    assert sum(p["counters"].get("arbiter_writes", 0) for p in payloads) == 3 + polls
    assert sum(p["counters"].get("admission_retry_receive", 0) for p in payloads) == 2
    assert sum(len(p["observations"]) for p in payloads) == 1
    registry = CollectorRegistry()
    prom = StoreMetricsProm(
        partial(PromCounter, registry=registry),
        partial(Gauge, registry=registry),
        partial(Histogram, registry=registry),
        ["engine"],
        {0: ["0"]},
    )
    for payload in payloads:
        prom.observe(payload, 0)
    assert (
        registry.get_sample_value(
            "vllm:store_events_total", {"engine": "0", "event": "arbiter_writes"}
        )
        == 3 + polls
    )
    for metric, value in (("store_snapshot_unix_s", 1180), ("store_snapshot_idle", 0)):
        assert (
            registry.get_sample_value(
                "vllm:store_state", {"engine": "0", "metric": metric}
            )
            == value
        )


def test_disabled_store_metrics_preserves_blocking_input(monkeypatch):
    scheduler = create_scheduler()
    scheduler.store_metrics = None
    assert scheduler.get_idle_stats() == (None, None)
    engine = object.__new__(EngineCoreProc)
    engine.scheduler = scheduler
    engine.has_work = Mock(side_effect=[False, True])
    engine.is_running = lambda: True
    engine._notify_idle_state_callbacks = Mock()
    engine.aborts_queue = queue.Queue()
    engine.output_queue = queue.Queue()
    engine.input_queue = Mock()
    engine.input_queue.empty.return_value = True
    engine.input_queue.get.return_value = ("request", None)
    engine._handle_client_request = Mock()
    engine.process_input_queue_block = True
    dog = SimpleNamespace(disable=nullcontext, feed=lambda: None)
    monkeypatch.setattr("vllm.v1.engine.core.get_watch_dog", lambda: dog)
    engine._process_input_queue()
    engine.input_queue.get.assert_called_once_with(block=True, timeout=None)
    engine._handle_client_request.assert_called_once_with("request", None)
    assert engine.output_queue.empty()
