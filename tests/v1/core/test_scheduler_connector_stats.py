# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side KV connector stats are collected every step, also when no
worker transfer reported stats in that step."""

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput


class _Stats:
    def __init__(self, data: dict):
        self.data = dict(data)

    def __bool__(self) -> bool:
        return bool(self.data)

    def aggregate(self, other: "_Stats") -> "_Stats":
        merged = dict(self.data)
        for k, v in other.data.items():
            merged[k] = merged.get(k, 0) + v
        return _Stats(merged)


class _Connector:
    def __init__(self, stats: "_Stats | None"):
        self.stats = stats
        self.calls = 0

    def get_kv_connector_stats(self):
        self.calls += 1
        return self.stats


def _collect(connector, output):
    fake = SimpleNamespace(connector=connector)
    return Scheduler._collect_kv_connector_stats(fake, output)


def test_scheduler_stats_without_worker_stats():
    connector = _Connector(_Stats({"evictions": 3}))
    assert _collect(connector, None).data == {"evictions": 3}
    assert _collect(connector, KVConnectorOutput()).data == {"evictions": 3}
    assert connector.calls == 2


def test_scheduler_stats_aggregate_with_worker_stats():
    connector = _Connector(_Stats({"evictions": 3}))
    output = KVConnectorOutput(kv_connector_stats=_Stats({"bytes": 7}))
    assert _collect(connector, output).data == {"bytes": 7, "evictions": 3}


def test_no_scheduler_stats_keeps_worker_stats():
    worker = _Stats({"bytes": 7})
    output = KVConnectorOutput(kv_connector_stats=worker)
    assert _collect(_Connector(None), output) is worker
    assert _collect(_Connector(_Stats({})), output) is worker
    assert _collect(None, output) is worker
    assert _collect(_Connector(None), None) is None
