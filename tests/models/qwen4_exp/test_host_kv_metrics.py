# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-0 telemetry in the CPU Triton interpreter, including normal gathers."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv.shadow import HotCacheMetrics
from vllm.v1.metrics.store import StoreMetrics

from .test_host_kv_history import Engine, _interpreter_boolean_comparison, _shape

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="CPU Triton interpreter required"
)
__all__ = ["_interpreter_boolean_comparison"]


def test_accumulation_snapshot_poll_reset_and_large_totals():
    meter = HotCacheMetrics(torch.device("cpu"))
    store = SimpleNamespace(
        wanted_count=torch.tensor(9, dtype=torch.int32),
        miss_count=torch.tensor(4, dtype=torch.int32),
    )
    meter.counters[:2] = 2**33
    for _ in range(3):
        meter.count(store)
    owner = torch.tensor([-2, -1, 3, 7, -1, 0], dtype=torch.int32)
    assert meter.harvest(10, owner) is None
    # Snapshot is a copy, later device work cannot change it.
    meter.count(store)
    assert meter.harvest(11, owner) == (10, 2**33 + 27, 2**33 + 12, 3)
    meter.next_snapshot = 0
    assert meter.harvest(12, owner) is None
    meter.event = SimpleNamespace(query=lambda: False)
    assert meter.harvest(13, owner) is None
    meter.event = SimpleNamespace(query=lambda: True)
    assert meter.harvest(14, owner) == (12, 2**33 + 36, 2**33 + 16, 3)
    meter.reset()
    assert not meter.counters.any() and meter.pending is None


def test_normal_gathers_count_layer_zero_only_and_leave_payload_identical():
    engines = [
        Engine(_shape(rows_per_pass=1, requests=1, blocks=12), seed=3) for _ in range(2)
    ]
    measured = engines[1].store
    measured.hot_metrics = HotCacheMetrics(torch.device("cpu"))
    for engine in engines:
        engine.layer = engine.store.layers[0]
    expected_wanted = expected_misses = 0
    for engine in engines:
        engine.write("a", list(range(100)))
    for q in range(100):
        for engine in engines:
            engine.step([("a", q)])
            if engine is engines[1]:
                expected_wanted += int(measured.wanted_count)
                expected_misses += int(measured.miss_count)
    assert measured.hot_metrics.counters[:2].tolist() == [
        expected_wanted,
        expected_misses,
    ]
    assert expected_wanted >= expected_misses > 0
    plain = engines[0].store
    for name in ("hot", "group_to_slot", "slot_owner", "age", "host", "ready"):
        assert torch.equal(getattr(plain, name), getattr(measured, name)), name
    # The second layer shares scratch but contributes no telemetry.
    before = measured.hot_metrics.counters.clone()
    engines[1].layer = measured.layers[1]
    engines[1].write("a", list(range(100)))
    engines[1].step([("a", 99)])
    assert torch.equal(before, measured.hot_metrics.counters)
    measured.reset_hot_state()
    assert not measured.hot_metrics.counters.any()


def test_rank_metadata_delta_export_and_duplicate_receipts():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )

    first = OffloadingWorkerMetadata(hot_metrics={0: (10, 100, 30, 7)})
    second = OffloadingWorkerMetadata(hot_metrics={1: (10, 90, 20, 9)})
    reports = first.aggregate(second).hot_metrics
    m = StoreMetrics()
    m.worker_hot(reports)
    m.worker_hot(reports)
    out = m.export(1.0)
    assert out["counters"]["gpu_hot_wanted_groups_total"] == 190
    assert out["counters"]["gpu_hot_miss_groups_total"] == 50
    assert out["gauges"]["gpu_hot_valid_slots"] == 7
    m.worker_hot({0: (20, 130, 40, 5), 1: (20, 120, 25, 8)})
    m.worker_hot(reports)  # delayed old output
    out = m.export(61.0)
    assert out["counters"]["gpu_hot_wanted_groups_total"] == 60
    assert out["counters"]["gpu_hot_miss_groups_total"] == 15
    assert out["gauges"]["gpu_hot_valid_slots"] == 5


def test_scheduler_receives_worker_hot_metadata():
    from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
        setup_receive,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.v1.outputs import KVConnectorOutput

    scheduler, _, _, _ = setup_receive()
    scheduler.store_metrics = StoreMetrics()
    scheduler.update_host_kv_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                hot_metrics={0: (10, 100, 30, 7)}
            )
        )
    )
    assert scheduler.store_metrics.counters["gpu_hot_wanted_groups_total"] == 100
    assert scheduler.store_metrics.hot_valid_slots == 7
