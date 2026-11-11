# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinator-only real CUDA events, graph accounting, and service budgets."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.test_dflash2_lookup_stats import REG
from tests.v1.spec_decode.test_dflash2_native_lookup import fixture_components
from vllm.v1.spec_decode.lookup_stats import VECTOR_SIZE, LookupStatsAccumulator
from vllm.v1.worker.gpu.spec_decode.dflash2.stats import LookupStatsExporter

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        os.environ.get("W21_TEST_DEVICE") != "cuda", reason="coordinator CUDA window"
    ),
]


def assert_zero_counters(counters):
    # UInt64 CUDA any/or_cuda lacks dispatch in torch 2.10. Read back only
    # in this test; compare Python integers on the CPU.
    assert counters.cpu().tolist() == [0] * VECTOR_SIZE


def test_graph_and_real_events_exact_counts_and_final_receipt():
    native, states, mapping, draft, logits, ids, scores, batch, sampled = (
        fixture_components([4, 5, 9, 8, 4, 5])
    )
    exporter = LookupStatsExporter(REG, torch.device("cuda"))
    native.stats = exporter
    # Recording=0 during warmup and capture suppresses every statistics atom.
    native.prepare(batch, sampled, real_run=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            native.apply(states, mapping, draft, 2, logits, ids, scores)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        native.apply(states, mapping, draft, 2, logits, ids, scores)
    assert_zero_counters(exporter.counters)
    native.prepare(batch, sampled, real_run=True)
    for _ in range(23):
        graph.replay()
        exporter.submitted()
    # First token=9, lookup starts9, match2, valid3. Padded row never counts.
    expected = [23, 23, 23, 23] + [0] * 14
    expected[5], expected[-1] = 23, 46
    message, quiet = exporter.drain(time.monotonic() + 5)
    assert quiet and message.snapshot.totals == tuple(expected)
    state = LookupStatsAccumulator(REG)
    state.merge(message)
    assert state.producer == state.covered == 23 and not state.pending
    assert exporter.cpu_receipt()[1]
    # An outstanding real D2H causes conservative UniProc isolation evidence.
    with pytest.raises(RuntimeError, match="retired"):
        exporter.submitted()


def test_uint64_export_preserves_high_bits_with_real_events():
    exporter = LookupStatsExporter(REG, torch.device("cuda"))
    assert_zero_counters(exporter.counters)
    expected = tuple([0, 1, 2**63 - 1, 2**63, 2**64 - 1] + [7] * 13)
    # This H2D write is test-only. Production updates use Triton atomics.
    exporter.counters.copy_(torch.tensor(expected, dtype=torch.uint64))
    exporter.submitted()
    message, quiet = exporter.drain(time.monotonic() + 5)
    assert quiet and message.snapshot.totals == expected
    assert exporter.cpu_receipt()[1]


class CountedEvent:
    def __init__(self, event):
        self.event = event
        self.queries = 0
        self.last_ready = None

    def query(self):
        self.queries += 1
        self.last_ready = self.event.query()
        return self.last_ready

    def record(self, stream):
        self.event.record(stream)


def timed_service(exporter, state):
    busy = exporter.busy
    before = exporter.export_done.queries
    wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
    state.merge(exporter.service())
    cpu_ns, wall_ns = time.thread_time_ns() - cpu, time.perf_counter_ns() - wall
    queries = exporter.export_done.queries - before
    assert queries == int(busy), "F4_EVENT_QUERY_COUNT"
    label = (
        ("completed" if exporter.export_done.last_ready else "pending")
        if busy
        else "idle"
    )
    if label == "completed":
        assert not exporter.busy and state.covered == exporter.producer, (
            "F4_COMPLETED_PAYLOAD"
        )
    return label, cpu_ns, wall_ns


@pytest.mark.parametrize("scenario", ["zero", "no_forward", "inflight_activity"])
def test_real_events_final_checkpoint_without_new_producer(scenario):
    exporter = LookupStatsExporter(REG, torch.device("cuda"))
    if scenario != "zero":
        exporter.submitted()
        exporter.service(force=True)
        if scenario == "no_forward":
            until = time.monotonic() + 5
            while exporter.busy:
                exporter.service()
                assert time.monotonic() < until
            assert exporter.quiescent
        exporter.begin_activity()
    producer, sequence = exporter.producer, exporter.sequence
    message, quiet = exporter.drain(time.monotonic() + 5)
    assert quiet, "F2_CUDA_FINAL_CHECKPOINT"
    assert message.snapshot.covered == producer == exporter.producer
    assert message.snapshot.sequence > sequence
    assert message.snapshot.totals == (0,) * VECTOR_SIZE
    assert exporter.completed_activity == exporter.activity


def test_real_export_service_budget_and_resources():
    exporter = LookupStatsExporter(REG, torch.device("cuda"))
    state = LookupStatsAccumulator(REG)
    exporter.export_done = CountedEvent(exporter.export_done)
    samples: dict[str, dict[str, list[int]]] = {
        name: {"cpu": [], "wall": []} for name in ("idle", "pending", "completed")
    }
    trigger, wall = [], []

    def observe():
        label, cpu_ns, wall_ns = timed_service(exporter, state)
        samples[label]["cpu"].append(cpu_ns)
        samples[label]["wall"].append(wall_ns)
        return label

    before = exporter.export_done.queries
    for _ in range(10000):
        assert observe() == "idle"
    assert exporter.export_done.queries == before
    pending_queries = exporter.export_done.queries
    until = time.monotonic() + 30
    while len(samples["pending"]["cpu"]) < 10000:
        if not exporter.busy:
            # Test-only GPU work delays the real D2H completion event. The
            # service timer excludes this setup; no event result is stubbed.
            with torch.cuda.stream(exporter.export_stream):
                torch.cuda._sleep(100_000_000)
            exporter.submitted()
            exporter.service(force=True)
        observe()
        assert time.monotonic() < until
    assert exporter.export_done.queries - pending_queries >= 10000
    exporter.export_done.event.synchronize()
    if exporter.busy:
        assert observe() == "completed"
    before = exporter.export_done.queries
    for _ in range(10000):
        exporter.submitted()
        exporter.service(force=True)
        exporter.export_done.event.synchronize()
        assert observe() == "completed"
    assert exporter.export_done.queries - before == 10000
    for _ in range(1000):
        exporter.submitted()
        start_wall, start_cpu = time.perf_counter_ns(), time.thread_time_ns()
        state.merge(exporter.service(force=True))
        trigger.append(time.thread_time_ns() - start_cpu)
        wall.append(time.perf_counter_ns() - start_wall)
        deadline = time.monotonic() + 5
        while exporter.busy:
            state.merge(exporter.service())
            assert time.monotonic() < deadline
    import sys

    metadata_bytes = sys.getsizeof(vars(exporter)) + sum(
        sys.getsizeof(value)
        for value in vars(exporter).values()
        if not isinstance(value, torch.Tensor)
    )
    metadata_bytes += sys.getsizeof(exporter.last_snapshot.totals)
    report = {
        "fixed_cpu_metadata_bytes": metadata_bytes,
        "driver_event_stream_bytes": "opaque CUDA runtime allocation",
        "query_merge_p99_us": {
            name: float(np.percentile(values["cpu"], 99) / 1000)
            for name, values in samples.items()
        },
        "query_service": {
            name: {
                "samples": len(values["cpu"]),
                "cpu_p99_us": float(np.percentile(values["cpu"], 99) / 1000),
                "cpu_max_us": max(values["cpu"]) / 1000,
                "wall_p99_us": float(np.percentile(values["wall"], 99) / 1000),
                "wall_max_us": max(values["wall"]) / 1000,
            }
            for name, values in samples.items()
        },
        "actual_event_queries": exporter.export_done.queries,
        "trigger_p99_us": float(np.percentile(trigger, 99) / 1000),
        "trigger_cpu_max_us": max(trigger) / 1000,
        "service_wall_p99_us": float(np.percentile(wall, 99) / 1000),
        "service_wall_max_us": max(wall) / 1000,
        "logical_bytes": metadata_bytes
        + sum(
            t.numel() * t.element_size()
            for t in (exporter.counters, exporter.snapshot, exporter.pinned)
        ),
        "events": 2,
        "export_streams": 1,
        "allocator_reserved_bytes": torch.accelerator.memory_reserved(),
    }
    path = Path(os.environ.get("W21_RESOURCE_OUT", "/tmp"))
    path.mkdir(parents=True, exist_ok=True)
    (path / "stats-budget.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["logical_bytes"] <= 65536
    assert all(value <= 50 for value in report["query_merge_p99_us"].values())
    assert report["trigger_p99_us"] <= 250
    assert report["service_wall_p99_us"] <= 1000


def test_active_and_resumed_delivery_with_real_events():
    from vllm.v1.spec_decode.lookup_stats import ZERO

    exporter = LookupStatsExporter(REG, torch.device("cuda"))
    frontend = LookupStatsAccumulator(REG)
    # Research observation timestamps bracket enqueue and normal output merge.
    first = time.monotonic()
    exporter.submitted()
    while frontend.covered < 1:
        frontend.merge(exporter.service())
        assert time.monotonic() - first <= 3
        time.sleep(0.001)
    exporter.submitted()
    frontend.merge(exporter.message())
    assert frontend.pending and frontend.producer == 2
    previous_time = frontend.timestamp
    # No service during idle, despite a wall-clock export period elapsing.
    time.sleep(1.1)
    assert frontend.pending and frontend.timestamp == previous_time
    resumed = time.monotonic()
    exporter.submitted()
    while frontend.covered < 3:
        frontend.merge(exporter.service())
        assert time.monotonic() - resumed <= 3
        time.sleep(0.001)
    assert frontend.totals == ZERO and not frontend.pending


def run_shutdown_probe(path):
    """The isolated owner lives until this dedicated CUDA process exits."""
    from types import SimpleNamespace as NS
    from unittest.mock import Mock, patch

    import msgspec

    from tests.v1.spec_decode.test_dflash2_lookup_shutdown import engine
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.spec_decode.lookup_stats import LookupStatsRegistration
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    first = LookupStatsExporter(REG, torch.device("cuda"))
    first.submitted()
    assert first.drain(time.monotonic() + 5)[1]
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = NS(
        worker=NS(model_runner=NS(speculator=NS(_native_lookup=NS(stats=first)))),
        shutdown=Mock(),
    )
    complete = engine(executor)
    with patch("vllm.v1.engine.core.cleanup_dist_env_and_memory"):
        complete.shutdown()
    assert complete.lookup_stats_shutdown.receipt.complete
    reg = LookupStatsRegistration(0, REG.owner, REG.generation + "-isolated", 0)
    last = LookupStatsExporter(reg, torch.device("cuda"))
    last.submitted()
    last.service(force=True)
    runner = object.__new__(GPUModelRunner)
    runner.speculator = NS(_native_lookup=NS(stats=last))
    worker = object.__new__(Worker)
    worker.model_runner = runner
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = worker
    executor2 = object.__new__(UniProcExecutor)
    executor2.driver_worker = wrapper
    isolated = engine(executor2)
    isolated.lookup_stats = LookupStatsAccumulator(reg)
    with (
        patch.object(
            last.export_done, "query", side_effect=AssertionError("shutdown query")
        ),
        patch(
            "torch.accelerator.synchronize", side_effect=AssertionError("shutdown sync")
        ),
        patch(
            "torch.accelerator.empty_cache", side_effect=AssertionError("shutdown free")
        ),
    ):
        isolated.shutdown()
        receipt = isolated.lookup_stats_shutdown.receipt
        assert not receipt.complete and receipt.isolated
        try:
            isolated.shutdown()
        finally:
            wrapper.shutdown()
            worker.shutdown()
            runner.shutdown()
        with pytest.raises(RuntimeError, match="process restart"):
            GPUModelRunner(None, None)
    path.write_text(
        json.dumps(
            {
                "complete": msgspec.to_builtins(complete.lookup_stats_shutdown.receipt),
                "isolated": msgspec.to_builtins(receipt),
            },
            indent=2,
        )
        + "\n"
    )


def test_uniproc_real_cuda_receipts_and_process_lifetime_isolation():
    import subprocess
    import sys

    path = Path(os.environ.get("W21_RESOURCE_OUT", "/tmp"))
    path.mkdir(parents=True, exist_ok=True)
    code = (
        "import vllm;"
        "vllm.__path__.append('/usr/local/lib/python3.12/dist-packages/vllm');"
        "from pathlib import Path;"
        "from tests.v1.spec_decode.test_dflash2_lookup_stats_cuda "
        "import run_shutdown_probe;"
        f"run_shutdown_probe(Path({str(path / 'uniproc-receipts.json')!r}))"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90)
    receipt = json.loads((path / "uniproc-receipts.json").read_text())
    assert receipt["complete"]["complete"] and receipt["isolated"]["isolated"]
