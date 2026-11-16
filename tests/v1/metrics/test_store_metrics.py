# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from vllm.v1.metrics.store import PrefixAttempt, StoreMetrics, StoreMetricsProm, cover


def request():
    return SimpleNamespace(
        prefix_cache_attempt=0,
        store_attempt=PrefixAttempt(0, queued=10.0),
        num_tokens=100,
        num_prompt_tokens=100,
    )


def test_attempt_idempotence_retries_failure_and_unique_coverage():
    m = StoreMetrics()
    r = request()
    for _ in range(7):
        m.lookup(r, "local", 100, 20)
        m.retry(r, "capacity", 11.0)
    m.admit(r, 20, 60, True, 12.0)
    m.admit(r, 20, 60, True, 12.0)
    m.restored(r, 13.0)
    m.restored(r, 14.0)
    m.first_token(r, 15.0)
    m.first_token(r, 16.0)
    assert m.counters["prefix_lookup_local_queries"] == 100
    assert m.counters["prefix_admitted_local_queries"] == 100
    assert m.counters["prefix_restored_external_hits"] == 60
    assert m.counters["admission_retry_capacity"] == 7
    assert {k: v for k, v in m.counters.items() if k.startswith("prompt_tokens_")} == {
        "prompt_tokens_host_reuse": 20,
        "prompt_tokens_host_load": 60,
        "prompt_tokens_compute": 20,
    }
    # A post-token preemption must not emit a second request TTFT/token bill.
    r.prefix_cache_attempt += 1
    m.admit(r, 0, 0, False, 17.0)
    m.first_token(r, 18.0)
    assert len(m.observations) == 3


def test_failed_restore_never_counts_restored_and_retry_is_a_new_attempt():
    m, r = StoreMetrics(), request()
    m.admit(r, 20, 60, False, 11.0)
    m.fail(r)
    m.restored(r, 12.0)
    assert not m.counters["prefix_restored_external_hits"]
    r.prefix_cache_attempt += 1
    m.lookup(r, "local", 100, 20)
    m.admit(r, 20, 0, False, 13.0)
    m.first_token(r, 14.0)
    assert m.counters["prompt_tokens_gpu"] == 20
    assert m.counters["prompt_tokens_compute"] == 80
    assert not m.counters["prompt_tokens_host_load"]


def test_overlap_assignment_preserves_unique_half_open_coverage():
    intervals = [(0, 100, "compute")]
    cover(intervals, 0, 40, "gpu")
    cover(intervals, 20, 70, "host_reuse")
    cover(intervals, 70, 90, "host_load")
    cover(intervals, 75, 80, "ssd")
    assert sum(b - a for a, b, _ in intervals) == 100
    assert all(
        intervals[i][1] == intervals[i + 1][0] for i in range(len(intervals) - 1)
    )
    for i in range(1000):
        cover(intervals, 90 + i, 91 + i, "compute")
    assert len(intervals) == 6


def test_export_is_bounded_low_frequency_delta_and_prometheus():
    m = StoreMetrics()
    for _ in range(9000):
        m.step_cpu(1000000)
    m.observe("ttft", "mixed", 2.0)
    m.counters["host_ssd_pages_written"] += 3
    first = m.export(60.0, ledger=SimpleNamespace(hot_used=7))
    assert first["gauges"]["step_cpu_seconds_p99"] == 0.001
    assert first["gauges"]["step_cpu_samples"] == 8192
    assert first["gauges"]["gpu_hot_committed_groups"] == 7
    assert "gpu_hot_used" not in first["gauges"]
    assert m.export(119.99) is None
    second = m.export(120.0)
    assert not second["counters"]
    registry = CollectorRegistry()
    prom = StoreMetricsProm(
        partial(Counter, registry=registry),
        partial(Gauge, registry=registry),
        partial(Histogram, registry=registry),
        ["engine"],
        {0: ["0"]},
    )
    prom.observe(first, 0)
    prom.observe(second, 0)
    assert (
        registry.get_sample_value(
            "vllm:store_events_total",
            {"engine": "0", "event": "host_ssd_pages_written"},
        )
        == 3
    )
    assert (
        registry.get_sample_value(
            "vllm:store_ttft_seconds_count", {"engine": "0", "source": "mixed"}
        )
        == 1
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_mfu_disabled_does_not_register(enabled):
    from vllm.v1.metrics.perf import PerfMetricsProm

    config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_mfu_metrics=enabled)
    )
    registry = CollectorRegistry()

    class Isolated(PerfMetricsProm):
        _counter_cls = staticmethod(partial(Counter, registry=registry))

    Isolated(config, ["engine"], {0: ["0"]})
    assert bool(list(registry.collect())) == enabled


def test_receive_ssd_requires_this_attempt_verified_hold_and_phases_partition():
    from vllm.v1.kv_offload.host.page_table import PageHandle
    from vllm.v1.kv_offload.host.receive import ReceiveEntry, ReceiveSource, ReceiveTxn

    m, r = StoreMetrics(), request()
    m.retry(r, "ssd_prefetch", 11.0)
    m.prefetch_ready(r, 12.0)
    m.state(r).ssd_handles.add(PageHandle(1, 0))
    m.admit(r, 0, 80, True, 13.0)
    txn = ReceiveTxn(
        1,
        "r",
        1,
        (
            ReceiveEntry(0, ReceiveSource.NEW_LOAD, 0, 1, 1, PageHandle(1, 0)),
            ReceiveEntry(1, ReceiveSource.NEW_LOAD, 0, 2, 1, PageHandle(2, 0)),
        ),
    )
    m.received(r, txn, 40, 14.0)
    m.state(r).scheduled = 15.0
    m.first_token(r, 16.0)
    assert m.counters["prompt_tokens_ssd"] == 40
    assert m.counters["prompt_tokens_host_load"] == 40
    assert m.counters["prompt_tokens_compute"] == 20
    observations = {phase: seconds for phase, source, seconds in m.observations}
    assert observations == {
        "ttft": 6.0,
        "queue": 3.0,
        "scheduled_to_token": 1.0,
        "ssd_prefetch": 1.0,
        "receive": 1.0,
    }
    assert all(source == "mixed" for _, source, _ in m.observations)


def test_store_metrics_cli_switch():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    assert parser.parse_args([]).enable_store_metrics
    assert parser.parse_args(["--enable-store-metrics"]).enable_store_metrics
    assert not parser.parse_args(["--no-enable-store-metrics"]).enable_store_metrics


def test_late_job_receipt_cannot_restore_or_fail_a_new_attempt_or_reused_id():
    m, r = StoreMetrics(), request()
    m.admit(r, 0, 80, True, 11.0)
    old = m.state(r)
    # Generation advances before the next lookup materializes its state.
    r.prefix_cache_attempt += 1
    m.complete_receive(r, old, now=12.0)
    m.admit(r, 0, 60, True, 13.0)
    new = m.state(r)
    m.complete_receive(r, old, ok=False, now=14.0)
    assert not new.failed and not new.restored
    # A different Request may reuse the same string ID and local generation.
    other = request()
    m.admit(other, 0, 70, True, 15.0)
    m.complete_receive(other, old, now=16.0)
    assert not m.state(other).restored
    m.complete_receive(r, new, now=17.0)
    assert m.counters["prefix_restored_external_hits"] == 60
    assert m.counters["prefix_receipt_stale"] == 3


def test_cancelled_before_first_token_is_distinct_from_restored():
    m, r = StoreMetrics(), request()
    m.admit(r, 0, 80, True, 11.0)
    r.is_finished = lambda: True
    m.complete_receive(r, m.state(r), now=12.0)
    assert m.counters["prefix_restore_cancelled"] == 1
    assert not m.counters["prefix_restored_external_hits"]


def test_preemption_preserves_first_schedule_and_pre_schedule_wait_accounting():
    m, r = StoreMetrics(), request()
    m.admit(r, 0, 0, False, 11.0)
    m.state(r).scheduled = 12.0
    m.preempt(r, 13.0)
    r.prefix_cache_attempt += 1
    m.admit(r, 0, 80, False, 14.0)
    m.restored(r, 15.0)
    m.first_token(r, 17.0)
    phases = {phase: value for phase, source, value in m.observations}
    assert phases["queue"] == 2.0
    assert phases["scheduled_to_token"] == 5.0
    assert phases["receive"] == 1.0
    assert phases["ttft"] == 7.0
    assert m.counters["prefix_attempt_preempted"] == 1


def test_each_lookup_retry_has_a_new_id_but_prefetch_wait_is_request_scoped():
    m, r = StoreMetrics(), request()
    m.begin_lookup(r)
    first = m.state(r)
    m.lookup(r, "local", 100, 0)
    m.lookup(r, "local", 100, 0)
    m.retry(r, "ssd_prefetch", 11.0)
    m.begin_lookup(r)
    m.lookup(r, "local", 100, 20)
    assert m.state(r) is not first
    m.prefetch_ready(r, 13.0)
    m.admit(r, 20, 0, False, 14.0)
    assert m.counters["prefix_lookup_local_queries"] == 200
    assert m.counters["prefix_lookup_local_hits"] == 20
    assert m.counters["prefix_admitted_local_queries"] == 100
    assert m.state(r).prefetch_seconds == 2.0


def test_explicit_bypass_and_absent_tiers_do_not_create_queries():
    m, r = StoreMetrics(), request()
    r.skip_reading_prefix_cache = True
    m.lookup(r, "local", 100, 0)
    m.lookup(r, "external", 100, 0)
    m.admit(r, 0, 0, False, 11.0, local_query=False, external_query=False)
    assert not m.state(r).lookup
    assert not any(v for k, v in m.counters.items() if "queries" in k)


def test_restored_query_denominators_are_frozen_at_admission():
    m, r = StoreMetrics(), request()
    m.lookup(r, "local", 100, 20)
    m.lookup(r, "external", 80, 60)
    m.admit(r, 20, 60, False, 11.0)
    r.num_tokens = 200
    m.restored(r, 12.0)
    for phase in ("admitted", "restored"):
        assert m.counters[f"prefix_{phase}_local_queries"] == 100
        assert m.counters[f"prefix_{phase}_external_queries"] == 80


def test_full_external_hit_keeps_mandatory_final_token_replay_in_compute():
    from vllm.v1.kv_offload.host.page_table import PageHandle
    from vllm.v1.kv_offload.host.receive import ReceiveEntry, ReceiveSource, ReceiveTxn

    m, r = StoreMetrics(), request()
    m.admit(r, 0, 100, True, 11.0)
    txn = ReceiveTxn(
        1,
        "r",
        1,
        (ReceiveEntry(0, ReceiveSource.NEW_LOAD, 0, 1, 1, PageHandle(1, 0)),),
    )
    m.received(r, txn, 100, 12.0)
    m.first_token(r, 13.0)
    assert m.counters["prompt_tokens_host_load"] == 99
    assert m.counters["prompt_tokens_compute"] == 1
    assert m.counters["prefix_restored_external_hits"] == 100


@pytest.mark.parametrize("local_blocks", [0, 2, 4])
def test_external_prefix_receipt_classifies_actual_main_sources(local_blocks):
    from tests.v1.core.long_context_kv_utils import SPAN
    from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
        lookup_allocate,
        setup_receive,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
    from vllm.v1.kv_offload.base import make_offload_key
    from vllm.v1.kv_offload.host.page_table import Result
    from vllm.v1.outputs import KVConnectorOutput

    scheduler, connector, r, _ = setup_receive(local_blocks=local_blocks)
    hits = scheduler.kv_cache_manager.host_kv_hits
    table = scheduler.host_history_pool.host_pages
    chain = chain_hash_from_request(r.block_hashes, 16)
    for index in range(local_blocks, 4):
        h = table.reserve()
        table.producer_result(h, compute=Result.OK, seal=Result.OK)
        table.publish(h, make_offload_key(chain(hits.rule, index), hits.rule.group_id))
    m = connector.store_metrics = StoreMetrics()
    m.begin_lookup(r)
    job_id, txn = lookup_allocate(scheduler, connector, r)
    # The complete multi-group local prefix is zero even with four bound MAINs.
    m.admit(r, 0, 4 * SPAN, True, 11.0)
    assert connector._jobs[job_id].metric_attempt is r.store_attempt
    table.complete_step(2, table.all_ranks)
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: connector.config.num_workers},
                receive_results={job_id: (table.all_ranks, 0)},
            )
        )
    )
    assert job_id not in connector._jobs
    assert r.store_attempt.restored and not r.store_attempt.failed
    m.first_token(r, 13.0)
    assert m.counters["prefix_restored_external_hits"] == 4 * SPAN
    assert m.counters["prompt_tokens_host_reuse"] == local_blocks * SPAN
    assert m.counters["prompt_tokens_host_load"] == (4 - local_blocks) * SPAN
    assert m.counters["prompt_tokens_compute"] == r.num_prompt_tokens - 4 * SPAN
    assert not m.counters["prompt_tokens_gpu"]
    assert not m.counters["prompt_tokens_ssd"]
    assert len(txn.new_loads) == 4 - local_blocks
