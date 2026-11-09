# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.v1.core.test_async_scheduler import _make_model_runner_output
from tests.v1.core.test_history_shared_region import _host_scheduler
from tests.v1.core.test_long_context_admission import make_requests
from tests.v1.core.test_scheduler_kv_capacity import step
from tests.v1.core.utils import create_requests, create_scheduler


@pytest.mark.parametrize("hybrid", [False, True], ids=["single-group", "hybrid"])
@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_first_token_accounting_after_reset(hybrid, async_scheduling, monkeypatch):
    from tests.v1.core.test_long_context_admission import build
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    if hybrid:
        # Plan the deployed QSA/GDN geometry with the async in-flight allowance.
        base = build(None, async_scheduling=async_scheduling)
        scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
        scheduler = scheduler_cls(
            vllm_config=base.vllm_config,
            kv_cache_config=base.kv_cache_config,
            structured_output_manager=base.structured_output_manager,
            block_size=16,
            hash_block_size=16,
            log_stats=True,
        )
        assert len(scheduler.kv_cache_config.kv_cache_groups) > 1
        (request,) = make_requests(1, 128, 3, tag="reset-metrics")
    else:
        scheduler = create_scheduler(
            async_scheduling=async_scheduling, enable_prefix_caching=True
        )
        (request,) = create_requests(num_requests=1, num_tokens=64, max_tokens=3)

    assert scheduler.vllm_config.observability_config.enable_store_metrics
    metrics = scheduler.store_metrics
    assert metrics is not None
    metrics.next_export = float("inf")
    now = 100.0
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.monotonic", lambda: now)
    scheduler.add_request(request)
    output = scheduler.schedule()
    if async_scheduling:
        assert request.num_output_placeholders == 1
        assert scheduler.reset_prefix_cache(reset_running_requests=True)
        assert request.async_tokens_to_discard == 1
        now = 110.0
        scheduler.update_from_output(output, _make_model_runner_output(output))
        assert request.async_tokens_to_discard == 0
        assert request.num_output_tokens == 0
        assert not request.store_attempt.first_token
        assert not metrics.observations
        assert not any(
            count
            for name, count in metrics.counters.items()
            if name.startswith("prompt_tokens_")
        )
        output = scheduler.schedule()

    now = 120.0
    scheduler.update_from_output(output, _make_model_runner_output(output))
    assert request.num_output_tokens == 1
    assert request.store_attempt.first_token
    assert [item for item in metrics.observations if item[0] == "ttft"] == [
        ("ttft", "compute", 20.0)
    ]
    assert (
        sum(
            count
            for name, count in metrics.counters.items()
            if name.startswith("prompt_tokens_")
        )
        == request.num_prompt_tokens
    )
    first_counters = metrics.counters.copy()
    first_observations = list(metrics.observations)
    while scheduler.has_unfinished_requests():
        now += 10.0
        output = scheduler.schedule()
        scheduler.update_from_output(output, _make_model_runner_output(output))
    assert request.num_output_tokens == 3
    assert metrics.counters == first_counters
    assert list(metrics.observations) == first_observations


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_first_token_accounting_precedes_streaming_turn_end(
    async_scheduling, monkeypatch
):
    scheduler = create_scheduler(
        async_scheduling=async_scheduling, enable_prefix_caching=True
    )
    (request,) = create_requests(num_requests=1, num_tokens=64, max_tokens=1)
    request.resumable = True
    metrics = scheduler.store_metrics
    assert metrics is not None
    metrics.next_export = float("inf")
    end_turn = scheduler._end_streaming_turn
    ended = []

    def checked_end_turn(request, status_before_stop):
        assert request.store_attempt.first_token
        assert metrics.counters["prompt_tokens_compute"] == request.num_prompt_tokens
        assert sum(phase == "ttft" for phase, _, _ in metrics.observations) == 1
        ended.append(request.request_id)
        end_turn(request, status_before_stop)

    monkeypatch.setattr(scheduler, "_end_streaming_turn", checked_end_turn)
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(output, _make_model_runner_output(output))
    assert ended == [request.request_id]
    assert request.num_output_tokens == 1


def test_hybrid_failed_receive_relooks_up_retained_local_prefix():
    from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
        SPAN,
        publish_sources,
        setup_receive,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.metrics.stats import PrefixCacheStats
    from vllm.v1.metrics.store import StoreMetrics
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import RequestStatus

    scheduler, offload, request, _ = setup_receive(
        local_blocks=2, keep_local_state=True
    )
    publish_sources(scheduler, request, (2, 3))
    connector = object.__new__(OffloadingConnector)
    connector.connector_scheduler, connector.connector_worker = offload, None
    scheduler.connector = connector
    scheduler.log_stats = scheduler.kv_cache_manager.log_stats = True
    scheduler.kv_cache_manager.prefix_cache_stats = PrefixCacheStats()
    scheduler.connector_prefix_cache_stats = PrefixCacheStats()
    metrics = scheduler.store_metrics = offload.store_metrics = StoreMetrics()
    metrics.next_export = float("inf")
    scheduler.current_step = 1
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    attempt = request.store_attempt
    assert attempt.local == 2 * SPAN
    assert attempt.external == 2 * SPAN
    job_id = next(iter(output.kv_connector_metadata.load_jobs))
    table = scheduler.host_history_pool.host_pages
    receipt = KVConnectorOutput(
        finished_recving={request.request_id},
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            receive_results={job_id: (table.all_ranks, 1)},
            completed_jobs={job_id: 4},
            completed_steps={2: table.all_ranks},
        ),
    )
    scheduler.update_host_kv_output(receipt)
    scheduler._update_from_kv_xfer_finished(receipt)
    assert request.num_computed_tokens == 0
    assert request.request_id in scheduler.failed_recving_kv_req_ids
    assert scheduler._try_promote_blocked_waiting_request(request)
    assert request.num_computed_tokens == 0
    assert all(
        not ids for ids in scheduler.kv_cache_manager.get_block_ids(request.request_id)
    )
    assert request.prefix_cache_attempt > attempt.attempt
    assert not attempt.first_token

    while scheduler.has_unfinished_requests():
        step(scheduler)
    fresh = request.store_attempt
    assert fresh is not attempt
    assert fresh.lookup == {"local", "external"}
    assert fresh.local == 2 * SPAN
    assert fresh.external == 0
    assert metrics.counters["prompt_tokens_host_reuse"] == 2 * SPAN
    assert (
        metrics.counters["prompt_tokens_compute"]
        == request.num_prompt_tokens - 2 * SPAN
    )
    assert (
        sum(
            count
            for name, count in metrics.counters.items()
            if name.startswith("prompt_tokens_")
        )
        == request.num_prompt_tokens
    )
    assert sum(phase == "ttft" for phase, _, _ in metrics.observations) == 1


def test_lookup_retries_do_not_inflate_admitted_prefix_tokens(monkeypatch):
    from vllm.v1.metrics.stats import PrefixCacheStats
    from vllm.v1.metrics.store import StoreMetrics

    scheduler = _host_scheduler(max_num_seqs=2)
    scheduler.log_stats = True
    scheduler.kv_cache_manager.log_stats = True
    scheduler.kv_cache_manager.prefix_cache_stats = PrefixCacheStats()
    scheduler.store_metrics = StoreMetrics()
    (request,) = make_requests(1, 1664 + 5, 2, tag="admit")
    scheduler.add_request(request)
    manager = scheduler.kv_cache_manager
    real_allocate = manager.allocate_slots
    monkeypatch.setattr(manager, "allocate_slots", lambda *a, **kw: None)
    for _ in range(3):
        scheduler.schedule()
    assert manager.prefix_cache_stats.queries == 0
    assert (
        scheduler.store_metrics.counters["prefix_lookup_local_queries"]
        == 3 * request.num_tokens
    )
    assert scheduler.store_metrics.counters["admission_retry_capacity"] == 3
    monkeypatch.setattr(manager, "allocate_slots", real_allocate)
    scheduler.schedule()
    assert manager.prefix_cache_stats.queries == request.num_tokens
    scheduler.schedule()
    assert manager.prefix_cache_stats.queries == request.num_tokens


def test_disabled_metrics_does_not_create_request_accounting_or_export():
    scheduler = _host_scheduler(max_num_seqs=2)
    scheduler.store_metrics = None
    (request,) = make_requests(1, 1664 + 5, 1, tag="disabled")
    scheduler.add_request(request)
    while scheduler.has_unfinished_requests():
        step(scheduler)
    assert request.store_attempt is None
    assert (
        not {
            "store_attempt",
            "store_queued_ts",
            "store_first_token",
            "prefix_cache_recorded_attempt",
        }
        & request.__dict__.keys()
    )
    assert scheduler._export_store_stats() is None
    assert not scheduler._store_schedule_cpu


def test_pending_external_lookup_has_no_admitted_query_until_allocation(monkeypatch):
    from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
        setup_receive,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.metrics.stats import PrefixCacheStats
    from vllm.v1.metrics.store import StoreMetrics

    scheduler, offload, request, _ = setup_receive(local_blocks=0)
    connector = object.__new__(OffloadingConnector)
    connector.connector_scheduler, connector.connector_worker = offload, None
    scheduler.connector = connector
    scheduler.log_stats = scheduler.kv_cache_manager.log_stats = True
    scheduler.kv_cache_manager.prefix_cache_stats = PrefixCacheStats()
    scheduler.connector_prefix_cache_stats = PrefixCacheStats()
    metrics = scheduler.store_metrics = StoreMetrics()
    scheduler.add_request(request)
    real_lookup = connector.get_num_new_matched_tokens
    monkeypatch.setattr(
        connector, "get_num_new_matched_tokens", lambda *a, **kw: (None, False)
    )
    for _ in range(3):
        scheduler.schedule()
    assert not scheduler.kv_cache_manager.prefix_cache_stats.queries
    assert not scheduler.connector_prefix_cache_stats.queries
    assert metrics.counters["prefix_lookup_external_queries"] == 3 * request.num_tokens
    assert not metrics.counters["prefix_lookup_external_hits"]
    assert metrics.counters["admission_retry_connector"] == 3
    monkeypatch.setattr(connector, "get_num_new_matched_tokens", real_lookup)
    scheduler.schedule()
    assert scheduler.kv_cache_manager.prefix_cache_stats.queries == request.num_tokens
    assert scheduler.connector_prefix_cache_stats.queries == request.num_tokens
    scheduler.schedule()
    assert scheduler.connector_prefix_cache_stats.queries == request.num_tokens


def test_cpu_sampling_pairs_schedule_and_output_without_retaining_old_outputs():
    from vllm.v1.metrics.stats import PrefixCacheStats
    from vllm.v1.metrics.store import CPU_SAMPLE_STRIDE, StoreMetrics

    scheduler = _host_scheduler(max_num_seqs=2)
    scheduler.log_stats = scheduler.kv_cache_manager.log_stats = True
    scheduler.kv_cache_manager.prefix_cache_stats = PrefixCacheStats()
    metrics = scheduler.store_metrics = StoreMetrics()
    metrics.next_export = float("inf")
    (request,) = make_requests(1, 1664 + 5, 20, tag="sampled")
    scheduler.add_request(request)
    while scheduler.has_unfinished_requests():
        step(scheduler)
    assert len(metrics.cpu_ns) == scheduler.current_step // CPU_SAMPLE_STRIDE
    assert metrics.cpu_ns and all(ns > 0 for ns in metrics.cpu_ns)
    assert not scheduler._store_schedule_cpu


def test_disabled_local_cache_and_absent_connector_do_not_count_queries():
    from tests.v1.core.test_long_context_admission import build, settings
    from vllm.v1.metrics.stats import PrefixCacheStats
    from vllm.v1.metrics.store import StoreMetrics

    scheduler = build(settings(main_kv_placement="host"), prefix_caching=False)
    scheduler.log_stats = scheduler.kv_cache_manager.log_stats = True
    scheduler.kv_cache_manager.prefix_cache_stats = PrefixCacheStats()
    metrics = scheduler.store_metrics = StoreMetrics()
    (request,) = make_requests(1, 1664 + 5, 2, tag="no-query")
    scheduler.add_request(request)
    scheduler.schedule()
    assert not scheduler.kv_cache_manager.prefix_cache_stats.queries
    assert not metrics.state(request).lookup
    assert not any(v for k, v in metrics.counters.items() if "queries" in k)
