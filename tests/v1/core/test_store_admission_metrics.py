# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from tests.v1.core.test_history_shared_region import _host_scheduler
from tests.v1.core.test_long_context_admission import make_requests
from tests.v1.core.test_scheduler_kv_capacity import step


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
