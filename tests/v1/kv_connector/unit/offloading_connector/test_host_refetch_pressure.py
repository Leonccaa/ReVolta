# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from tests.v1.core.test_value_arbiter_pool import _arbiter
from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
    lookup_allocate,
    setup_receive,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput


def test_index_shrink_then_admitted_verified_receive_counts_refetch():
    scheduler, connector, request, bindings = setup_receive(
        local_blocks=4, page_state="ready"
    )
    pool = scheduler.host_history_pool
    arb = scheduler.value_arbiter = _arbiter(pool)
    connector.value_restore_hook = scheduler._value_restore
    # Shrink the index cache under shared-region pressure. Host canonical
    # pages remain READY; subsequent NEW_LOAD must restore those exact hashes.
    assert pool.donate_cached_for_value(len(bindings)) == len(bindings) == 4
    assert arb.windows[0].idx_evicted == 4
    arb.begin_step(2, 0.01)
    job, txn = lookup_allocate(scheduler, connector, request)
    table = pool.host_pages
    receipt = KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            receive_results={job: (table.all_ranks, 0)},
            completed_steps={2: table.all_ranks},
        )
    )
    scheduler.update_host_kv_output(receipt)
    connector.update_connector_output(receipt)
    assert arb.windows[0].idx_refetch_blocks == 4
    assert arb.windows[0].idx_refetch_txns == 1
    # Duplicate receipts are ignored by the production receive ledger.
    connector.update_connector_output(receipt)
    assert arb.windows[0].idx_refetch_txns == 1


def test_receive_job_carries_its_metric_attempt_until_physical_retirement():
    from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
        SPAN,
        publish_sources,
    )
    from vllm.v1.metrics.store import StoreMetrics

    for stale in (False, True):
        scheduler, connector, request, _ = setup_receive(local_blocks=0)
        publish_sources(scheduler, request, range(4))
        metrics = connector.store_metrics = StoreMetrics()
        job, txn = lookup_allocate(scheduler, connector, request)
        metrics.admit(request, 0, 4 * SPAN, True, 10.0)
        old = metrics.state(request)
        assert connector._jobs[job].metric_attempt is old
        if stale:
            request.prefix_cache_attempt += 1
            metrics.admit(request, 0, 0, True, 11.0)
        table = scheduler.host_history_pool.host_pages
        receipt = KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                receive_results={job: (table.all_ranks, 0)},
                completed_steps={2: table.all_ranks},
            )
        )
        scheduler.update_host_kv_output(receipt)
        connector.update_connector_output(receipt)
        assert metrics.counters["prefix_restored_external_hits"] == (
            0 if stale else 4 * SPAN
        )
        assert metrics.counters["prefix_receipt_stale"] == int(stale)
