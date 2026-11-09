# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real PRIVATE/SHADOW G0 reuse joined to unified STATE component leases."""

from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
    lookup_allocate,
    setup_receive,
)
from tests.v1.kv_offload.test_unified_units import publish
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.unified.manager import UnifiedStoreManager
from vllm.v1.outputs import KVConnectorOutput

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "page_state,canonical", [("private", "absent"), ("shadow", "ready")]
)
@pytest.mark.parametrize("failed", [False, True])
def test_components_join_attempt_without_revoking_shared_g0(
    page_state, canonical, failed
):
    scheduler, connector, request, bindings = setup_receive(
        page_state=page_state, canonical=canonical
    )
    pool = scheduler.host_history_pool
    table = pool.host_pages
    table.num_workers, table.all_ranks = 4, 15
    m = UnifiedStoreManager(
        SimpleNamespace(
            kv_cache_config=connector.kv_cache_config,
            extra_config={},
            cpu_page_size_per_worker=16,
        )
    )
    m.attach(pool)
    connector.manager = connector.unified = m
    scheduler.kv_cache_manager.host_kv_hits.release_components = m.release_lookup
    status = connector._req_status[request.request_id]
    for group in connector.config.kv_group_configs:
        k = status.group_states[group.group_idx].offload_keys[3]
        publish(table, k, m.kind(k))
    job_id, txn = lookup_allocate(scheduler, connector, request)
    job = connector._jobs[job_id]
    assert len(txn.components) == len(m.kind_by_group)
    assert not txn.new_loads and len(txn.entries) == 4
    assert all(c.load_generation == job.generation for c in txn.components)
    assert all(c.handle not in {e.handle for e in txn.entries} for c in txn.components)
    # G0 completion alone never retires component leases or reports success.
    table.complete_step(2, table.all_ranks)
    output = KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            receive_results={job_id: (table.all_ranks, 0)},
            completed_jobs={job_id: 4},
        )
    )
    connector.update_connector_output(output)
    assert not output.finished_recving and job_id in connector._jobs
    assert all(c.lease_id in table.get(c.handle).leases for c in txn.components)
    # Both component groups retire under this attempt/generation. Duplicate
    # rank receipts do not imply all workers completed.
    for _ in range(3):
        connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=OffloadingWorkerMetadata(
                    job_results={(job_id, job.generation): (1, 1 if failed else 0)},
                )
            )
        )
    assert job_id in connector._jobs
    output = KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            job_results={(job_id, job.generation): (table.all_ranks, 0)},
        )
    )
    connector.update_connector_output(output)
    assert output.finished_recving == {request.request_id}
    assert bool(connector.take_failed_loads()) == failed
    assert all(c.lease_id not in table.get(c.handle).leases for c in txn.components)
    # A failed STATE receive only voids this receiver; original bindings and
    # PRIVATE/SHADOW physical versions remain owned by their original IDs.
    for hid, (hgen, handle) in bindings.items():
        assert pool.bindings[hid] == handle
        assert (hid, hgen) in table.get(handle).binds


def test_seal_only_failure_exit_reclaims_keyless_cached_pages():
    scheduler, connector, request, bindings = setup_receive(page_state="private")
    pool = scheduler.host_history_pool
    m = UnifiedStoreManager(
        SimpleNamespace(
            kv_cache_config=connector.kv_cache_config,
            extra_config={},
            cpu_page_size_per_worker=16,
        )
    )
    m.attach(pool)
    assert pool.elastic
    assert all(pool.host_pages.get(h).key is None for _, h in bindings.values())
    assert any(pool._cached_page_reclaimable([hid]) for hid in bindings)
    # Exhaust all unbound capacity so unified admission must revoke a cached
    # binding left by successful computation / failed seal / producer exit.
    table = pool.host_pages
    spare = []
    while table.net_headroom() > 0:
        h = table.reserve()
        assert h is not None
        table.pin(h, 9876)
        spare.append(h)
    assert m.admit_build(request.request_id, 1)
    assert pool.stats["cached_evicted_for_host"] > 0
    assert m.ledger.requests[request.request_id].unused == 1
    m.cancel_request(request.request_id)
    for h in spare:
        table.release(h, 9876)


@pytest.mark.parametrize("local_blocks,quota", [(0, 16), (4, 16), (0, 8)])
def test_connector_restores_seven_cold_state_groups(tmp_path, local_blocks, quota):
    import time

    from tests.v1.kv_offload.test_host_page_ssd import LAYOUT, ROW, options
    from vllm.v1.kv_offload.host.persist import HostPageSsd

    scheduler, connector, request, _ = setup_receive(local_blocks=local_blocks)
    pool = scheduler.host_history_pool
    table = pool.host_pages
    m = UnifiedStoreManager(
        SimpleNamespace(
            kv_cache_config=connector.kv_cache_config,
            extra_config={},
            cpu_page_size_per_worker=16,
        )
    )
    m.attach(pool)
    connector.manager = connector.unified = m
    hits = scheduler.kv_cache_manager.host_kv_hits
    hits.release_components = m.release_lookup
    backing = bytearray(len(table.pages) * ROW)
    spec, opts = options(tmp_path)
    opts["prefetch_max_per_request"] = quota
    ssd = HostPageSsd(
        table,
        memoryview(backing).cast("B", shape=[len(table.pages), ROW]),
        spec,
        LAYOUT,
        opts,
    )
    m.attach_ssd(ssd)
    connector.host_page_ssd = ssd
    status = connector._req_status[request.request_id]
    keys = [
        status.group_states[g.group_idx].offload_keys[3]
        for g in connector.config.kv_group_configs
    ]
    assert len(keys) == 7
    try:
        handles = [publish(table, k, m.kind(k)) for k in keys]
        if not local_blocks:
            from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
            from vllm.v1.kv_offload.base import make_offload_key

            chain = chain_hash_from_request(request.block_hashes, 16)
            handles += [
                publish(
                    table, make_offload_key(chain(hits.rule, i), hits.rule.group_id)
                )
                for i in range(4)
            ]
        deadline = time.monotonic() + 5
        while not all(table.is_persisted(table.get(h)) for h in handles):
            assert time.monotonic() < deadline
            ssd.step()
            time.sleep(0.002)
        for h in handles:
            table._recycle(table.get(h))
        _, local = scheduler.kv_cache_manager.get_computed_blocks(request)
        result = (None, False)
        while result[0] is None:
            assert time.monotonic() < deadline
            result = connector.get_num_new_matched_tokens(request, local)
            ssd.step()
            time.sleep(0.002)
        if quota < len(handles):
            assert result == (0, False)
            assert ssd.stats_counter["prefetch_issued"] == quota
            assert not ssd.holds(request.request_id)
            return
        assert result[0] > 0 and result[1]
        assert all(m.is_ready_resident(k) for k in keys)
        assert ssd.stats_counter["prefetch_verified"] == len(handles)
        assert not ssd.holds(request.request_id)
        assert m._lookup_holds[request.request_id]
    finally:
        hits.release_lookup(request.request_id)
        ssd.close(io_grace_s=5)


def test_idle_worker_receipt_does_not_scan_transfer_table():
    _, connector, _, _ = setup_receive()

    class NoScan(dict):
        def items(self):
            raise AssertionError("idle receipt scanned unrelated jobs")

    connector._jobs = NoScan(connector._jobs)
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(),
        )
    )
