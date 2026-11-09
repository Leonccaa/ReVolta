# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""receive transactions source splitting through the real allocator and connector entry
points."""

from unittest.mock import Mock

import pytest

from tests.v1.core.test_long_context_admission import build, make_requests
from tests.v1.core.test_long_context_host_main_kv import SPAN, host_settings
from tests.v1.kv_connector.unit.offloading_connector.test_host_kv_hit import (
    connector_for,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.host.page_table import Result
from vllm.v1.kv_offload.host.receive import HostReceiveSpec, ReceiveSource
from vllm.v1.outputs import KVConnectorOutput

pytestmark = pytest.mark.cpu_test


def setup_receive(
    *,
    local_blocks=4,
    page_state="private",
    canonical="absent",
    workers=4,
    keep_local_state=False,
    speculative_tokens=0,
):
    scheduler = build(
        host_settings(host_pool_tokens=24 * SPAN),
        max_num_seqs=2,
        speculative_tokens=speculative_tokens,
    )
    manager = scheduler.kv_cache_manager
    pool = scheduler.host_history_pool
    hits = manager.host_kv_hits
    producer_tokens = local_blocks * SPAN if keep_local_state else 6 * SPAN
    (producer,) = make_requests(1, producer_tokens, 1, tag="producer")
    pool.start_step(1)
    if local_blocks:
        assert manager.allocate_slots(producer, local_blocks * SPAN) is not None
    tickets = pool.finish_schedule([producer.request_id])
    bindings = {}
    for ticket in tickets:
        if page_state == "shadow":
            if canonical == "loading":
                other = pool.host_pages.reserve(loading_key=ticket.key)
            else:
                other = pool.host_pages.reserve()
                record = pool.host_pages.get(other)
                record.compute_result = record.seal_result = Result.OK
                pool.host_pages.publish(other, ticket.key)
            assert other is not None
        pool.host_pages.seal_result(
            ticket,
            pool.host_pages.all_ranks,
            pool.host_pages.all_ranks if page_state == "private" else 0,
        )
        if page_state == "shadow" and canonical == "evicted":
            pool.host_pages._recycle(pool.host_pages.get(other))
        bindings[ticket.hid] = (ticket.hgen, ticket.handle)
    pool.host_pages.complete_step(1, pool.host_pages.all_ranks)
    manager.free(producer)
    manager.take_new_block_ids()
    pool.binding_updates.clear()
    manager.new_step_starts()
    # Leave only G0 on the GPU. The common local hit is L=0.
    for group_id, group_manager in enumerate(manager.coordinator.single_type_managers):
        if not keep_local_state and group_id != pool.host_group_id:
            group_manager.block_pool.evict_blocks(
                {
                    b.block_id
                    for b in group_manager.block_pool.blocks
                    if b.block_hash is not None
                    and get_group_id(b.block_hash) == group_id
                }
            )
    (request,) = make_requests(1, 6 * SPAN, 1, tag="receiver")
    connector = connector_for(
        manager, dcp_world=2, workers=workers, eagle=bool(speculative_tokens)
    )
    hits.attach_connector(connector)
    connector.on_new_request(request)
    status = connector._req_status[request.request_id]
    status.update_offload_keys()
    keys = [
        group.offload_keys[3]
        for group_id, group in enumerate(status.group_states)
        if group_id in connector._group_config_by_idx
    ]
    connector.manager.prepare_store(keys, status.req_context)
    connector.manager.complete_store(keys, status.req_context)
    pool.start_step(2)
    return scheduler, connector, request, bindings


def lookup_allocate(scheduler, connector, request):
    manager = scheduler.kv_cache_manager
    local, tokens = manager.get_computed_blocks(request)
    external, asynchronous = connector.get_num_new_matched_tokens(request, tokens)
    assert external == 4 * SPAN - tokens and asynchronous
    assert (
        manager.allocate_slots(
            request,
            0,
            num_new_computed_tokens=tokens,
            new_computed_blocks=local,
            num_external_computed_tokens=external,
            delay_cache_blocks=True,
        )
        is not None
    )
    connector.update_state_after_alloc(
        request, manager.get_blocks(request.request_id), external
    )
    assert not connector.take_failed_loads()
    job_id, job = next(iter(connector._current_batch_load_jobs.items()))
    assert isinstance(job.transfer_spec[0], HostReceiveSpec)
    return job_id, job.transfer_spec[0].txn


@pytest.mark.parametrize(
    "page_state,canonical",
    [
        ("private", "absent"),
        ("shadow", "ready"),
        ("shadow", "loading"),
        ("shadow", "evicted"),
        ("ready", "ready"),
    ],
)
def test_reuse_private_shadow_without_canonical_probe_or_pin(
    monkeypatch, page_state, canonical
):
    scheduler, connector, request, bindings = setup_receive(
        page_state=page_state, canonical=canonical
    )
    pool = scheduler.host_history_pool
    table = pool.host_pages
    borrowed_keys = {
        make_offload_key(
            request.block_hashes[(i + 1) * (SPAN // 16) - 1], pool.host_group_id
        )
        for i in range(4)
    }
    lookup = table.lookup

    def external_lookup(key):
        assert key not in borrowed_keys, "canonical probe of a borrowed G0 block"
        return lookup(key)

    monkeypatch.setattr(table, "lookup", Mock(side_effect=external_lookup))
    monkeypatch.setattr(table, "pin", Mock(side_effect=AssertionError("canonical pin")))
    job_id, txn = lookup_allocate(scheduler, connector, request)
    assert len(txn.entries) == 4 and not txn.new_loads
    assert all(e.source == ReceiveSource.REUSE_LOCAL for e in txn.entries)
    assert {(e.hid, e.hgen, e.handle) for e in txn.entries} == {
        (hid, hgen, handle) for hid, (hgen, handle) in bindings.items()
    }
    assert (
        connector._current_batch_load_jobs[job_id].transfer_spec[1].group_sizes[0] == 0
    )
    table.complete_step(2, table.all_ranks)
    output = KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            completed_jobs={job_id: 1},
            receive_results={job_id: (table.all_ranks, 0)},
        )
    )
    connector.update_connector_output(output)
    assert output.finished_recving == {request.request_id}
    assert not connector.take_failed_loads()
    assert all(
        call.args[0] not in borrowed_keys for call in table.lookup.call_args_list
    )
    assert not table.pin.called
    manager = scheduler.kv_cache_manager
    manager.cache_blocks(request, 4 * SPAN)
    manager.free(request)
    for hid, (_, handle) in bindings.items():
        assert pool.bindings[hid] == handle and pool.blocks[hid].block_hash is not None


def test_partial_reuse_and_new_load_keep_absolute_source_indices():
    scheduler, connector, request, bindings = setup_receive(local_blocks=2)
    hits = scheduler.kv_cache_manager.host_kv_hits
    table = scheduler.host_history_pool.host_pages
    chain_hash = chain_hash_from_request(request.block_hashes, 16)
    handles = {}
    for index in (2, 3):
        key = make_offload_key(chain_hash(hits.rule, index), hits.rule.group_id)
        handle = table.reserve()
        page = table.get(handle)
        page.compute_result = page.seal_result = Result.OK
        table.publish(handle, key)
        handles[index] = handle
    job_id, txn = lookup_allocate(scheduler, connector, request)
    assert [e.logical_index for e in txn.entries] == [0, 1, 2, 3]
    assert [e.source for e in txn.entries] == [
        ReceiveSource.REUSE_LOCAL,
        ReceiveSource.REUSE_LOCAL,
        ReceiveSource.NEW_LOAD,
        ReceiveSource.NEW_LOAD,
    ]
    assert [index for index, _ in hits.pinned_pages(request.request_id)] == [2, 3]
    for entry in txn.new_loads:
        assert entry.handle == handles[entry.logical_index]
        assert scheduler.host_history_pool.bindings[entry.hid] == entry.handle
        assert table.get(entry.handle).io == {job_id}
    assert set(bindings).isdisjoint(e.hid for e in txn.new_loads)


def test_receive_masks_deduplicate_and_failure_waits_for_watermark():
    scheduler, connector, request, _ = setup_receive(workers=2)
    table = scheduler.host_history_pool.host_pages
    # This fixture's scheduler has one worker. Exercise the same page table's
    # multi-rank completion interface without changing physical storage.
    table.num_workers, table.all_ranks = 2, 3
    job_id, txn = lookup_allocate(scheduler, connector, request)
    for _ in range(3):
        output = KVConnectorOutput(
            finished_recving={request.request_id},
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: 1},
                receive_results={job_id: (1, 1)},
            ),
        )
        connector.update_connector_output(output)
        assert output.finished_recving == set()
        assert job_id in connector._jobs
    assert txn.completed_ranks == 1
    output.kv_connector_worker_meta = OffloadingWorkerMetadata(
        completed_jobs={job_id: 1}, receive_results={job_id: (2, 0)}
    )
    connector.update_connector_output(output)
    assert output.finished_recving == set() and job_id in connector._jobs
    table.complete_step(2, 3)
    output.kv_connector_worker_meta = OffloadingWorkerMetadata()
    connector.update_connector_output(output)
    assert output.finished_recving == {request.request_id}
    assert connector.take_failed_loads() == {request.request_id}
    assert job_id not in connector._jobs
    connector.update_connector_output(output)
    assert output.finished_recving == set()
    assert connector.take_failed_loads() == set()


def test_state_check_keeps_local_L_when_g0_reuses_to_R(monkeypatch):
    scheduler, connector, request, _ = setup_receive()
    seen = []
    original = connector.hit_contract.check

    def check(end, start, block_ok, chain_hash, **kwargs):
        seen.append((end, start, kwargs.get("group_starts")))
        return original(end, start, block_ok, chain_hash, **kwargs)

    monkeypatch.setattr(connector.hit_contract, "check", check)
    assert connector.get_num_new_matched_tokens(request, 0) == (4 * SPAN, True)
    assert seen == [(4 * SPAN, 0, {0: 4 * SPAN})]
    connector.manager.reset_cache()
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)


def test_only_failed_borrower_returns_its_reference(monkeypatch):
    scheduler, connector, request, bindings = setup_receive()
    manager, pool = scheduler.kv_cache_manager, scheduler.host_history_pool
    job_id, txn = lookup_allocate(scheduler, connector, request)
    (other,) = make_requests(1, 6 * SPAN, 1, tag="other-reader")
    group = manager.coordinator.single_type_managers[pool.host_group_id]
    borrowed = [pool.blocks[e.hid] for e in txn.entries]
    pool.touch(borrowed)
    group.req_to_blocks[other.request_id] = list(borrowed)
    refs = {b.block_id: b.ref_cnt for b in borrowed}
    hashes = {b.block_id: b.block_hash for b in borrowed}
    table = pool.host_pages
    table.complete_step(2, table.all_ranks)
    output = KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            completed_jobs={job_id: 1},
            failed_jobs={job_id: 1},
            receive_results={job_id: (table.all_ranks, table.all_ranks)},
        )
    )
    connector.update_connector_output(output)
    assert connector.take_failed_loads() == {request.request_id}
    manager.free(request)
    for hid, (hgen, handle) in bindings.items():
        assert pool.blocks[hid].ref_cnt == refs[hid] - 1
        assert pool.blocks[hid].block_hash == hashes[hid]
        assert pool.bindings[hid] == handle and pool.hgen[hid] == hgen
        assert table.get(handle).compute_result == Result.OK
    manager.free(other)


def publish_sources(scheduler, request, indices):
    hits = scheduler.kv_cache_manager.host_kv_hits
    table = scheduler.host_history_pool.host_pages
    chain_hash = chain_hash_from_request(request.block_hashes, 16)
    handles = {}
    for index in indices:
        key = make_offload_key(chain_hash(hits.rule, index), hits.rule.group_id)
        handle = table.reserve()
        page = table.get(handle)
        page.compute_result = page.seal_result = Result.OK
        table.publish(handle, key)
        handles[index] = handle
    return handles


@pytest.mark.parametrize("cancel", [False, True])
def test_new_load_failure_keeps_sources_and_targets_until_retirement(cancel):
    from vllm.v1.request import RequestStatus

    scheduler, connector, request, _ = setup_receive(local_blocks=0)
    manager, pool = scheduler.kv_cache_manager, scheduler.host_history_pool
    table = pool.host_pages
    handles = publish_sources(scheduler, request, range(4))
    job_id, txn = lookup_allocate(scheduler, connector, request)
    assert len(txn.new_loads) == 4
    if cancel:
        request.status = RequestStatus.FINISHED_ABORTED
        connector.request_finished(request)
        assert txn.cancelled
    output = KVConnectorOutput(
        finished_recving={request.request_id},
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            completed_jobs={job_id: 4},
            failed_jobs={job_id: 1},
            receive_results={job_id: (table.all_ranks, 1)},
        ),
    )
    connector.update_connector_output(output)
    assert output.finished_recving == set()
    assert not connector.take_failed_loads()
    assert job_id in connector._jobs
    table.complete_step(3, table.all_ranks)
    assert table.watermark == 1
    for entry in txn.entries:
        page = table.get(entry.handle)
        assert entry.handle == handles[entry.logical_index]
        assert page.io == {job_id} and page.leases
        assert pool.blocks[entry.hid].ref_cnt == 1
        assert pool.bindings[entry.hid] == entry.handle
        assert not table.reclaimable(page)
    table.complete_step(2, table.all_ranks)
    output.kv_connector_worker_meta = OffloadingWorkerMetadata()
    connector.update_connector_output(output)
    assert output.finished_recving == {request.request_id}
    assert connector.take_failed_loads() == (set() if cancel else {request.request_id})
    manager.free(request)
    for entry in txn.entries:
        page = table.get(entry.handle)
        assert pool.blocks[entry.hid].ref_cnt == 0
        assert pool.blocks[entry.hid].block_hash is None
        assert entry.hid not in pool.bindings
        assert not (page.io or page.leases or page.binds)
        assert table.reclaimable(page) and table.is_ready(entry.handle)


@pytest.mark.parametrize("full_sequence", [False, True])
def test_source_plan_rejection_releases_only_lookup_leases(full_sequence):
    scheduler, connector, request, bindings = setup_receive(local_blocks=2)
    manager, pool = scheduler.kv_cache_manager, scheduler.host_history_pool
    handles = publish_sources(scheduler, request, (2, 3))
    local, tokens = manager.get_computed_blocks(request)
    assert connector.get_num_new_matched_tokens(request, tokens) == (4 * SPAN, True)
    assert manager.host_kv_hits._pending[request.request_id].pages
    assert (
        manager.allocate_slots(
            request,
            0,
            new_computed_blocks=local,
            num_external_computed_tokens=4 * SPAN,
            delay_cache_blocks=True,
            reserved_blocks=pool.get_num_free_blocks(),
            full_sequence_must_fit=full_sequence,
        )
        is None
    )
    assert not manager.host_kv_hits._pending and not manager.host_kv_hits._reuse
    assert all(not pool.host_pages.get(h).leases for h in handles.values())
    for hid, (hgen, handle) in bindings.items():
        assert pool.hgen[hid] == hgen and pool.bindings[hid] == handle
        assert pool.blocks[hid].ref_cnt == 0 and pool.blocks[hid].block_hash is not None


@pytest.mark.parametrize("damage", ["compute", "generation", "epoch", "invalid"])
def test_local_validator_rejects_stale_or_failed_producers(damage):
    from dataclasses import replace

    scheduler, _, request, bindings = setup_receive()
    hits, pool = scheduler.kv_cache_manager.host_kv_hits, scheduler.host_history_pool
    hid, (_, handle) = next(iter(bindings.items()))
    key_hash = chain_hash_from_request(request.block_hashes, 16)(hits.rule, 0)
    assert hits.local_validator(0, 0, key_hash, hid)
    page = pool.host_pages.get(handle)
    if damage == "compute":
        page.compute_result = Result.FAILED
    elif damage == "generation":
        pool.hgen[hid] += 1
    elif damage == "epoch":
        page.handle = replace(handle, epoch=handle.epoch + 1)
    else:
        page.valid = False
    assert not hits.local_validator(0, 0, key_hash, hid)
    assert hits.plan_reuse(request, 0, 16) == 0


def test_full_local_private_hit_neither_probes_nor_pins_canonical(monkeypatch):
    scheduler, _, request, bindings = setup_receive(keep_local_state=True)
    manager, pool = scheduler.kv_cache_manager, scheduler.host_history_pool
    table = pool.host_pages
    monkeypatch.setattr(
        table, "lookup", Mock(side_effect=AssertionError("local probe"))
    )
    monkeypatch.setattr(table, "pin", Mock(side_effect=AssertionError("local pin")))
    local, tokens = manager.get_computed_blocks(request)
    assert tokens == 4 * SPAN
    result = manager.allocate_slots(
        request, 1, new_computed_blocks=local, num_new_computed_tokens=tokens
    )
    assert result is not None, (
        manager.admission_ledger.blocked,
        [p.get_num_free_blocks() for p in manager.coordinator.block_pools],
    )
    assert all(pool.bindings[hid] == handle for hid, (_, handle) in bindings.items())
    assert not table.lookup.called and not table.pin.called
    manager.free(request)


@pytest.mark.parametrize("need", ["state", "window"])
def test_source_specific_start_does_not_skip_state_or_window(monkeypatch, need):
    from dataclasses import replace

    from vllm.v1.core.kv_cache_hit_contract import HitNeed

    _, connector, request, _ = setup_receive()
    if need == "window":
        rule = connector.hit_contract.rules[2]
        connector.hit_contract.rules[2] = replace(
            rule, need=HitNeed.WINDOW, window_tokens=4 * SPAN - 1
        )
    # Force a backend overclaim, so the independent contract must check the
    # missing state/window from L. L=2 blocks, R=E=4 blocks.
    monkeypatch.setattr(connector, "_lookup", lambda *args: 2 * SPAN)
    connector.manager.reset_cache()
    assert connector.get_num_new_matched_tokens(request, 2 * SPAN) == (0, False)
    assert connector.hit_contract.stats.rejected_hits == 1


@pytest.mark.parametrize("local_blocks", [0, 2])
@pytest.mark.parametrize("failed", [False, True])
def test_scheduler_receive_completion_and_recompute(local_blocks, failed):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.request import RequestStatus

    scheduler, connector, request, bindings = setup_receive(local_blocks=local_blocks)
    publish_sources(scheduler, request, range(local_blocks, 4))
    wrapper = object.__new__(OffloadingConnector)
    wrapper.connector_scheduler, wrapper.connector_worker = connector, None
    scheduler.connector = wrapper
    scheduler.current_step = 1
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    job_id, job = next(iter(output.kv_connector_metadata.load_jobs.items()))
    txn = job.transfer_spec[0].txn
    assert [entry.source for entry in txn.entries] == (
        [ReceiveSource.REUSE_LOCAL] * local_blocks
        + [ReceiveSource.NEW_LOAD] * (4 - local_blocks)
    )
    assert set(bindings).isdisjoint(output.new_block_ids_to_zero or ())
    assert {e.hid for e in txn.new_loads}.issubset(output.new_block_ids_to_zero or ())
    assert set(bindings).isdisjoint(hid for hid, _ in output.host_kv.bindings)
    table = scheduler.host_history_pool.host_pages
    receipt = KVConnectorOutput(
        finished_recving={request.request_id},
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            receive_results={job_id: (table.all_ranks, int(failed))},
            completed_jobs={job_id: 4},
            completed_steps={2: table.all_ranks},
        ),
    )
    scheduler.update_host_kv_output(receipt)
    scheduler._update_from_kv_xfer_finished(receipt)
    assert request.num_computed_tokens == (0 if failed else 4 * SPAN)
    assert (request.request_id in scheduler.failed_recving_kv_req_ids) is failed
    assert scheduler._try_promote_blocked_waiting_request(request)
    assert request.status == RequestStatus.WAITING
    pool = scheduler.host_history_pool
    for hid, (hgen, handle) in bindings.items():
        assert pool.hgen[hid] == hgen and pool.bindings[hid] == handle
        assert pool.blocks[hid].block_hash is not None
    assert (request.request_id in connector._skip_external_once) is failed
    if not failed:
        # Loaded READY pages publish GPU hashes without becoming new compute
        # writers. A same-step preemption must leave their sources intact.
        pool.cancel_scheduled(request.request_id)
        assert pool.finish_schedule([request.request_id]) == []
        for entry in txn.new_loads:
            assert pool.blocks[entry.hid].block_hash is not None
            assert table.is_ready(entry.handle)
            assert table.get(entry.handle).compute_result == Result.OK
        scheduler.kv_cache_manager.free(request)
    assert not scheduler.kv_cache_manager.host_kv_hits._active


def test_g0_source_cap_still_records_a_missing_state_junction():
    _, connector, request, _ = setup_receive()
    status = connector._req_status[request.request_id]
    connector.manager.reset_cache()
    keys = [
        g.offload_keys[1]
        for i, g in enumerate(status.group_states)
        if i in connector._group_config_by_idx
    ]
    connector.manager.prepare_store(keys, status.req_context)
    connector.manager.complete_store(keys, status.req_context)
    assert connector.get_num_new_matched_tokens(request, 0) == (2 * SPAN, True)
    assert request.shared_prefix_boundary == 4 * SPAN


def test_external_slice_returns_prefix_with_logical_indices():
    scheduler, _, request, _ = setup_receive(local_blocks=2)
    handles = publish_sources(scheduler, request, (2, 3))
    hits = scheduler.kv_cache_manager.host_kv_hits
    prefix = hits.pin_hit(
        request.request_id,
        chain_hash_from_request(request.block_hashes, 16),
        start=2 * SPAN,
        end=4 * SPAN,
    )
    assert prefix.start_block == 2 and prefix.available_tokens == 4 * SPAN
    assert [(index, page.handle) for index, page in prefix.entries] == list(
        handles.items()
    )
    assert hits.pinned_pages(request.request_id) == prefix.entries
    hits.release_lookup(request.request_id)
    assert all(
        not scheduler.host_history_pool.host_pages.get(h).leases
        for h in handles.values()
    )


def test_pin_exception_rolls_back_new_slice_and_preserves_prior_prefix(monkeypatch):
    scheduler, _, request, _ = setup_receive(local_blocks=0)
    handles = publish_sources(scheduler, request, range(4))
    hits = scheduler.kv_cache_manager.host_kv_hits
    table = scheduler.host_history_pool.host_pages
    chain = chain_hash_from_request(request.block_hashes, 16)
    prior = hits.pin_hit(request.request_id, chain, start=0, end=2 * SPAN)
    pin = table.pin

    def fail(handle, lease):
        if handle == handles[3]:
            raise ValueError("source changed during slice pin")
        pin(handle, lease)

    monkeypatch.setattr(table, "pin", fail)
    assert hits.pin_hit(request.request_id, chain, start=2 * SPAN, end=4 * SPAN) is None
    assert hits._pending[request.request_id] is prior
    assert table.get(handles[0]).leases and table.get(handles[1]).leases
    assert not table.get(handles[2]).leases and not table.get(handles[3]).leases
    hits.release_lookup(request.request_id)


@pytest.mark.parametrize("mode", ["all", "window", "none"])
def test_host_draft_prefix_configuration_is_checked_at_startup(mode):
    from tests.v1.kv_connector.unit.offloading_connector.test_cpu_draft_offload import (
        _kv_config,
        _vllm_config,
    )
    from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    config = _kv_config()
    config.host_kv_layers = config.kv_cache_groups[0].layer_names
    spec = MockOffloadingSpec(_vllm_config(mode), config)
    if mode == "none":
        with pytest.raises(ValueError, match="draft WINDOW"):
            OffloadingConnectorScheduler(spec)
    else:
        connector = OffloadingConnectorScheduler(spec)
        assert 0 not in connector._group_config_by_idx


def test_partial_g0_boundary_fails_before_rebinding_an_immutable_source():
    scheduler, connector, request, _ = setup_receive(local_blocks=2)
    publish_sources(scheduler, request, (2, 3))
    lookup_allocate(scheduler, connector, request)
    manager = scheduler.kv_cache_manager
    pool = scheduler.host_history_pool
    before = dict(pool.bindings)
    with pytest.raises(ValueError, match="complete G0 block boundaries"):
        manager.host_kv_hits.build_receive(
            request,
            manager.get_blocks(request.request_id).blocks,
            0,
            3 * SPAN + 1,
            999,
        )
    assert pool.bindings == before
    assert all(999 not in page.io for page in pool.host_pages.pages)


def test_receive_metadata_unions_rank_masks_across_duplicate_packets():
    first = OffloadingWorkerMetadata(completed_jobs={7: 1}, receive_results={7: (1, 1)})
    duplicate = first.aggregate(first)
    assert duplicate.completed_jobs == {7: 2}
    assert duplicate.receive_results == {7: (1, 1)}
    last = duplicate.aggregate(OffloadingWorkerMetadata(receive_results={7: (2, 0)}))
    assert last.receive_results == {7: (3, 1)}
    assert first.receive_results == {7: (1, 1)}


def test_external_new_load_cannot_use_a_loading_canonical_page():
    from vllm.v1.kv_offload.host.page_table import PageState

    scheduler, connector, request, _ = setup_receive(local_blocks=0)
    publish_sources(scheduler, request, (0, 2, 3))
    hits = scheduler.kv_cache_manager.host_kv_hits
    table = scheduler.host_history_pool.host_pages
    block_hash = chain_hash_from_request(request.block_hashes, 16)(hits.rule, 1)
    handle = table.reserve(loading_key=make_offload_key(block_hash, hits.rule.group_id))
    table.start_io(handle, 987)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    page = table.get(handle)
    assert page.state == PageState.LOADING and page.io == {987} and not page.leases
    assert not hits._pending and not connector._current_batch_load_jobs
    table.finish_io(handle, 987)


@pytest.mark.parametrize("local_blocks", [2, 4])
def test_dflash_source_split_keeps_state_lookbehind_and_full_window(local_blocks):
    from vllm.v1.core.kv_cache_hit_contract import HitNeed

    scheduler, connector, request, _ = setup_receive(
        local_blocks=local_blocks, speculative_tokens=7
    )
    if local_blocks == 2:
        publish_sources(scheduler, request, (2, 3))
    status = connector._req_status[request.request_id]
    connector.manager.reset_cache()
    keys: list[OffloadKey] = []
    for group_id, rule in connector.hit_contract.rules.items():
        if rule.need == HitNeed.STATE:
            indices = range(2, 4)
        elif rule.need == HitNeed.WINDOW:
            indices = range(6)
        else:
            continue
        keys.extend(status.group_states[group_id].offload_keys[i] for i in indices)
    connector.manager.prepare_store(keys, status.req_context)
    connector.manager.complete_store(keys, status.req_context)
    manager = scheduler.kv_cache_manager
    local, tokens = manager.get_computed_blocks(request)
    assert tokens == 0
    # G0 reaches four pages. Preserve the established EAGLE/DFlash rewind,
    # even though an unusable checkpoint also exists on the fourth page.
    assert connector.get_num_new_matched_tokens(request, 0) == (3 * SPAN, True)
    assert (
        manager.allocate_slots(
            request,
            0,
            new_computed_blocks=local,
            num_external_computed_tokens=3 * SPAN,
            delay_cache_blocks=True,
        )
        is not None
    )
    connector.update_state_after_alloc(
        request, manager.get_blocks(request.request_id), 3 * SPAN
    )
    assert not connector.take_failed_loads()
    job_id, job = next(iter(connector._current_batch_load_jobs.items()))
    txn = job.transfer_spec[0].txn
    assert [entry.logical_index for entry in txn.entries] == [0, 1, 2]
    assert len(txn.new_loads) == max(0, 3 - local_blocks)
    window_groups = [
        r.group_id
        for r in connector.hit_contract.rules.values()
        if r.need == HitNeed.WINDOW
    ]
    assert window_groups
    for group_id in window_groups:
        required = connector.hit_contract.rules[group_id].required_blocks(3 * SPAN, 0)
        assert job.transfer_spec[1].group_sizes[group_id] == len(required)
        assert job.transfer_spec[1].block_indices[group_id] == required.start
    assert [i for i, _ in manager.host_kv_hits.pinned_pages(request.request_id)] == (
        [2] if local_blocks == 2 else []
    )
