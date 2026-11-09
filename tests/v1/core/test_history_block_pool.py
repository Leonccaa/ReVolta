# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.history_block_pool import HistoryBlockPool
from vllm.v1.core.long_context_kv import HOST
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host.page_table import PageState


def pool():
    result = HistoryBlockPool(5, True, 16, False)
    result.configure_host(4, 2, 0)
    result.admission_observer = SimpleNamespace(
        _scope=(HOST,), on_allocated=lambda _: None, on_released=lambda _: None
    )
    return result


def request(name="a"):
    return SimpleNamespace(request_id=name, block_hashes=[b"a" * 32])


def cache(p, req, blocks):
    p.cache_full_blocks(req, blocks, 0, 1, 16, 0)


def test_upstream_free_keeps_same_step_uncomputed_hash():
    p = BlockPool(4, True, 16, False)
    blocks = p.get_new_blocks(1)
    cache(p, request(), blocks)
    p.free_blocks(blocks)
    assert p.get_cached_block(b"a" * 32, [0]) == blocks


def test_preempted_step_revokes_new_hash_and_does_not_issue_seal():
    p = pool()
    p.start_step(1)
    blocks = p.get_new_blocks(1)
    cache(p, request(), blocks)
    p.cancel_scheduled("a")
    p.free_blocks(blocks)
    assert p.finish_schedule([]) == []
    assert p.get_cached_block(b"a" * 32, [0]) is None
    assert not p.bindings


def test_only_final_executed_set_gets_seals_and_active_bindings_survive_free():
    p = pool()
    p.start_step(1)
    a = p.get_new_blocks(1)
    b = p.get_new_blocks(1)
    cache(p, request("a"), a)
    cache(p, request("b"), b)
    p.cancel_scheduled("b")
    tickets = p.finish_schedule(["a"])
    assert len(tickets) == 1 and tickets[0].hid == a[0].block_id
    assert b[0].block_hash is None
    ticket = tickets[0]
    page = p.bindings[ticket.hid]
    p.host_pages.seal_result(ticket, 3, 0)
    p.host_pages.complete_step(1, 3)
    p.start_step(2)
    p.finish_schedule(["a"])
    p.start_step(3)
    p.finish_schedule(["a"])
    p.release_request("a", a)
    p.free_blocks(a)
    assert p.bindings[ticket.hid] == page
    p.unbind(ticket.hid)
    assert p.host_pages.get(page).retirements[0].retire_after == 3
    assert p.host_pages.get(page).state == PageState.READY
    p.host_pages.complete_step(3, 3)
    assert p.host_pages.watermark == 1
    p.host_pages.complete_step(2, 3)
    assert p.host_pages.watermark == 3


def test_ring_does_not_allocate_host_page_and_binding_updates_are_last_writer():
    p = pool()
    p.admission_observer._scope = ()
    ring = p.get_new_blocks(1)
    assert ring[0].block_id not in p.bindings
    assert p.host_pages.available() == 3
    p.free_blocks(ring)
    p.admission_observer._scope = (HOST,)
    fresh = p.get_new_blocks(1)[0]
    handle = p.bindings[fresh.block_id]
    assert fresh.block_id == ring[0].block_id
    assert p.hgen[fresh.block_id] == 2
    assert p.binding_updates[fresh.block_id] == handle.page_id


def test_transfer_job_keeps_detached_page_until_all_worker_completions():
    p = pool()
    block = p.get_new_blocks(1)[0]
    old = p.bindings[block.block_id]
    p.track_job(7, [block.block_id])
    p.unbind(block.block_id)
    assert p.host_pages.get(old).io == {7}
    p.complete_jobs({7: 1})
    assert p.host_pages.get(old).io == {7}
    p.complete_jobs({7: 1})
    assert p.host_pages.pages[old.page_id].handle.epoch == old.epoch + 1


def test_reset_with_pending_seal_waits_for_receipt_and_watermark():
    p = pool()
    p.start_step(1)
    blocks = p.get_new_blocks(1)
    cache(p, request(), blocks)
    [ticket] = p.finish_schedule(["a"])
    p.release_request("a", blocks)
    p.free_blocks(blocks)
    assert p.reset_prefix_cache()
    assert p.host_pages.ssd_epoch == 1
    assert p.get_cached_block(b"a" * 32, [0]) is None
    p.host_pages.seal_result(ticket, 3, 0)
    assert p.host_pages.lookup(ticket.key) is None
    p.host_pages.complete_step(1, 3)
    assert p.host_pages.pages[ticket.handle.page_id].state == PageState.FREE


@pytest.mark.parametrize("compute_failed", [0, 2])
def test_scheduler_classifies_compute_failure_and_preserves_seal_only_local_hash(
    compute_failed,
):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.outputs import KVConnectorOutput

    p = pool()
    p.start_step(1)
    blocks = p.get_new_blocks(1)
    cache(p, request(), blocks)
    [ticket] = p.finish_schedule(["a"])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.host_history_pool = p
    meta = OffloadingWorkerMetadata(
        seal_results={ticket: (3, 2, compute_failed)}, completed_steps={1: 3}
    )
    scheduler.update_host_kv_output(KVConnectorOutput(kv_connector_worker_meta=meta))
    assert (p.get_cached_block(b"a" * 32, [0]) is not None) == (not compute_failed)
    assert p.host_pages.get(ticket.handle).state == PageState.PRIVATE
    assert p.host_pages.watermark == 1


def test_schedule_preempts_an_earlier_executed_candidate_and_revokes_its_hash(
    monkeypatch,
):
    from tests.v1.core.long_context_kv_utils import SPAN
    from tests.v1.core.test_long_context_admission import build, make_requests, settings
    from vllm.v1.core.sched.scheduler import SchedulingPolicy
    from vllm.v1.request import RequestStatus

    scheduler = build(settings(main_kv_placement="host", hot_pool_gib=0.8))
    scheduler.policy = SchedulingPolicy.PRIORITY
    a, b = make_requests(2, 2 * SPAN + 100, 1)
    a.priority, b.priority = 5, 0
    for req in (a, b):
        scheduler.add_request(req)
        scheduler.waiting.pop_request()
        req.status = RequestStatus.RUNNING
    scheduler.running = [a, b]
    manager = scheduler.kv_cache_manager
    original = manager.allocate_slots
    p = scheduler.host_history_pool
    offered: list[tuple[int, int, bytes]] = []

    def allocate(req, *args, **kwargs):
        if req is b and not offered:
            # The allocator failure is injected only after A has obtained
            # full-block hashes in this very schedule() invocation.
            offered.extend(p.new_hashes[a.request_id])
            assert offered
            return None
        return original(req, *args, **kwargs)

    monkeypatch.setattr(manager, "allocate_slots", allocate)
    output = scheduler.schedule()
    assert a.request_id not in output.num_scheduled_tokens
    assert b.request_id in output.num_scheduled_tokens
    assert a.num_preemptions == 1
    cancelled = {(hid, hgen) for hid, hgen, _ in offered}
    assert all((t.hid, t.hgen) not in cancelled for t in output.host_kv.seals)
    assert a.request_id not in p.new_hashes
    assert all(
        p.get_cached_block(block_hash, [p.host_group_id]) is None
        for _, _, block_hash in offered
    )


def test_failed_receipt_after_request_free_unbinds_cache_and_waits_for_watermark():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.outputs import KVConnectorOutput

    p = pool()
    p.start_step(1)
    blocks = p.get_new_blocks(1)
    cache(p, request(), blocks)
    [ticket] = p.finish_schedule(["a"])
    p.release_request("a", blocks)
    p.free_blocks(blocks)
    assert ticket.hid in p.bindings
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.host_history_pool = p
    meta = OffloadingWorkerMetadata(seal_results={ticket: (3, 2, 2)})
    scheduler.update_host_kv_output(KVConnectorOutput(kv_connector_worker_meta=meta))
    assert ticket.hid not in p.bindings
    page = p.host_pages.get(ticket.handle)
    assert page.retirements[0].retire_after == 1
    assert not p.host_pages.reclaimable(page)
    p.host_pages.complete_step(1, 3)
    assert p.host_pages.pages[ticket.handle.page_id].state == PageState.FREE


@pytest.mark.parametrize("hash_blocks_per_page", [1, 2])
def test_loaded_prefix_hash_and_binding_survive_deferral_without_execution(
    hash_blocks_per_page,
):
    # Async receive completion caches valid loaded blocks before admission.
    # Budget or hot-pool pressure can defer that request in the same schedule.
    p = pool()
    p.start_step(1)
    blocks = p.get_new_blocks(1)
    req = request()
    req.block_hashes = [b"a" * 32, b"b" * 32][:hash_blocks_per_page]
    p.cache_full_blocks(req, blocks, 0, 1, 16 * hash_blocks_per_page, 0)
    handle = p.bindings[blocks[0].block_id]
    assert p.finish_schedule([]) == []
    assert p.get_cached_block(b"".join(req.block_hashes), [0]) == blocks
    assert p.bindings[blocks[0].block_id] == handle
    assert p.host_pages.get(handle).compute_result.value == "pending"
    p.start_step(2)
    [ticket] = p.finish_schedule(["a"])
    assert ticket.handle == handle and ticket.step == 2
    assert ticket.key == make_offload_key(req.block_hashes[-1], 0)
    p.host_pages.seal_result(ticket, 3, 0)
    p.host_pages.complete_step(1, 3)
    p.host_pages.complete_step(2, 3)
    assert p.host_pages.is_ready(handle)
    p.evict_blocks({blocks[0].block_id})
    p.unbind(blocks[0].block_id)
    assert p.host_pages.lookup(ticket.key) == handle


def test_real_scheduler_restored_prefix_deferral_eventually_publishes(monkeypatch):
    from tests.v1.core.test_async_kv_admission import HybridMockConnector, complete
    from tests.v1.core.test_long_context_admission import (
        SPAN,
        build,
        make_requests,
        settings,
    )
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.request import RequestStatus

    s = build(settings(main_kv_placement="host", hot_pool_gib=0.8))
    cfg = s.vllm_config
    cfg.kv_transfer_config = KVTransferConfig(
        kv_connector="MockKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"matched_tokens": SPAN, "is_async": True},
    )
    s.connector = HybridMockConnector(cfg, KVConnectorRole.SCHEDULER, s.kv_cache_config)
    [req] = make_requests(1, 2 * SPAN + 100, 1)
    s.add_request(req)
    first = s.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    complete(s, first, received={req.request_id})
    original = s.kv_cache_manager.allocate_slots
    monkeypatch.setattr(s.kv_cache_manager, "allocate_slots", lambda *a, **kw: None)
    deferred = s.schedule()
    assert req.request_id not in deferred.num_scheduled_tokens
    assert not deferred.host_kv.seals
    p = s.host_history_pool
    restored = {hid: p.bindings[hid] for hid, _, _ in p.new_hashes[req.request_id]}
    assert restored
    complete(s, deferred)
    monkeypatch.setattr(s.kv_cache_manager, "allocate_slots", original)
    resumed = s.schedule()
    assert req.request_id in resumed.num_scheduled_tokens
    assert restored.keys() <= {t.hid for t in resumed.host_kv.seals}
    complete(s, resumed)
    for hid, handle in restored.items():
        assert p.host_pages.is_ready(handle)
        key = p.host_pages.get(handle).key
        p.evict_blocks({hid})
        p.unbind(hid)
        assert p.host_pages.lookup(key) == handle
