# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU counterexamples for independent state/history spaces and recovery."""

from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from tests.v1.core.long_context_kv_utils import production_config, production_specs
from tests.v1.core.test_kv_cache_hit_contract import BS, _full, _request, _window
from tests.v1.core.test_long_context_admission import build, make_requests
from tests.v1.core.test_long_context_split_pool import SPAN, _layer_specs, settings
from tests.v1.kv_connector.unit.offloading_connector.test_host_kv_hit import (
    connector_for,
    fill,
    paths,
)
from tests.v1.kv_offload.test_host_kv_prefix import PAGE_BYTES, cache, key, pool
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.utils.hashing import sha256
from vllm.v1.attention.kv_cache_view import packed_kv_members
from vllm.v1.core.kv_cache_hit_contract import (
    KVHitContract,
    chain_hash_from_request,
    local_hit_rules,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, init_none_hash
from vllm.v1.kv_cache_interface import (
    KVBlockSegments,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_offload.base import CanonicalKVCaches
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.host.hit import HostKVHitManager
from vllm.v1.kv_offload.host.pool import HostKVPool
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _hash_ready():
    init_none_hash(sha256)


@pytest.mark.parametrize("tiering", [False, True])
def test_unpartitioned_cpu_backing_fits_budget(monkeypatch, tiering):
    config = production_config(max_model_len=8 * SPAN, long_context=settings())
    config.model_config.original_max_model_len = 8 * SPAN
    config.parallel_config.world_size = 4
    config.parallel_config.rank = 3
    config.instance_id = uuid4().hex
    config.cache_config.block_size = SPAN
    config.cache_config.hash_block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_cache_retention_interval = 0
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_extra_config={"cpu_bytes_to_use": 4 << 30},
    )
    (cfg,) = get_kv_cache_configs(config, [production_specs()], [int(6.05 * (1 << 30))])
    # Keep production page specs, packed owners and independent tensor counts;
    # shrink GPU row counts only, so real CPU views fit in this CPU-only test.
    cfg.kv_cache_tensors = [
        replace(t, size=t.size // t.num_blocks * 2, num_blocks=2)
        for t in cfg.kv_cache_tensors
    ]
    cfg.num_blocks = 2
    spec_cls = TieringOffloadingSpec if tiering else CPUOffloadingSpec
    spec = spec_cls(config, cfg)
    assert not spec.partition_by_group
    specs = _layer_specs(cfg)
    packed = packed_kv_members(cfg)
    caches = {}
    for tensor in cfg.kv_cache_tensors:
        raw = torch.empty(tensor.size, dtype=torch.uint8)
        for name in tensor.shared_by:
            view = raw[packed.get(name, (0, 1))[0] :]
            caches[name] = [view] if isinstance(specs[name][1], MambaSpec) else view
    worker = object.__new__(OffloadingConnectorWorker)
    worker.spec = spec
    captured: list[CanonicalKVCaches] = []
    worker._register_handlers = captured.append
    worker.register_kv_caches(caches)
    (canonical,) = captured
    assert len(canonical.tensors) == 24
    required = (
        sum(t.page_size_bytes for t in canonical.tensors) * spec.block_size_factor
    )
    assert required == 11_501_568
    budget = required * 4 * 2 + 123
    config.kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"] = budget
    spec = spec_cls(config, cfg)

    # Exercise the real ordinary allocator / Tiering mmap constructor and view
    # allocator. Only CUDA transfer-handler construction and pinning are stubbed.
    monkeypatch.setattr(gpu_worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(
        gpu_worker,
        "SingleDirectionOffloadingHandler",
        lambda **kw: SimpleNamespace(**kw),
    )
    handlers = spec.create_handlers(canonical)
    cpu = handlers.gpu_to_cpu_handler.cpu_tensors
    try:
        assert sum(t.numel() for t in cpu) == required * spec.num_blocks
        assert sum(t.numel() for t in cpu) * 4 <= budget
        assert spec.cpu_page_size_per_worker == required
        assert spec.num_blocks == 2
        if tiering:
            region = handlers._mmap_region
            assert region.total_size_bytes == required * 4 * spec.num_blocks <= budget
            assert region._worker_offset == region._worker_area_end
            assert all(t.stride(0) == required * 4 for t in cpu)
            # Last byte of rank 3 in the last row must remain addressable.
            cpu[-1][-1, -1] = 71
            assert region._base[-1].item() == 71
    finally:
        handlers.gpu_to_cpu_handler.cpu_tensors.clear()
        handlers.cpu_to_gpu_handler.cpu_tensors.clear()
        if tiering:
            region.cleanup()


def _same_id_store(monkeypatch):
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as module

    cfg = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], _full()),
            KVCacheGroupSpec(["window"], _window()),
        ],
        block_segments=KVBlockSegments(("state", "history"), (32, 32), (1, 0), 1),
    )
    manager = KVCacheManager(
        cfg, max_model_len=4096, hash_block_size=BS, enable_caching=True
    )
    request = _request(2)
    assert manager.allocate_slots(request, 2 * BS)
    ids = manager.get_block_ids(request.request_id)
    assert ids[0] == ids[1] == [1, 2]
    connector = connector_for(manager)
    connector.on_new_request(request)
    monkeypatch.setattr(
        module,
        "yield_req_data",
        lambda output: iter([(request.request_id, ids, False)]),
    )
    jobs = connector._build_store_jobs(
        SimpleNamespace(num_scheduled_tokens={request.request_id: 2 * BS})
    )
    assert len(jobs) == 1
    return manager, request, connector, next(iter(jobs))


@pytest.mark.parametrize("finished_first", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_same_id_store_completes_once(monkeypatch, finished_first, failed):
    manager, request, connector, jid = _same_id_store(monkeypatch)
    job = connector._jobs[jid]
    if finished_first:
        request.status = RequestStatus.FINISHED_STOPPED
        connector.request_finished(request)
        manager.free(request)
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={jid: 1},
                failed_jobs={jid: 1} if failed else {},
            )
        )
    )
    assert job.non_sliding_window_block_ids == [(1, 1), (1, 2)]
    assert job.sliding_window_block_ids == [(0, 1), (0, 2)]
    assert not connector._jobs
    assert not connector._block_id_to_pending_jobs
    if not finished_first:
        request.status = RequestStatus.FINISHED_STOPPED
        connector.request_finished(request)
        manager.free(request)
    assert not connector._req_status
    for pool_ in manager.coordinator.block_pools:
        assert pool_.get_num_free_blocks() == pool_.num_gpu_blocks - 1
    for offload_key in job.keys:
        assert connector.manager.lookup(offload_key, None) is not failed


def test_same_id_store_fences_only_its_segment(monkeypatch):
    manager, request, connector, jid = _same_id_store(monkeypatch)
    # Running full-attention sources remain ref-count protected. Only the
    # window/state sources are registered; equal history IDs must not flush.
    assert set(connector._block_id_to_pending_jobs) == {(0, 1), (0, 2)}
    assert not any(
        connector._block_id_to_pending_jobs.get(identity)
        for identity in connector._transfer_block_identities([1, 2], [2, 0])
    )
    assert all(
        connector._block_id_to_pending_jobs[identity] == {jid}
        for identity in connector._transfer_block_identities([1, 2], [0, 2])
    )
    request.status = RequestStatus.FINISHED_STOPPED
    connector.request_finished(request)
    # Separate jobs watching one segment's same ID survive the other's cleanup.
    connector._block_id_to_pending_jobs[(0, 1)].add(999)
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs={jid: 1})
        )
    )
    assert connector._block_id_to_pending_jobs == {(0, 1): {999}}
    connector._remove_pending_job(jid, [(0, 1), (1, 1)])  # repeated cleanup
    assert connector._block_id_to_pending_jobs == {(0, 1): {999}}
    manager.free(request)


@pytest.mark.parametrize("site", ["load", "allocation", "zero"])
@pytest.mark.parametrize("matching", [False, True])
def test_store_fence_sites_keep_segment_identity(monkeypatch, site, matching):
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as module
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    manager, request, connector, jid = _same_id_store(monkeypatch)
    # This source is protected in exactly one space, with an equal numeric ID
    # in the other space. Drive the actual fence sites, not only key helpers.
    segment = 1 if matching else 0
    connector._block_id_to_pending_jobs = {(segment, 1): {jid}}
    if site == "load":
        other = _request(2, "loading")
        connector.on_new_request(other)
        status = connector._req_status[other.request_id]
        status.update_offload_keys()
        keys = [key_ for group in status.group_states for key_ in group.offload_keys]
        connector.manager.prepare_store(keys, status.req_context)
        connector.manager.complete_store(keys, status.req_context)
        # Restore only history: window already holds a valid local prefix.
        from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock

        window_blocks = [
            KVCacheBlock(block_id=i, _block_hash=BlockHash(b"already-valid"))
            for i in (3, 4)
        ]
        blocks = KVCacheBlocks(
            (
                [KVCacheBlock(block_id=i) for i in (1, 2)],
                window_blocks,
            )
        )
        connector.hit_contract = None
        connector._req_status[other.request_id].num_locally_computed_tokens = 0
        connector.update_state_after_alloc(other, blocks, 2 * BS)
    elif site == "allocation":
        monkeypatch.setattr(
            module,
            "yield_req_data",
            lambda output: iter(
                [
                    (request.request_id, ([1], []), False),
                ]
            ),
        )
        connector._build_store_jobs(
            SimpleNamespace(
                num_scheduled_tokens={request.request_id: 0},
            )
        )
    else:
        monkeypatch.setattr(connector, "_build_store_jobs", lambda output: {})
        monkeypatch.setattr(
            connector, "_build_boundary_state_store_jobs", lambda output: {}
        )
        metadata = connector.build_connector_meta(
            SimpleNamespace(
                preempted_req_ids=None,
                new_block_ids_to_zero=[1],
            )
        )
        assert metadata.jobs_to_flush == ({jid} if matching else set())
        return
    assert connector._current_batch_jobs_to_flush == ({jid} if matching else set())


def test_store_failure_waits_for_all_rank_completions(monkeypatch):
    manager, request, connector, jid = _same_id_store(monkeypatch)
    job = connector._jobs[jid]
    job.pending_count = 4
    request.status = RequestStatus.FINISHED_ABORTED
    connector.request_finished(request)
    manager.free(request)
    for count, failed in ((1, True), (2, False)):
        connector.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=OffloadingWorkerMetadata(
                    completed_jobs={jid: count},
                    failed_jobs={jid: 1} if failed else {},
                )
            )
        )
        assert jid in connector._jobs
        assert connector._block_id_to_pending_jobs
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs={jid: 1})
        )
    )
    assert not connector._jobs and not connector._block_id_to_pending_jobs
    assert not connector._req_status
    assert all(not connector.manager.lookup(k, None) for k in job.keys)


def test_split_connector_factory_rejects_unadapted_implementation(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole

    class Unadapted:
        def __init__(self, *args):
            pytest.fail("must reject before creating the connector")

    monkeypatch.setattr(
        KVConnectorFactory, "get_connector_class", lambda config: Unadapted
    )
    with pytest.raises(ValueError, match="request-scoped load failures"):
        KVConnectorFactory.create_connector(
            SimpleNamespace(
                kv_transfer_config=SimpleNamespace(engine_id="split-review"),
                scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
            ),
            KVConnectorRole.SCHEDULER,
            SimpleNamespace(block_segments=object()),
        )


@pytest.mark.parametrize("late", [False, True])
def test_finished_failure_does_not_retain_skip_marker(tmp_path, late):
    c = cache(tmp_path, pages=pool(slots=4), enabled=False)
    try:
        manager, connector, request, hits = paths(c)
        if not late:
            connector._abandon_load(request.request_id, "injected plan failure")
            connector.take_failed_loads()
        request.status = RequestStatus.FINISHED_ABORTED
        connector.request_finished(request)
        if late:
            connector._abandon_load(request.request_id, "completion after cancellation")
            connector.take_failed_loads()
        manager.free(request)
        assert request.request_id not in connector._req_status
        assert request.request_id not in connector._skip_external_once
        connector.reset_cache()
        assert not connector._skip_external_once
    finally:
        c.close()


def test_skip_survives_deferred_lookup_until_admitted(tmp_path):
    c = cache(tmp_path, pages=pool(slots=4), enabled=False)
    try:
        manager, connector, request, hits = paths(c)
        connector._abandon_load(request.request_id, "injected plan failure")
        connector.take_failed_loads()
        # Model two lookups followed by scheduler budget/capacity deferral.
        for _ in range(2):
            assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
            assert request.request_id in connector._skip_external_once
        connector.reset_cache()
        assert request.request_id in connector._skip_external_once
        connector.update_state_after_alloc(request, manager.empty_kv_cache_blocks, 0)
        assert request.request_id not in connector._skip_external_once
    finally:
        c.close()


@pytest.mark.parametrize("full", [False, True])
def test_split_rejection_releases_pending_host_pin(tmp_path, full):
    scheduler = build(settings(state_blocks=14))
    manager = scheduler.kv_cache_manager
    manager.hit_contract = KVHitContract(
        local_hit_rules(manager.coordinator.single_type_managers), "gpu-prefix-cache"
    )
    connector = connector_for(manager)
    c = cache(
        tmp_path,
        pages=HostKVPool(
            memoryview(bytearray(4 * PAGE_BYTES)).cast("B", [4, PAGE_BYTES]),
            group_idx=100,
            block_tokens=SPAN,
        ),
        enabled=False,
    )
    hits = HostKVHitManager.attach(c, manager, connector)
    occupants = []
    try:
        for tag in ("occupant1", "occupant2"):
            (request,) = make_requests(1, 8 * SPAN, 1, tag=tag)
            assert manager.allocate_slots(request, 2 * SPAN)
            occupants.append(request)
        (restoring,) = make_requests(1, 4 * SPAN, 1, tag="restoring")
        fill(c, restoring, 4)
        assert hits.pin_hit(
            restoring.request_id,
            chain_hash_from_request(restoring.block_hashes, 16),
            4 * SPAN,
        )
        assert (
            manager.allocate_slots(
                restoring,
                1,
                num_external_computed_tokens=4 * SPAN,
                delay_cache_blocks=True,
                full_sequence_must_fit=full,
            )
            is None
        )
        assert not hits._pending
        assert not c.pool._leases
        page = c.pool.reserve(key(99, group=100))
        assert page is not None
        c.pool.release(page)
    finally:
        for request in occupants:
            manager.free(request)
        c.close()


def test_split_failed_restore_releases_all_domains_and_host_leases(tmp_path):
    scheduler = build(settings(state_blocks=28))
    manager = scheduler.kv_cache_manager
    manager.hit_contract = KVHitContract(
        local_hit_rules(manager.coordinator.single_type_managers), "gpu-prefix-cache"
    )
    connector = connector_for(manager)
    c = cache(
        tmp_path,
        pages=HostKVPool(
            memoryview(bytearray(4 * PAGE_BYTES)).cast("B", [4, PAGE_BYTES]),
            group_idx=100,
            block_tokens=SPAN,
        ),
        enabled=False,
    )
    hits = HostKVHitManager.attach(c, manager, connector)
    try:
        (request,) = make_requests(1, 4 * SPAN + 1, 1)
        fill(c, request, 4)
        assert hits.pin_hit(
            request.request_id,
            chain_hash_from_request(request.block_hashes, 16),
            4 * SPAN,
        )
        assert manager.allocate_slots(
            request, 0, num_external_computed_tokens=4 * SPAN, delay_cache_blocks=True
        )
        assert request.request_id in hits._active
        assert manager.admission_ledger.hot_used > 0
        scheduler.connector = object()
        scheduler.requests[request.request_id] = request
        scheduler._inflight_prefills.add(request)
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        request.num_computed_tokens = 4 * SPAN
        scheduler._void_failed_loads({request.request_id})
        scheduler.finished_recving_kv_req_ids.add(request.request_id)
        scheduler._update_waiting_for_remote_kv(request)
        assert not hits._active and not hits._pending and not c.pool._leases
        assert manager.admission_ledger.hot_used == 0
        assert manager.admission_ledger.in_use == [0, 0, 0]
        for pool_ in manager.coordinator.block_pools:
            assert pool_.get_num_free_blocks() == pool_.num_gpu_blocks - 1
        manager.free(request)
        assert manager.admission_ledger.hot_used == 0
    finally:
        c.close()
