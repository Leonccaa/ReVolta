# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host reader-ready probes and request pins through the normal complete-prefix contract
paths."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.v1.core.test_kv_cache_hit_contract import (
    BS,
    _cache,
    _full,
    _mamba,
    _manager,
    _request,
    _window,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from tests.v1.kv_offload.test_host_kv_prefix import (
    PAGE_BYTES,
    cache,
    commit,
    key,
    payload,
    pool,
    stored,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_hit_contract import (
    GroupHitRule,
    HitNeed,
    KVHitContract,
    chain_hash_from_request,
    local_hit_rules,
)
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.host.hit import HostKVHitManager
from vllm.v1.kv_offload.host.pool import HostKVPool

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def caches():
    opened = []

    def make(*args, **kwargs):
        result = cache(*args, **kwargs)
        opened.append(result)
        return result

    yield make
    for result in reversed(opened):
        result.close()


def connector_for(manager):
    config = MagicMock()
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.world_size = 1
    config.cache_config.enable_prefix_caching = True
    config.cache_config.block_size = BS
    config.cache_config.hash_block_size = None
    config.cache_config.prefix_cache_retention_interval = 0
    config.kv_transfer_config.kv_connector_extra_config = {}
    config.speculative_config = SimpleNamespace(use_eagle_kv_cache=lambda: False)
    config.kernel_config.qsa_cross_group_hit_check = True
    connector = OffloadingConnectorScheduler(
        MockOffloadingSpec(config, manager.kv_cache_config)
    )
    connector.manager = CPUOffloadingManager(256)
    return connector


def paths(c, *, local_state=(3, 6), binding=lambda bid, key: True):
    manager = _manager([_full(), _mamba(), _window()], check=True)
    request = _request(10)
    connector = connector_for(manager)
    connector.on_new_request(request)
    state = connector._req_status[request.request_id]
    state.update_offload_keys()
    _cache(
        manager, request, {0: set(range(10)), 1: set(local_state), 2: set(range(10))}
    )
    keys = [
        key
        for g, group in enumerate(state.group_states)
        for i, key in enumerate(group.offload_keys)
        if g != 1 or i in (3, 6)
    ]
    connector.manager.prepare_store(keys, state.req_context)
    connector.manager.complete_store(keys, state.req_context)
    hits = HostKVHitManager.attach(c, manager, connector, gpu_binding_ready=binding)
    assert hits is not None
    return manager, connector, request, hits


def host_keys(c, request, n):
    factor = c.pool.block_tokens // BS
    return [
        make_offload_key(request.block_hashes[(i + 1) * factor - 1], c.pool.group_idx)
        for i in range(n)
    ]


def fill(c, request, n):
    keys = host_keys(c, request, n)
    for i, k in enumerate(keys):
        commit(c.pool, k, payload(i))
    return keys


@pytest.mark.parametrize("group", [0, 3])
def test_both_paths_intersect_the_reader_ready_host_prefix(tmp_path, caches, group):
    c = caches(tmp_path, pages=pool(slots=5, group=group), enabled=False)
    manager, connector, request, hits = paths(c)
    keys = fill(c, request, 5)
    # Five main-KV pages plus the recurrent checkpoint at four pages.
    assert manager.get_computed_blocks(request)[1] == 4 * BS
    assert connector.get_num_new_matched_tokens(request, 0) == (4 * BS, True)
    assert hits.rule.group_id in manager.hit_contract._probes
    assert hits.rule.group_id in connector.hit_contract._probes
    assert all(c.reader_ready(k) for k in keys)
    hits.release_uncommitted()


@pytest.mark.parametrize("missing", ["main", "index", "state", "draft"])
def test_any_missing_group_prevents_a_hit(tmp_path, caches, missing):
    c = caches(tmp_path, pages=pool(slots=7), enabled=False)
    manager, connector, request, hits = paths(c)
    if missing != "main":
        fill(c, request, 7)
        group = {"index": 0, "state": 1, "draft": 2}[missing]
        # Evict this group's cached identities while keeping the other groups.
        blocks = {
            block.block_id
            for block in manager.block_pool.blocks
            if block.block_hash is not None and get_group_id(block.block_hash) == group
        }
        manager.block_pool.evict_blocks(blocks)
        state = connector._req_status[request.request_id]
        connector.manager.reset_cache()
        keys = [
            k
            for g, entries in enumerate(state.group_states)
            if g != group
            for k in entries.offload_keys
        ]
        connector.manager.prepare_store(keys, state.req_context)
        connector.manager.complete_store(keys, state.req_context)
    assert manager.get_computed_blocks(request)[1] == 0
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert not hits._pending


def test_unpublished_write_through_is_unreadable(tmp_path, caches):
    c = caches(tmp_path, pages=pool(slots=4), enabled=False)
    manager, connector, request, hits = paths(c)
    keys = host_keys(c, request, 4)
    page = c.pool.reserve(keys[0])
    page.view[:] = payload()
    for k in keys[1:]:
        commit(c.pool, k)
    assert c.pool.lookup(keys[0]) is None
    assert not c.reader_ready(keys[0])
    assert manager.get_computed_blocks(request)[1] == 0
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    c.pool.publish(page)
    c.pool.release(page)
    assert manager.get_computed_blocks(request)[1] == 4 * BS
    assert connector.get_num_new_matched_tokens(request, 0) == (4 * BS, True)
    hits.release_uncommitted()


@pytest.mark.parametrize("path", ["gpu", "connector"])
def test_lookup_pins_survive_admission_and_load_until_free(tmp_path, caches, path):
    c = caches(tmp_path, pages=pool(slots=4), enabled=False)
    manager, connector, request, hits = paths(c)
    keys = fill(c, request, 4)
    if path == "gpu":
        blocks, tokens = manager.get_computed_blocks(request)
        args = dict(num_new_computed_tokens=tokens, new_computed_blocks=blocks)
    else:
        tokens, _ = connector.get_num_new_matched_tokens(request, 0)
        args = dict(num_external_computed_tokens=tokens, delay_cache_blocks=True)
    assert tokens == 4 * BS
    assert c.pool.reserve(key(99)) is None  # lookup -> allocation
    assert manager.allocate_slots(request, 1, **args) is not None
    assert request.request_id in hits._active and not hits._pending
    assert [p.key for p in hits.pinned_pages(request.request_id)] == keys
    if path == "connector":
        connector.update_state_after_alloc(
            request, manager.get_blocks(request.request_id), tokens
        )
        assert connector._current_batch_load_jobs
    assert c.pool.reserve(key(99)) is None  # allocation -> load/consumption
    assert all(c.reader_ready(k) for k in keys)
    manager.free(request)
    assert not hits._active
    assert hits.pinned_pages(request.request_id) == ()
    replacement = c.pool.reserve(key(99))
    assert replacement is not None
    c.pool.release(replacement)


@pytest.mark.parametrize("full_sequence", [False, True])
def test_failed_admission_and_deferred_step_release_lookup_pins(
    tmp_path, caches, full_sequence
):
    c = caches(tmp_path, pages=pool(slots=4), enabled=False)
    manager, _, request, hits = paths(c)
    fill(c, request, 4)
    blocks, tokens = manager.get_computed_blocks(request)
    assert tokens == 4 * BS
    assert (
        manager.allocate_slots(
            request,
            1,
            num_new_computed_tokens=tokens,
            new_computed_blocks=blocks,
            reserved_blocks=manager.block_pool.get_num_free_blocks(),
            full_sequence_must_fit=full_sequence,
        )
        is None
    )
    assert not hits._pending
    assert manager.get_computed_blocks(request)[1] == tokens
    hits.release_uncommitted()
    replacement = c.pool.reserve(key(99))
    assert replacement is not None
    c.pool.release(replacement)


def test_external_extension_keeps_the_local_head_pinned(tmp_path, caches):
    c = caches(tmp_path, pages=pool(slots=7), enabled=False)
    manager, connector, request, hits = paths(c, local_state=(3,))
    fill(c, request, 7)
    assert manager.get_computed_blocks(request)[1] == 4 * BS
    assert connector.get_num_new_matched_tokens(request, 4 * BS) == (3 * BS, True)
    assert len(hits._pending[request.request_id].pages) == 7
    assert c.pool.reserve(key(99)) is None
    hits.release_uncommitted()


def test_failed_external_extension_preserves_local_leases(
    tmp_path, caches, monkeypatch
):
    c = caches(tmp_path, pages=pool(slots=7), enabled=False)
    manager, connector, request, hits = paths(c, local_state=(3,))
    keys = fill(c, request, 7)
    assert manager.get_computed_blocks(request)[1] == 4 * BS
    pin = c.pool.pin
    monkeypatch.setattr(c.pool, "pin", lambda k: None if k == keys[-1] else pin(k))
    assert connector.get_num_new_matched_tokens(request, 4 * BS) == (0, False)
    assert [p.key for p in hits.pinned_pages(request.request_id)] == keys[:4]
    assert len(c.pool._leases) == 4
    hits.release_uncommitted()
    assert not c.pool._leases


@pytest.mark.parametrize("path", ["gpu", "connector"])
def test_pin_failure_is_a_miss_with_no_partial_lease(
    tmp_path, caches, monkeypatch, path
):
    c = caches(tmp_path, pages=pool(slots=4), enabled=False)
    manager, connector, request, hits = paths(c)
    keys = fill(c, request, 4)
    pin = c.pool.pin
    monkeypatch.setattr(c.pool, "pin", lambda k: None if k == keys[1] else pin(k))
    if path == "gpu":
        assert manager.get_computed_blocks(request)[1] == 0
    else:
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert not hits._pending and not c.pool._leases


@pytest.mark.parametrize("binding", [None, False, "error"])
def test_gpu_binding_must_attest_the_current_generation(tmp_path, caches, binding):
    c = caches(tmp_path, pages=pool(slots=7, group=0), enabled=False)

    def attest(bid, k):
        if binding == "error":
            raise KeyError(bid)
        return binding

    manager, connector, request, hits = paths(
        c, binding=None if binding is None else attest
    )
    fill(c, request, 7)
    assert manager.get_computed_blocks(request)[1] == 0
    # External lookup has no destination ID yet. Binding belongs to host-history
    # integration destination binding.
    assert connector.get_num_new_matched_tokens(request, 0) == (7 * BS, True)
    hits.release_uncommitted()


def test_probe_checks_live_gpu_binding_with_the_connector_key(tmp_path, caches):
    c = caches(tmp_path, pages=pool(slots=7, group=0), enabled=False)
    bindings: dict[int, OffloadKey] = {}
    manager, _, request, hits = paths(c, binding=lambda bid, k: bindings.get(bid) == k)
    keys = fill(c, request, 7)
    assert manager.get_computed_blocks(request)[1] == 0
    for block in manager.block_pool.blocks:
        if block.block_hash is not None and get_group_id(block.block_hash) == 0:
            bindings[block.block_id] = make_offload_key(
                get_block_hash(block.block_hash), 0
            )
    assert manager.get_computed_blocks(request)[1] == 7 * BS
    assert [p.key for p in hits._pending[request.request_id].pages] == keys
    # Simulate a block ID reassigned to another generation/key by its owner.
    bindings.clear()
    assert manager.get_computed_blocks(request)[1] == 0
    assert not hits._pending


@pytest.mark.parametrize("damage", [None, "hash", "truncate"])
def test_cold_promotion_reports_only_verified_common_prefix(tmp_path, caches, damage):
    first = caches(tmp_path, pages=pool(slots=7), max_pending_blocks=8)
    request = _request(10)
    keys = host_keys(first, request, 7)
    stored(first, keys)
    paths_on_disk = [Path(first._tier.file_mapper.get_file_name(k)) for k in keys]
    first.close()
    if damage == "hash":
        shutil.copy2(paths_on_disk[0], paths_on_disk[1])
    elif damage == "truncate":
        with paths_on_disk[1].open("r+b") as f:
            f.truncate(PAGE_BYTES - 1)
    second = caches(tmp_path, pages=pool(slots=7), max_pending_blocks=8)
    manager, connector, request, hits = paths(second)
    assert manager.get_computed_blocks(request)[1] == 0
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    for k in keys:
        assert second.prefetch(k)
        assert not second.reader_ready(k)
    assert second.flush(5)
    expected = 7 * BS if damage is None else 0
    assert manager.get_computed_blocks(request)[1] == expected
    assert connector.get_num_new_matched_tokens(request, 0) == (
        expected,
        bool(expected),
    )
    if damage is None:
        assert [bytes(p.view) for p in hits._pending[request.request_id].pages] == [
            payload(i) for i in range(7)
        ]
    else:
        assert second.stats()["load_failures"] == 1
        assert second.pool.lookup(keys[1]) is False
    hits.release_uncommitted()


@pytest.mark.parametrize("reason", ["disabled", "incompatible"])
def test_registration_is_atomic_and_off_contracts_stay_untouched(
    tmp_path, caches, reason
):
    c = caches(tmp_path, enabled=False)
    manager = _manager([_full()], check=reason != "disabled")
    connector = connector_for(manager)
    if reason == "incompatible":
        connector.hit_contract.rules[c.pool.group_idx] = GroupHitRule(
            c.pool.group_idx, HitNeed.STATE, BS
        )
    assert HostKVHitManager.attach(c, manager, connector) is None
    assert manager.host_kv_hits is None and connector.host_kv_hits is None
    assert not connector.hit_contract.has_probes
    assert manager.hit_contract is None or not manager.hit_contract.has_probes


def test_m3_external_restore_is_still_subject_to_joint_admission(tmp_path, caches):
    from tests.v1.core.test_long_context_admission import (
        MAX_MODEL_LEN,
        SPAN,
        build,
        make_requests,
        settings,
    )
    from vllm.v1.core.long_context_kv import HOST

    scheduler = build(settings(host_pool_tokens=MAX_MODEL_LEN))
    manager = scheduler.kv_cache_manager
    manager.hit_contract = KVHitContract(
        local_hit_rules(manager.coordinator.single_type_managers), "gpu-prefix-cache"
    )
    connector = connector_for(manager)
    backing = bytearray(4 * PAGE_BYTES)
    pages = HostKVPool(
        memoryview(backing).cast("B", [4, PAGE_BYTES]),
        group_idx=100,
        block_tokens=SPAN,
    )
    c = caches(tmp_path, pages=pages, enabled=False)
    hits = HostKVHitManager.attach(c, manager, connector)
    assert hits is not None
    (occupant,) = make_requests(1, 8 * SPAN, 1, tag="occupant")
    assert manager.allocate_slots(occupant, 0, num_external_computed_tokens=8 * SPAN)
    (restoring,) = make_requests(1, 4 * SPAN, 1, tag="restoring")
    fill(c, restoring, 4)
    assert hits.pin_hit(
        restoring.request_id,
        chain_hash_from_request(restoring.block_hashes, BS),
        4 * SPAN,
    )
    assert (
        manager.allocate_slots(
            restoring, 0, num_external_computed_tokens=4 * SPAN, delay_cache_blocks=True
        )
        is None
    )
    assert manager.admission_ledger.blocked[HOST] > 0
    assert not hits._pending
    manager.free(occupant)
    assert hits.pin_hit(
        restoring.request_id,
        chain_hash_from_request(restoring.block_hashes, BS),
        4 * SPAN,
    )
    assert (
        manager.allocate_slots(
            restoring, 0, num_external_computed_tokens=4 * SPAN, delay_cache_blocks=True
        )
        is not None
    )
    assert c.pool.reserve(key(99, group=100)) is None
    manager.free(restoring)
    assert not hits._active
    # An unadmitted lookup must also release pins when schedule exits before
    # allocate_slots, such as a connector/encoder/budget deferral.
    assert hits.pin_hit(
        restoring.request_id,
        chain_hash_from_request(restoring.block_hashes, BS),
        4 * SPAN,
    )
    scheduler.schedule()
    assert not hits._pending
