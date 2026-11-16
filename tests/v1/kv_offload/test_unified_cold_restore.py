# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold restore of production-shaped inventories, using real CPU file IO."""

import threading
import time
from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_host_page_ssd import ROW, key, payload
from tests.v1.kv_offload.test_unified_ssd_transactions import UnitEnv
from tests.v1.kv_offload.test_unified_units import assert_conserved, publish
from vllm.utils.hashing import sha256
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.host import ssd as ssd_module
from vllm.v1.kv_offload.unified.units import VariantOffer

pytestmark = pytest.mark.cpu_test
SPAN = 1664


def large_unit(env):
    main = tuple(key(i) for i in range(231))
    state = [key(230, g) for g in range(1, 8)]
    draft = [key(i, 8) for i in range(227, 231)]
    expected = {}
    for n, k in enumerate([*state, *draft, *main]):
        h = publish(env.table, k, env.table.kind_for_key(k))
        expected[k] = payload(n)
        env.backing[h.page_id * ROW : (h.page_id + 1) * ROW] = expected[k]
    boundary = env.manager.units.begin(sha256(230), [*state, *draft])
    assert env.manager.units.finish(boundary.uid, boundary.gen)
    unit = env.manager.units.register(
        VariantOffer(
            boundary.uid,
            boundary.gen,
            sha256(230),
            main,
            231 * SPAN,
            231 * SPAN,
            (227 * SPAN, 231 * SPAN),
        )
    )
    assert unit is not None and len(unit.keys) == 242
    return unit, expected


@pytest.mark.parametrize("resident", ["empty", "ready", "loading"])
def test_large_restart_same_prefix(tmp_path, monkeypatch, resident):
    writer = UnitEnv(tmp_path, pages=300)
    unit, expected = large_unit(writer)
    writer.persist(unit)
    writer.ssd.close(io_grace_s=5)
    reader = UnitEnv(tmp_path, pages=300)
    release = threading.Event()
    original = ssd_module.load_verified_block

    def gated(*args, **kwargs):
        assert release.wait(10)
        return original(*args, **kwargs)

    try:
        assert len(reader.store.manifests) == 1
        assert unit.uid in reader.manager.units.units
        assert not any(reader.manager.is_ready_resident(k) for k in unit.keys)
        if resident == "ready":
            for k in [key(i) for i in range(5)]:
                h = publish(reader.table, k)
                reader.backing[h.page_id * ROW : (h.page_id + 1) * ROW] = expected[k]
        elif resident == "loading":
            monkeypatch.setattr(ssd_module, "load_verified_block", gated)
            assert reader.ssd.prefetch("other", [key(0)]) == [key(0)]
        # The continuation is longer than the committed prefix. Use MAIN-only
        # keys, just as the connector's initial source-aware prefetch does.
        got = reader.ssd.prefetch("restart", [key(i) for i in range(234)])
        print(resident, "selected", len(got), "events", dict(reader.store.events))
        assert set(got) == set(unit.keys)
        assert reader.store.events["prefetch_transactions_started"] == 1
        release.set()
        assert reader.settle(lambda: not reader.ssd.waiting("restart"))
        assert all(
            reader.manager.lookup(k, ReqContext("restart")) is True for k in unit.keys
        )
        for k in unit.keys:
            h = reader.table.lookup(k)
            assert (
                reader.backing[h.page_id * ROW : (h.page_id + 1) * ROW] == expected[k]
            )
        assert reader.ssd.stats_counter["prefetch_verified"] == (
            237 if resident == "ready" else 242
        )
        print(
            resident,
            "restored_tokens",
            231 * SPAN,
            "verified",
            reader.ssd.stats_counter["prefetch_verified"],
        )
        reader.ssd.cancel("restart")
        reader.ssd.cancel("other")
        assert not reader.store.prefetches
        assert_conserved(reader.manager)
    finally:
        release.set()
        reader.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("resident", ["empty", "ready", "loading"])
def test_connector_large_cold_continuation(tmp_path, monkeypatch, resident):
    from tests.v1.core import test_long_context_admission as admission
    from tests.v1.core.test_long_context_host_main_kv import host_settings
    from tests.v1.kv_connector.unit.offloading_connector.test_host_kv_hit import (
        connector_for,
    )
    from tests.v1.kv_connector.unit.offloading_connector.test_host_ssd_prefetch import (
        attach_ssd,
    )
    from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
    from vllm.v1.kv_offload.base import make_offload_key
    from vllm.v1.kv_offload.unified.manager import UnifiedStoreManager

    monkeypatch.setattr(admission, "MAX_MODEL_LEN", 240 * SPAN)
    scheduler = admission.build(
        host_settings(host_pool_tokens=400 * SPAN),
        max_num_seqs=1,
        speculative_tokens=1,
    )
    connector = connector_for(scheduler.kv_cache_manager, dcp_world=2, workers=4)
    scheduler.kv_cache_manager.host_kv_hits.attach_connector(connector)
    (request,) = admission.make_requests(1, 234 * SPAN + 1, 1)
    connector.on_new_request(request)
    status = connector._req_status[request.request_id]
    status.update_offload_keys()
    pool = scheduler.host_history_pool
    m = UnifiedStoreManager(
        SimpleNamespace(
            kv_cache_config=connector.kv_cache_config,
            extra_config={},
            cpu_page_size_per_worker=ROW,
        )
    )
    m.attach(pool)
    connector.manager = connector.unified = m
    chain = chain_hash_from_request(request.block_hashes, connector._hash_block_size)
    rule = connector.host_kv_hits.rule
    main = [make_offload_key(chain(rule, i), m.main_group) for i in range(231)]
    boundary = 231 * SPAN
    components = []
    for group in connector.config.kv_group_configs:
        end = boundary // group.offloaded_block_size
        low = (
            end - 1
            if group.requires_exact_boundary_source
            else max(0, end - (group.sliding_window_size_in_blocks or end))
        )
        components.extend(status.group_states[group.group_idx].offload_keys[low:end])
    ssd, backing = attach_ssd(scheduler, connector, tmp_path)
    m.attach_ssd(ssd)
    try:
        for n, k in enumerate([*components, *main]):
            h = publish(m.table, k, m.table.kind_for_key(k))
            backing[h.page_id * ROW : (h.page_id + 1) * ROW] = payload(n)
        b = m.units.begin(chain(rule, 230), components)
        assert m.units.finish(b.uid, b.gen)
        u = m.units.register(
            VariantOffer(
                b.uid,
                b.gen,
                chain(rule, 230),
                tuple(main),
                boundary,
                boundary,
                (boundary - 4 * SPAN, boundary),
            )
        )
        assert u is not None
        assert ssd.unit_store.start_write(u)
        deadline = time.monotonic() + 10
        while not ssd.unit_store.manifest_ok(u):
            assert time.monotonic() < deadline
            ssd.step()
            time.sleep(0.001)
        ssd.close(io_grace_s=5)
        # Reconstruct the table, manager and SSD controller for the same
        # request; no in-memory copy of the inventory survives this boundary.
        from vllm.v1.kv_offload.host.page_table import HostPageTable

        pool.host_pages = HostPageTable(len(m.table.pages), num_workers=4)
        m = UnifiedStoreManager(m.spec)
        m.attach(pool)
        connector.manager = connector.unified = m
        ssd, backing = attach_ssd(scheduler, connector, tmp_path)
        m.attach_ssd(ssd)
        release = threading.Event()
        original = ssd_module.load_verified_block

        def gated(*args, **kwargs):
            assert release.wait(10)
            return original(*args, **kwargs)

        if resident == "ready":
            for k in main[:5]:
                publish(m.table, k)
        elif resident == "loading":
            monkeypatch.setattr(ssd_module, "load_verified_block", gated)
            assert ssd.prefetch("other", main[:1])
        try:
            result = connector.get_num_new_matched_tokens(request, 0)
            print(
                "connector",
                resident,
                "initial",
                result,
                "events",
                dict(ssd.unit_store.events),
            )
            release.set()
            deadline = time.monotonic() + 10
            while result[0] is None:
                assert time.monotonic() < deadline
                time.sleep(0.002)
                result = connector.get_num_new_matched_tokens(request, 0)
            print("connector", resident, "final", result)
            assert result == (boundary, True)
            assert ssd.unit_store.events["prefetch_transactions_started"] == 1
        finally:
            release.set()
            ssd.cancel("other")
            m.release_lookup(request.request_id)
    finally:
        ssd.close(io_grace_s=5)


@pytest.mark.parametrize("legacy", [False, True])
def test_large_inventory_outlives_one_window_and_legacy_round(tmp_path, legacy):
    now = [1000.0]
    env = UnitEnv(tmp_path, pages=300, clock=lambda: now[0])
    try:
        unit, _ = large_unit(env)
        env.persist(unit)
        env.evict_cpu(unit)
        if legacy:
            assert env.ssd.prefetch("restart", [key(0)])
            assert env.settle(lambda: not env.ssd.waiting("restart"))
        assert env.ssd.prefetch("restart", [key(i) for i in range(234)])
        now[0] += 6  # Healthy multi-window IO need not fit a single T_pf.
        env.ssd.poll()
        assert "restart" not in env.ssd._expired_rounds
        assert env.store.events["prefetch_waiter_timeout"] == 0
        assert env.settle(lambda: not env.ssd.waiting("restart"))
        assert all(env.manager.is_ready_resident(k) for k in unit.keys)
        assert env.ssd.stats_counter["prefetch_verified"] == 242
        env.ssd.cancel("restart")
        assert not env.store.prefetches
    finally:
        env.ssd.close(io_grace_s=5)


def test_cancel_original_unit_preserves_shared_pending_reads(tmp_path):
    env = UnitEnv(tmp_path, prefetch_max_per_request=1, prefetch_max_inflight=1)
    try:
        first = env.variant(ahead=6)
        second = env.variant(ahead=7)
        env.persist(first)
        env.persist(second)
        env.evict_cpu(first)
        env.evict_cpu(second)
        # Select distinct branches explicitly; their STATE/DRAFT and MAIN
        # prefix are shared, but their terminal chain hashes differ.
        assert env.ssd.prefetch("first", [key(6)])
        assert env.ssd.prefetch("second", [key(7)])
        env.ssd.cancel("first")
        assert env.settle(lambda: not env.ssd.waiting("second"))
        assert all(env.manager.is_ready_resident(k) for k in second.keys)
        assert env.store.events["prefetch_adopted_pending_pages"] > 0
        env.ssd.cancel("second")
        assert not env.store.prefetches
        assert not env.store.cache._pins
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


def test_mismatched_continuation_does_not_restore_unrelated_unit(tmp_path):
    env = UnitEnv(tmp_path, pages=300)
    try:
        unit, _ = large_unit(env)
        env.persist(unit)
        env.evict_cpu(unit)
        assert env.store.prefetch("diverged", [key(i) for i in range(5)]) is None
        assert env.store.events["prefetch_no_matching_endpoint"] == 1
        assert not env.store.prefetches
    finally:
        env.ssd.close(io_grace_s=5)
