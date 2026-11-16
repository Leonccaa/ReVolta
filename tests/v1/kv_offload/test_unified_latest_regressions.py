# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retention regressions through normal connector and SSD entry points."""

import threading
from pathlib import Path

import pytest

from tests.v1.kv_offload.test_host_page_ssd import ROW, key
from tests.v1.kv_offload.test_unified_latest import enable, env_point, point
from tests.v1.kv_offload.test_unified_ssd_transactions import UnitEnv
from tests.v1.kv_offload.test_unified_units import manager
from vllm.v1.kv_offload.base import ReqContext


@pytest.mark.parametrize("ssd", [False, True])
def test_ram_quiescence_uses_boundary_variants_without_global_scan(ssd):
    from types import SimpleNamespace

    m = enable(manager(pages=80, workers=1))
    old = point(m, [0], request_length=16)
    point(m, [0, 1, 2], request_length=16)
    r = m.retention
    r.refresh()

    class NoScan(dict):
        def __iter__(self):
            raise AssertionError("RAM quiescence scanned every variant")

        def items(self):
            raise AssertionError("RAM quiescence scanned every variant")

    r.variants = NoScan(r.variants)
    if ssd:
        store = SimpleNamespace(writes={}, prefetches={})
        m.ssd = SimpleNamespace(unit_store=store)
        for transactions in (store.writes, store.prefetches):
            transactions[old.uid] = object()
            assert not r.quiescent(old.parent)
            transactions.clear()
    assert r.quiescent(old.parent)


@pytest.mark.parametrize("transaction", ["writes", "prefetches"])
def test_disk_only_quiescence_still_waits_for_whole_unit(tmp_path, transaction):
    env = UnitEnv(tmp_path, pages=140, latest_only=True)
    try:
        old = env_point(env, 2, request_length=16)
        env.persist(old)
        env_point(env, 4, request_length=16)
        b = env.manager.units.units[old.parent]
        assert env.manager.units.revoke(b.uid, b.gen, "value")
        r = env.manager.retention
        r.refresh()
        assert old.parent not in env.manager.units.units
        assert old.uid in r.variants
        transactions = getattr(env.store, transaction)
        # The transaction indices are what quiescent consults. Real asynchronous
        # write/prefetch retirement has separate real-IO regression coverage.
        transactions[old.uid] = object()
        try:
            assert not r.quiescent(old.parent)
        finally:
            transactions.pop(old.uid)
        assert r.quiescent(old.parent)
    finally:
        env.ssd.close(io_grace_s=5)


def test_existing_variant_receives_junction_before_cleanup():
    m = enable(manager(pages=80, workers=1))
    root = point(m, [0], request_length=16)
    point(m, [0, 1, 2], request_length=16)
    reused = point(m, [0], request_length=16, junction=True)
    assert reused is root and root.retention["junction"]
    m.retention.step()
    assert root.parent in m.units.units
    assert root.parent in m.retention.branches


def test_each_branch_keeps_its_own_latest_request_resend():
    m = enable(manager(pages=100, workers=1))
    resend_a = point(m, [0], resend=0, request_length=32)
    fork = point(m, [0, 1], resend=0, request_length=32, junction=True)
    a = point(m, [0, 1, 2, 3], resend=0, request_length=32)
    b = point(m, [0, 1, 4, 5, 6], resend=1, request_length=48)
    m.retention.step()
    assert a.parent in m.retention.latest and b.parent in m.retention.latest
    assert fork.parent in m.retention.branches
    assert resend_a.parent in m.units.units
    assert m.lookup(key(0, 1), ReqContext("resend-a")) is True


@pytest.mark.parametrize("after_fsync", [False, True])
@pytest.mark.parametrize("protection", ["branch", "resend"])
def test_new_branch_registered_while_old_manifest_delete_is_inflight(
    tmp_path, monkeypatch, after_fsync, protection
):
    env = UnitEnv(tmp_path, pages=140, latest_only=True)
    gate, started = threading.Event(), threading.Event()
    original = env.store._delete_manifests

    def gated(txn):
        if after_fsync:
            original(txn)
        started.set()
        assert gate.wait(15)
        if not after_fsync:
            original(txn)

    monkeypatch.setattr(env.store, "_delete_manifests", gated)
    try:
        old = env_point(env, 2, request_length=16)
        env.persist(old)
        env_point(env, 4, request_length=16)
        env.manager.retention.step()
        assert started.wait(5)
        if protection == "branch":
            # New complete B shares old MAIN through B=2, then takes another arm.
            fork = env.variant(7, prefix=[key(0), key(1), key(2)])
            fork.retention.update(
                request_length=64, resend=old.uid.boundary.hex(), junction=False
            )
            env.manager.units.on_event("observe", fork)
        else:
            env_point(env, 6, request_length=64, resend=2)
        env.manager.retention.refresh()
        assert old.parent in env.manager.retention.resends
        if protection == "branch":
            assert old.parent in env.manager.retention.branches
        else:
            assert old.parent not in env.manager.retention.branches
        expected = {
            k: bytes(env.backing[h.page_id * ROW : (h.page_id + 1) * ROW])
            for k in old.keys
            if (h := env.table.lookup(k)) is not None
        }
        old_path = env.store.manifest_paths[old.uid]
        gate.set()
        assert env.settle(lambda: not env.store.manifest_jobs)
        assert old.parent in env.manager.units.units
        assert old.uid in env.store.manifests and Path(old_path).exists()
        assert env.settle(lambda: env.store.manifest_ok(old))
        assert not env.manager.units.registration_blocked
        assert not env.manager.units.protected
        env.evict_cpu(old)
        assert env.ssd.prefetch("repaired", [key(2, 1)])
        assert env.settle(lambda: not env.ssd.waiting("repaired"))
        for k, value in expected.items():
            h = env.table.lookup(k)
            assert bytes(env.backing[h.page_id * ROW : (h.page_id + 1) * ROW]) == value
        env.ssd.cancel("repaired")
    finally:
        gate.set()
        env.ssd.close(io_grace_s=5)


def test_legacy_branch_metadata_keeps_legacy_resend_fallback(tmp_path):
    writer = UnitEnv(tmp_path, pages=160)
    old = writer.variant(2)
    first = writer.variant(4, ahead=30)
    second = writer.variant(4, ahead=31)
    for unit in (old, first, second):
        writer.persist(unit)
    writer.ssd.close(io_grace_s=5)
    reader = UnitEnv(tmp_path, pages=160, latest_only=True)
    try:
        reader.manager.retention.refresh()
        assert old.parent in reader.manager.retention.resends
        assert first.parent in reader.manager.retention.branches
        # A normal new MAIN publication makes the retention tree dirty again.
        from tests.v1.kv_offload.test_unified_units import publish

        publish(reader.table, key(99))
        reader.manager.units.on_event("observe", reader.manager.units.units[first.uid])
        reader.manager.retention.refresh()
        assert old.parent in reader.manager.retention.resends
        assert old.parent not in reader.manager.retention.pending
    finally:
        reader.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("evidence", [False, True])
def test_junction_update_through_real_connector_metadata_entry(evidence):
    from tests.v1.kv_connector.unit.offloading_connector.test_unified_lifecycle import (
        SPAN,
        advance,
        make_requests,
        scheduler_with_connector,
    )
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    scheduler = scheduler_with_connector()
    m = scheduler.connector.connector_scheduler.unified
    m.enable_junction_evidence = evidence
    m.retention.evidence_enabled = evidence
    m.units.on_retention_offer = (
        m.retention.offer_evidence if evidence else lambda unit, data: None
    )
    (producer,) = make_requests(1, 8 * SPAN, 16, tag="junction-producer")
    scheduler.add_request(producer)
    connector = scheduler.connector.connector_scheduler
    for _ in range(12):
        advance(scheduler)
    m = connector.unified
    variant = min(
        (u for u in m.units.units.values() if u.parent is not None),
        key=lambda u: u.boundary_position,
    )
    assert not variant.retention.get("junction")
    (sibling,) = make_requests(1, 8 * SPAN, 16, tag="junction-reuser")
    connector.on_new_request(sibling)
    sibling.shared_prefix_boundary = variant.boundary_position
    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={sibling.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    connector.build_connector_meta(output)
    assert variant.retention.get("junction") is True
    m.retention.refresh()
    assert (variant.parent in m.retention.branches) is not evidence
    # This synthetic capture sees only its own child: it remains a hint.


def test_branch_resend_manifest_survives_other_branch_growth(tmp_path):
    env = UnitEnv(tmp_path, pages=160, latest_only=True)

    def metadata(unit, prompt, resend, junction=False):
        unit.retention.update(
            request_length=prompt,
            resend=resend.uid.boundary.hex(),
            junction=junction,
        )
        env.manager.units.on_event("observe", unit)
        return unit

    try:
        resend_a = env.variant(2)
        metadata(resend_a, 64, resend_a)
        env.persist(resend_a)
        old_path = env.store.manifest_paths[resend_a.uid]
        fork = env.variant(3)
        metadata(fork, 64, resend_a, junction=True)
        a = metadata(env.variant(5), 64, resend_a)
        b = env.variant(9, prefix=[key(i) for i in (0, 1, 2, 3, 7, 8)])
        metadata(b, 80, fork)
        env.manager.retention.refresh()
        assert a.parent in env.manager.retention.latest
        assert b.parent in env.manager.retention.latest
        assert fork.parent in env.manager.retention.branches
        # Drain several ordinary idle steps and all real file IO receipts.
        for _ in range(20):
            env.ssd.step()
        assert env.settle(lambda: not env.store.manifest_jobs)
        assert resend_a.parent in env.manager.units.units
        assert resend_a.uid in env.store.manifests
        assert Path(old_path).exists()
    finally:
        env.ssd.close(io_grace_s=5)
