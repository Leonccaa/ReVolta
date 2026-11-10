# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normal completion and accounting regressions for accepted cache work."""

from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_host_page_ssd import ROW, Clock
from tests.v1.kv_offload.test_unified_ssd_transactions import UnitEnv
from tests.v1.kv_offload.test_unified_units import key, manager, publish
from vllm.utils.hashing import sha256
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.host.page_table import PageState, Result
from vllm.v1.kv_offload.unified.units import VariantOffer

pytestmark = pytest.mark.cpu_test


def test_inflight_cpu_receive_is_excluded_from_writable_wait(tmp_path):
    from vllm.v1.kv_offload.host.hit import HostKVHitManager
    from vllm.v1.kv_offload.host.receive import ReceiveEntry, ReceiveSource, ReceiveTxn

    clock = Clock()
    env = UnitEnv(tmp_path, clock=clock)
    h = publish(env.table, key(77))
    receive = ReceiveTxn(
        12345,
        "read",
        1,
        (ReceiveEntry(0, ReceiveSource.NEW_LOAD, 0, 7, 1, h),),
    )
    hits = SimpleNamespace(
        history_pool=SimpleNamespace(host_pages=env.table, last_use={})
    )
    try:
        clock.now += 90
        before = env.store.snapshot()
        assert before["writable_dirty_bytes"] == ROW
        assert before["oldest_writable_dirty_age_s"] == 90
        HostKVHitManager.start_receive(hits, receive)
        snapshot = env.store.snapshot()
        assert snapshot["dirty_inflight_io_bytes"] == ROW
        assert snapshot["writable_dirty_bytes"] == 0
        assert snapshot["oldest_writable_dirty_age_s"] == 0
    finally:
        HostKVHitManager.retire_receive(hits, receive)
        env.ssd.close(io_grace_s=2)


def pending_variant(m, *, ahead=1):
    keys = [key(1, g) for g in m.kind_by_group if g != 8]
    assert m.admit_build("r", len(keys) + 2)
    b = m.begin_boundary("r", sha256(1), keys)
    assert b is not None
    publish(m.table, key(1))
    main = (key(1),) if ahead == 1 else (key(1), key(ahead))
    m.offer_variant("r", VariantOffer(b.uid, b.gen, sha256(ahead), main))
    out = m.prepare_store(keys, ReqContext("r"))
    m.bind_job(1, "r", out.store_spec, True)
    return b, out.store_spec.gen


def assert_waiters_released(m):
    assert not m.waiting and not m.waiting_on and not m._wait_keys
    assert not m._missing_main and not m._waiting_boundaries
    assert not m._draining_variants and not m._draining_main


def test_normal_finish_before_dma_registers_and_commits_manifest(tmp_path):
    clock = Clock()
    env = UnitEnv(tmp_path, clock=clock)
    m = env.manager
    try:
        b, gen = pending_variant(m)
        assert m.hold_lookup("r", [key(1)])
        m.on_request_finished(ReqContext("r"))
        assert not m._lookup_holds
        assert "r" in m.waiting
        m.receipt(1, gen, m.table.all_ranks, 0)
        assert m.units.units[b.uid].complete
        assert len(m.units.units[b.uid].variants) == 1
        variant = m.units.units[next(iter(m.units.units[b.uid].variants))]
        clock.now += 301
        env.ssd.step()
        assert env.settle(lambda: env.store.manifest_ok(variant))
        assert env.store.manifest_ok(variant)
        assert_waiters_released(m)
        assert not m.builds and not m.jobs and not m.ledger.charges
    finally:
        env.ssd.close(io_grace_s=2)


@pytest.mark.parametrize("main_first", [True, False])
def test_finished_offer_waits_only_for_accepted_main_seal(main_first):
    m = manager()
    b, gen = pending_variant(m, ahead=2)
    h = m.table.reserve()
    m.table.bind(h, 7, 1)
    ticket = m.table.open_seal(h, 7, 1, key(2), 1)
    m.on_request_finished(ReqContext("r"))
    if main_first:
        m.table.seal_result(ticket, m.table.all_ranks, 0)
        m.receipt(1, gen, m.table.all_ranks, 0)
    else:
        m.receipt(1, gen, m.table.all_ranks, 0)
        m.table.seal_result(ticket, m.table.all_ranks, 0)
    assert m.units.units[b.uid].variants
    assert_waiters_released(m)


def test_finished_offer_needing_future_main_is_terminal():
    m = manager()
    b, gen = pending_variant(m, ahead=2)
    m.on_request_finished(ReqContext("r"))
    m.receipt(1, gen, m.table.all_ranks, 0)
    assert not m.units.units[b.uid].variants
    assert m.events["variant_finish_incomplete"] == 1
    assert_waiters_released(m)


@pytest.mark.parametrize(
    "reason", ["cancel", "preempt", "reset", "dma_failed_mask", "source_not_dispatched"]
)
def test_build_failure_reason_counted_once(reason):
    m = manager()
    b, gen = pending_variant(m)
    if reason in ("cancel", "preempt"):
        m.cancel_request("r", reason=reason)
    elif reason == "reset":
        m.reset_cache()
    elif reason == "source_not_dispatched":
        # One additional component was accepted but never handed to a job.
        extra = [key(2, g) for g in m.kind_by_group]
        assert m.admit_build("other", 11)
        other = m.begin_boundary("other", sha256(2), extra)
        m.end_step()
        assert other is not None
    m.receipt(1, gen, m.table.all_ranks, 1 if reason == "dma_failed_mask" else 0)
    assert m.units.events[f"build_failed_{reason}"] == 1
    if reason != "source_not_dispatched":
        assert b.uid not in m.units.units
    assert not m.builds and not m.jobs


def test_private_shadow_do_not_age_writable_dirty(tmp_path):
    clock = Clock()
    env = UnitEnv(tmp_path, clock=clock)
    try:
        private = env.table.reserve()
        env.table.pin(private, 999)
        canonical = publish(env.table, key(1))
        shadow = env.table.reserve()
        env.table.pin(shadow, 998)
        env.table.producer_result(shadow, compute=Result.OK, seal=Result.OK)
        assert not env.table.publish(shadow, key(1))
        assert env.table.get(shadow).state == PageState.SHADOW
        env.table.mark_persisted(canonical)
        env.store.snapshot()
        clock.now += 3600
        snap = env.store.snapshot()
        assert snap["total_unpersisted_physical_bytes"] == 2 * ROW
        assert snap["total_dirty_bytes"] == 2 * ROW
        assert snap["dirty_private_bytes"] == snap["dirty_shadow_bytes"] == ROW
        assert snap["dirty_private_oldest_age_s"] == 3600
        assert snap["writable_dirty_bytes"] == 0
        assert snap["oldest_writable_dirty_age_s"] == 0
        assert not env.store.writes
        env.table.release(private, 999)
        env.table.release(shadow, 998)
        env.table.collect()
        assert env.store.snapshot()["total_unpersisted_physical_bytes"] == 0
    finally:
        env.ssd.close(io_grace_s=2)


def test_snapshot_partition_and_publication_age(tmp_path):
    clock = Clock()
    env = UnitEnv(tmp_path, clock=clock)
    try:
        unit = env.variant()
        zero = publish(env.table, key(99))
        clock.now += 301
        snap = env.store.snapshot()
        assert snap["dirty_ready_queued_due_bytes"] == len(unit.keys) * ROW
        assert snap["dirty_ready_unreferenced_bytes"] == ROW
        assert snap["oldest_writable_dirty_age_s"] == 301
        assert snap["writable_dirty_bytes"] == (len(unit.keys) + 1) * ROW
        env.manager.drop_pending_secondary_writes([key(5, 1)])
        filtered = env.store.snapshot()
        assert filtered["dirty_ready_filtered_bytes"] >= ROW
        assert env.store.start_write(unit)
        active = env.store.snapshot()
        assert active["dirty_inflight_io_bytes"] > 0
        for snapshot in (snap, filtered, active):
            assert (
                sum(
                    v
                    for k, v in snapshot.items()
                    if k.startswith("dirty_")
                    and k.endswith("_bytes")
                    and k
                    not in (
                        "dirty_main_bytes",
                        "dirty_state_bytes",
                        "dirty_draft_bytes",
                        "dirty_limit_bytes",
                    )
                )
                == snapshot["total_unpersisted_physical_bytes"]
            )
        assert env.table.get(zero).valid
    finally:
        env.ssd.close(io_grace_s=2)


def test_orphans_split_without_deletion(tmp_path):
    legacy = tmp_path / "previous-format" / "page.bin"
    legacy.parent.mkdir()
    legacy.write_bytes(b"a" * 19)
    temporary = tmp_path / "leftover.tmp"
    temporary.write_bytes(b"b" * 23)
    env = UnitEnv(tmp_path)
    try:
        h = publish(env.table, key(7))
        env.ssd._new_job(env.table.get(h), False)
        assert env.settle(lambda: env.table.is_persisted(env.table.get(h)))
        snap = env.store.snapshot()
        assert snap["ssd_orphan_old_namespace_zero_ref_bytes"] == 19
        assert snap["ssd_orphan_current_namespace_no_manifest_bytes"] == ROW
        assert snap["ssd_temporary_bytes"] == 23
        assert snap["ssd_orphan_bytes"] == ROW + 19
        assert legacy.exists() and temporary.exists()
    finally:
        env.ssd.close(io_grace_s=2)


def test_union_failure_has_its_own_reason(monkeypatch):
    m = manager()
    b, gen = pending_variant(m)
    monkeypatch.setattr(m.units, "finish", lambda uid, generation: False)
    m.receipt(1, gen, m.table.all_ranks, 0)
    assert m.units.events["build_failed_finish_union_incomplete"] == 1
    assert m.units.events["revoke_build_failed"] == 1
    assert b.uid not in m.units.units
    assert_waiters_released(m)


def test_draining_registration_metadata_stays_bounded():
    m = manager(pages=512)
    for n in range(40):
        keys = [key(n, g) for g in m.kind_by_group]
        assert m.admit_build("r", 11)
        b = m.begin_boundary("r", sha256(n), keys)
        publish(m.table, key(n))
        m.offer_variant("r", VariantOffer(b.uid, b.gen, sha256(n), (key(n),)))
        out = m.prepare_store(keys, ReqContext("r"))
        m.bind_job(n, "r", out.store_spec, True)
        m.on_request_finished(ReqContext("r"))
        assert len(m.waiting["r"]) <= m.max_pending_variants
        m.receipt(n, out.store_spec.gen, m.table.all_ranks, 0)
        assert m.units.units[b.uid].variants
        assert_waiters_released(m)
        assert not m._latest_variant_offers and not m.ledger.charges


def test_variant_wait_and_terminal_absence_are_distinct():
    m = manager()
    b, _ = pending_variant(m)
    assert m.units.events["variant_b_building_wait"] == 1
    assert m.units.events["variant_b_absent_terminal"] == 0
    m.units.revoke(b.uid, b.gen, "value")
    m.retry_variants("r")
    m.retry_variants("r")
    assert m.units.events["variant_b_absent_terminal"] == 1
    assert_waiters_released(m)


def test_private_age_restarts_when_published(tmp_path):
    clock = Clock()
    env = UnitEnv(tmp_path, clock=clock)
    try:
        h = env.table.reserve()
        env.table.pin(h, 11)
        env.store.snapshot()
        clock.now += 3600
        env.table.producer_result(h, compute=Result.OK, seal=Result.OK)
        env.table.publish(h, key(1))
        assert env.store.snapshot()["oldest_writable_dirty_age_s"] == 0
        clock.now += 12
        assert env.store.snapshot()["oldest_writable_dirty_age_s"] == 12
    finally:
        env.ssd.close(io_grace_s=2)


def test_shared_live_variant_overrides_dropped_single_page_queue(tmp_path):
    clock = Clock()
    env = UnitEnv(tmp_path, pages=64, clock=clock)
    try:
        env.variant(ahead=6)
        live = env.variant(ahead=7)
        env.manager.drop_pending_secondary_writes([key(5, 1)])
        env.store.schedule(live, clock.now + 300)
        snap = env.store.snapshot()
        assert snap["dirty_ready_queued_not_due_bytes"] == len(live.keys) * ROW
        assert snap["dirty_ready_filtered_bytes"] == ROW  # only old A
        env.persist(live)
    finally:
        env.ssd.close(io_grace_s=2)
