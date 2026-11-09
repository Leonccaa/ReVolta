# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""unified host budgeting event sequences: content references never substitute for
physical owners."""

from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_host_kv_hit import (
    _mamba,
    _window,
)
from vllm.utils.hashing import sha256
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.host.page_table import (
    HostPageTable,
    PageHandle,
    PageKind,
    Result,
)
from vllm.v1.kv_offload.unified.ledger import HostReservationLedger
from vllm.v1.kv_offload.unified.manager import UnifiedStoreManager
from vllm.v1.kv_offload.unified.units import UnitIndex, ValueClass, VariantOffer

pytestmark = pytest.mark.cpu_test


def key(i, group=0):
    return make_offload_key(sha256(i), group)


def manager(pages=32, workers=4):
    spec = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            host_kv_group_id=0,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(prefix_cacheable=True)),
                SimpleNamespace(kv_cache_spec=_mamba()),
                SimpleNamespace(kv_cache_spec=_window()),
            ],
        ),
        extra_config={},
        cpu_page_size_per_worker=16,
    )
    m = UnifiedStoreManager(spec)
    m.attach(SimpleNamespace(host_pages=HostPageTable(pages, workers), elastic=False))
    return m


def publish(table, k, kind=PageKind.MAIN):
    h = table.reserve()
    assert h is not None
    table.start_io(h, 100000)
    table.get(h).kind = kind
    table.producer_result(h, compute=Result.OK, seal=Result.OK)
    assert table.publish(h, k)
    table.finish_io(h, 100000)
    return h


def store(m, keys, req="r", job=20, success=True):
    out = m.prepare_store(keys, ReqContext(req))
    assert out is not None
    if out.keys_to_store:
        m.bind_job(job, req, out.store_spec, True)
        m.receipt(job, out.store_spec.gen, m.table.all_ranks, 0 if success else 1)
    return out


def complete_boundary(m, n, keys, req="r", job=20):
    assert m.admit_build(req, 11)
    b = m.begin_boundary(req, sha256(n), keys)
    assert b is not None
    store(m, keys, req, job)
    assert m.units.units[b.uid].complete
    return b


def assert_conserved(m):
    u = m.units
    for k, c in u.components.items():
        assert c.refs == {uid for uid, unit in u.units.items() if k in unit.keys}
        if c.cpu is not None:
            assert m.table.lookup(k) == c.cpu
            assert m.table.is_ready(c.cpu)
    for unit in u.units.values():
        for tier in ("cpu", "ssd", "union"):
            assert unit.missing[tier] == sum(
                not u._resident(k, tier) for k in unit.keys
            )
    credits = [*m.ledger.requests.values(), *m.ledger.draining.values()]
    assert len(m.ledger.charges) == sum(c.owned for c in credits)
    for credit in credits:
        assert 0 <= credit.unused + credit.owned <= credit.limit
        assert not credit.draining or credit.unused == 0
    assert m.table.available() == m.table.num_free() + sum(
        p.state.value != "free" and m.table.reclaimable(p) for p in m.table.pages[1:]
    )
    assert len(m.value.heap) <= 4 * len(m.table.pages) + 64


def test_b1_atomic_registration_gate_and_two_arrival_orders():
    m = manager()
    ks = [key(1, 1), key(1, 2)]
    b = complete_boundary(m, 1, ks)
    offer = VariantOffer(b.uid, b.gen, sha256(1), (key(0), key(1)))
    assert m.units.register(offer) is None
    assert m.units.events["variant_main_missing"] == 1
    m.offer_variant("r", offer)
    publish(m.table, key(1))  # A exists but an earlier MAIN page does not.
    m.retry_variants("r")
    assert len(m.units.units) == 1
    publish(m.table, key(0))
    m.retry_variants("r")
    assert len(m.units.units) == 2
    assert not m.waiting
    assert m.units.register(offer) is m.units.register(offer)
    assert_conserved(m)


def test_same_b_two_a_branches_and_last_copy_loss_is_local():
    m = manager()
    b = complete_boundary(m, 2, [key(2, 1), key(2, 2)])
    prefix = publish(m.table, key(0))
    publish(m.table, key(2))
    x, y = publish(m.table, key("X")), publish(m.table, key("Y"))
    vx = m.units.register(
        VariantOffer(b.uid, b.gen, sha256("X"), (key(0), key(2), key("X")))
    )
    vy = m.units.register(
        VariantOffer(b.uid, b.gen, sha256("Y"), (key(0), key(2), key("Y")))
    )
    assert vx.uid != vy.uid
    m.table.invalidate(x)
    assert vx.uid not in m.units.units and vy.uid in m.units.units
    assert m.units.components[key(0)].refs == {vy.uid}
    assert m.table.get(prefix).valid and m.table.get(y).valid
    assert_conserved(m)


def test_cpu_ssd_union_and_gpu_unknown():
    m = manager()
    b = complete_boundary(m, 2, [key(2, 1), key(2, 2)])
    h = publish(m.table, key(2))
    v = m.units.register(VariantOffer(b.uid, b.gen, sha256(2), (key(2),)))
    assert m.table.mark_persisted(h)
    m.table._recycle(m.table.get(h))
    assert not m.units.recoverable(v.uid, "cpu")
    assert not m.units.recoverable(v.uid, "ssd")
    assert m.units.recoverable(v.uid, "union")
    with pytest.raises(ValueError, match="GPU"):
        m.units.recoverable(v.uid, "gpu")
    assert_conserved(m)


def test_b2_three_inventories_and_draining_do_not_double_charge():
    m = manager(pages=13)
    m.kind_by_group = {g: PageKind.STATE for g in range(1, 8)} | {8: PageKind.DRAFT}
    old = [key(1, g) for g in range(1, 8)]
    for k in old:
        publish(m.table, k, PageKind.STATE)
    assert m.admit_build("r", 11)
    b = m.begin_boundary("r", sha256(1), old + [key(i, 8) for i in range(4)])
    assert b.cost == 11
    assert len(b.newly_protected) == 7 and not b.already_owned
    assert m.ledger.snapshot()["build_reserved"] == 0
    assert m.table.available() == 1  # 7 existing + 4 fresh, each charged once.
    out = m.prepare_store(b.new, ReqContext("r"))
    m.bind_job(1, "r", out.store_spec, True)
    m.cancel_request("r")
    assert m.ledger.snapshot()["build_draining"] == 11
    assert not m.admit_build("r", 11)
    assert not m.admit_build("other", 2)
    m.receipt(1, out.store_spec.gen, 15, 0)
    assert "r" not in m.ledger.requests
    assert b.uid not in m.units.units
    assert_conserved(m)


def test_existing_owner_cost_zero_and_foreground_reserve_protected():
    m = manager(pages=15)
    h = publish(m.table, key(1, 1), PageKind.STATE)
    m.table.pin(h, 999)
    assert m.admit_build("r", 11)
    b = m.begin_boundary("r", sha256(1), [key(1, 1), key(1, 2)])
    assert b.cost == 1
    assert m.ledger.requests["r"].unused == 10
    m.table.committed_reserve = lambda: m.ledger.committed() + 2
    assert m.ledger.reserve_frames(1, "background") is None
    m.end_step()
    assert b.uid not in m.units.units
    assert m.ledger.requests["r"].unused == 11
    m.table.release(h, 999)
    assert_conserved(m)


def test_missing_group_source_does_not_survive_step_and_rebuild_new_gen():
    m = manager()
    ks = [key(1, 1), key(1, 2)]
    assert m.admit_build("r", 11)
    b = m.begin_boundary("r", sha256(1), ks)
    out = m.prepare_store([ks[0]], ReqContext("r"))
    m.bind_job(3, "r", out.store_spec, True)
    m.end_step()
    assert b.uid not in m.units.units
    assert m.begin_boundary("r", sha256(1), ks) is None
    assert m.ledger.snapshot()["build_owned"] == 2
    m.receipt(3, out.store_spec.gen, 15, 0)
    fresh = m.begin_boundary("r", sha256(1), ks)
    assert fresh.gen > b.gen
    assert not m.units.revoke(b.uid, b.gen, "build_failed")
    store(m, ks, job=4)
    assert m.units.units[fresh.uid].complete
    assert_conserved(m)


@pytest.mark.parametrize("reset", [False, True])
def test_rank_masks_duplicate_stale_generation_failure_and_cancel(reset):
    m = manager()
    assert m.admit_build("r", 11)
    ks = [key(1, 1)]
    b = m.begin_boundary("r", sha256(1), ks)
    out = m.prepare_store(ks, ReqContext("r"))
    m.bind_job(8, "r", out.store_spec, True)
    gen = out.store_spec.gen
    assert not m.receipt(8, gen + 1, 15, 0)
    for _ in range(10):
        assert not m.receipt(8, gen, 1, 0)
    assert not m.is_ready_resident(ks[0])
    if reset:
        m.reset_cache()
        m.table.reset()
    assert not m.receipt(8, gen, 2, 2)
    assert m.receipt(8, gen, 12, 0)
    assert b.uid not in m.units.units
    assert not m.receipt(8, gen, 15, 0)
    assert not m.jobs and not m.pending and not m.builds
    assert_conserved(m)


def test_b3_shared_draft_all_live_refs_reclassify_after_revoke():
    m = manager()
    shared = key(3, 2)
    b1 = complete_boundary(m, 1, [key(1, 1), shared])
    b2 = complete_boundary(m, 2, [key(2, 1), shared], job=21)
    m.units.demote([key(1, 1)], True)
    assert m.units.value(shared, PageKind.DRAFT) == ValueClass.LEAF
    m.units.revoke(b1.uid, b1.gen, "value")
    assert m.units.components[shared].refs == {b2.uid}
    assert m.units.value(shared, PageKind.DRAFT) == ValueClass.LEAF
    assert_conserved(m)


def test_b4_manifest_only_change_and_file_loss_are_distinct_revisions():
    m = manager()
    b = complete_boundary(m, 1, [key(1, 1)])
    h = publish(m.table, key(1))
    v = m.units.register(VariantOffer(b.uid, b.gen, sha256(1), (key(1),)))
    for k in v.keys:
        m.table.mark_persisted(m.table.lookup(k))
    assert m.units.value(key(1), PageKind.MAIN) == ValueClass.SSD_REDUNDANT
    old_revision = m.table.get(h).revision
    m.units.manifest_ok = lambda unit: unit.uid != v.uid
    m.units.evidence_changed(v.uid)  # Empty actual file deletion set.
    assert m.table.get(h).revision > old_revision
    assert m.units.files_complete(v.uid) and not m.units.cold_recoverable(v.uid)
    assert m.units.value(key(1), PageKind.MAIN) == ValueClass.LEAF
    m.units.ssd_lost(key(1), h)
    assert m.units.recoverable(v.uid, "union")
    m.table.invalidate(h)
    assert v.uid not in m.units.units
    assert_conserved(m)


def test_stale_cpu_loss_does_not_revoke_new_incarnation():
    units = UnitIndex()
    k = key(1, 1)
    old, new = PageHandle(1, 0), PageHandle(1, 1)
    units.cpu_ready(k, old, PageKind.STATE)
    units.cpu_ready(k, new, PageKind.STATE)
    b = units.begin(sha256(1), [k])
    assert units.finish(b.uid, b.gen)
    units.cpu_lost(k, old)
    assert units.recoverable(b.uid, "cpu")


def test_metadata_gc_with_long_lived_shared_main():
    m = manager()
    h = publish(m.table, key(0))
    m.table.pin(h, 888)
    for i in range(1, 1001):
        b = complete_boundary(m, i, [key(i, 1)], job=100 + i)
        tail = publish(m.table, key(i))
        v = m.units.register(VariantOffer(b.uid, b.gen, sha256(i), (key(0), key(i))))
        assert v is not None
        m.table.invalidate(m.table.lookup(key(i, 1)))
        m.table.collect()
        assert not m.units.units
        if tail != h:
            m.table._recycle(m.table.get(tail))
        assert len(m.units.components) == 1
        assert_conserved(m)
    m.table.release(h, 888)


def test_eviction_class_order_and_finite_shared_leaf_exit():
    m = manager(pages=8)
    orphan_main = publish(m.table, key(0))
    orphan_component = publish(m.table, key(9, 1), PageKind.STATE)
    # Credit quota exceeds this tiny pool, so explicitly use a smaller quota.
    assert m.admit_build("r", 2)
    b = m.begin_boundary("r", sha256(1), [key(1, 1), key(1, 2)])
    store(m, b.handles)
    while m.table.num_free():
        publish(m.table, key(100 + m.table.num_free()))
    first = m.table.reserve()
    assert first.page_id == orphan_main.page_id
    m.table.start_io(first, 888)
    # Remaining zero-reference MAIN precedes even older zero-reference STATE.
    assert m.table.reserve().page_id != orphan_component.page_id
    assert_conserved(m)


def test_ledger_failure_is_atomic_and_cached_reclaim_shared_channel():
    table = HostPageTable(4)
    held = [publish(table, key(i)) for i in range(3)]
    for i, h in enumerate(held):
        table.bind(h, i, 0)
    calls: list[int] = []

    def reclaim():
        if len(calls) == 3:
            return False
        i = len(calls)
        calls.append(i)
        table.unbind(held[i], i, 0, 0)
        return True

    ledger = HostReservationLedger(table, reclaim)
    assert ledger.admit("r", 2)
    assert len(calls) == 2
    assert ledger.reserve_frames(2, "background") is None
    assert ledger.consume("r", 2)
    fresh = ledger.reserve_frames(2, "build", reserved=True)
    assert fresh is not None and len(fresh) == 2
    assert len({h.page_id for h in fresh}) == 2
    # Each fresh page has a physical owner before another reserve can run.
    assert all(not table.reclaimable(table.get(h)) for h in fresh)
    ledger.release_temporary(fresh)
    ledger.retire("r", 2)
    assert ledger.snapshot()["build_reserved"] == 2


def test_request_waiting_does_not_protect_components_and_finishes_cleanly():
    m = manager()
    b = complete_boundary(m, 1, [key(1, 1)])
    m.offer_variant("r", VariantOffer(b.uid, b.gen, sha256(3), (key(0),)))
    m.offer_variant("r", VariantOffer(b.uid, b.gen, sha256(4), (key(2),)))
    h = m.table.lookup(key(1, 1))
    assert m.table.reclaimable(m.table.get(h))
    m.on_request_finished(ReqContext("r"))
    assert not m.waiting and not m.ledger.requests
    assert_conserved(m)


def test_receive_component_inventory_and_out_of_order_jobs():
    m = manager()
    complete_boundary(m, 1, [key(1, 1), key(1, 2)])
    a = m.prepare_load([key(1, 1)], ReqContext("a"))
    inventory = m.receive_components(a)
    assert inventory[0].handle == a.handles[0]
    assert inventory[0].group == 1 and inventory[0].load_generation == a.gen
    m.bind_job(1, "a", a, False)
    b = m.prepare_load([key(1, 2)], ReqContext("a"))
    m.bind_job(2, "a", b, False)
    m.receipt(2, b.gen, 15, 0)
    assert a.gen in m.table.get(a.handles[0]).leases
    m.receipt(1, a.gen, 15, 0)
    assert not m._loads and not m._prepared
    assert_conserved(m)


def test_normal_finish_keeps_accepted_dma_in_draining_until_success():
    m = manager()
    assert m.admit_build("r", 11)
    ks = [key(1, 1)]
    b = m.begin_boundary("r", sha256(1), ks)
    out = m.prepare_store(ks, ReqContext("r"))
    m.bind_job(1, "r", out.store_spec, True)
    m.on_request_finished(ReqContext("r"))
    assert m.ledger.snapshot()["build_draining"] == 1
    assert b.uid in m.units.units and not m.units.units[b.uid].complete
    m.receipt(1, out.store_spec.gen, 15, 0)
    assert m.units.units[b.uid].complete
    assert not m.ledger.requests
    assert_conserved(m)


def test_optional_extras_cannot_refuse_an_already_accepted_boundary():
    m = manager(pages=5)
    assert m.admit_build("r", 2)
    b = m.begin_boundary("r", sha256(1), [key(1, 1), key(1, 2)])
    for i in range(2):
        h = publish(m.table, key(i))
        m.table.pin(h, 99)
    # No surplus for optional window pages; the two accepted frames exist.
    out = m.prepare_store([*b.handles, key(7, 2)], ReqContext("r"))
    assert set(out.keys_to_store) == set(b.handles)
    m.bind_job(1, "r", out.store_spec, True)
    m.receipt(1, out.store_spec.gen, 15, 0)
    assert m.units.units[b.uid].complete
    assert_conserved(m)


def test_foreground_lookup_lease_handoff_and_background_surplus():
    m = manager(pages=6)
    h = publish(m.table, key(1, 1), PageKind.STATE)
    assert m.admit_build("r", 4)
    assert m.hold_lookup("r", [key(1, 1)])
    assert m.table.net_headroom() == 0
    assert m.ledger.reserve_frames(1, "background") is None
    spec = m.prepare_load([key(1, 1)], ReqContext("r"))
    assert spec.gen in m.table.get(h).leases and not m._lookup_holds
    m.bind_job(1, "r", spec, False)
    m.cancel_request("r")  # Cancellation cannot retire the load's physical lease.
    assert spec.gen in m.table.get(h).leases
    m.receipt(1, spec.gen, 15, 0)
    assert m.table.reclaimable(m.table.get(h))
    assert_conserved(m)


def test_shared_inflight_draft_has_one_dma_and_one_payment_until_last_owner():
    m = manager()
    shared = key(1, 2)
    assert m.admit_build("a", 11) and m.admit_build("b", 11)
    a = m.begin_boundary("a", sha256(1), [key(1, 1), shared])
    first = m.prepare_store([shared], ReqContext("a"))
    m.bind_job(1, "a", first.store_spec, True)
    b = m.begin_boundary("b", sha256(2), [key(2, 1), shared])
    assert a.cost == 2 and b.cost == 1
    # A and B share the original physical handle, never a second allocation.
    assert a.handles[shared] == b.handles[shared]
    m.receipt(1, first.store_spec.gen, 15, 0)
    store(m, [key(1, 1)], "a", job=2)
    assert m.units.units[a.uid].complete
    assert m.ledger.requests["a"].owned == 1  # Shared B owner still uses it.
    second = m.prepare_store([key(2, 1), shared], ReqContext("b"))
    m.bind_job(3, "b", second.store_spec, True)
    m.on_request_finished(ReqContext("a"))
    assert m.ledger.snapshot()["build_draining"] == 1
    m.receipt(3, second.store_spec.gen, 15, 0)
    assert m.units.units[b.uid].complete
    assert "a" not in m.ledger.requests
    assert m.ledger.requests["b"].unused == 11
    assert not m.ledger.charges and not m.dependents
    assert m.num_stored_blocks == 3
    assert_conserved(m)


def test_failed_boundary_does_not_discard_healthy_siblings_shared_dma():
    m = manager()
    shared = key(1, 2)
    assert m.admit_build("a", 11) and m.admit_build("b", 11)
    a = m.begin_boundary("a", sha256(1), [key(1, 1), shared])
    b = m.begin_boundary("b", sha256(2), [key(2, 1), shared])
    out = m.prepare_store([shared, key(2, 1)], ReqContext("b"))
    m.bind_job(1, "b", out.store_spec, True)
    m.end_step()  # A's private STATE source was never dispatched.
    assert a.uid not in m.units.units and b.uid in m.units.units
    m.receipt(1, out.store_spec.gen, 15, 0)
    assert m.units.units[b.uid].complete
    assert m.is_ready_resident(shared)
    assert_conserved(m)


def test_b5_transfer_increment_and_b6_shrink_obligation_event_sequences():
    """Actual graph + test-only credit oracle; the unit SSD transactions grant policy is
    not enabled.

    Best-effort recovery is covered by six finite examples
    without implementing or claiming production guarantee credit before unit SSD
    transactions.
    """
    units = UnitIndex()
    made = []
    for number, count in [(1, 2), (2, 5), (3, 4)]:
        keys = [key(number, i + 1) for i in range(count)]
        for i, k in enumerate(keys):
            units.cpu_ready(k, PageHandle(number * 10 + i, 0), PageKind.STATE)
        b = units.begin(sha256(number), keys)
        assert units.finish(b.uid, b.gen)
        main = key(number)
        units.cpu_ready(main, PageHandle(number * 10 + count, 0), PageKind.MAIN)
        variant = units.register(VariantOffer(b.uid, b.gen, sha256(number), (main,)))
        assert variant is not None
        made.append(variant.uid)
    units.manifest_ok = lambda u: False
    old, new, other = made

    def obligation(grants):
        files, manifests = set(), 0
        for uid in grants:
            missing, manifest = units.persistence_obligations(uid)
            files.update(missing)
            manifests += manifest
        return len(files) + manifests

    # B5: old 4 + other 6 = 10. Removing old before adding new must check
    # the 13-byte union; copying an old numeric credit would violate the cap.
    granted = {old, other}
    assert obligation(granted) == 10
    granted.remove(old)
    assert obligation(granted | {new}) == 13
    assert obligation(granted) <= 10
    # B6: downshift 10 -> 5 requires immediate demotion, including metadata
    # obligations. Physical in-flight writes remain owned, independently.
    inflight = 8
    while obligation(granted) > 5:
        granted.remove(other)
    assert obligation(granted) == 0 and inflight == 8
    # A missing-manifest-only change keeps all logical references alive.
    for k in units.units[new].keys:
        units.persisted(k, units.components[k].cpu)
    assert units.persistence_obligations(new) == (frozenset(), True)
    parent = units.units[new].parent
    assert parent is not None
    for k in units.units[new].keys:
        expected = {new, parent} if k in units.units[parent].keys else {new}
        assert units.components[k].refs == expected
    assert units.snapshot()["guaranteed_bytes"] == 0


def test_randomized_presence_and_reference_conservation():
    import random

    rng = random.Random(20261108)
    u = UnitIndex()
    epoch = 0
    for step in range(5000):
        k = key(rng.randrange(24), 1)
        c = u.components.get(k)
        action = rng.randrange(8)
        if action == 0:
            epoch += 1
            u.cpu_ready(k, PageHandle(rng.randrange(100), epoch), PageKind.STATE)
        elif action == 1 and c is not None and c.cpu is not None:
            u.persisted(k, c.cpu)
        elif action == 2 and c is not None and c.cpu is not None:
            u.cpu_lost(k, c.cpu)
        elif action == 3 and c is not None and c.persisted is not None:
            u.ssd_lost(k, c.persisted)
        elif action == 4 and c is not None and (c.cpu or c.persisted):
            b = u.begin(k[:-4], [k])
            u.finish(b.uid, b.gen)
        elif action == 5 and u.units:
            b = rng.choice(list(u.units.values()))
            u.revoke(b.uid, b.gen, "value")
        elif action == 6:
            u.demote([k], bool(step % 2))
        elif action == 7 and step % 97 == 0:
            u.reset()
        for unit in u.units.values():
            for tier in ("cpu", "ssd", "union"):
                assert unit.missing[tier] == sum(
                    not u._resident(x, tier) for x in unit.keys
                )
            assert u.recoverable(unit.uid, "union")
        for component_key, component in u.components.items():
            assert component.refs == {
                uid for uid, unit in u.units.items() if component_key in unit.keys
            }


def test_leaf_page_spill_retains_union_until_unit_files_are_complete():
    m = manager(pages=4)
    assert m.admit_build("r", 2)
    b = m.begin_boundary("r", sha256(1), [key(1, 1), key(1, 2)])
    store(m, b.handles)
    m.ledger.finish("r")
    h = m.table.lookup(key(1, 1))
    m.table.mark_persisted(h)
    other = publish(m.table, key(0))
    m.table.pin(other, 999)
    fresh = m.table.reserve()
    assert fresh.page_id == h.page_id
    assert b.uid in m.units.units and m.units.recoverable(b.uid, "union")
    assert not m.units.recoverable(b.uid, "cpu")
    assert not m.units.files_complete(b.uid)
    assert_conserved(m)


def test_main_seal_actual_byte_receipts_are_rank_and_ticket_idempotent():
    m = manager()
    t = m.table
    h = t.reserve()
    t.bind(h, 1, 1)
    ticket = t.open_seal(h, 1, 1, key(1), 1)
    for _ in range(3):
        assert not t.seal_result(ticket, 1, 0, dma_bytes={0: 42})
    assert t.dma_bytes["dma_d2h_main"] == 42
    assert t.seal_result(ticket, 14, 2, dma_bytes={1: 42, 2: 42, 3: 42})
    assert t.dma_bytes["dma_d2h_main"] == 126  # Failed rank's bytes are unknown.
    assert not t.seal_result(ticket, 15, 0, dma_bytes={0: 42})
    assert t.dma_bytes["dma_d2h_main"] == 126


def test_main_batch_keeps_later_allocations_out_of_writeback_surplus():
    m = manager(pages=5)
    states = [key(1, 1), key(1, 2)]
    for k in states:
        publish(m.table, k, m.kind(k))
    b = m.units.begin(sha256(1), states)
    assert m.units.finish(b.uid, b.gen)
    main = [key(0), key(1)]
    for k in main:
        publish(m.table, k)
    assert m.units.register(VariantOffer(b.uid, b.gen, sha256(1), tuple(main)))
    seen = []

    class Persistence:
        def try_writeback(self, page, headroom):
            seen.append(headroom)
            return False

        def on_evict(self, page):
            pass

    m.table.persistence = Persistence()
    held = m.ledger.reserve_frames(1, "main", reserved=True, pending=4)
    assert held is not None and seen and max(seen) == 0
    m.ledger.release_temporary(held)
    assert_conserved(m)


def test_cpu_serviceability_requires_variant_main_not_only_complete_b():
    m = manager()
    states = [key(1, 1), key(1, 2)]
    b = complete_boundary(m, 1, states)
    assert all(m.units.cpu_status(k) == "unknown" for k in states)
    main = [key(0), key(1)]
    handles = [publish(m.table, k) for k in main]
    v = m.units.register(VariantOffer(b.uid, b.gen, sha256(1), tuple(main)))
    assert v is not None
    assert all(
        m.units.cpu_status(k) == "potentially_servable" for k in (*states, *main)
    )
    assert m.table.mark_persisted(handles[0])
    m.table._recycle(m.table.get(handles[0]))
    assert m.units.recoverable(b.uid, "cpu")
    assert m.units.recoverable(v.uid, "union")
    assert not m.units.recoverable(v.uid, "cpu")
    assert all(m.units.cpu_status(k) == "missing_dependency" for k in states)
    assert m.residency_totals()["state", "missing_dependency"] == 1
    assert m.units.snapshot()["gpu_residency_unknown"] == 1
    assert_conserved(m)


def test_optional_inflight_component_has_one_physical_producer():
    m = manager()
    keys = [key(1, 2)]
    first = m.prepare_store(keys, ReqContext("r"))
    assert first is not None and first.keys_to_store == keys
    m.bind_job(1, "r", first.store_spec, True)
    second = m.prepare_store(keys, ReqContext("other"))
    assert second is not None and not second.keys_to_store
    h = first.store_spec.handles[0]
    assert m.table.get(h).io == {first.store_spec.gen}
    assert m.receipt(1, first.store_spec.gen, m.table.all_ranks, 0)
    assert m.is_ready_resident(keys[0])
    assert not m._dispatched and not m.pending
    assert_conserved(m)


def test_build_can_share_optional_inflight_component_without_second_dma():
    m = manager()
    draft = key(1, 2)
    extra = m.prepare_store([draft], ReqContext("extra"))
    assert extra is not None
    m.bind_job(1, "extra", extra.store_spec, True)
    assert m.admit_build("r", 11)
    b = m.begin_boundary("r", sha256(1), [key(1, 1), draft])
    assert b is not None
    state = m.prepare_store([key(1, 1), draft], ReqContext("r"))
    assert state is not None and state.keys_to_store == [key(1, 1)]
    m.bind_job(2, "r", state.store_spec, True)
    m.receipt(2, state.store_spec.gen, m.table.all_ranks, 0)
    assert not m.units.units[b.uid].complete
    m.receipt(1, extra.store_spec.gen, m.table.all_ranks, 0)
    assert m.units.units[b.uid].complete
    assert not m._dispatched and not m.pending
    assert_conserved(m)


def test_partial_state_demotion_refreshes_chain_only_on_value_class_change():
    m = manager()
    states = [key(1, 1), key(1, 3)]
    for k in states:
        publish(m.table, k, PageKind.STATE)
    b = m.units.begin(sha256(1), states)
    assert m.units.finish(b.uid, b.gen)
    main = [key(0), key(1)]
    handles = [publish(m.table, k) for k in main]
    assert m.units.register(VariantOffer(b.uid, b.gen, sha256(1), tuple(main)))
    h = handles[0]
    m.table.pin(h, 99)
    revision = m.table.get(h).revision
    m.units.demote([states[0]], True)
    assert m.units.value(main[0], PageKind.MAIN) == ValueClass.LEAF
    assert m.table.get(h).revision == revision
    m.units.demote([states[1]], True)
    assert m.units.value(main[0], PageKind.MAIN) == ValueClass.INTERIOR
    assert m.table.get(h).revision > revision
    revision = m.table.get(h).revision
    m.units.demote([states[0]], False)
    assert m.units.value(main[0], PageKind.MAIN) == ValueClass.LEAF
    assert m.table.get(h).revision > revision
    revision = m.table.get(h).revision
    m.units.demote([states[1]], False)
    assert m.table.get(h).revision == revision
    m.table.release(h, 99)
    assert_conserved(m)


def test_readmission_credit_is_independent_of_old_draining_owner():
    m = manager()
    assert m.admit_build("r", 11)
    old = m.ledger.requests["r"]
    b = m.begin_boundary("r", sha256(1), [key(1, 1), key(1, 2)])
    out = m.prepare_store(b.handles, ReqContext("r"))
    m.bind_job(1, "r", out.store_spec, True)
    m.cancel_request("r")
    assert old.draining and old.unused == 0 and old.owned == 2
    assert m.admit_build("r", 11)
    fresh = m.ledger.requests["r"]
    assert fresh.generation > old.generation and fresh.unused == 11
    assert_conserved(m)
    m.receipt(1, out.store_spec.gen, m.table.all_ranks, 0)
    assert old.owned == 0 and not m.ledger.draining and not m.ledger.charges
    assert m.ledger.requests["r"] is fresh and fresh.unused == 11
    assert not m.units.units


def test_ready_event_rechecks_inventory_lost_while_waiting():
    m = manager()
    b = complete_boundary(m, 1, [key(1, 1), key(1, 2)])
    first = publish(m.table, key(0))
    offer = VariantOffer(b.uid, b.gen, sha256(1), (key(0), key(1)))
    m.offer_variant("reader", offer)
    m.table._recycle(m.table.get(first))
    publish(m.table, key(1))
    assert not m.units.units[b.uid].variants
    assert m._wait_keys["reader"] == {key(0)}
    publish(m.table, key(0))
    assert m.units.units[b.uid].variants and not m.waiting
    assert not m._missing_main and not m._waiting_boundaries
    assert_conserved(m)


def test_boundary_completion_wakes_other_request_without_step_retry():
    m = manager()
    assert m.admit_build("producer", 11)
    keys = [key(1, 1), key(1, 2)]
    b = m.begin_boundary("producer", sha256(1), keys)
    for k in (key(0), key(1)):
        publish(m.table, k)
    m.offer_variant("reader", VariantOffer(b.uid, b.gen, sha256(1), (key(0), key(1))))
    assert "reader" in m.waiting
    store(m, keys, req="producer")
    assert m.units.units[b.uid].variants and not m.waiting
