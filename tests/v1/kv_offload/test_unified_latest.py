# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latest/resend retention through the real RAM and SSD transaction owners."""

import threading
from pathlib import Path

import pytest

from tests.v1.kv_offload.test_host_page_ssd import ROW, key
from tests.v1.kv_offload.test_unified_ssd_transactions import UnitEnv
from tests.v1.kv_offload.test_unified_units import assert_conserved, manager, publish
from vllm.utils.hashing import sha256
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.unified.retention import LatestCheckpoints
from vllm.v1.kv_offload.unified.units import UnitId, VariantOffer

pytestmark = pytest.mark.cpu_test


def enable(m, keep=False, evidence=False):
    m.latest_only = True
    m.enable_junction_evidence = evidence
    m.retention = LatestCheckpoints(m, keep)
    m.units.on_event = m.retention.event
    if evidence:
        m.units.on_retention_offer = m.retention.offer_evidence
    m.spec.unified_frame_bytes = ROW
    return m


def point(m, path, *, resend=None, request_length=None, junction=False, draft=()):
    boundary = path[-1]
    components = (key(boundary, 1), *draft)
    for k in (*map(key, path), *components):
        if not m.is_ready_resident(k):
            publish(m.table, k, m.table.kind_for_key(k))
    b = m.units.begin(sha256(boundary), components)
    assert m.units.finish(b.uid, b.gen)
    offer = VariantOffer(
        b.uid,
        b.gen,
        sha256(boundary),
        tuple(map(key, path)),
        len(path) * 16,
        len(path) * 16,
        (max(0, len(path) - 2) * 16, len(path) * 16),
        {
            "request_length": request_length or len(path) * 16 + 1,
            "resend": sha256(resend).hex() if resend is not None else None,
            "junction": junction,
        },
    )
    m.offer_variant("producer", offer)
    return m.units.units[UnitId(b.uid.boundary, b.uid.reset, offer.ahead)]


def boundaries(m):
    return {u.uid.boundary for u in m.units.units.values() if u.parent is None}


def test_linear_latest_and_exact_resend_with_continuous_resends():
    m = enable(manager(pages=100, workers=1))
    prefix = []
    for turn in range(8):
        prefix.extend([turn * 2, turn * 2 + 1])
        resend = point(
            m, prefix[:-1], resend=prefix[-2], request_length=len(prefix) * 16
        )
        latest = point(m, prefix, resend=prefix[-2], request_length=len(prefix) * 16)
        m.retention.step()
        for retry in range(4):
            # A new request instance reuses precisely the same content point.
            assert (
                m.lookup(key(prefix[-2], 1), ReqContext(f"retry-{turn}-{retry}"))
                is True
            )
            point(m, prefix, resend=prefix[-2], request_length=len(prefix) * 16)
            m.retention.step()
        assert boundaries(m) == {resend.uid.boundary, latest.uid.boundary}
        assert m.retention.snapshot()["branch_points"] == 0
        assert m.retention.snapshot()["resend_points"] == 1
        assert_conserved(m)
    assert m.events["superseded_revoked"] == 14
    assert m.events["superseded_freed_bytes"] == 14 * ROW


def test_structural_successors_and_precise_junction_survive():
    m = enable(manager(pages=80, workers=1))
    root = point(m, [0], junction=True)
    point(m, [0, 1])
    point(m, [0, 2])
    m.retention.step()
    assert root.parent in m.retention.branches
    assert boundaries(m) == {sha256(0), sha256(1), sha256(2)}
    # Serial descendants on each arm do not turn their ancestors into branches.
    point(m, [0, 1, 3])
    point(m, [0, 2, 4])
    m.retention.step()
    assert boundaries(m) == {sha256(0), sha256(3), sha256(4)}
    assert m.retention.snapshot()["branch_points"] == 1
    assert_conserved(m)


def test_branch_observed_before_ancestor_registration():
    m = enable(manager(pages=80, workers=1))
    point(m, [0, 1])
    point(m, [0, 2])
    root = point(m, [0])
    m.retention.step()
    assert root.parent in m.retention.branches
    assert sha256(0) in boundaries(m)


@pytest.mark.parametrize("hold", ["lease", "io", "bind", "watermark"])
def test_inflight_component_use_delays_revocation(hold):
    m = enable(manager(pages=40, workers=1))
    old = point(m, [0])
    handle = m.table.lookup(key(0, 1))
    if hold == "lease":
        assert m.hold_lookup("receive", [key(0, 1)])
    elif hold == "io":
        m.table.start_io(handle, 99)
    elif hold == "bind":
        m.table.bind(handle, 99, 0)
    else:
        m.table.touch(handle, 2)
    point(m, [0, 1])
    m.retention.step()
    assert old.parent in m.units.units
    if hold == "lease":
        m.release_lookup("receive")
    elif hold == "io":
        m.table.finish_io(handle, 99)
    elif hold == "bind":
        m.table.unbind(handle, 99, 0, 0)
    else:
        m.table.watermark = 2
    m.retention.step()
    assert old.parent not in m.units.units
    assert_conserved(m)


def test_shared_draft_reference_count_and_exactly_once_free():
    m = enable(manager(pages=40, workers=1))
    shared = key(7, 2)
    old = point(m, [0], draft=(shared,))
    point(m, [0, 1], draft=(shared,))
    m.retention.step()
    assert old.parent not in m.units.units
    assert m.is_ready_resident(shared)
    assert len(m.units.components[shared].refs) == 2
    point(m, [0, 1, 2])
    m.retention.step()
    assert not m.is_ready_resident(shared)
    assert m.events["superseded_freed_bytes"] == 3 * ROW
    m.retention.step()
    assert m.events["superseded_freed_bytes"] == 3 * ROW
    assert_conserved(m)


def env_point(env, boundary, *, resend=None, request_length=None):
    unit = env.variant(boundary)
    unit.retention.update(
        request_length=request_length or (boundary + 1) * 16 + 1,
        resend=None if resend is None else sha256(resend).hex(),
        junction=False,
    )
    env.manager.units.on_event("observe", unit)
    return unit


def test_value_eviction_keeps_ssd_cleanup_obligation(tmp_path):
    env = UnitEnv(tmp_path, pages=100, latest_only=True)
    env.manager.spec.unified_frame_bytes = ROW
    try:
        old = env_point(env, 2)
        env.persist(old)
        old_paths = [env.store.path(key(2, g)) for g in range(1, 8)]
        old_manifest = env.store.manifest_paths[old.uid]
        env_point(env, 4)
        env.manager.retention.refresh()
        assert old.parent in env.manager.retention.pending
        env.manager.units.revoke(
            old.parent, env.manager.units.units[old.parent].gen, "value"
        )
        assert old.uid in env.store.manifests
        assert env.settle(lambda: old.uid not in env.store.manifests)
        assert all(not Path(p).exists() for p in [old_manifest, *old_paths])
        assert env.manager.events["superseded_revoked"] == 0
        assert env.manager.events["superseded_ssd_deleted_bytes"] > 7 * ROW
        assert env.store.cache.lookup(env.store.path(key(0)))
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


def test_delete_receipt_blocks_rebuild_and_protects_witness(tmp_path, monkeypatch):
    env = UnitEnv(tmp_path, pages=100, latest_only=True)
    gate, started = threading.Event(), threading.Event()
    original = env.store._delete_manifests

    def gated(txn):
        started.set()
        assert gate.wait(10)
        original(txn)

    monkeypatch.setattr(env.store, "_delete_manifests", gated)
    try:
        old = env_point(env, 2)
        env.persist(old)
        new = env_point(env, 4)
        env.manager.retention.step()
        assert started.wait(5)
        assert env.manager.lookup(key(2, 1), ReqContext("late")) is False
        assert not env.manager.units.revoke(
            new.parent, env.manager.units.units[new.parent].gen, "value"
        )
        assert not env.store.start_write(old)
        gate.set()
        assert env.settle(lambda: old.uid not in env.store.manifests)
        assert old.parent not in env.manager.units.units
        assert not env.manager.units.protected
        assert_conserved(env.manager)
    finally:
        gate.set()
        env.ssd.close(io_grace_s=5)


def test_keep_ssd_frees_ram_then_restores_identical_bytes(tmp_path):
    env = UnitEnv(tmp_path, pages=100, latest_only=True, keep_ssd=True)
    env.manager.spec.unified_frame_bytes = ROW
    try:
        old = env_point(env, 2)
        expected = {
            k: bytes(
                env.backing[
                    env.table.lookup(k).page_id * ROW : (
                        env.table.lookup(k).page_id + 1
                    )
                    * ROW
                ]
            )
            for k in old.keys
        }
        env_point(env, 4)
        env.manager.retention.step()
        assert env.manager.is_ready_resident(key(2, 1))
        env.persist(old)
        assert env.settle(lambda: not env.manager.is_ready_resident(key(2, 1)))
        assert old.parent in env.manager.units.units
        assert old.uid in env.store.manifests
        assert env.manager.events["superseded_revoked"] == 0
        assert env.manager.events["superseded_ssd_deleted_bytes"] == 0
        assert env.manager.events["superseded_freed_bytes"] >= 7 * ROW
        assert env.ssd.prefetch("cold", [key(2, 1)])
        assert env.settle(lambda: not env.ssd.waiting("cold"))
        for k, value in expected.items():
            h = env.table.lookup(k)
            assert env.backing[h.page_id * ROW : (h.page_id + 1) * ROW] == value
        env.ssd.cancel("cold")
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


def test_restart_compacts_committed_manifests_and_new_generation_is_safe(tmp_path):
    writer = UnitEnv(tmp_path, pages=120)
    old = writer.variant(2)
    latest = writer.variant(4)
    writer.persist(old)
    writer.persist(latest)
    writer.ssd.close(io_grace_s=5)
    reader = UnitEnv(tmp_path, pages=120, latest_only=True)
    try:
        # Legacy manifests retain the immediate predecessor for resend safety.
        newer = env_point(reader, 6)
        reader.persist(newer)
        assert reader.settle(lambda: old.uid not in reader.store.manifests)
        assert not Path(reader.store.manifest_path(old)).exists()
        # Rebuild the same content in this process. A retired deletion cannot
        # touch its new generation or its new manifest path.
        rebuilt = env_point(reader, 2, request_length=200)
        assert rebuilt.gen != old.gen
        reader.manager.retention.keep_ssd = True
        reader.persist(rebuilt)
        rebuilt_path = reader.store.manifest_paths[rebuilt.uid]
        assert Path(rebuilt_path).exists()
        reader.ssd.close(io_grace_s=5)
        again = UnitEnv(tmp_path, pages=120, latest_only=True, keep_ssd=True)
        try:
            assert rebuilt.uid in again.store.manifests
            assert again.store.manifest_paths[rebuilt.uid] == rebuilt_path
        finally:
            again.ssd.close(io_grace_s=5)
    finally:
        reader.ssd.close(io_grace_s=5)


def test_disabled_has_no_new_stats_or_manifest_fields(tmp_path):
    env = UnitEnv(tmp_path, pages=100)
    try:
        old = env.variant(2)
        env.variant(4)
        env.persist(old)
        assert env.manager.retention is None
        assert len(boundaries(env.manager)) == 2
        assert "retention" not in env.store.document(old)
        assert not any(
            "superseded" in k or k in ("branch_points", "chain_latest_units")
            for k in env.manager.snapshot()
        )
        assert not any(k.startswith("superseded_") for k in env.manager.events)
    finally:
        env.ssd.close(io_grace_s=5)


def test_store_stats_shape_223_checkpoints():
    m = enable(manager(pages=700, workers=1))
    m.spec.unified_frame_bytes = 23_003_136
    for i in range(223):
        point(m, list(range(i + 1)), resend=221, request_length=223 * 16)
    before = m.snapshot()
    for _ in range(30):
        m.retention.step()
    after = m.snapshot()
    assert before["allocated_state_bytes"] == 223 * m.spec.unified_frame_bytes
    assert after["allocated_state_bytes"] == 2 * m.spec.unified_frame_bytes
    assert m.events["superseded_freed_bytes"] == 221 * m.spec.unified_frame_bytes
    assert before["allocated_main_bytes"] == after["allocated_main_bytes"]
    assert m.retention.snapshot()["superseded_pending"] == 0


@pytest.mark.parametrize("transaction", ["write", "prefetch"])
def test_unit_transactions_retire_before_revoke(tmp_path, transaction):
    env = UnitEnv(tmp_path, pages=120, latest_only=True)
    try:
        old = env_point(env, 2)
        if transaction == "write":
            assert env.store.start_write(old)
        else:
            env.persist(old)
            env.evict_cpu(old)
            assert env.ssd.prefetch("receiver", [key(2, 1)])
        env_point(env, 4)
        env.manager.retention.step()
        assert old.parent in env.manager.units.units
        if transaction == "prefetch":
            assert env.settle(lambda: not env.ssd.waiting("receiver"))
            # A completed prefetch still owns its sources until handoff/cancel.
            env.manager.retention.step()
            assert old.parent in env.manager.units.units
            env.ssd.cancel("receiver")
        assert env.settle(lambda: old.parent not in env.manager.units.units)
        assert old.uid not in env.store.manifests
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


def test_two_a_variants_latch_branch(tmp_path):
    env = UnitEnv(tmp_path, pages=120, latest_only=True)
    try:
        first = env.variant(2, ahead=30)
        second = env.variant(2, ahead=31)
        env.manager.retention.step()
        assert first.parent in env.manager.retention.branches
        env.persist(first)
        env.persist(second)
        env.manager.units.revoke(second.uid, second.gen, "value")
        assert env.settle(
            lambda: env.store.manifests[first.uid].get("retention", {}).get("branch")
        )
        env.ssd.close(io_grace_s=5)
        reader = UnitEnv(tmp_path, pages=120, latest_only=True)
        try:
            reader.manager.retention.refresh()
            assert first.parent in reader.manager.retention.branches
        finally:
            reader.ssd.close(io_grace_s=5)
    finally:
        env.ssd.close(io_grace_s=5)


def test_only_committed_old_point_survives_restart_without_new_evidence(tmp_path):
    env = UnitEnv(tmp_path, pages=120, latest_only=True)
    old = env_point(env, 2)
    env.persist(old)
    env.ssd.close(io_grace_s=5)
    reader = UnitEnv(tmp_path, pages=120, latest_only=True)
    try:
        for _ in range(5):
            reader.ssd.step()
        assert old.uid in reader.store.manifests
        assert not reader.manager.retention.pending
    finally:
        reader.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("seed", range(8))
def test_randomized_linear_resend_and_shared_components(seed):
    import random

    rng = random.Random(seed)
    m = enable(manager(pages=300, workers=1))
    path: list[int] = []
    for turn in range(30):
        path.extend(range(len(path), len(path) + rng.randrange(2, 5)))
        resend = path[-2]
        for index in range(max(0, len(path) - 4), len(path)):
            point(
                m,
                path[: index + 1],
                resend=resend,
                request_length=len(path) * 16,
                draft=(key(path[max(0, index - 1)], 2), key(path[index], 2)),
            )
        for _ in range(rng.randrange(1, 6)):
            point(m, path, resend=resend, request_length=len(path) * 16)
            m.retention.step()
        assert boundaries(m) == {sha256(resend), sha256(path[-1])}
        assert not m.retention.branches
        assert_conserved(m)


def test_keep_ssd_waits_for_manifest_not_only_component_files(tmp_path):
    env = UnitEnv(tmp_path, pages=120, latest_only=True, keep_ssd=True)
    try:
        old = env_point(env, 2)
        env_point(env, 4)
        # Component files alone are insufficient to archive an old checkpoint.
        for key_ in old.keys:
            handle = env.table.lookup(key_)
            env.table.mark_persisted(handle)
        env.manager.retention.step()
        assert env.manager.is_ready_resident(key(2, 1))
        assert not env.manager.units.cold_recoverable(old.uid)
        assert env.manager.events["superseded_freed_bytes"] == 0
    finally:
        env.ssd.close(io_grace_s=5)


def test_required_metrics_export_through_store_stats_and_prometheus():
    from functools import partial

    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    from vllm.v1.metrics.store import StoreMetrics, StoreMetricsProm

    m = enable(manager(pages=40, workers=1))
    point(m, [0])
    point(m, [0, 1])
    m.retention.step()
    payload = StoreMetrics().export(60.0, manager=m)
    registry = CollectorRegistry()
    prom = StoreMetricsProm(
        partial(Counter, registry=registry),
        partial(Gauge, registry=registry),
        partial(Histogram, registry=registry),
        ["engine"],
        {0: ["0"]},
    )
    prom.observe(payload, 0)
    for key_, expected in (
        ("unified_superseded_revoked", 1),
        ("unified_superseded_freed_bytes", ROW),
        ("unified_superseded_ssd_deleted_bytes", 0),
    ):
        assert (
            registry.get_sample_value(
                "vllm:store_events_total", {"engine": "0", "event": key_}
            )
            == expected
        )
    for key_, expected in (
        ("unified_branch_points", 0),
        ("unified_chain_latest_units", 1),
    ):
        assert (
            registry.get_sample_value(
                "vllm:store_state", {"engine": "0", "metric": key_}
            )
            == expected
        )


def test_new_generation_does_not_inherit_uncommitted_branch_latch():
    m = enable(manager(pages=80, workers=1))
    root = point(m, [0])
    a = point(m, [0, 1])
    b = point(m, [0, 2])
    m.retention.refresh()
    assert root.parent in m.retention.branches
    for u in (a, b, root):
        parent = m.units.units[u.parent]
        m.units.revoke(parent.uid, parent.gen, "value")
    # No idle refresh between value eviction and same-content reconstruction.
    rebuilt = point(m, [0])
    point(m, [0, 3])
    m.retention.step()
    assert rebuilt.parent not in m.retention.branches
    assert rebuilt.parent not in m.units.units


def test_unknown_boundary_positions_are_preserved():
    m = enable(manager(pages=40, workers=1))
    unit = point(m, [0])
    m.retention.reset()
    unit.boundary_position = 0
    m.retention.observe(unit)
    point(m, [0, 1])
    m.retention.step()
    assert unit.parent in m.units.units


def test_reusing_old_latest_does_not_advance_request_resend_exemption():
    m = enable(manager(pages=50, workers=1))
    old_resend = point(m, [0], resend=0, request_length=32)
    point(m, [0, 1], resend=0, request_length=32)
    m.retention.step()
    # The next request reuses the old B/A, but has not published a new B yet.
    point(m, [0, 1], resend=3, request_length=64)
    m.retention.step()
    assert old_resend.parent in m.units.units
    point(m, [0, 1, 2], resend=3, request_length=64)
    m.retention.step()
    assert old_resend.parent not in m.units.units


def test_generation_path_reuse_after_restart_has_no_stale_delete_intent(tmp_path):
    writer = UnitEnv(tmp_path, pages=120, latest_only=True)
    old = env_point(writer, 2)
    writer.persist(old)
    old_path = writer.store.manifest_paths[old.uid]
    env_point(writer, 4)  # The replacement stays in RAM only.
    assert writer.settle(lambda: old.uid not in writer.store.manifests)
    writer.ssd.close(io_grace_s=5)
    reader = UnitEnv(tmp_path, pages=120, latest_only=True)
    try:
        assert not reader.store.manifests
        rebuilt = env_point(reader, 2)
        reader.persist(rebuilt)
        # Reusing a fully deleted path is safe. No prior-process deletion
        # intent or job can survive to target this new committed inventory.
        assert reader.store.manifest_paths[rebuilt.uid] == old_path
        assert not reader.store.retired_manifests
        assert not reader.store.manifest_jobs
        reader.ssd.close(io_grace_s=5)
        again = UnitEnv(tmp_path, pages=120, latest_only=True)
        try:
            assert rebuilt.uid in again.store.manifests
            assert again.store.manifest_paths[rebuilt.uid] == old_path
        finally:
            again.ssd.close(io_grace_s=5)
    finally:
        reader.ssd.close(io_grace_s=5)


def test_disk_witness_remains_valid_after_both_ram_generations_are_evicted(tmp_path):
    env = UnitEnv(tmp_path, pages=120, latest_only=True)
    try:
        old = env_point(env, 2)
        env.persist(old)
        handle = env.table.lookup(key(2, 1))
        env.table.pin(handle, 876)
        new = env_point(env, 4)
        env.persist(new)
        for unit in (old, new):
            b = env.manager.units.units[unit.parent]
            assert env.manager.units.revoke(b.uid, b.gen, "value")
        env.table.release(handle, 876)
        assert env.settle(lambda: old.uid not in env.store.manifests)
        assert new.uid in env.store.manifests
        assert env.manager.events["superseded_revoked"] == 0
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


def test_keep_ssd_overrides_legacy_pending_write_filter(tmp_path):
    env = UnitEnv(tmp_path, pages=120, latest_only=True, keep_ssd=True)
    try:
        old = env_point(env, 2)
        env_point(env, 4)
        env.manager.state_latest_only = True
        env.manager.drop_pending_secondary_writes(old.keys, req_id="producer")
        assert old.uid in env.store.due
        assert old.uid not in env.store.filtered_variants
        assert old.uid not in env.store.credit.excluded
        env.persist(old)
        assert env.settle(lambda: not env.manager.is_ready_resident(key(2, 1)))
        assert old.uid in env.store.manifests
    finally:
        env.ssd.close(io_grace_s=5)


def test_waiting_candidates_do_not_starve_other_chains_or_later_points():
    m = enable(manager(pages=80, workers=1))
    held = []
    for i in range(12):
        point(m, list(range(i + 1)))
        if i < 9:
            handle = m.table.lookup(key(i, 1))
            m.table.pin(handle, 456)
            held.append(handle)
    m.retention.step()
    m.retention.step()
    assert sha256(9) not in boundaries(m)
    assert sha256(10) not in boundaries(m)
    for handle in held:
        m.table.release(handle, 456)
    for _ in range(3):
        m.retention.step()
    assert boundaries(m) == {sha256(11)}


def test_restart_after_reset_rebuilds_retention_in_restored_epoch(tmp_path):
    writer = UnitEnv(tmp_path, pages=120, latest_only=True)
    writer.manager.reset_cache()
    old = env_point(writer, 2)
    writer.persist(old)
    assert old.uid.reset > 0
    writer.ssd.close(io_grace_s=5)
    reader = UnitEnv(tmp_path, pages=120, latest_only=True)
    try:
        reader.manager.retention.refresh()
        assert old.parent in reader.manager.retention.latest
        assert reader.manager.retention.epoch == old.uid.reset
        env_point(reader, 4)
        assert reader.settle(lambda: old.uid not in reader.store.manifests)
        assert_conserved(reader.manager)
    finally:
        reader.ssd.close(io_grace_s=5)
