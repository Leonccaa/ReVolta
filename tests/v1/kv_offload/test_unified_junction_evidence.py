# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evidence retention through natural captures, registered graphs and real SSD IO."""

import random
import threading
from pathlib import Path

import pytest

from tests.v1.kv_offload.junction_utils import (
    SPAN,
    advance,
    build,
    decode_step,
    evict_gpu_prefix_only,
    host_settings,
    make_requests,
    new_request,
)
from tests.v1.kv_offload.test_host_page_ssd import ROW, key
from tests.v1.kv_offload.test_unified_latest import enable, env_point, point
from tests.v1.kv_offload.test_unified_ssd_transactions import UnitEnv
from tests.v1.kv_offload.test_unified_units import assert_conserved, manager, publish
from vllm.v1.kv_offload.host.page_table import PageKind
from vllm.v1.kv_offload.unified.units import VariantOffer
from vllm.v1.request import RequestStatus


def evidenced(m):
    return enable(m, evidence=True)


def confirm(m, unit, reason="short_tail_junction"):
    data = dict(unit.retention, evidence=m.retention.evidence(unit, reason))
    b = m.units.units[unit.parent]
    main = tuple(k for k in unit.keys if m.table.kind_for_key(k) == PageKind.MAIN)
    offer = VariantOffer(
        b.uid,
        b.gen,
        unit.uid.ahead,
        main,
        unit.boundary_position,
        unit.ahead_position,
        unit.draft_range,
        data,
    )
    assert m.units.register(offer) is unit
    m.retention.refresh()


@pytest.mark.parametrize("speculative_tokens", [0, 1])
@pytest.mark.parametrize("tool_pages", [0, 2])
def test_c1_natural_six_turn_decode_append(speculative_tokens, tool_pages, monkeypatch):
    monkeypatch.setattr(
        "tests.v1.core.test_long_context_admission.MAX_MODEL_LEN", 64 * SPAN
    )
    scheduler = build(
        host_settings(host_pool_tokens=256 * SPAN),
        max_num_seqs=2,
        with_connector=True,
        speculative_tokens=speculative_tokens,
    )
    connector = scheduler.connector.connector_scheduler
    m = connector.unified
    assert m.enable_junction_evidence
    request = new_request("turn-0", [11] * (2 * SPAN + 17), SPAN + 64)
    captures, live = [], []
    for turn in range(7):
        scheduler.add_request(request)
        for _ in range(4 * SPAN + 200):
            decode_step(scheduler)
            if request.is_finished():
                break
        assert request.is_finished()
        for _ in range(12):
            connector.persistence_step()
        r = m.retention
        r.refresh()
        assert not r.branches
        assert len(r.latest) == 1
        if turn:
            assert request.shared_prefix_boundary > 0
            captures.append(request.shared_prefix_boundary)
            assert request.shared_prefix_boundary < request.num_prompt_tokens
            if tool_pages:
                assert (
                    request.shared_prefix_boundary
                    < request.num_prompt_tokens // SPAN * SPAN
                )
        live.append(sum(u.parent is None for u in m.units.units.values()))
        # Latest + exact resend + current captures waiting on physical retirement.
        assert live[-1] <= 4
        evict_gpu_prefix_only(scheduler)
        request = new_request(
            f"turn-{turn + 1}",
            [*request.all_token_ids, *([19] * (37 + tool_pages * SPAN))],
            (1 + turn % 2) * SPAN + 64,
        )
    assert captures == sorted(set(captures))
    assert m.events["superseded_revoked"] > 0
    assert m.events["superseded_freed_bytes"] > 0


@pytest.mark.parametrize("output", [1, 32])
def test_c2_no_page_crossing_and_hot_exact_resend(output):
    scheduler = build(host_settings(host_pool_tokens=64 * SPAN), with_connector=True)
    request = new_request("hot-parent", [11] * (2 * SPAN + 17), output)
    scheduler.add_request(request)
    for _ in range(SPAN):
        decode_step(scheduler)
        if request.is_finished():
            break
    assert request.is_finished()
    m = scheduler.connector.connector_scheduler.unified
    for n in range(3):
        retry = new_request(f"retry-{n}", request.prompt_token_ids, 1)
        scheduler.add_request(retry)
        for _ in range(100):
            decode_step(scheduler)
            if retry.is_finished():
                break
        assert retry.is_finished()
        assert not m.retention.snapshot()["branch_points"]


def test_c2_small_linear_tail_identities_never_confirm():
    m = evidenced(manager(pages=100, workers=1))
    u = point(m, [0, 1], request_length=33, junction=True)
    original = dict(u.retention)
    for length in range(34, 48):
        again = point(m, [0, 1], request_length=length, junction=True)
        assert again is u
        assert u.retention == original
    point(m, [0, 1, 2], request_length=49)
    m.retention.step()
    assert not m.retention.branches and u.parent not in m.units.units


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 0, 2)])
def test_c3_fork_order_and_own_child(order):
    m = evidenced(manager(pages=100, workers=1))
    paths = [[0, 1], [0, 1, 2, 3], [0, 1, 4, 5]]
    units = {}
    for index in order:
        units[index] = point(m, paths[index], resend=1 if index == 2 else 0)
    r = m.retention
    assert r.successor_reason(list(map(key, paths[1])), 2, key(2)) == "other_successor"
    r.refresh()
    assert units[0].parent in r.branches
    assert {units[1].parent, units[2].parent} <= r.latest
    assert units[0].parent in r.resends
    assert units[0].retention["evidence"]["reason"] == "distinct_successors"
    only = evidenced(manager(pages=80, workers=1))
    point(only, paths[1])
    assert only.retention.successor_reason(list(map(key, paths[1])), 2, key(2)) is None
    assert not only.retention.branches


@pytest.mark.parametrize("speculative_tokens", [0, 1])
@pytest.mark.parametrize("suffix", [832, 1664, 3328])
def test_c4_natural_short_and_full_forks(speculative_tokens, suffix):
    scheduler = build(
        host_settings(host_pool_tokens=128 * SPAN),
        max_num_seqs=2,
        with_connector=True,
        speculative_tokens=speculative_tokens,
    )
    (parent,) = make_requests(1, 8 * SPAN, 16, tag="fork-parent")
    scheduler.add_request(parent)
    for _ in range(12):
        advance(scheduler)
    scheduler.finish_requests([parent.request_id], RequestStatus.FINISHED_ABORTED)
    connector = scheduler.connector.connector_scheduler
    connector.persistence_step()
    common = 4 * SPAN
    child = new_request(
        "fork-child", [*parent.prompt_token_ids[:common], *([2000] * suffix)]
    )
    scheduler.add_request(child)
    for _ in range(12):
        advance(scheduler)
    assert child.shared_prefix_boundary == common
    r = connector.unified.retention
    r.refresh()
    captured = [u for u in r.variants.values() if u.boundary_position == common]
    assert captured and captured[0].parent in r.branches
    reason = captured[0].retention["evidence"]["reason"]
    assert reason == ("short_tail_junction" if suffix < SPAN else "other_successor")


def test_c4_partial_hash_last_token_is_only_identity_not_a_branch():
    scheduler = build(host_settings(host_pool_tokens=64 * SPAN), with_connector=True)
    c = scheduler.connector.connector_scheduler
    a = new_request("partial-a", [11] * (2 * SPAN + 7))
    b = new_request("partial-b", [*a.prompt_token_ids[:-1], 12])
    c.on_new_request(a)
    c.on_new_request(b)
    assert c._req_status[a.request_id].prompt_identity(16) != c._req_status[
        b.request_id
    ].prompt_identity(16)
    from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request

    assert chain_hash_from_request(a.block_hashes, 16)(
        c.host_kv_hits.rule, 1
    ) == chain_hash_from_request(b.block_hashes, 16)(c.host_kv_hits.rule, 1)
    assert not c.unified.retention.branches


def test_c5_unindexed_tail_can_be_superseded_then_recaptured():
    m = evidenced(manager(pages=100, workers=1))
    old = point(m, [0, 1], junction=True)
    for value in (2, 3):
        publish(m.table, key(value), PageKind.MAIN)
    assert m.retention.successor_reason([key(0), key(1)], 2, None) is None
    point(m, [0, 1, 2, 3])
    m.retention.step()
    assert old.parent not in m.units.units
    assert all(m.is_ready_resident(key(v)) for v in range(4))
    new = point(m, [0, 1], junction=True)
    assert new.gen != old.gen
    assert "evidence" not in new.retention
    confirm(m, new)
    assert new.parent in m.retention.branches
    assert_conserved(m)


@pytest.mark.parametrize(
    "legacy", [{"junction": True}, {"branch": True}, {"branch": True, "junction": True}]
)
@pytest.mark.parametrize("seed", [1, 7])
def test_c6_legacy_flags_do_not_retain_or_rewrite(tmp_path, legacy, seed):
    writer = UnitEnv(tmp_path, pages=160, latest_only=True)
    old = env_point(writer, 2, request_length=16)
    old.retention.update(legacy)
    writer.manager.units.on_event("observe", old)
    tip = env_point(writer, 5, request_length=16)
    writer.manager.retention.refresh()
    writer.persist(old)
    writer.persist(tip)
    documents = {
        u: Path(p).read_bytes() for u, p in writer.store.manifest_paths.items()
    }
    writer.ssd.close(io_grace_s=5)
    # Exercise the effective full-graph classifier with shuffled observations.
    # Full rebuild already defers collection until every valid record loaded.
    reader = UnitEnv(tmp_path, pages=160, latest_only=True, evidence=True)
    try:
        r = reader.manager.retention
        variants = list(r.variants.values())
        random.Random(seed).shuffle(variants)
        r.reset()
        for unit in variants:
            r.observe(unit)
        r.refresh()
        assert not r.branches and old.parent in r.pending
        for uid, p in reader.store.manifest_paths.items():
            assert Path(p).read_bytes() == documents[uid]
        assert not reader.store.retention_updates_pending()
        assert reader.settle(lambda: old.uid not in reader.store.manifests)
        assert reader.manager.events["superseded_revoked"] > 0
        assert all(
            reader.store.path(key(v)) and Path(reader.store.path(key(v))).exists()
            for v in range(6)
        )
    finally:
        reader.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("legacy_branch", [False, True])
def test_c6_new_reason_survives_child_eviction_and_two_restarts(
    tmp_path, legacy_branch
):
    writer = UnitEnv(tmp_path, pages=160, latest_only=True, evidence=True)
    u = env_point(writer, 2, request_length=16)
    child = env_point(writer, 5, request_length=16)
    u.retention["branch"] = legacy_branch
    confirm(writer.manager, u)
    writer.persist(u)
    assert u.retention["evidence"]["version"] == 1
    b = writer.manager.units.units[child.parent]
    assert writer.manager.units.revoke(b.uid, b.gen, "value")
    writer.manager.retention.refresh()
    assert u.parent in writer.manager.retention.branches
    writer.ssd.close(io_grace_s=5)
    for _ in range(2):
        reader = UnitEnv(tmp_path, pages=160, latest_only=True, evidence=True)
        try:
            r = reader.manager.retention
            r.refresh()
            assert u.parent in r.branches
            assert r.valid_evidence(
                r.variants[u.uid], r.variants[u.uid].retention["evidence"]
            )
        finally:
            reader.ssd.close(io_grace_s=5)


def test_c6_invalid_binding_and_unique_legacy_checkpoint():
    m = evidenced(manager(pages=80, workers=1))
    u = point(m, [0, 1], junction=True)
    u.retention["branch"] = True
    u.retention["evidence"] = dict(
        m.retention.evidence(u, "other_successor"), content="invalid"
    )
    m.units.on_event("observe", u)
    m.retention.step()
    assert not m.retention.branches and u.parent in m.units.units


def test_c7_no_watch_reobserve_reset_and_new_generation():
    m = evidenced(manager(pages=80, workers=1))
    u = point(m, [0, 1], junction=True)
    for _ in range(20):
        m.units.on_event("observe", u)
        m.retention.step()
    assert not hasattr(m.retention, "watch")
    assert not m.retention.reasons
    confirm(m, u)
    m.units.revoke(u.parent, m.units.units[u.parent].gen, "value")
    new = point(m, [0, 1])
    m.retention.refresh()
    assert new.parent not in m.retention.branches
    assert not m.retention.reasons
    m.retention.reset()
    assert not m.retention.pending and not m.retention.branches


@pytest.mark.parametrize("after_unlink", [False, True])
def test_c8_structural_evidence_retracts_inflight_delete(
    tmp_path, monkeypatch, after_unlink
):
    env = UnitEnv(tmp_path, pages=160, latest_only=True, evidence=True)
    started, gate = threading.Event(), threading.Event()
    original = env.store._delete_manifests

    def gated(txn):
        if after_unlink:
            original(txn)
        started.set()
        assert gate.wait(20)
        if not after_unlink:
            original(txn)

    monkeypatch.setattr(env.store, "_delete_manifests", gated)
    try:
        old = env_point(env, 2, request_length=16)
        env.persist(old)
        env_point(env, 5, request_length=16)
        env.manager.retention.step()
        assert started.wait(5)
        # The exact old UID is blocked once delete has started. A successfully
        # registered sibling can still supply structural evidence.
        sibling = env.variant(9, prefix=[key(0), key(1), key(2), key(7), key(8)])
        sibling.retention.update(request_length=16, resend=None)
        env.manager.units.on_event("observe", sibling)
        env.manager.retention.refresh()
        assert old.parent not in env.manager.retention.pending
        gate.set()
        assert env.settle(lambda: not env.store.manifest_jobs)
        assert old.parent in env.manager.units.units
        assert env.settle(lambda: env.store.manifest_ok(old))
        assert (
            env.store.manifests[old.uid]["retention"]["evidence"]["reason"]
            == "distinct_successors"
        )
        assert env.manager.events["superseded_revoked"] == 0
        assert not env.manager.units.registration_blocked
    finally:
        gate.set()
        env.ssd.close(io_grace_s=5)


def test_c8_disk_only_new_structural_reason_is_published(tmp_path):
    env = UnitEnv(tmp_path, pages=180, latest_only=True, evidence=True)
    try:
        old = env_point(env, 2, request_length=16)
        env.persist(old)
        b = env.manager.units.units[old.parent]
        env.manager.units.revoke(b.uid, b.gen, "value")
        env_point(env, 5, request_length=16)
        sibling = env.variant(9, prefix=[key(0), key(1), key(2), key(7), key(8)])
        sibling.retention.update(request_length=16, resend=None)
        env.manager.units.on_event("observe", sibling)
        env.manager.retention.refresh()
        assert old.parent in env.manager.retention.branches
        assert old.uid in env.manager.units.units
        assert env.settle(
            lambda: env.store.manifests[old.uid]["retention"].get("evidence")
            is not None
        )
    finally:
        env.ssd.close(io_grace_s=5)


def test_c8_short_tail_upgrade_with_lease_before_delete(tmp_path):
    env = UnitEnv(tmp_path, pages=160, latest_only=True, evidence=True)
    try:
        old = env_point(env, 2, request_length=16)
        env.persist(old)
        assert env.manager.hold_lookup("reader", [key(2, 1)])
        env_point(env, 5, request_length=16)
        env.manager.retention.step()
        assert old.parent in env.manager.retention.pending
        confirm(env.manager, old)
        env.manager.release_lookup("reader")
        assert env.settle(
            lambda: not env.store.manifest_jobs
            and env.store.manifests[old.uid]["retention"].get("evidence") is not None
        )
        assert old.parent in env.manager.retention.branches
        assert env.manager.events["superseded_revoked"] == 0
    finally:
        env.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("keep", [False, True])
def test_c10_delayed_write_idle_and_exact_frame_accounting(tmp_path, keep):
    clock = [0.0]
    env = UnitEnv(
        tmp_path,
        pages=160,
        clock=lambda: clock[0],
        latest_only=True,
        keep_ssd=keep,
        evidence=True,
    )
    try:
        env.manager.spec.unified_frame_bytes = ROW
        old = env_point(env, 2, request_length=16)
        newer = env_point(env, 5, request_length=16)
        r = env.manager.retention
        r.step()
        if keep:
            assert old.parent in env.manager.units.units
            assert old.uid in env.store.due
            clock[0] = 301.0
            assert env.settle(
                lambda: env.store.manifest_ok(old) and not env.store.writes
            )
            r.step()
            assert env.manager.units.cold_recoverable(old.uid)
        else:
            assert old.parent not in env.manager.units.units
            assert old.uid not in env.store.due
            clock[0] = 301.0
        assert env.settle(lambda: not env.store.writes and not env.store.manifest_jobs)
        freed = env.manager.events["superseded_freed_bytes"]
        deleted = env.manager.events["superseded_ssd_deleted_bytes"]
        assert freed > 0
        for _ in range(20):
            env.ssd.step()
        assert env.manager.events["superseded_freed_bytes"] == freed
        assert env.manager.events["superseded_ssd_deleted_bytes"] == deleted
        assert all(env.manager.is_ready_resident(key(i)) for i in range(6))
        assert env.manager.units.recoverable(newer.uid, "union")
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("latest", [False, True])
@pytest.mark.parametrize("evidence", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_c9_two_layer_switch_matrix(latest, evidence, keep):
    scheduler = build(
        host_settings(
            host_pool_tokens=64 * SPAN,
            latest_only=latest,
            enable_junction_evidence=evidence,
            superseded_keep_ssd=False,
        ),
        with_connector=True,
    )
    m = scheduler.connector.connector_scheduler.unified
    assert m.latest_only is latest
    assert m.enable_junction_evidence is (latest and evidence)
    assert (m.retention is not None) is latest
    if latest:
        assert m.retention.evidence_enabled is evidence
    # keep_ssd's real-file behavior is exercised by C10, including delayed IO.
    # It does not alter the evidence classifier.
    m.superseded_keep_ssd = keep
    if latest:
        m.retention.keep_ssd = keep
        assert not m.retention.branches
        assert not m.retention.reasons
