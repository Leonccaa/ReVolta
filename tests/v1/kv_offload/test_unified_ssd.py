# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real buffered file IO through the unified host budgeting controller, with independent
MAIN ABI."""

import time
from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_host_page_ssd import ROW, Env, key, payload
from tests.v1.kv_offload.test_unified_units import manager, publish
from vllm.utils.hashing import sha256
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.host.page_table import PageKind
from vllm.v1.kv_offload.unified.units import ValueClass

pytestmark = pytest.mark.cpu_test


def attach(env):
    m = manager(pages=len(env.table.pages), workers=1)
    m.attached = False
    m.attach(SimpleNamespace(host_pages=env.table, elastic=False))
    m.attach_ssd(env.ssd)
    return m


def settle(env, predicate):
    assert env.settle(predicate)


def test_components_use_new_namespace_and_main_namespace_is_unchanged(tmp_path):
    env = Env(tmp_path)
    try:
        old_config = env.ssd._tier.file_mapper.get_run_config()
        old_name = env.ssd._tier.file_mapper.get_file_name(env.ssd._tier_key(key(0)))
        m = attach(env)
        assert old_config == env.ssd._tier.file_mapper.get_run_config()
        assert old_name == env.ssd._tier.file_mapper.get_file_name(
            env.ssd._tier_key(key(0))
        )
        component = key(1, 1)
        h = publish(m.table, component, PageKind.STATE)
        a = h.page_id * ROW
        env.backing[a : a + ROW] = payload(19)
        settle(env, lambda: m.table.is_persisted(m.table.get(h)))
        tier_key = env.ssd._tier_key(component)
        assert env.ssd._component_tier.lookup(tier_key)
        assert not env.ssd._tier.lookup(tier_key)
        m.table._recycle(m.table.get(h))
        assert m.lookup(component, ReqContext("reader")) is None
        settle(env, lambda: m.is_ready_resident(component))
        restored = m.table.lookup(component)
        assert m.table.get(restored).kind == PageKind.STATE
        assert env.backing[
            restored.page_id * ROW : (restored.page_id + 1) * ROW
        ] == payload(19)
    finally:
        env.ssd.close(io_grace_s=5)


def test_eviction_before_delay_writes_component_instead_of_discard(tmp_path):
    env = Env(tmp_path, pages=5, write_delay_s=300)
    try:
        m = attach(env)
        ks = [key(1, 1)]
        assert m.admit_build("r", 1)
        b = m.begin_boundary("r", sha256(1), ks)
        from tests.v1.kv_offload.test_unified_units import store

        store(m, ks)
        m.ledger.finish("r")
        h = m.table.lookup(ks[0])
        a = h.page_id * ROW
        env.backing[a : a + ROW] = payload(11)
        # Remaining candidates are physical-owner protected. Leave one spare
        # to pay IO, then ask for another allocation beyond it.
        others = [publish(m.table, key(i)) for i in range(2)]
        for other in others:
            m.table.pin(other, 900)
        spare = m.table.reserve()
        m.table.start_io(spare, 901)
        # try_writeback is the same pre-eviction hook as MAIN, not latest-only.
        assert env.ssd.try_writeback(m.table.get(h), 1)
        assert not m.table.reclaimable(m.table.get(h))
        settle(env, lambda: m.table.is_persisted(m.table.get(h)))
        assert env.ssd.stats_counter["writeback_submitted"] == 1
        assert m.units.files_complete(b.uid)
        m.table._recycle(m.table.get(h))
        assert m.units.recoverable(b.uid, "union")
    finally:
        env.ssd.close(io_grace_s=5)


def test_real_quota_loss_revokes_evidence_and_refreshes_cpu_value(tmp_path):
    env = Env(tmp_path, pages=8, max_files=1)
    try:
        m = attach(env)
        h = publish(m.table, key(1, 1), PageKind.STATE)
        settle(env, lambda: m.table.is_persisted(m.table.get(h)))
        b = m.units.begin(sha256(1), [key(1, 1)])
        m.units.finish(b.uid, b.gen)
        assert m.units.value(key(1, 1), PageKind.STATE) == ValueClass.SSD_REDUNDANT
        h2 = publish(m.table, key(2, 1), PageKind.STATE)
        settle(env, lambda: m.table.is_persisted(m.table.get(h2)))
        env.ssd.poll()
        assert not m.table.is_persisted(m.table.get(h))
        assert not m.units.files_complete(b.uid)
        assert m.units.value(key(1, 1), PageKind.STATE) == ValueClass.LEAF
        assert m.units.recoverable(b.uid, "cpu")
    finally:
        env.ssd.close(io_grace_s=5)


def test_shutdown_completes_unexpired_component_write(tmp_path):
    env = Env(tmp_path, write_delay_s=300)
    try:
        m = attach(env)
        h = publish(m.table, key(2, 2), PageKind.DRAFT)
        assert not m.table.is_persisted(m.table.get(h))
        report = env.ssd.final_flush(budget_s=3)
        assert report.completed == 1 and not report.stuck
        assert m.table.is_persisted(m.table.get(h))
        assert env.ssd._component_tier.lookup(env.ssd._tier_key(key(2, 2)))
    finally:
        env.ssd.close(io_grace_s=5)


def test_quota_deletion_before_store_receipt_cannot_create_false_evidence(tmp_path):
    env = Env(tmp_path, pages=8, max_files=1, write_threads=2, write_delay_s=300)
    try:
        m = attach(env)
        h1 = publish(m.table, key(1, 1), PageKind.STATE)
        h2 = publish(m.table, key(2, 1), PageKind.STATE)
        env.ssd._new_job(m.table.get(h1), False)
        env.ssd._pump()
        tier = env.ssd._component_tier
        deadline = time.monotonic() + 5
        while not tier._pool._finished_q and time.monotonic() < deadline:
            time.sleep(0.001)
        assert tier._pool._finished_q
        env.ssd._new_job(m.table.get(h2), False)
        env.ssd._pump()
        while len(tier._pool._finished_q) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(tier._pool._finished_q) == 2
        # Both writes succeeded, but the second file replaced the first in
        # the one-file root before either scheduler receipt was consumed.
        env.ssd.poll()
        assert not m.table.is_persisted(m.table.get(h1))
        assert m.table.is_persisted(m.table.get(h2))
        assert m.units.components[key(1, 1)].persisted is None
        assert not env.ssd._jobs
    finally:
        env.ssd.close(io_grace_s=5)


def test_connector_only_reset_keeps_main_and_hides_old_component_files(tmp_path):
    env = Env(tmp_path, write_delay_s=0)
    try:
        m = attach(env)
        main = publish(m.table, key(0))
        m.table.bind(main, 99, 0)
        component = publish(m.table, key(1, 1), PageKind.STATE)
        settle(env, lambda: m.table.is_persisted(m.table.get(component)))
        old_key = env.ssd._tier_key(key(1, 1))
        old_main_key = env.ssd._tier_key(key(0))
        m.reset_cache()
        assert m.table.lookup(key(0)) == main and m.table.is_ready(main)
        assert (99, 0) in m.table.get(main).binds
        assert m.table.lookup(key(1, 1)) is None
        assert env.ssd._tier_key(key(0)) == old_main_key
        assert env.ssd._tier_key(key(1, 1)) != old_key
        assert m.lookup(key(1, 1), ReqContext("reader")) is False
        assert not m.units.units and not m.waiting and not m.builds
    finally:
        env.ssd.close(io_grace_s=5)


def test_soft_demote_latest_only_and_eviction_writeback_are_independent(tmp_path):
    now = [0.0]
    env = Env(tmp_path, clock=lambda: now[0], write_delay_s=300)
    try:
        m = attach(env)
        k = key(1, 1)
        h = publish(m.table, k, PageKind.STATE)
        b = m.units.begin(sha256(1), [k])
        assert m.units.finish(b.uid, b.gen)
        assert env.ssd._queue.depth == 1
        m.demote([k], ReqContext("r"))
        assert m.units.value(k, PageKind.STATE) == ValueClass.INTERIOR
        assert env.ssd._queue.depth == 1
        m.drop_pending_secondary_writes([k])
        now[0] = 301
        env.ssd.step()
        assert env.ssd._queue.depth == 0
        assert env.ssd._queue.stats["dropped_superseded"] == 1
        assert not env.ssd._jobs
        assert m.units.recoverable(b.uid, "cpu")
        assert m.units.value(k, PageKind.STATE) == ValueClass.INTERIOR
        m.touch([k], ReqContext("r"))
        assert m.units.value(k, PageKind.STATE) == ValueClass.LEAF
        # Latest-only removed delayed eligibility, not the eviction backstop.
        assert env.ssd.try_writeback(m.table.get(h), m.table.net_headroom())
        settle(env, lambda: m.table.is_persisted(m.table.get(h)))
        assert m.units.files_complete(b.uid)
    finally:
        env.ssd.close(io_grace_s=5)


def test_delayed_file_loss_cannot_withdraw_same_path_new_epoch(tmp_path):
    from pathlib import Path

    env = Env(tmp_path, write_delay_s=300)
    try:
        m = attach(env)
        k = key(1, 1)
        old = publish(m.table, k, PageKind.STATE)
        assert env.ssd.try_writeback(m.table.get(old), m.table.net_headroom())
        settle(env, lambda: m.table.is_persisted(m.table.get(old)))
        b = m.units.begin(sha256(1), [k])
        assert m.units.finish(b.uid, b.gen)
        tier = env.ssd._component_tier
        path = tier.file_mapper.get_file_name(env.ssd._tier_key(k))
        m.table._recycle(m.table.get(old))
        Path(path).unlink()
        assert not tier.lookup(env.ssd._tier_key(k))  # Queue old-epoch loss.
        fresh = publish(m.table, k, PageKind.STATE)
        assert fresh != old
        job = env.ssd._new_job(m.table.get(fresh), False)
        env.ssd._pump()
        deadline = time.monotonic() + 5
        while not tier._pool._finished_q and time.monotonic() < deadline:
            time.sleep(0.001)
        result = list(tier.get_finished_jobs())
        assert len(result) == 1 and result[0].success
        # Model deletion racing after _poll's first loss drain, followed by a
        # queued same-path rewrite completing before the final loss drain.
        env.ssd._finish(job, True)
        assert env.ssd._file_evidence[path] == (k, fresh)
        env.ssd._poll_losses()
        assert m.table.is_persisted(m.table.get(fresh))
        assert m.units.components[k].persisted == fresh
        assert m.units.files_complete(b.uid)
        assert Path(path).is_file()
    finally:
        env.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("frames", [1, 2, 3])
def test_full_leaf_writeback_cannot_refuse_reserved_foreground_frame(tmp_path, frames):
    from vllm.v1.kv_offload.unified.units import VariantOffer

    env = Env(tmp_path, pages=5, write_delay_s=300)
    try:
        m = attach(env)
        states = [key(1, 1), key(1, 2)]
        for k in states:
            publish(m.table, k, m.kind(k))
        b = m.units.begin(sha256(1), states)
        assert m.units.finish(b.uid, b.gen)
        main = [key(0), key(1)]
        old_main = [publish(m.table, k) for k in main]
        v = m.units.register(VariantOffer(b.uid, b.gen, sha256(1), tuple(main)))
        assert v is not None and m.table.num_free() == 0
        # One physical unit spans every frame. Its writes may use surplus,
        # but must leave a path to the already-admitted foreground frame.
        held = m.ledger.reserve_frames(frames, "main", reserved=True)
        assert held is not None and len(held) == frames
        assert held[0].page_id == old_main[0].page_id
        assert v.uid not in m.units.units
        assert m.units.events["revoke_value"] >= 1
        assert env.ssd._jobs
        for job in env.ssd._jobs.values():
            assert job.handle not in held
            assert job.job_id in m.table.get(job.handle).io
        m.ledger.release_temporary(held)
        settle(env, lambda: not env.ssd._jobs)
        assert not m.table.diagnostics()["io"]
    finally:
        env.ssd.close(io_grace_s=5)


def test_two_cold_components_share_round_and_keep_holds(tmp_path):
    env = Env(tmp_path)
    try:
        m = attach(env)
        keys = [key(1, 1), key(1, 2)]
        handles = [publish(m.table, k, m.kind(k)) for k in keys]
        for n, h in enumerate(handles):
            env.backing[h.page_id * ROW : (h.page_id + 1) * ROW] = payload(n + 1)
        settle(env, lambda: all(m.table.is_persisted(m.table.get(h)) for h in handles))
        for h in handles:
            m.table._recycle(m.table.get(h))
        ctx = ReqContext("reader")
        for k in keys:
            assert m.lookup(k, ctx) is None
            settle(env, lambda k=k: m.is_ready_resident(k))
            assert m.lookup(k, ctx) is True
        assert env.ssd.stats_counter["prefetch_issued"] == 2
        assert len(env.ssd.holds("reader")) == 2
        assert m.hold_lookup("reader", keys)
        env.ssd.cancel("reader")
        assert all(m.table.get(m.table.lookup(k)).leases for k in keys)
        spec = m.prepare_load(keys, ctx)
        for n, h in enumerate(spec.handles):
            assert env.backing[h.page_id * ROW : (h.page_id + 1) * ROW] == payload(
                n + 1
            )
        m.complete_load(keys, ctx)
        assert not m._lookup_holds and not m._loads
    finally:
        env.ssd.close(io_grace_s=5)


def test_old_namespace_loss_does_not_unpersist_new_generation(tmp_path):
    from pathlib import Path

    env = Env(tmp_path, pages=8, write_delay_s=300)
    try:
        m = attach(env)
        k = key(1, 1)
        old = publish(m.table, k, PageKind.STATE)
        tier = env.ssd._component_tier
        old_path = tier.file_mapper.get_file_name(env.ssd._tier_key(k))
        env.ssd._new_job(m.table.get(old), False)
        env.ssd._pump()
        deadline = time.monotonic() + 5
        while not tier._pool._finished_q:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        m.reset_cache()
        env.ssd.poll()
        assert old_path not in env.ssd._file_evidence
        assert k not in m.units.components
        fresh = publish(m.table, k, PageKind.STATE)
        fresh_path = tier.file_mapper.get_file_name(env.ssd._tier_key(k))
        assert fresh_path != old_path
        env.ssd._new_job(m.table.get(fresh), False)
        env.ssd._pump()
        settle(env, lambda: m.table.is_persisted(m.table.get(fresh)))
        Path(old_path).unlink()
        assert not tier._cache.lookup(old_path)
        env.ssd.poll()
        assert Path(fresh_path).is_file()
        assert m.table.is_persisted(m.table.get(fresh))
        assert m.units.components[k].persisted == fresh
    finally:
        env.ssd.close(io_grace_s=5)


def test_extended_round_keeps_total_quota_after_verified_reads(tmp_path):
    env = Env(tmp_path, prefetch_max_per_request=2)
    try:
        m = attach(env)
        keys = [key(i, 1) for i in range(3)]
        handles = [publish(m.table, k, PageKind.STATE) for k in keys]
        settle(env, lambda: all(m.table.is_persisted(m.table.get(h)) for h in handles))
        for h in handles:
            m.table._recycle(m.table.get(h))
        ctx = ReqContext("r")
        for k in keys[:2]:
            assert m.lookup(k, ctx) is None
            settle(env, lambda k=k: m.is_ready_resident(k))
        assert not env.ssd.waiting("r")
        assert m.lookup(keys[2], ctx) is False
        assert env.ssd.stats_counter["prefetch_issued"] == 2
        assert len(env.ssd.holds("r")) == 2
        env.ssd.cancel("r")
    finally:
        env.ssd.close(io_grace_s=5)


def test_extended_round_keeps_original_deadline_until_episode_ends(tmp_path):
    now = [0.0]
    env = Env(tmp_path, clock=lambda: now[0], prefetch_timeout_s=5)
    try:
        m = attach(env)
        keys = [key(i, 1) for i in range(2)]
        handles = [publish(m.table, k, PageKind.STATE) for k in keys]
        settle(env, lambda: all(m.table.is_persisted(m.table.get(h)) for h in handles))
        for h in handles:
            m.table._recycle(m.table.get(h))
        ctx = ReqContext("r")
        assert m.lookup(keys[0], ctx) is None
        settle(env, lambda: m.is_ready_resident(keys[0]))
        now[0] = 6
        assert m.lookup(keys[1], ctx) is False
        assert not env.ssd.holds("r") and not env.ssd.waiting("r")
        assert env.ssd.stats_counter["prefetch_issued"] == 1
        env.ssd.cancel("r")
        assert m.lookup(keys[1], ctx) is None
        settle(env, lambda: m.is_ready_resident(keys[1]))
    finally:
        env.ssd.close(io_grace_s=5)
