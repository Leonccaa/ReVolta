# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit-sized transactions over real files and the ordinary page IO queue."""

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_host_page_ssd import (
    LAYOUT,
    ROW,
    Env,
    key,
    options,
    payload,
)
from tests.v1.kv_offload.test_unified_units import assert_conserved, manager, publish
from vllm.utils.hashing import sha256
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.host.page_table import HostPageTable, PageKind
from vllm.v1.kv_offload.host.persist import HostPageSsd
from vllm.v1.kv_offload.unified.units import VariantOffer

pytestmark = pytest.mark.cpu_test


class UnitEnv(Env):
    def __init__(
        self,
        root,
        pages=40,
        clock=None,
        latest_only=False,
        keep_ssd=False,
        evidence=False,
        **overrides,
    ):
        self.table = HostPageTable(pages, num_workers=1)
        self.backing = bytearray(pages * ROW)
        self.manager = manager(pages=pages, workers=1)
        self.manager.attached = False
        self.manager.latest_only = latest_only
        self.manager.enable_junction_evidence = latest_only and evidence
        self.manager.superseded_keep_ssd = keep_ssd
        self.manager.attach(SimpleNamespace(host_pages=self.table, elastic=False))
        self.manager.kind_by_group = {g: PageKind.STATE for g in range(1, 8)} | {
            8: PageKind.DRAFT
        }
        view = memoryview(self.backing).cast("B", shape=[pages, ROW])
        spec, opts = options(root, write_delay_s=300, **overrides)
        self.clock = clock or time.monotonic
        self.ssd = HostPageSsd(self.table, view, spec, LAYOUT, opts, clock=self.clock)
        self.manager.attach_ssd(self.ssd)
        self.store = self.ssd.unit_store

    def variant(self, boundary=5, ahead=None, prefix=None):
        ahead = boundary if ahead is None else ahead
        main = list(prefix or [key(i) for i in range(boundary)]) + [key(boundary)]
        if ahead != boundary:
            main.append(key(ahead))
        states = [key(boundary, g) for g in range(1, 8)]
        drafts = [key(boundary - 1, 8), key(boundary, 8)]
        keys = states + drafts + main
        for n, k in enumerate(keys):
            if self.table.lookup(k) is None:
                h = publish(self.table, k, self.table.kind_for_key(k))
                self.backing[h.page_id * ROW : (h.page_id + 1) * ROW] = payload(n)
        b = self.manager.units.begin(sha256(boundary), states + drafts)
        assert self.manager.units.finish(b.uid, b.gen)
        unit = self.manager.units.register(
            VariantOffer(
                b.uid,
                b.gen,
                sha256(ahead),
                tuple(main),
                (len(main) - int(ahead != boundary)) * 16,
                len(main) * 16,
                (
                    (len(main) - int(ahead != boundary)) * 16 - 32,
                    (len(main) - int(ahead != boundary)) * 16,
                ),
            )
        )
        assert unit is not None
        return unit

    def persist(self, unit):
        assert self.store.start_write(unit)
        assert self.settle(lambda: self.store.manifest_ok(unit))
        assert not self.store.writes
        assert_conserved(self.manager)

    def evict_cpu(self, unit):
        for k in unit.keys:
            h = self.table.lookup(k)
            if h is not None:
                self.table._recycle(self.table.get(h))


@pytest.fixture
def env(tmp_path):
    value = UnitEnv(tmp_path)
    yield value
    value.ssd.close(io_grace_s=5)


def test_whole_inventory_protected_and_batched_write_and_prefetch(tmp_path):
    env = UnitEnv(
        tmp_path,
        max_pending_blocks=2,
        prefetch_max_per_request=1,
        prefetch_max_inflight=2,
    )
    try:
        unit = env.variant(ahead=6)
        expected = {
            k: bytes(
                env.backing[
                    env.table.lookup(k).page_id * ROW : (
                        env.table.lookup(k).page_id + 1
                    )
                    * ROW
                ]
            )
            for k in unit.keys
        }
        assert len(unit.keys) > env.ssd._max_pending
        assert env.store.start_write(unit)
        txn = env.store.writes[unit.uid]
        assert len(txn.handles) == len(unit.keys)
        assert all(txn.owner in env.table.get(h).leases for h in txn.handles.values())
        assert len(env.ssd._jobs) <= 2
        assert not env.manager.units.cold_recoverable(unit.uid)
        assert env.settle(lambda: env.store.manifest_ok(unit))
        assert env.store.credit.bytes == 0
        assert env.store.cache._reserved == 0
        record = env.store.manifests[unit.uid]
        assert record["B"] == 96 and record["A"] == 112
        assert record["main"][-1] == key(6).hex()
        assert len(record["state"]) == 7 and len(record["draft"]) == 2
        env.evict_cpu(unit)
        assert env.ssd.prefetch("a", [unit.keys[0]])
        assert len(env.store.prefetches) == 1
        txn = env.store.prefetches[unit.uid]
        assert len(txn.handles) == len(unit.keys)
        assert len(txn.active) <= 1
        assert env.ssd.prefetch("b", [unit.keys[0]])
        assert txn.waiters == {"a", "b"}
        env.ssd.cancel("a")
        assert txn.waiters == {"b"}
        assert env.settle(lambda: txn.complete)
        assert all(env.manager.is_ready_resident(k) for k in unit.keys)
        for k, h in txn.handles.items():
            assert env.backing[h.page_id * ROW : (h.page_id + 1) * ROW] == expected[k]
        env.ssd.cancel("b")
        assert not env.store.prefetches and not env.store.cache._pins
        assert env.manager.units.cold_recoverable(unit.uid)
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


def test_true_new_process_rebuild_and_verified_hit(tmp_path):
    env = UnitEnv(tmp_path)
    unit = env.variant()
    env.persist(unit)
    uid = unit.uid
    env.ssd.close(io_grace_s=5)
    script = """
import sys
from pathlib import Path
from tests.v1.kv_offload.test_unified_ssd_transactions import UnitEnv
from vllm.v1.kv_offload.base import ReqContext
root = Path(sys.argv[1])
e = UnitEnv(root)
assert len(e.store.manifests) == 1
u = next(u for u in e.manager.units.units.values() if u.parent is not None)
assert e.manager.units.cold_recoverable(u.uid)
assert not e.manager.is_ready_resident(u.keys[0])
assert e.manager.lookup(u.keys[0], ReqContext("restart")) is None
assert e.settle(lambda: not e.ssd.waiting("restart"))
assert all(e.manager.lookup(k, ReqContext("restart")) is True for k in u.keys)
assert e.store.events["prefetch_verified_bytes"] == len(u.keys) * e.ssd.page_bytes
e.ssd.cancel("restart")
e.ssd.close(io_grace_s=5)
print("REBUILT_VERIFIED_HIT")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REBUILT_VERIFIED_HIT" in result.stdout
    assert uid.reset == 0


def test_manifest_only_drop_updates_value_and_redirty_with_shared_files(env):
    first = env.variant(ahead=6)
    second = env.variant(ahead=7)
    env.persist(first)
    env.persist(second)
    # All components of this branch stay pinned, so F is empty.
    paths = {k: env.store.path(k) for k in first.keys}
    pins = [env.store.cache.read(p) for p in paths.values()]
    for pin in pins:
        pin.__enter__()
    try:
        revision = env.table.get(env.table.lookup(first.keys[0])).revision
        assert env.store.drop(first.uid)
        assert not env.store.manifest_ok(first)
        assert env.manager.units.recoverable(first.uid, "union")
        assert env.table.get(env.table.lookup(first.keys[0])).revision > revision
        assert all(Path(p).exists() for p in paths.values())
        assert env.store.manifest_ok(second)
        assert env.store.credit.bytes <= env.store.credit.d_dyn
    finally:
        for pin in reversed(pins):
            pin.__exit__(None, None, None)


def test_shared_component_last_reference_reclaimed(env):
    first = env.variant(ahead=6)
    second = env.variant(ahead=7)
    env.persist(first)
    env.persist(second)
    shared = set(first.keys) & set(second.keys)
    paths = {k: env.store.path(k) for k in shared}
    env.evict_cpu(first)
    env.evict_cpu(second)
    assert env.store.drop(first.uid)
    assert all(Path(p).exists() for p in paths.values())
    assert env.store.drop(second.uid)
    assert all(not Path(p).exists() for p in paths.values())
    assert not env.store.manifests
    assert env.store.cache._reserved == 0


def test_credit_deduplicates_shrinks_and_best_effort_still_cached(env):
    first = env.variant(ahead=6)
    second = env.variant(ahead=7)
    credit = env.store.credit
    a, b = env.store.obligations(first), env.store.obligations(second)
    assert credit.bytes == sum((a | b).values())
    assert credit.bytes < sum(a.values()) + sum(b.values())
    credit.capacity_bytes = credit.io_bytes + sum(a.values())
    credit.reconcile(promote=False)
    assert credit.bytes <= credit.d_dyn
    assert len(credit.guaranteed) == 1
    third = env.variant(boundary=8)
    assert third.uid not in credit.guaranteed
    assert env.manager.units.recoverable(third.uid, "cpu")
    assert env.store.snapshot()["best_effort_variants"] >= 1
    credit.rate = credit.floor_rate = 1
    credit.capacity_bytes = 1
    credit.reconcile(promote=False)
    assert credit.bytes == credit.d_dyn == 0
    assert env.manager.units.recoverable(first.uid, "cpu")
    assert credit.events["downgraded_shrink"] >= 1


def test_shutdown_guarantees_before_best_effort_and_component_before_manifest(
    env, monkeypatch
):
    first = env.variant(ahead=6)
    second = env.variant(ahead=7)
    env.store.credit.pressure(second.uid)
    order = []
    original = env.store._finish_write

    def finished(txn, success):
        if success:
            assert all(Path(p).exists() for p in txn.paths.values())
            assert Path(txn.manifest_path).exists()
            order.append(txn.uid)
        return original(txn, success)

    monkeypatch.setattr(env.store, "_finish_write", finished)
    report = env.ssd.final_flush(budget_s=5)
    assert order == [first.uid, second.uid]
    assert report.guaranteed_completed_bytes > 0
    assert report.best_effort_completed_bytes > 0
    assert report.unfinished_bytes == 0
    assert not report.stuck and not report.deadline_hit


def test_two_writers_share_uncommitted_files_and_abort_reclaims_only_last_ref(env):
    first = env.variant(ahead=6)
    second = env.variant(ahead=7)
    assert env.store.start_write(first)
    assert env.store.start_write(second)
    assert len(env.store.writes) == 2
    env.manager.units.revoke(first.uid, first.gen, "value")
    assert env.settle(lambda: env.store.manifest_ok(second))
    assert not env.store.writes
    assert all(Path(env.store.path(k)).exists() for k in second.keys)
    assert not env.store.cache._claims and not env.store.cache._pins
    assert env.store.events["write_aborted"] == 1
    assert env.manager.units.cold_recoverable(second.uid)
    assert_conserved(env.manager)


def test_quota_reservation_cannot_split_unit_during_commit(env):
    unit = env.variant()
    assert env.store.start_write(unit)
    txn = env.store.writes[unit.uid]
    paths = dict(txn.reservation)
    assert not env.store.reserve(
        {
            str(
                Path(env.store.directory) / "oversize.bin"
            ): env.store.cache.max_size_bytes
        }
    )
    assert unit.uid in env.store.writes
    assert set(paths) <= env.store.cache._claims.keys()
    assert all(txn.owner in env.table.get(h).leases for h in txn.handles.values())
    assert env.settle(lambda: env.store.manifest_ok(unit))
    assert all(Path(p).exists() for p in paths)


def test_startup_stats_do_not_evict_manifests_or_old_main_by_mtime(tmp_path):
    env = UnitEnv(tmp_path)
    unit = env.variant()
    env.persist(unit)
    main_paths = [
        env.store.path(k)
        for k in unit.keys
        if env.table.kind_for_key(k) == PageKind.MAIN
    ]
    orphan = tmp_path / "abandoned.tmp"
    orphan.write_bytes(b"x" * 123)
    expected_bytes = env.store.cache._bytes + 123
    env.ssd.close(io_grace_s=5)
    restarted = UnitEnv(tmp_path, max_files=2)
    try:
        assert len(restarted.store.manifests) == 1
        assert restarted.store.cache._bytes == expected_bytes
        assert all(Path(p).exists() for p in main_paths)
        assert orphan.exists()
    finally:
        restarted.ssd.close(io_grace_s=5)


def test_uncommitted_components_are_not_a_cold_hit(tmp_path):
    env = UnitEnv(tmp_path)
    k = key(1, 1)
    h = publish(env.table, k, PageKind.STATE)
    assert env.ssd.try_writeback(env.table.get(h), env.table.net_headroom())
    assert env.settle(lambda: env.table.is_persisted(env.table.get(h)))
    env.ssd.close(io_grace_s=5)
    restarted = UnitEnv(tmp_path)
    try:
        assert not restarted.store.manifests
        assert restarted.manager.lookup(k, ReqContext("cold")) is False
        assert not restarted.ssd._jobs
    finally:
        restarted.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("point", [*range(1, 16), "before_manifest", "after_manifest"])
def test_restart_at_each_component_and_manifest_commit(tmp_path, point):
    script = """
import os, sys
from pathlib import Path
from tests.v1.kv_offload.test_unified_ssd_transactions import UnitEnv
from vllm.v1.kv_offload.unified import ssd as module
root, point = Path(sys.argv[1]), sys.argv[2]
e = UnitEnv(root, max_pending_blocks=1)
u = e.variant()
original_finish = e.ssd._finish
finished = [0]
def finish(job, success):
    original_finish(job, success)
    if success and not job.loading:
        finished[0] += 1
        if point == str(finished[0]):
            os._exit(0)
e.ssd._finish = finish
original_json = module.atomic_json
def atomic(path, value, *args):
    if path.endswith(".unit.json") and point == "before_manifest":
        os._exit(0)
    result = original_json(path, value, *args)
    if path.endswith(".unit.json") and point == "after_manifest":
        os._exit(0)
    return result
module.atomic_json = atomic
assert e.store.start_write(u)
assert e.settle(lambda: e.store.manifest_ok(u))
raise AssertionError("Commit point did not execute")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(point)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    env = UnitEnv(tmp_path, max_pending_blocks=1)
    try:
        assert len(env.store.manifests) == int(point == "after_manifest")
        assert env.store.cache._bytes == sum(
            p.stat().st_size
            for p in tmp_path.rglob("*")
            if p.is_file()
            and (
                p.suffix == ".bin"
                or p.name.endswith(".unit.json")
                or p.suffix == ".tmp"
            )
        )
        if point == "after_manifest":
            unit = next(
                u for u in env.manager.units.units.values() if u.parent is not None
            )
            assert env.ssd.prefetch("restart", unit.keys)
            assert env.settle(lambda: not env.ssd.waiting("restart"))
            assert all(
                env.manager.lookup(k, ReqContext("restart")) is True for k in unit.keys
            )
            env.ssd.cancel("restart")
        else:
            assert env.manager.lookup(key(5, 1), ReqContext("restart")) is False
    finally:
        env.ssd.close(io_grace_s=5)


def test_legacy_main_file_is_read_by_unified_reader(tmp_path):
    legacy = Env(tmp_path)
    h = legacy.publish(17)
    assert legacy.settle(lambda: legacy.table.is_persisted(legacy.table.get(h)))
    path = legacy.ssd._tier.file_mapper.get_file_name(legacy.ssd._tier_key(key(17)))
    identity = Path(path).read_bytes()
    legacy.ssd.close(io_grace_s=5)
    unified = UnitEnv(tmp_path)
    try:
        assert not unified.store.manifests
        assert unified.ssd.prefetch("legacy", [key(17)]) == [key(17)]
        assert unified.settle(lambda: not unified.ssd.waiting("legacy"))
        restored = unified.table.lookup(key(17))
        assert unified.table.is_ready(restored)
        assert unified.backing[
            restored.page_id * ROW : (restored.page_id + 1) * ROW
        ] == payload(17)
        assert Path(path).read_bytes() == identity
        unified.ssd.cancel("legacy")
    finally:
        unified.ssd.close(io_grace_s=5)


def test_redirty_manifest_only_obligation_downgrades_synchronously(env):
    first = env.variant(ahead=6)
    env.persist(first)
    assert first.uid not in env.store.credit.guaranteed
    second = env.variant(ahead=7)
    ahead = env.table.lookup(key(7))
    env.ssd._new_job(env.table.get(ahead), False)
    env.ssd._pump()
    assert env.settle(lambda: env.table.is_persisted(env.table.get(ahead)))
    credit = env.store.credit
    assert second.uid in credit.guaranteed
    assert credit.bytes == env.store.manifest_bytes(second)
    credit.capacity_bytes = credit.io_bytes + credit.bytes
    credit.reconcile(promote=False)
    assert env.store.drop(first.uid)
    assert second.uid not in credit.guaranteed
    assert credit.events["downgraded_redirty"] == 1
    assert credit.bytes == 0
    assert env.manager.units.recoverable(second.uid, "cpu")


def test_credit_transfer_promote_and_faster_production_remain_bounded(tmp_path):
    env = UnitEnv(tmp_path, pages=160)
    try:
        credit = env.store.credit
        first = env.variant(boundary=10, prefix=[key(0)])
        second = env.variant(boundary=11, prefix=[key(0)])
        credit.pressure(second.uid)
        credit.supersede({first.uid}, second.uid)
        assert first.uid not in credit.guaranteed
        assert credit.bytes == sum(env.store.obligations(second).values())
        assert credit.events["transfer"] == 1
        amount = credit.bytes
        credit.capacity_bytes = credit.io_bytes
        credit.reconcile(promote=False)
        assert credit.bytes == 0
        credit.capacity_bytes = credit.io_bytes + amount
        credit.reconcile()
        assert second.uid in credit.guaranteed
        assert credit.events["promote"] >= 1
        for n in range(12, 22):
            unit = env.variant(boundary=n, prefix=[key(0)])
            assert credit.bytes <= credit.d_dyn
            assert env.manager.units.recoverable(unit.uid, "cpu")
        assert env.store.snapshot()["best_effort_variants"] >= 10
        assert not env.store.writes
        credit.rate = credit.floor_rate = 1
        credit.reconcile(promote=False)
        assert credit.bytes <= credit.d_dyn
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("poll_gap_s", [0.0, 0.001])
def test_credit_rate_counts_full_busy_interval_for_batched_receipts(
    tmp_path, poll_gap_s
):
    now = [1000.0]
    env = UnitEnv(
        tmp_path,
        clock=lambda: now[0],
        max_size_bytes=176 << 30,
        write_bytes_per_second=64 << 20,
    )
    try:
        credit = env.store.credit
        assert credit.rate == 64 << 20
        credit.started()
        now[0] += 16
        credit.started()  # The second queued transaction shares the interval.
        assert credit.measurement_start == 1000.0
        now[0] += 16
        credit.committed(1 << 30, busy=True)
        now[0] += poll_gap_s
        credit.committed(1 << 30, busy=False)
        print(f"B1 replay gap={poll_gap_s}s W={credit.rate / (1 << 20):.6f} MiB/s")
        assert credit.rate == pytest.approx(64 << 20, rel=0.0001)
        assert credit.events["committed_bytes"] == 2 << 30
        assert credit.measurement_start is None

        # An idle gap belongs to neither busy interval; retain the rate floor
        # and the existing no-completion degradation rule.
        now[0] += 100
        credit.started()
        assert credit.measurement_start == now[0]
        now[0] += 32
        credit.committed(1 << 30)
        assert credit.rate == 32 << 20
        env.variant()
        assert credit.bytes
        credit.started()
        now[0] += credit.window_s + 1
        credit.tick()
        assert credit.rate == credit.floor_rate
        assert credit.events["rate_stalled"] == 1
        credit.committed(1 << 20)
        assert credit.rate == credit.floor_rate
    finally:
        env.ssd.close(io_grace_s=5)


def test_credit_rate_real_shared_transactions_with_batched_manifests(
    tmp_path, monkeypatch
):
    # Match the review's production-sized, unaligned frames and healthy IO.
    row = 23_003_136
    monkeypatch.setattr(sys.modules[__name__], "ROW", row)
    monkeypatch.setattr(sys.modules[__name__], "payload", lambda n: bytes([n]) * row)
    env = UnitEnv(
        tmp_path,
        pages=24,
        max_size_bytes=2 << 30,
        write_threads=4,
        max_pending_blocks=8,
        write_bytes_per_second=64 << 20,
    )
    try:
        first, second = env.variant(ahead=6), env.variant(ahead=7)
        assert set(first.keys) & set(second.keys)
        submit = env.store._submit_manifest
        monkeypatch.setattr(env.store, "_submit_manifest", lambda txn, unit: None)
        started = time.monotonic()
        assert env.store.start_write(first)
        assert env.store.start_write(second)
        assert env.settle(
            lambda: all(
                not txn.remaining and not txn.active
                for txn in env.store.writes.values()
            ),
            timeout=25,
        )
        monkeypatch.setattr(env.store, "_submit_manifest", submit)
        env.store.advance_writes()
        assert len(env.store.manifest_jobs) == 2
        deadline = time.monotonic() + 5
        while len(env.ssd._tier._pool._finished_q) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        # Both real manifests (including fsync) precede one scheduler poll.
        env.ssd._poll()
        elapsed = time.monotonic() - started
        assert not env.store.writes
        assert env.store.manifest_ok(first) and env.store.manifest_ok(second)
        assert env.ssd.stats_counter["store_failures"] == 0
        nbytes = env.store.events["committed_bytes"]
        assert nbytes == env.store.credit.events["committed_bytes"]
        observed = nbytes / elapsed
        rate = env.store.credit.rate
        print(
            f"B1 real bytes={nbytes} wall={elapsed:.6f}s "
            f"observed={observed / (1 << 20):.6f} MiB/s "
            f"W={rate / (1 << 20):.6f} MiB/s errors=0"
        )
        # Allow small differences between the test and credit clock endpoints.
        assert rate <= observed * 1.1
        assert_conserved(env.manager)
    finally:
        env.ssd.close(io_grace_s=5)


def test_reset_inflight_retains_sources_then_restarts_current_epoch(
    tmp_path, monkeypatch
):
    import threading

    from vllm.v1.kv_offload.host.ssd import _HostTier

    env = UnitEnv(tmp_path, max_pending_blocks=1)
    original = _HostTier._write_page
    entered, release = threading.Event(), threading.Event()

    def blocked(self, *args):
        entered.set()
        assert release.wait(10)
        return original(self, *args)

    monkeypatch.setattr(_HostTier, "_write_page", blocked)
    try:
        unit = env.variant()
        assert env.store.start_write(unit)
        txn = env.store.writes[unit.uid]
        assert entered.wait(5)
        env.manager.reset_cache()
        assert all(txn.owner in env.table.get(h).leases for h in txn.handles.values())
        assert not env.store.manifests
        release.set()
        assert env.settle(lambda: not env.store.writes and not env.ssd._jobs)
        assert not env.store.cache._pins and not env.store.cache._claims
        current = env.variant()
        assert current.uid.reset == 1
        env.persist(current)
        env.ssd.close(io_grace_s=5)
        restarted = UnitEnv(tmp_path)
        try:
            assert restarted.manager.units.reset_epoch == 1
            assert len(restarted.store.manifests) == 1
            restored = next(
                u
                for u in restarted.manager.units.units.values()
                if u.parent is not None
            )
            assert restored.uid.reset == 1
            assert restarted.ssd.prefetch("reset-restart", restored.keys)
            assert restarted.settle(lambda: not restarted.ssd.waiting("reset-restart"))
            assert all(
                restarted.manager.lookup(k, ReqContext("reset-restart")) is True
                for k in restored.keys
            )
            restarted.ssd.cancel("reset-restart")
        finally:
            restarted.ssd.close(io_grace_s=5)
    finally:
        release.set()
        env.ssd.close(io_grace_s=5)


def test_zero_shutdown_budget_reports_unfinished_union(env):
    first = env.variant(ahead=6)
    second = env.variant(ahead=7)
    obligations = env.store.obligations(first) | env.store.obligations(second)
    publish(env.table, key(100), PageKind.MAIN)
    report = env.ssd.final_flush(budget_s=0)
    assert report.unfinished_bytes == sum(obligations.values()) + ROW
    assert report.guaranteed_completed_bytes == report.best_effort_completed_bytes == 0
    assert report.skipped == "no budget"
    assert not env.ssd._jobs
    assert env.store.credit.window_s == 0
    assert env.store.credit.bytes == 0
    assert env.store.credit.events["downgraded_shrink"] >= 1


def test_shutdown_prioritizes_guarantee_over_unsubmitted_best_effort(
    tmp_path, monkeypatch
):
    import threading

    from vllm.v1.kv_offload.host.ssd import _HostTier

    env = UnitEnv(tmp_path, max_pending_blocks=2)
    entered, release = threading.Event(), threading.Event()
    original = _HostTier._write_page

    def blocked(self, *args):
        entered.set()
        assert release.wait(10)
        return original(self, *args)

    monkeypatch.setattr(_HostTier, "_write_page", blocked)
    try:
        best = env.variant(boundary=5)
        guaranteed = env.variant(boundary=6)
        env.store.credit.pressure(best.uid)
        assert env.store.start_write(best)
        assert entered.wait(5)
        assert env.ssd._submit
        order = []
        finish = env.store._finish_write

        def finished(txn, success):
            if success:
                order.append(txn.uid)
            return finish(txn, success)

        monkeypatch.setattr(env.store, "_finish_write", finished)
        release.set()
        report = env.ssd.final_flush(budget_s=5)
        assert order == [guaranteed.uid, best.uid]
        assert report.unfinished_bytes == 0
        assert not env.store.cache._pins and not env.store.cache._claims
    finally:
        release.set()
        env.ssd.close(io_grace_s=5)


def test_request_byte_quota_is_shared_by_multiple_unit_transactions(tmp_path):
    env = UnitEnv(tmp_path, prefetch_max_per_request=1, prefetch_max_inflight=4)
    try:
        first = env.variant(boundary=20, prefix=[key(-1)])
        second = env.variant(boundary=40, prefix=[key(-2)])
        env.persist(first)
        env.persist(second)
        env.evict_cpu(first)
        env.evict_cpu(second)
        assert env.ssd.prefetch("shared-quota", first.keys)
        assert env.ssd.prefetch("shared-quota", second.keys)
        assert len(env.store.prefetches) == 2
        assert sum(len(t.active) for t in env.store.prefetches.values()) == 1
        assert env.settle(lambda: not env.ssd.waiting("shared-quota"))
        assert all(
            env.manager.is_ready_resident(k) for k in (*first.keys, *second.keys)
        )
        env.ssd.cancel("shared-quota")
        assert not env.store.prefetches
    finally:
        env.ssd.close(io_grace_s=5)


def test_latest_only_transfers_to_same_requests_new_leaf(env):
    first = env.variant(boundary=10, prefix=[key(0)])
    unrelated = env.variant(boundary=11, prefix=[key(0)])
    latest = env.variant(boundary=12, prefix=[key(0)])
    manager = env.manager
    for req_id, unit in [("r", first), ("other", unrelated), ("r", latest)]:
        boundary = manager.units.units[unit.parent]
        main = tuple(k for k in unit.keys if env.table.kind_for_key(k) == PageKind.MAIN)
        manager.offer_variant(
            req_id,
            VariantOffer(
                boundary.uid,
                boundary.gen,
                unit.uid.ahead,
                main,
                unit.boundary_position,
                unit.ahead_position,
                unit.draft_range,
            ),
        )
    credit = env.store.credit
    credit.pressure(unrelated.uid)
    credit.pressure(latest.uid)
    credit.capacity_bytes = credit.io_bytes + credit.bytes
    credit.reconcile(promote=False)
    credit.excluded.discard(unrelated.uid)
    manager.drop_pending_secondary_writes([key(10, 1)], req_id="r")
    assert credit.guaranteed == {latest.uid}
    assert credit.events["transfer"] == 1
    assert credit.events["guarantee_superseded"] == 1
    manager.on_request_finished(ReqContext("r"))
    assert "r" not in manager._latest_variant_offers


def test_prefetch_rejects_manifest_from_another_generation(env):
    unit = env.variant()
    env.persist(unit)
    unit.gen += 100
    env.evict_cpu(unit)
    assert env.ssd.prefetch("stale-manifest", unit.keys) == []
    assert not env.ssd.waiting("stale-manifest")
    assert not env.store.prefetches and not env.ssd._jobs
    assert env.store.events["prefetch_manifest_generation_mismatch"] == 1


def test_cached_credit_bills_match_live_graph_after_each_ssd_event(env):
    def assert_credit():
        expected = {}
        for uid in env.store.credit.guaranteed:
            unit = env.manager.units.units[uid]
            txn = env.store.writes.get(uid)
            missing = (
                txn.missing
                if txn is not None and txn.gen == unit.gen
                else {
                    k
                    for k in unit.keys
                    if env.manager.units.components[k].persisted is None
                }
            )
            expected.update({("component", k): ROW for k in missing})
            if not env.store.manifest_ok(unit):
                expected["manifest", (uid, unit.gen)] = env.store.manifest_bytes(unit)
        assert env.store.credit.bytes == sum(expected.values())
        assert env.store.credit.bytes <= env.store.credit.d_dyn
        assert set(env.store._obligation_cache) <= env.manager.units.units.keys()

    first = env.variant(ahead=6)
    second = env.variant(ahead=7)
    assert_credit()
    assert env.store.start_write(first)
    assert env.store.start_write(second)
    deadline = time.monotonic() + 5
    while env.store.writes and time.monotonic() < deadline:
        env.ssd.step()
        assert_credit()
        time.sleep(0.001)
    assert not env.store.writes
    third = env.variant(ahead=8)
    ahead = env.table.lookup(key(8))
    env.ssd._new_job(env.table.get(ahead), False)
    env.ssd._pump()
    assert env.settle(lambda: env.table.is_persisted(env.table.get(ahead)))
    assert_credit()
    assert env.store.drop(first.uid)
    assert_credit()
    assert env.store.drop(second.uid)
    assert_credit()
    env.persist(third)
    assert_credit()


def test_full_queue_skips_source_scans_and_shrink_drains_owners(tmp_path, monkeypatch):
    import threading

    from vllm.v1.kv_offload.host.ssd import _HostTier

    env = UnitEnv(tmp_path, max_pending_blocks=2)
    entered, release = threading.Event(), threading.Event()
    original = _HostTier._write_page

    def blocked(self, *args):
        entered.set()
        assert release.wait(10)
        return original(self, *args)

    monkeypatch.setattr(_HostTier, "_write_page", blocked)
    try:
        unit = env.variant()
        assert env.store.start_write(unit)
        txn = env.store.writes[unit.uid]
        assert entered.wait(5)

        def no_scan(*args, **kwargs):
            raise AssertionError("A full IO queue must not rescan transaction sources")

        with monkeypatch.context() as check:
            check.setattr(env.store.cache, "lookup", no_scan)
            env.store.advance_writes()
        env.store.credit.rate = env.store.credit.floor_rate = 1
        env.store.credit.reconcile(promote=False)
        env.ssd.step()
        assert txn.failed
        assert not env.ssd._submit
        assert all(txn.owner in env.table.get(h).leases for h in txn.handles.values())
        release.set()
        assert env.settle(lambda: not env.store.writes and not env.ssd._jobs)
        assert not env.store.cache._pins and not env.store.cache._claims
        assert env.manager.units.recoverable(unit.uid, "cpu")
        assert env.store.credit.bytes == 0
        assert_conserved(env.manager)
    finally:
        release.set()
        env.ssd.close(io_grace_s=5)


@pytest.mark.parametrize("factor", [1, 3])
@pytest.mark.parametrize("adapted", [False, True])
def test_incremental_offload_keys_seek_only_new_hashes(factor, adapted):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestOffloadState,
    )
    from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
    from vllm.v1.kv_offload.base import make_offload_key

    class CountedHashes(BlockHashListWithBlockSize):
        calls = 0

        def _get_value_at(self, idx):
            self.calls += 1
            return super()._get_value_at(idx)

    base = [sha256(i) for i in range(18)]
    hashes = CountedHashes(base, 16, 32) if adapted else base
    group = SimpleNamespace(group_idx=0, hash_block_size_factor=factor)
    group_state = SimpleNamespace(offload_keys=[])
    state = SimpleNamespace(
        req=SimpleNamespace(block_hashes=hashes),
        config=SimpleNamespace(kv_group_configs=[group]),
        group_states=(group_state,),
    )
    for extra in ([], [], [sha256(i) for i in range(18, 26)], []):
        before = len(group_state.offload_keys)
        if isinstance(hashes, CountedHashes):
            hashes.calls = 0
        base.extend(extra)
        RequestOffloadState.update_offload_keys(state)
        if isinstance(hashes, CountedHashes):
            assert hashes.calls == len(group_state.offload_keys) - before
        assert group_state.offload_keys == [
            make_offload_key(hashes[i], 0)
            for i in range(factor - 1, len(hashes), factor)
        ]
