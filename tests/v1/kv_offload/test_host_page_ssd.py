# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""host SSD persistence host-page SSD persistence on CPU, with real files and processes.

Covers design 9 items 1-8 and the host SSD persistence exit items: delayed write without
a pin,
write-back bounded by net headroom (parked successors protected), final flush
budget/report/quarantine, SIGKILL half-writes, reset epoch and layout
namespaces, prefetch quotas, and shutdown-window truncation.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_revolta_fs import checkpoint_spec
from vllm.config import KVTransferConfig
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host import persist
from vllm.v1.kv_offload.host import ssd as ssd_module
from vllm.v1.kv_offload.host.page_table import HostPageTable, PageState, Result
from vllm.v1.kv_offload.host.persist import (
    SHUTDOWN_MARGIN_S,
    HostPageSsd,
    plan_final_flush,
    run_final_flush,
    shutdown_window_warning,
)
from vllm.v1.kv_offload.host.ssd import HostKVLayout
from vllm.v1.kv_offload.tiering.fs.config import host_kv_ssd_config

ROW = 257  # deliberately unaligned row bytes
LAYOUT = HostKVLayout(
    version="host-ssd-fixture-v1",
    storage_layout="row=[main|index]; uint8",
    encoding="uint8-fixture",
    layer_names=("qsa.0",),
    kv_scales=((1.0, 1.0),),
)


def key(n, group=0):
    return make_offload_key(sha256(n), group)


def payload(n):
    return bytes((i * 7 + n) % 256 for i in range(ROW))


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def options(root, **overrides):
    spec, manifest = checkpoint_spec(root)
    raw = {
        "enabled": True,
        "root_dir": str(root),
        "max_size_bytes": 1 << 20,
        "min_free_bytes": 0,
        "content_manifest_path": manifest,
        "max_pending_blocks": 8,
        "write_bytes_per_second": 1 << 30,
        "write_delay_s": 0,
        "write_threads": 1,
        **overrides,
    }
    config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"host_kv.ssd": raw},
    )
    spec.vllm_config.kv_transfer_config = config
    kv_cache_utils.init_none_hash(
        sha256, kv_transfer_config=config, vllm_config=spec.vllm_config
    )
    return spec, host_kv_ssd_config(config)


class Env:
    def __init__(self, root, pages=6, layout=LAYOUT, clock=None, **overrides):
        self.table = HostPageTable(pages, num_workers=1)
        self.backing = bytearray(pages * ROW)
        view = memoryview(self.backing).cast("B", shape=[pages, ROW])
        spec, opts = options(root, **overrides)
        self.clock = clock or time.monotonic
        self.ssd = HostPageSsd(self.table, view, spec, layout, opts, clock=self.clock)
        assert self.ssd.enabled

    def publish(self, n, *, bind=None):
        handle = self.table.reserve()
        assert handle is not None
        start = handle.page_id * ROW
        self.backing[start : start + ROW] = payload(n)
        if bind is not None:
            self.table.bind(handle, bind, 0)
        self.table.producer_result(handle, compute=Result.OK, seal=Result.OK)
        self.table.publish(handle, key(n))
        return handle

    def settle(self, cond, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.ssd.step()
            if cond():
                return True
            time.sleep(0.005)
        return cond()


@pytest.fixture
def envs():
    made = []

    def make(*args, **kwargs):
        env = Env(*args, **kwargs)
        made.append(env)
        return env

    yield make
    for env in reversed(made):
        env.ssd.close(io_grace_s=5)


@pytest.fixture
def gate(monkeypatch):
    """Block host-page writes for chosen page ids until released."""
    blocked: set[int] = set()
    started = threading.Event()
    release = threading.Event()
    write = ssd_module.store_verified_block

    def gated(path, view, offset, size, *args, **kwargs):
        if offset // size in blocked:
            started.set()
            release.wait()
        return write(path, view, offset, size, *args, **kwargs)

    monkeypatch.setattr(ssd_module, "store_verified_block", gated)
    yield SimpleNamespace(blocked=blocked, started=started, release=release)
    release.set()


def page_files(root):
    return [
        p
        for p in Path(root).rglob("*")
        if p.is_file()
        and ".tmp" not in p.name
        and not p.name.startswith(".")
        and p.suffix != ".json"
    ]


# ---------------------------------------------------------------- item 1


def test_published_page_is_queued_without_pin_and_dropped_if_evicted_first(
    tmp_path, envs
):
    clock = Clock()
    env = envs(tmp_path / "ssd", pages=3, clock=clock, write_delay_s=300)
    first = env.publish(1)
    page = env.table.get(first)
    assert not (page.leases or page.io) and env.ssd.stats()["delayed_depth"] == 1
    env.publish(2)
    # R_host reserves everything: no write-back, direct eviction (D-B).
    env.table.committed_reserve = lambda: 2
    assert env.table.reserve() is not None
    stats = env.ssd.stats()
    assert stats["evict_unwritten"] == 1 and stats.get("writeback_submitted", 0) == 0
    assert env.table.lookup(key(1)) is None
    clock.now += 301
    env.table.committed_reserve = lambda: 0
    env.settle(lambda: not env.ssd._jobs)
    stats = env.ssd.stats()
    assert stats["delayed_dropped_evicted"] == 1
    assert stats["pages_written"] == 1  # only the surviving page 2
    assert env.table.is_persisted(env.table.get(env.table.lookup(key(2))))


def test_delay_is_respected_and_due_pages_are_written(tmp_path, envs):
    clock = Clock()
    env = envs(tmp_path / "ssd", clock=clock, write_delay_s=300)
    handle = env.publish(1)
    env.ssd.step()
    assert not env.ssd._jobs and not env.table.is_persisted(env.table.get(handle))
    clock.now += 300
    assert env.settle(lambda: env.table.is_persisted(env.table.get(handle)))
    assert len(page_files(tmp_path / "ssd")) == 1


def test_write_volume_counts_completed_writes_only(tmp_path, envs, monkeypatch):
    """delayed_written and the 24 h volume are charged when a store completes;
    a failed store is not charged (it wrote nothing durable)."""
    env = envs(tmp_path / "ssd")
    write = ssd_module.store_verified_block

    def broken(*args, **kwargs):
        raise OSError(28, "injected: no space left")

    monkeypatch.setattr(ssd_module, "store_verified_block", broken)
    failed = env.publish(1)
    assert env.settle(lambda: env.ssd.stats().get("store_failures") == 1)
    stats = env.ssd.stats()
    assert stats.get("delayed_written", 0) == 0
    assert stats["delayed_written_bytes_24h"] == 0
    assert not env.table.is_persisted(env.table.get(failed))
    monkeypatch.setattr(ssd_module, "store_verified_block", write)
    done = env.publish(2)
    assert env.settle(lambda: env.table.is_persisted(env.table.get(done)))
    stats = env.ssd.stats()
    assert stats["delayed_written"] == 1 and stats["pages_written"] == 1
    assert stats["delayed_written_bytes_24h"] == ROW


# ---------------------------------------------------------------- item 2


def test_writeback_is_skipped_when_it_would_take_a_parked_successor_page(
    tmp_path, envs
):
    # Review counterexample: 3 reclaimable pages, n = 1, R_host = 2.
    env = envs(tmp_path / "ssd", pages=4, write_delay_s=300)
    for n in (1, 2, 3):
        env.publish(n)
    env.table.committed_reserve = lambda: 2
    assert env.table.net_headroom(1) == 0
    taken = env.table.reserve()
    assert taken is not None
    env.table.bind(taken, 0, 0)  # the new request's page
    stats = env.ssd.stats()
    assert stats.get("writeback_submitted", 0) == 0
    assert stats["evict_unwritten"] == 1 and stats["writeback_no_headroom"] == 1
    # The two successor pages stay reclaimable for the parked request.
    assert env.table.available() == 2 and not env.ssd._jobs


def test_writeback_with_headroom_skips_the_page_and_evicts_the_next(tmp_path, envs):
    env = envs(tmp_path / "ssd", pages=4, write_delay_s=300)
    for n in (1, 2, 3):
        env.publish(n)
    assert env.table.reserve() is not None
    stats = env.ssd.stats()
    # LRU walk: keys 1 and 2 are written back while headroom (2) pays for
    # them; key 3 is then evicted unwritten.
    assert stats["writeback_submitted"] == 2 and stats["evict_unwritten"] == 1
    assert env.table.lookup(key(3)) is None
    assert env.settle(lambda: not env.ssd._jobs)
    for n in (1, 2):
        assert env.table.is_persisted(env.table.get(env.table.lookup(key(n))))


def test_writeback_counts_the_rest_of_an_admitted_batch(tmp_path, envs):
    env = envs(tmp_path / "ssd", pages=3, write_delay_s=300)
    env.publish(1)
    env.publish(2)
    # Admission promised two pages; a write-back would leave only one.
    assert env.table.reserve(pending=2) is not None
    assert env.table.reserve(pending=1) is not None
    assert env.ssd.stats().get("writeback_submitted", 0) == 0


def test_delayed_write_charges_net_headroom_only_for_reclaimable_pages(tmp_path, envs):
    env = envs(tmp_path / "ssd", pages=4)
    bound = env.publish(1, bind=7)  # active binding: write costs nothing
    free = env.publish(2)  # reclaimable: write costs one page
    env.table.committed_reserve = lambda: env.table.available()
    assert env.table.net_headroom() == 0
    env.ssd.step()
    assert set(env.ssd._page_jobs) == {bound.page_id}
    assert env.ssd.stats()["deferred_no_headroom"] == 1
    assert env.settle(lambda: env.table.is_persisted(env.table.get(bound)))
    assert not env.table.is_persisted(env.table.get(free))
    env.table.committed_reserve = lambda: 0
    assert env.settle(lambda: env.table.is_persisted(env.table.get(free)))


# ---------------------------------------------------------------- item 3


def test_page_under_writeback_cannot_be_evicted(tmp_path, envs, gate):
    env = envs(tmp_path / "ssd", pages=3, write_delay_s=300)
    first = env.publish(1)
    gate.blocked.add(first.page_id)
    env.publish(2)
    taken = env.table.reserve()  # writes back 1, evicts 2
    assert taken is not None
    env.table.bind(taken, 0, 0)
    assert gate.started.wait(5)
    page = env.table.get(first)
    assert page.io and not env.table.reclaimable(page)
    assert env.table.reserve() is None  # nothing else may be taken
    assert env.table.get(first).state == PageState.READY
    gate.release.set()
    assert env.settle(lambda: env.table.is_persisted(env.table.get(first)))
    assert env.table.reserve() is not None  # persisted page evicts directly
    assert env.ssd.stats()["evicted_persisted"] == 1


# ---------------------------------------------------------------- item 4


def test_persisted_page_not_rewritten_and_other_copy_is_not_persisted(tmp_path, envs):
    env = envs(tmp_path / "ssd", pages=4)
    first = env.publish(1)
    assert env.settle(lambda: env.table.is_persisted(env.table.get(first)))
    env.ssd.on_published(first)  # a duplicate queue entry
    env.settle(lambda: not env.ssd._jobs)
    assert env.ssd.stats()["pages_written"] == 1
    assert env.ssd.stats()["already_persisted"] == 1
    # A SHADOW copy of the same hash is not READY and never inherits it.
    handle = env.table.reserve()
    page = env.table.get(handle)
    env.table.producer_result(handle, compute=Result.OK, seal=Result.OK)
    env.table.publish(handle, key(1))
    assert page.state == PageState.SHADOW and not env.table.is_persisted(page)
    # A new incarnation after eviction is not persisted until its own write.
    env.table._lru.move_to_end(first.page_id, last=False)
    env.table.committed_reserve = lambda: 0
    victim = env.table.get(first)
    env.table._recycle(victim)
    again = env.publish(1)
    assert not env.table.is_persisted(env.table.get(again))
    assert env.settle(lambda: env.table.is_persisted(env.table.get(again)))


# ---------------------------------------------------------------- item 5


def test_final_flush_lifts_the_rate_limit(tmp_path, envs):
    env = envs(tmp_path / "ssd", write_bytes_per_second=1, write_delay_s=300)
    handles = [env.publish(n) for n in range(4)]
    started = time.monotonic()
    report = env.ssd.final_flush(budget_s=10, io_grace_s=1)
    assert time.monotonic() - started < 5  # 4 * 257 B at 1 B/s would be ~17 min
    assert report.candidates == 4 and report.completed == 4
    assert not report.deadline_hit and not report.quarantined
    assert all(env.table.is_persisted(env.table.get(h)) for h in handles)
    assert len(page_files(tmp_path / "ssd")) == 4


def test_final_flush_deadline_cancels_queued_and_drains_started_writes(
    tmp_path, envs, gate
):
    env = envs(tmp_path / "ssd", write_delay_s=300)
    handles = [env.publish(n) for n in range(4)]
    newest = handles[-1]
    gate.blocked.add(newest.page_id)  # newest goes first and blocks
    timer = threading.Timer(0.6, gate.release.set)
    timer.start()
    report = env.ssd.final_flush(budget_s=0.3, io_grace_s=5)
    timer.join()
    assert report.deadline_hit and report.candidates == 4
    assert report.completed == 0 and report.failed == 0
    assert report.in_flight_at_deadline == 1 and report.dropped_unsubmitted == 3
    assert report.late_completed == 1 and report.stuck == 0
    assert report.candidates == (
        report.completed
        + report.failed
        + report.dropped_unsubmitted
        + report.in_flight_at_deadline
    )
    assert report.in_flight_at_deadline == (
        report.late_completed + report.late_failed + report.stuck
    )
    persisted = [h for h in handles if env.table.is_persisted(env.table.get(h))]
    assert persisted == [newest] and len(page_files(tmp_path / "ssd")) == 1
    assert not any(p.io for p in env.table.pages)


def _shutdown_with(env, plan=None, deadline=None):
    from vllm.v1.core.sched.scheduler import Scheduler

    calls = []

    class Region:
        mmap_path = "/dev/shm/fake"

        def cleanup(self):
            calls.append(any(p.io for p in env.table.pages))

    fake = SimpleNamespace(
        host_kv_ssd=env.ssd,
        host_kv_flush_plan=plan,
        kv_event_publisher=None,
        connector=None,
        ec_connector=None,
        host_region=Region(),
        host_history_pool=SimpleNamespace(host_pages=env.table),
    )
    Scheduler.shutdown(fake, deadline=deadline)
    return calls


def test_scheduler_unlinks_only_after_every_io_owner_retired(tmp_path, envs, gate):
    env = envs(tmp_path / "ssd", write_delay_s=300)
    first = env.publish(1)
    env.publish(2)
    gate.blocked.add(first.page_id)
    env.ssd.on_published(first)
    env.ssd._queue.delay_s = 0
    env.ssd.step()
    threading.Timer(0.3, gate.release.set).start()
    calls = _shutdown_with(env)
    assert calls == [False]  # cleanup ran once, with no IO owner left
    assert all(env.table.is_persisted(env.table.get(h)) for h in (first,))


# ---------------------------------------------------------------- item 6


def test_io_that_never_finishes_quarantines_without_hanging(tmp_path, envs, gate):
    env = envs(tmp_path / "ssd", write_delay_s=300)
    first = env.publish(1)
    gate.blocked.add(first.page_id)
    plan = persist.FlushPlan(0.2, None, None, "test")
    env.ssd.options["shutdown_io_grace_s"] = 0.2
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="retaining region"):
        _shutdown_with(env, plan)
    assert time.monotonic() - started < 3
    assert env.ssd.quarantined
    assert env.table.get(first).io  # the hung row keeps its owner


def test_scheduler_deadline_bounds_ssd_without_engine_plan(tmp_path, envs, gate):
    env = envs(tmp_path / "ssd", shutdown_io_grace_s=5)
    handle = env.publish(1)
    gate.blocked.add(handle.page_id)
    env.ssd.step()
    assert gate.started.wait(1)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _shutdown_with(env, deadline=started + 0.1)
    assert time.monotonic() - started < 1
    assert env.ssd.quarantined and env.table.get(handle).io


# ---------------------------------------------------------------- item 7

_CHILD = r"""
import os, sys, time
from tests.v1.kv_offload import test_host_page_ssd as t
root = sys.argv[1]
env = t.Env(t.Path(root), write_delay_s=0)
done = env.publish(1)
assert env.settle(lambda: env.table.is_persisted(env.table.get(done)))
real = os.write
def half(fd, data):
    real(fd, bytes(data)[: len(data) // 2])
    print("HALF", flush=True)
    time.sleep(120)
import vllm.v1.kv_offload.tiering.fs.revolta as r
r.os.write = half  # the page writer's os.write
env.publish(2)
env.ssd.step()
time.sleep(120)
"""


def test_sigkill_mid_write_rejects_half_file_and_keeps_complete_file(tmp_path):
    root = tmp_path / "ssd"
    options(root)  # create fixtures before the child runs
    env_vars = dict(os.environ)
    env_vars["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[3]), env_vars.get("PYTHONPATH", "")]
    )
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_vars,
    )
    try:
        assert child.stdout is not None
        seen = []
        for line in child.stdout:
            if line.strip() == "HALF":
                break
            seen.append(line)
        else:
            pytest.fail("child exited before the half write:\n" + "".join(seen))
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(30)
    assert any(".tmp" in p.name for p in root.rglob("*") if p.is_file())
    fresh = Env(root)
    try:
        assert fresh.ssd.prefetch("r", [key(1), key(2)]) == [key(1)]
        assert fresh.settle(lambda: not fresh.ssd._jobs)
        handle = fresh.table.lookup(key(1))
        assert fresh.table.is_ready(handle)
        start = handle.page_id * ROW
        assert bytes(fresh.backing[start : start + ROW]) == payload(1)
        assert fresh.table.lookup(key(2)) is None
    finally:
        fresh.ssd.close(io_grace_s=5)


def test_truncated_or_corrupt_page_is_rejected_and_recycled(tmp_path, envs):
    root = tmp_path / "ssd"
    writer = envs(root)
    handle = writer.publish(1)
    assert writer.settle(lambda: writer.table.is_persisted(writer.table.get(handle)))
    writer.ssd.close(io_grace_s=5)
    (path,) = page_files(root)
    with open(path, "r+b") as f:
        f.truncate(ROW - 1)
    reader = envs(root)
    assert reader.ssd.prefetch("r", [key(1)]) == [key(1)]
    assert reader.settle(lambda: not reader.ssd._jobs)
    assert reader.table.lookup(key(1)) is None
    assert reader.ssd.stats()["prefetch_failed"] == 1
    assert reader.table.available() == len(reader.table.pages) - 1


# ---------------------------------------------------------------- item 8


def test_reset_epoch_and_layout_change_never_hit_old_files(tmp_path, envs):
    root = tmp_path / "ssd"
    writer = envs(root)
    handle = writer.publish(1)
    assert writer.settle(lambda: writer.table.is_persisted(writer.table.get(handle)))
    assert writer.ssd._tier.lookup(writer.ssd._tier_key(key(1))) is True
    writer.table.reset()
    assert writer.table.ssd_epoch == 1
    assert writer.ssd.prefetch("r", [key(1)]) == []
    writer.ssd.close(io_grace_s=5)
    other = HostKVLayout(
        version="host-ssd-fixture-v2",
        storage_layout=LAYOUT.storage_layout,
        encoding=LAYOUT.encoding,
        layer_names=LAYOUT.layer_names,
        kv_scales=LAYOUT.kv_scales,
    )
    changed = envs(root, layout=other)
    assert changed.ssd.prefetch("r", [key(1)]) == []
    changed.ssd.close(io_grace_s=5)
    same = envs(root)  # positive control: a restart with the same ABI hits
    assert same.ssd.prefetch("r", [key(1)]) == [key(1)]
    assert same.settle(lambda: not same.ssd._jobs)
    assert same.table.is_ready(same.table.lookup(key(1)))


def test_full_length_host_page_key_is_written_and_hits_after_restart(tmp_path, envs):
    """A host page key concatenates one 32-byte hash per hash block (98 for a
    1568-token page). Its hex alone is 6272 characters, far over the 255-byte
    file name limit; the SSD key must stay short and stable across restarts."""
    root = tmp_path / "ssd"
    long_key = make_offload_key(b"".join(sha256(i) for i in range(98)), 0)
    assert len(long_key) == 98 * 32 + 4
    writer = envs(root, write_delay_s=300)
    handle = writer.table.reserve()
    assert handle is not None
    start = handle.page_id * ROW
    writer.backing[start : start + ROW] = payload(5)
    writer.table.producer_result(handle, compute=Result.OK, seal=Result.OK)
    writer.table.publish(handle, long_key)
    report = writer.ssd.final_flush(budget_s=5.0, io_grace_s=5.0)
    assert (report.candidates, report.completed, report.failed) == (1, 1, 0)
    files = page_files(root)
    assert len(files) == 1 and all(len(f.name) < 255 for f in files)
    writer.ssd.close(io_grace_s=5)
    same = envs(root)
    assert same.ssd.prefetch("r", [long_key]) == [long_key]
    assert same.settle(lambda: not same.ssd._jobs)
    loaded = same.table.lookup(long_key)
    assert same.table.is_ready(loaded)
    row = loaded.page_id * ROW
    assert bytes(same.backing[row : row + ROW]) == payload(5)


def test_reset_during_store_and_load_never_publishes_old_content(tmp_path, envs, gate):
    root = tmp_path / "ssd"
    seed = envs(root)
    h = seed.publish(9)
    assert seed.settle(lambda: seed.table.is_persisted(seed.table.get(h)))
    seed.ssd.close(io_grace_s=5)
    env = envs(root)
    first = env.publish(1)
    gate.blocked.add(first.page_id)
    env.ssd.step()
    assert gate.started.wait(5)
    assert env.ssd.prefetch("r", [key(9)]) == [key(9)]
    env.table.reset()
    gate.release.set()
    assert env.settle(lambda: not env.ssd._jobs)
    assert env.table.lookup(key(9)) is None and env.table.lookup(key(1)) is None
    assert env.table.available() == len(env.table.pages) - 1
    assert env.ssd.stats()["store_failures"] == 1


# ---------------------------------------------------------------- prefetch


def _seed(root, envs, n):
    env = envs(root, pages=n + 2)
    hs = [env.publish(i) for i in range(n)]
    assert env.settle(lambda: all(env.table.is_persisted(env.table.get(h)) for h in hs))
    env.ssd.close(io_grace_s=5)


def test_prefetch_quotas_round_limit_holds_and_timeout(tmp_path, envs):
    root = tmp_path / "ssd"
    _seed(root, envs, 5)
    clock = Clock()
    env = envs(
        root,
        pages=8,
        clock=clock,
        prefetch_max_inflight=3,
        prefetch_max_per_request=2,
        prefetch_timeout_s=5,
    )
    keys = [key(i) for i in range(5)]
    assert env.ssd.prefetch("a", keys) == keys[:2]  # per-request quota
    assert env.ssd.prefetch("a", keys[2:]) == []  # one round per T_pf
    assert env.ssd.prefetch("b", keys[2:]) == keys[2:3]  # global quota
    assert env.ssd.stats()["prefetch_request_quota"] == 1
    assert env.ssd.stats()["prefetch_global_quota"] == 1
    assert env.settle(lambda: not env.ssd._jobs)
    held = env.ssd.holds("a")
    assert len(held) == 2 and all(env.table.get(h).leases for h in held)
    # Holds keep READY pages from eviction until consumed or T_pf passes.
    assert all(not env.table.reclaimable(env.table.get(h)) for h in held)
    env.ssd.release_holds("a")
    assert all(not env.table.get(h).leases for h in held)
    clock.now += 5
    env.ssd.step()
    assert not env.ssd._rounds and env.ssd.stats()["prefetch_timeouts"] == 1
    assert not env.table.get(env.table.lookup(keys[2])).leases
    assert env.ssd.prefetch("a", keys[3:]) == keys[3:5]  # a new window


def test_prefetch_timeout_revokes_waiting_but_io_owner_waits_for_read(
    tmp_path, envs, monkeypatch
):
    root = tmp_path / "ssd"
    _seed(root, envs, 1)
    clock = Clock()
    env = envs(root, clock=clock, prefetch_timeout_s=1)
    release = threading.Event()
    read = ssd_module.load_verified_block

    def slow(*args, **kwargs):
        release.wait()
        return read(*args, **kwargs)

    monkeypatch.setattr(ssd_module, "load_verified_block", slow)
    assert env.ssd.prefetch("a", [key(0)]) == [key(0)]
    handle = env.table.lookup(key(0))
    clock.now += 1
    env.ssd.step()
    assert not env.ssd._rounds
    page = env.table.get(handle)
    assert page.state == PageState.LOADING and page.io  # owner stays
    release.set()
    assert env.settle(lambda: not env.ssd._jobs)
    page = env.table.get(handle)
    assert page.state == PageState.READY and not page.leases
    assert env.table.is_persisted(page) and env.table.reclaimable(page)


def test_prefetch_respects_net_headroom(tmp_path, envs):
    root = tmp_path / "ssd"
    _seed(root, envs, 1)
    env = envs(root)
    env.table.committed_reserve = lambda: env.table.available()
    assert env.ssd.prefetch("a", [key(0)]) == []
    assert env.ssd.stats()["prefetch_no_headroom"] == 1


# ---------------------------------------------------------------- shutdown window


def test_flush_plan_truncates_to_shutdown_window():
    from vllm.v1.engine.core import EngineCore

    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = SimpleNamespace(shutdown_timeout=30)
    engine._shutdown_requested_at = 100.0
    deadline = engine.shutdown_deadline()
    assert deadline == 100 + 30 - SHUTDOWN_MARGIN_S
    assert plan_final_flush(10, 30, None, 50).budget_s == 10
    plan = plan_final_flush(10, 30, deadline, 104.0)
    assert plan.budget_s == 10 and plan.deadline == deadline
    assert plan_final_flush(10, 30, deadline, 118.0).budget_s == pytest.approx(7.0)
    plan = plan_final_flush(10, 30, deadline, 126.0)
    assert plan.budget_s == 0 and plan.reason == "shutdown window exhausted"
    engine.vllm_config.shutdown_timeout = 0
    deadline = engine.shutdown_deadline()
    plan = plan_final_flush(10, 0, deadline, 100.0)
    assert plan.budget_s == 0 and "shutdown_timeout=0" in plan.reason
    assert plan.deadline == deadline == 95.0


def test_startup_window_warning():
    assert shutdown_window_warning(10, 15, 30) is None
    assert "exceeds" in shutdown_window_warning(10, 20, 30)
    assert "skips" in shutdown_window_warning(10, 0, 0)


def test_flush_shares_the_deadline_left_after_the_fs_drain(tmp_path, envs):
    env = envs(tmp_path / "ssd", write_delay_s=300)
    env.publish(1)
    now = env.ssd.clock()
    # The fs drain used the whole shared window: nothing new is written.
    plan = persist.FlushPlan(10.0, now - 0.1, now - 30, "ok")
    report = run_final_flush(env.ssd, plan, now)
    assert report.skipped == "no budget" and report.candidates == 0
    assert not page_files(tmp_path / "ssd")
    report = run_final_flush(env.ssd, persist.FlushPlan(0.0, None, now, "x"), now)
    assert report.skipped == "x"


def test_tiering_drain_is_capped_by_the_shared_deadline():
    from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
    from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

    tier = FileSystemTierManager.__new__(FileSystemTierManager)
    tier._pool = DualQueueThreadPool(0, 0, drain_timeout_s=20)
    tier._cache = SimpleNamespace(release=lambda: None)
    tier._closed = False
    tier.tier_type = "fs"
    manager = object.__new__(TieringOffloadingManager)
    manager.secondary_tiers = [tier]
    manager._delayed = {}
    manager.primary_tier = SimpleNamespace(shutdown=lambda: None)
    deadline = time.monotonic() + 3
    manager.set_shutdown_deadline(deadline)
    manager.shutdown()
    assert manager._shutdown_deadline == tier._shutdown_deadline == deadline
    assert tier._closed


def test_disabled_switch_attaches_nothing(tmp_path):
    spec, _ = checkpoint_spec(tmp_path / "unused")
    config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={},
    )
    assert host_kv_ssd_config(config) is None
    vllm_config = SimpleNamespace(kv_transfer_config=config, shutdown_timeout=30)
    table = HostPageTable(3)
    assert persist.create_host_page_ssd(vllm_config, None, table, None, None) is None
    assert table.persistence is None


def test_engine_core_shutdown_plans_flush_from_signal_time(monkeypatch):
    import vllm.v1.engine.core as core_module
    from vllm.v1.engine.core import EngineCore

    monkeypatch.setattr(core_module, "cleanup_dist_env_and_memory", lambda: None)
    plans = []
    scheduler = SimpleNamespace(
        host_history_pool=SimpleNamespace(),
        host_kv_ssd=SimpleNamespace(options={"shutdown_flush_timeout_s": 10.0}),
        host_kv_flush_plan=None,
    )
    deadlines = []

    def shutdown(deadline=None):
        plans.append(scheduler.host_kv_flush_plan)
        deadlines.append(deadline)

    scheduler.shutdown = shutdown
    for timeout, elapsed, budget in ((30, 20.0, 5.0), (30, 1.0, 10.0), (0, 0.0, 0.0)):
        engine = EngineCore.__new__(EngineCore)
        engine.scheduler = scheduler
        engine.model_executor = None
        engine.structured_output_manager = SimpleNamespace(clear_backend=lambda: None)
        engine.vllm_config = SimpleNamespace(shutdown_timeout=timeout)
        engine._shutdown_requested_at = time.monotonic() - elapsed
        engine.shutdown()
        assert plans[-1].budget_s == pytest.approx(budget, abs=0.2)
        assert plans[-1].deadline == deadlines[-1] == engine.shutdown_deadline()
    # A shared deadline provided by the shutdown owner takes precedence.
    engine._host_kv_shutdown_complete = False
    engine.vllm_config.shutdown_timeout = 30
    deadline = time.monotonic() + 3
    engine.shutdown_deadline = lambda: deadline
    engine.shutdown()
    assert plans[-1].budget_s == pytest.approx(3.0, abs=0.2)
    assert plans[-1].deadline == deadlines[-1] == deadline


def test_engine_fs_and_ssd_share_deadline_without_extra_grace(
    tmp_path, envs, gate, monkeypatch
):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
    from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

    env = envs(tmp_path / "ssd", shutdown_io_grace_s=5)
    handle = env.publish(1)
    gate.blocked.add(handle.page_id)
    env.ssd.step()
    assert gate.started.wait(1)
    assert all(job_id < 0 for job_id in env.table.get(handle).io)

    fs_gate = threading.Event()
    fs_entered = threading.Event()
    tier = FileSystemTierManager.__new__(FileSystemTierManager)
    tier._pool = DualQueueThreadPool(0, 1, drain_timeout_s=30)
    tier._cache = SimpleNamespace(release=lambda: None)
    tier._closed = False
    tier.tier_type = "fs"

    def fs_io():
        fs_entered.set()
        fs_gate.wait()

    tier._pool.enqueue_store(1, 1, [fs_io])
    assert fs_entered.wait(1)
    manager = TieringOffloadingManager.__new__(TieringOffloadingManager)
    manager._delayed = {}
    manager.secondary_tiers = [tier]
    manager.primary_tier = SimpleNamespace(shutdown=lambda: None)
    connector_scheduler = OffloadingConnectorScheduler.__new__(
        OffloadingConnectorScheduler
    )
    connector_scheduler.manager = manager
    connector = OffloadingConnector.__new__(OffloadingConnector)
    connector.connector_worker = None
    connector.connector_scheduler = connector_scheduler
    released = []
    region = SimpleNamespace(cleanup=lambda: released.append("region"))
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_event_publisher = scheduler.ec_connector = None
    scheduler.connector = connector
    scheduler.host_region = region
    scheduler.host_kv_ssd = env.ssd
    scheduler.host_history_pool = SimpleNamespace(
        _jobs={}, host_pages=env.table, pending_seals={}
    )
    scheduler.current_step = env.table.watermark
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = scheduler
    engine.vllm_config = SimpleNamespace(shutdown_timeout=5.2)
    engine.model_executor = SimpleNamespace(
        collective_rpc=lambda *_, **__: [(0, None, None)], shutdown=lambda: None
    )
    engine.structured_output_manager = SimpleNamespace(clear_backend=lambda: None)
    monkeypatch.setattr("vllm.v1.engine.core.cleanup_dist_env_and_memory", lambda: None)
    timer = threading.Timer(0.04, fs_gate.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            engine.shutdown()
        assert time.monotonic() - started < 1
        assert tier._closed and engine._host_kv_drained
        assert manager._shutdown_deadline == tier._shutdown_deadline
        assert manager._shutdown_deadline == engine.shutdown_deadline()
        assert scheduler.host_kv_flush_plan.deadline == engine.shutdown_deadline()
        assert env.ssd.quarantined and engine._host_kv_quarantined
        assert not getattr(engine, "_host_kv_shutdown_complete", False)
        assert scheduler.host_region is region and scheduler.host_kv_ssd is env.ssd
        assert env.table.get(handle).io and not released
        with pytest.raises(RuntimeError, match="quarantined"):
            engine.shutdown()
    finally:
        fs_gate.set()
        timer.join()
        tier._pool.shutdown()


def test_history_pool_batch_admission_survives_writeback(tmp_path):
    """get_new_blocks passes the rest of its batch as ``pending``: write-back
    may never take a page the admitted batch still needs."""
    from vllm.v1.core.history_block_pool import HistoryBlockPool
    from vllm.v1.core.long_context_kv import HOST

    pool = HistoryBlockPool(9, True, 16, False)
    pool.configure_host(4, 1, 0)
    pool.admission_observer = SimpleNamespace(
        _scope=(HOST,), on_allocated=lambda _: None, on_released=lambda _: None
    )
    table = pool.host_pages
    backing = bytearray(4 * ROW)
    spec, opts = options(tmp_path / "ssd", write_delay_s=300)
    ssd = HostPageSsd(
        table, memoryview(backing).cast("B", shape=[4, ROW]), spec, LAYOUT, opts
    )
    try:
        for n in range(3):
            handle = table.reserve()
            table.producer_result(handle, compute=Result.OK, seal=Result.OK)
            table.publish(handle, key(n))
        assert table.available() == 3
        blocks = pool.get_new_blocks(2)
        assert len(blocks) == 2 and all(b.block_id in pool.bindings for b in blocks)
        stats = ssd.stats()
        assert stats["writeback_submitted"] == 1 and stats["evict_unwritten"] == 2
    finally:
        ssd.close(io_grace_s=5)


def test_idle_engine_services_host_ssd_persistence():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    calls = []
    fake = SimpleNamespace(
        manager=SimpleNamespace(),
        host_page_ssd=SimpleNamespace(
            has_pending=lambda: True, step=lambda: calls.append("step")
        ),
    )
    assert OffloadingConnectorScheduler.has_pending_persistence(fake)
    OffloadingConnectorScheduler.persistence_step(fake)
    assert calls == ["step"]
    fake.host_page_ssd = None
    assert not OffloadingConnectorScheduler.has_pending_persistence(fake)
