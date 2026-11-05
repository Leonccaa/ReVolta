# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped jobs retain deadline, CPU pin and physical backpressure rules."""

import errno
import gc
import os
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_fs_tier import drain, key, make_job
from tests.v1.kv_offload.test_revolta_fs import _view, make_tier
from vllm.v1.kv_offload.tiering.fs import revolta


def test_backpressure_counts_blocks_within_one_group(tmp_path, monkeypatch):
    tensor, view = _view()
    tier = make_tier(tmp_path, view, max_inflight_store_blocks=3, store_timeout_s=0)
    entered, release = threading.Event(), threading.Event()
    real_write = tier._write_page

    def block_head(k, path, bid):
        if k == key(1):
            entered.set()
            assert release.wait(5)
        real_write(k, path, bid)

    monkeypatch.setattr(tier, "_write_page", block_head)
    try:
        tensor.fill_(1)
        tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
        assert entered.wait(5)
        assert tier._inflight_stores == 2
        tier.submit_store(make_job(2, [key(3), key(4)], [2, 3]))
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(2, True)]
        assert tier.get_revolta_stats()["store_blocks_skipped_backpressure"] == 2
        release.set()
        assert [(r.job_id, r.success) for r in drain(tier)] == [(1, True)]
        assert tier._inflight_stores == 0
        assert [tier.lookup(key(i)) for i in range(1, 5)] == [True, True, False, False]
    finally:
        release.set()
        tier.shutdown()


def test_expiry_between_group_pages_cancels_remaining_sources(tmp_path, monkeypatch):
    tensor, view = _view()
    tier = make_tier(tmp_path, view, store_timeout_s=0.05)
    entered, release, done = (threading.Event() for _ in range(3))
    real_sync, real_group = revolta._fsync_dir, tier._store_group

    def block_after_snapshot(path):
        entered.set()
        assert release.wait(5)
        real_sync(path)

    def group(pages):
        try:
            real_group(pages)
        finally:
            done.set()

    monkeypatch.setattr(revolta, "_fsync_dir", block_after_snapshot)
    monkeypatch.setattr(tier, "_store_group", group)
    try:
        tensor.fill_(1)
        tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
        assert entered.wait(5)
        time.sleep(0.08)
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(1, False)]
        # CPU slots may now be reused; page two must never snapshot this data.
        tensor[1].fill_(9)
        release.set()
        assert done.wait(5)
        assert list(tier.get_finished_jobs()) == []
        assert not os.path.exists(tier.file_mapper.get_file_name(key(2)))
        assert tier.get_revolta_stats()["store_tasks_cancelled"] == 1
        assert tier._inflight_stores == 0
    finally:
        release.set()
        tier.shutdown()


def test_deadline_remains_armed_after_second_page_snapshot(tmp_path, monkeypatch):
    tensor, view = _view()
    tier = make_tier(tmp_path, view, store_timeout_s=0.1)
    in_snapshot, release_snapshot, after_snapshot, release_io = (
        threading.Event() for _ in range(4)
    )
    real_crc, real_sync = revolta.crc32c, revolta._fsync_dir
    crc_calls, sync_calls = 0, 0

    def block_second_snapshot(payload, crc=0):
        nonlocal crc_calls
        crc_calls += 1
        if crc_calls == 2:
            in_snapshot.set()
            assert release_snapshot.wait(5)
        return real_crc(payload, crc)

    def block_second_sync(path):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            after_snapshot.set()
            assert release_io.wait(5)
        real_sync(path)

    monkeypatch.setattr(revolta, "crc32c", block_second_snapshot)
    monkeypatch.setattr(revolta, "_fsync_dir", block_second_sync)
    try:
        tensor.fill_(1)
        tier.submit_store(make_job(1, [key(1), key(2), key(3)], [0, 1, 2]))
        assert in_snapshot.wait(5)
        time.sleep(0.15)
        assert list(tier.get_finished_jobs()) == []
        release_snapshot.set()
        assert after_snapshot.wait(5)
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(1, False)]
        release_io.set()
        assert drain(tier) == []
        assert crc_calls == 2  # the third source was never touched
        assert not os.path.exists(tier.file_mapper.get_file_name(key(3)))
    finally:
        release_snapshot.set()
        release_io.set()
        tier.shutdown()


def test_drain_deadline_stops_active_group_between_pages(tmp_path, monkeypatch):
    tensor, view = _view()
    tier = make_tier(tmp_path, view, store_timeout_s=0, drain_timeout_s=0.05)
    entered, release = threading.Event(), threading.Event()
    real_write = tier._write_page

    def block_head(k, path, bid):
        if k == key(1):
            entered.set()
            assert release.wait(5)
        real_write(k, path, bid)

    monkeypatch.setattr(tier, "_write_page", block_head)
    closer = threading.Thread(target=tier.shutdown)
    try:
        tensor.fill_(1)
        tier.submit_store(make_job(1, [key(1), key(2), key(3)], [0, 1, 2]))
        assert entered.wait(5)
        closer.start()
        deadline = time.monotonic() + 5
        while not tier._pool.stopping and time.monotonic() < deadline:
            time.sleep(0.005)
        assert tier._pool.stopping
        assert closer.is_alive()  # the active page still owns its CPU memory
        release.set()
        closer.join(5)
        assert not closer.is_alive()
        assert os.path.exists(tier.file_mapper.get_file_name(key(1)))
        assert not os.path.exists(tier.file_mapper.get_file_name(key(2)))
        assert not os.path.exists(tier.file_mapper.get_file_name(key(3)))
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(5)
        tier.shutdown()


@pytest.mark.parametrize("limit", [1, 3])
def test_timeout_cannot_bypass_physical_backpressure(tmp_path, monkeypatch, limit):
    from vllm.v1.kv_offload.tiering.fs import thread_pool

    tensor, view = _view()
    tier = make_tier(
        tmp_path,
        view,
        n_read_threads=0,
        n_write_threads=1,
        max_inflight_store_blocks=limit,
        store_timeout_s=1,
        drain_timeout_s=0,
    )
    entered, release = threading.Event(), threading.Event()
    clock = [0.0]
    monkeypatch.setattr(
        thread_pool, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    real_sync, real_enqueue = revolta._fsync_dir, tier._pool.enqueue_store
    references: list[weakref.ReferenceType] = []

    def block(path):
        entered.set()
        assert release.wait(10)
        real_sync(path)

    def record(*args, **kwargs):
        references.extend(weakref.ref(task) for task in args[2])
        return real_enqueue(*args, **kwargs)

    monkeypatch.setattr(revolta, "_fsync_dir", block)
    monkeypatch.setattr(tier._pool, "enqueue_store", record)
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert entered.wait(5)
        for job in range(1, 13):
            if job > 1:
                tier.submit_store(make_job(job, [key(job)], [0]))
            clock[0] += 2
            result = list(tier.get_finished_jobs())
            assert [(r.job_id, r.success) for r in result] == [(job, job > limit)]
            # Logical completion releases CPU ownership independently.
            assert tier._inflight_stores == 0 and not tier._inflight_jobs
            tensor.fill_(job)
            gc.collect()
            assert len(tier._pool._store_q) <= limit - 1
            assert sum(ref() is not None for ref in references) <= limit
            assert tier._pool.pending_store_blocks <= limit
            assert tier._cache._reserved <= tier._block_size
        assert tier.get_revolta_stats()["store_jobs_skipped_backpressure"] == 12 - limit
        release.set()
        tier.shutdown()
        assert tier._pool.pending_store_blocks == 0
        assert tier._cache._reserved == 0 and not tier._cache._pins
        assert list(tier.get_finished_jobs()) == []
    finally:
        release.set()
        tier.shutdown()


def test_retry_backoff_uses_remaining_monotonic_budget(tmp_path, monkeypatch):
    from vllm.v1.kv_offload.tiering.fs import thread_pool

    tensor, view = _view()
    tier = make_tier(
        tmp_path,
        view,
        max_io_retries=10,
        drain_timeout_s=0,
        n_read_threads=0,
        n_write_threads=1,
    )
    clock = [0.0]
    waits, calls = [], []
    monkeypatch.setattr(
        thread_pool, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    state = thread_pool.JobState(1, 1, deadline=0.035)
    monkeypatch.setattr(thread_pool._current, "state", state, raising=False)

    real_wait = tier._pool._condition.wait_for

    def wait(predicate, timeout=None):
        if timeout is None:
            return real_wait(predicate)
        waits.append(timeout)
        clock[0] += timeout
        return predicate()

    def fail():
        calls.append(clock[0])
        raise OSError(errno.EAGAIN, "retry")

    monkeypatch.setattr(tier._pool._condition, "wait_for", wait)
    try:
        with pytest.raises(revolta.StoreCancelled):
            tier._with_retries(fail)
        assert waits == pytest.approx([0.01, 0.025])
        assert calls == [0.0, 0.01]
        assert clock[0] == pytest.approx(state.deadline)
        # Deadline alone prevents the next attempt, without completion polling.
        with pytest.raises(revolta.StoreCancelled):
            tier._with_retries(fail)
        assert len(calls) == 2
    finally:
        monkeypatch.setattr(thread_pool._current, "state", None)
        tier.shutdown()


def test_retry_backoff_is_capped_without_job_deadline(tmp_path, monkeypatch):
    from vllm.v1.kv_offload.tiering.fs import thread_pool

    tensor, view = _view()
    tier = make_tier(
        tmp_path,
        view,
        max_io_retries=10,
        drain_timeout_s=0,
        n_read_threads=0,
        n_write_threads=1,
    )
    clock = [0.0]
    waits = []
    monkeypatch.setattr(
        thread_pool, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )

    real_wait = tier._pool._condition.wait_for

    def wait(predicate, timeout=None):
        if timeout is None:
            return real_wait(predicate)
        waits.append(timeout)
        clock[0] += timeout
        return predicate()

    def fail():
        raise OSError(errno.EAGAIN, "retry")

    monkeypatch.setattr(tier._pool._condition, "wait_for", wait)
    try:
        with pytest.raises(OSError):
            tier._with_retries(fail)
        assert waits == pytest.approx([0.01, 0.04, 0.16, 0.64] + [1.0] * 6)
        assert max(waits) <= 1.0 and sum(waits) < 7
    finally:
        tier.shutdown()


def test_shutdown_wakes_retry_backoff(tmp_path, monkeypatch):
    tensor, view = _view()
    tier = make_tier(
        tmp_path, view, max_io_retries=100, store_timeout_s=0, drain_timeout_s=0
    )
    waiting = threading.Event()
    real_wait = tier._pool.wait_for_retry
    calls = []

    def wait(delay):
        waiting.set()
        return real_wait(1.0)

    def fail(*args):
        calls.append(True)
        raise OSError(errno.EAGAIN, "retry")

    monkeypatch.setattr(tier._pool, "wait_for_retry", wait)
    monkeypatch.setattr(tier, "_write_page", fail)
    closer = threading.Thread(target=tier.shutdown)
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert waiting.wait(5)
        closer.start()
        closer.join(0.5)
        assert not closer.is_alive()
        assert len(calls) == 1
        assert tier._pool.pending_store_blocks == 0
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(1, False)]
    finally:
        tier._pool.shutdown()
        if closer.ident is not None:
            closer.join(5)
        tier.shutdown()


def test_transient_read_error_is_retried_without_losing_page(tmp_path, monkeypatch):
    from tests.v1.kv_offload.test_revolta_fs import _load, _store

    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    path = _store(tier, tensor, key(1), value=7)
    real_readv = os.readv
    calls = []

    def readv(fd, buffers):
        calls.append(True)
        if len(calls) == 1:
            raise OSError(errno.EAGAIN, "transient read")
        return real_readv(fd, buffers)

    monkeypatch.setattr(os, "readv", readv)
    try:
        assert _load(tier, key(1)) == [True]
        assert len(calls) == 2 and os.path.exists(path)
        assert tier.get_revolta_stats()["io_retries"] == 1
        assert tensor[1].eq(7).all()
    finally:
        tier.shutdown()


def test_expiry_wakes_retry_backoff(tmp_path, monkeypatch):
    from vllm.v1.kv_offload.tiering.fs import thread_pool

    tensor, view = _view()
    tier = make_tier(
        tmp_path,
        view,
        n_read_threads=0,
        n_write_threads=1,
        max_io_retries=100,
        store_timeout_s=10,
        drain_timeout_s=0,
    )
    clock = [0.0]
    waiting, finished = threading.Event(), threading.Event()
    monkeypatch.setattr(
        thread_pool, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    real_wait, real_group = tier._pool.wait_for_retry, tier._store_group
    calls = []

    def wait(delay):
        waiting.set()
        return real_wait(1.0)

    def fail(*args):
        calls.append(True)
        raise OSError(errno.EAGAIN, "retry")

    def group(pages):
        try:
            real_group(pages)
        finally:
            finished.set()

    monkeypatch.setattr(tier._pool, "wait_for_retry", wait)
    monkeypatch.setattr(tier, "_write_page", fail)
    monkeypatch.setattr(tier, "_store_group", group)
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert waiting.wait(5)
        clock[0] = 11
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(1, False)]
        assert finished.wait(0.5)
        assert len(calls) == 1
        assert drain(tier) == []
    finally:
        tier.shutdown()


def test_expired_physical_job_does_not_retain_cpu_pins(tmp_path, monkeypatch):
    from tests.v1.kv_offload.test_tiering_offloading import _mock_mmap_region
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.fs import thread_pool
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor, view = _view()
    tier = make_tier(
        tmp_path,
        view,
        n_read_threads=0,
        n_write_threads=1,
        max_inflight_store_blocks=1,
        store_timeout_s=1,
        drain_timeout_s=0,
    )
    primary = CPUPrimaryTierOffloadingManager(2, _mock_mmap_region(2))
    manager = TieringOffloadingManager(primary, [tier])
    clock = [0.0]
    monkeypatch.setattr(
        thread_pool, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    entered, release = threading.Event(), threading.Event()
    real_sync = revolta._fsync_dir

    def block(path):
        entered.set()
        assert release.wait(10)
        real_sync(path)

    monkeypatch.setattr(revolta, "_fsync_dir", block)
    ctx = ReqContext(req_id="timeout-test")
    try:
        for i in range(1, 9):
            manager.prepare_store([key(i)], ctx)
            manager.complete_store([key(i)], ctx, success=True)
            assert primary._policy.get(key(i)).ref_cnt == 1
            if i == 1:
                assert entered.wait(5)
            clock[0] += 2
            manager._process_finished_jobs()
            assert primary._policy.get(key(i)).ref_cnt == 0
            assert not manager._transfer_jobs
            assert tier._pool.pending_store_blocks == 1
            assert not tier._pool._store_q
        assert tier.get_revolta_stats()["store_jobs_skipped_backpressure"] == 7
    finally:
        release.set()
        manager.shutdown()
