# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped store jobs retain per-page timeout, pin and backpressure rules."""

import os
import threading
import time

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
