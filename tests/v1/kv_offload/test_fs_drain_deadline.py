# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real blocked filesystem threads respect the supplied shutdown deadline."""

import threading
import time

import pytest

from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool


def test_shutdown_deadline_retains_active_thread_until_retry():
    entered = threading.Event()
    release = threading.Event()

    def blocked():
        entered.set()
        release.wait(5)

    pool = DualQueueThreadPool(1, 0, drain_timeout_s=10)
    try:
        pool.enqueue_store(1, 1, [blocked])
        assert entered.wait(2)
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="IO still owns backing"):
            pool.shutdown(deadline=start + 0.05)
        assert time.monotonic() - start < 1
        assert any(t.is_alive() for t in pool._threads)
    finally:
        release.set()
        pool.shutdown(deadline=time.monotonic() + 2)
    assert all(not t.is_alive() for t in pool._threads)


def test_tiering_timeout_keeps_primary_backing_until_all_tiers_retire():
    from types import SimpleNamespace

    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

    calls = []
    retired = False

    def drain():
        calls.append("secondary")
        if not retired:
            raise RuntimeError("IO still active")

    manager = TieringOffloadingManager.__new__(TieringOffloadingManager)
    manager._delayed = {}
    manager.secondary_tiers = [
        SimpleNamespace(
            set_shutdown_deadline=lambda d: calls.append("deadline"), shutdown=drain
        )
    ]
    manager.primary_tier = SimpleNamespace(shutdown=lambda: calls.append("primary"))
    manager.set_shutdown_deadline(time.monotonic() + 2)
    with pytest.raises(RuntimeError, match="IO still active"):
        manager.shutdown()
    assert calls == ["deadline", "secondary"]
    retired = True
    manager.shutdown()
    assert calls[-3:] == ["deadline", "secondary", "primary"]
