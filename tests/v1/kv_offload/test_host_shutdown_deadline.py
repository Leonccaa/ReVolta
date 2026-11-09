# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One monotonic budget across request drain, worker control and tiering IO."""

import time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

import pytest


@pytest.mark.parametrize("timeout", [0, 3, 30])
def test_request_time_is_shared_and_repeated_signals_do_not_extend_it(
    monkeypatch, timeout
):
    from vllm.v1.engine import core

    now = [100.0]
    monkeypatch.setattr(core.time, "monotonic", lambda: now[0])
    engine = core.EngineCoreProc.__new__(core.EngineCoreProc)
    engine.vllm_config = SimpleNamespace(shutdown_timeout=timeout)
    engine.shutdown_state = core.EngineShutdownState.RUNNING
    assert engine.shutdown_deadline() is None
    engine.request_shutdown()
    assert engine.shutdown_state == core.EngineShutdownState.REQUESTED
    assert engine.shutdown_deadline() == 100 + timeout - 5
    now[0] += 12
    engine.request_shutdown()
    assert engine._shutdown_requested_at == 100
    assert engine.shutdown_deadline() == 100 + timeout - 5


def test_tiering_uses_scheduler_deadline_and_never_unlinks_live_io():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_offload.host.region import HostRegionLayout, HostRegionOwner
    from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
    from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

    layout = HostRegionLayout(uuid4().hex, 4, 1, 128, ())
    owner = HostRegionOwner(layout)
    region = layout.open(0)
    gate, entered, finished = Event(), Event(), Event()
    released = []
    errors = []
    tier = FileSystemTierManager.__new__(FileSystemTierManager)
    tier._pool = DualQueueThreadPool(0, 1, drain_timeout_s=30)
    tier._cache = SimpleNamespace(release=lambda: released.append("cache"))
    tier._closed = False
    tier.tier_type = "fs"

    def io():
        entered.set()
        gate.wait()

    tier._pool.enqueue_store(1, 1, [io])
    assert entered.wait(1)
    manager = TieringOffloadingManager.__new__(TieringOffloadingManager)
    manager._delayed = {}
    manager.secondary_tiers = [tier]
    manager.primary_tier = SimpleNamespace(shutdown=lambda: released.append("primary"))
    connector_scheduler = OffloadingConnectorScheduler.__new__(
        OffloadingConnectorScheduler
    )
    connector_scheduler.manager = manager
    connector = OffloadingConnector.__new__(OffloadingConnector)
    connector.connector_worker = None
    connector.connector_scheduler = connector_scheduler
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_event_publisher = scheduler.ec_connector = None
    scheduler.connector = connector
    scheduler.host_region = owner
    deadline = time.monotonic() + 0.1

    def shutdown():
        try:
            scheduler.shutdown(deadline=deadline)
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    thread = Thread(target=shutdown, daemon=True)
    thread.start()
    try:
        assert finished.wait(1), "tier drain hung beyond the shared deadline"
        assert errors and isinstance(errors[0], TimeoutError)
        assert manager._shutdown_deadline == tier._shutdown_deadline == deadline
        assert not released and not tier._closed
        assert scheduler.host_region is owner
        assert Path(owner.mmap_path).exists()
        assert region.fd is not None and region.mmap_obj is not None
    finally:
        gate.set()
        thread.join(2)
        tier._pool.shutdown()
        tier.shutdown()
        region.cleanup()
        owner.cleanup()
