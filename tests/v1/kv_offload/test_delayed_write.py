# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Delayed write-through from the CPU tier to a secondary tier:
queue semantics, manager integration, revolta_fs options,
shutdown flush, 24 h written-bytes warning and the state_latest_only hook."""

import json
from collections.abc import Iterable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_offload.test_revolta_fs import _view, make_tier
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.delayed_write import (
    VOLUME_FILE,
    DelayedBatch,
    DelayedWriteQueue,
    WriteVolume,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

CTX = ReqContext(req_id="r")


def k(i: int, group: int = 0) -> OffloadKey:
    return make_offload_key(str(i).encode(), group)


def region(view):
    mock = MagicMock()
    mock.create_kv_memoryview.return_value = view
    return mock


class FakeTier(SecondaryTierManager):
    """Records submitted store jobs; completes them on the next poll."""

    def __init__(self, view, delay_s=None, latest_only=False, volume_path=None, warn=0):
        super().__init__(SimpleNamespace(), view, "fake")
        self.delay_s, self.latest_only = delay_s, latest_only
        self.volume_path, self.warn = volume_path, warn
        self.jobs: list[JobMetadata] = []
        self.done: list[JobResult] = []
        self.shutdown_calls = 0
        self.jobs_at_shutdown: list[JobMetadata] | None = None

    def delayed_write_config(self):
        if self.delay_s is None:
            return None
        return {
            "delay_s": self.delay_s,
            "page_bytes": 100,
            "volume_path": self.volume_path,
            "warn_bytes_24h": self.warn,
            "state_latest_only": self.latest_only,
        }

    def lookup(self, key, req_context):
        return False

    def submit_store(self, job_metadata):
        self.jobs.append(job_metadata)
        self.done.append(JobResult(job_metadata.job_id, True))

    def submit_load(self, job_metadata):
        raise AssertionError("no loads in these tests")

    def get_finished_jobs(self) -> Iterable[JobResult]:
        out, self.done = self.done, []
        return out

    def touch(self, keys, req_context):
        pass

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    def on_request_finished(self, req_context):
        pass

    def shutdown(self):
        self.shutdown_calls += 1
        self.jobs_at_shutdown = list(self.jobs)


def setup(num_blocks=4, **tier_kwargs):
    view = memoryview(torch.zeros((num_blocks, 100), dtype=torch.int8).numpy())
    primary = CPUPrimaryTierOffloadingManager(num_blocks, region(view))
    tier = FakeTier(view, **tier_kwargs)
    manager = TieringOffloadingManager(primary, [tier])
    clock = [1000.0]
    if tier in manager._delayed:
        manager._delayed[tier].clock = lambda: clock[0]
    return manager, primary, tier, clock


def store(manager, keys):
    assert manager.prepare_store(keys, CTX) is not None
    manager.complete_store(keys, CTX, success=True)


def step(manager):
    list(manager.take_events())


def written(tier):
    return [list(j.keys) for j in tier.jobs]


# ---------------------------------------------------------------- queue only


def test_queue_due_order_and_filters():
    q = DelayedWriteQueue(delay_s=300, page_bytes=10, clock=lambda: 0.0)
    q.enqueue([k(1), k(2), k(3)], CTX, now=0)
    q.enqueue([k(4)], CTX, now=100)
    assert q.depth == 4
    assert q.take_due(now=299) == []
    (batch,) = q.take_due(now=300)
    resident = {k(1), k(3)}
    live = q.select(batch, lambda key: key in resident, lambda key: key == k(3))
    assert live == [k(1)]
    assert q.stats["dropped_evicted"] == 1 and q.stats["dropped_superseded"] == 1
    assert [b.keys for b in q.take_all_newest_first()] == [[k(4)]]
    assert q.depth == 1  # select() of that batch has not run yet


def test_queue_drop_marks_all_pending_copies_then_clears():
    q = DelayedWriteQueue(delay_s=1, page_bytes=10, clock=lambda: 0.0)
    q.enqueue([k(1), k(2)], CTX, now=0)
    q.enqueue([k(1)], CTX, now=0)
    assert q.drop([k(1), k(9)]) == 1  # k(9) is not pending
    yes = lambda key: True  # noqa: E731
    no = lambda key: False  # noqa: E731
    live = [q.select(b, yes, no) for b in q.take_due(now=5)]
    assert live == [[k(2)], []]
    assert q.stats["dropped_latest"] == 2
    q.enqueue([k(1)], CTX, now=10)  # a later copy is no longer marked
    assert q.select(q.take_due(now=20)[0], yes, no) == [k(1)]


def test_queue_rejects_bad_config():
    with pytest.raises(ValueError):
        DelayedWriteQueue(delay_s=-1, page_bytes=1)
    with pytest.raises(ValueError):
        DelayedWriteQueue(delay_s=1, page_bytes=0)


def test_write_volume_rolls_persists_and_warns(tmp_path, monkeypatch):
    from vllm.v1.kv_offload.tiering import delayed_write

    warnings: list[str] = []
    monkeypatch.setattr(
        delayed_write.logger, "warning", lambda msg, *a: warnings.append(msg % a)
    )
    t = [10 * 3600.0]
    path = str(tmp_path / VOLUME_FILE)
    vol = WriteVolume(path, warn_bytes=250, clock=lambda: t[0], persist_interval_s=0)
    q = DelayedWriteQueue(delay_s=1, page_bytes=100, volume=vol)
    q.account([k(1), k(2)])  # 200 bytes
    assert q.stats["over_threshold_jobs"] == 0
    q.account([k(3)])  # 300 > 250: warn, still counted as written
    q.account([k(4)])  # within the hour: no second warning
    assert q.stats["written"] == 4
    assert (
        q.stats["over_threshold_jobs"] == 2 and q.stats["over_threshold_warnings"] == 1
    )
    assert len(warnings) == 1 and "writes continue" in warnings[0]
    # persisted; a restart sees the same count
    assert WriteVolume(path, 250, clock=lambda: t[0]).used() == 400
    # 24 h later the old hour has rolled out
    t[0] += 24 * 3600
    assert WriteVolume(path, 250, clock=lambda: t[0]).used() == 0


def test_write_volume_corrupt_file_starts_empty(tmp_path):
    path = tmp_path / VOLUME_FILE
    path.write_text("{not json")
    assert WriteVolume(str(path), 1).used() == 0
    path.write_text(json.dumps({"version": 1, "bucket_s": 60, "buckets": {}}))
    assert WriteVolume(str(path), 1).used() == 0  # other bucket size: ignored


# ------------------------------------------------------- manager integration


def test_without_config_cascade_is_immediate_and_pinned():
    manager, primary, tier, _ = setup(delay_s=None)
    store(manager, [k(1), k(2)])
    assert written(tier) == [[k(1), k(2)]]
    assert primary._policy.get(k(1)).ref_cnt == 1  # pinned until the job finishes
    step(manager)  # prepare_store already polled this step
    step(manager)
    assert primary._policy.get(k(1)).ref_cnt == 0


def test_delayed_rows_are_not_pinned_and_written_after_delay():
    manager, primary, tier, clock = setup(delay_s=300)
    store(manager, [k(1), k(2)])
    step(manager)
    assert written(tier) == []
    assert primary._policy.get(k(1)).ref_cnt == 0  # not pinned while waiting
    clock[0] += 299
    step(manager)
    assert written(tier) == []
    clock[0] += 1
    step(manager)
    assert written(tier) == [[k(1), k(2)]]
    # the written job pinned its rows like the immediate path, until done
    assert primary._policy.get(k(1)).ref_cnt == 1
    step(manager)
    assert primary._policy.get(k(1)).ref_cnt == 0
    assert manager.delayed_write_stats()["fake"]["written"] == 2


def test_superseded_rows_are_skipped_unless_touched_again():
    manager, primary, tier, clock = setup(delay_s=60)
    store(manager, [k(1), k(2), k(3)])
    manager.demote([k(1), k(2)], CTX)
    manager.touch([k(2)], CTX)  # hit again: live, written
    clock[0] += 60
    step(manager)
    assert written(tier) == [[k(2), k(3)]]
    assert manager.delayed_write_stats()["fake"]["dropped_superseded"] == 1


def test_evicted_rows_are_skipped_and_reused_slot_is_not_written_as_old_key():
    manager, primary, tier, clock = setup(num_blocks=2, delay_s=60)
    store(manager, [k(1), k(2)])
    slot = primary._policy.get(k(1)).block_id
    store(manager, [k(3)])  # LRU evicts k(1); k(3) takes its slot
    assert primary._policy.get(k(1)) is None
    assert primary._policy.get(k(3)).block_id == slot
    clock[0] += 60
    step(manager)
    jobs = tier.jobs
    assert [list(j.keys) for j in jobs] == [[k(2)], [k(3)]]
    # block ids are resolved by key at write time
    assert list(jobs[1].block_ids) == [slot]
    assert manager.delayed_write_stats()["fake"]["dropped_evicted"] == 1


def test_shutdown_submits_live_pending_newest_first_before_tier_shutdown():
    manager, primary, tier, clock = setup(delay_s=300)
    store(manager, [k(1)])
    clock[0] += 10
    store(manager, [k(2)])
    clock[0] += 10
    store(manager, [k(3)])
    manager.demote([k(2)], CTX)
    manager.shutdown()
    assert tier.shutdown_calls == 1
    assert [list(j.keys) for j in tier.jobs_at_shutdown] == [[k(3)], [k(1)]]
    assert manager.delayed_write_stats()["fake"]["dropped_superseded"] == 1


def test_drop_pending_only_on_latest_only_tiers():
    manager, primary, tier, clock = setup(delay_s=60, latest_only=False)
    assert manager.state_latest_only is False
    store(manager, [k(1)])
    assert manager.drop_pending_secondary_writes([k(1)]) == 0
    manager, primary, tier, clock = setup(delay_s=60, latest_only=True)
    assert manager.state_latest_only is True
    store(manager, [k(1), k(2)])
    assert manager.drop_pending_secondary_writes([k(1)]) == 1
    clock[0] += 60
    step(manager)
    assert written(tier) == [[k(2)]]
    assert manager.delayed_write_stats()["fake"]["dropped_latest"] == 1


def test_volume_warning_never_blocks_writes(tmp_path):
    manager, primary, tier, clock = setup(
        delay_s=1, volume_path=str(tmp_path / VOLUME_FILE), warn=150
    )
    store(manager, [k(1), k(2)])  # 2 x 100 bytes > 150
    clock[0] += 1
    step(manager)
    assert written(tier) == [[k(1), k(2)]]
    stats = manager.delayed_write_stats()["fake"]
    assert stats["over_threshold_warnings"] == 1 and stats["written_bytes_24h"] == 200


# --------------------------------------------------------------- revolta_fs


def test_revolta_defaults_and_config(tmp_path):
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    try:
        cfg = tier.delayed_write_config()
        assert cfg["delay_s"] == 300.0
        assert cfg["warn_bytes_24h"] == 10**12
        assert cfg["state_latest_only"] is False
        assert cfg["page_bytes"] == view.strides[0]
        assert cfg["volume_path"].endswith(VOLUME_FILE)
    finally:
        tier.shutdown()
    tier = make_tier(tmp_path, view, write_delay_s=0)
    try:
        assert tier.delayed_write_config() is None  # immediate cascade, unchanged
    finally:
        tier.shutdown()
    with pytest.raises(ValueError):
        make_tier(tmp_path, view, write_delay_s=-1)


def test_revolta_delayed_page_written_after_delay_and_rejected_when_corrupt(tmp_path):
    tensor, view = _view()
    tier = make_tier(tmp_path, view, write_delay_s=30, drain_timeout_s=5)
    primary = CPUPrimaryTierOffloadingManager(len(view), region(view))
    manager = TieringOffloadingManager(primary, [tier])
    clock = [0.0]
    manager._delayed[tier].clock = lambda: clock[0]
    keys = [k(11)]
    try:
        store(manager, keys)
        tensor[primary._policy.get(k(11)).block_id].fill_(7)
        step(manager)
        assert not list(tmp_path.rglob("*.bin"))  # nothing on disk yet
        clock[0] += 30
        step(manager)
        for _ in range(200):
            step(manager)
            if list(tmp_path.rglob("*.bin")):
                break
            import time

            time.sleep(0.01)
        pages = list(tmp_path.rglob("*.bin"))
        assert len(pages) == 1
    finally:
        manager.shutdown()
    # corrupt the delayed page: the tier's CRC still rejects it on read
    page = pages[0]
    data = bytearray(page.read_bytes())
    data[len(data) // 2] ^= 0xFF
    page.write_bytes(bytes(data))
    tensor2, view2 = _view()
    tier2 = make_tier(tmp_path, view2, write_delay_s=0)
    try:
        assert tier2.lookup(k(11), CTX) in (True, None)
        from tests.v1.kv_offload.test_fs_tier import drain, make_job

        tier2.submit_load(make_job(1, [k(11)], [0], is_promotion=True))
        (result,) = drain(tier2)
        assert result.success is False
        assert tier2.get_revolta_stats()["checksum_failures"] == 1
    finally:
        tier2.shutdown()


# ------------------------------------------------------ scheduler (latest only)


def test_scheduler_drops_older_own_states_and_extended_hit_state():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    sched = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    state_group = SimpleNamespace(
        group_idx=1,
        sliding_window_size_in_blocks=1,
        requires_exact_boundary_source=True,
        offloaded_block_size=16,
    )
    target_group = SimpleNamespace(
        group_idx=0,
        sliding_window_size_in_blocks=None,
        requires_exact_boundary_source=False,
        offloaded_block_size=16,
    )
    sched.config = SimpleNamespace(kv_group_configs=[target_group, state_group])
    sched._req_status = {}
    sched.manager = MagicMock()
    keys = [k(i, 1) for i in range(10)]
    req_status = SimpleNamespace(
        req=SimpleNamespace(shared_prefix_boundary=4 * 16),  # junction = block 3
        group_states={1: SimpleNamespace(offload_keys=keys, num_hit_blocks=3)},
        stored_state_blocks={1: {3, 7, 8, 9}},
    )
    sched._drop_superseded_state_writes(req_status)
    (call,) = sched.manager.drop_pending_secondary_writes.call_args_list
    # older own states 7, 8 and the extended hit state 2; junction (3) and the
    # newest (9) are kept
    assert call.args[0] == [keys[2], keys[7], keys[8]]


def test_delayed_batch_dataclass_is_plain():
    b = DelayedBatch(1.0, [k(1)], CTX, 3)
    assert b.seq == 3 and b.keys == [k(1)]


# --------------------------------------------- grouped pools (one root, N groups)


def test_shared_volume_counts_all_groups_and_warns_once(tmp_path, monkeypatch):
    from vllm.v1.kv_offload.tiering import delayed_write

    warnings: list[str] = []
    monkeypatch.setattr(
        delayed_write.logger, "warning", lambda msg, *a: warnings.append(msg % a)
    )
    path = str(tmp_path / VOLUME_FILE)
    a = WriteVolume.shared(path, 150)
    b = WriteVolume.shared(path, 150)
    assert a is b
    with pytest.raises(ValueError):
        WriteVolume.shared(path, 999)
    qa = DelayedWriteQueue(delay_s=1, page_bytes=100, volume=a)
    qb = DelayedWriteQueue(delay_s=1, page_bytes=100, volume=b)
    qa.account([k(1)])
    qb.account([k(2)])  # 200 > 150 across both groups
    qa.account([k(3)])
    assert a.used() == 300
    assert len(warnings) == 1
    assert qb.stats["over_threshold_warnings"] == 1
    assert (
        qa.stats["over_threshold_jobs"] == 1
        and qa.stats["over_threshold_warnings"] == 0
    )


def test_grouped_manager_forwards_latest_only_drop_and_stats():
    from vllm.v1.kv_offload.cpu.manager import GroupedCPUOffloadingManager

    groups = {}
    clocks = {}
    for g in (0, 1):
        manager, primary, tier, clock = setup(delay_s=60, latest_only=(g == 1))
        groups[g], clocks[g] = (manager, tier), clock
    grouped = GroupedCPUOffloadingManager({g: m for g, (m, _) in groups.items()})
    assert grouped.state_latest_only is True
    store(groups[0][0], [k(1, 0)])
    store(groups[1][0], [k(2, 1), k(3, 1)])
    # only group 1 is latest-only
    assert grouped.drop_pending_secondary_writes([k(1, 0), k(2, 1)]) == 1
    for g in (0, 1):
        clocks[g][0] += 60
    list(grouped.take_events())
    assert written(groups[0][1]) == [[k(1, 0)]]
    assert written(groups[1][1]) == [[k(3, 1)]]
    stats = grouped.delayed_write_stats()
    assert stats["fake[group 1]"]["dropped_latest"] == 1
    assert stats["fake[group 0]"]["written"] == 1


# ------------------------------------------- idle engine (no steps) persistence


def test_idle_step_writes_due_rows_and_releases_pins_without_engine_steps():
    manager, primary, tier, clock = setup(delay_s=60)
    assert manager.has_pending_delayed_writes() is False
    store(manager, [k(1)])
    assert manager.has_pending_delayed_writes() is True  # queued
    manager.delayed_write_step()  # not due yet
    assert written(tier) == []
    clock[0] += 60
    manager.delayed_write_step()  # due: written, CPU row pinned until polled
    assert written(tier) == [[k(1)]]
    assert primary._policy.get(k(1)).ref_cnt == 1
    assert manager.has_pending_delayed_writes() is True  # job in flight
    manager.delayed_write_step()  # completion polled
    assert primary._policy.get(k(1)).ref_cnt == 0
    assert manager.has_pending_delayed_writes() is False


def test_no_delayed_tier_never_reports_pending():
    manager, primary, tier, clock = setup(delay_s=None)
    store(manager, [k(1)])  # immediate cascade in flight
    assert manager.has_pending_delayed_writes() is False


def test_scheduler_forwards_persistence_hooks():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    sched = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    calls = []
    sched.manager = SimpleNamespace(
        has_pending_delayed_writes=lambda: True,
        delayed_write_step=lambda: calls.append("step"),
    )
    assert sched.has_pending_persistence() is True
    sched.persistence_step()
    assert calls == ["step"]
    sched.manager = SimpleNamespace()  # plain CPU manager: no delayed writes
    assert sched.has_pending_persistence() is False
    sched.persistence_step()


@pytest.mark.parametrize("pending", [True, False])
def test_idle_engine_bounds_wait_only_with_pending_writes(pending, monkeypatch):
    import queue
    from contextlib import nullcontext

    from vllm.v1.engine import core

    observed, steps = [], []

    class Input:
        def empty(self):
            return True

        def get(self, **kwargs):
            observed.append(kwargs)
            raise queue.Empty

    connector = SimpleNamespace(
        has_pending_persistence=lambda: pending,
        persistence_step=lambda: steps.append(1),
    )
    monkeypatch.setattr(
        core,
        "get_watch_dog",
        lambda: SimpleNamespace(disable=nullcontext, feed=lambda: None),
    )
    instance = SimpleNamespace(
        has_work=lambda: False,
        is_running=lambda: True,
        _notify_idle_state_callbacks=lambda: None,
        input_queue=Input(),
        aborts_queue=queue.Queue(),
        process_input_queue_block=True,
        scheduler=SimpleNamespace(
            get_kv_connector=lambda: connector,
            get_idle_stats=lambda: (None, None),
        ),
    )
    core.EngineCoreProc._process_input_queue(instance)
    if pending:
        assert observed == [{"block": True, "timeout": 1.0}] and steps == [1]
    else:
        assert observed == [{"block": True, "timeout": None}] and steps == []
