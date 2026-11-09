# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""value arbitration arbiter on the shared hot region shared region: the pool adapter,
eviction signals and
the scheduler wiring (CPU only)."""

from vllm.v1.core.value_arbiter import (
    HOT_FORCED_RECENT,
    HOT_REFETCH,
    NUM_HOT_COUNTERS,
    ArbiterConfig,
    ValueArbiter,
)

from .test_history_shared_region import K, cache, pool


def cached_pool(n=5, blocks=9, target=None):
    p = pool(blocks=blocks, target=target)
    p.start_step(1)
    taken = p.get_new_blocks(n)
    for i, block in enumerate(taken):
        cache(p, f"r{i}", [block], bytes([65 + i]))
    p.free_blocks(taken)
    p.finish_roles()
    return p, taken


def test_snapshot_counts_cached_donated_and_in_use():
    p, _ = cached_pool(n=5)
    snap = p.arbiter_snapshot()
    assert (snap.num_ids, snap.cached, snap.floor_blocks) == (8, 5, 0)
    assert snap.inuse == 0 and snap.upper == 8
    p.start_step(2)
    p.set_hot_floor(2 * K)
    held = p.get_new_blocks(1)
    snap = p.arbiter_snapshot()
    assert snap.inuse == 1 and snap.floor_blocks == 2
    assert snap.upper == 8 - 1 - 2
    p.free_blocks(held)


def test_value_donation_gives_lru_cached_ids_to_the_hot_region():
    p, taken = cached_pool(n=3)
    seen: list[bytes] = []
    p.value_observer = seen.append
    donated = p.num_donated()
    assert p.donate_cached_for_value(2) == 2
    assert p.num_donated() == donated + 2 and p.num_cached_free() == 1
    assert p.stats["cached_evicted_for_value"] == 2
    assert seen == [b"A" * 32, b"B" * 32]
    roles = p.finish_roles()
    assert sorted(roles.donate) == sorted(b.block_id for b in taken[:2])
    assert p.donate_cached_for_value(5) == 1  # only what is cached


def test_every_cached_eviction_reaches_the_observer():
    p, _ = cached_pool(n=2, target=0)
    seen: list[bytes] = []
    p.value_observer = seen.append
    p.start_step(2)
    p.get_new_blocks(1)  # cached above T_idx=0 goes first
    assert seen == [b"A" * 32]
    assert p.stats["cached_evicted_for_allocation"] == 1


def _arbiter(p):
    config = ArbiterConfig(bw_uva=1.0, bw_h2d=1.0, w_async=1.0, w_ttft=0.0)
    arb = ValueArbiter(
        config,
        p,
        default_target=p.index_cache_target,
        slot_bytes=1,
        index_block_bytes=1,
        forced_counts=lambda: p.stats,
    )
    p.value_observer = arb.on_index_evicted
    return arb


def test_arbiter_moves_the_pool_target_and_donates_at_most_q():
    p, _ = cached_pool(n=6, blocks=9, target=6)
    arb = _arbiter(p)
    assert arb.q == 1 and p.index_cache_target == 6
    step = 1
    for window in range(4):
        for i in range(arb.config.window_steps):
            step += 1
            arb.begin_step(step, 0.01)
            values = [0] * NUM_HOT_COUNTERS
            if i == 0:
                values[HOT_FORCED_RECENT], values[HOT_REFETCH] = 10, 10
            arb.add_hot_counters(step, values)
            before = p.num_cached_free()
            arb.end_step(step)
            assert before - p.num_cached_free() <= arb.q
            p.finish_roles()
    lowered = [d for d in arb.decisions if d.action == "lower"]
    assert len(lowered) == 3  # the fourth window is still open
    assert p.index_cache_target == 6 - 3
    assert p.stats["cached_evicted_for_value"] == 3
    assert arb.windows  # the open window keeps collecting


def test_scheduler_wiring_on_and_off():
    from .test_history_shared_region import _host_scheduler

    on = _host_scheduler(max_num_seqs=4)
    off = _host_scheduler(max_num_seqs=4, value_arbitration=False)
    for scheduler, enabled in ((on, True), (off, False)):
        pool = scheduler.host_history_pool
        plan = scheduler.kv_cache_manager.admission_ledger.plan
        assert pool is not None and pool.elastic
        assert (scheduler.value_arbiter is not None) == enabled
        assert plan.value_shadow == enabled
        assert (pool.value_observer is not None) == enabled
        assert scheduler.connector is None  # the restore hook needs one
    # Off keeps the shared hot region plan; on bills one shadow byte per four-token
    # group.
    p_on = on.kv_cache_manager.admission_ledger.plan
    p_off = off.kv_cache_manager.admission_ledger.plan
    assert (
        p_on.history_bytes_per_block - p_off.history_bytes_per_block
        == p_on.geometry.local_block_tokens // 4
    )
    assert off.host_history_pool.index_cache_target == (
        (
            off.host_history_pool.num_gpu_blocks
            - 1
            - off.host_history_pool.floor_blocks()
        )
        // 2
    )


def test_scheduler_steps_carry_epochs_and_consume_rank_counters():
    from tests.v1.core.test_long_context_admission import make_requests
    from tests.v1.core.test_scheduler_kv_capacity import step
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.v1.outputs import KVConnectorOutput

    from .test_history_shared_region import _host_scheduler

    s = _host_scheduler(max_num_seqs=4, value_arbiter_settings={"window_steps": 4})
    arb = s.value_arbiter
    all_ranks = s.host_history_pool.host_pages.all_ranks
    for request in make_requests(4, 3 * 1664 + 5, 40, tag="v"):
        s.add_request(request)
    epochs = []
    for _ in range(30):
        output = s.schedule()
        epochs.append(output.host_kv.arbiter_epoch)
        s.schedule = lambda output=output: output
        step(s)
        del s.schedule
        values = [0] * NUM_HOT_COUNTERS
        values[HOT_FORCED_RECENT], values[HOT_REFETCH] = 4, 4
        meta = OffloadingWorkerMetadata(
            value_counters={output.host_kv.step: (all_ranks, tuple(values))}
        )
        s.update_host_kv_output(KVConnectorOutput(kv_connector_worker_meta=meta))
    assert epochs == sorted(epochs) and epochs[-1] >= 5
    assert arb.decisions
    # Every evaluated window had all counters (hot side observed).
    assert all(d.reason != "counters_missing" for d in arb.decisions)
    assert all(d.value_donated <= arb.q for d in arb.decisions)
    snap = s.host_history_pool.arbiter_snapshot()
    assert 0 <= arb.t_idx <= max(snap.upper, arb.t_idx)
