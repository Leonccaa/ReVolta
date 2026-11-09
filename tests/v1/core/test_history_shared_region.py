# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""shared hot region shared region on the scheduler thread: ID roles, the hot floor,
step
role updates, HOST reclaim of cached bindings and pass plans."""

from types import SimpleNamespace

import pytest

from vllm.models.qwen4_exp.nvidia.host_kv.planning import (
    plan_passes,
    plan_step_passes,
)
from vllm.v1.core.history_block_pool import HistoryBlockPool
from vllm.v1.core.long_context_kv import (
    GPU_HISTORY,
    HOST,
    HOT,
    LongContextAdmissionLedger,
)

K = 4


def pool(blocks=9, pages=9, floor=0, target=None):
    p = HistoryBlockPool(blocks, True, 16, False)
    p.configure_host(pages, 1, 0)
    p.admission_observer = SimpleNamespace(
        _scope=(HOST,), on_allocated=lambda _: None, on_released=lambda _: None
    )
    p.configure_elastic(K, min_floor_slots=floor, index_cache_target=target)
    return p


def request(name, digest):
    return SimpleNamespace(request_id=name, block_hashes=[digest * 32])


def cache(p, name, blocks, digest=b"a"):
    p.cache_full_blocks(request(name, digest), blocks, 0, 1, 16, 0)


def retire(p, step):
    p.host_pages.complete_step(step, p.host_pages.all_ranks)


def test_initial_roles_every_non_null_id_donated_and_floor_reserved():
    p = pool(floor=10)  # ceil(10 / 4) = 3 IDs
    assert p.num_donated() == 8 and p.hot_slots() == 32
    assert p.get_num_free_blocks() == 8 - 3
    assert p.finish_roles().donate == []  # the worker starts in this state
    with pytest.raises(ValueError, match="above the hot floor"):
        p.get_new_blocks(6)


def test_allocation_claims_and_free_donates_with_net_step_updates():
    p = pool()
    p.start_step(1)
    a = p.get_new_blocks(2)
    roles = p.finish_roles()
    assert roles.claim == sorted(b.block_id for b in a) and roles.donate == []
    assert roles.hot_slots == 6 * K
    # Allocated and freed within one step: no role change reaches the worker,
    # and the new ID is never zeroed (its page belongs to the hot region).
    p.start_step(2)
    b = p.get_new_blocks(1)
    p.free_blocks(b)
    roles = p.finish_roles()
    assert roles.claim == [] and roles.donate == []
    assert p.filter_zero_ids([b[0].block_id]) is None
    # Freed without a hash: donated at the next step head.
    p.free_blocks(a)
    assert sorted(p.finish_roles().donate) == sorted(x.block_id for x in a)
    assert p.hot_slots() == 8 * K


def test_cached_ids_keep_reserved_slots_and_survive_short_request_churn():
    # Index-cache target 2: a cached prefix below the target is kept while
    # DONATED IDs above the floor exist (short requests churn those).
    p = pool(target=2)
    p.start_step(1)
    prefix = p.get_new_blocks(1)
    cache(p, "sys", prefix, b"s")
    p.release_request("sys", prefix)
    p.free_blocks(prefix)
    roles = p.finish_roles()
    assert prefix[0].block_id not in roles.donate  # CACHED keeps owner -2
    for step in range(2, 40):
        p.start_step(step)
        short = p.get_new_blocks(1)  # a 128-token request: one partial block
        assert short[0].block_id != prefix[0].block_id
        p.free_blocks(short)
        p.finish_roles()
    assert p.get_cached_block(b"s" * 32, [0]) == prefix
    assert p.stats["cached_evicted_for_allocation"] == 0
    assert p.hot_slots() == 7 * K  # every idle ID except the prefix is hot


def test_cached_above_target_evicted_before_hot_region_shrinks():
    p = pool(target=0)
    p.start_step(1)
    blocks = p.get_new_blocks(2)
    cache(p, "a", blocks[:1], b"x")
    cache(p, "b", blocks[1:], b"y")
    p.free_blocks(blocks)
    p.finish_roles()
    donated = p.num_donated()
    [taken] = p.get_new_blocks(1)
    assert taken.block_id == blocks[0].block_id  # LRU cached first
    assert p.num_donated() == donated
    assert p.stats["cached_evicted_for_allocation"] == 1


def test_floor_growth_donates_cached_ids_and_never_goes_below_floor():
    p = pool(blocks=6, target=10)
    p.start_step(1)
    blocks = p.get_new_blocks(5)
    p.finish_roles()
    for i, block in enumerate(blocks):
        cache(p, f"r{i}", [block], bytes([65 + i]))
    p.free_blocks(blocks)
    assert p.num_donated() == 0
    p.set_hot_floor(6)  # 2 IDs
    assert p.num_donated() == 2 and p.stats["cached_evicted_for_floor"] == 2
    roles = p.finish_roles()
    assert len(roles.donate) == 2 and roles.hot_slots == 8
    assert p.get_num_free_blocks() == 3
    p.get_new_blocks(3)
    with pytest.raises(ValueError):
        p.get_new_blocks(1)


def test_in_flight_transfer_holds_the_id_out_of_the_hot_region():
    p = pool()
    p.start_step(1)
    [block] = p.get_new_blocks(1)
    p.finish_roles()
    p.track_job(5, [block.block_id])
    p.free_blocks([block])
    roles = p.finish_roles()
    assert block.block_id not in roles.donate and block.block_id in p.held
    assert p.get_num_free_blocks() == 7
    p.complete_jobs({5: 1})
    assert block.block_id in p.finish_roles().donate


def test_draining_target_donated_only_after_its_accesses_retire():
    p = pool()
    p.start_step(1)
    [block] = p.get_new_blocks(1)
    p.finish_roles()
    p.mark_draining([block.block_id], retire_after=3)
    p.free_blocks([block])
    assert block.block_id not in p.finish_roles().donate
    retire(p, 1)
    retire(p, 2)
    assert block.block_id not in p.finish_roles().donate
    retire(p, 3)
    assert block.block_id in p.finish_roles().donate


def _cached_page_pool():
    # Two host pages besides ZERO_PAGE; two GPU IDs cached on them.
    p = pool(blocks=6, pages=3, target=10)
    p.start_step(1)
    blocks = p.get_new_blocks(2)
    cache(p, "a", blocks[:1], b"a")
    cache(p, "b", blocks[1:], b"b")
    for ticket in p.finish_schedule(["a", "b"]):
        p.host_pages.seal_result(ticket, 1, 0)
    p.release_request("a", blocks[:1])
    p.release_request("b", blocks[1:])
    p.free_blocks(blocks)
    p.finish_roles()
    return p, blocks


def test_full_host_pool_of_retired_cached_bindings_still_admits_new_request():
    p, blocks = _cached_page_pool()
    assert p.host_pages.available() == 0
    assert p.host_reclaimable_after() == 0  # step 1 is not retired yet
    retire(p, 1)
    assert p.host_reclaimable_after() == 2
    [fresh] = p.get_new_blocks(1)
    # The victim ID lost its hash and became DONATED; its page went to `fresh`.
    victim = next(b for b in blocks if b.block_hash is None)
    assert p.stats["cached_evicted_for_host"] == 1
    assert fresh.block_id in p.bindings
    assert victim.block_id not in p.bindings
    roles = p.finish_roles()
    assert victim.block_id in roles.donate and fresh.block_id in roles.claim


def test_hit_pages_and_pages_with_io_are_not_reclaimable():
    p, blocks = _cached_page_pool()
    retire(p, 1)
    p.keep_hids = frozenset({blocks[0].block_id})  # this candidate's hit
    assert p.host_reclaimable_after() == 1
    p.host_pages.start_io(p.bindings[blocks[1].block_id], 9)
    assert p.host_reclaimable_after() == 0
    assert p.host_pages.reserve() is None
    assert not p._reclaim_cached_page()


def test_reset_donates_every_id():
    p, blocks = _cached_page_pool()
    assert p.reset_prefix_cache()
    assert p.num_donated() == 5 and p.num_cached_free() == 0
    assert sorted(p.finish_roles().donate) == sorted(b.block_id for b in blocks)


def _plan(k=K, floor=0):
    geometry = SimpleNamespace(
        hot_reserve_groups=lambda n: n // 4, group_domains=((GPU_HISTORY, HOST),)
    )
    return SimpleNamespace(
        geometry=geometry,
        capacity=lambda d: 100,
        hot_floor_groups=lambda r: max(floor, r),
        hot_pool_groups=0,
    )


def test_ledger_hot_gate_reserves_the_floor_next_to_new_ids():
    p = pool(blocks=9)
    ledger = LongContextAdmissionLedger(_plan(), 9, physical_segments=True)
    ledger.elastic_pool = p
    req = SimpleNamespace(request_id="a", num_prompt_tokens=80, max_tokens=0)
    ledger.plan.geometry.max_model_len = 10**6
    # Reserve 20 slots = 5 IDs; with 3 new IDs that needs 8 <= 8 free.
    assert ledger.blocking_domain(req, [3]) is None
    assert ledger.blocking_domain(req, [4]) == HOT
    p.get_new_blocks(3)
    ledger.commit(req)
    assert p.floor_slots == 20 and p.get_num_free_blocks() == 0
    ledger.release("a")
    assert p.floor_slots == 0 and p.get_num_free_blocks() == 5


def test_ledger_host_gate_uses_reclaimable_after():
    p, _ = _cached_page_pool()
    ledger = LongContextAdmissionLedger(_plan(), 6, physical_segments=True)
    ledger.elastic_pool = p
    ledger.plan.geometry.max_model_len = 10**6
    req = SimpleNamespace(request_id="c", num_prompt_tokens=4, max_tokens=0)
    assert ledger.blocking_domain(req, [1]) == HOST
    retire(p, 1)
    assert ledger.blocking_domain(req, [2]) is None
    assert ledger.blocking_domain(req, [3]) == HOST


def test_two_long_prefills_plan_128_passes_of_16_rows():
    passes = plan_step_passes(
        ["a", "b"], 2048, {"a": (1024, 65536, 1), "b": (1024, 65536, 1)}, 8242, 256
    )
    assert len(passes) == 128 and {e - s for s, e in passes} == {16}


def test_512_rows_of_short_history_are_bounded_by_r_cap():
    passes = plan_step_passes(["a"], 512, {"a": (512, 64, 1)}, 78, 256)
    assert passes == [(0, 256), (256, 512)]


def test_128_token_requests_one_pass_and_padding_rows_bound():
    reqs = {f"r{i}": (128, 32, 1) for i in range(8)}
    passes = plan_step_passes(list(reqs), 1024 + 3, reqs, 8 * 32, 256)
    assert passes[-1][1] == 1027 and all(e - s <= 256 for s, e in passes)
    # All requests' whole histories fit: only R_cap splits the rows.
    assert len(passes) == 5


def test_every_rank_plans_from_scheduler_metadata_not_local_budgets():
    # F1 counterexample: unequal worker budgets used to yield 80/89 rows per
    # pass. The shared region plans from U_step, identical on every rank.
    reqs = {"a": (2048, 65536, 1)}
    u_step = 41_440
    plans = [plan_step_passes(["a"], 2048, reqs, u_step, 256) for _ in range(4)]
    assert all(plan == plans[0] for plan in plans)
    assert plans[0] == plan_passes([2048], [65536], u_step, 256)


def test_scheduler_shares_one_common_plan_across_unequal_worker_budgets():
    from tests.v1.core.long_context_kv_utils import (
        production_config,
        production_specs,
    )
    from vllm.config.kernel import LongContextKVConfig
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    GiB = 1 << 30
    settings = LongContextKVConfig(enabled=True, host_pool_gib=112)
    config = production_config(max_model_len=32, long_context=settings)
    config.model_config.original_max_model_len = 32
    specs = production_specs(speculative_blocks=0)
    budgets = [5 * GiB, 3 * GiB, 4 * GiB, 3 * GiB + 100_000]
    workers = get_kv_cache_configs(config, [specs] * 4, budgets)
    plans = {w.long_context_plan for w in workers}
    assert len(plans) == 1
    (plan,) = plans
    assert plan.hot_slots_per_block > 0 and plan.hot_pool_min_groups == 0
    assert len({w.block_segments for w in workers}) == 1


def _host_scheduler(max_num_seqs=64, **kw):
    from tests.v1.core.test_long_context_admission import build, settings

    kw.setdefault("host_pool_tokens", 40 * 1664)
    kw.setdefault("hot_pool_gib", "auto")
    return build(settings(main_kv_placement="host", **kw), max_num_seqs=max_num_seqs)


def test_scheduler_short_request_burst_uses_idle_gpu_for_hot_region():
    """Leon's scenario: many short requests. The GPU index pool is not stuck
    evicting: the floor tracks the admitted 128-token working sets, freed IDs
    rejoin the hot region, and every step carries identical U_step/roles."""
    from tests.v1.core.test_long_context_admission import make_requests
    from tests.v1.core.test_scheduler_kv_capacity import step

    s = _host_scheduler(host_pool_tokens=64 * 1664)  # host pages are no limit
    pool = s.host_history_pool
    assert pool is not None and pool.elastic
    usable = pool.num_gpu_blocks - 1
    assert pool.hot_slots() == usable * pool.slots_per_block
    requests = make_requests(48, 128, 4, tag="short")
    for request in requests:
        s.add_request(request)
    outputs = []
    peak_running = 0
    for _ in range(64):
        output = s.schedule()
        outputs.append(output)
        peak_running = max(peak_running, len(output.num_scheduled_tokens))
        # Reuse the shared step helper for receipts and sampled tokens.
        s.schedule = lambda output=output: output
        step(s)
        del s.schedule
        if all(r.is_finished() for r in requests):
            break
    assert all(r.is_finished() for r in requests)
    outputs.append(s.schedule())  # the next step head donates the freed IDs
    # All 48 short requests run together: neither the GPU history pool nor
    # the floor of their tiny working sets limits concurrency.
    assert peak_running == 48
    assert s.kv_cache_manager.admission_ledger.blocked[3] == 0
    steps = [o.host_kv for o in outputs]
    assert all(h.hot_slots >= h.floor_slots for h in steps)
    assert max(h.floor_slots for h in steps) <= 48 * 32
    claimed = {hid for h in steps for hid in h.claim}
    donated = {hid for h in steps for hid in h.donate}
    assert claimed and claimed <= donated  # every claimed ID came back
    # After the burst every idle ID is hot again; none is stuck as index.
    assert pool.floor_slots == 0 and not pool.held
    assert pool.hot_slots() == usable * pool.slots_per_block - (
        pool.num_cached_free() * pool.slots_per_block
    )


def test_scheduler_new_zero_list_never_names_a_donated_page():
    from tests.v1.core.test_long_context_admission import make_requests
    from tests.v1.core.test_scheduler_kv_capacity import step

    s = _host_scheduler(max_num_seqs=4)
    pool = s.host_history_pool
    for request in make_requests(4, 3 * 1664 + 5, 2, tag="z"):
        s.add_request(request)
    for _ in range(40):
        output = s.schedule()
        donated = {b.block_id for b in pool.donated_queue.get_all_free_blocks()}
        assert not set(output.new_block_ids_to_zero or ()) & donated
        assert set(output.host_kv.claim).isdisjoint(output.host_kv.donate)
        s.schedule = lambda output=output: output
        step(s)
        del s.schedule


def test_static_hot_pool_rollback_switch(monkeypatch):
    from tests.v1.core.long_context_kv_utils import (
        production_config,
        production_specs,
    )
    from vllm.config.kernel import LongContextKVConfig
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    def plan():
        settings = LongContextKVConfig(enabled=True, host_pool_gib=112)
        config = production_config(max_model_len=32, long_context=settings)
        config.model_config.original_max_model_len = 32
        specs = production_specs(speculative_blocks=0)
        (worker,) = get_kv_cache_configs(config, [specs], [4 << 30])
        return worker.long_context_plan

    assert plan().hot_slots_per_block > 0
    monkeypatch.setenv("VLLM_SM70_HOST_KV_STATIC_HOT_POOL", "1")
    static = plan()
    assert static.hot_slots_per_block == 0 and static.hot_pool_groups > 0
