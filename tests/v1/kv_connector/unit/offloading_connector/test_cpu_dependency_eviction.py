# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency removals must preserve every hit, including partial GPU prefixes."""

import random
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from tests.v1.kv_connector.unit.offloading_connector.test_cpu_draft_offload import (
    DRAFT,
    FULL,
    MAMBA,
    MODES,
    PROMPT,
    _hashes,
    _kv_config,
    _request,
    _standard_run,
    _vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    CPU_RESIDENT_KEY,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.kv_offload.base import ReqContext, RequestOffloadingContext
from vllm.v1.kv_offload.cpu.dependency import KINDS, STATUSES
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

pytestmark = pytest.mark.cpu_test
CTX = ReqContext("test")


def make(
    *,
    mode="all",
    grouped=False,
    eviction=True,
    stats=False,
    capacity=128,
    secondary=False,
    target_swa=False,
    eagle=False,
    policy="lru",
    state_groups=1,
    production=False,
):
    config = _vllm_config(mode)
    config.kv_transfer_config.kv_connector_extra_config.update(
        {
            "cpu.enable_dependency_eviction": eviction,
            "cpu.residency_stats": stats,
            "superseded_state_rule": "extend",
        }
    )
    if eagle:
        config.speculative_config.use_eagle_kv_cache = lambda: True
    kv = _kv_config("model.layers.2.self_attn.attn") if target_swa else _kv_config()
    if production:
        for i, group in enumerate(kv.kv_cache_groups):
            group.kv_cache_spec = replace(
                group.kv_cache_spec,
                block_size=832 if i == DRAFT else 1664,
                **({"sliding_window": 2880} if i == DRAFT else {}),
            )
    for i in range(1, state_groups):
        kv.kv_cache_groups.append(
            KVCacheGroupSpec(
                [f"model.layers.{i + 4}.linear_attn"],
                kv.kv_cache_groups[MAMBA].kv_cache_spec,
            )
        )
    spec = MockOffloadingSpec(config, kv)
    pools = [
        CPUOffloadingManager(capacity, cache_policy=policy, enable_events=True)
        for _ in range(len(kv.kv_cache_groups) if grouped else 1)
    ]
    if grouped:
        spec.manager = GroupedCPUOffloadingManager(dict(enumerate(pools)))
    else:
        spec.manager = pools[0]
    if secondary:
        spec.manager = TieringOffloadingManager(spec.manager, [SimpleNamespace()])
    return OffloadingConnectorScheduler(spec), pools


def populate(scheduler, *, states=(96, 192), junction=0, salt="p", finish=True):
    req = _request(salt, PROMPT, _hashes(12, salt), junction=junction)
    scheduler.on_new_request(req)
    status = scheduler._req_status[salt]
    status.update_offload_keys()
    groups = [g.offload_keys for g in status.group_states]
    keys = list(groups[FULL]) + [
        groups[group.group_idx][b // group.offloaded_block_size - 1]
        for group in scheduler.config.kv_group_configs
        if group.requires_exact_boundary_source
        for b in states
    ]
    if scheduler.config.cpu_draft_offload != "none":
        keys += groups[DRAFT]
    out = scheduler._prepare_store(keys, status)
    assert out is not None
    scheduler.manager.complete_store(out.keys_to_store, status.req_context)
    if finish:
        req.is_finished.return_value = True
        scheduler.request_finished(req)
    return status, groups


def discard(pools, key):
    for pool in pools:
        if pool._policy.get(key) is not None:
            pool.remove_dependency_blocks([key])
            return
    raise AssertionError("missing test block")


def oracle(scheduler, paths):
    """Enumerate real _lookup results for all prompt ends and GPU start tokens.

    Using every integer GPU prefix includes partial prefix-cache entries and
    windows shorter than the full draft window. No dependency-index logic is
    used to decide which blocks can be returned.
    """
    useful = set()
    hits = []
    for original in paths:
        for end in range(33, PROMPT + 1, 32):
            req = _request("oracle", end, original.req.block_hashes)
            status = RequestOffloadState(
                scheduler.config, req, CTX, RequestOffloadingContext()
            )
            status.update_offload_keys()
            for local in range(end):
                status.num_locally_computed_tokens = local
                hit = scheduler._lookup(status)
                hits.append(hit)
                if not hit:
                    continue
                for group in scheduler.config.kv_group_configs:
                    keys = status.group_states[group.group_idx].offload_keys
                    size = group.offloaded_block_size
                    loaded = keys[local // size : (local + hit + size - 1) // size]
                    if group.sliding_window_size_in_blocks is not None:
                        loaded = loaded[-group.sliding_window_size_in_blocks :]
                    useful.update(loaded)
    return useful, hits


@pytest.mark.parametrize(
    "key", ["cpu.enable_dependency_eviction", "cpu.residency_stats"]
)
@pytest.mark.parametrize("value", [0, 1, "true", "false", None, [], {}])
def test_bool_validation(key, value):
    config = _vllm_config("all")
    config.kv_transfer_config.kv_connector_extra_config[key] = value
    with pytest.raises(ValueError, match=key):
        MockOffloadingSpec(config, _kv_config())


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_removal_categories_and_closure(grouped, mode):
    scheduler, pools = make(grouped=grouped, mode=mode)
    status, g = populate(scheduler)
    index = scheduler._dependency_index
    # A lost mandatory target page invalidates the state at that boundary.
    discard(pools, g[FULL][5])
    classes = index.classify()
    assert classes[g[MAMBA][5]] == "chain_broken"
    if mode != "none":
        assert classes[g[DRAFT][0]] == "outside_window"
        # The tail past the last remaining state has no possible lookup.
        assert classes[g[FULL][4]] == "no_state"
    else:
        assert classes[g[MAMBA][2]] == "missing_dependency"
    useful, before = oracle(scheduler, [status])
    removed = set(index.evict(set()))
    assert removed.isdisjoint(useful)
    assert oracle(scheduler, [status])[1] == before
    assert g[MAMBA][5] in removed
    assert index.evict(set()) == []
    for pool in pools:
        assert (
            len(pool.resident_keys()) + pool._get_num_free_blocks() == pool._num_blocks
        )


def test_missing_draft_tail_invalidates_state_and_cleans_target_tail():
    scheduler, pools = make()
    _, g = populate(scheduler)
    discard(pools, g[DRAFT][-1])
    index = scheduler._dependency_index
    assert index.classify()[g[MAMBA][-1]] == "missing_dependency"
    removed = index.evict(set())
    assert g[MAMBA][-1] in removed
    assert g[FULL][-1] in removed


def test_eagle_probe_page_is_a_dependency_even_when_not_loaded():
    scheduler, pools = make(eagle=True)
    status, g = populate(scheduler, states=(160,))
    useful, before = oracle(scheduler, [status])
    assert g[FULL][-1] not in useful  # full reach is shifted back before loads
    assert g[FULL][-1] not in scheduler._dependency_index.evict(set())
    assert oracle(scheduler, [status])[1] == before


def test_unannotated_resident_or_incomplete_active_hashes_keep_unknown_rows():
    scheduler, pools = make()
    _, g = populate(scheduler)
    index = scheduler._dependency_index
    record = index.records[g[MAMBA][-1]]
    index.records[g[MAMBA][-1]] = None
    assert set(index.classify().values()) == {"unknown"}
    assert index.evict(set()) == []
    index.records[g[MAMBA][-1]] = record
    req = _request("unhashed", PROMPT, hashes=[])
    scheduler.on_new_request(req)
    protected = scheduler._dependency_protected_keys()
    assert protected == index.records.keys()
    assert index.evict(protected) == []


def test_active_prefix_hit_and_pending_later_state_keep_target_tail():
    scheduler, pools = make()
    status, g = populate(scheduler, states=(96,), finish=False)
    status.group_states[MAMBA].num_hit_blocks = 3
    status.pending_boundary_offloads = [(MAMBA, 999, 192)]
    removed = scheduler._dependency_index.evict(scheduler._dependency_protected_keys())
    assert not removed
    assert g[MAMBA][2] in scheduler._dependency_index.records
    assert g[FULL][-1] in scheduler._dependency_index.records


def test_gpu_prefix_can_bridge_missing_cpu_head_and_draft_window_head():
    scheduler, pools = make()
    status, g = populate(scheduler)
    discard(pools, g[FULL][0])
    discard(pools, g[DRAFT][3])  # head of the window at 96, GPU can supply it
    pools[0].demote([g[MAMBA][2]], CTX)  # extend remains a soft demotion
    useful, before = oracle(scheduler, [status])
    assert g[MAMBA][2] in useful
    assert g[FULL][1] in useful
    removed = set(scheduler._dependency_index.evict(set()))
    assert removed.isdisjoint(useful)
    assert oracle(scheduler, [status])[1] == before


@pytest.mark.parametrize("pin", ["load", "store", "active", "junction"])
def test_must_keep_dependencies(pin):
    scheduler, pools = make()
    status, g = populate(
        scheduler,
        states=(192,),
        junction=192 if pin == "junction" else 0,
        finish=pin != "active",
    )
    if pin == "load":
        pools[0].prepare_load([g[MAMBA][-1]], CTX)
    elif pin == "store":
        pools[0]._policy.get(g[MAMBA][-1]).ref_cnt = -1
    discard(pools, g[DRAFT][-1])
    before = set(scheduler._dependency_index.records)
    protected = scheduler._dependency_protected_keys()
    removed = scheduler._dependency_index.evict(protected)
    required = set(g[FULL] + [g[MAMBA][-1]] + g[DRAFT][-3:-1])
    assert set(removed).isdisjoint(required)
    assert required <= scheduler._dependency_index.records.keys()
    if pin == "active":
        assert not removed
        assert set(scheduler._dependency_index.records) == before


@pytest.mark.parametrize("case", ["secondary", "target_swa", "arc"])
def test_unmodelled_tiers_and_groups_are_kept(case):
    scheduler, pools = make(
        secondary=case == "secondary",
        target_swa=case == "target_swa",
        policy="arc" if case == "arc" else "lru",
    )
    # Store directly: no annotations are required in the unknown case.
    status = _request("r", PROMPT)
    rs = RequestOffloadState(scheduler.config, status, CTX, RequestOffloadingContext())
    rs.update_offload_keys()
    keys = rs.group_states[0].offload_keys
    out = pools[0].prepare_store(keys, CTX)
    pools[0].complete_store(out.keys_to_store, CTX)
    index = scheduler._dependency_index
    assert set(index.classify().values()) == {"unknown"}
    assert index.evict(set()) == []
    pools[0].reset_cache()
    assert not index.records and not index.owners


@pytest.mark.parametrize("grouped", [False, True])
def test_seven_state_groups_production_geometry_and_missing_peer(grouped):
    scheduler, pools = make(grouped=grouped, state_groups=7, production=True)
    req = _request("production-shape", 6 * 1664 + 8, _hashes(12))
    scheduler.on_new_request(req)
    rs = scheduler._req_status[req.request_id]
    rs.update_offload_keys()
    keys = []
    for group in scheduler.config.kv_group_configs:
        group_keys = rs.group_states[group.group_idx].offload_keys
        keys.extend(
            group_keys[-1:] if group.requires_exact_boundary_source else group_keys
        )
    out = scheduler._prepare_store(keys, rs)
    scheduler.manager.complete_store(out.keys_to_store, CTX)
    req.is_finished.return_value = True
    scheduler.request_finished(req)
    index = scheduler._dependency_index
    state_key = rs.group_states[MAMBA].offload_keys[-1]
    assert len(index.records[state_key].states) == 7
    assert len(index.records[state_key].windows) == 4
    discard(pools, rs.group_states[8].offload_keys[-1])
    removed = index.evict(set())
    assert state_key in removed
    assert not index.records  # no complete checkpoint remains


@pytest.mark.parametrize("grouped", [False, True])
def test_bounded_turnover_reset_and_failed_stores(grouped):
    scheduler, pools = make(grouped=grouped, capacity=24)
    index = scheduler._dependency_index
    for i in range(100):
        populate(scheduler, salt=str(i))
        resident = {key for pool in pools for key in pool.resident_keys()}
        assert index.records.keys() == index.owners.keys() == resident
        assert len(resident) <= 24 * len(pools)
        assert (
            sum(
                len(r.windows) + len(r.states) + len(r.targets)
                for r in index.records.values()
                if r
            )
            <= len(resident) * 8
        )
        if i % 7 == 0:
            scheduler.manager.reset_cache()
            assert not index.records and not index.owners
    for pool in pools:
        pool.reset_cache()
    req = _request("failed", PROMPT)
    scheduler.on_new_request(req)
    rs = scheduler._req_status["failed"]
    rs.update_offload_keys()
    out = scheduler._prepare_store(rs.group_states[0].offload_keys, rs)
    scheduler.manager.complete_store(out.keys_to_store, CTX, success=False)
    assert not index.records and not index.owners


def test_grouped_reservation_rollback_cleans_index_after_capacity_eviction():
    scheduler, pools = make(grouped=True, capacity=2)
    req = _request("old", PROMPT)
    scheduler.on_new_request(req)
    rs = scheduler._req_status["old"]
    rs.update_offload_keys()
    g = [group.offload_keys for group in rs.group_states]
    out = scheduler._prepare_store(g[FULL][:2] + [g[MAMBA][1]] + g[DRAFT][2:4], rs)
    scheduler.manager.complete_store(out.keys_to_store, CTX)
    req.is_finished.return_value = True
    scheduler.request_finished(req)
    pools[MAMBA].prepare_load([g[MAMBA][1]], CTX)
    req = _request("new", PROMPT, _hashes(12, "new"))
    scheduler.on_new_request(req)
    rs = scheduler._req_status["new"]
    rs.update_offload_keys()
    offered = (
        rs.group_states[FULL].offload_keys[:2] + rs.group_states[MAMBA].offload_keys[:2]
    )
    assert scheduler._prepare_store(offered, rs) is None
    index = scheduler._dependency_index
    assert not (set(offered) & index.records.keys())
    assert index.records.keys() == {k for p in pools for k in p.resident_keys()}
    assert pools[FULL].num_evicted_blocks == 2
    pools[MAMBA].complete_load([g[MAMBA][1]], CTX)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("eagle", [False, True])
def test_randomized_against_lookup_oracle(grouped, mode, eagle):
    rng = random.Random(17)
    for trial in range(12):
        scheduler, pools = make(grouped=grouped, mode=mode, eagle=eagle)
        a, _ = populate(scheduler, salt="a")
        # Shared prefix followed by two branches (hashes represent chained keys).
        b_req = _request("b", PROMPT, a.req.block_hashes[:4] + _hashes(8, "b"))
        scheduler.on_new_request(b_req)
        b = scheduler._req_status["b"]
        b.update_offload_keys()
        keys = [k for g in b.group_states for k in g.offload_keys]
        if mode == "none":
            keys = [k for g in b.group_states[:2] for k in g.offload_keys]
        out = scheduler._prepare_store(keys, b)
        scheduler.manager.complete_store(out.keys_to_store, CTX)
        b_req.is_finished.return_value = True
        scheduler.request_finished(b_req)
        index = scheduler._dependency_index
        for key in list(index.records):
            if rng.random() < 0.25:
                discard(pools, key)
        useful, before = oracle(scheduler, [a, b])
        order = {id(p): p.resident_keys() for p in pools}
        removed = set(index.evict(set()))
        assert removed.isdisjoint(useful), (trial, removed & useful)
        assert oracle(scheduler, [a, b])[1] == before
        for pool in pools:
            assert pool.resident_keys() == [
                k for k in order[id(pool)] if k not in removed
            ]


def test_pressure_triggers_dependency_events_and_metrics():
    scheduler, pools = make(capacity=20)
    populate(scheduler)
    pool = pools[0]
    list(pool.take_events())
    before = pool.num_evicted_blocks
    populate(scheduler, salt="new", states=())
    assert pool.evicted_blocks_by_reason["dependency"] > 0
    removed = sum(len(e.keys) for e in pool.take_events() if e.removed)
    assert removed == pool.num_evicted_blocks - before
    reasons = pool.evicted_blocks_by_reason
    assert pool.num_evicted_blocks == sum(
        reasons[r] for r in ("capacity", "superseded", "dependency")
    )


def test_default_has_no_index_and_stats_only_does_not_evict():
    off, _ = make(eviction=False)
    stats, _ = make(eviction=False, stats=True)
    assert off._dependency_index is None
    assert off.cpu_residency_snapshot() is None
    left = _standard_run(off, complete=True)
    right = _standard_run(stats, complete=True)

    # CPU specs contain physical IDs rather than mock offload keys.
    def signature(jobs):
        return [
            (a.block_ids.tolist(), a.group_sizes, a.block_indices, b.block_ids.tolist())
            for job in jobs
            for a, b in [job.transfer_spec]
        ]

    assert signature(left.jobs) == signature(right.jobs)
    assert stats._dependency_index.records
    assert stats.manager.evicted_blocks_by_reason["dependency"] == 0


def test_c7_throttle_log_consumption_reset_and_prometheus():
    scheduler, pools = make(eviction=False, stats=True)
    populate(scheduler)
    connector = object.__new__(OffloadingConnector)
    connector.connector_worker = None
    connector.connector_scheduler = scheduler
    registry = CollectorRegistry()

    def bind(cls):
        return lambda *a, **kw: cls(*a, registry=registry, **kw)

    prom = OffloadPromMetrics(
        SimpleNamespace(kv_transfer_config=None),
        {Gauge: bind(Gauge), Counter: bind(Counter), Histogram: bind(Histogram)},
        ["engine"],
        {0: ["0"]},
    )
    assert not any("cpu_resident" in m.name for m in registry.collect())
    with patch("time.monotonic", return_value=100):
        scheduler.cpu_residency_snapshot(
            consume=False
        )  # log must not eat metric sample
        sample = connector.get_kv_connector_stats()
        assert sample is not None and CPU_RESIDENT_KEY in sample.data
        assert sum(sample.reduce().values()) == len(scheduler._dependency_index.records)
        assert connector.get_kv_connector_stats() is None
    prom.observe(sample.data)
    with patch("time.monotonic", return_value=159):
        assert scheduler.cpu_residency_snapshot() is None
    scheduler.manager.reset_cache()
    with patch("time.monotonic", return_value=160):
        zero = connector.get_kv_connector_stats()
    merged = sample.aggregate(zero)
    prom.observe(merged.data)
    assert len(merged.data[CPU_RESIDENT_KEY]) == 1
    for kind in KINDS:
        for status in STATUSES:
            assert (
                registry.get_sample_value(
                    "vllm:kv_offload_cpu_resident_blocks",
                    {"engine": "0", "kind": kind, "status": status},
                )
                == 0
            )
