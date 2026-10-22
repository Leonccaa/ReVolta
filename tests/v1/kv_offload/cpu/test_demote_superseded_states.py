# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Superseded recurrent checkpoints are the host tier's first eviction victims."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestGroupState,
)
from vllm.v1.kv_offload.base import OffloadingSpec, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)

pytestmark = pytest.mark.cpu_test

CTX = ReqContext("test")


@pytest.mark.parametrize("override,expected", [(None, True), (False, False)])
def test_demotion_defaults_on_and_accepts_disable_override(override, expected):
    extra = {} if override is None else {"demote_superseded_states": override}
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(block_size=16),
    )
    spec = SimpleNamespace()
    OffloadingSpec.__init__(spec, config, SimpleNamespace(kv_cache_groups=[]))
    assert spec.demote_superseded_states is expected


def key(group, index):
    return make_offload_key(index.to_bytes(8, "big"), group)


def store(m, keys):
    out = m.prepare_store(keys, CTX)
    assert out is not None
    m.complete_store(out.keys_to_store, CTX)
    return out


def test_lru_demote_makes_keys_the_next_victims():
    m = CPUOffloadingManager(3, cache_policy="lru")
    a, b, c, d = (key(0, i) for i in range(4))
    store(m, [a, b, c])
    # Without demotion ``a`` (the oldest) would go first.
    m.demote([c, key(0, 99)], CTX)  # missing keys are ignored
    out = store(m, [d])
    assert out.evicted_keys == [c]
    assert all(m.lookup(k, CTX) for k in (a, b, d))


def test_lru_demote_skips_referenced_blocks():
    m = CPUOffloadingManager(2, cache_policy="lru")
    a, b, c = (key(0, i) for i in range(3))
    store(m, [a, b])
    m.prepare_load([b], CTX)  # b is being read
    m.demote([b], CTX)
    out = store(m, [c])
    assert out.evicted_keys == [a]
    m.complete_load([b], CTX)


def test_grouped_demote_partitions_by_group():
    m = GroupedCPUOffloadingManager(
        {g: CPUOffloadingManager(2, cache_policy="lru") for g in (0, 2)}
    )
    store(m, [key(0, 1), key(0, 2), key(2, 1), key(2, 2)])
    m.demote([key(2, 2)], CTX)
    out = store(m, [key(0, 3), key(2, 3)])
    assert sorted(out.evicted_keys) == sorted([key(0, 1), key(2, 2)])


def _scheduler(enabled, configs, manager):
    s = object.__new__(OffloadingConnectorScheduler)
    s.config = SimpleNamespace(
        demote_superseded_states=enabled, kv_group_configs=configs
    )
    s.manager = manager
    return s


@pytest.mark.parametrize("enabled", [True, False])
def test_request_completion_respects_demotion_policy(enabled):
    manager = MagicMock()
    config = SimpleNamespace(
        group_idx=0,
        sliding_window_size_in_blocks=1,
        requires_exact_boundary_source=True,
    )
    keys = [key(0, i) for i in range(6)]
    state = SimpleNamespace(
        group_states=(SimpleNamespace(offload_keys=keys, num_hit_blocks=0),),
        req=SimpleNamespace(shared_prefix_boundary=0),
        req_context=CTX,
        transfer_jobs=[],
    )
    scheduler = _scheduler(enabled, (config,), manager)
    scheduler._req_status = {"test": state}
    scheduler._drop_pending_boundary_offloads = MagicMock()
    assert scheduler.request_finished(SimpleNamespace(request_id="test")) == (
        False,
        None,
    )
    manager.on_request_finished.assert_called_once_with(CTX)
    if enabled:
        manager.demote.assert_called_once_with(keys[:3], CTX)
    else:
        manager.demote.assert_not_called()
    assert "test" not in scheduler._req_status


def test_request_finished_demotes_all_but_the_tail_states():
    calls = []
    manager = SimpleNamespace(demote=lambda keys, ctx: calls.append(list(keys)))
    configs = (
        SimpleNamespace(
            group_idx=0,
            sliding_window_size_in_blocks=None,
            requires_exact_boundary_source=False,
        ),
        SimpleNamespace(
            group_idx=1,
            sliding_window_size_in_blocks=1,
            requires_exact_boundary_source=True,
        ),
    )
    attn = [key(0, i) for i in range(10)]
    state = [key(1, i) for i in range(10)]
    req_status = SimpleNamespace(
        group_states=(
            SimpleNamespace(offload_keys=attn, num_hit_blocks=0),
            SimpleNamespace(offload_keys=state, num_hit_blocks=0),
        ),
        req=SimpleNamespace(shared_prefix_boundary=0),
        req_context=CTX,
    )
    _scheduler(True, configs, manager)._demote_superseded_states(req_status)
    # Attention blocks stay; the last three state positions (replay
    # boundaries) stay; everything older is demoted.
    assert calls == [state[:7]]


def test_short_requests_demote_nothing():
    calls = []
    manager = SimpleNamespace(demote=lambda keys, ctx: calls.append(list(keys)))
    configs = (
        SimpleNamespace(
            group_idx=0,
            sliding_window_size_in_blocks=1,
            requires_exact_boundary_source=True,
        ),
    )
    req_status = SimpleNamespace(
        group_states=(
            SimpleNamespace(offload_keys=[key(0, 0), key(0, 1)], num_hit_blocks=0),
        ),
        req=SimpleNamespace(shared_prefix_boundary=0),
        req_context=CTX,
    )
    _scheduler(True, configs, manager)._demote_superseded_states(req_status)
    assert calls == []


@pytest.mark.parametrize(
    "hits,junction,block_size,expected",
    [
        (0, 0, 32, [0, 1, 2, 3, 4, 5, 6]),
        (3, 0, 32, [3, 4, 5, 6]),
        (7, 0, 32, []),
        (10, 0, 32, []),
        (12, 0, 32, []),
        (0, 32, 32, [1, 2, 3, 4, 5, 6]),
        (0, 160, 32, [0, 1, 2, 3, 5, 6]),
        (0, 161, 32, [0, 1, 2, 3, 4, 5, 6]),
        (0, 256, 32, [0, 1, 2, 3, 4, 5, 6]),
        (0, 1024, 32, [0, 1, 2, 3, 4, 5, 6]),
        (3, 160, 32, [3, 5, 6]),
        (0, 80, 16, [0, 1, 2, 3, 5, 6]),
    ],
)
def test_demotion_excludes_hits_and_exact_junction(
    hits, junction, block_size, expected
):
    manager = MagicMock()
    config = SimpleNamespace(
        group_idx=0,
        sliding_window_size_in_blocks=1,
        requires_exact_boundary_source=True,
        offloaded_block_size=block_size,
    )
    keys = [key(0, i) for i in range(10)]
    state = SimpleNamespace(
        group_states=(RequestGroupState(offload_keys=keys, num_hit_blocks=hits),),
        req=SimpleNamespace(shared_prefix_boundary=junction),
        req_context=CTX,
    )
    _scheduler(True, (config,), manager)._demote_superseded_states(state)
    if expected:
        manager.demote.assert_called_once_with([keys[i] for i in expected], CTX)
    else:
        manager.demote.assert_not_called()
    # Candidate selection must not change block identity or lookup metadata.
    assert state.group_states[0].offload_keys == keys
    assert state.group_states[0].num_hit_blocks == hits
    assert state.req.shared_prefix_boundary == junction


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_junctions_remain_evictable_and_pool_capacity_stays_bounded(policy):
    manager = CPUOffloadingManager(4, cache_policy=policy)
    keys = [key(0, i) for i in range(10)]
    # Several historical checkpoints: two in the inherited hit range, one
    # at the current junction, and one unknown to this request.
    historical = [keys[i] for i in (0, 2, 4, 6)]
    store(manager, historical)
    config = SimpleNamespace(
        group_idx=0,
        sliding_window_size_in_blocks=1,
        requires_exact_boundary_source=True,
        offloaded_block_size=32,
    )
    state = SimpleNamespace(
        group_states=(RequestGroupState(offload_keys=keys, num_hit_blocks=3),),
        req=SimpleNamespace(shared_prefix_boundary=160),
        req_context=CTX,
    )
    _scheduler(True, (config,), manager)._demote_superseded_states(state)
    evicted = []
    for i in range(4):
        evicted.extend(store(manager, [key(0, 100 + i)]).evicted_keys)
    if policy == "lru":
        assert evicted[0] == keys[6]  # unrecorded historical junction
    assert set(evicted) == set(historical)
    assert not any(manager.lookup(k, CTX) for k in historical)
    assert all(manager.lookup(key(0, 100 + i), CTX) for i in range(4))
