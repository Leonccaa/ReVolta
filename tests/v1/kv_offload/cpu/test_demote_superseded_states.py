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


# --- superseded_state_rule -------------------------------------------------

_BS = 1664


def _rule_scheduler(rule, manager):
    s = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    s.config = SimpleNamespace(
        demote_superseded_states=True,
        superseded_state_rule=rule,
        kv_group_configs=(
            SimpleNamespace(
                group_idx=0,
                sliding_window_size_in_blocks=1,
                requires_exact_boundary_source=True,
                offloaded_block_size=_BS,
            ),
        ),
    )
    s.manager = manager
    s.num_demoted_states = 0
    return s


def _req(prompt_blocks, output_blocks, hit, stored=(), junction=0):
    """A finished request: offload keys cover prompt + decode blocks."""
    keys = [key(0, i) for i in range(prompt_blocks + output_blocks)]
    return SimpleNamespace(
        group_states=(SimpleNamespace(offload_keys=keys, num_hit_blocks=hit),),
        req=SimpleNamespace(
            shared_prefix_boundary=junction,
            num_prompt_tokens=prompt_blocks * _BS + 100,
        ),
        req_context=CTX,
        stored_state_blocks={0: set(stored)} if stored else {},
    ), keys


def _demoted(rule, req_status):
    calls = []
    manager = SimpleNamespace(demote=lambda keys, ctx: calls.append(list(keys)))
    s = _rule_scheduler(rule, manager)
    s._demote_superseded_states(req_status)
    return (calls[0] if calls else []), s.num_demoted_states


def test_request_rule_counts_decode_output_and_demotes_resend_state():
    # 20 prompt blocks (resend states at blocks 18, 19), 5 decode blocks.
    req, keys = _req(20, 5, hit=0, stored=(18, 19))
    demoted, n = _demoted("request", req)
    # Historical behaviour: the tail is taken from 25 blocks, so the prompt-end
    # replay states the next turn resumes from (18, 19) are demoted.
    assert keys[18] in demoted and keys[19] in demoted
    assert n == len(demoted)


def test_request_fixed_keeps_prompt_end_states_after_long_output():
    req, keys = _req(20, 5, hit=0, stored=(18, 19))
    demoted, _ = _demoted("request_fixed", req)
    assert demoted == keys[:17]
    assert keys[18] not in demoted and keys[19] not in demoted


def test_request_fixed_equals_request_for_short_output():
    req, _ = _req(20, 0, hit=3, stored=(18, 19))
    assert _demoted("request", req)[0] == _demoted("request_fixed", req)[0]


def test_extend_demotes_previous_end_state_once_a_newer_state_is_stored():
    # Continued turn: hit at 19 blocks (previous end state at block 18),
    # this request stored new states at blocks 21 and 22.
    req, keys = _req(23, 1, hit=19, stored=(21, 22))
    demoted, _ = _demoted("extend", req)
    # [hit, last - 3) = [19, 20) as before, plus the hit point (block 18).
    assert demoted == [keys[19], keys[18]]


def test_extend_keeps_hit_state_when_nothing_newer_was_stored():
    # Preempted or hand-off dropped: no state stored past the hit.
    req, keys = _req(23, 1, hit=19, stored=())
    assert _demoted("extend", req)[0] == [keys[19]]
    req, keys = _req(23, 1, hit=19, stored=(10,))
    assert _demoted("extend", req)[0] == [keys[19]]


def test_extend_never_demotes_the_junction():
    req, keys = _req(23, 1, hit=19, stored=(21, 22), junction=19 * _BS)
    assert keys[18] not in _demoted("extend", req)[0]


def test_cold_request_extend_equals_request_fixed():
    req, _ = _req(20, 5, hit=0, stored=(18, 19))
    assert _demoted("extend", req)[0] == _demoted("request_fixed", req)[0]


@pytest.mark.parametrize("rule", ["request", "request_fixed", "extend"])
def test_rule_config_accepted(rule):
    extra = {"superseded_state_rule": rule}
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(block_size=16),
    )
    spec = SimpleNamespace()
    OffloadingSpec.__init__(spec, config, SimpleNamespace(kv_cache_groups=[]))
    assert spec.superseded_state_rule == rule


def test_rule_config_default_and_rejects_unknown():
    def make(extra):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1, prefill_context_parallel_size=1
            ),
            cache_config=SimpleNamespace(block_size=16),
        )
        spec = SimpleNamespace()
        OffloadingSpec.__init__(spec, config, SimpleNamespace(kv_cache_groups=[]))
        return spec

    assert make({}).superseded_state_rule == "request"
    with pytest.raises(ValueError, match="superseded_state_rule"):
        make({"superseded_state_rule": "extend_all"})


def test_extend_keeps_the_checkpoint_an_identical_resend_uses():
    # 33,280-token prompt (exactly 20 blocks): hit at 31,616 (19 blocks), new
    # state stored at 33,280 (block 19). An identical resend is capped at
    # prompt_length - 1 = 33,279 and resumes from 31,616 (block 18), inside the
    # protected tail, so extend must not demote it.
    keys = [key(0, i) for i in range(21)]
    req = SimpleNamespace(
        group_states=(SimpleNamespace(offload_keys=keys, num_hit_blocks=19),),
        req=SimpleNamespace(shared_prefix_boundary=0, num_prompt_tokens=20 * _BS),
        req_context=CTX,
        stored_state_blocks={0: {19}},
    )
    demoted, _ = _demoted("extend", req)
    assert keys[18] not in demoted
    assert demoted == []


def _with_other(rule, finishing, other_hit_blocks, other_keys):
    calls = []
    manager = SimpleNamespace(demote=lambda keys, ctx: calls.append(list(keys)))
    s = _rule_scheduler(rule, manager)
    other = SimpleNamespace(
        group_states=(
            SimpleNamespace(offload_keys=other_keys, num_hit_blocks=other_hit_blocks),
        )
    )
    s._req_status = {"finishing": finishing, "other": other}
    s._demote_superseded_states(finishing)
    return calls[0] if calls else []


def test_extend_keeps_a_checkpoint_another_active_request_resumed_from():
    # A and B both hit checkpoint E (block 18); no junction is recorded. A
    # extends past it and finishes while B is still running.
    a, keys = _req(23, 1, hit=19, stored=(21, 22))
    assert keys[18] in _with_other("extend", a, 0, [])
    a, keys = _req(23, 1, hit=19, stored=(21, 22))
    demoted = _with_other("extend", a, 19, keys[:20])
    assert keys[18] not in demoted
    assert demoted == [keys[19]]


def test_request_fixed_also_respects_other_requests_hit_states():
    # B resumed from A's block 10 (e.g. a sibling branch): A's own range must
    # not demote it under the new rules.
    a, keys = _req(20, 5, hit=0, stored=(18, 19))
    demoted = _with_other("request_fixed", a, 11, keys[:12])
    assert keys[10] not in demoted and keys[9] in demoted


def test_request_rule_unchanged_by_other_requests():
    a, keys = _req(20, 5, hit=0, stored=(18, 19))
    assert _with_other("request", a, 11, keys[:12]) == _demoted("request", a)[0]


def test_extend_protects_the_shifted_resend_checkpoint_with_eagle():
    # 23-block prompt (+100 tokens): resend checkpoint is block 22. Hit at 22
    # blocks (state at block 21), new state stored at block 22. Without a
    # shift block 21 may be demoted; with an EAGLE/MTP shift it is the usable
    # resend state and must be kept.
    req, keys = _req(23, 1, hit=22, stored=(22,))
    assert keys[21] in _demoted("extend", req)[0]
    calls = []
    manager = SimpleNamespace(demote=lambda k, ctx: calls.append(list(k)))
    s = _rule_scheduler("extend", manager)
    s._eagle_state_shift = 1664
    req, keys = _req(23, 1, hit=22, stored=(22,))
    s._demote_superseded_states(req)
    assert not calls or keys[21] not in calls[0]
