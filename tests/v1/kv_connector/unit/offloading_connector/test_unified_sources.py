# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inventory admission requires an exact source in the current step."""

from types import SimpleNamespace

import pytest

from tests.v1.core.long_context_kv_utils import SPAN
from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
    setup_receive,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.unified.manager import UnifiedStoreManager

pytestmark = pytest.mark.cpu_test


def test_junction_without_state_handoff_is_not_admitted():
    scheduler, connector, request, _ = setup_receive(speculative_tokens=1)
    m = UnifiedStoreManager(
        SimpleNamespace(
            kv_cache_config=connector.kv_cache_config,
            extra_config={},
            cpu_page_size_per_worker=16,
        )
    )
    m.attach(scheduler.host_history_pool)
    connector.manager = connector.unified = m
    request.shared_prefix_boundary = 4 * SPAN
    status = connector._req_status[request.request_id]
    assert not status.pending_boundary_offloads
    assert m.admit_build(request.request_id, 11)
    out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 16},
        total_num_scheduled_tokens=16,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    connector._prepare_unified_boundaries(out)
    assert not m.builds and not m.pending
    assert m.events["junction_host_b_unavailable"] == 1
    m.end_step()
    assert not m.units.events["revoke_build_failed"]


@pytest.mark.parametrize(
    "source",
    [
        "ready",
        "exact",
        "partial",
        "capped",
        "draft",
        "draft_lost",
        "draft_capped",
        "draft_shared",
    ],
)
def test_inventory_matches_current_dispatch_sources(source):
    from tests.v1.kv_offload.test_unified_units import publish
    from vllm.v1.kv_offload.base import ReqContext

    scheduler, connector, request, _ = setup_receive(speculative_tokens=1)
    m = UnifiedStoreManager(
        SimpleNamespace(
            kv_cache_config=connector.kv_cache_config,
            extra_config={},
            cpu_page_size_per_worker=16,
        )
    )
    m.attach(scheduler.host_history_pool)
    connector.manager = connector.unified = m
    boundary = request.shared_prefix_boundary = 4 * SPAN
    status = connector._req_status[request.request_id]
    inventory = []
    states = []
    drafts: list[tuple[GroupOffloadConfig, int, OffloadKey]] = []
    for group in connector.config.kv_group_configs:
        end = boundary // group.offloaded_block_size
        keys = status.group_states[group.group_idx].offload_keys
        if group.requires_exact_boundary_source:
            states.append((group, keys[end - 1]))
            inventory.append(keys[end - 1])
        elif group.draft_window_tokens is not None:
            window = group.sliding_window_size_in_blocks
            drafts.extend((group, i, keys[i]) for i in range(max(0, end - window), end))
            inventory.extend(keys[max(0, end - window) : end])
    draft_case = source.startswith("draft")
    missing = {drafts[-1][2]} if draft_case else {k for _, k in states}
    if source == "ready":
        missing.clear()
    for k in inventory:
        if k not in missing:
            publish(m.table, k, m.kind(k))
    if source in ("exact", "partial", "capped"):
        handoffs = states[:-1] if source == "partial" else states
        status.pending_boundary_offloads = [
            (g.group_idx, 123 + g.group_idx, boundary) for g, _ in handoffs
        ]
    if source in ("capped", "draft_capped"):
        status.max_offload_tokens = boundary - SPAN
    shared = None
    if source == "draft_shared":
        shared = m.prepare_store(list(missing), ReqContext("owner"))
        m.bind_job(70, "owner", shared.store_spec, True)
    groups: list[list[int]] = [[] for _ in status.group_states]
    if draft_case:
        group, index, _ = drafts[-1]
        groups[group.group_idx] = [123] * (
            (index + 1) * connector.config.block_size_factor
        )
        if source == "draft_lost":
            request.num_computed_tokens = request.num_tokens
    out = SimpleNamespace(
        num_scheduled_tokens={request.request_id: boundary},
        scheduled_new_reqs=[
            SimpleNamespace(req_id=request.request_id, block_ids=tuple(groups))
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
    )
    assert m.admit_build(request.request_id, 11)
    connector._prepare_unified_boundaries(out)
    expected = source in ("ready", "exact", "draft", "draft_shared")
    assert bool(m.units.units) == expected
    if shared is not None:
        # The new boundary waits for the original job, without a second read.
        assert not m.prepare_store(
            list(missing), ReqContext(request.request_id)
        ).keys_to_store
        m.end_step()
        m.receipt(70, shared.store_spec.gen, m.table.all_ranks, 0)
        assert m.units.units[next(iter(m.units.units))].complete
    elif expected and missing:
        result = m.prepare_store(list(missing), ReqContext(request.request_id))
        m.bind_job(1, request.request_id, result.store_spec, True)
        m.end_step()
        m.receipt(1, result.store_spec.gen, m.table.all_ranks, 0)
        assert m.units.units[next(iter(m.units.units))].complete
    assert not m.units.events["revoke_build_failed"]
