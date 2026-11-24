# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inventory admission requires an exact source in the current step."""

from types import SimpleNamespace

import pytest

from tests.v1.core.long_context_kv_utils import SPAN
from tests.v1.core.test_long_context_admission import make_requests
from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
    setup_receive,
)
from tests.v1.kv_offload.test_unified_units import publish
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
    assert m.events["boundary_reject_state"] == 1
    assert not m.events["boundary_reject_inventory"]
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


def test_rejected_boundary_does_not_block_another_requests_exact_source():
    from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
    from vllm.v1.kv_offload.base import make_offload_key

    scheduler, connector, rejected, _ = setup_receive(speculative_tokens=1)
    [accepted] = make_requests(1, rejected.num_tokens, 1, tag="accepted")
    connector.on_new_request(accepted)
    m = UnifiedStoreManager(
        SimpleNamespace(
            kv_cache_config=connector.kv_cache_config,
            extra_config={},
            cpu_page_size_per_worker=16,
        )
    )
    m.attach(scheduler.host_history_pool)
    connector.manager = connector.unified = m
    boundary = 4 * SPAN
    status = connector._req_status[accepted.request_id]
    status.update_offload_keys()
    sources = []
    for group in connector.config.kv_group_configs:
        end = boundary // group.offloaded_block_size
        keys = status.group_states[group.group_idx].offload_keys
        if group.requires_exact_boundary_source:
            sources.append((group.group_idx, 123 + group.group_idx, boundary))
        elif group.draft_window_tokens is not None:
            for key in keys[max(0, end - group.sliding_window_size_in_blocks) : end]:
                publish(m.table, key, m.kind(key))
    assert len(sources) > 1
    chain = chain_hash_from_request(accepted.block_hashes, 16)
    for i in range(5):
        key = make_offload_key(chain(connector.host_kv_hits.rule, i), m.main_group)
        if not m.is_ready_resident(key):
            publish(m.table, key)
    out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={
            rejected.request_id: boundary,
            accepted.request_id: boundary,
        },
        total_num_scheduled_tokens=2 * boundary,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        boundary_state_offloads={
            rejected.request_id: [(g, bid + 1000, b) for g, bid, b in sources[:1]],
            accepted.request_id: sources,
        },
    )
    meta = connector.build_connector_meta(out)
    assert m.events["boundary_reject_state"] == 1
    assert len(meta.store_jobs) == len(sources)
    assert {job.req_id for job in meta.store_jobs.values()} == {accepted.request_id}
    assert {
        int(job.transfer_spec[0].block_ids[0]) for job in meta.store_jobs.values()
    } == {block_id for _, block_id, _ in sources}
    assert not m.units.events["revoke_build_failed"]
    [build] = m.builds.values()
    unit = m.units.units[build.uid]
    assert not unit.complete and not unit.variants
    for job_id, job in meta.store_jobs.items():
        m.receipt(job_id, job.generation, m.table.all_ranks, 0)
    assert unit.complete and len(unit.variants) == 1
