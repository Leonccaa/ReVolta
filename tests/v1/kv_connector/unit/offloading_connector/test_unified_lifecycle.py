# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-loss and publication sequences through the real CPU scheduler."""

import time

import pytest

from tests.v1.core.long_context_kv_utils import SPAN, host_step_receipts
from tests.v1.core.test_long_context_admission import build, make_requests
from tests.v1.core.test_long_context_host_main_kv import host_settings
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


def scheduler_with_connector():
    scheduler = build(
        host_settings(host_pool_tokens=128 * SPAN),
        max_num_seqs=2,
        with_connector=True,
    )
    assert scheduler.connector.connector_scheduler.unified.attached
    return scheduler


def advance(scheduler, *, complete_stores=True, complete_main=True):
    out = scheduler.schedule()
    ids = list(out.num_scheduled_tokens)
    meta = host_step_receipts(scheduler, out) or OffloadingWorkerMetadata()
    if not complete_main:
        meta.seal_results = {}
    connector = scheduler.connector.connector_scheduler
    if complete_stores:
        mask = scheduler.host_history_pool.host_pages.all_ranks
        meta.job_results = {
            (jid, status.generation): (mask, 0)
            for jid, status in connector._jobs.items()
            if status.generation is not None
        }
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[[] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=KVConnectorOutput(kv_connector_worker_meta=meta),
        ),
    )
    return out


def start_with_inflight_boundary():
    scheduler = scheduler_with_connector()
    (req,) = make_requests(1, 8 * SPAN, 16)
    scheduler.add_request(req)
    connector = scheduler.connector.connector_scheduler
    for _ in range(8):
        advance(scheduler, complete_stores=False)
        if connector.unified.builds:
            break
    assert connector.unified.builds and connector.unified.jobs
    return scheduler, connector, req


def test_preemption_without_resume_releases_unused_and_drains_dma():
    scheduler, connector, req = start_with_inflight_boundary()
    m = connector.unified
    credit = m.ledger.requests[req.request_id]
    assert credit.unused > 0 and credit.owned > 0
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    assert credit.draining and credit.unused == 0
    assert m.ledger.snapshot()["build_reserved"] == 0
    assert all(b.failed for b in m.builds.values())
    # No resumption, no scheduler step and no worker receipt for a long wait.
    for _ in range(100):
        assert m.ledger.snapshot()["build_reserved"] == 0
        assert m.ledger.snapshot()["build_draining"] == credit.owned > 0
        assert all(m.table.get(h).io for _, _, _, hs, _ in m.jobs.values() for h in hs)
    mask = m.table.all_ranks
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                job_results={
                    (jid, job.generation): (mask, 0)
                    for jid, job in connector._jobs.items()
                }
            ),
        )
    )
    assert not m.jobs and not m.builds and not m.ledger.charges
    assert req.request_id not in m.ledger.requests
    assert not m.units.units


def test_resume_first_step_boundary_survives_delayed_preemption(monkeypatch):
    from unittest.mock import Mock

    scheduler = scheduler_with_connector()
    (req,) = make_requests(1, 8 * SPAN, 16)
    scheduler.add_request(req)
    advance(scheduler)
    connector = scheduler.connector.connector_scheduler
    m = connector.unified
    old_generation = m.ledger.requests[req.request_id].generation
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    handoffs = Mock(wraps=connector._queue_boundary_state_offloads)
    monkeypatch.setattr(connector, "_queue_boundary_state_offloads", handoffs)
    # Include a committed state boundary in the very first resumed step.
    scheduler.scheduler_config.long_prefill_token_threshold = 2 * SPAN
    scheduler.scheduler_config.prefill_pressure_token_budget = 2 * SPAN
    out = advance(scheduler, complete_stores=False)
    assert req.request_id in out.num_scheduled_tokens
    assert req.request_id in out.preempted_req_ids
    assert handoffs.called
    assert m.ledger.requests[req.request_id].generation > old_generation
    assert not m.ledger.requests[req.request_id].draining
    assert m.builds and all(not b.failed for b in m.builds.values())
    assert m.jobs


@pytest.mark.parametrize("main_first", [True, False])
def test_real_publication_arrival_orders(main_first):
    scheduler = scheduler_with_connector()
    (req,) = make_requests(1, 8 * SPAN, 16)
    scheduler.add_request(req)
    connector = scheduler.connector.connector_scheduler
    m = connector.unified
    outputs = []
    for _ in range(8):
        out = advance(
            scheduler, complete_stores=not main_first, complete_main=main_first
        )
        outputs.append(out)
        if m.waiting:
            break
    assert m.waiting
    offer = next(iter(m.waiting[req.request_id].values()))
    b = m.units.units[offer.boundary]
    assert b.complete != main_first
    assert all(m.is_ready_resident(k) for k in offer.main) == main_first
    mask = m.table.all_ranks
    if main_first:
        meta = OffloadingWorkerMetadata(
            job_results={
                (jid, job.generation): (mask, 0) for jid, job in connector._jobs.items()
            }
        )
        connector.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=meta)
        )
    else:
        for out in outputs:
            scheduler.update_host_kv_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=host_step_receipts(scheduler, out),
                )
            )
    assert b.variants
    assert all(m.units.recoverable(uid, "cpu") for uid in b.variants)
    assert not m.waiting


def test_another_request_registers_distinct_eagle_ahead_branch():
    from types import SimpleNamespace

    from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
        setup_receive,
    )
    from tests.v1.kv_offload.test_unified_units import publish, store
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    from vllm.v1.kv_offload.base import make_offload_key
    from vllm.v1.kv_offload.unified.manager import UnifiedStoreManager
    from vllm.v1.request import Request

    scheduler, connector, original, _ = setup_receive(speculative_tokens=1)
    m = UnifiedStoreManager(
        SimpleNamespace(
            kv_cache_config=connector.kv_cache_config,
            extra_config={},
            cpu_page_size_per_worker=16,
        )
    )
    m.attach(scheduler.host_history_pool)
    connector.manager = connector.unified = m
    assert connector._eagle_state_shift == SPAN
    boundary = 4 * SPAN
    chain = chain_hash_from_request(original.block_hashes, 16)
    h_b = chain(connector.host_kv_hits.rule, 3)
    state = connector._req_status[original.request_id]
    components = []
    for group in connector.config.kv_group_configs:
        end = boundary // group.offloaded_block_size
        low = end - (group.sliding_window_size_in_blocks or end)
        components.extend(state.group_states[group.group_idx].offload_keys[low:end])
    assert m.admit_build(original.request_id, len(components))
    b = m.begin_boundary(original.request_id, h_b, components)
    assert b is not None
    store(m, components, req=original.request_id)
    main = [
        make_offload_key(chain(connector.host_kv_hits.rule, i), m.main_group)
        for i in range(5)
    ]
    for k in main:
        publish(m.table, k)
    fork = Request(
        request_id="other-branch",
        prompt_token_ids=original.prompt_token_ids[:boundary] + [42] * (2 * SPAN),
        sampling_params=original.sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    connector.on_new_request(fork)
    fork_chain = chain_hash_from_request(fork.block_hashes, 16)
    assert fork_chain(connector.host_kv_hits.rule, 3) == h_b
    fork_a = fork_chain(connector.host_kv_hits.rule, 4)
    assert fork_a != chain(connector.host_kv_hits.rule, 4)
    for req in (original, fork):
        req.shared_prefix_boundary = boundary
        # The real metadata entry handles a reused B at a shared junction.
        out = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={req.request_id: 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
        connector.build_connector_meta(out)
    assert len(b.uid.boundary) > 0 and len(m.units.units[b.uid].variants) == 1
    assert fork.request_id in m.waiting
    publish(m.table, make_offload_key(fork_a, m.main_group))
    variants = [m.units.units[uid] for uid in m.units.units[b.uid].variants]
    assert len(variants) == 2 and len({v.uid.ahead for v in variants}) == 2
    assert all(
        v.parent == b.uid and m.units.recoverable(v.uid, "cpu") for v in variants
    )
    assert not m.waiting


def test_delayed_preemption_keeps_fresh_resumed_load():
    from vllm.v1.core.kv_cache_utils import get_group_id

    scheduler = scheduler_with_connector()
    (req,) = make_requests(1, 8 * SPAN, 16)
    scheduler.add_request(req)
    connector = scheduler.connector.connector_scheduler
    m = connector.unified
    for _ in range(4):
        advance(scheduler)
    old_generation = m.ledger.requests[req.request_id].generation
    assert m.units.units and not m.jobs
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    manager = scheduler.kv_cache_manager
    for group_id, single in enumerate(manager.coordinator.single_type_managers):
        if group_id != scheduler.host_history_pool.host_group_id:
            single.block_pool.evict_blocks(
                {
                    block.block_id
                    for block in single.block_pool.blocks
                    if block.block_hash is not None
                    and get_group_id(block.block_hash) == group_id
                }
            )
    out = scheduler.schedule()
    assert req.request_id in out.preempted_req_ids
    assert out.kv_connector_metadata.load_jobs
    assert m.ledger.requests[req.request_id].generation > old_generation
    assert not m.ledger.requests[req.request_id].draining
    assert m._loads[req.request_id]
    assert all(not b.failed for b in m.builds.values())


@pytest.mark.parametrize("speculative_tokens", [0, 1])
def test_latest_metadata_uses_actual_resend_fallback(speculative_tokens):
    from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request

    scheduler = build(
        host_settings(host_pool_tokens=128 * SPAN),
        max_num_seqs=2,
        with_connector=True,
        speculative_tokens=speculative_tokens,
    )
    (request,) = make_requests(1, 8 * SPAN + 1, 16)
    scheduler.add_request(request)
    connector = scheduler.connector.connector_scheduler
    m = connector.unified
    assert m.latest_only
    for _ in range(12):
        advance(scheduler)
    span = connector.host_kv_hits.rule.block_size
    position = (request.num_prompt_tokens - 1) // span * span
    position -= connector._eagle_state_shift or 0
    chain = chain_hash_from_request(request.block_hashes, connector._hash_block_size)
    expected = chain(connector.host_kv_hits.rule, position // span - 1)
    variants = [u for u in m.units.units.values() if u.parent is not None]
    assert variants
    assert all(u.retention["resend"] == expected.hex() for u in variants)
    assert any(u.uid.boundary == expected for u in variants)
    assert not m.retention.branches


@pytest.mark.parametrize("latest_only", [False, True])
@pytest.mark.parametrize("speculative_tokens", [0, 1])
@pytest.mark.parametrize("prompt_extra", [0, SPAN // 2])
def test_continuous_identical_resend_hit_does_not_decrease(
    latest_only, speculative_tokens, prompt_extra
):
    from vllm.v1.request import RequestStatus

    settings = host_settings(host_pool_tokens=128 * SPAN)
    settings.latest_only = latest_only
    scheduler = build(
        settings,
        max_num_seqs=2,
        with_connector=True,
        speculative_tokens=speculative_tokens,
    )
    connector = scheduler.connector.connector_scheduler
    m = connector.unified
    hits = []
    for retry in range(4):
        (request,) = make_requests(
            1, 8 * SPAN + prompt_extra, 16, tag=f"resend-{retry}"
        )
        scheduler.add_request(request)
        for _ in range(12):
            advance(scheduler)
        status = connector._req_status[request.request_id]
        hits.append(status.num_locally_computed_tokens)
        scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
        connector.persistence_step()
        if latest_only:
            assert not m.retention.branches
    assert hits[1] > 0
    assert hits[1:] == [hits[1]] * 3


@pytest.mark.parametrize("speculative_tokens", [0, 1])
@pytest.mark.parametrize("tail_tokens", [SPAN // 2, SPAN, 2 * SPAN])
def test_short_sibling_junction_survives(speculative_tokens, tail_tokens):
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.kv_offload.unified.units import UnitId
    from vllm.v1.request import Request, RequestStatus

    scheduler = build(
        host_settings(host_pool_tokens=128 * SPAN),
        max_num_seqs=2,
        with_connector=True,
        speculative_tokens=speculative_tokens,
    )
    (parent,) = make_requests(1, 8 * SPAN, 16, tag="parent")
    scheduler.add_request(parent)
    for _ in range(12):
        advance(scheduler)
    scheduler.finish_requests([parent.request_id], RequestStatus.FINISHED_ABORTED)
    connector = scheduler.connector.connector_scheduler
    connector.persistence_step()
    common = 4 * SPAN
    child = Request(
        request_id="short-sibling",
        prompt_token_ids=parent.prompt_token_ids[:common] + [2000] * tail_tokens,
        sampling_params=parent.sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    scheduler.add_request(child)
    for _ in range(12):
        advance(scheduler)
    assert child.shared_prefix_boundary == common
    chain = chain_hash_from_request(child.block_hashes, connector._hash_block_size)
    uid = UnitId(
        chain(connector.host_kv_hits.rule, common // SPAN - 1),
        connector.unified.units.reset_epoch,
    )
    assert uid in connector.unified.retention.branches
    assert uid in connector.unified.units.units


@pytest.mark.parametrize("speculative_tokens", [0, 1])
def test_prompt_identity_includes_incomplete_hash_block(speculative_tokens):
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request, RequestStatus

    scheduler = build(
        host_settings(host_pool_tokens=128 * SPAN),
        max_num_seqs=2,
        with_connector=True,
        speculative_tokens=speculative_tokens,
    )
    connector = scheduler.connector.connector_scheduler
    (parent,) = make_requests(1, 8 * SPAN + SPAN // 2 + 1, 16, tag="parent")
    scheduler.add_request(parent)
    for _ in range(12):
        advance(scheduler)
    identity = connector._req_status[parent.request_id].prompt_identity(16)
    scheduler.finish_requests([parent.request_id], RequestStatus.FINISHED_ABORTED)
    connector.persistence_step()
    # The only differing token lies beyond all complete hash blocks, not just
    # beyond the last storage block. Length and every block hash still match.
    child = Request(
        request_id="different-last-token",
        prompt_token_ids=parent.prompt_token_ids[:-1] + [2000],
        sampling_params=parent.sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    assert child.block_hashes == parent.block_hashes[: len(child.block_hashes)]
    scheduler.add_request(child)
    for _ in range(12):
        advance(scheduler)
    status = connector._req_status[child.request_id]
    assert status.prompt_identity(16) != identity
