# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-window ownership across admission, window advance and cache reset."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.v1.core.partial_prefix_trace import make_request
from tests.v1.kv_connector.unit.offloading_connector import (
    test_cpu_draft_offload as h,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _complete(scheduler, jobs):
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=h.OffloadingWorkerMetadata(
                completed_jobs={jid: 1 for jid in jobs}
            )
        )
    )


@pytest.fixture(params=["small", "production"])
def geometry(request, monkeypatch):
    if request.param == "production":
        for key, value in dict(
            TARGET_BLOCK=1664,
            DRAFT_BLOCK=832,
            DRAFT_WINDOW_TOKENS=2880,
            DRAFT_WINDOW_BLOCKS=4,
            PROMPT=20000,
        ).items():
            monkeypatch.setattr(h, key, value)
    return request.param


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("delayed_preempt", [False, True])
def test_readmission_load_advance_then_reset(geometry, grouped, delayed_preempt):
    scheduler = h._make("window", manager=h._cpu_manager(grouped))
    if geometry == "small":
        seed = h._standard_run(scheduler, complete=True)
    else:
        seed = h.Run(
            scheduler, h._request("r1", h.PROMPT, junction=16640), complete=True
        )
        chunks = [(4096, ())] * 4 + [
            (256, (16640,)),
            (1664, (18304,)),
            (1664, (19968,)),
            (32, ()),
        ]
        for tokens, boundaries in chunks:
            seed.step(tokens, boundaries=boundaries)
    seed.finish()
    req = h._sibling(scheduler, num_tokens=h.PROMPT + 2 * h.TARGET_BLOCK)
    load_hit = h.PROMPT // h.TARGET_BLOCK * h.TARGET_BLOCK
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == load_hit
    # An external preemption frees the old admission before its ID is
    # delivered. The next schedule can already have admitted a fresh load.
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == load_hit
    ids = h._block_ids(req.num_tokens)
    first_loaded = load_hit // h.DRAFT_BLOCK - h.DRAFT_WINDOW_BLOCKS
    ids[h.DRAFT][:first_loaded] = [0] * first_loaded
    ids[h.MAMBA][load_hit // h.TARGET_BLOCK - 1] = 600
    null = h.KVCacheBlock(block_id=0, is_null=True)
    blocks = h.KVCacheBlocks(
        tuple(
            [null if bid == 0 else h.KVCacheBlock(block_id=bid) for bid in group]
            for group in ids
        )
    )
    scheduler.update_state_after_alloc(req, blocks, load_hit)
    req.num_computed_tokens = load_hit
    state = scheduler._req_status[req.request_id]
    before = set(state.draft_reofferable[h.DRAFT])
    skipped = scheduler.num_skipped_draft_blocks
    meta = scheduler.build_connector_meta(
        h._output(
            req,
            0,
            preempted_req_ids={req.request_id} if delayed_preempt else set(),
        )
    )
    after = set(state.draft_reofferable.get(h.DRAFT, ()))
    assert meta.load_jobs.keys().isdisjoint(meta.jobs_to_flush)
    _complete(scheduler, meta.load_jobs)
    # Move the frontier past the loaded window before resetting the CPU tier.
    meta = scheduler.build_connector_meta(h._output(req, h.DRAFT_BLOCK, new=ids))
    _complete(scheduler, meta.store_jobs)
    req.num_computed_tokens += h.DRAFT_BLOCK
    scheduler.reset_cache()
    boundary = load_hit + h.TARGET_BLOCK
    meta = scheduler.build_connector_meta(
        h._output(req, h.DRAFT_BLOCK, cached=True, boundaries=(boundary,))
    )
    sources = []
    for job in meta.store_jobs.values():
        src = job.transfer_spec[0]
        if count := src.group_sizes[h.DRAFT]:
            sources.extend(src.block_ids.tolist()[-count:])
    _complete(scheduler, meta.store_jobs)
    sibling = h._request("r3", req.num_tokens, hashes=list(req.block_hashes))
    scheduler.on_new_request(sibling)
    assert scheduler.get_num_new_matched_tokens(sibling, 0)[0] == boundary
    assert before == after
    end = boundary // h.DRAFT_BLOCK
    assert sources == ids[h.DRAFT][end - h.DRAFT_WINDOW_BLOCKS : end]
    assert scheduler.num_skipped_draft_blocks == skipped


@pytest.mark.parametrize("pages", [16, 64, 128])
@pytest.mark.parametrize("grouped", [False, True])
def test_periodic_history_is_bounded_by_held_blocks(geometry, pages, grouped):
    cfg = replace(h._kv_config(), num_blocks=1024)
    core = KVCacheManager(
        cfg,
        max_model_len=h.TARGET_BLOCK * (pages + 1) + 32,
        hash_block_size=16,
        enable_caching=True,
        use_eagle=False,
        prefix_cache_retention_interval=h.TARGET_BLOCK,
    )
    config = h._vllm_config("window")
    config.cache_config.hash_block_size = 16
    config.cache_config.prefix_cache_retention_interval = h.TARGET_BLOCK
    spec = h.MockOffloadingSpec(config, cfg)
    spec.manager = h._cpu_manager(grouped)
    off = h.OffloadingConnectorScheduler(spec)
    tokens = list(range(h.TARGET_BLOCK * (pages + 1) + 32))
    req = make_request("long", tokens)
    off.on_new_request(req)
    assert off.get_num_new_matched_tokens(req, 0)[0] == 0
    state = off._req_status[req.request_id]
    gc = next(g for g in off.config.kv_group_configs if g.group_idx == h.DRAFT)
    for i in range(pages):
        allocated = core.allocate_slots(req, h.TARGET_BLOCK)
        assert allocated is not None
        handoffs = core.take_boundary_state_offloads()
        assert len(handoffs[req.request_id]) == 1
        output = h._output(
            req,
            h.TARGET_BLOCK,
            new=core.get_blocks(req.request_id).get_block_ids() if i == 0 else None,
            cached=i != 0,
        )
        if i:
            output.scheduled_cached_reqs.new_block_ids = [allocated.get_block_ids()]
        output.boundary_state_offloads = handoffs
        meta = off.build_connector_meta(output)
        owned = {
            b.block_id
            for b in core.get_blocks(req.request_id).blocks[h.DRAFT]
            if not b.is_null
        }
        for job in meta.store_jobs.values():
            src = job.transfer_spec[0]
            if count := src.group_sizes[h.DRAFT]:
                assert set(src.block_ids.tolist()[-count:]) <= owned
        first_held = off._first_held_draft_block(state, gc)
        tracked = state.draft_reofferable[h.DRAFT]
        assert all(b >= first_held for b in tracked)
        # The current batch may accept two more blocks before its output.
        assert len(tracked) <= h.DRAFT_WINDOW_BLOCKS + 2
        assert len(state.draft_window_boundaries) <= h.DRAFT_WINDOW_BLOCKS + 2
        # Each page hands off a needed window and all offers are accepted.
        # Retiring those records creates no new skipped obligations.
        assert off.num_skipped_draft_blocks == 0
        req.num_computed_tokens += h.TARGET_BLOCK
        _complete(off, meta.store_jobs)
        core.new_step_starts()
    # A long history still supports a reset and the next state hand-off.
    off.reset_cache()
    allocated = core.allocate_slots(req, h.TARGET_BLOCK)
    assert allocated is not None
    output = h._output(req, h.TARGET_BLOCK, cached=True)
    output.scheduled_cached_reqs.new_block_ids = [allocated.get_block_ids()]
    output.boundary_state_offloads = core.take_boundary_state_offloads()
    meta = off.build_connector_meta(output)
    owned = {
        b.block_id
        for b in core.get_blocks(req.request_id).blocks[h.DRAFT]
        if not b.is_null
    }
    for job in meta.store_jobs.values():
        src = job.transfer_spec[0]
        if count := src.group_sizes[h.DRAFT]:
            assert set(src.block_ids.tolist()[-count:]) <= owned
    _complete(off, meta.store_jobs)
    sibling = make_request("sibling", tokens)
    off.on_new_request(sibling)
    assert off.get_num_new_matched_tokens(sibling, 0)[0] == (
        h.TARGET_BLOCK * (pages + 1)
    )
    req.status = RequestStatus.FINISHED_ABORTED
    off.request_finished(req)
    core.free(req)
    assert not state.draft_reofferable
    assert not state.draft_window_boundaries
    assert off.num_skipped_draft_blocks == 0


@pytest.mark.parametrize("grouped", [False, True])
def test_optimistic_advance_rollback_then_reset_keeps_held_window(grouped):
    scheduler = h._make("window", manager=h._cpu_manager(grouped))
    req = h._request("r1", h.PROMPT, junction=160)
    run = h.Run(scheduler, req, complete=True)
    run.step(128)
    run.step(16)
    state = scheduler._req_status[req.request_id]
    before = set(state.draft_reofferable[h.DRAFT])
    # Only 144 tokens have been processed. Pending speculative work may roll
    # back, so it cannot permanently retire accepted blocks 7 and 8.
    req.num_computed_tokens = 192
    req.num_in_flight_tokens = 48
    run.step(0)
    assert before <= state.draft_reofferable[h.DRAFT]
    req.num_computed_tokens = 144
    req.num_in_flight_tokens = 0
    scheduler.reset_cache()
    run.step(16, boundaries=(160,))
    assert scheduler.get_num_new_matched_tokens(h._sibling(scheduler), 0)[0] == 160
    assert scheduler.num_skipped_draft_blocks == 7
    run.finish()
    # Optimistically scanned blocks 10 and 11 were rejected and never
    # recomputed; finish settles each remaining obligation once.
    assert scheduler.num_skipped_draft_blocks == 9
    scheduler.request_finished(req)
    assert scheduler.num_skipped_draft_blocks == 9
