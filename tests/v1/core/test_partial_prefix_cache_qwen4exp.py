# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partial prefix-cache hits on the Qwen4Exp (Flash-Next) cache layout.

Groups: DCP2-sharded main K/V, compressed indexer keys (ratio 4), the QSA raw
key ring (per-request scratch), GDN and PLE ``align`` states, and the DFlash2
drafter's sliding window, at a 256-token global block (1664 in production).
"""

import json
from pathlib import Path

import pytest

from tests.v1.core.partial_prefix_trace import (
    BLOCK,
    DCP,
    DRAFT_BLOCK,
    GROUP_NAMES,
    NUM_BLOCKS,
    WINDOW,
    allocation_trace,
    make_manager,
    make_request,
    run_chunk,
    split_trace,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.request import PREFIX_CACHE_BOUNDARY_ALIGNMENT

pytestmark = pytest.mark.cpu_test

MAIN, COMPRESSED, RING, GDN, PLE, DRAFTER = range(len(GROUP_NAMES))
GOLDEN = json.loads(
    (Path(__file__).parent / "partial_prefix_rc1_golden.json").read_text()
)
PROMPT = [(7 * i + 3) % 1000 for i in range(700)]
# 2 * 256 + 64: inside the third block, a multiple of the GDN chunk.
P = 576


def _normalize(value):
    return json.loads(json.dumps(value))


# --------------------------------------------------------------------------
# Switched off: behaviour of the base tree.


@pytest.mark.parametrize("nsb", [0, 2])
def test_switched_off_allocation_matches_base_tree(nsb):
    manager = make_manager(num_speculative_blocks=nsb)
    assert manager.partial_cache is None
    key = "alloc_nsb2" if nsb else "alloc"
    assert _normalize(allocation_trace(manager)) == GOLDEN[key]
    assert manager.take_block_copies() == []


def test_switched_off_chunking_matches_base_tree():
    # No partial cache: boundaries carried by requests add no chunk stops.
    assert split_trace() == GOLDEN["split"]
    assert (
        split_trace(partial_cache=None, stops=lambda r, s, e: (576,))
        == (GOLDEN["split"])
    )


def test_switched_off_ignores_request_boundaries():
    manager = make_manager()
    request = make_request("r", PROMPT, boundaries=[P])
    assert request.prefix_cache_boundaries == (P,)
    assert manager.partial_prefix_stops(request, 0, 700) == ()
    run_chunk(manager, request, 512)
    run_chunk(manager, request, 64)
    assert manager._pending_snapshots == []
    assert manager.take_block_copies() == []


def test_switched_on_splits_at_boundaries():
    split = split_trace(partial_cache=object(), stops=lambda r, s, e: (576,))
    assert split != GOLDEN["split"]
    # From 512 with room for 300 tokens the chunk ends at the boundary.
    assert 64 in split


# --------------------------------------------------------------------------
# Layout and enablement.


def test_layout_enables_partial_cache_with_ring_as_scratch():
    manager = make_manager(partial_prefix_cache=True)
    managers = manager.coordinator.single_type_managers
    assert [type(m) for m in managers] == [
        FullAttentionManager,
        FullAttentionManager,
        CircularBufferManager,
        MambaManager,
        MambaManager,
        SlidingWindowManager,
    ]
    # DCP2: the sharded main K/V spans the global block; the rest is replicated.
    assert [m.block_size for m in managers] == [BLOCK, BLOCK, 16, BLOCK, BLOCK, 128]
    assert [m.dcp_world_size for m in managers] == [DCP, 1, 1, 1, 1, 1]
    assert manager.partial_cache is not None
    assert manager._partial_scratch == (False, False, True, False, False, False)
    assert manager.partial_block_size == BLOCK
    assert manager.partial_cache.max_blocks == int(NUM_BLOCKS * 0.125)


def test_replayed_speculative_states_disable_partial_cache():
    manager = make_manager(
        partial_prefix_cache=True,
        num_speculative_blocks=1,
        num_replayed_speculative_blocks=1,
    )
    assert manager.partial_cache is None


def test_pool_fraction_is_configurable():
    manager = make_manager(
        partial_prefix_cache=True, partial_prefix_cache_max_fraction=0.5
    )
    assert manager.partial_cache.max_blocks == NUM_BLOCKS // 2


def test_boundaries_round_down_to_gdn_chunks():
    align = PREFIX_CACHE_BOUNDARY_ALIGNMENT
    assert align == 64
    request = make_request("r", PROMPT, boundaries=[600, 63, 640, 699, 700])
    assert request.prefix_cache_boundaries == (576, 640)
    # Compressed keys close a group every 4 tokens: boundaries never split one.
    assert all(b % 4 == 0 for b in request.prefix_cache_boundaries)
    # Malformed client values are ignored, not an error.
    assert make_request("bad", PROMPT, boundaries=["x"]).prefix_cache_boundaries == ()
    assert make_request("bad", PROMPT, boundaries=5).prefix_cache_boundaries == ()


# --------------------------------------------------------------------------
# Snapshot, lookup and copy-on-write.


def _snapshot(manager, prompt=PROMPT, boundary=P):
    first = make_request("first", prompt, boundaries=[boundary])
    assert manager.partial_prefix_stops(first, 512, 700) == (boundary,)
    run_chunk(manager, first, 512)
    blocks_before = [list(b) for b in manager.get_block_ids("first")]
    run_chunk(manager, first, boundary - 512)  # snapshot materializes here
    copies = manager.take_block_copies()
    return first, blocks_before, copies


@pytest.mark.parametrize("nsb", [0, 2])
def test_snapshot_copies_each_group_and_skips_the_ring(nsb):
    manager = make_manager(partial_prefix_cache=True, num_speculative_blocks=nsb)
    first, _, copies = _snapshot(manager)
    [entry] = manager.partial_cache.entries.values()
    assert entry.length == P
    ids = manager.get_block_ids("first")
    last = (P - 1) // BLOCK
    sources = {
        MAIN: [ids[MAIN][last]],
        COMPRESSED: [ids[COMPRESSED][last]],
        RING: [],
        GDN: [ids[GDN][last]],
        PLE: [ids[PLE][last]],
        # The whole drafter window before P: blocks 2..4 of 128 tokens.
        DRAFTER: ids[DRAFTER][
            (P - (WINDOW - 1)) // DRAFT_BLOCK : (P - 1) // DRAFT_BLOCK + 1
        ],
    }
    assert len(sources[DRAFTER]) == 3
    expected = [
        (src, dst.block_id)
        for group in range(len(GROUP_NAMES))
        for src, dst in zip(sources[group], entry.blocks[group])
    ]
    assert copies == expected
    assert entry.blocks[RING] == []
    assert entry.copy_last == [True, True, False, True, True, True]
    # Entry blocks are cache-owned and outside the hash table.
    for blocks in entry.blocks:
        for block in blocks:
            assert block.ref_cnt == 1 and block.block_hash is None
    # Materialized at the start of this step; its copies run in this step's
    # worker pass, so lookups in this step must not use it.
    assert entry.ready_step == manager._step
    manager.free(first)


@pytest.mark.parametrize("nsb", [0, 2])
def test_partial_hit_resumes_at_boundary_with_copy_on_write(nsb):
    manager = make_manager(partial_prefix_cache=True, num_speculative_blocks=nsb)
    first, _, _ = _snapshot(manager)
    first_ids = [list(b) for b in manager.get_block_ids("first")]
    run_chunk(manager, first, 700 - P)
    [entry] = manager.partial_cache.entries.values()
    entry_ids = [[b.block_id for b in blocks] for blocks in entry.blocks]

    second = make_request("second", PROMPT[:P] + [999] * 200, boundaries=[700])
    hit_blocks, hit = manager.get_computed_blocks(second)
    assert hit == P
    fallback_blocks, aligned = second.partial_prefix_fallback
    # Retention 0 keeps no recurrent state at 256/512: the aligned hit is short.
    assert aligned < P and aligned % BLOCK == 0
    got = hit_blocks.get_block_ids()
    null = manager.block_pool.null_block.block_id
    # Attention groups: the cached full blocks, then the entry's partial page.
    for group in (MAIN, COMPRESSED):
        assert got[group] == first_ids[group][:2] + entry_ids[group]
    for group in (GDN, PLE):
        assert got[group] == [null, null] + entry_ids[group]
    first_window = (P - (WINDOW - 1)) // DRAFT_BLOCK
    assert got[DRAFTER] == [null] * first_window + entry_ids[DRAFTER]
    assert got[RING] == []

    blocks = manager.allocate_slots(
        second, 64, num_new_computed_tokens=hit, new_computed_blocks=hit_blocks
    )
    assert blocks is not None
    copies = manager.take_block_copies()
    ids = manager.get_block_ids("second")
    # Copy-on-write: the request writes private copies of the partial pages.
    expected = []
    for group in (MAIN, COMPRESSED, GDN, PLE, DRAFTER):
        private = ids[group][len(got[group]) - 1]
        assert private != entry_ids[group][-1]
        expected.append((entry_ids[group][-1], private))
    assert sorted(copies) == sorted(expected)
    # Earlier drafter window blocks are full and shared read-only.
    assert ids[DRAFTER][first_window : first_window + 2] == entry_ids[DRAFTER][:2]
    # The ring is claimed fresh, never copied.
    assert len(ids[RING]) == 1 and ids[RING][0] not in sum(entry_ids, [])
    # The entry itself is unchanged and stays cached.
    assert [[b.block_id for b in blocks] for blocks in entry.blocks] == entry_ids
    assert manager.partial_cache.contains(entry.key)
    # The private partial page is cached normally once it fills.
    for group in (MAIN, COMPRESSED):
        mgr = manager.coordinator.single_type_managers[group]
        assert mgr.num_cached_block["second"] == 2
    manager.free(first)
    manager.free(second)


def test_dcp2_block_index_follows_global_tokens():
    # With DCP2 every rank holds BLOCK // 2 local slots of global block
    # t // BLOCK; the scheduler hands each rank the same block ids.
    manager = make_manager(partial_prefix_cache=True)
    main = manager.coordinator.single_type_managers[MAIN]
    assert main.block_size == BLOCK and main.kv_cache_spec.block_size == BLOCK // DCP
    first, _, copies = _snapshot(manager)
    ids = manager.get_block_ids("first")
    assert copies[0][0] == ids[MAIN][P // BLOCK]
    # Each rank's partial page holds (P % BLOCK) / DCP filled slots.
    assert (P % BLOCK) % DCP == 0
    manager.free(first)


def test_entry_is_ready_only_after_its_copy_step():
    manager = make_manager(partial_prefix_cache=True)
    first = make_request("first", PROMPT, boundaries=[P])
    run_chunk(manager, first, 512)
    manager.allocate_slots(first, P - 512)
    first.num_computed_tokens = P
    second = make_request("second", PROMPT[:P] + [5] * 100)
    # The snapshot is only queued; the next step materializes it.
    assert manager.get_computed_blocks(second)[1] != P
    manager.new_step_starts()
    assert manager.get_computed_blocks(second)[1] != P
    manager.new_step_starts()
    assert manager.get_computed_blocks(second)[1] == P


def test_lookup_respects_tail_salt_and_longest_match():
    manager = make_manager(partial_prefix_cache=True)
    first, _, _ = _snapshot(manager)
    run_chunk(manager, first, 700 - P)
    other = list(PROMPT[:P])
    other[530] += 1
    assert manager.get_computed_blocks(make_request("x", other + [1] * 50))[1] < P
    short = make_request("y", PROMPT[: P - 1] + [123] * 50)
    assert manager.get_computed_blocks(short)[1] < P
    manager.free(first)


def test_entries_are_lru_bounded_and_yield_to_running_requests():
    manager = make_manager(
        partial_prefix_cache=True, partial_prefix_cache_max_fraction=0.2
    )
    cache = manager.partial_cache
    entry_blocks = None
    keys = []
    for i in range(4):
        prompt = [(13 * j + i) % 997 for j in range(700)]
        request = make_request(f"r{i}", prompt, boundaries=[P])
        run_chunk(manager, request, 512)
        run_chunk(manager, request, P - 512)
        manager.free(request)
        manager.new_step_starts()
        manager.new_step_starts()
        manager.new_step_starts()
        keys.append(next(reversed(cache.entries)))
        entry_blocks = cache.entries[keys[-1]].num_blocks
        assert cache.num_blocks <= max(cache.max_blocks, entry_blocks)
    # 7 blocks per entry (4 single-block groups + a 3-block drafter window),
    # bound 19: two entries survive.
    assert entry_blocks == 7
    assert list(cache.entries) == keys[-2:]
    # The entry a request is resuming from is never reclaimed for it.
    free = manager.block_pool.get_num_free_blocks()
    hit_entry = cache.entries[keys[-2]]
    assert not manager._reclaim_partial_blocks(free + 100, 0, keep=hit_entry)
    assert list(cache.entries) == [keys[-2]]
    # Allocation reclaims entry blocks, oldest first, before giving up.
    cache.clear()
    for i in range(2):
        prompt = [(13 * j + 50 + i) % 997 for j in range(700)]
        request = make_request(f"s{i}", prompt, boundaries=[P])
        run_chunk(manager, request, 512)
        run_chunk(manager, request, P - 512)
        manager.free(request)
        for _ in range(3):
            manager.new_step_starts()
    keys = list(cache.entries)
    assert len(keys) == 2
    free = manager.block_pool.get_num_free_blocks()
    assert manager._reclaim_partial_blocks(free + 7, 0)
    assert list(cache.entries) == keys[-1:]
    assert not manager._reclaim_partial_blocks(free + 100, 0)
    assert not cache.entries and cache.num_blocks == 0
    # Reserved blocks count against the free pool as before.
    assert not manager._reclaim_partial_blocks(1, NUM_BLOCKS)


def test_reset_prefix_cache_releases_entries():
    manager = make_manager(partial_prefix_cache=True)
    first, _, _ = _snapshot(manager)
    manager.free(first)
    for _ in range(4):
        manager.new_step_starts()
    assert manager.partial_cache.entries
    assert manager.reset_prefix_cache()
    assert not manager.partial_cache.entries
    assert manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


def test_reasoning_requests_cache_output_blocks():
    # The thinking-mode output exclusion from upstream is not ported: our
    # template keeps reasoning in later turns, so outputs stay cacheable.
    manager = make_manager(partial_prefix_cache=True)
    request = make_request("r", PROMPT[:250], reasoning_ended=False)
    run_chunk(manager, request, 250)
    for token in range(10):
        request.append_output_token_ids(token)
        run_chunk(manager, request, 1)
    main = manager.coordinator.single_type_managers[MAIN]
    assert main.num_cached_block["r"] == 1
    assert main.req_to_blocks["r"][0].block_hash is not None


def test_boundary_state_handoffs_stay_block_aligned():
    # Partial pages and state copies never become connector hand-offs; only
    # full-block boundaries are offered.
    manager = make_manager(partial_prefix_cache=True)
    first, _, _ = _snapshot(manager)
    run_chunk(manager, first, 700 - P)
    manager.take_boundary_state_offloads()
    second = make_request("second", PROMPT[:P] + [999] * 400)
    hit_blocks, hit = manager.get_computed_blocks(second)
    assert hit == P
    run_chunk(manager, second, BLOCK * 3 - P, hit, hit_blocks)
    entry_ids = {
        b.block_id
        for entry in manager.partial_cache.entries.values()
        for blocks in entry.blocks
        for b in blocks
    }
    offers = manager.take_boundary_state_offloads()
    for request_offers in offers.values():
        for _, block_id, boundary in request_offers:
            assert boundary % BLOCK == 0
            assert block_id not in entry_ids
    manager.free(first)
    manager.free(second)
