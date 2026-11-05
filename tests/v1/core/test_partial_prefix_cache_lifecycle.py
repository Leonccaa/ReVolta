# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Review findings on the partial prefix cache, as regression tests.

Covers boundary parsing (client values and the switch-off path), page
lifetimes of copy sources and destinations (entry eviction, aborts, reset),
and the identity of partial entries (multimodal inputs, cache salt).
"""

import math
from typing import Any

import pytest

from tests.v1.core.partial_prefix_trace import (
    BLOCK,
    HASH_BLOCK,
    NUM_BLOCKS,
    allocation_trace,
    ensure_none_hash,
    make_manager,
    make_request,
    run_chunk,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.request import Request, parse_prefix_cache_boundaries

pytestmark = pytest.mark.cpu_test

PROMPT = [(7 * i + 3) % 1000 for i in range(700)]
P = 576


# --------------------------------------------------------------------------
# Boundary parsing (findings 1 and 5).


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, []),
        ((), []),
        ([], []),
        (100, [100]),
        ("100", [100]),
        ([100, 64.0, "200", 100], [64, 100, 200]),
        (
            [
                "x",
                -5,
                0,
                1000,
                10**40,
                math.inf,
                -math.inf,
                math.nan,
                1e309,
                3.5,
                True,
                None,
                [1],
                {"a": 1},
                "1e3",
                " 7 ",
                "١٢",
            ],
            [7],
        ),
        ({"a": 1}, []),
        (object(), []),
    ],
)
def test_parse_boundaries_ignores_anything_invalid(raw, expected):
    assert parse_prefix_cache_boundaries(raw, 1000) == expected


def test_parse_boundaries_reads_a_bounded_number_of_values():
    assert len(parse_prefix_cache_boundaries(list(range(1, 10_000)), 10**6)) == 64


@pytest.mark.parametrize(
    "raw", [[math.inf], [1e309], ["nan"], [10**400], "garbage", [[64]]]
)
def test_request_creation_never_fails_on_boundaries(raw):
    request = make_request("r", PROMPT, boundaries=raw)
    assert request.prefix_cache_boundaries == ()


def test_switched_off_never_parses_boundaries():
    # Finding 5: with the switch off the raw value is never interpreted.
    manager = make_manager()
    request = make_request("r", PROMPT, boundaries=[1e309, P])
    run_chunk(manager, request, 512)
    run_chunk(manager, request, 188)
    manager.free(request)
    assert request._prefix_cache_boundaries is None
    # The full switch-off trace never reads them either.
    allocation_trace(make_manager())


# --------------------------------------------------------------------------
# Page lifetimes of copy endpoints (findings 2 and 6).


def _stop_at_boundary(manager, request_id, seed):
    prompt = [(13 * j + seed) % 997 for j in range(700)]
    request = make_request(request_id, prompt, boundaries=[P])
    manager.allocate_slots(request, 512)
    request.num_computed_tokens = 512
    manager.allocate_slots(request, P - 512)
    request.num_computed_tokens = P
    return request


def _holders(manager):
    """block id -> number of live owners (requests and entries)."""
    owners: dict[int, int] = {}
    for m in manager.coordinator.single_type_managers:
        for blocks in m.req_to_blocks.values():
            for block in blocks:
                if not block.is_null:
                    owners[block.block_id] = owners.get(block.block_id, 0) + 1
    for entry in manager.partial_cache.entries.values():
        for blocks in entry.blocks:
            for block in blocks:
                owners[block.block_id] = owners.get(block.block_id, 0) + 1
    return owners


def test_entry_cap_eviction_never_recycles_pending_copy_destinations():
    # Codex repro: three requests stop at P=576 in one step; the default entry
    # cap (12 blocks, 7 per entry) evicts during materialization.
    manager = make_manager(partial_prefix_cache=True)
    requests = [_stop_at_boundary(manager, f"r{i}", i) for i in range(3)]
    manager.new_step_starts()
    copies = manager.take_block_copies()
    destinations = [dst for _, dst in copies]
    assert len(copies) == 3 * 7
    assert len(set(destinations)) == len(destinations)
    pool = manager.block_pool
    # Destinations of evicted entries stay allocated until the copy has run.
    for dst in destinations:
        assert pool.blocks[dst].ref_cnt >= 1
    assert len(manager.partial_cache.entries) == 1
    # Sources that requests release this step are guarded as well.
    for request in requests:
        manager.free(request)
    for src, _ in copies:
        assert pool.blocks[src].ref_cnt >= 1
    for _ in range(3):
        manager.new_step_starts()
    # Afterwards only the surviving entry holds pages.
    in_use = NUM_BLOCKS - 1 - pool.get_num_free_blocks()
    assert in_use == manager.partial_cache.num_blocks == 7


def test_no_page_is_a_copy_destination_while_another_holder_uses_it():
    manager = make_manager(partial_prefix_cache=True)
    for step in range(4):
        pair = [
            _stop_at_boundary(manager, f"a{step}", step),
            _stop_at_boundary(manager, f"b{step}", 50 + step),
        ]
        manager.new_step_starts()
        copies = manager.take_block_copies()
        holders = _holders(manager)
        for _, dst in copies:
            # Exactly one holder: the entry it is copied into, if it survived.
            assert holders.get(dst, 0) <= 1
        dsts = [d for _, d in copies]
        assert len(set(dsts)) == len(dsts)
        # One launch copies all pairs at once: no destination is also read.
        assert not set(dsts) & {src for src, _ in copies}
        for request in pair:
            manager.free(request)


def test_copy_on_write_target_outlives_an_abort_before_the_copy():
    manager = make_manager(partial_prefix_cache=True)
    first = make_request("first", PROMPT, boundaries=[P])
    run_chunk(manager, first, 512)
    run_chunk(manager, first, P - 512)
    run_chunk(manager, first, 700 - P)
    manager.free(first)
    manager.take_block_copies()  # the snapshot's own copies
    second = make_request("second", PROMPT[:P] + [999] * 200)
    hit_blocks, hit = manager.get_computed_blocks(second)
    assert hit == P
    manager.allocate_slots(second, 64, P, hit_blocks)
    copies = manager.take_block_copies()
    assert copies
    ids = manager.get_block_ids("second")
    # Copy-on-write targets are in the request's block table, so connectors
    # see them as newly allocated pages (and fence them) as usual.
    flat = {b for group in ids for b in group}
    assert all(dst in flat for _, dst in copies)
    manager.free(second)  # aborted before its worker step ran
    pool = manager.block_pool
    for src, dst in copies:
        assert pool.blocks[src].ref_cnt >= 1
        assert pool.blocks[dst].ref_cnt >= 1
    for _ in range(3):
        manager.new_step_starts()
    for _, dst in copies:
        assert pool.blocks[dst].ref_cnt == 0


@pytest.mark.parametrize("event", ["finished", "preempted"])
def test_no_snapshot_for_a_request_gone_before_materialization(event):
    manager = make_manager(partial_prefix_cache=True)
    request = _stop_at_boundary(manager, "r", 1)
    if event == "preempted":
        request.num_preemptions += 1
    manager.free(request)
    manager.new_step_starts()
    assert manager.take_block_copies() == []
    assert not manager.partial_cache.entries
    assert manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


def test_reset_keeps_guards_while_requests_hold_blocks():
    manager = make_manager(partial_prefix_cache=True)
    request = _stop_at_boundary(manager, "r", 1)
    manager.new_step_starts()
    copies = manager.take_block_copies()
    assert copies
    other = _stop_at_boundary(manager, "other", 2)
    # A running request blocks the reset; copy endpoints stay guarded.
    assert not manager.reset_prefix_cache()
    assert not manager.partial_cache.entries
    for src, dst in copies:
        assert manager.block_pool.blocks[dst].ref_cnt >= 1
    manager.free(request)
    manager.free(other)
    # With no request left the guards are released and the reset succeeds;
    # copies never handed to a worker are dropped with them.
    assert manager.reset_prefix_cache()
    assert manager.take_block_copies() == []
    assert manager._pending_snapshots == []
    assert manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


# --------------------------------------------------------------------------
# Identity of partial entries (finding 4).


def _mm_request(request_id, image, salt=None):
    ensure_none_hash()
    params = SamplingParams(max_tokens=4, extra_args={"prefix_cache_boundaries": [P]})
    params.update_from_generation_config({}, eos_token_id=100_000)
    tokens = list(PROMPT)
    tokens[530:562] = [7] * 32  # image placeholder tokens, identical
    feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=530, length=32),
        identifier=image,
        modality="image",
    )
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        mm_features=[feature],
        cache_salt=salt,
        block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
    )


def _snapshot_then_lookup(first, second):
    manager = make_manager(partial_prefix_cache=True)
    run_chunk(manager, first, 512)
    run_chunk(manager, first, P - 512)
    run_chunk(manager, first, len(first.prompt_token_ids) - P)
    manager.new_step_starts()
    assert len(manager.partial_cache.entries) == 1
    _, hit = manager.get_computed_blocks(second)
    manager.free(first)
    return hit


def test_different_images_with_identical_placeholders_do_not_share_entries():
    # Codex repro: B=256, P=576, images at 530..561 with equal token ids.
    first, other = _mm_request("a", "image-a"), _mm_request("b", "image-b")
    assert (
        first.block_hashes[P // HASH_BLOCK - 1]
        != (other.block_hashes[P // HASH_BLOCK - 1])
    )
    assert _snapshot_then_lookup(first, other) < P
    same = _mm_request("c", "image-a")
    assert _snapshot_then_lookup(_mm_request("a2", "image-a"), same) == P


def test_cache_salt_isolates_entries():
    hit = _snapshot_then_lookup(
        _mm_request("a", "image-a", salt="s1"), _mm_request("b", "image-a", "s2")
    )
    assert hit == 0
    hit = _snapshot_then_lookup(
        _mm_request("a", "image-a", salt="s1"), _mm_request("b", "image-a", "s1")
    )
    assert hit == P


def test_lookup_rejects_entries_whose_chained_hash_differs():
    manager = make_manager(partial_prefix_cache=True)
    first = make_request("first", PROMPT, boundaries=[P])
    run_chunk(manager, first, 512)
    run_chunk(manager, first, P - 512)
    manager.new_step_starts()
    [entry] = manager.partial_cache.entries.values()
    assert entry.key[4] == first.block_hashes[P // HASH_BLOCK - 1]
    parent = entry.key[0]
    found = manager.partial_cache.lookup(
        parent,
        first.all_token_ids,
        P // BLOCK * BLOCK,
        699,
        None,
        0,
        manager._step + 1,
        tail_hash_at=lambda length: b"other",
    )
    assert found is None


def test_resume_and_snapshot_in_one_step_never_chain_copies():
    # A request resumed from an entry (copy-on-write) in the same step as
    # other snapshots materialize: entries made this step are not hit, so no
    # page is written and read by the same copy launch.
    manager = make_manager(
        partial_prefix_cache=True, partial_prefix_cache_max_fraction=0.5
    )
    first = make_request("first", PROMPT, boundaries=[P])
    run_chunk(manager, first, 512)
    run_chunk(manager, first, P - 512)
    run_chunk(manager, first, 700 - P)
    manager.take_block_copies()
    other = _stop_at_boundary(manager, "other", 9)
    manager.new_step_starts()  # materializes other's snapshot
    second = make_request("second", PROMPT[:P] + [999] * 200, boundaries=[640])
    hit_blocks, hit = manager.get_computed_blocks(second)
    assert hit == P
    manager.allocate_slots(second, 64, P, hit_blocks)
    copies = manager.take_block_copies()
    assert len(copies) == 7 + 5
    assert not {dst for _, dst in copies} & {src for src, _ in copies}
    for request in (first, other, second):
        manager.free(request)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_copy_blocks_held_until_the_queued_batch_has_executed(depth):
    """Copy blocks stay referenced until the batch that emitted the copy has
    finished, for every batch-queue depth the engine allows (the SM70 queue
    depth override can raise it above the default two)."""
    from collections import deque
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCore

    m = make_manager(partial_prefix_cache=True, max_concurrent_batches=depth)
    state: dict[str, Any] = {
        "scheduled": 0,
        "completed": 0,
        "source": None,
        "premature": [],
    }

    class Completion:
        def __init__(self, step):
            self.step, self.finished = step, False

        def done(self):
            return self.finished

        def result(self):
            self.finished = True
            state["completed"] = max(state["completed"], self.step)
            return SimpleNamespace()

    def schedule():
        state["scheduled"] += 1
        step = state["scheduled"]
        m.new_step_starts()
        if step == 1:
            [source] = m.block_pool.get_new_blocks(1)
            state["source"] = source
            m._defer_release([source])
        elif state["completed"] < 1 and state["source"].ref_cnt == 0:
            state["premature"].append((step, state["completed"]))
        return SimpleNamespace(
            total_num_scheduled_tokens=1,
            pending_structured_output_tokens=False,
            step=step,
        )

    last: list = [None]

    def execute(output, non_block):
        last[0] = Completion(output.step)
        return last[0]

    engine = object.__new__(EngineCore)
    engine.batch_queue = deque(maxlen=depth)
    engine.batch_queue_size = depth
    engine._sm70_async_cpu_trace_step = 0
    engine.scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=schedule,
        get_grammar_bitmask=lambda output: None,
        update_from_output=lambda output, result: {},
    )
    engine.model_executor = SimpleNamespace(
        execute_model=execute, sample_tokens=lambda grammar, non_block: last[0]
    )
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine.use_spec_decode = False
    engine.log_error_detail = lambda output: nullcontext()
    engine.log_iteration_details = lambda output: nullcontext()
    engine._process_aborts_queue = lambda: None
    for _ in range(depth + 4):
        EngineCore.step_with_batch_queue(engine)

    assert state["completed"] >= 1
    assert state["source"].ref_cnt == 0  # released eventually
    assert state["premature"] == []
