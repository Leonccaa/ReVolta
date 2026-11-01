# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partial prefix-cache hits next to the offloading connector.

A partial hit ends inside a block; the connector only loads and stores whole
blocks. The scheduler asks the connector from the block-aligned hit ``A`` the
entry extends and keeps the partial hit unless ``A + external`` reaches
further (``Scheduler._resolve_partial_prefix_hit``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
    SchedulerOffloadConfig,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_offload.base import OffloadingManager, make_offload_key

pytestmark = pytest.mark.cpu_test

BLOCK = 16
NUM_TOKENS = 100
# The host tier holds attention blocks 0..3 and the recurrent state at 64.
STORED_BLOCKS = 4


def _scheduler(rule: str = "extend") -> OffloadingConnectorScheduler:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.config = SchedulerOffloadConfig(
        kv_group_configs=(
            GroupOffloadConfig(
                group_idx=0,
                gpu_block_size=BLOCK,
                offloaded_block_size=BLOCK,
                hash_block_size_factor=1,
                sliding_window_size_in_blocks=None,
            ),
            GroupOffloadConfig(
                group_idx=1,
                gpu_block_size=BLOCK,
                offloaded_block_size=BLOCK,
                hash_block_size_factor=1,
                sliding_window_size_in_blocks=1,
                requires_exact_boundary_source=True,
            ),
        ),
        block_size_factor=1,
        num_workers=1,
        offload_prompt_only=True,
        num_kv_cache_groups=2,
        superseded_state_rule=rule,
    )
    scheduler._group_config_by_idx = {
        group.group_idx: group for group in scheduler.config.kv_group_configs
    }
    scheduler._sliding_window_groups = (1,)
    scheduler._lookup_groups = (0, 1)
    scheduler._mamba_align_size = BLOCK
    scheduler._eagle_state_shift = None
    scheduler._req_status = {}
    scheduler.num_demoted_states = 0
    scheduler._current_batch_load_jobs = {}
    scheduler._current_batch_jobs_to_flush = set()
    scheduler._blocks_being_loaded = set()
    scheduler._job_counter = 0
    scheduler._stale_job_threshold = 0
    scheduler._jobs = {}
    scheduler._block_id_to_pending_jobs = {}

    hashes = [BlockHash(f"h{i}".encode()) for i in range(NUM_TOKENS // BLOCK)]
    stored = {make_offload_key(hashes[i], 0) for i in range(STORED_BLOCKS)}
    stored.add(make_offload_key(hashes[STORED_BLOCKS - 1], 1))
    manager = MagicMock(spec=OffloadingManager)
    manager.lookup.side_effect = lambda key, req_context: key in stored
    manager.on_new_request.return_value = SimpleNamespace(policy=None)
    manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    scheduler.manager = manager

    request = MagicMock()
    request.request_id = "req"
    request.kv_transfer_params = None
    request.block_hashes = hashes
    request.num_computed_tokens = 0
    request.num_prompt_tokens = NUM_TOKENS
    request.num_tokens = NUM_TOKENS
    request.skip_reading_prefix_cache = False
    request.shared_prefix_boundary = 0
    request.is_finished.return_value = False
    scheduler.on_new_request(request)
    scheduler.request = request
    return scheduler


def _blocks(num_blocks: int) -> KVCacheBlocks:
    # Attention keeps every block; the recurrent group only its last state.
    null = KVCacheBlock(block_id=0, is_null=True)
    return KVCacheBlocks(
        (
            [KVCacheBlock(block_id=100 + i) for i in range(num_blocks)],
            [null] * (num_blocks - 1) + [KVCacheBlock(block_id=200)],
        )
    )


def _partial_request(scheduler, aligned: int, partial: int):
    request = scheduler.request
    request.partial_prefix_entry = object()
    request.partial_prefix_fallback = (_blocks(aligned // BLOCK), aligned)
    return request


def test_unaligned_local_hit_trips_the_connector_assertion():
    # Without the scheduler's protection the connector is asked from the
    # partial hit: it reports external blocks after it and then cannot place
    # them, since loads start on a block boundary.
    scheduler = _scheduler()
    ext, _ = scheduler.get_num_new_matched_tokens(scheduler.request, 40)
    assert ext == STORED_BLOCKS * BLOCK - 40
    with pytest.raises(AssertionError):
        scheduler.update_state_after_alloc(
            scheduler.request, _blocks(STORED_BLOCKS), ext
        )


def test_longer_external_hit_replaces_the_partial_hit():
    scheduler = _scheduler()
    request = _partial_request(scheduler, aligned=16, partial=40)
    aligned_blocks = request.partial_prefix_fallback[0]
    ext, load_async = scheduler.get_num_new_matched_tokens(request, 16)
    assert ext == STORED_BLOCKS * BLOCK - 16
    blocks, local, ext, load_async = Scheduler._resolve_partial_prefix_hit(
        request, _blocks(3), 40, ext, load_async
    )
    assert blocks is aligned_blocks and local == 16
    assert ext == 48 and load_async
    assert request.partial_prefix_entry is None
    assert request.partial_prefix_fallback is None
    # The load starts on the block-aligned local hit.
    scheduler.update_state_after_alloc(request, _blocks(STORED_BLOCKS), ext)
    [job] = scheduler._current_batch_load_jobs.values()
    _, dst = job.transfer_spec
    # Attention blocks 1..3 and the single recurrent state at 64.
    assert dst.group_sizes == [3, 1]
    assert dst.block_indices == [1, 3]


@pytest.mark.parametrize("partial", [64 + 16, 88])
def test_partial_hit_reaching_further_is_kept_and_nothing_is_loaded(partial):
    scheduler = _scheduler()
    request = _partial_request(scheduler, aligned=16, partial=partial)
    partial_blocks = _blocks(partial // BLOCK + 1)
    ext, load_async = scheduler.get_num_new_matched_tokens(request, 16)
    assert 16 + ext <= partial
    blocks, local, ext, load_async = Scheduler._resolve_partial_prefix_hit(
        request, partial_blocks, partial, ext, load_async
    )
    assert blocks is partial_blocks and local == partial
    assert ext == 0 and not load_async
    assert request.partial_prefix_entry is not None
    # Nothing is read back from the host tier for a kept partial hit.
    scheduler.update_state_after_alloc(request, blocks, 0)
    assert scheduler._current_batch_load_jobs == {}
    scheduler.manager.prepare_load.assert_not_called()


def test_extend_rule_uses_the_block_aligned_hit():
    # R1 "extend": the state at the hit point is demoted once the request has
    # stored a newer state. The hit must be the block-aligned one the connector
    # served, not the partial length (88 // 16 = 5 would demote block 4).
    scheduler = _scheduler(rule="extend")
    request = _partial_request(scheduler, aligned=16, partial=88)
    ext, load_async = scheduler.get_num_new_matched_tokens(request, 16)
    Scheduler._resolve_partial_prefix_hit(request, _blocks(6), 88, ext, load_async)
    req_status = scheduler._req_status["req"]
    assert [g.num_hit_blocks for g in req_status.group_states] == [4, 4]
    req_status.stored_state_blocks = {1: {5}}
    scheduler._demote_superseded_states(req_status)
    keys = req_status.group_states[1].offload_keys
    scheduler.manager.demote.assert_called_once_with([keys[3]], req_status.req_context)


def test_boundary_handoffs_only_accept_block_boundaries():
    # Partial pages and their state copies are never offered whole-block
    # hand-offs; an unaligned boundary is dropped by the connector as well.
    scheduler = _scheduler()
    request = scheduler.request
    request.num_computed_tokens = 0
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={"req": 96},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        boundary_state_offloads={"req": [(1, 99, 40), (1, 98, 48)]},
    )
    jobs = scheduler._build_boundary_state_store_jobs(output)
    assert len(jobs) == 1
    [job] = jobs.values()
    src, _ = job.transfer_spec
    assert src.block_ids.tolist() == [98]


def _step_output(**overrides):
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        boundary_state_offloads=None,
    )
    for name, value in overrides.items():
        setattr(output, name, value)
    return output


def test_copy_destinations_are_fenced_against_pending_stores():
    # Codex repro: page 89 is still the source of store 123 of a finished
    # request; a snapshot copy into it must wait for that store.
    scheduler = _scheduler()
    scheduler._block_id_to_pending_jobs = {89: {123}, 50: {7}}
    meta = scheduler.build_connector_meta(_step_output(kv_block_copies=[(8, 89)]))
    assert meta.jobs_to_flush == {123}
    # Copy sources are only read and need no fence; outputs without the field
    # (older producers, test doubles) are unaffected.
    meta = scheduler.build_connector_meta(_step_output(kv_block_copies=[(50, 9)]))
    assert meta.jobs_to_flush == set()
    assert scheduler.build_connector_meta(_step_output()).jobs_to_flush == set()
