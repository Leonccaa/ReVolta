# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pending stores are flushed before the worker zeroes a reused page."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
    SchedulerOffloadConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import OffloadingManager
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

# Qwen3.8-Flash-Next with DCP2 and the DFlash2 drafter: 1664-token global
# block, DCP-sharded main K/V, compressed index keys, the per-request QSA
# compressor ring, GDN and PLE recurrent state, and the drafter's window.
BLOCK = 1664
DRAFT_BLOCK = 832
HASH_BLOCK = 16
DCP = 2

init_none_hash(sha256)


def _kv_cache_config(num_blocks: int = 96) -> KVCacheConfig:
    specs = {
        "main_kv": FullAttentionSpec(
            block_size=BLOCK // DCP,
            num_kv_heads=1,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
            dcp_sharded=True,
        ),
        "compressed_key": MLAAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
            dcp_sharded=False,
        ),
        "ring": CircularBufferSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
            dcp_sharded=False,
        ),
        "gdn": MambaSpec(
            block_size=BLOCK,
            shapes=((4, 64),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        "ple": MambaSpec(
            block_size=BLOCK,
            shapes=((2, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            tp_replicated=True,
        ),
        "drafter": SlidingWindowSpec(
            block_size=DRAFT_BLOCK,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            sliding_window=2048,
            dcp_sharded=False,
        ),
    }
    groups = [
        KVCacheGroupSpec([name], spec, is_eagle_group=(name == "drafter"))
        for name, spec in specs.items()
    ]
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


def _request(request_id: str, prompt: list[int]) -> Request:
    params = SamplingParams(max_tokens=64)
    params.update_from_generation_config({}, eos_token_id=100_000)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
    )


def _scheduler() -> OffloadingConnectorScheduler:
    """Offload groups as SchedulerOffloadConfig.from_spec builds them: the
    prefix-cacheable main K/V (group 0) and the GDN state (group 3)."""
    s = object.__new__(OffloadingConnectorScheduler)
    s.config = SchedulerOffloadConfig(
        kv_group_configs=(
            GroupOffloadConfig(0, BLOCK, BLOCK, BLOCK // HASH_BLOCK, None),
            GroupOffloadConfig(
                3,
                BLOCK,
                BLOCK,
                BLOCK // HASH_BLOCK,
                1,
                requires_exact_boundary_source=True,
            ),
        ),
        block_size_factor=1,
        num_workers=1,
        offload_prompt_only=True,
        num_kv_cache_groups=6,
    )
    s._group_config_by_idx = {g.group_idx: g for g in s.config.kv_group_configs}
    s._sliding_window_groups = (3,)
    s._lookup_groups = (0, 3)
    s._mamba_align_size = BLOCK
    s._eagle_state_shift = None
    s._req_status = {}
    s._current_batch_load_jobs = {}
    s._current_batch_jobs_to_flush = set()
    s._blocks_being_loaded = set()
    s._job_counter = 0
    s._stale_job_threshold = 0
    s._jobs = {}
    s._block_id_to_pending_jobs = {}
    s.manager = MagicMock(spec=OffloadingManager)
    s.manager.on_new_request.return_value = SimpleNamespace(policy=None)
    s.manager.prepare_store.side_effect = lambda keys, ctx: generate_store_output(keys)
    return s


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


def test_zeroed_pages_are_fenced_against_pending_stores():
    scheduler = _scheduler()
    scheduler._block_id_to_pending_jobs = {89: {123}, 50: {7}}
    meta = scheduler.build_connector_meta(_step_output(new_block_ids_to_zero=[3, 89]))
    assert meta.jobs_to_flush == {123}
    meta = scheduler.build_connector_meta(_step_output(new_block_ids_to_zero=None))
    assert meta.jobs_to_flush == set()
    assert scheduler.build_connector_meta(_step_output()).jobs_to_flush == set()


def test_parked_request_ring_page_is_fenced_before_zeroing():
    """A request parked for an async CPU load claims, as its QSA compressor-
    ring page, the source page of another request's pending boundary store.
    The ring group is not offloaded and the parked request is not scheduled,
    so only the zeroing fence covers the page."""
    prompt_len = 4500
    boundary = 2 * BLOCK
    m = KVCacheManager(
        _kv_cache_config(),
        max_model_len=8192,
        hash_block_size=HASH_BLOCK,
        enable_caching=True,
        dcp_world_size=DCP,
        prefix_cache_retention_interval=0,
    )
    s = _scheduler()

    # A's boundary store is accepted in its offer step.
    a = _request("A", list(range(prompt_len)))
    s.on_new_request(a)
    assert m.allocate_slots(a, boundary)
    offers = m.take_boundary_state_offloads()
    source = next(bid for g, bid, _ in offers["A"] if g == 3)
    meta_a = s.build_connector_meta(
        _step_output(
            boundary_state_offloads=offers, num_scheduled_tokens={"A": boundary}
        )
    )
    a.num_computed_tokens = boundary
    [job] = [
        j
        for j, t in meta_a.store_jobs.items()
        if t.transfer_spec[0].block_ids.tolist() == [source]
    ]
    m.new_step_starts()
    assert m.allocate_slots(a, prompt_len - boundary)
    a.num_computed_tokens = prompt_len
    m.new_step_starts()
    a.append_output_token_ids(42)
    assert m.allocate_slots(a, 1)
    a.num_computed_tokens = prompt_len + 1
    # A's draft window moved past the page; the store is still pending.
    assert m.block_pool.blocks[source].ref_cnt == 0
    assert job in s._block_id_to_pending_jobs[source]

    # Order the free queue so that C's ring page (group 2) is the released one.
    m.new_step_starts()
    taken = m.block_pool.get_new_blocks(m.block_pool.get_num_free_blocks())
    page = next(b for b in taken if b.block_id == source)
    rest = [b for b in taken if b.block_id != source]
    m.block_pool.free_blocks(rest[6:])
    m.block_pool.free_blocks([page])
    m.block_pool.free_blocks(rest[:6])

    c = _request("C", [50000 + i for i in range(prompt_len)])
    assert (
        m.allocate_slots(
            c, 0, num_external_computed_tokens=BLOCK, delay_cache_blocks=True
        )
        is not None
    )
    ring = m.coordinator.single_type_managers[2]
    assert m.block_pool.blocks[source] in ring.req_to_blocks["C"]
    zero_ids = m.take_new_block_ids()
    assert source in zero_ids

    s.on_new_request(c)
    status = s._req_status["C"]
    status.update_offload_keys()
    status.num_locally_computed_tokens = 0
    s.update_state_after_alloc(c, m.get_blocks("C"), BLOCK)
    # C is parked for its load and not scheduled this step.
    meta = s.build_connector_meta(_step_output(new_block_ids_to_zero=zero_ids))
    assert job in meta.jobs_to_flush
