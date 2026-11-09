# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate state and history block spaces of the long-context mode.

With ``kernel_config.long_context_kv`` enabled the KV cache is two block
spaces: a GPU state segment (GDN, PLE and drafter states) and a GPU history
segment (main K/V while it stays on the GPU, compressed index, compressor
ring). Each has its own block pool and its own tensors; block IDs repeat
across the two. These tests use the deployed CSA/GDN/PLE geometry (TP4/DCP2,
1,664-token blocks) and check that the tensors match the capacity plan, that
every group's block IDs stay inside its own space, and that a full segment
makes requests wait or be preempted, never fail.
"""

from collections import Counter

import pytest
import torch

from vllm.config.kernel import LongContextKVConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.kv_cache_view import packed_kv_members
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.long_context_kv import (
    HISTORY_SEGMENT,
    HOST,
    STATE_SEGMENT,
    GiB,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .long_context_kv_utils import (
    INDEX_BLOCK_BYTES,
    MAIN_KV_BLOCK_BYTES,
    POOL_BLOCK_BYTES,
    SPAN,
    production_config,
    production_specs,
)
from .test_scheduler_kv_capacity import make_config, step

pytestmark = pytest.mark.cpu_test

WORKERS = 4
GPU_BUDGET = int(6.05 * GiB)
HISTORY_BLOCK_BYTES = INDEX_BLOCK_BYTES + MAIN_KV_BLOCK_BYTES  # main K/V on GPU
STATE_BLOCK_BYTES = MAIN_KV_BLOCK_BYTES  # six owner-shaped 851,968-byte pages
assert HISTORY_BLOCK_BYTES == POOL_BLOCK_BYTES

_hash_ready = False


def _init_hash() -> None:
    global _hash_ready
    if not _hash_ready:
        init_none_hash(sha256)
        _hash_ready = True


def settings(**kw) -> LongContextKVConfig:
    if "host_pool_tokens" not in kw:
        kw.setdefault("host_pool_gib", 200.0)
    kw.setdefault("hot_pool_gib", 0.6)
    return LongContextKVConfig(enabled=True, **kw)


# -- worker configs -------------------------------------------------------------


def worker_configs(
    lc: LongContextKVConfig | None,
    *,
    max_model_len: int = 512 * 1024,
    memory: int = GPU_BUDGET,
    speculative_blocks: int = 7,
    max_num_seqs: int = 8,
) -> list[KVCacheConfig]:
    config = production_config(
        max_model_len=max_model_len, long_context=lc, max_num_seqs=max_num_seqs
    )
    config.model_config.original_max_model_len = max_model_len
    specs = production_specs(speculative_blocks=speculative_blocks)
    return get_kv_cache_configs(config, [specs] * WORKERS, [memory] * WORKERS)


def _layer_specs(config: KVCacheConfig) -> dict[str, tuple[int, KVCacheSpec]]:
    out = {}
    for group_id, group in enumerate(config.kv_cache_groups):
        spec = group.kv_cache_spec
        for name in group.layer_names:
            layer_spec = (
                spec.kv_cache_specs[name]
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else spec
            )
            out[name] = (group_id, layer_spec)
    return out


def test_split_tensors_match_the_plan():
    (cfg, *rest) = worker_configs(settings(prefill_staging_gib=0.25))
    plan = cfg.long_context_plan
    segments = cfg.block_segments
    assert plan is not None and segments is not None
    assert plan.main_kv_on_gpu
    num_state, num_history = plan.segment_num_blocks()
    assert segments.num_blocks == (num_state, num_history)
    assert cfg.num_blocks == num_history
    # Replay off, 8 requests: 6 GDN x 9 + PLE 2 + drafter 15 state blocks each.
    assert plan.state_blocks == 8 * 71
    assert plan.geometry.state_bytes_per_block == STATE_BLOCK_BYTES
    assert plan.history_bytes_per_block == HISTORY_BLOCK_BYTES

    # Group placement: compressed-sparse (main K/V + index) and the ring in
    # the history segment; six GDN groups, PLE and the drafter in the state.
    kinds = []
    for group, segment in zip(cfg.kv_cache_groups, segments.group_segments):
        spec = group.kv_cache_spec
        member = (
            next(iter(spec.kv_cache_specs.values()))
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else spec
        )
        kinds.append((type(member), segment))
    assert kinds[0] == (FullAttentionSpec, HISTORY_SEGMENT)
    assert kinds[1] == (CircularBufferSpec, HISTORY_SEGMENT)
    assert Counter(kinds[2:]) == {
        (MambaSpec, STATE_SEGMENT): 7,
        (SlidingWindowSpec, STATE_SEGMENT): 1,
    }

    # Bytes: the state segment is exactly (S + 1) state blocks, the history
    # segment (H + 1) index + main K/V blocks, and both fit the budget.
    state_bytes = sum(t.size for t in cfg.kv_cache_tensors if t.num_blocks == num_state)
    history_bytes = sum(
        t.size for t in cfg.kv_cache_tensors if t.num_blocks == num_history
    )
    assert num_state != num_history
    assert state_bytes + history_bytes == sum(t.size for t in cfg.kv_cache_tensors)
    assert state_bytes == num_state * STATE_BLOCK_BYTES == plan.gpu_state_bytes
    assert history_bytes == num_history * HISTORY_BLOCK_BYTES == plan.gpu_history_bytes
    staging = int(0.25 * GiB)
    assert state_bytes + history_bytes <= GPU_BUDGET - staging
    # The history segment took everything left (less than one block spare).
    assert GPU_BUDGET - staging - state_bytes - history_bytes < HISTORY_BLOCK_BYTES

    # Every layer is in exactly one tensor, and the worker's reshape (tensor
    # bytes // (page x packed members)) yields its own segment's block count.
    specs = _layer_specs(cfg)
    packed = packed_kv_members(cfg)
    owners = Counter(name for t in cfg.kv_cache_tensors for name in t.shared_by)
    assert set(owners) == set(specs) and set(owners.values()) == {1}
    for tensor in cfg.kv_cache_tensors:
        for name in tensor.shared_by:
            group_id, spec = specs[name]
            members = packed.get(name, (0, 1))[1]
            assert tensor.size % (spec.page_size_bytes * members) == 0
            blocks = tensor.size // (spec.page_size_bytes * members)
            assert blocks == cfg.group_num_blocks(group_id) == tensor.num_blocks
    # Main K/V stays packed two layers per physical owner page.
    assert sum(1 for t in cfg.kv_cache_tensors if t.packed_members) == 6
    # All workers agree.
    for other in rest:
        assert other.block_segments == segments
        assert [t.size for t in other.kv_cache_tensors] == [
            t.size for t in cfg.kv_cache_tensors
        ]


def test_disabled_mode_keeps_the_unified_pool():
    (off,) = worker_configs(None)[:1]
    assert off.block_segments is None and off.long_context_plan is None
    assert all(t.num_blocks is None for t in off.kv_cache_tensors)
    assert off.num_blocks == GPU_BUDGET // POOL_BLOCK_BYTES
    assert all(off.group_num_blocks(g) == off.num_blocks for g in range(10))


def test_startup_checks_only_the_segments():
    # 300 history blocks per request; one request needs 72 state blocks of
    # 5.11 MB (71 + null) and 302 history blocks of 6.39 MB (+ ring + null):
    # 2.298 GB. The unified pool would need 372 blocks of 6.39 MB (2.377 GB).
    max_len = 300 * SPAN
    memory = 2_330_000_000
    lc = settings(state_blocks=71)
    (cfg, *_) = worker_configs(lc, max_model_len=max_len, memory=memory)
    assert cfg.long_context_plan.gpu_history_blocks >= 302
    with pytest.raises(ValueError, match="estimated maximum model length"):
        worker_configs(None, max_model_len=max_len, memory=memory)
    # And the per-segment check refuses what the segments cannot hold.
    with pytest.raises(ValueError, match="GPU history segment"):
        worker_configs(
            settings(state_blocks=8 * 71), max_model_len=max_len, memory=memory
        )
    with pytest.raises(ValueError, match="state_blocks >= 71"):
        worker_configs(settings(state_blocks=70), max_model_len=max_len)


# -- manager: production groups, drafter included --------------------------------


def _request(req_id: str, num_tokens: int, token: int = 7) -> Request:
    _init_hash()
    params = SamplingParams(max_tokens=16, ignore_eos=True)
    params.update_from_generation_config({}, 0)
    return Request(
        request_id=req_id,
        prompt_token_ids=[token] * num_tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def _manager(**kw) -> KVCacheManager:
    (cfg, *_) = worker_configs(settings(**kw), max_model_len=64 * SPAN)
    sched_cfg = generate_scheduler_kv_cache_config([cfg])
    return KVCacheManager(
        sched_cfg,
        max_model_len=64 * SPAN,
        hash_block_size=16,
        max_in_flight_tokens=8192,
        enable_caching=True,
        use_eagle=True,
        dcp_world_size=2,
    )


def _check_ids_in_segments(manager: KVCacheManager) -> None:
    coordinator = manager.coordinator
    for group_id, single in enumerate(coordinator.single_type_managers):
        pool = coordinator.group_block_pool(group_id)
        assert single.block_pool is pool
        for blocks in single.req_to_blocks.values():
            for block in blocks:
                assert pool.blocks[block.block_id] is block


def _used(pool) -> int:
    return pool.num_gpu_blocks - 1 - pool.get_num_free_blocks()


def test_manager_allocates_each_group_from_its_segment():
    manager = _manager()
    coordinator = manager.coordinator
    state_pool, history_pool = coordinator.block_pools
    assert manager.block_pool is history_pool
    assert manager.admission_ledger.physical_segments
    assert state_pool.num_gpu_blocks == 8 * 71 + 1
    seg = manager.kv_cache_config.block_segments.group_segments
    held = []
    for i in range(3):
        req = _request(f"r{i}", 5 * SPAN + 3 + i, token=100 + i)
        new = manager.allocate_slots(req, req.num_tokens, num_lookahead_tokens=7)
        assert new is not None
        held.append(req)
        _check_ids_in_segments(manager)
    # State blocks come only from the state pool; history only from history.
    state_ids: Counter[int] = Counter()
    history_ids: Counter[int] = Counter()
    for req in held:
        for group_id, ids in enumerate(manager.get_block_ids(req.request_id)):
            target = state_ids if seg[group_id] == STATE_SEGMENT else history_ids
            target.update(i for i in ids if i)
    assert _used(state_pool) == len(state_ids) > 0
    assert _used(history_pool) == len(history_ids) > 0
    # Each history block ID backs both the main K/V and the index pages of the
    # compressed-sparse group; the ring has its own history block.
    assert max(state_ids) < state_pool.num_gpu_blocks
    assert max(history_ids) < history_pool.num_gpu_blocks
    # Zeroing only names history blocks (the workers zero only that segment).
    zero = manager.take_new_block_ids()
    assert zero and set(zero) <= set(history_ids)
    for req in held:
        manager.free(req)
    assert _used(state_pool) == _used(history_pool) == 0
    assert manager.admission_ledger.in_use[HOST] == 0


def test_full_state_segment_refuses_without_touching_history():
    # Room for one request's worst-case states only.
    manager = _manager(state_blocks=71)
    state_pool, history_pool = manager.coordinator.block_pools
    first = _request("a", 3 * SPAN, token=1)
    assert manager.allocate_slots(first, first.num_tokens, num_lookahead_tokens=7)
    free_history = history_pool.get_num_free_blocks()
    blocked = 0
    for i in range(6):
        req = _request(f"b{i}", 3 * SPAN, token=10 + i)
        if manager.allocate_slots(req, req.num_tokens, num_lookahead_tokens=7) is None:
            blocked += 1
            break
    assert blocked
    # The refusal came from the state pool; the history pool still had room
    # and nothing was taken from it for the refused request.
    assert history_pool.get_num_free_blocks() > 100
    assert state_pool.get_num_free_blocks() < 71
    manager.free(first)
    assert history_pool.get_num_free_blocks() > free_history - 1


# -- scheduler: queueing, preemption, sharing ----------------------------------


MAX_MODEL_LEN = 10 * SPAN


def build(
    lc: LongContextKVConfig | None,
    *,
    memory_blocks: int = 200,
    memory: int | None = None,
    max_num_seqs: int = 8,
    prefix_caching: bool = True,
    reserve_full_isl: bool = False,
) -> Scheduler:
    config = make_config(max_model_len=MAX_MODEL_LEN, speculative_tokens=0)
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.scheduler_reserve_full_isl = reserve_full_isl
    config.cache_config.enable_prefix_caching = prefix_caching
    if lc is not None:
        config.kernel_config.long_context_kv = lc
    specs = production_specs(speculative_blocks=0, drafter=False)
    if memory is None:
        memory = memory_blocks * POOL_BLOCK_BYTES
    (worker,) = get_kv_cache_configs(config, [specs], [memory])
    return Scheduler(
        vllm_config=config,
        kv_cache_config=generate_scheduler_kv_cache_config([worker]),
        structured_output_manager=StructuredOutputManager(config),
        block_size=16,
        hash_block_size=16,
    )


def make_requests(n, prompt_len, max_tokens, *, same_prompt=False, tag="r"):
    _init_hash()
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 0)
    return [
        Request(
            request_id=f"{tag}{i}",
            prompt_token_ids=[7 if same_prompt else 1000 + i] * prompt_len,
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(16, sha256),
        )
        for i in range(n)
    ]


def check_pools(scheduler: Scheduler) -> None:
    """Each pool's used count equals the distinct blocks its groups hold."""
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    held: dict[int, set[int]] = {id(pool): set() for pool in coordinator.block_pools}
    for group_id, single in enumerate(coordinator.single_type_managers):
        pool = coordinator.group_block_pool(group_id)
        for blocks in single.req_to_blocks.values():
            for block in blocks:
                assert pool.blocks[block.block_id] is block
                if not block.is_null:
                    assert block.ref_cnt > 0
                    held[id(pool)].add(block.block_id)
    for pool in coordinator.block_pools:
        assert _used(pool) == len(held[id(pool)])
    ledger = manager.admission_ledger
    host: set[int] = set()
    for blocks in coordinator.single_type_managers[0].req_to_blocks.values():
        host.update(b.block_id for b in blocks if not b.is_null)
    assert ledger.in_use[HOST] == len(host) <= ledger.capacity[HOST]


def run(scheduler, requests, *, max_steps=20000):
    for request in requests:
        scheduler.add_request(request)
    peak, preemptions = 0, 0
    pools = scheduler.kv_cache_manager.coordinator.block_pools
    min_free = [pool.num_gpu_blocks for pool in pools]
    for _ in range(max_steps):
        if not scheduler.has_unfinished_requests():
            break
        output = step(scheduler)
        preemptions += len(output.preempted_req_ids or ())
        peak = max(peak, len(scheduler.running))
        check_pools(scheduler)
        min_free = [
            min(m, pool.get_num_free_blocks()) for m, pool in zip(min_free, pools)
        ]

    assert not scheduler.has_unfinished_requests(), "workload did not finish"
    for request in requests:
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    for pool in scheduler.kv_cache_manager.coordinator.block_pools:
        assert _used(pool) == 0
    return peak, preemptions, min_free


def test_scheduler_runs_on_two_block_spaces():
    scheduler = build(settings())
    assert len(scheduler.kv_cache_manager.coordinator.block_pools) == 2
    requests = make_requests(6, 3 * SPAN + 77, 300) + make_requests(
        3, 2 * SPAN, 50, same_prompt=True, tag="s"
    )
    peak, _, _ = run(scheduler, requests)
    assert peak >= 2


@pytest.mark.parametrize("reserve_full_isl", [False, True])
def test_full_state_segment_queues(reserve_full_isl):
    # No speculation, no drafter: 6 GDN x 2 + PLE 2 = 14 state blocks/request.
    scheduler = build(settings(state_blocks=2 * 14), reserve_full_isl=reserve_full_isl)
    state_pool, history_pool = scheduler.kv_cache_manager.coordinator.block_pools
    assert state_pool.num_gpu_blocks == 2 * 14 + 1
    requests = make_requests(8, 2 * SPAN, 8)
    peak, _, (state_free, history_free) = run(scheduler, requests)
    # A running request holds fewer than its worst-case 14 state blocks, so
    # more than two run at once, but the state pool ran dry and held the rest
    # back while the history pool always had room for all eight.
    assert peak < 8
    assert state_free < 14
    assert history_free > 8 * 4


def test_full_history_segment_preempts_and_finishes():
    # A history segment of a few blocks: requests grow from 1 to 3 blocks.
    lc = settings(state_blocks=8 * 14)
    probe = build(lc, memory_blocks=200)
    plan = probe.kv_cache_manager.admission_ledger.plan
    # memory for the state segment plus 12 history blocks (+ null)
    memory = plan.gpu_state_bytes + 13 * HISTORY_BLOCK_BYTES
    scheduler = build(lc, memory=memory)
    state_pool, history_pool = scheduler.kv_cache_manager.coordinator.block_pools
    assert history_pool.num_gpu_blocks == 13
    requests = make_requests(5, SPAN - 64, 2 * SPAN)
    _, preemptions, _ = run(scheduler, requests)
    assert preemptions > 0
    assert sum(r.num_preemptions for r in requests) == preemptions


def test_host_pool_still_admits_by_ledger():
    scheduler = build(settings(host_pool_tokens=MAX_MODEL_LEN))
    ledger = scheduler.kv_cache_manager.admission_ledger
    assert ledger.capacity[HOST] == 10 and ledger.physical_segments
    requests = make_requests(4, 4 * SPAN - 100, 4)
    peak, _, _ = run(scheduler, requests)
    assert ledger.blocked[HOST] > 0
    assert peak >= 2


def test_shared_prefix_is_held_once_in_each_segment():
    scheduler = build(settings())
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    seg = coordinator.kv_cache_config.block_segments.group_segments
    hits: list[tuple[int, set[int]]] = []
    lookup = manager.get_computed_blocks

    def recording_lookup(request):
        computed, num_tokens = lookup(request)
        for group_id, blocks in enumerate(computed.blocks):
            pool = coordinator.group_block_pool(group_id)
            for block in blocks:
                assert pool.blocks[block.block_id] is block
        segments = {
            seg[g]
            for g, blocks in enumerate(computed.blocks)
            if any(not b.is_null for b in blocks)
        }
        hits.append((num_tokens, segments))
        return computed, num_tokens

    manager.get_computed_blocks = recording_lookup
    requests = make_requests(3, 4 * SPAN + 10, 64, same_prompt=True)
    scheduler.add_request(requests[0])
    for _ in range(4):
        step(scheduler)
    for request in requests[1:]:
        scheduler.add_request(request)
    shared = {pool_index: False for pool_index in range(2)}
    while scheduler.has_unfinished_requests():
        step(scheduler)
        check_pools(scheduler)
        for group_id, single in enumerate(coordinator.single_type_managers):
            pool_index = coordinator.kv_cache_config.block_segments.group_segments[
                group_id
            ]
            refs = [
                b.ref_cnt
                for blocks in single.req_to_blocks.values()
                for b in blocks
                if not b.is_null
            ]
            shared[pool_index] |= any(ref > 1 for ref in refs)
    assert shared[HISTORY_SEGMENT]
    # The later requests hit the first one's prefix in both block spaces:
    # history blocks (main K/V + index) and the GDN/PLE checkpoint states.
    assert any(
        n > 0 and segments == {STATE_SEGMENT, HISTORY_SEGMENT} for n, segments in hits
    )
    for pool in coordinator.block_pools:
        assert _used(pool) == 0
    # Reset clears both block spaces, all or nothing.
    assert manager.reset_prefix_cache()
    for pool in coordinator.block_pools:
        assert len(pool.cached_block_hash_to_block) == 0


# -- complete-prefix contract prefix regression: a second full-attention spec group is
# trimmed to the hit ----------


def test_second_full_attention_group_is_trimmed_with_block_segments():
    """Two full-attention specs (e.g. main K/V and index as separate groups)
    and a Mamba state that limits the hit. The unified coordinator trims only
    its first full-attention spec group (complete-prefix contract prefix regression);
    with block segments every
    full-attention group is trimmed to the hit, so no block past it is handed
    to the request."""
    import torch

    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    from vllm.v1.kv_cache_interface import KVBlockSegments, KVCacheGroupSpec

    bs = 16
    specs: list[KVCacheSpec] = [
        FullAttentionSpec(
            block_size=bs, num_kv_heads=1, head_size=h, dtype=torch.float32
        )
        for h in (1, 2)
    ] + [
        MambaSpec(
            block_size=bs,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        )
    ]
    request = _request("r", 10 * bs + 1, token=9)
    present = {0: set(range(10)), 1: set(range(10)), 2: {3}}
    results = {}
    for split in (False, True):
        config = KVCacheConfig(
            num_blocks=256,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)
            ],
            block_segments=(
                KVBlockSegments(
                    names=("state", "history"),
                    num_blocks=(64, 256),
                    group_segments=(HISTORY_SEGMENT, HISTORY_SEGMENT, STATE_SEGMENT),
                    primary=HISTORY_SEGMENT,
                )
                if split
                else None
            ),
        )
        manager = KVCacheManager(
            config, max_model_len=4096, hash_block_size=bs, enable_caching=True
        )
        coordinator = manager.coordinator
        assert coordinator.truncate_all_full_attention_hits is split
        for group_id, indices in present.items():
            pool = coordinator.group_block_pool(group_id)
            for index in sorted(indices):
                (block,) = pool.get_new_blocks(1)
                block.block_hash = make_block_hash_with_group_id(
                    request.block_hashes[index], group_id
                )
                pool.cached_block_hash_to_block.insert(block.block_hash, block)
        blocks, num_tokens = manager.get_computed_blocks(request)
        assert num_tokens == 4 * bs
        results[split] = [len(b) for b in blocks.blocks]
        if split:
            # Each group's hit blocks come from its own pool.
            for group_id, group_blocks in enumerate(blocks.blocks):
                pool = coordinator.group_block_pool(group_id)
                for block in group_blocks:
                    assert pool.blocks[block.block_id] is block
    assert results[False][:2] == [4, 10]  # unified: existing behaviour
    assert results[True][:2] == [4, 4]


# -- worker allocation ---------------------------------------------------------


def test_worker_allocates_separate_tensors_per_segment():
    """The worker's raw allocation follows the split config: every layer gets
    a tensor of its own segment's blocks, and no state layer shares storage
    with a history layer (their block IDs overlap)."""
    from vllm.v1.worker.gpu.attn_utils import _allocate_kv_cache

    lc = settings(state_blocks=71)
    max_len = 4 * SPAN
    plan_bytes = 72 * STATE_BLOCK_BYTES + 9 * HISTORY_BLOCK_BYTES
    (cfg, *_) = worker_configs(
        lc, max_model_len=max_len, memory=plan_bytes, max_num_seqs=1
    )
    assert cfg.block_segments.num_blocks == (72, 9)
    raw = _allocate_kv_cache(cfg, {}, torch.device("cpu"))
    specs = _layer_specs(cfg)
    packed = packed_kv_members(cfg)
    storages: dict[int, set[int]] = {STATE_SEGMENT: set(), HISTORY_SEGMENT: set()}
    for name, tensor in raw.items():
        group_id, spec = specs[name]
        members = packed.get(name, (0, 1))[1]
        assert tensor.numel() == (
            cfg.group_num_blocks(group_id) * spec.page_size_bytes * members
        )
        segment = cfg.block_segments.group_segments[group_id]
        storages[segment].add(tensor.untyped_storage().data_ptr())
    assert not storages[STATE_SEGMENT] & storages[HISTORY_SEGMENT]
    distinct = {id(t): t for t in raw.values()}.values()
    total = sum(t.untyped_storage().nbytes() for t in distinct)
    assert total == plan_bytes
    # Zeroing touches only the history (primary) space's layers.
    seg = cfg.block_segments.group_segments
    assert [cfg.in_primary_block_space(g) for g in range(len(seg))] == [
        s == HISTORY_SEGMENT for s in seg
    ]


@pytest.mark.parametrize("drafter", [True, False])
def test_cpu_offload_pages_are_the_same_on_two_block_spaces(drafter: bool):
    """The CPU offload tier sizes its pages from the GPU tensors. With block
    segments each tensor holds its own space's blocks; a page must still fit
    one block of either space, which here equals the unified pool block."""
    from types import SimpleNamespace

    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    def offload_spec(lc):
        config = production_config(max_model_len=8 * SPAN, long_context=lc)
        config.model_config.original_max_model_len = 8 * SPAN
        config.parallel_config.world_size = WORKERS
        config.cache_config.block_size = SPAN
        config.cache_config.hash_block_size = 16
        config.cache_config.enable_prefix_caching = True
        config.cache_config.prefix_cache_retention_interval = 0
        config.kv_transfer_config = SimpleNamespace(
            kv_connector="OffloadingConnector",
            kv_connector_extra_config={"cpu_bytes_to_use": 4 * GiB},
        )
        if not drafter:
            config.speculative_config = None
        specs = production_specs(speculative_blocks=7, drafter=drafter)
        (worker,) = get_kv_cache_configs(config, [specs], [GPU_BUDGET])
        return CPUOffloadingSpec(config, worker)

    unified = offload_spec(None)
    split = offload_spec(settings())
    assert split.kv_cache_config.block_segments is not None
    if drafter:
        # The drafter's 832-token blocks keep the shared-page CPU layout: a
        # page holds one block of one space, the largest being a history block.
        assert not split.partition_by_group and not unified.partition_by_group
        assert split.cpu_page_size_per_worker == unified.cpu_page_size_per_worker
        assert split.cpu_page_size_per_worker == (
            POOL_BLOCK_BYTES * split.block_size_factor
        )
    else:
        # Per-group pages: state groups keep their owner-shaped pages, the
        # compressed-sparse group main K/V + index, the ring its index page.
        assert split.partition_by_group and unified.partition_by_group
        assert split.cpu_group_page_sizes == unified.cpu_group_page_sizes
        assert split.cpu_group_page_sizes[0] == POOL_BLOCK_BYTES
        assert split.cpu_group_num_blocks == unified.cpu_group_num_blocks
