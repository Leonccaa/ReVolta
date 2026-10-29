# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for retention under interleaved long aligned prefills."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

# These tests allocate only block metadata; no device/distributed state exists.
pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _request(name, length, hash_size=8, request_id=None):
    return SimpleNamespace(
        request_id=request_id or name,
        num_tokens=length,
        shared_prefix_boundary=0,
        skip_reading_prefix_cache=False,
        block_hashes=[
            hashlib.sha256(f"{name}:{i}".encode()).digest()
            for i in range(length // hash_size)
        ],
    )


def _mamba(pool, block_size, group):
    return MambaManager(
        MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=group,
    )


def _prefill(managers, request, block_size, alignment=None, interval=None, eagle=False):
    boundaries = KVCacheCoordinator.get_replay_boundaries(
        SimpleNamespace(eagle_group_ids={0} if eagle else set()),
        request,
        alignment or block_size,
    )
    for start in range(0, request.num_tokens, block_size):
        end = min(start + block_size, request.num_tokens)
        for manager in managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start)
            manager.allocate_new_blocks(request.request_id, end, end)
            kwargs = (
                {"retention_interval": interval, "replay_boundaries": boundaries}
                if isinstance(manager, MambaManager)
                else {}
            )
            manager.cache_blocks(
                request, end, alignment_tokens=alignment or block_size, **kwargs
            )
    for manager in managers:
        manager.free(request.request_id)


def _hit(manager, request, pool, block_size, eagle):
    hashes = BlockHashListWithBlockSize(request.block_hashes, 8, block_size)
    return (
        len(
            manager.find_longest_cache_hit(
                hashes,
                request.num_tokens - 1,
                [manager.kv_cache_group_id],
                pool,
                manager.kv_cache_spec,
                eagle,
                block_size,
            )[0]
        )
        * block_size
    )


@pytest.mark.parametrize(
    "length", [799, 800, 801, 15999, 16000, 16001, 16799, 16800, 16801]
)
@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("interval", [0, 16000])
def test_sparse_retains_both_prompt_replay_boundaries(length, eagle, interval):
    pool = BlockPool(80, True, 8)
    manager = _mamba(pool, 800, 1)
    request = _request("boundary", length)
    _prefill([manager], request, 800, interval=interval, eagle=eagle)
    assert (
        _hit(manager, request, pool, 800, eagle)
        == max(0, (length - 1) // 800 - int(eagle)) * 800
    )
    assert pool.get_num_free_blocks() == 79


@pytest.mark.parametrize(
    "interval,expected", [(None, 0), (0, 169 * 16), (320, 169 * 16)]
)
def test_long_b_does_not_flush_a_when_sparse_enabled(interval, expected):
    pool = BlockPool(650, True, 8)
    full = FullAttentionManager(
        FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )
    managers = [full] + [_mamba(pool, 16, group) for group in range(1, 5)]
    a, b = _request("a", 170 * 16 + 3), _request("b", 175 * 16 + 7)
    _prefill(managers, a, 16, interval=interval, eagle=True)
    assert min(_hit(m, a, pool, 16, True) for m in managers) == 169 * 16
    _prefill(managers, b, 16, interval=interval, eagle=True)
    assert min(_hit(m, a, pool, 16, True) for m in managers) == expected
    assert pool.get_num_free_blocks() == 649


@pytest.mark.parametrize("alignment", [800, 1600])
def test_admission_falls_back_for_different_alignment(alignment):
    def retained(interval):
        pool = BlockPool(80, True, 8)
        manager = _mamba(pool, 800, 1)
        request = _request("fallback", 16001)
        _prefill([manager], request, 800, alignment, interval=interval)
        hashes = BlockHashListWithBlockSize(request.block_hashes, 8, 800)
        return [i for i, h in enumerate(hashes) if pool.get_cached_block(h, [1])]

    dense, sparse = retained(None), retained(16000)
    if alignment == 800:
        assert len(sparse) < len(dense)
    else:
        assert sparse == dense


@pytest.mark.parametrize("enable_caching", [False, True])
def test_scratch_reuse_preserves_cached_and_shared_blocks(enable_caching):
    pool = BlockPool(7, enable_caching, 8)
    cached, first, second, shared = pool.get_new_blocks(4)
    untouched = list(pool.free_block_queue.get_all_free_blocks())
    request = _request("cached", 8)
    if enable_caching:
        pool.cache_full_blocks(request, [cached], 0, 1, 8, 0)
    pool.touch([shared])
    pool.free_blocks([cached])
    pool.free_blocks([first, second, shared])
    assert shared.ref_cnt == 1
    reused = pool.get_new_blocks(2)
    assert reused == ([first, second] if enable_caching else untouched)
    if enable_caching:
        assert pool.get_cached_block(request.block_hashes[0], [0]) == [cached]
    pool.free_blocks(reused + [shared])
    assert pool.get_num_free_blocks() == 6


def _cache_manager(block_size=32, interval=0, eagle=True, capacity=1000):
    full = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        capacity,
        [],
        [KVCacheGroupSpec(["full"], full), KVCacheGroupSpec(["mamba"], mamba)],
    )
    return KVCacheManager(
        config, 100000, 8, use_eagle=eagle, prefix_cache_retention_interval=interval
    )


def _coordinated_prefill(cm, request, block_size=32):
    for start in range(0, request.num_tokens, block_size):
        end = min(start + block_size, request.num_tokens)
        for manager in cm.coordinator.single_type_managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start)
            manager.allocate_new_blocks(request.request_id, end, end)
        cm.coordinator.cache_blocks(request, end)
    cm.coordinator.free(request.request_id)


@pytest.mark.parametrize("interval", [None, 0, 32, 16000])
def test_upstream_config_semantics(interval):
    assert CacheConfig().prefix_cache_retention_interval == 0
    assert (
        CacheConfig(
            prefix_cache_retention_interval=interval
        ).prefix_cache_retention_interval
        == interval
    )
    manager = _cache_manager(interval=interval)
    assert manager.coordinator.retention_interval == interval
    assert (
        CacheConfig(prefix_cache_retention_interval=interval).compute_hash()
        == CacheConfig().compute_hash()
    )


@pytest.mark.parametrize("interval", [-1, 1000])
def test_invalid_interval_fails_after_geometry_resolution(interval):
    with pytest.raises(ValueError, match="resolved alignment"):
        _cache_manager(block_size=800, interval=interval)


@pytest.mark.parametrize("length", [127, 128, 129, 160, 161])
@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("interval", [0, 320])
def test_hybrid_identical_resend_and_longer_sibling(length, eagle, interval):
    cm = _cache_manager(interval=interval, eagle=eagle)
    a = _request("same", length)
    _coordinated_prefill(cm, a)
    _, resend = cm.get_computed_blocks(a)
    sibling = _request("same", length + 64)
    _, extension = cm.get_computed_blocks(sibling)
    assert resend == max(0, (length - 1) // 32 - int(eagle)) * 32
    assert extension == max(0, length // 32 - int(eagle)) * 32
    assert cm.block_pool.get_num_free_blocks() == 999


@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("retain_junction", [False, True])
def test_detected_junction_survives_semantic_only_retention(eagle, retain_junction):
    cm = _cache_manager(interval=0, eagle=eagle)
    a, b, c = [
        _request(name, length * 32 + 3)
        for name, length in [("a", 80), ("b", 60), ("c", 70)]
    ]
    shared_hashes = 30 * 32 // 8
    b.block_hashes[:shared_hashes] = a.block_hashes[:shared_hashes]
    c.block_hashes[:shared_hashes] = a.block_hashes[:shared_hashes]
    _coordinated_prefill(cm, a)
    _, first_hit = cm.get_computed_blocks(b)
    assert first_hit == 0
    assert b.shared_prefix_boundary == (30 - int(eagle)) * 32
    if not retain_junction:
        b.shared_prefix_boundary = 0
    _coordinated_prefill(cm, b)
    _, later_hit = cm.get_computed_blocks(c)
    assert later_hit == ((30 - int(eagle)) * 32 if retain_junction else 0)
    assert cm.block_pool.get_num_free_blocks() == 999


@pytest.mark.parametrize("block_size", [16, 512, 800, 1024])
@pytest.mark.parametrize("interval_blocks", [1, 5, 10, 20, 40])
def test_token_interval_uses_resolved_block_size(block_size, interval_blocks):
    pool = BlockPool(100, True, 8)
    manager = _mamba(pool, block_size, 1)
    request = _request("periodic", 45 * block_size + 3)
    _prefill([manager], request, block_size, interval=interval_blocks * block_size)
    hashes = BlockHashListWithBlockSize(request.block_hashes, 8, block_size)
    retained = [i + 1 for i, h in enumerate(hashes) if pool.get_cached_block(h, [1])]
    assert retained == sorted(set(range(interval_blocks, 46, interval_blocks)) | {45})
    assert pool.get_num_free_blocks() == 99


@pytest.mark.parametrize("eagle_groups", [{0}, {1}, {2}, {0, 2}])
@pytest.mark.parametrize("length", [127, 128, 129])
def test_mixed_main_and_draft_groups_retain_joint_restore_point(eagle_groups, length):
    cm = _cache_manager()
    config = cm.coordinator.kv_cache_config
    config.kv_cache_groups.append(
        KVCacheGroupSpec(["draft_mamba"], config.kv_cache_groups[1].kv_cache_spec)
    )
    for index, group in enumerate(config.kv_cache_groups):
        group.is_eagle_group = index in eagle_groups
    cm = KVCacheManager(
        config, 100000, 8, use_eagle=True, prefix_cache_retention_interval=0
    )
    request = _request("mixed", length)
    _coordinated_prefill(cm, request)
    _, hit = cm.get_computed_blocks(request)
    assert hit == max(0, (length - 1) // 32 - 1) * 32
    _, extension = cm.get_computed_blocks(_request("mixed", length + 64))
    assert extension == max(0, length // 32 - 1) * 32


@pytest.mark.parametrize("block_size", [32, 800])
@pytest.mark.parametrize("interval_blocks", [0, 5])
@pytest.mark.parametrize("decode_extra", [0, 1, 2])
def test_non_eagle_decode_preserves_new_completed_boundary(
    block_size, interval_blocks, decode_extra
):
    cm = _cache_manager(
        block_size=block_size, interval=interval_blocks * block_size, eagle=False
    )
    request = _request("growing", 2 * block_size + 1)

    def step(start, end):
        for manager in cm.coordinator.single_type_managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start)
            manager.allocate_new_blocks(request.request_id, end, end)
        cm.coordinator.cache_blocks(request, end)

    for start in range(0, request.num_tokens, block_size):
        step(start, min(start + block_size, request.num_tokens))
    final_length = 4 * block_size + decode_extra
    for length in range(request.num_tokens + 1, final_length + 1):
        request.num_tokens = length
        request.block_hashes = _request("growing", length).block_hashes
        step(length - 1, length)
    cm.coordinator.free(request.request_id)
    _, hit = cm.get_computed_blocks(_request("growing", final_length + block_size))
    assert hit == 4 * block_size
    assert cm.block_pool.get_num_free_blocks() == 999


@pytest.mark.parametrize("eagle", [False, True])
def test_sparse_mamba_publishes_only_retained_state_tokens(eagle):
    config = _cache_manager().coordinator.kv_cache_config
    cm = KVCacheManager(
        config,
        100000,
        8,
        use_eagle=eagle,
        prefix_cache_retention_interval=0,
        enable_kv_cache_events=True,
    )
    request = _request("events", 97)
    request.all_token_ids = list(range(97))
    request.lora_request = None
    request.mm_features = []
    request.cache_salt = None
    request.prompt_embeds = None
    _coordinated_prefill(cm, request)
    events = [
        e for e in cm.block_pool.take_events() if getattr(e, "group_idx", None) == 1
    ]
    boundary = 64 if eagle else 96
    assert len(events) == 1
    assert len(events[0].block_hashes) == 1
    assert events[0].token_ids == list(range(boundary - 32, boundary))


@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("length", [8160, 8192, 32768])
@pytest.mark.parametrize("interval_blocks", [0, 5])
def test_bulk_prefill_materializes_every_admitted_state(eagle, length, interval_blocks):
    block_size = 816
    interval = interval_blocks * block_size
    cm = _cache_manager(block_size=block_size, interval=interval, eagle=eagle)
    request = _request("bulk", length)
    request.num_computed_tokens = 0
    request.num_prompt_tokens = length
    request.shared_prefix_boundary = 3 * block_size
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=8),
        mamba_state_block_size=block_size,
        mamba_state_retention_interval=interval,
        kv_cache_manager=cm,
        use_eagle=eagle,
    )
    materialized = {}
    ends = []
    while request.num_computed_tokens < length:
        start = request.num_computed_tokens
        chunk = Scheduler._mamba_block_aligned_split(
            scheduler, request, min(8192, length - start)
        )
        assert chunk > 0
        end = start + chunk
        for manager in cm.coordinator.single_type_managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start)
            manager.allocate_new_blocks(request.request_id, end, end)
            if isinstance(manager, MambaManager):
                # The GPU only writes the final state of this chunk. Verify
                # no admitted interior block is advertised as that state.
                slot = (end - 1) // block_size
                block = manager.req_to_blocks[request.request_id][slot]
                assert not block.is_null
                materialized[block.block_id] = end
        cm.coordinator.cache_blocks(request, end)
        request.num_computed_tokens = end
        ends.append(end)
    manager = cm.coordinator.single_type_managers[1]
    hashes = BlockHashListWithBlockSize(request.block_hashes, 8, block_size)
    for i, block_hash in enumerate(hashes):
        if cached := cm.block_pool.get_cached_block(block_hash, [1]):
            assert materialized[cached[0].block_id] == (i + 1) * block_size
    cm.coordinator.free(request.request_id)
    _, hit = cm.get_computed_blocks(_request("bulk", length))
    assert hit == max(0, (length - 1) // block_size - int(eagle)) * block_size
    assert request.shared_prefix_boundary in ends
    if interval == 0:
        assert len(ends) <= length // 8000 + 4
    assert cm.block_pool.get_num_free_blocks() == 999


# ---------------------------------------------------------------------------
# DFlash2: a drafter that owns a *sub-page* cache must not rewind the state.
#
# Measured on 0.1.4: the FIRST reuse of a 32,768 / 99,000 / 131,072-token
# prompt hit 29,952 / 96,512 / 128,128 tokens -- exactly one 1,664-token state
# block below the deepest cacheable boundary 31,616 / 98,176 / 129,792 that
# every later reuse reaches. ``get_replay_boundaries`` rewound whenever *any*
# group was EAGLE-flagged, but DFlash2 flags only its own drafter group
# (kv_cache_utils.py:2446-2462) and that drop pops one sub-page block and
# re-aligns inside the page (single_type_kv_cache_manager.py:947-956), so no
# checkpoint page was ever owed. MTP4 rewinds for real (its EAGLE drop is
# page-granular) and must keep the rewound boundary.
# ---------------------------------------------------------------------------

STATE_BLOCK = 1664
"""Recurrent-state grid of the DFlash2 profile (docs/dflash2.md:88-90)."""

DRAFT_BLOCK = 128
"""The drafter's own block: a divisor of the state grid, hence sub-page."""

DRAFTER_GROUP = 2
"""Index of the only EAGLE-flagged group in the measured DFlash2 layout."""

# prompt length -> deepest cacheable boundary floor((n - 1) / 1664) * 1664.
FIRST_REUSE_DEFICIT = {32_768: 31_616, 99_000: 98_176, 131_072: 129_792}


def rewound_replay_boundaries(num_tokens: int, alignment: int) -> tuple[int, ...]:
    """The rule before the fix: rewind whenever *any* group is EAGLE-flagged."""
    resend = (num_tokens - 1) // alignment * alignment
    extension = num_tokens // alignment * alignment
    return tuple(sorted({max(resend - alignment, 0), max(extension - alignment, 0)}))


def state_lookbehind(coordinator) -> bool:
    """The look-behind a coordinator applies, also before this attribute existed."""
    return getattr(
        coordinator, "mamba_state_lookbehind", bool(coordinator.eagle_group_ids)
    )


def dflash2_groups(
    drafter_block: int = DRAFT_BLOCK,
    *,
    flag_drafter: bool = True,
    state_block: int = STATE_BLOCK,
) -> list[KVCacheGroupSpec]:
    """Target KV + align-mode state on one grid, drafter window on a sub-grid."""
    return [
        KVCacheGroupSpec(
            ["target"],
            FullAttentionSpec(
                block_size=state_block,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
            ),
        ),
        KVCacheGroupSpec(
            ["state"],
            MambaSpec(
                block_size=state_block,
                shapes=((1,),),
                dtypes=(torch.float16,),
                mamba_cache_mode="align",
            ),
        ),
        KVCacheGroupSpec(
            ["drafter"],
            SlidingWindowSpec(
                block_size=drafter_block,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
                sliding_window=3 * drafter_block,
            ),
            is_eagle_group=flag_drafter,
        ),
    ]


def dflash2_manager(
    *,
    length: int,
    drafter_block: int = DRAFT_BLOCK,
    flag_drafter: bool = True,
    use_eagle: bool = False,
    interval: int = 0,
    num_blocks: int | None = None,
) -> KVCacheManager:
    """KVCacheManager as the launch profile builds it: sparse retention, grid B."""
    groups = dflash2_groups(drafter_block, flag_drafter=flag_drafter)
    if num_blocks is None:
        # Roomy: the checkpoint set under test must never be evicted.
        num_blocks = 2 * (length // DRAFT_BLOCK) + 4 * (length // STATE_BLOCK) + 256
    return KVCacheManager(
        KVCacheConfig(num_blocks, [], groups),
        max_model_len=length + 1,
        hash_block_size=DRAFT_BLOCK,
        max_in_flight_tokens=8192,
        use_eagle=use_eagle,
        prefix_cache_retention_interval=interval,
    )


def _prompt(name: str, length: int) -> SimpleNamespace:
    request = _request(name, length, DRAFT_BLOCK)
    request.num_prompt_tokens = length
    request.num_computed_tokens = 0
    return request


def cold_prefill(cm: KVCacheManager, request, *, use_eagle: bool, budget: int = 8192):
    """Drive a cold prefill with the scheduler's own stop rule.

    Returns the chunk ends plus, for every recurrent-state slot, the chunk ends
    that could have written it: the align kernel stores the state of a chunk in
    the slot of its last whole state block only when the chunk ends there
    (worker/gpu/mamba_align.py:243-252).
    """
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=DRAFT_BLOCK),
        mamba_state_block_size=STATE_BLOCK,
        mamba_state_retention_interval=cm.coordinator.retention_interval,
        use_eagle=use_eagle,
        kv_cache_manager=cm,
    )
    ends: list[int] = []
    materialized: dict[int, set[int]] = {}
    while request.num_computed_tokens < request.num_tokens:
        start = request.num_computed_tokens
        chunk = Scheduler._mamba_block_aligned_split(
            scheduler, request, min(budget, request.num_tokens - start)
        )
        assert chunk > 0
        end = start + chunk
        for manager in cm.coordinator.single_type_managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(
                request.request_id, start, request.num_prompt_tokens
            )
            # The EAGLE-flagged drafter peeks one of its own blocks past the
            # boundary so that block is eligible to be cached, exactly as
            # HybridKVCacheCoordinator.cache_blocks does for it.
            manager.allocate_new_blocks(request.request_id, end + DRAFT_BLOCK, end)
            if isinstance(manager, MambaManager):
                slot = (end - 1) // STATE_BLOCK
                block = manager.req_to_blocks[request.request_id][slot]
                assert not block.is_null
                materialized.setdefault(block.block_id, set()).add(end)
        cm.coordinator.cache_blocks(request, end)
        request.num_computed_tokens = end
        ends.append(end)
    return ends, materialized


def published_state_blocks(cm: KVCacheManager, request) -> list[int]:
    """Indices of the state-group blocks that reached the prefix-cache map."""
    hashes = BlockHashListWithBlockSize(request.block_hashes, DRAFT_BLOCK, STATE_BLOCK)
    return [
        i
        for i, block_hash in enumerate(hashes)
        if cm.block_pool.get_cached_block(block_hash, [1])
    ]


@pytest.mark.parametrize("length,deepest", sorted(FIRST_REUSE_DEFICIT.items()))
def test_sub_page_drafter_publishes_the_deepest_boundary(length, deepest):
    cm = dflash2_manager(length=length, num_blocks=8)
    coordinator = cm.coordinator
    # The measured geometry: one grid, one sub-page drafter group, flagged alone.
    assert coordinator.lcm_block_size == STATE_BLOCK
    assert coordinator.eagle_group_ids == {DRAFTER_GROUP}

    boundaries = coordinator.get_replay_boundaries(
        _request("doc", length, DRAFT_BLOCK), STATE_BLOCK
    )
    # The pre-fix rule published only the rewound boundary -- the reported bug.
    rewound = rewound_replay_boundaries(length, STATE_BLOCK)
    assert rewound == (deepest - STATE_BLOCK,)
    assert deepest not in rewound
    # Both checkpoints are retained: the unrewound one a resend lands on, plus
    # the rewound one so the cold chunk stops do not move.
    assert boundaries == tuple(sorted({deepest - STATE_BLOCK, deepest}))
    assert state_lookbehind(coordinator) is False

    mask = MambaManager.reachable_block_mask(
        0,
        deepest // STATE_BLOCK + 1,
        STATE_BLOCK,
        coordinator.single_type_managers[1].kv_cache_spec,
        0,
        boundaries,
    )
    assert [i for i, keep in enumerate(mask) if keep] == [
        deepest // STATE_BLOCK - 2,
        deepest // STATE_BLOCK - 1,
    ]

    # The drafter keeps its window at the boundary that is actually admitted.
    drafter = coordinator.single_type_managers[DRAFTER_GROUP]
    assert drafter.mamba_replay_lookbehind is False
    need = SlidingWindowManager._contiguous_blocks_for_hit(
        drafter.kv_cache_spec.sliding_window, DRAFT_BLOCK, True
    )
    end = deepest // DRAFT_BLOCK + 1
    assert drafter._cached_window_at_last_boundary(length) == (end - need, end)


@pytest.mark.parametrize("variant", ["engine_fallback", "page_granular_flag"])
@pytest.mark.parametrize("length,deepest", sorted(FIRST_REUSE_DEFICIT.items()))
def test_page_granular_eagle_drop_still_rewinds(variant, length, deepest):
    # MTP4 takes the drop with a page-granular group, either because nothing is
    # labelled and the coordinator flags every group, or because the flagged
    # group's block *is* the checkpoint page.
    cm = dflash2_manager(
        length=length,
        num_blocks=8,
        drafter_block=STATE_BLOCK if variant == "page_granular_flag" else DRAFT_BLOCK,
        flag_drafter=variant == "page_granular_flag",
        use_eagle=variant == "engine_fallback",
    )
    coordinator = cm.coordinator
    assert state_lookbehind(coordinator) is True
    assert all(m.mamba_replay_lookbehind for m in coordinator.single_type_managers)
    assert (
        coordinator.get_replay_boundaries(
            _request("doc", length, DRAFT_BLOCK), STATE_BLOCK
        )
        == rewound_replay_boundaries(length, STATE_BLOCK)
        == (deepest - STATE_BLOCK,)
    )


def test_retention_rules_over_a_sweep_of_prompt_lengths():
    sub_page = dflash2_manager(length=12 * STATE_BLOCK, num_blocks=8).coordinator
    page = dflash2_manager(
        length=12 * STATE_BLOCK,
        drafter_block=STATE_BLOCK,
        use_eagle=True,
        num_blocks=8,
    ).coordinator
    lengths = [
        1,
        2,
        STATE_BLOCK - 1,
        STATE_BLOCK,
        STATE_BLOCK + 1,
        2 * STATE_BLOCK,
        5 * STATE_BLOCK + 3,
        11 * STATE_BLOCK + 7,
    ]
    for length in lengths:
        request = _request("doc", length, DRAFT_BLOCK)
        old = rewound_replay_boundaries(length, STATE_BLOCK)
        # A page-granular drop keeps today's answer to the token.
        assert page.get_replay_boundaries(request, STATE_BLOCK) == old
        boundaries = sub_page.get_replay_boundaries(request, STATE_BLOCK)
        # The fix only ever adds the unrewound boundary, so the scheduler's
        # *earliest* stop -- and with it the cold chunk geometry -- cannot move.
        assert set(boundaries) >= set(old)
        assert max(boundaries) == length // STATE_BLOCK * STATE_BLOCK


def test_sub_page_drafter_first_reuse_hits_the_full_prefix():
    length, deepest = 32_768, FIRST_REUSE_DEFICIT[32_768]
    cm = dflash2_manager(length=length)
    request = _prompt("doc", length)
    ends, materialized = cold_prefill(cm, request, use_eagle=False)
    # DFlash2 does not rewind the scheduler's `last_cache_position`, so the
    # cold run already ends a chunk at the deepest boundary.
    assert ends == [6656, 13312, 19968, 26624, 29952, 31616, 32768]

    published = published_state_blocks(cm, request)
    assert published == [deepest // STATE_BLOCK - 2, deepest // STATE_BLOCK - 1]
    for index in published:
        block = cm.block_pool.get_cached_block(
            BlockHashListWithBlockSize(request.block_hashes, DRAFT_BLOCK, STATE_BLOCK)[
                index
            ],
            [1],
        )[0]
        # Only a chunk that ended exactly on the boundary writes that state.
        assert (index + 1) * STATE_BLOCK in materialized[block.block_id]

    cm.coordinator.free(request.request_id)
    _, hit = cm.get_computed_blocks(_prompt("doc", length))
    # Before the fix this was `deepest - STATE_BLOCK` and the junction repair
    # cost one extra 1,664-token re-prefill on this very request.
    assert hit == deepest
    # Every cacheable group now reaches the same boundary: no junction at all.
    assert cm.coordinator.shared_prefix_boundary == 0
    # The pool is only metadata here; nothing leaked.
    assert cm.block_pool.get_num_free_blocks() == cm.block_pool.num_gpu_blocks - 1


def test_engine_level_rewind_first_reuse_stays_rewound():
    # An engine that rewinds `last_cache_position` (MTP/EAGLE semantics) never
    # produces the deepest state, even when only a sub-page group is flagged:
    # publishing it would add a chunk stop and change the cold prefill shape.
    length, deepest = 32_768, FIRST_REUSE_DEFICIT[32_768]
    cm = dflash2_manager(length=length, use_eagle=True)
    request = _prompt("doc", length)
    ends, materialized = cold_prefill(cm, request, use_eagle=True)
    # The last chunk crosses the deepest boundary, so no state exists there.
    assert ends == [6656, 13312, 19968, 26624, 29952, 32768]
    assert deepest not in ends
    published = published_state_blocks(cm, request)
    assert published == [deepest // STATE_BLOCK - 2]
    for index in published:
        block = cm.block_pool.get_cached_block(
            BlockHashListWithBlockSize(request.block_hashes, DRAFT_BLOCK, STATE_BLOCK)[
                index
            ],
            [1],
        )[0]
        assert (index + 1) * STATE_BLOCK in materialized[block.block_id]
    cm.coordinator.free(request.request_id)
    reuse = _prompt("doc", length)
    _, hit = cm.get_computed_blocks(reuse)
    assert hit == deepest - STATE_BLOCK
    # Denser groups still reach the deepest boundary, so the junction that
    # repairs the gap is recorded -- the behaviour MTP4 keeps today.
    assert cm.coordinator.shared_prefix_boundary == deepest
