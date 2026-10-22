# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A DFlash drafter's sliding-window cache in the Qwen4Exp CSA+linear layout."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import (
    _get_csa_linear_tensor_layout,
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skip_global_cleanup

DRAFT = "model.layers.{}.self_attn.attn"


class _ModelConfig:
    max_model_len = 8192

    def get_num_kv_heads(self, parallel_config) -> int:
        del parallel_config
        return 1

    def get_total_num_hidden_layers(self) -> int:
        return 8


def _vllm_config(dflash: bool = True, dcp: int = 1):
    return SimpleNamespace(
        model_config=_ModelConfig(),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=8192,
            max_num_seqs=2,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            mamba_cache_mode="none",
        ),
        speculative_config=(
            SimpleNamespace(use_dflash=lambda: True) if dflash else None
        ),
    )


def _target_specs(block_size: int = 16, dcp_sharded: bool = False):
    specs = {}
    for layer in (3, 7):
        prefix = f"model.layers.{layer}.self_attn"
        specs[prefix] = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.float16,
            dcp_sharded=dcp_sharded,
        )
        specs[f"{prefix}.compressed"] = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
            dcp_sharded=dcp_sharded,
        )
        specs[f"{prefix}.compressor_state"] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
        )
    for layer in (0, 1, 2, 4, 5, 6):
        specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
            block_size=block_size,
            shapes=((1, 64),),
            dtypes=(torch.float16,),
        )
    return specs


def _draft_specs(layers, num_kv_heads: int = 1, head_size: int = 64, **kw):
    return {
        DRAFT.format(layer): SlidingWindowSpec(
            block_size=kw.get("block_size", 16),
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.float16,
            sliding_window=32,
            dcp_sharded=kw.get("dcp_sharded", False),
        )
        for layer in layers
    }


def _drafter_group(groups):
    drafter = [g for g in groups if type(g.kv_cache_spec) is SlidingWindowSpec]
    assert len(drafter) == 1
    return drafter[0]


def test_drafter_layers_share_main_kv_owner_slots() -> None:
    config = _vllm_config()
    baseline = _get_csa_linear_tensor_layout(
        get_kv_cache_groups(config, _target_specs())
    )
    groups = get_kv_cache_groups(config, _target_specs() | _draft_specs((8, 9)))
    layout = _get_csa_linear_tensor_layout(groups)
    assert baseline is not None and layout is not None

    group = _drafter_group(groups)
    spec = group.kv_cache_spec
    assert group.layer_names == [DRAFT.format(8), DRAFT.format(9)]
    # Only the drafter group takes the EAGLE last-block drop.
    assert group.is_eagle_group
    assert not any(g.is_eagle_group for g in groups if g is not group)
    assert spec.block_size == 16
    # One extra block in the manager's window (vllm-project/vllm#60560).
    assert spec.sliding_window == 32 + 16
    assert spec.real_page_size_bytes == 16 * 64 * 2 * 2
    assert spec.page_size_bytes == layout.owner_page_size(0) == 16_384

    # Sharing owner slots costs no extra bytes per pool block.
    assert layout.bytes_per_block == baseline.bytes_per_block
    shared, standalone = layout.auxiliary_tensor_plan()
    assert shared == [[DRAFT.format(8)], [DRAFT.format(9)]]
    assert standalone == []

    caches = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 30)
    owners = [t for t in caches.kv_cache_tensors if DRAFT.format(8) in t.shared_by]
    assert len(owners) == 1 and "model.layers.3.self_attn" in owners[0].shared_by
    assert len(caches.kv_cache_tensors) == 4


def test_drafter_layers_beyond_owner_slots_get_their_own_tensors() -> None:
    config = _vllm_config()
    baseline = _get_csa_linear_tensor_layout(
        get_kv_cache_groups(config, _target_specs())
    )
    groups = get_kv_cache_groups(config, _target_specs() | _draft_specs((8, 9, 10)))
    layout = _get_csa_linear_tensor_layout(groups)
    assert baseline is not None and layout is not None

    _, standalone = layout.auxiliary_tensor_plan()
    assert standalone == [(DRAFT.format(10), 16_384)]
    assert layout.bytes_per_block == baseline.bytes_per_block + 16_384
    caches = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 30)
    assert [t.shared_by for t in caches.kv_cache_tensors][-1] == [DRAFT.format(10)]
    assert len(caches.kv_cache_tensors) == 5


def test_drafter_page_larger_than_a_slot_keeps_its_own_page() -> None:
    config = _vllm_config()
    groups = get_kv_cache_groups(
        config, _target_specs() | _draft_specs((8,), num_kv_heads=8, head_size=256)
    )
    layout = _get_csa_linear_tensor_layout(groups)
    assert layout is not None
    spec = _drafter_group(groups).kv_cache_spec
    assert spec.page_size_padded is None
    assert spec.page_size_bytes == 16 * 8 * 256 * 2 * 2
    _, standalone = layout.auxiliary_tensor_plan()
    assert standalone == [(DRAFT.format(8), spec.page_size_bytes)]


def test_drafter_reblocks_to_a_divisor_of_the_target_span() -> None:
    # A 64-token target block: the largest 16-aligned divisor whose draft page
    # fits one 64-token main-KV slot is 64 itself for a narrow drafter, and 32
    # for one whose 64-token page would overflow the slot.
    config = _vllm_config()
    narrow = _drafter_group(
        get_kv_cache_groups(config, _target_specs(64) | _draft_specs((8,)))
    ).kv_cache_spec
    assert narrow.block_size == 64
    slot = 64 * 1 * 256 * 2 * 2
    wide = _drafter_group(
        get_kv_cache_groups(
            config, _target_specs(64) | _draft_specs((8,), head_size=384)
        )
    ).kv_cache_spec
    assert wide.block_size == 32 and wide.page_size_bytes == slot
    assert wide.sliding_window == 32 + 32


def test_drafter_cache_requires_dflash() -> None:
    with pytest.raises(ValueError, match="DFlash drafter"):
        get_kv_cache_groups(
            _vllm_config(dflash=False), _target_specs() | _draft_specs((8,))
        )


def _dcp2_target_specs():
    # Production layout: QSA main K/V sharded, recurrent and ring state replicated.
    return {
        name: (
            replace(spec, dcp_sharded=False)
            if type(spec) in (MambaSpec, CircularBufferSpec)
            else spec
        )
        for name, spec in _target_specs(dcp_sharded=True).items()
    }


def test_drafter_cache_is_replicated_across_dcp() -> None:
    # Drafter layers report AttentionSpec's default dcp_sharded=True; the group
    # is replicated anyway, so hybrid DCP accepts it and every rank keeps the
    # whole window (the drafter's attention backend has no DCP combine).
    config = _vllm_config(dcp=2)
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.hash_block_size = None
    config.kv_transfer_config = None
    groups = get_kv_cache_groups(
        config, _dcp2_target_specs() | _draft_specs((8,), dcp_sharded=True)
    )
    drafter = _drafter_group(groups).kv_cache_spec
    assert not drafter.dcp_sharded
    # The sharded 16-slot target block spans 32 global tokens; a 32-token
    # drafter page (8 KiB) still fits one 16 KiB owner slot.
    assert drafter.block_size == 32

    caches = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 30)
    assert resolve_kv_cache_block_sizes(caches, config) == (32, 32)
    scheduler_config = generate_scheduler_kv_cache_config([caches])
    scheduler_config.num_blocks = 64
    coordinator = get_kv_cache_coordinator(
        scheduler_config,
        max_model_len=8192,
        max_in_flight_tokens=128,
        use_eagle=True,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=2,
        pcp_world_size=1,
        hash_block_size=16,
    )
    manager = next(
        m
        for m in coordinator.single_type_managers
        if isinstance(m, SlidingWindowManager)
    )
    assert manager.dcp_world_size == 1
    assert manager.block_size == 32

    # The engine sizes memory per request from the group spec before it
    # builds the pool; a replicated window is sized as at DCP1.
    config.model_config.max_model_len = 8192
    config.max_in_flight_tokens = 128
    blocks = drafter.max_admission_blocks_per_request(
        max_in_flight_tokens=128, max_model_len=8192
    )
    assert drafter.max_memory_usage_bytes(config) == blocks * drafter.page_size_bytes
    with pytest.raises(AssertionError, match="DCP not support sliding window"):
        replace(drafter, dcp_sharded=True).max_memory_usage_bytes(config)


def test_block_table_lays_out_a_replicated_group_on_every_rank() -> None:
    tables = BlockTables.__new__(BlockTables)
    tables.cp_size, tables.cp_rank, tables._dcp_sharded = 2, 1, [True, False]
    assert tables.group_cp(0) == (2, 1)
    assert tables.group_cp(1) == (1, 0)


def test_drafter_under_dcp_uses_the_global_target_span() -> None:
    # A DCP-sharded 16-token target block spans 32 global tokens with DCP=2;
    # the replicated drafter keeps that span so the scheduler block is unchanged.
    config = _vllm_config(dcp=2)
    specs = _target_specs(dcp_sharded=True)
    specs = {
        name: (replace(spec, dcp_sharded=False) if type(spec) is MambaSpec else spec)
        for name, spec in specs.items()
    }
    groups = get_kv_cache_groups(config, specs | _draft_specs((8,)))
    spec = _drafter_group(groups).kv_cache_spec
    assert spec.block_size in (32, 16)
    assert 32 % spec.block_size == 0
    assert not spec.dcp_sharded


def test_scheduler_builds_a_sliding_window_manager_for_the_drafter() -> None:
    config = _vllm_config()
    groups = get_kv_cache_groups(config, _target_specs() | _draft_specs((8, 9)))
    caches = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 30)
    scheduler_config = generate_scheduler_kv_cache_config([caches])
    scheduler_config.num_blocks = 64
    coordinator = get_kv_cache_coordinator(
        scheduler_config,
        max_model_len=8192,
        max_in_flight_tokens=128,
        use_eagle=True,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=4,
    )
    managers = [
        m
        for m in coordinator.single_type_managers
        if isinstance(m, SlidingWindowManager)
    ]
    assert len(managers) == 1
    assert managers[0].sliding_window == 48
    drafter_index = coordinator.single_type_managers.index(managers[0])
    assert coordinator.eagle_group_ids == {drafter_index}


def test_sharded_flag_is_harmless_without_dcp() -> None:
    # AttentionSpec defaults to dcp_sharded=True; with DCP=1 that is a no-op.
    groups = get_kv_cache_groups(
        _vllm_config(), _target_specs() | _draft_specs((8,), dcp_sharded=True)
    )
    assert _drafter_group(groups).kv_cache_spec.block_size == 16


def test_dcp2_prefix_hit_reaches_through_the_drafter_window() -> None:
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    init_none_hash(sha256)
    # Sharded 16-slot target (32 global tokens), replicated 32-token drafter
    # window, replicated recurrent state: the window-2 DCP2 geometry in small.
    target = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    drafter = SlidingWindowSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=64 + 32,
        dcp_sharded=False,
    )
    state = MambaSpec(
        block_size=32, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="all"
    )
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["target"], target),
            KVCacheGroupSpec(["drafter"], drafter, is_eagle_group=True),
            KVCacheGroupSpec(["state"], state),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=512,
        hash_block_size=32,
        use_eagle=True,
        dcp_world_size=2,
        enable_caching=True,
    )
    owners = manager.coordinator.single_type_managers
    assert [m.block_size for m in owners] == [32, 32, 32]
    assert [m.dcp_world_size for m in owners] == [2, 1, 1]
    assert manager.coordinator.eagle_group_ids == {1}

    common = list(range(256))
    first = make_request("first", common + [91] * 5, 32, sha256)
    assert manager.allocate_slots(first, 261) is not None
    second = make_request("second", common + [92] * 5, 32, sha256)
    blocks, tokens = manager.get_computed_blocks(second)
    # Only the drafter group gives up its last block (EAGLE drop); the hit is
    # still most of the shared prefix and stays block aligned.
    assert tokens % 32 == 0
    assert 256 - 64 <= tokens < 256
    manager.free(first)


def test_offload_spec_budgets_the_drafter_group() -> None:
    from tests.v1.core.test_qwen4_exp_kv_cache import _mixed_dcp_specs
    from tests.v1.core.test_qwen4_exp_kv_cache import _vllm_config as _base_config
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    config = _base_config()
    config.parallel_config.decode_context_parallel_size = 2
    config.parallel_config.world_size = 4
    config.cache_config.block_size = 16
    config.cache_config.hash_block_size = 4
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_cache_retention_interval = 0
    config.cache_config.mamba_cache_mode = "align"
    config.speculative_config = SimpleNamespace(use_dflash=lambda: True)
    config.kv_transfer_config = SimpleNamespace(
        kv_connector_extra_config={"cpu_bytes_to_use": 16 * 1024**2}
    )
    specs = _mixed_dcp_specs()
    for name, spec in list(specs.items()):
        if isinstance(spec, MambaSpec):
            specs[name] = replace(
                spec, block_size=32, mamba_cache_mode="align", num_speculative_blocks=4
            )
    specs |= _draft_specs((8, 9), dcp_sharded=True)
    groups = get_kv_cache_groups(config, specs)
    drafter = _drafter_group(groups)
    assert not drafter.kv_cache_spec.dcp_sharded
    worker_cache = get_kv_cache_config_from_groups(
        config, groups, available_memory=1 << 20
    )
    scheduler_cache = generate_scheduler_kv_cache_config([worker_cache])
    worker = CPUOffloadingSpec(config, worker_cache)
    scheduler = CPUOffloadingSpec(config, scheduler_cache)
    drafter_index = groups.index(drafter)
    assert worker.gpu_block_size == scheduler.gpu_block_size
    assert worker.gpu_block_size[drafter_index] == drafter.kv_cache_spec.block_size
    assert worker.cpu_group_num_blocks == scheduler.cpu_group_num_blocks
    assert worker.cpu_group_page_sizes == scheduler.cpu_group_page_sizes
