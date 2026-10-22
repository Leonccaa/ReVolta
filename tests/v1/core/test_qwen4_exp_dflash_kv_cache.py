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
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

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


def test_drafter_cache_must_be_replicated_across_dcp() -> None:
    specs = _target_specs(dcp_sharded=True) | _draft_specs((8,), dcp_sharded=True)
    with pytest.raises(ValueError, match="replicated across DCP"):
        get_kv_cache_groups(_vllm_config(dcp=2), specs)


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
