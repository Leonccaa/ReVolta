# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP drafting in the long-context host-KV layout (CPU).

The MTP draft layer stays in the target's KV cache groups (one block table,
one hash chain), keeps its caches as GPU history tensors and rides in the
host row next to the target's index pages.
"""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm.config.kernel import LongContextKVConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    get_kv_cache_groups,
    get_long_context_geometry,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from vllm.v1.kv_offload.host.region import host_region_layout

from .long_context_kv_utils import (
    SPAN,
    mtp_config,
    mtp_specs,
    production_config,
    production_specs,
)
from .test_long_context_split_pool import GPU_BUDGET, WORKERS

pytestmark = pytest.mark.cpu_test

DRAFT = "mtp.layers.48.self_attn"
TARGET_MAIN = [f"model.layers.{layer}.self_attn" for layer in range(3, 48, 4)]


def settings(**kw) -> LongContextKVConfig:
    kw.setdefault("host_pool_gib", 200.0)
    kw.setdefault("hot_pool_gib", 0.6)
    return LongContextKVConfig(enabled=True, **kw)


def build(lc, specs, *, mtp=True):
    config = (mtp_config if mtp else production_config)(
        max_model_len=512 * 1024, long_context=lc
    )
    if not mtp:
        config.speculative_config = None
    config.model_config.original_max_model_len = 512 * 1024
    if lc.main_kv_placement == "host":
        config.kv_transfer_config = SimpleNamespace(kv_connector="OffloadingConnector")
    (cfg, *_) = get_kv_cache_configs(config, [specs] * WORKERS, [GPU_BUDGET] * WORKERS)
    return config, cfg


def test_replicated_draft_stays_on_gpu_and_joins_the_host_row():
    specs = mtp_specs()
    _, cfg = build(settings(), specs)
    plan = cfg.long_context_plan
    assert plan is not None and not plan.main_kv_on_gpu
    # A replicated draft has its own page geometry: only the target's main K/V
    # moves to the host store, the draft keeps GPU history tensors.
    assert sorted(cfg.host_kv_layers) == sorted(TARGET_MAIN)
    main_group = cfg.host_kv_group_id
    assert main_group is not None
    names = cfg.kv_cache_groups[main_group].layer_names
    assert DRAFT in names and f"{DRAFT}.compressed" in names
    num_state, num_history = plan.segment_num_blocks()
    by_name = {n: t for t in cfg.kv_cache_tensors for n in t.shared_by}
    assert Counter(n for t in cfg.kv_cache_tensors for n in t.shared_by)[DRAFT] == 1
    main, index = specs[DRAFT], specs[f"{DRAFT}.compressed"]
    assert by_name[DRAFT].shared_by == [DRAFT]
    assert by_name[DRAFT].num_blocks == num_history
    assert by_name[DRAFT].size == main.page_size_bytes * num_history
    assert by_name[DRAFT].page_stride_bytes is None
    ring = by_name[f"{DRAFT}.compressor_state"]
    assert ring.shared_by == [f"{DRAFT}.compressed", f"{DRAFT}.compressor_state"]
    assert ring.size == index.page_size_bytes * num_history
    # Host row: target index pages, then the draft's main and index pages.
    layout = host_region_layout(cfg, "test")
    draft_bytes = main.page_size_bytes + index.page_size_bytes
    assert sorted(layout.index_bytes)[-1] == main.page_size_bytes
    geo = plan.geometry
    assert geo.host_bytes_per_block == layout.worker_bytes * layout.num_workers
    assert geo.logical_index_bytes_per_block == sum(layout.index_bytes)
    assert geo.index_bytes_per_block == sum(geo.index_page_strides) + draft_bytes


def test_replicated_draft_does_not_change_the_state_segment():
    lc = settings()
    specs = mtp_specs()
    target = {name: spec for name, spec in specs.items() if "mtp" not in name}
    _, base = build(lc, target, mtp=False)
    _, cfg = build(lc, specs)
    base_geo, geo = base.long_context_plan.geometry, cfg.long_context_plan.geometry
    assert geo.state_bytes_per_block == base_geo.state_bytes_per_block
    assert geo.main_kv_bytes_per_block == base_geo.main_kv_bytes_per_block
    assert geo.hot_layers == base_geo.hot_layers == len(TARGET_MAIN)
    assert geo.index_page_strides == base_geo.index_page_strides
    state = lambda c: sorted(  # noqa: E731
        (t.size, tuple(t.shared_by))
        for t in c.kv_cache_tensors
        if t.num_blocks == c.long_context_plan.segment_num_blocks()[0]
    )
    assert [shared for _, shared in state(cfg)] == [shared for _, shared in state(base)]


def test_sharded_draft_main_is_one_more_host_layer():
    specs = mtp_specs(shard_index=True, shard_draft=True)
    target = {name: spec for name, spec in specs.items() if "mtp" not in name}
    _, base = build(settings(), target, mtp=False)
    _, cfg = build(settings(draft_main_placement="host"), specs)
    plan, base_plan = cfg.long_context_plan, base.long_context_plan
    # The sharded draft main K/V has the target's geometry: it joins the host
    # store as the last layer, with no GPU tensor and no state-segment page.
    assert cfg.host_kv_layers == [*TARGET_MAIN, DRAFT]
    owners = {n for t in cfg.kv_cache_tensors for n in t.shared_by}
    assert DRAFT not in owners
    # Its index (and ring) is a padded history tensor beside the target's,
    # whose pages carry the draft layer's hot slots.
    index = next(
        t for t in cfg.kv_cache_tensors if f"{DRAFT}.compressed" in t.shared_by
    )
    assert index.shared_by == [f"{DRAFT}.compressed", f"{DRAFT}.compressor_state"]
    padded = [t for t in cfg.kv_cache_tensors if t.page_stride_bytes is not None]
    assert len(padded) == len(cfg.host_kv_layers) and padded[-1] is index
    geo, base_geo = plan.geometry, base_plan.geometry
    layer_page = specs[DRAFT].page_size_bytes
    assert geo.hot_layers == len(TARGET_MAIN) + 1
    assert geo.main_kv_bytes_per_block == base_geo.main_kv_bytes_per_block + layer_page
    assert geo.hot_slot_bytes // geo.hot_layers == (
        base_geo.hot_slot_bytes // base_geo.hot_layers
    )
    assert geo.shared_slots_per_block == base_geo.shared_slots_per_block > 0
    assert geo.index_page_strides == (
        *base_geo.index_page_strides,
        base_geo.index_page_strides[-1],
    )
    assert geo.state_bytes_per_block == base_geo.state_bytes_per_block
    # Host row: 13 main K/V layers and 13 index pages.
    layout = host_region_layout(cfg, "test")
    assert layout.main_bytes == geo.main_kv_bytes_per_block
    assert len(layout.index_bytes) == len(cfg.host_kv_layers)
    assert geo.host_bytes_per_block == layout.worker_bytes * layout.num_workers
    # GPU history pays for the draft index, not for its main K/V.
    gpu = plan.history_bytes_per_block - base_plan.history_bytes_per_block
    assert 0 < gpu < layer_page


@pytest.mark.parametrize("placement", ["gpu", "host"])
def test_planar_mtp_main_carries_scales_in_existing_component(placement):
    specs = {
        name: (
            replace(spec, kv_quant_mode=KVQuantMode.FP8_E4M3_PLANAR)
            if type(spec) is FullAttentionSpec
            else spec
        )
        for name, spec in mtp_specs(shard_index=True, shard_draft=True).items()
    }
    _, cfg = build(settings(draft_main_placement=placement), specs)
    plan = cfg.long_context_plan
    geo = plan.geometry
    layout = host_region_layout(cfg, "test-planar-mtp")
    draft_bytes = specs[DRAFT].block_size * 520
    assert specs[DRAFT].page_size_bytes == draft_bytes
    assert geo.hot_slot_bytes == len(cfg.host_kv_layers) * 2080
    if placement == "host":
        assert cfg.host_kv_layers == [*TARGET_MAIN, DRAFT]
        assert layout.main_bytes == 13 * draft_bytes
    else:
        assert cfg.host_kv_layers == TARGET_MAIN
        tensor = next(t for t in cfg.kv_cache_tensors if DRAFT in t.shared_by)
        assert tensor.size == draft_bytes * tensor.num_blocks
        assert draft_bytes in layout.index_bytes
        assert layout.main_bytes == 12 * draft_bytes
    assert layout.worker_bytes * WORKERS == geo.host_bytes_per_block


def test_draft_main_placement_defaults_to_host():
    # A sharded draft's main K/V joins the host store unless "gpu" is asked
    # for explicitly (1M-context concurrency is history-ID bound).
    assert LongContextKVConfig().draft_main_placement == "host"
    specs = mtp_specs(shard_index=True, shard_draft=True)
    _, cfg = build(settings(), specs)
    assert cfg.long_context_plan.draft_main_on_host
    assert cfg.host_kv_layers == [*TARGET_MAIN, DRAFT]


def test_sharded_draft_main_follows_draft_main_placement():
    specs = mtp_specs(shard_index=True, shard_draft=True)
    _, gpu = build(settings(draft_main_placement="gpu"), specs)
    _, host = build(settings(), specs)  # default: "host"
    # "gpu": the draft stays beside the index in GPU history, as before.
    assert gpu.host_kv_layers == TARGET_MAIN
    assert not gpu.long_context_plan.draft_main_on_host
    assert host.host_kv_layers == [*TARGET_MAIN, DRAFT]
    assert host.long_context_plan.draft_main_on_host
    # "host" lowers the GPU bytes of every history ID and so raises the number
    # of 512K contexts the same budget can hold.
    gpu_bytes = gpu.long_context_plan.geometry.index_bytes_per_block
    host_bytes = host.long_context_plan.geometry.index_bytes_per_block
    assert host_bytes < gpu_bytes
    assert (
        host.long_context_plan.gpu_history_blocks
        > gpu.long_context_plan.gpu_history_blocks
    )


def test_draft_main_placement_is_told_apart_by_layout_not_by_the_setting():
    from vllm.v1.kv_offload.tiering.fs.identity import canonical_digest
    from vllm.v1.kv_offload.tiering.fs.revolta import _persistent_kv_tensors

    # The setting is not in the persisted identity; the page layout is: the
    # host layer list and tensors differ.
    specs = mtp_specs(shard_index=True, shard_draft=True)
    _, gpu = build(settings(draft_main_placement="gpu"), specs)
    _, host = build(settings(), specs)

    def layout(cfg):
        return canonical_digest(
            {
                "groups": [list(g.layer_names) for g in cfg.kv_cache_groups],
                "tensors": _persistent_kv_tensors(cfg),
                "host": list(cfg.host_kv_layers),
            }
        )

    assert layout(gpu) != layout(host)


def test_draft_main_placement_falls_back_to_gpu(caplog):
    # A draft that does not share the target's geometry, or main K/V on the
    # GPU, leaves the layout of "gpu" and says so.
    caplog.set_level("WARNING")
    _, replicated = build(settings(draft_main_placement="host"), mtp_specs())
    _, base = build(settings(draft_main_placement="gpu"), mtp_specs())
    assert replicated.host_kv_layers == base.host_kv_layers == TARGET_MAIN
    assert "draft_main_placement='host' needs a draft" in caplog.text
    caplog.clear()
    _, gpu_main = build(
        settings(main_kv_placement="gpu", draft_main_placement="host"),
        mtp_specs(shard_index=True, shard_draft=True),
    )
    assert (
        not gpu_main.host_kv_layers
        and not gpu_main.long_context_plan.draft_main_on_host
    )
    assert "needs main_kv_placement='host'" in caplog.text


def test_mtp_admission_and_other_drafters():
    config = mtp_config(long_context=settings())
    config.kv_transfer_config = SimpleNamespace(kv_connector="OffloadingConnector")
    groups = get_kv_cache_groups(config, mtp_specs())
    geo = get_long_context_geometry(config, groups)
    assert geo.block_tokens == SPAN and geo.decode_rows_per_request == 5
    config.speculative_config.method = "eagle"
    with pytest.raises(ValueError, match="DFlash or MTP"):
        get_long_context_geometry(config, groups)


def test_gpu_placement_charges_the_draft_to_history():
    specs = mtp_specs()
    _, cfg = build(settings(main_kv_placement="gpu"), specs)
    plan = cfg.long_context_plan
    assert plan.main_kv_on_gpu and not cfg.host_kv_layers
    draft = specs[DRAFT].page_size_bytes + specs[f"{DRAFT}.compressed"].page_size_bytes
    _, base = build(
        settings(main_kv_placement="gpu"),
        production_specs(speculative_blocks=0, drafter=False),
        mtp=False,
    )
    assert (
        plan.geometry.index_bytes_per_block
        == base.long_context_plan.geometry.index_bytes_per_block + draft
    )


def test_sharded_draft_layer_gets_its_own_hot_slot_index_tensor():
    import torch

    from vllm.models.qwen4_exp.nvidia.host_kv.engine import shared_index_tensors

    _, cfg = build(
        settings(draft_main_placement="host"),
        mtp_specs(shard_index=True, shard_draft=True),
    )
    padded = [t for t in cfg.kv_cache_tensors if t.page_stride_bytes is not None]
    kv_caches = {t.shared_by[0]: torch.empty(t.size, dtype=torch.uint8) for t in padded}
    views = shared_index_tensors(cfg, kv_caches, len(cfg.host_kv_layers))
    # One distinct index tensor per host layer, the draft's last like its layer.
    assert len(views) == len(cfg.host_kv_layers) == len(TARGET_MAIN) + 1
    assert views[-1] is kv_caches[f"{DRAFT}.compressed"]
