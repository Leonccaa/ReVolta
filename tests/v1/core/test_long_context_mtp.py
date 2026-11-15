# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP drafting in the long-context host-KV layout (CPU).

The MTP draft layer stays in the target's KV cache groups (one block table,
one hash chain), keeps its caches as GPU history tensors and rides in the
host row next to the target's index pages.
"""

from collections import Counter
from types import SimpleNamespace

import pytest

from vllm.config.kernel import LongContextKVConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    get_kv_cache_groups,
    get_long_context_geometry,
)
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


@pytest.mark.parametrize("shard_draft", [False, True])
def test_draft_caches_stay_on_gpu_and_join_the_host_row(shard_draft):
    specs = mtp_specs(shard_index=shard_draft, shard_draft=shard_draft)
    _, cfg = build(settings(), specs)
    plan = cfg.long_context_plan
    assert plan is not None and not plan.main_kv_on_gpu
    # Only the target's main K/V moves to the host store.
    assert sorted(cfg.host_kv_layers) == sorted(TARGET_MAIN)
    # The draft stays in the target's main and ring groups.
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


@pytest.mark.parametrize("shard_draft", [False, True])
def test_draft_does_not_change_the_state_segment(shard_draft):
    lc = settings()
    specs = mtp_specs(shard_index=shard_draft, shard_draft=shard_draft)
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
