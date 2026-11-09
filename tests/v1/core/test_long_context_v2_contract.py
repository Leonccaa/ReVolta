# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""static host allocation startup minimums, placement boundary and physical index
plans."""

from dataclasses import replace

import pytest
import torch

from tests.v1.core.long_context_kv_utils import production_config, production_specs
from tests.v1.core.test_long_context_kv_capacity import GPU_BUDGET, geometry
from vllm.config.kernel import LongContextKVConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.long_context_kv import (
    GiB,
    plan_long_context_capacity,
    validate_long_context_plan,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("value", [0.01, 1.0])
def test_manual_gib_is_a_total_budget_slot_floor(value):
    geo = geometry()
    settings = LongContextKVConfig(
        enabled=True, host_pool_gib=96.13, hot_pool_gib=value
    )
    plan = plan_long_context_capacity(settings, geo, GPU_BUDGET, 8)
    validate_long_context_plan(plan)
    converted = geo.hot_groups_for_bytes(int(value * GiB), 8)
    assert plan.hot_pool_min_groups == converted
    assert plan.hot_pool_groups >= max(
        converted, geo.hot_reserve_groups(geo.max_model_len)
    )
    assert plan.hot_floor_groups(converted + 5) == converted + 5
    assert plan.hot_floor_groups(0) == converted
    assert geo.hot_bytes_for_groups(converted + 1, 8) > int(value * GiB)


def test_manual_minimum_is_never_silently_cut_to_fit_active_history():
    geo = geometry()
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=96.13, hot_pool_gib=5.5),
        geo,
        GPU_BUDGET,
        8,
    )
    assert plan.hot_pool_groups >= plan.hot_pool_min_groups
    with pytest.raises(ValueError, match="GPU history segment"):
        validate_long_context_plan(plan)


@pytest.mark.parametrize("transfer", [None, "custom"])
def test_host_requires_builtin_connector(transfer):
    from types import SimpleNamespace

    settings = LongContextKVConfig(enabled=True, host_pool_gib=96.13, hot_pool_gib=1.0)
    config = production_config(long_context=settings)
    config.model_config.original_max_model_len = config.model_config.max_model_len
    config.kv_transfer_config = (
        None
        if transfer is None
        else SimpleNamespace(
            kv_connector="OffloadingConnector", kv_connector_module_path="custom"
        )
    )
    with pytest.raises(ValueError, match="OffloadingConnector"):
        get_kv_cache_configs(config, [production_specs()], [GPU_BUDGET])


def test_standin_plan_pads_only_the_index_tensor_not_the_spec():
    settings = LongContextKVConfig(
        enabled=True, host_pool_tokens=100 * 1568, hot_pool_gib=0.1
    )
    config = production_config(
        max_model_len=8192,
        max_num_seqs=4,
        num_speculative_tokens=0,
        long_context=settings,
    )
    config.model_config.original_max_model_len = 8192
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.model_config.get_num_kv_heads = lambda parallel: 2
    config.scheduler_config.max_num_batched_tokens = 2048
    specs = production_specs(speculative_blocks=0, drafter=False)
    prefix = "model.layers.3.self_attn"
    specs = {
        name: spec
        for name, spec in specs.items()
        if name.startswith(prefix) or name == "model.layers.0.linear_attn"
    }
    specs[prefix] = FullAttentionSpec(
        block_size=1568, num_kv_heads=2, head_size=256, dtype=torch.float16
    )
    index = MLAAttentionSpec(
        block_size=1568,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
    )
    specs[prefix + ".compressed"] = index
    specs["model.layers.0.linear_attn"] = replace(
        specs["model.layers.0.linear_attn"], block_size=1568
    )
    (cfg,) = get_kv_cache_configs(config, [specs], [3 * GiB])
    (tensor,) = [
        t for t in cfg.kv_cache_tensors if prefix + ".compressed" in t.shared_by
    ]
    assert tensor.page_stride_bytes == 106496
    assert tensor.size == 106496 * tensor.num_blocks
    assert cfg.long_context_plan.geometry.index_bytes_per_block == 106496
    assert index.page_size_bytes == 100352 and index.page_size_padded is None
    assert prefix + ".compressor_state" in tensor.shared_by


@pytest.mark.parametrize(
    "capture_sizes,mode", [([264], "FULL_DECODE_ONLY"), ([64], "FULL")]
)
def test_worker_checks_resolved_graph_collection_before_allocation(
    monkeypatch, capture_sizes, mode
):
    from types import SimpleNamespace

    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.qwen4_exp.nvidia.host_kv.engine import build_host_history_store

    settings = LongContextKVConfig(
        enabled=True, host_pool_tokens=8 * 1664, hot_pool_gib=0.2
    )
    config = production_config(
        max_model_len=4 * 1664, max_num_seqs=1, long_context=settings
    )
    config.model_config.original_max_model_len = 4 * 1664
    (cfg,) = get_kv_cache_configs(config, [production_specs()], [GPU_BUDGET])
    layers = {
        name: SimpleNamespace(
            qsa_dcp_sharded=False, indexer=SimpleNamespace(output_width=2051)
        )
        for name in cfg.host_kv_layers
    }
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode[mode], cudagraph_capture_sizes=capture_sizes
    )

    def no_allocation(*args, **kwargs):
        pytest.fail("Unsafe graph configuration reached the allocator")

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.host_kv.engine.HostHistoryStore", no_allocation
    )
    with pytest.raises(ValueError, match="R_cap|mixed_mode"):
        build_host_history_store(
            cfg, config, layers, [832] * len(cfg.kv_cache_groups), torch.device("cpu")
        )
