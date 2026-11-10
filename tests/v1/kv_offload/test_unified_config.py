# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""legacy-budget conversion conversion is planning-only and cannot compound across
workers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.v1.core.long_context_kv_utils import production_config, production_specs
from vllm.config.kernel import LongContextKVConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_offload.host.region import host_region_layout

pytestmark = pytest.mark.cpu_test


def plan(settings, extra=None):
    c = production_config(max_model_len=524288, long_context=settings, max_num_seqs=8)
    c.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector", kv_connector_extra_config=extra or {}
    )
    c.model_config.original_max_model_len = 524288
    c.kernel_config.qsa_dcp_index_shard = True
    specs = production_specs()
    for name, spec in list(specs.items()):
        if name.endswith(".compressed"):
            specs[name] = replace(spec, block_size=832, dcp_sharded=True)
    workers = get_kv_cache_configs(c, [specs] * 4, [12 << 30] * 4)
    return c, workers


def test_budget_frame_width_zero_page_and_legacy_conversion():
    settings = LongContextKVConfig(enabled=True, host_store_gib=98, hot_pool_gib=2)
    c, workers = plan(settings)
    layout = host_region_layout(workers[0], "test")
    assert layout.worker_bytes == 5_750_784
    assert layout.num_workers * layout.worker_bytes == 23_003_136
    assert layout.num_pages == 4574  # Includes ZERO_PAGE, 4573 data frames.
    legacy = LongContextKVConfig(enabled=True, host_pool_gib=90, hot_pool_gib=2)
    _, old = plan(legacy, {"cpu_bytes_to_use": 8 << 30})
    assert all(
        w.long_context_plan.host_pool_bytes
        == workers[0].long_context_plan.host_pool_bytes
        for w in old
    )
    assert legacy.host_pool_gib == 90 and legacy.host_store_gib is None
    _, again = plan(legacy, {"cpu_bytes_to_use": 8 << 30})
    assert again[0].long_context_plan == old[0].long_context_plan


@pytest.mark.parametrize("other", ["host_pool_gib", "host_pool_tokens"])
def test_new_and_old_host_fields_conflict(other):
    cfg = LongContextKVConfig(enabled=True, host_store_gib=98, **{other: 90})
    with pytest.raises(ValueError, match="exactly one"):
        cfg.validate_settings()


def test_new_budget_rejects_connector_cpu_even_if_equal():
    with pytest.raises(ValueError, match="conflicts"):
        plan(
            LongContextKVConfig(enabled=True, host_store_gib=98, hot_pool_gib=2),
            {"cpu_bytes_to_use": 98 << 30},
        )


def test_disabled_configuration_is_inert():
    LongContextKVConfig(
        enabled=False, host_store_gib=98, host_pool_gib=90
    ).validate_settings()


def test_retention_defaults_and_compile_identity():
    from vllm.config import KernelConfig

    settings = LongContextKVConfig(enabled=True, host_store_gib=4)
    assert settings.latest_only and not settings.superseded_keep_ssd
    config = KernelConfig(long_context_kv=settings)
    original = config.compute_hash()
    settings.latest_only = False
    settings.superseded_keep_ssd = True
    assert config.compute_hash() == original
