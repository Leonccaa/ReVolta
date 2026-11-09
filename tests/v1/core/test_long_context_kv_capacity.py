# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context (Active Host KV) configuration, capacity plan and startup check."""

from dataclasses import replace

import pydantic
import pytest
import torch

from vllm.config.kernel import KernelConfig, LongContextKVConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    get_kv_cache_groups,
    get_long_context_geometry,
)
from vllm.v1.core.long_context_kv import (
    GPU_HISTORY,
    HOST,
    STATE,
    GiB,
    plan_long_context_capacity,
    validate_long_context_plan,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from .long_context_kv_utils import (
    INDEX_BLOCK_BYTES,
    MAIN_KV_BLOCK_BYTES,
    MI,
    POOL_BLOCK_BYTES,
    SPAN,
    production_config,
    production_specs,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Deployed KV memory per GPU (11-06 startup log: 6.05 GiB).
GPU_BUDGET = int(6.05 * GiB)
WORKERS = 4
HOST_BLOCK_BYTES = WORKERS * MAIN_KV_BLOCK_BYTES  # 12,288 B/token over 4 GPUs
STATE_BLOCK_BYTES = MAIN_KV_BLOCK_BYTES  # six 851,968-byte owner pages


def geometry(*, max_model_len=MI, speculative_blocks=7, num_spec=7):
    config = production_config(
        max_model_len=max_model_len, num_speculative_tokens=num_spec
    )
    groups = get_kv_cache_groups(
        config, production_specs(speculative_blocks=speculative_blocks)
    )
    return get_long_context_geometry(config, groups)


def test_production_geometry_reproduces_the_m0_ledger():
    geo = geometry()
    assert geo.unified_bytes_per_block == POOL_BLOCK_BYTES
    assert geo.block_tokens == SPAN
    assert geo.main_kv_bytes_per_block == MAIN_KV_BLOCK_BYTES
    assert geo.main_kv_bytes_per_block // SPAN == 3072  # B/token/GPU
    assert geo.index_bytes_per_block == INDEX_BLOCK_BYTES  # 768 B/token/GPU
    assert geo.state_bytes_per_block == STATE_BLOCK_BYTES
    assert geo.num_workers == WORKERS
    assert geo.host_bytes_per_block // SPAN == 12_288
    # 6 GDN groups x (2 + 7 speculative) + PLE 2 + DFlash2 window 15.
    assert geo.state_blocks_per_request == 6 * 9 + 2 + 15
    assert geometry(speculative_blocks=0).state_blocks_per_request == 6 * 2 + 2 + 15
    assert geo.ring_blocks_per_request == 1
    assert geo.history_blocks(524_288) == 316
    assert geo.history_blocks(MI) == 631
    # Main KV + index stay together; ring is GPU only; states are GPU state.
    assert geo.group_domains[0] == (GPU_HISTORY, HOST)
    assert geo.group_domains[1] == (GPU_HISTORY,)
    assert set(geo.group_domains[2:]) == {(STATE,)}
    # 512 top-k groups + tail group per row, 8 verify rows, 9 pinned groups.
    assert (geo.hot_row_groups, geo.decode_rows_per_request) == (513, 8)
    assert geo.hot_reserve_groups(MI) == 8 * 513 + 9
    assert geo.hot_reserve_groups(100) == 25  # never more than its history
    assert geometry(num_spec=0).hot_reserve_groups(MI) == 513 + 9


def expected(n, max_len, *, replay, shard, hot_gib, host_gib):
    """Hand-written capacity arithmetic for the table cases."""
    state_per_req = 6 * (2 if replay else 9) + 2 + 15
    state_blocks = n * state_per_req
    index_bytes = INDEX_BLOCK_BYTES // (2 if shard else 1)
    hot_groups = int(hot_gib * GiB) * SPAN // (MAIN_KV_BLOCK_BYTES * 4)
    hot_bytes = -(-hot_groups * MAIN_KV_BLOCK_BYTES * 4 // SPAN)
    left = GPU_BUDGET - (state_blocks + 1) * STATE_BLOCK_BYTES - hot_bytes
    gpu_history = max(left // index_bytes - 1, 0)
    host_blocks = int(host_gib * GiB) // HOST_BLOCK_BYTES
    history = -(-max_len // SPAN)
    return dict(
        state_blocks=state_blocks,
        gpu_history_blocks=gpu_history,
        host_history_blocks=host_blocks,
        hot_pool_groups=hot_groups,
        concurrency=min(
            gpu_history // (history + 1),
            host_blocks // history,
            hot_groups // min(-(-max_len // 4), 8 * 513 + 9),
            n,
        ),
    )


CASES = [
    # (requests, tokens, host GiB covering exactly those requests)
    (8, MI, 96.13),
    (8, 800_000, 73.28),
    (4, MI, 48.07),
    (8, 524_288, 48.15),
]


@pytest.mark.parametrize("n,max_len,host_gib", CASES)
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("shard", [False, True])
@pytest.mark.parametrize("hot_gib", [0.5, 1.0])
def test_capacity_combinations(n, max_len, host_gib, replay, shard, hot_gib):
    geo = geometry(max_model_len=max_len, speculative_blocks=0 if replay else 7)
    if shard:  # index sharding halves the per-GPU index pages
        geo = replace(geo, index_bytes_per_block=geo.index_bytes_per_block // 2)
    settings = LongContextKVConfig(
        enabled=True, host_pool_gib=host_gib, hot_pool_gib=hot_gib
    )
    plan = plan_long_context_capacity(settings, geo, GPU_BUDGET, n)
    want = expected(
        n, max_len, replay=replay, shard=shard, hot_gib=hot_gib, host_gib=host_gib
    )
    assert plan.state_blocks == want["state_blocks"]
    assert plan.gpu_history_blocks == want["gpu_history_blocks"]
    assert plan.host_history_blocks == want["host_history_blocks"]
    assert plan.host_history_blocks >= n * geo.history_blocks(max_len)
    assert plan.hot_pool_groups == want["hot_pool_groups"]
    assert min(plan.max_concurrency(max_len).values()) == want["concurrency"]
    # Every case holds at least one full request, so startup accepts it; the
    # GPU history segment is what limits concurrency below n.
    assert plan.startup_problems() == []


def test_capacity_table_headlines():
    """TP4/DCP2 capacity derived from the reference geometry below."""
    # 8 x 1Mi, state replay on, index sharded (index sharding), 0.5 GiB hot pool: the
    # GPU history segment holds 7,461 blocks, 11 x (631 + 1).
    geo = replace(
        geometry(speculative_blocks=0), index_bytes_per_block=INDEX_BLOCK_BYTES // 2
    )
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=96.13, hot_pool_gib=0.5),
        geo,
        GPU_BUDGET,
        8,
    )
    assert plan.gpu_history_blocks == 7461
    assert plan.max_concurrency(MI) == {
        "GPU state segment": 8,
        "GPU history segment": 11,
        "host history pool": 8,
        "GPU hot-page pool": 10,
        "max_num_seqs": 8,
    }
    # Replicated index, replay off: only 3 x 1Mi fit on the GPU side.
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=96.13, hot_pool_gib=0.5),
        geometry(),
        GPU_BUDGET,
        8,
    )
    assert plan.max_concurrency(MI)["GPU history segment"] == 3


def test_hot_pool_ratio_defaults_to_ten_percent_of_host_history():
    geo = geometry()
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_tokens=8 * 631 * SPAN),
        geo,
        GPU_BUDGET,
        8,
    )
    assert plan.host_history_blocks == 8 * 631
    assert plan.hot_pool_groups == int(8 * 631 * SPAN * 0.10) // 4
    half = plan_long_context_capacity(
        LongContextKVConfig(
            enabled=True, host_pool_tokens=8 * 631 * SPAN, hot_pool_ratio=0.05
        ),
        geo,
        GPU_BUDGET,
        8,
    )
    assert half.hot_pool_groups == int(8 * 631 * SPAN * 0.05) // 4
    # 10 % of 8 x 1Mi on every GPU is 2.4 GiB: too much next to the index.
    assert plan.hot_pool_bytes > 2.3 * GiB


@pytest.mark.parametrize(
    "settings,state_blocks,domain,fix",
    [
        (
            dict(host_pool_gib=10.0, hot_pool_gib=0.5),
            None,
            "host history pool",
            dict(host_pool_gib=None, host_pool_tokens=631 * SPAN),
        ),
        (
            dict(host_pool_gib=96.13, hot_pool_gib=0.01),
            None,
            "GPU hot-page pool",
            dict(hot_pool_gib=0.05),
        ),
        (
            dict(host_pool_gib=96.13, hot_pool_gib=0.5, state_blocks=10),
            None,
            "GPU state segment",
            dict(state_blocks=71),
        ),
        (
            dict(host_pool_gib=96.13, hot_pool_gib=5.0),
            None,
            "GPU history segment",
            dict(hot_pool_gib=0.5),
        ),
    ],
)
def test_startup_refuses_a_pool_that_cannot_hold_one_request(
    settings, state_blocks, domain, fix
):
    del state_blocks
    geo = geometry()
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, **settings), geo, GPU_BUDGET, 1
    )
    with pytest.raises(ValueError, match=domain) as info:
        validate_long_context_plan(plan)
    assert "max_model_len=1,048,576" in str(info.value)
    # The suggested setting is enough on its own.
    fixed = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, **{**settings, **fix}), geo, GPU_BUDGET, 1
    )
    validate_long_context_plan(fixed)


def test_startup_suggestion_values_round_trip():
    geo = geometry()
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=1.0, hot_pool_gib=0.01),
        geo,
        GPU_BUDGET,
        1,
    )
    problems = plan.startup_problems()
    host = next(p for p in problems if "host history pool" in p)
    hot = next(p for p in problems if "hot-page pool" in p)
    host_gib = float(host.split("host_pool_gib >= ")[1].split(" ")[0])
    hot_gib = float(hot.split("hot_pool_gib >= ")[1].rstrip("."))
    assert (host_gib, hot_gib) == (12.02, 0.05)
    fixed = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=host_gib, hot_pool_gib=hot_gib),
        geo,
        GPU_BUDGET,
        1,
    )
    assert fixed.startup_problems() == []
    # A 7 GiB pool fits a shorter context, which the error also reports.
    short = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=7.0, hot_pool_gib=0.05),
        geo,
        GPU_BUDGET,
        1,
    )
    assert short.max_fitting_len() == 367 * SPAN
    assert "largest max_model_len is 610,688" in short.startup_problems()[-1]
    # A hot pool below one request's verify working set caps the context at
    # the history it can pin whole: 3,495 groups x 4 tokens.
    tiny_hot = replace(short, hot_pool_groups=3495, host_history_blocks=631)
    assert tiny_hot.max_fitting_len() == 3495 * 4


def test_settings_validation():
    LongContextKVConfig().validate_settings()  # disabled: nothing required
    LongContextKVConfig(hot_pool_gib=1, hot_pool_ratio=0.1).validate_settings()
    with pytest.raises(ValueError, match="exactly one of host_pool_gib"):
        LongContextKVConfig(enabled=True).validate_settings()
    with pytest.raises(ValueError, match="exactly one of host_pool_gib"):
        LongContextKVConfig(
            enabled=True, host_pool_gib=1, host_pool_tokens=SPAN
        ).validate_settings()
    with pytest.raises(ValueError, match="at most one of"):
        LongContextKVConfig(
            enabled=True, host_pool_gib=1, hot_pool_gib=1, hot_pool_ratio=0.1
        ).validate_settings()
    for bad in (
        dict(host_pool_gib=0),
        dict(host_pool_tokens=-1),
        dict(hot_pool_gib=0),
        dict(hot_pool_ratio=1.5),
        dict(prefill_staging_gib=-1),
        dict(state_blocks=0),
    ):
        with pytest.raises(pydantic.ValidationError):
            LongContextKVConfig(**bad)
    config = KernelConfig(
        long_context_kv={"enabled": True, "host_pool_gib": 96, "hot_pool_gib": 0.5}
    )
    assert config.long_context_kv.enabled
    assert config.long_context_kv.prewarm_hot_pool


def test_disabled_mode_does_not_change_the_compile_hash():
    base = KernelConfig().compute_hash()
    inert = KernelConfig(
        long_context_kv=LongContextKVConfig(host_pool_gib=5, hot_pool_ratio=0.2)
    ).compute_hash()
    assert inert == base
    enabled = KernelConfig(
        long_context_kv=LongContextKVConfig(enabled=True, host_pool_gib=5)
    ).compute_hash()
    assert enabled != base


def kv_cache_configs(settings, *, max_model_len=MI, memory=GPU_BUDGET):
    config = production_config(max_model_len=max_model_len, long_context=settings)
    config.model_config.original_max_model_len = max_model_len
    specs = production_specs()
    return get_kv_cache_configs(config, [specs] * WORKERS, [memory] * WORKERS)


def test_startup_attaches_the_plan_without_changing_the_unified_pool():
    off = kv_cache_configs(LongContextKVConfig())
    on = kv_cache_configs(
        LongContextKVConfig(enabled=True, host_pool_gib=13.0, hot_pool_gib=0.1)
    )
    for disabled, enabled in zip(off, on):
        assert disabled.long_context_plan is None
        assert enabled.long_context_plan is not None
        assert (
            disabled.num_blocks
            == enabled.num_blocks
            == GPU_BUDGET // (POOL_BLOCK_BYTES)
        )
        assert disabled.kv_cache_tensors == enabled.kv_cache_tensors
        assert disabled.kv_cache_groups == enabled.kv_cache_groups
    plan = on[0].long_context_plan
    assert plan.gpu_budget_bytes == GPU_BUDGET
    assert plan.host_history_blocks == 13 * GiB // (WORKERS * MAIN_KV_BLOCK_BYTES)
    assert plan.geometry.max_model_len == MI


def test_startup_rejects_too_small_pools():
    with pytest.raises(ValueError, match="host history pool"):
        kv_cache_configs(
            LongContextKVConfig(enabled=True, host_pool_gib=1.0, hot_pool_gib=0.1)
        )


def test_unsupported_layouts_are_rejected_at_startup():
    settings = LongContextKVConfig(enabled=True, host_pool_gib=13.0)
    config = production_config(long_context=settings)
    plain = {
        "layer": FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
    }
    with pytest.raises(ValueError, match="compressed-sparse"):
        get_long_context_geometry(config, get_kv_cache_groups(config, plain))
    groups = get_kv_cache_groups(config, production_specs())
    mtp = production_config(long_context=settings, dflash=False)
    with pytest.raises(ValueError, match="DFlash drafting or no speculation"):
        get_long_context_geometry(mtp, groups)
    pp = production_config(long_context=settings)
    pp.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline parallelism"):
        get_long_context_geometry(pp, groups)
    no_spec = production_config(long_context=settings)
    no_spec.speculative_config = None
    no_drafter = get_kv_cache_groups(no_spec, production_specs(drafter=False))
    geo = get_long_context_geometry(no_spec, no_drafter)
    assert geo.decode_rows_per_request == 1
    assert geo.state_blocks_per_request == 6 * 9 + 2
