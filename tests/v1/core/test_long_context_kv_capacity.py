# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context (Active Host KV) configuration, capacity plan and startup check."""

from dataclasses import replace
from types import SimpleNamespace

import pydantic
import pytest
import torch

from vllm.config.kernel import KernelConfig, LongContextKVConfig
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
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
LOCAL = SPAN // 2  # rank-local main K/V tokens of a history block under DCP2
# Host placement: the host store's GPU metadata per history block, i.e. the
# hot-group map of 12 layers + the wanted map (int32 per 4 local tokens), the
# readiness bytes and the generation.
HOST_META_BYTES = 12 * LOCAL + LOCAL + LOCAL + 8
HOT_GROUP_BYTES = 12 * 4 * 2 * 256  # four local tokens of K and V, 12 layers


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
    assert geo.hot_reserve_groups(100) == 13  # never more than its history
    assert geometry(num_spec=0).hot_reserve_groups(MI) == 513 + 9
    assert (geo.dcp_world_size, geo.local_block_tokens) == (2, LOCAL)
    assert geo.hot_layers == 12
    assert geo.host_metadata_bytes_per_block == HOST_META_BYTES == 11_656
    assert geo.hot_slot_bytes == HOT_GROUP_BYTES


def test_hot_slots_use_worker_local_dcp_geometry():
    geo = geometry()
    # Twelve layers of [slot, K/V, four LOCAL tokens, head=1, dim=256] uint8.
    assert geo.hot_slot_bytes == 12 * 2 * 4 * 256 == 24_576
    rank0_tokens = [
        (4 * group + offset) // 2 for group in range(0, 1024, 2) for offset in (0, 2)
    ]
    assert len({token // 4 for token in rank0_tokens}) == 512
    parts = geo.hot_memory_bytes(4113, 1)
    assert parts["payload"] == 4113 * 24_576
    assert parts["metadata"] > 0
    assert parts["workspace"] > 0
    budget = geo.hot_bytes_for_groups(4113, 1)
    assert geo.hot_groups_for_bytes(budget, 1) == 4113
    assert geo.hot_groups_for_bytes(budget - 1, 1) == 4112


def test_hot_memory_uses_the_engine_store_workspace():
    from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryShape

    geo = geometry(max_model_len=128)
    shape = HostHistoryShape(
        num_blocks=2,
        kernel_blocks_per_block=1,
        kernel_block_tokens=832,
        num_layers=12,
        kv_heads=1,
        head_dim=256,
        dtype=torch.uint8,
        hot_slots=4113,
        max_requests=1,
        selection_width=2051,
        max_write_rows=8192,
    )
    parts = shape.fixed_device_memory_bytes()
    actual = geo.hot_memory_bytes(4113)
    assert actual["workspace"] == sum(
        parts[name] for name in ("indices", "staging", "staging_slots")
    )
    assert sum(actual.values()) == sum(parts.values())


def expected(n, max_len, *, replay, shard, hot_gib, host_gib):
    """Hand-written capacity arithmetic for the table cases (main K/V in
    the host store: a history block costs index pages + store metadata, and
    the history IDs never outnumber the host pages, one of which backs the
    null block)."""
    state_per_req = 6 * (2 if replay else 9) + 2 + 15
    state_blocks = n * state_per_req
    index_bytes = INDEX_BLOCK_BYTES // (2 if shard else 1) + HOST_META_BYTES
    geo = geometry(max_model_len=max_len, speculative_blocks=0 if replay else 7)
    hot_groups = geo.hot_groups_for_bytes(int(hot_gib * GiB), n)
    hot_bytes = geo.hot_bytes_for_groups(hot_groups, n)
    left = GPU_BUDGET - (state_blocks + 1) * STATE_BLOCK_BYTES - hot_bytes
    host_blocks = int(host_gib * GiB) // HOST_BLOCK_BYTES
    gpu_history = min(max(left // index_bytes - 1, 0), host_blocks - 1)
    history = -(-max_len // SPAN)
    return dict(
        state_blocks=state_blocks,
        gpu_history_blocks=gpu_history,
        host_history_blocks=host_blocks,
        hot_pool_groups=hot_groups,
        concurrency=min(
            gpu_history // (history + 1),
            host_blocks // history,
            hot_groups // min(-(-max_len // 2 // 4), 8 * 513 + 9),
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
    assert bool(plan.startup_problems()) is (want["concurrency"] == 0)
    assert plan.hot_pool_bytes == sum(
        geo.hot_memory_bytes(plan.hot_pool_groups, n).values()
    )
    if plan.hot_pool_groups:
        assert plan.hot_pool_bytes <= int(hot_gib * GiB)
        assert geo.hot_bytes_for_groups(plan.hot_pool_groups + 1, n) > int(
            hot_gib * GiB
        )


def test_capacity_table_headlines():
    """Domain arithmetic only; physical engine concurrency is unverified."""
    # Block-ID metadata is charged to history, not duplicated per request
    # inside the fixed hot-pool budget.
    geo = replace(
        geometry(speculative_blocks=0), index_bytes_per_block=INDEX_BLOCK_BYTES // 2
    )
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=96.13, hot_pool_gib=0.5),
        geo,
        GPU_BUDGET,
        8,
    )
    assert plan.hot_pool_groups > 0
    assert 0 < plan.max_concurrency(MI)["GPU hot-page pool"] < 8
    # One GiB now covers ten requests' decode unions; the scheduler's eight
    # sequence limit still applies. This is an arithmetic upper bound.
    larger = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=96.13, hot_pool_gib=1.0),
        geo,
        GPU_BUDGET,
        8,
    )
    assert larger.max_concurrency(MI)["GPU hot-page pool"] == 10

    # Replicated index, replay off: only 3 x 1Mi fit on the GPU side.
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=96.13, hot_pool_gib=0.5),
        geometry(),
        GPU_BUDGET,
        8,
    )
    assert plan.max_concurrency(MI)["GPU history segment"] == 3


def test_explicit_hot_pool_ratio_keeps_manual_sizing():
    geo = geometry()
    plan = plan_long_context_capacity(
        LongContextKVConfig(
            enabled=True, host_pool_tokens=8 * 631 * SPAN, hot_pool_ratio=0.10
        ),
        geo,
        GPU_BUDGET,
        8,
    )
    assert plan.host_history_blocks == 8 * 631
    assert plan.hot_pool_groups == int(8 * 631 * SPAN * 0.10 / 2) // 4
    half = plan_long_context_capacity(
        LongContextKVConfig(
            enabled=True, host_pool_tokens=8 * 631 * SPAN, hot_pool_ratio=0.05
        ),
        geo,
        GPU_BUDGET,
        8,
    )
    assert half.hot_pool_groups == int(8 * 631 * SPAN * 0.05 / 2) // 4
    # 10 % of 8 x 1Mi on every GPU is 2.4 GiB: too much next to the index.
    assert plan.hot_pool_bytes > 2.3 * GiB
    assert plan.hot_pool_source == half.hot_pool_source == "manual"


@pytest.mark.parametrize("host_gib", [24.0, 96.13, 112.0])
@pytest.mark.parametrize("explicit", [False, True])
def test_auto_uses_demand_sized_history_then_all_remaining_memory(host_gib, explicit):
    geo = replace(
        geometry(speculative_blocks=0), index_bytes_per_block=INDEX_BLOCK_BYTES // 2
    )
    settings = LongContextKVConfig(
        enabled=True,
        host_pool_gib=host_gib,
        hot_pool_gib="auto" if explicit else None,
        prefill_staging_gib=0.125,
    )
    plan = plan_long_context_capacity(settings, geo, GPU_BUDGET, 8)
    validate_long_context_plan(plan)
    assert plan.hot_pool_source == "auto"
    assert plan.gpu_history_blocks == min(8 * (631 + 1), plan.host_history_blocks - 1)
    assert plan.hot_pool_groups >= 8 * (8 * 513 + 9)
    used = (
        plan.gpu_state_bytes
        + plan.gpu_history_bytes
        + plan.gpu_hot_pool_bytes
        + plan.staging_bytes
    )
    next_slot = (
        geo.hot_bytes_for_groups(plan.hot_pool_groups + 1, 8) - plan.hot_pool_bytes
    )
    assert 0 <= plan.gpu_remaining_bytes == GPU_BUDGET - used < next_slot
    text = plan.describe()
    for key, value in (
        ("state", plan.gpu_state_bytes),
        ("history", plan.gpu_history_bytes),
        ("hot", plan.gpu_hot_pool_bytes),
        ("staging", plan.staging_bytes),
        ("remaining", plan.gpu_remaining_bytes),
        ("hot_pool_source", "auto"),
    ):
        assert f"{key}={value}" in text


@pytest.mark.parametrize("placement", ["host", "gpu"])
def test_auto_verify_minimum_and_budget_suggestion_round_trip(placement):
    geo = geometry(max_model_len=32, speculative_blocks=0)
    seqs = 8
    settings = LongContextKVConfig(
        enabled=True,
        host_pool_tokens=100 * SPAN,
        main_kv_placement=placement,
        prefill_staging_gib=0.125,
    )
    plan = plan_long_context_capacity(settings, geo, GPU_BUDGET, seqs)
    geo = plan.geometry
    groups = seqs * (
        (8 * 513 + 9) if placement == "gpu" else geo.hot_reserve_groups(32)
    )
    minimum = (
        plan.gpu_state_bytes
        + (plan.history_demand_blocks + 1) * plan.history_bytes_per_block
        + plan.staging_bytes
        + geo.hot_bytes_for_groups(groups, seqs)
    )
    enough = plan_long_context_capacity(settings, geo, minimum, seqs)
    validate_long_context_plan(enough)
    assert enough.hot_pool_groups == groups
    for budget in (minimum - 1, plan.gpu_state_bytes - 1, 0):
        small = plan_long_context_capacity(settings, geo, budget, seqs)
        with pytest.raises(
            ValueError, match=f"verify working sets need {groups}"
        ) as info:
            validate_long_context_plan(small)
        assert f"GPU KV budget >= {minimum} bytes" in str(info.value)
        assert "hot_pool_gib >= " in str(info.value)
        assert "largest max_model_len" not in str(info.value)


def test_gpu_auto_keeps_physical_history_and_only_budgets_hot_admission():
    geo = geometry(max_model_len=32, speculative_blocks=0)
    kwargs = dict(enabled=True, host_pool_gib=96.13, main_kv_placement="gpu")
    auto = plan_long_context_capacity(LongContextKVConfig(**kwargs), geo, GPU_BUDGET, 8)
    manual = plan_long_context_capacity(
        LongContextKVConfig(**kwargs, hot_pool_gib=1.0), geo, GPU_BUDGET, 8
    )
    validate_long_context_plan(auto)
    assert auto.gpu_hot_pool_bytes == 0
    assert auto.gpu_history_blocks == manual.gpu_history_blocks
    assert auto.segment_num_blocks() == manual.segment_num_blocks()
    assert auto.hot_pool_groups > 0
    assert "hot pool budgets admission only" in auto.describe()


def test_auto_still_requires_a_manually_sized_host_pool():
    with pytest.raises(
        ValueError, match="exactly one of (?:host_store_gib, )?host_pool_gib"
    ):
        LongContextKVConfig(enabled=True, hot_pool_gib="auto").validate_settings()
    with pytest.raises(ValueError, match="at most one of"):
        LongContextKVConfig(
            enabled=True, host_pool_gib=96, hot_pool_gib="auto", hot_pool_ratio=0.1
        ).validate_settings()


@pytest.mark.parametrize(
    "settings,state_blocks,domain,fix",
    [
        (
            dict(host_pool_gib=10.0, hot_pool_gib=0.5),
            None,
            "host history pool",
            # 631 history + 1 ring IDs, plus the null block's page.
            dict(host_pool_gib=None, host_pool_tokens=633 * SPAN),
        ),
        (
            dict(host_pool_gib=96.13, hot_pool_gib=0.01, prefill_staging_gib=6.0),
            None,
            "GPU history segment",
            dict(prefill_staging_gib=0),
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
    # A manual value is a lower bound: static host allocation can allocate more to fit
    # the
    # minimum active verify set. The host error remains actionable.
    assert plan.hot_pool_groups > plan.hot_pool_min_groups
    host_gib = float(host.split("host_pool_gib >= ")[1].split(" ")[0])
    hot_gib = 0.01
    assert host_gib == 12.06
    fixed = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=host_gib, hot_pool_gib=hot_gib),
        geo,
        GPU_BUDGET,
        1,
    )
    assert fixed.startup_problems() == []
    # A 7 GiB pool (367 pages: null + ring + 365 history IDs) fits a shorter
    # context, which the error also reports.
    short = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_gib=7.0, hot_pool_gib=0.3),
        geo,
        GPU_BUDGET,
        1,
    )
    assert short.host_history_blocks == 367
    assert short.max_fitting_len() == 365 * SPAN
    assert "largest max_model_len is 607,360" in short.startup_problems()[-1]
    # A hot pool below one request's verify working set caps the context at
    # the history it can pin whole: 3,495 local groups x eight global tokens.
    tiny_hot = replace(short, hot_pool_groups=3495, host_history_blocks=631)
    assert tiny_hot.max_fitting_len() == 3495 * 8


def test_settings_validation():
    LongContextKVConfig().validate_settings()  # disabled: nothing required
    LongContextKVConfig(hot_pool_gib=1, hot_pool_ratio=0.1).validate_settings()
    with pytest.raises(
        ValueError, match="exactly one of (?:host_store_gib, )?host_pool_gib"
    ):
        LongContextKVConfig(enabled=True).validate_settings()
    with pytest.raises(
        ValueError, match="exactly one of (?:host_store_gib, )?host_pool_gib"
    ):
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
        dict(hot_pool_gib=-1),
        dict(hot_pool_gib="automatic"),
        dict(hot_pool_gib=float("inf")),
        dict(hot_pool_gib=float("nan")),
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
    assert (
        KernelConfig(
            long_context_kv=LongContextKVConfig(hot_pool_gib="auto", host_pool_gib=5)
        ).compute_hash()
        == base
    )
    enabled = KernelConfig(
        long_context_kv=LongContextKVConfig(enabled=True, host_pool_gib=5)
    ).compute_hash()
    assert enabled != base


def kv_cache_configs(
    settings, *, max_model_len=MI, memory=GPU_BUDGET, speculative_blocks=7
):
    config = production_config(max_model_len=max_model_len, long_context=settings)
    config.model_config.original_max_model_len = max_model_len
    specs = production_specs(speculative_blocks=speculative_blocks)
    return get_kv_cache_configs(config, [specs] * WORKERS, [memory] * WORKERS)


def test_static_auto_uses_common_minimum_budget_for_every_worker(
    monkeypatch,
):
    # Static and shared hot pools both need the same cross-rank pass plan.
    monkeypatch.setattr("vllm.v1.core.long_context_kv.SHARED_HOT_REGION", False)
    settings = LongContextKVConfig(enabled=True, host_pool_gib=112)
    config = production_config(max_model_len=32, long_context=settings)
    config.model_config.original_max_model_len = 32
    specs = production_specs(speculative_blocks=0)
    budgets = [5 * GiB, 3 * GiB, 4 * GiB, 3 * GiB + 100_000]
    workers = get_kv_cache_configs(config, [specs] * 4, budgets)
    for worker, budget in zip(workers, budgets):
        plan = worker.long_context_plan
        assert plan.gpu_budget_bytes == min(budgets)
        assert plan.gpu_budget_bytes <= budget
        assert (
            0
            <= plan.gpu_remaining_bytes
            < (
                plan.geometry.hot_bytes_for_groups(plan.hot_pool_groups + 1, 8)
                - plan.hot_pool_bytes
            )
        )
        assert worker.block_segments == workers[0].block_segments
    scheduler = generate_scheduler_kv_cache_config(workers)
    assert scheduler.long_context_plan == workers[1].long_context_plan
    assert len({w.long_context_plan for w in workers}) == 1
    config.cache_config.num_gpu_blocks_override = 700
    overridden = get_kv_cache_configs(config, [specs] * 4, budgets)
    assert all(
        w.long_context_plan.gpu_budget_bytes == 700 * POOL_BLOCK_BYTES
        for w in overridden
    )
    assert len({w.long_context_plan for w in overridden}) == 1


def test_startup_splits_the_pool_into_state_and_history_segments():
    # With the main K/V still on the GPU a history block costs index + main
    # K/V (6.39 MB/GPU), so 8 x 1Mi needs GDN state replay (s = 0) in 6.05 GiB.
    off = kv_cache_configs(LongContextKVConfig(), speculative_blocks=0)
    on = kv_cache_configs(
        LongContextKVConfig(
            enabled=True, host_pool_gib=13.0, hot_pool_gib=1.0, main_kv_placement="gpu"
        ),
        speculative_blocks=0,
    )
    for disabled, enabled in zip(off, on):
        assert disabled.long_context_plan is None
        assert disabled.block_segments is None
        assert disabled.num_blocks == GPU_BUDGET // POOL_BLOCK_BYTES
        plan = enabled.long_context_plan
        assert plan is not None and plan.main_kv_on_gpu
        assert enabled.block_segments.num_blocks == plan.segment_num_blocks()
        assert enabled.num_blocks == plan.gpu_history_blocks + 1
        assert disabled.kv_cache_groups == enabled.kv_cache_groups
        assert sum(t.size for t in enabled.kv_cache_tensors) == (
            plan.gpu_state_bytes + plan.gpu_history_bytes
        )
    plan = on[0].long_context_plan
    assert plan.state_blocks == 8 * 29
    assert plan.gpu_budget_bytes == GPU_BUDGET
    assert plan.host_history_blocks == 13 * GiB // (WORKERS * MAIN_KV_BLOCK_BYTES)
    assert plan.geometry.max_model_len == MI
    assert plan.history_bytes_per_block == POOL_BLOCK_BYTES
    assert plan.gpu_history_blocks == (
        (GPU_BUDGET - 233 * STATE_BLOCK_BYTES) // POOL_BLOCK_BYTES - 1
    )
    # Without replay the 1Mi request does not fit next to 8 requests' states.
    with pytest.raises(ValueError, match="largest max_model_len is 930,176"):
        kv_cache_configs(
            LongContextKVConfig(
                enabled=True,
                host_pool_gib=13.0,
                hot_pool_gib=1.0,
                main_kv_placement="gpu",
            )
        )


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


@pytest.mark.parametrize("connector", ["SimpleCPUOffloadConnector", "MultiConnector"])
@pytest.mark.parametrize("placement", ["host", "gpu"])
def test_split_pool_rejects_connectors_with_ambiguous_failure_ids(connector, placement):
    config = production_config(
        long_context=LongContextKVConfig(
            enabled=True, host_pool_gib=200, hot_pool_gib=1, main_kv_placement=placement
        )
    )
    groups = get_kv_cache_groups(config, production_specs())
    config.kv_transfer_config = SimpleNamespace(kv_connector=connector)
    with pytest.raises(ValueError, match="requires OffloadingConnector"):
        get_long_context_geometry(config, groups)
    config.kv_transfer_config.kv_connector = "OffloadingConnector"
    assert get_long_context_geometry(config, groups).group_domains[0] == (
        GPU_HISTORY,
        HOST,
    )


def test_hot_layout_refuses_unsupported_dcp():
    config = production_config()
    groups = get_kv_cache_groups(config, production_specs())
    config.parallel_config.decode_context_parallel_size = 4
    with pytest.raises(ValueError, match="DCP1 or DCP2"):
        get_long_context_geometry(config, groups)


def test_split_pool_rejects_custom_connector_with_builtin_name():
    config = production_config()
    groups = get_kv_cache_groups(config, production_specs())
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector", kv_connector_module_path="custom.connector"
    )
    with pytest.raises(ValueError, match="built-in implementation"):
        get_long_context_geometry(config, groups)
