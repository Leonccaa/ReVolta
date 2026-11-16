# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context mode with the main QSA K/V in the host history store.

``main_kv_placement="host"`` (the default once the mode is enabled): the GPU
history segment keeps only the compressed index and ring tensors, every
history block ID owns one host page per QSA layer on each worker, and the
history segment can never have more IDs than the host pool has pages. These
tests use the deployed CSA/GDN/PLE/DFlash geometry (TP4/DCP2, 1,664-token
blocks).
"""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import LongContextKVConfig
from vllm.v1.attention.kv_cache_view import packed_kv_members
from vllm.v1.core.long_context_kv import HISTORY_SEGMENT, STATE_SEGMENT, GiB
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

from .long_context_kv_utils import (
    INDEX_BLOCK_BYTES,
    MAIN_KV_BLOCK_BYTES,
    SPAN,
    production_config,
    production_specs,
)
from .test_long_context_admission import build as build_scheduler
from .test_long_context_admission import make_requests, run
from .test_long_context_split_pool import GPU_BUDGET, WORKERS, _layer_specs

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _static_hot_pool(monkeypatch):
    # These page ownership regressions pin the static hot-pool arm (host-page-capped
    # history IDs). The shared hot region shared region has its own scheduler tests in
    # test_history_shared_region.py.
    monkeypatch.setattr("vllm.v1.core.long_context_kv.SHARED_HOT_REGION", False)


LOCAL = SPAN // 2
HOST_META_BYTES = 14 * LOCAL + 8  # 12 hot maps + wanted + ready, generation
HOST_PAGE_BYTES = MAIN_KV_BLOCK_BYTES  # one block, 12 layers, one worker
MAIN_KV = [f"model.layers.{layer}.self_attn" for layer in range(3, 48, 4)]


def host_settings(**kw) -> LongContextKVConfig:
    if "host_pool_tokens" not in kw:
        kw.setdefault("host_pool_gib", 200.0)
    kw.setdefault("hot_pool_gib", 0.6)
    return LongContextKVConfig(enabled=True, **kw)


def configs(lc, *, max_model_len=512 * 1024, memory=GPU_BUDGET, max_num_seqs=8):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    config = production_config(
        max_model_len=max_model_len, long_context=lc, max_num_seqs=max_num_seqs
    )
    config.model_config.original_max_model_len = max_model_len
    if lc.main_kv_placement == "host":
        config.kv_transfer_config = SimpleNamespace(kv_connector="OffloadingConnector")
    specs = production_specs()
    return get_kv_cache_configs(config, [specs] * WORKERS, [memory] * WORKERS)


def test_default_placement_is_host():
    assert LongContextKVConfig().main_kv_placement == "host"


def test_history_segment_holds_index_and_ring_only():
    (cfg, *rest) = configs(host_settings(prefill_staging_gib=0.25))
    plan = cfg.long_context_plan
    assert plan is not None and not plan.main_kv_on_gpu
    assert sorted(cfg.host_kv_layers) == sorted(MAIN_KV)
    num_state, num_history = plan.segment_num_blocks()
    assert cfg.block_segments.num_blocks == (num_state, num_history)
    assert plan.history_bytes_per_block == INDEX_BLOCK_BYTES + HOST_META_BYTES

    # No tensor carries a main K/V layer; every other layer is in exactly one
    # tensor of its own segment's size.
    specs = _layer_specs(cfg)
    owners = Counter(name for t in cfg.kv_cache_tensors for name in t.shared_by)
    assert set(owners) == set(specs) - set(MAIN_KV)
    assert set(owners.values()) == {1}
    packed = packed_kv_members(cfg)
    for tensor in cfg.kv_cache_tensors:
        assert tensor.packed_members is None  # only main K/V owners packed
        for name in tensor.shared_by:
            group_id, spec = specs[name]
            members = packed.get(name, (0, 1))[1]
            blocks = tensor.size // (spec.page_size_bytes * members)
            assert blocks == cfg.group_num_blocks(group_id) == tensor.num_blocks
    history = sum(t.size for t in cfg.kv_cache_tensors if t.num_blocks == num_history)
    state = sum(t.size for t in cfg.kv_cache_tensors if t.num_blocks == num_state)
    assert history == num_history * INDEX_BLOCK_BYTES
    assert state == plan.gpu_state_bytes
    # GPU total: tensors + store metadata + hot pool + staging fit the budget,
    # and the history segment took what is left (budget-bound here).
    used = (
        state
        + history
        + num_history * HOST_META_BYTES
        + plan.gpu_hot_pool_bytes
        + int(0.25 * GiB)
    )
    assert plan.gpu_hot_pool_bytes == plan.geometry.hot_bytes_for_groups(
        plan.hot_pool_groups, plan.max_num_seqs
    )
    assert used <= GPU_BUDGET < used + plan.history_bytes_per_block
    assert plan.gpu_history_blocks == plan.gpu_history_budget_blocks
    # Host pages: one per history ID on every worker, within the pool.
    assert num_history * HOST_PAGE_BYTES * WORKERS <= plan.host_pool_bytes
    for other in rest:
        assert other.host_kv_layers == cfg.host_kv_layers
        assert other.block_segments == cfg.block_segments


def test_history_ids_never_outnumber_host_pages():
    # 20 x 1,664 history tokens: 20 pages, one backs the null block.
    (cfg, *_) = configs(
        host_settings(host_pool_tokens=20 * SPAN), max_model_len=10 * SPAN
    )
    plan = cfg.long_context_plan
    assert plan.host_history_blocks == 20
    assert plan.gpu_history_budget_blocks > 20
    assert plan.gpu_history_blocks == 19
    assert cfg.num_blocks == cfg.block_segments.num_blocks[HISTORY_SEGMENT] == 20
    # One request needs 10 history + 1 ring IDs and the null page.
    assert plan.max_concurrency(10 * SPAN)["GPU history segment"] == 1
    with pytest.raises(ValueError, match="host_pool_tokens >= 19968"):
        configs(host_settings(host_pool_tokens=11 * SPAN), max_model_len=10 * SPAN)


def test_hot_pool_must_hold_one_row_and_every_request_tail():
    (cfg, *_) = configs(host_settings(hot_pool_gib=0.01), max_model_len=1000)
    plan = cfg.long_context_plan
    assert plan.hot_pool_groups >= plan.hot_pool_min_groups
    assert plan.hot_pool_groups >= plan.geometry.hot_reserve_groups(1000)
    assert plan.hot_pool_groups > plan.hot_pool_min_groups


def test_host_placement_accepts_builtin_connector_and_refuses_int8():
    config = production_config(long_context=host_settings(hot_pool_gib=1.0))
    config.model_config.original_max_model_len = config.model_config.max_model_len
    config.kv_transfer_config = SimpleNamespace(kv_connector="OffloadingConnector")
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    (host_cfg,) = get_kv_cache_configs(config, [production_specs()], [GPU_BUDGET])
    assert host_cfg.host_kv_layers == MAIN_KV
    specs = production_specs()
    for name in MAIN_KV:
        specs[name] = FullAttentionSpec(
            block_size=LOCAL,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.int8,
            kv_quant_mode=KVQuantMode.INT8_PLANAR,
            dcp_sharded=True,
        )
    with pytest.raises(ValueError, match="E4M3 or FP16"):
        get_kv_cache_configs(config, [specs], [GPU_BUDGET])
    # The GPU placement keeps accepting both (as before this change).
    gpu = production_config(
        max_model_len=512 * 1024,
        long_context=host_settings(main_kv_placement="gpu"),
    )
    gpu.model_config.original_max_model_len = gpu.model_config.max_model_len
    gpu.kv_transfer_config = None
    (cfg,) = get_kv_cache_configs(gpu, [production_specs()], [GPU_BUDGET])
    assert cfg.host_kv_layers == []


def test_worker_allocates_no_tensor_for_host_layers():
    from vllm.v1.worker.gpu.attn_utils import _allocate_kv_cache

    (cfg, *_) = configs(
        host_settings(host_pool_tokens=8 * SPAN, state_blocks=71),
        max_model_len=4 * SPAN,
        max_num_seqs=1,
    )
    raw = _allocate_kv_cache(cfg, {}, torch.device("cpu"))
    assert set(raw).isdisjoint(cfg.host_kv_layers)
    specs = _layer_specs(cfg)
    assert set(raw) | set(cfg.host_kv_layers) == set(specs)
    storages: dict[int, set[int]] = {STATE_SEGMENT: set(), HISTORY_SEGMENT: set()}
    for name, tensor in raw.items():
        group_id, _ = specs[name]
        segment = cfg.block_segments.group_segments[group_id]
        storages[segment].add(tensor.untyped_storage().data_ptr())
    assert not storages[STATE_SEGMENT] & storages[HISTORY_SEGMENT]
    distinct = {id(t): t for t in raw.values()}.values()
    assert sum(t.untyped_storage().nbytes() for t in distinct) == sum(
        t.size for t in cfg.kv_cache_tensors
    )


def test_scheduler_queues_when_host_pages_run_out():
    # Host placement: 12 pages = null + one 10-block request + its ring. The
    # history pool itself is the limit (no ledger refusal needed), and the
    # workload waits or preempts, then finishes.
    from .test_long_context_admission import MAX_MODEL_LEN
    from .test_long_context_admission import settings as admission_settings

    lc = admission_settings(host_pool_tokens=12 * SPAN, main_kv_placement="host")
    scheduler = build_scheduler(lc)
    manager = scheduler.kv_cache_manager
    assert MAX_MODEL_LEN == 10 * SPAN
    assert manager.admission_ledger.plan.gpu_history_blocks == 11
    history_pool = manager.coordinator.block_pools[HISTORY_SEGMENT]
    assert history_pool.num_gpu_blocks == 12
    requests = make_requests(4, 4 * SPAN - 100, 4)
    # Admission checks each step's allocation against what is free now (a
    # chunked prompt may be admitted and preempted later); run() checks the
    # ledger every step and that every request finishes.
    run(scheduler, requests)
    grow = make_requests(3, SPAN - 64, 2 * SPAN, tag="g")
    trace, _ = run(build_scheduler(lc), grow)
    assert any(preempted for _, _, preempted in trace)


def test_worker_binds_one_host_store_to_the_qsa_layers(monkeypatch):
    import os

    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("The CPU host store needs TRITON_INTERPRET=1")
    from vllm.models.qwen4_exp.nvidia.host_kv.engine import build_host_history_store
    from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryStore

    # This production-size geometry fixture uses ordinary CPU backing. The
    # shared row layout and registration are tested with bounded real mmaps.
    def cpu_store(*args, **kwargs):
        kwargs.pop("region_layout")
        kwargs.pop("rank")
        return HostHistoryStore(*args, **kwargs)

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.host_kv.engine.HostHistoryStore", cpu_store
    )
    monkeypatch.setattr(
        "vllm.distributed.get_world_group", lambda: SimpleNamespace(rank_in_group=0)
    )

    (cfg, *_) = configs(
        host_settings(host_pool_tokens=8 * SPAN, hot_pool_gib=0.2, state_blocks=71),
        max_model_len=4 * SPAN,
        max_num_seqs=1,
    )
    plan = cfg.long_context_plan
    layers = {
        name: SimpleNamespace(
            qsa_dcp_sharded=False,
            indexer=SimpleNamespace(output_width=2051),
            host_history=None,
        )
        for name in cfg.host_kv_layers
    }
    vllm_config = production_config(max_model_len=4 * SPAN, max_num_seqs=1)
    vllm_config.instance_id = "cpu-binding-fixture"
    store = build_host_history_store(
        cfg,
        vllm_config,
        layers,
        [LOCAL] * len(cfg.kv_cache_groups),
        torch.device("cpu"),
    )
    assert store is not None
    shape = store.shape
    assert shape.num_blocks == cfg.num_blocks == plan.gpu_history_blocks + 1 == 8
    assert (shape.kernel_block_tokens, shape.kernel_blocks_per_block) == (LOCAL, 1)
    assert (shape.num_layers, shape.kv_heads, shape.head_dim) == (12, 1, 256)
    assert shape.dtype == torch.uint8 and shape.hot_slots == plan.hot_pool_groups
    assert (
        shape.rows_per_pass
        == min(shape.read_rows, (plan.hot_pool_groups - 9) // 513)
        >= 1
    )
    # Host bytes: one page per history ID and layer, within the host pool.
    assert store.host.numel() == 8 * HOST_PAGE_BYTES
    assert store.host_page(3).numel() == HOST_PAGE_BYTES
    # GPU metadata per history block matches the plan's charge.
    meta = (
        store.group_to_slot.numel() * 4
        + store.wanted.numel() * 4
        + store.ready.numel()
        + store.generation.numel() * 4
        + store.page_of.numel() * 4
    )
    assert meta == shape.num_blocks * HOST_META_BYTES
    assert store.hot.numel() < plan.gpu_hot_pool_bytes
    assert store.fixed_device_memory_bytes() == shape.fixed_device_memory_bytes()
    assert sum(store.fixed_device_memory_bytes().values()) == plan.gpu_hot_pool_bytes
    for name, layer in layers.items():
        assert layer.host_history is store.layer(name)
    # No host layers: nothing to build.
    (gpu_cfg, *_) = configs(
        host_settings(main_kv_placement="gpu"), max_model_len=4 * SPAN
    )
    assert build_host_history_store(gpu_cfg, vllm_config, {}, [], "cpu") is None


@pytest.mark.parametrize("full_isl", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("matched", [0, 5 * SPAN, 9 * SPAN])
@pytest.mark.parametrize("slack", [1, 8])
def test_host_placement_preserves_r1_net_reservation(
    monkeypatch, full_isl, async_scheduling, matched, slack
):
    from . import test_long_context_async_admission as regression

    original_settings = regression.settings

    def host_admission_settings(**kwargs):
        # The same ten history pages, with backing for ring and null IDs.
        kwargs["host_pool_tokens"] += 2 * SPAN
        return original_settings(main_kv_placement="host", **kwargs)

    monkeypatch.setattr(regression, "settings", host_admission_settings)
    regression.test_dflash_single_request_fits_without_double_reservation(
        True, full_isl, async_scheduling, matched, slack
    )


@pytest.mark.parametrize("full_isl", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("initial", [1000, 2000])
def test_host_placement_preserves_r2_streaming_progress(
    monkeypatch, full_isl, async_scheduling, initial
):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from . import test_long_context_streaming_admission as regression
    from .test_long_context_admission import MAX_MODEL_LEN, settings

    def host_scheduler(enabled, full_isl, async_scheduling):
        assert enabled
        probe = build_scheduler(settings(main_kv_placement="host"))
        geometry = probe.kv_cache_manager.admission_ledger.plan.geometry
        groups = max(
            geometry.hot_reserve_groups(MAX_MODEL_LEN),
            geometry.hot_row_groups + 8 * geometry.hot_tail_groups,
        )
        cost = geometry.hot_bytes_for_groups(groups, 8)
        scheduler = build_scheduler(
            settings(main_kv_placement="host", hot_pool_gib=cost / GiB + 1e-9),
            reserve_full_isl=full_isl,
        )
        if async_scheduling:
            scheduler.vllm_config.scheduler_config.async_scheduling = True
            scheduler = AsyncScheduler(
                vllm_config=scheduler.vllm_config,
                kv_cache_config=scheduler.kv_cache_config,
                structured_output_manager=scheduler.structured_output_manager,
                block_size=16,
                hash_block_size=16,
            )
            scheduler.connector = None  # Scheduler-only fixture has no I/O worker.
            table = scheduler.host_history_pool.host_pages
            if table.reservation_ledger is not None:
                table.committed_reserve = table.reservation_ledger._external
                table.reservation_ledger = None
                table.value_policy = None
        return scheduler

    monkeypatch.setattr(regression, "_scheduler", host_scheduler)
    regression.test_two_streams_grow_and_finish(
        True, full_isl, async_scheduling, initial
    )
