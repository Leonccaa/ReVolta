# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU store allocations versus the engine and A/B tool's hot budgets."""

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.long_context_kv_utils import production_config, production_specs
from tools import long_context_kv_ab as ab
from vllm.config.kernel import LongContextKVConfig
from vllm.models.qwen4_exp.nvidia.host_kv.budget import (
    HISTORY_METADATA,
    history_metadata_bytes,
)
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
)
from vllm.v1.core.long_context_kv import (
    GiB,
    LongContextGeometry,
    plan_long_context_capacity,
    validate_long_context_plan,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.host.region import HostRegionLayout, host_region_layout

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def sparse_zero_payloads(tmp_path, monkeypatch):
    """Keep full production single-pass shapes in the 8-GiB CPU runner.

    Zero tensors over 2 GiB (large host pages / single-pass hot payloads)
    use private mappings of sparse files.
    They are real, initialized CPU tensor storages, with the requested byte
    lengths; only untouched physical zero pages are shared with the OS.
    Neither FakeTensor nor reduced dimensions replace the real store.
    """
    zeros = torch.zeros
    paths: list[Path] = []

    def sparse_zeros(*size, **kwargs):
        dims = (
            size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
        )
        numel = math.prod(dims)
        dtype = kwargs.get("dtype", torch.float32)
        nbytes = numel * dtype.itemsize
        if nbytes < 2 * 1024**3:
            return zeros(*size, **kwargs)
        assert torch.device(kwargs["device"]).type == "cpu"
        path = tmp_path / f"zero-{len(paths)}"
        with path.open("wb") as f:
            f.truncate(nbytes)
        paths.append(path)
        return torch.from_file(str(path), shared=False, size=numel, dtype=dtype).view(
            dims
        )

    monkeypatch.setattr(torch, "zeros", sparse_zeros)

    def sparse_region(layout, rank):
        # Preserve the production shared row geometry without populating a
        # 24-112 GiB mapping in this bounded CPU runner.
        path = tmp_path / f"region-{len(paths)}"
        paths.append(path)
        size = layout.num_pages * layout.worker_bytes * layout.num_workers
        region = SharedOffloadRegion.__new__(SharedOffloadRegion)
        region._base = torch.from_file(
            str(path), shared=True, size=size, dtype=torch.int8
        )
        region._views = []
        region._row_stride = layout.worker_bytes * layout.num_workers
        region._worker_offset = rank * layout.worker_bytes
        region._worker_area_end = (rank + 1) * layout.worker_bytes
        region.num_blocks, region.rank = layout.num_pages, rank
        region.total_size_bytes = size
        region.is_pinned = region._creator = region._scheduler_unlink = False
        region.mmap_obj = region.fd = None
        return region

    monkeypatch.setattr(HostRegionLayout, "open", sparse_region)


def _geometry(model, dcp, max_write_rows):
    if model == "production":
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_groups,
            get_long_context_geometry,
        )

        config = production_config()
        geo = get_long_context_geometry(
            config, get_kv_cache_groups(config, production_specs())
        )
        return (
            dataclasses.replace(
                geo,
                dcp_world_size=dcp,
                main_kv_bytes_per_block=geo.main_kv_bytes_per_block * 2 // dcp,
                hot_max_write_rows=max_write_rows,
            ),
            torch.uint8,
            1,
        )
    # Four layers, one QSA layer: recorded FP16 1.9B substitute, block 1568.
    # The DCP2 variant also uses TP2: one of its two global KV heads per rank
    # and half the block's tokens (the TP1/DCP1 receipt itself is unchanged).
    return (
        LongContextGeometry(
            block_tokens=1568,
            num_workers=dcp,
            main_kv_bytes_per_block=3_211_264 // (dcp * dcp),
            index_bytes_per_block=100_352,
            state_bytes_per_block=3_211_264,
            unified_bytes_per_block=3_311_616,
            state_blocks_per_request=8,
            ring_blocks_per_request=1,
            max_model_len=8192 if max_write_rows == 2048 else 4096,
            hot_group_tokens=4,
            hot_row_groups=513,
            decode_rows_per_request=1,
            group_domains=((1, 2), (1,), (0,), (0,), (0,), (0,)),
            dcp_world_size=dcp,
            hot_layers=1,
            hot_max_write_rows=max_write_rows,
            hot_query_heads=24 // dcp,
            hot_head_dim=256,
            hot_kv_heads=2 // dcp,
        ),
        torch.float16,
        2 // dcp,
    )


@pytest.mark.parametrize(
    "model,write_rows,seqs",
    [("production", 8192, 8), ("standin", 2048, 4), ("standin", 1024, 2)],
)
@pytest.mark.parametrize("dcp", [1, 2])
@pytest.mark.parametrize("budget", [0.5, 1.0, "single-pass"])
def test_real_store_fits_the_total_budget(
    model, write_rows, seqs, dcp, budget, sparse_zero_payloads
):
    geo, dtype, heads = _geometry(model, dcp, write_rows)
    receipt = dataclasses.asdict(geo)
    gib = (
        ab.hot_pool_gib_for_rows(receipt, min(write_rows, geo.hot_read_rows), seqs)
        if budget == "single-pass"
        else budget
    )
    host_blocks = max(
        8, geo.history_blocks(geo.max_model_len) + geo.ring_blocks_per_request + 1
    )
    plan = plan_long_context_capacity(
        LongContextKVConfig(
            enabled=True,
            host_pool_tokens=host_blocks * geo.block_tokens,
            hot_pool_gib=gib,
        ),
        geo,
        512 * GiB,  # CPU arithmetic, not a claim of this GPU capacity.
        seqs,
    )
    assert not plan.startup_problems(), plan.startup_problems()
    shape = HostHistoryShape(
        num_blocks=plan.gpu_history_blocks + 1,
        kernel_blocks_per_block=1,
        kernel_block_tokens=geo.local_block_tokens,
        num_layers=geo.hot_layers,
        kv_heads=heads,
        head_dim=256,
        dtype=dtype,
        hot_slots=plan.hot_pool_groups,
        max_requests=seqs,
        selection_width=2051,
        max_write_rows=write_rows,
        dcp_world_size=dcp,
        dcp_rank=dcp - 1,
    )
    store = HostHistoryStore(
        shape, [f"l{i}" for i in range(geo.hot_layers)], device="cpu"
    )
    expected = shape.fixed_device_memory_bytes()
    actual = store.fixed_device_memory_bytes()
    assert actual == expected  # Every tensor, not just the hot payload.
    assert all(
        getattr(store, name).untyped_storage().nbytes() == size
        for name, size in actual.items()
    )
    fixed = sum(actual.values())
    assert fixed == plan.gpu_hot_pool_bytes <= int(gib * GiB)
    # Unused user budget is less than the next whole slot's cost (including
    # an indices row / victim tile jump when crossing their boundaries).
    next_cost = geo.hot_bytes_for_groups(plan.hot_pool_groups + 1, seqs)
    assert next_cost > int(gib * GiB)
    assert int(gib * GiB) - fixed < next_cost - fixed
    per_block = history_metadata_bytes(geo.local_block_tokens, geo.hot_layers)
    assert set(per_block) == HISTORY_METADATA
    assert sum(per_block.values()) == geo.host_metadata_bytes_per_block
    assert {
        name: getattr(store, name).numel() * getattr(store, name).element_size()
        for name in HISTORY_METADATA
    } == {name: size * shape.num_blocks for name, size in per_block.items()}
    assert (
        store.host.numel() * store.host.element_size()
        == shape.num_blocks * geo.main_kv_bytes_per_block
    )
    if budget == "single-pass":
        assert shape.rows_per_pass == min(write_rows, geo.hot_read_rows)


def test_store_budget_handles_split_kernel_blocks(sparse_zero_payloads):
    geo, dtype, heads = _geometry("standin", 2, 2048)
    slots = geo.hot_groups_for_bytes(GiB, 4)
    shape = HostHistoryShape(
        num_blocks=8,
        kernel_blocks_per_block=2,
        kernel_block_tokens=392,
        num_layers=1,
        kv_heads=heads,
        head_dim=256,
        dtype=dtype,
        hot_slots=slots,
        max_requests=4,
        selection_width=2051,
        max_write_rows=2048,
        dcp_world_size=2,
    )
    store = HostHistoryStore(shape, ["l0"], device="cpu")
    assert sum(store.fixed_device_memory_bytes().values()) == geo.hot_bytes_for_groups(
        slots, 4
    )
    assert store.fixed_device_memory_bytes() == shape.fixed_device_memory_bytes()


@pytest.mark.parametrize(
    "model,host_gib",
    [
        ("production", 96.13),
        ("production", 24.0),
        ("production", 112.0),
        ("standin", 0.5),
        ("standin", 0.125),
    ],
)
def test_auto_real_store_uses_all_remaining_bytes(
    model, host_gib, sparse_zero_payloads, monkeypatch
):
    # Static hot-pool arm: the shared region bills its slots per history ID.
    monkeypatch.setattr("vllm.v1.core.long_context_kv.SHARED_HOT_REGION", False)
    settings = LongContextKVConfig(
        enabled=True,
        host_pool_gib=host_gib,
        hot_pool_gib="auto",
        prefill_staging_gib=0.125,
    )
    if model == "production":
        from vllm.models.qwen4_exp.nvidia.host_kv.engine import build_host_history_store
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        config = production_config(long_context=settings)
        config.instance_id = "cpu-budget-fixture"
        config.model_config.original_max_model_len = config.model_config.max_model_len
        specs = production_specs(speculative_blocks=0)  # State replay enabled.
        for name, spec in list(specs.items()):
            if name.endswith(".compressed"):
                specs[name] = dataclasses.replace(
                    spec, block_size=spec.block_size // 2, dcp_sharded=True
                )  # index sharding index sharding, full global history span preserved.
        budget = int(6.05 * GiB)
        configs = get_kv_cache_configs(config, [specs] * 4, [budget] * 4)
        cfg = configs[0]
        plan = cfg.long_context_plan
        geo = plan.geometry
        assert geo.index_bytes_per_block == 638_976
        assert geo.state_blocks_per_request == 29
        assert geo.decode_rows_per_request == 8
        layers = {
            name: SimpleNamespace(
                qsa_dcp_sharded=True,
                indexer=SimpleNamespace(output_width=2051),
                host_history=None,
            )
            for name in cfg.host_kv_layers
        }
        monkeypatch.setattr(
            "vllm.distributed.get_dcp_group",
            lambda: SimpleNamespace(world_size=2, rank_in_group=1),
        )
        monkeypatch.setattr(
            "vllm.distributed.get_world_group", lambda: SimpleNamespace(rank_in_group=1)
        )
        config.parallel_config.cp_kv_cache_interleave_size = 1
        store = build_host_history_store(
            cfg,
            config,
            layers,
            [geo.local_block_tokens] * len(cfg.kv_cache_groups),
            torch.device("cpu"),
        )
        assert all(
            layer.host_history is store.layer(name) for name, layer in layers.items()
        )
        # Real worker tensor descriptions plus the store's per-ID metadata.
        assert sum(t.size for t in cfg.kv_cache_tensors) == (
            plan.gpu_state_bytes
            + plan.gpu_history_bytes
            - cfg.num_blocks * geo.host_metadata_bytes_per_block
        )
    else:
        geo, dtype, heads = _geometry("standin", 1, 2048)
        budget = 3 * GiB
        plan = plan_long_context_capacity(settings, geo, budget, 4)
        store = HostHistoryStore(
            HostHistoryShape(
                num_blocks=plan.gpu_history_blocks + 1,
                kernel_blocks_per_block=2,
                kernel_block_tokens=geo.local_block_tokens // 2,
                num_layers=geo.hot_layers,
                kv_heads=heads,
                head_dim=256,
                dtype=dtype,
                hot_slots=plan.hot_pool_groups,
                max_requests=4,
                selection_width=2051,
                max_write_rows=2048,
            ),
            ["l0"],
            device="cpu",
        )
    validate_long_context_plan(plan)
    shape = store.shape
    assert plan.gpu_history_blocks == min(
        plan.max_num_seqs
        * (geo.history_blocks(geo.max_model_len) + geo.ring_blocks_per_request),
        plan.host_history_blocks - 1,
    )
    actual = store.fixed_device_memory_bytes()
    assert actual == shape.fixed_device_memory_bytes()
    assert all(
        getattr(store, name).untyped_storage().nbytes() == size
        for name, size in actual.items()
    )
    assert sum(actual.values()) == plan.gpu_hot_pool_bytes
    assert {
        name: getattr(store, name).untyped_storage().nbytes()
        for name in HISTORY_METADATA
    } == {
        name: size * shape.num_blocks
        for name, size in history_metadata_bytes(
            geo.local_block_tokens, geo.hot_layers
        ).items()
    }
    if store.region is None:
        assert (
            store.host.untyped_storage().nbytes()
            == shape.num_blocks * geo.main_kv_bytes_per_block
        )
    else:
        layout = host_region_layout(cfg, config.instance_id)
        assert (
            store.host.untyped_storage().nbytes()
            == layout.num_pages * geo.host_bytes_per_block
        )
        assert store.host.stride(0) * shape.dtype.itemsize == geo.host_bytes_per_block
        assert (
            store.host.numel() * shape.dtype.itemsize
            == layout.num_pages * geo.main_kv_bytes_per_block
        )
    assert shape.num_blocks * geo.host_bytes_per_block <= plan.host_pool_bytes
    used = (
        plan.gpu_state_bytes
        + plan.gpu_history_bytes
        + sum(actual.values())
        + plan.staging_bytes
    )
    assert 0 <= plan.gpu_remaining_bytes == budget - used
    assert plan.gpu_remaining_bytes < (
        geo.hot_bytes_for_groups(shape.hot_slots + 1, plan.max_num_seqs)
        - sum(actual.values())
    )
    assert shape.hot_slots >= plan.max_num_seqs * (
        geo.decode_rows_per_request * geo.hot_row_groups + 9
    )
