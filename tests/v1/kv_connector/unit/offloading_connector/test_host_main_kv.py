# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host history transfer registration, backing budgets and publication on CPU."""

import ctypes
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from tests.v1.core.long_context_kv_utils import (
    SPAN,
    production_config,
    production_specs,
)
from tests.v1.core.test_long_context_host_main_kv import host_settings
from tests.v1.core.test_long_context_split_pool import _layer_specs
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("tiering", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_host_transfer_registration_and_real_cpu_budget(
    monkeypatch, request, tiering, grouped, padded
):
    config = production_config(
        max_model_len=4 * SPAN,
        max_num_seqs=1,
        long_context=host_settings(host_pool_tokens=8 * SPAN, hot_pool_gib=0.2),
    )
    config.model_config.original_max_model_len = 4 * SPAN
    config.parallel_config.world_size = 4
    config.parallel_config.rank = 3
    config.instance_id = uuid4().hex
    config.cache_config.block_size = SPAN
    config.cache_config.hash_block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_cache_retention_interval = 0
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_extra_config={"cpu_bytes_to_use": 4 << 30},
    )
    # Preserve both legacy input geometries (with/without drafter). Long
    # context now uses the same unified backing instead of grouped pools.
    (cfg,) = get_kv_cache_configs(
        config, [production_specs(drafter=not grouped)], [int(6.05 * (1 << 30))]
    )
    cfg.kv_cache_tensors = [
        replace(t, size=t.size // t.num_blocks * 2, num_blocks=2)
        for t in cfg.kv_cache_tensors
    ]
    if padded:
        # G0 padding belongs only to the GPU index arena. Both backing budget
        # paths must exclude its entire allocation, including ring aliases.
        cfg.kv_cache_tensors = [
            replace(
                t,
                size=t.size + 4096 * t.num_blocks,
                page_stride_bytes=t.page_stride_bytes + 4096,
            )
            if t.page_stride_bytes is not None
            else t
            for t in cfg.kv_cache_tensors
        ]
    cfg.num_blocks = 2
    from vllm.v1.kv_offload.host.region import HostRegionOwner, host_region_layout

    layout = replace(host_region_layout(cfg, config.instance_id), num_pages=2)
    cfg._host_region_layout = layout
    request.addfinalizer(HostRegionOwner(layout).cleanup)
    store = HostHistoryStore(
        HostHistoryShape(
            num_blocks=2,
            kernel_blocks_per_block=1,
            kernel_block_tokens=SPAN // 2,
            num_layers=12,
            kv_heads=1,
            head_dim=256,
            dtype=torch.uint8,
            hot_slots=522,
            max_requests=1,
            selection_width=2051,
            max_write_rows=16,
        ),
        cfg.host_kv_layers,
        device="cpu",
        region_layout=layout,
        rank=3,
    )
    request.addfinalizer(store.close)
    config.compilation_config = SimpleNamespace(
        static_forward_context={
            name: SimpleNamespace(host_history=store.layer(name))
            for name in cfg.host_kv_layers
        }
    )
    request.addfinalizer(config.compilation_config.static_forward_context.clear)
    if tiering and grouped:
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.nnodes = 1
        config.attention_config = SimpleNamespace(backend="FLASH_ATTN")
        config.model_config.revision = None
        config.compute_hash = lambda: "cpu-host-fixture"
    cls = TieringOffloadingSpec if tiering else CPUOffloadingSpec
    spec = cls(config, cfg)
    assert not spec.partition_by_group
    assert spec.unified_host and spec.num_blocks == 2
    specs = _layer_specs(cfg)
    caches = {name: torch.empty(0) for name in cfg.host_kv_layers}
    for tensor in cfg.kv_cache_tensors:
        raw = torch.empty(tensor.size, dtype=torch.uint8)
        for name in tensor.shared_by:
            caches[name] = [raw] if isinstance(specs[name][1], MambaSpec) else raw
    worker = OffloadingConnectorWorker(spec)
    captured: list[CanonicalKVCaches] = []
    monkeypatch.setattr(worker, "_register_handlers", captured.append)
    worker.register_kv_caches(caches)
    (canonical,) = captured
    assert not canonical.group_data_refs[cfg.host_kv_group_id]
    assert all(t.tensor.data_ptr() != store.host.data_ptr() for t in canonical.tensors)
    excluded = {name for name in caches if name not in cfg.connector_transfer_layers()}
    excluded_storage = {
        caches[name].untyped_storage()._cdata
        for name in excluded
        if isinstance(caches[name], torch.Tensor)
    }
    assert all(
        t.tensor.untyped_storage()._cdata not in excluded_storage
        for t in canonical.tensors
    )
    assert cfg.host_kv_group_id not in spec.cpu_group_page_sizes
    required = (
        sum(t.page_size_bytes for t in canonical.tensors) * spec.block_size_factor
        if not spec.partition_by_group
        else sum(spec.cpu_group_page_sizes.values())
    )
    config.kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"] = (
        required * 4 + 123
    )
    spec = cls(config, cfg)
    monkeypatch.setattr(gpu_worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(
        gpu_worker,
        "SingleDirectionOffloadingHandler",
        lambda **kw: SimpleNamespace(**kw),
    )
    # Worker registration supplies the owner mapping, never an independent
    # allocation from cpu_bytes_to_use. The second spec tests the same ABI.
    spec.unified_region = store.region
    handlers = spec.create_handlers(canonical)
    request.addfinalizer(handlers.gpu_to_cpu_handler.cpu_tensors.clear)
    request.addfinalizer(handlers.cpu_to_gpu_handler.cpu_tensors.clear)
    cpu = handlers.gpu_to_cpu_handler.cpu_tensors
    assert spec.cpu_page_size_per_worker == required
    assert handlers._mmap_region is None
    assert spec.unified_region is store.region
    assert sum(t.numel() for t in cpu) * 4 <= store.region.total_size_bytes
    assert spec.unified_frame_bytes * spec.num_blocks == store.region.total_size_bytes
    cursor = 0
    for tensor in cpu:
        assert tensor.data_ptr() == store.host.data_ptr() + cursor
        assert tensor.stride(0) == spec.unified_frame_bytes
        cursor += tensor.shape[1]
    if tiering:
        assert (
            spec.persistent_layout["host_main_kv"]["page_bytes"]
            == store.shape.page_bytes
        )
        assert not handlers.gpu_to_cpu_handler.group_mmap_regions


def _streams(monkeypatch):
    state = SimpleNamespace(retired=False, fail_drain=False, fences=0)

    def synchronize():
        state.fences += 1
        if state.fail_drain:
            raise RuntimeError("pending host transfer drain")
        state.retired = True

    stream = SimpleNamespace(
        wait_stream=lambda _: None, wait_event=lambda _: None, synchronize=synchronize
    )

    def event(**kwargs):
        return SimpleNamespace(
            record=lambda _: None,
            query=lambda: state.retired,
            synchronize=synchronize,
            elapsed_time=lambda _: 1,
        )

    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch, "Event", event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    return state


@pytest.mark.parametrize("factor", [1, 2])
def test_host_store_waits_for_producer_and_copies_the_correct_subpage(
    monkeypatch, factor
):
    state = _streams(monkeypatch)
    host = torch.arange(4 * 32, dtype=torch.int8).view(4, 32)
    cpu = torch.zeros((4, factor * 32), dtype=torch.int8)
    handler = gpu_worker.SingleDirectionOffloadingHandler(
        gpu_tensors=[host],
        cpu_tensors=[cpu],
        block_size_factor=factor,
        kv_cache_groups_data_refs=[[CanonicalKVCacheRef(0, 32)]],
        gpu_to_cpu=True,
    )
    gpu = GPULoadStoreSpec([2], [1], [factor - 1])
    assert handler.transfer_async(1, (gpu, CPULoadStoreSpec([1])))
    assert state.retired and state.fences == 1
    assert torch.equal(cpu[1, (factor - 1) * 32 :], host[2])
    assert not cpu[0].any() and not cpu[2:].any()
    if factor == 2:
        assert not cpu[1, :32].any()
    (result,) = handler.get_finished()
    assert result.success and result.transfer_size == 32
    handler.shutdown()


@pytest.mark.parametrize("failure", [None, "copy", "end"])
@pytest.mark.parametrize("factor", [1, 2])
def test_ordinary_cpu_tensor_copy_cannot_publish_host_receive(
    monkeypatch, failure, factor
):
    state = _streams(monkeypatch)
    shape = HostHistoryShape(
        num_blocks=4,
        kernel_blocks_per_block=1,
        kernel_block_tokens=4,
        num_layers=1,
        kv_heads=1,
        head_dim=16,
        dtype=torch.uint8,
        hot_slots=10,
        max_requests=1,
        selection_width=4,
        max_write_rows=8,
        row_groups=1,
        tail_groups=0,
    )
    store = HostHistoryStore(shape, ["main"], device="cpu")
    host = store.host.view(torch.int8).view(4, shape.page_bytes)
    cpu = (
        torch.arange(4 * shape.page_bytes * factor, dtype=torch.int64)
        .to(torch.int8)
        .view(4, shape.page_bytes * factor)
    )
    index = torch.empty((4, 16), dtype=torch.int8)
    cpu_index = torch.arange(4 * 16 * factor, dtype=torch.int8).view(4, 16 * factor)
    handler = gpu_worker.SingleDirectionOffloadingHandler(
        gpu_tensors=[host, index],
        cpu_tensors=[cpu, cpu_index],
        block_size_factor=factor,
        kv_cache_groups_data_refs=[
            [],
            [CanonicalKVCacheRef(0, shape.page_bytes), CanonicalKVCacheRef(1, 16)],
        ],
        gpu_to_cpu=False,
    )
    # The index stands in for device memory. Execute its batch copy on real
    # CPU bytes after the same stream fence used by the CUDA implementation.
    handler._host_tensor_indices = {0}

    def copy_index(src, dst, sizes, **kwargs):
        assert state.retired
        for source, target, size in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(target, source, size)
        if failure == "copy":
            state.fail_drain = True
            raise RuntimeError("index batch copy")

    monkeypatch.setattr(gpu_worker.ops, "swap_blocks_batch", copy_index)
    worker = OffloadingConnectorWorker(
        SimpleNamespace(
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(qsa_cross_group_hit_check=True)
            )
        )
    )
    worker._host_history, worker._host_group_id = store, 1
    worker.worker.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, handler)
    for wave in range(3):
        store.begin_blocks([2])
        state.fail_drain = False
        start, end = torch.Event(), torch.Event()
        if failure == "end":

            def fail_record(_):
                state.fail_drain = True
                raise RuntimeError("end")

            end.record = fail_record
        handler._event_pool = [end, start]
        metadata = OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
        metadata.load_jobs[wave] = TransferJob(
            req_id="request",
            transfer_spec=(
                CPULoadStoreSpec([1]),
                GPULoadStoreSpec([2], [0, 1], [0, factor - 1]),
            ),
        )
        worker.start_kv_transfers(metadata)
        assert torch.equal(host[2], cpu[1, (factor - 1) * shape.page_bytes :])
        assert torch.equal(index[2], cpu_index[1, (factor - 1) * 16 :])
        assert not store.ready[8:12].any()
        if failure is not None:
            assert worker.get_finished(set()) == (set(), set())
            assert worker.build_connector_worker_meta() is None
            assert wave in handler._failed_transfers
            state.fail_drain = False
        assert worker.get_finished(set()) == (set(), {"request"})
        meta = worker.build_connector_worker_meta()
        assert meta.completed_jobs == {wave: 1}
        assert meta.failed_jobs == ({wave: 1} if failure else {})
        # G0 publication now requires explicit ReceiveTxn source identities.
        assert not store.ready[8:12].any()
        assert not worker._host_load_blocks
        assert worker.get_finished(set()) == (set(), set())
    worker.shutdown()
