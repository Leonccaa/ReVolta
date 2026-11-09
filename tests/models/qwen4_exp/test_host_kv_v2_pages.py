# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real shared host rows, alias strides, indirection, null and logical seals."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
    HostKVResourceError,
)
from vllm.v1.attention.kv_cache_view import pack_page_payload
from vllm.v1.kv_offload.host.page_table import PageHandle, SealTicket
from vllm.v1.kv_offload.host.region import HostRegionLayout


@pytest.mark.parametrize("failure", ["registration", "metadata", "registered_metadata"])
def test_constructor_failure_closes_local_mapping_but_owner_unlinks(
    monkeypatch, failure
):
    from vllm.models.qwen4_exp.nvidia.host_kv import history

    shape = HostHistoryShape(
        num_blocks=4,
        kernel_blocks_per_block=1,
        kernel_block_tokens=16,
        num_layers=1,
        kv_heads=1,
        head_dim=8,
        dtype=torch.uint8,
        hot_slots=32,
        max_requests=1,
        selection_width=4,
        max_write_rows=4,
        row_groups=2,
    )
    layout = HostRegionLayout(uuid4().hex, 4, 1, shape.page_bytes, (96,))
    owner = layout.open(None)
    regions = []
    mappings = []
    unregisters = []
    original_open = HostRegionLayout.open

    def record_open(self, rank):
        region = original_open(self, rank)
        regions.append(region)
        mappings.append(region.mmap_obj)
        return region

    def fail(*_, **__):
        raise RuntimeError("injected metadata allocation failure")

    def unregister(ptr):
        unregisters.append(ptr)
        return SimpleNamespace(value=0)

    monkeypatch.setattr(HostRegionLayout, "open", record_open)
    monkeypatch.setattr(torch, "full", fail)
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: SimpleNamespace(
            cudaHostRegister=lambda *_: SimpleNamespace(
                value=int(failure == "registration")
            ),
            cudaHostUnregister=unregister,
        ),
    )
    monkeypatch.setattr(torch.accelerator, "device_index", lambda _: nullcontext())
    monkeypatch.setattr(history, "get_accelerator_view_from_cpu_tensor", lambda t: t)
    try:
        with pytest.raises((HostKVResourceError, RuntimeError)):
            HostHistoryStore(
                shape,
                ["l"],
                device="cpu" if failure == "metadata" else "cuda:0",
                region_layout=layout,
            )
        [region] = regions
        assert region.fd is None and region.mmap_obj is None
        assert mappings[0].closed
        assert len(unregisters) == int(failure == "registered_metadata")
        assert Path(region.mmap_path).exists()  # Worker never owns unlink.
    finally:
        for region in regions:
            region.cleanup()
        owner.cleanup()
    assert not Path(owner.mmap_path).exists()


@pytest.mark.parametrize("ratio", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_shared_alias_indirection_subpages_and_masked_writes(ratio, dtype):
    shape = HostHistoryShape(
        num_blocks=5,
        kernel_blocks_per_block=ratio,
        kernel_block_tokens=16,
        num_layers=2,
        kv_heads=1,
        head_dim=8,
        dtype=dtype,
        hot_slots=32,
        max_requests=1,
        selection_width=4,
        max_write_rows=4,
        row_groups=2,
    )
    layout = HostRegionLayout(uuid4().hex, 7, 2, shape.page_bytes, (32,))
    scheduler = layout.open(None)
    stores = [
        HostHistoryStore(
            shape, ["l0", "l1"], device="cpu", region_layout=layout, rank=rank
        )
        for rank in range(2)
    ]
    try:
        for rank, store in enumerate(stores):
            store.bind_pages([(1, 6), (2, -1), (3, 2)])
            store.begin_blocks([1, 2, 3])
            layer = store.layer("l1")
            view = layer.host_view
            assert view.untyped_storage()._cdata == store.host.untyped_storage()._cdata
            assert (
                view.data_ptr()
                == store.host.untyped_storage().data_ptr()
                + view.storage_offset() * dtype.itemsize
            )
            assert view.stride(0) * dtype.itemsize == layout.worker_bytes * 2
            assert view.stride(1) * dtype.itemsize == shape.page_bytes // ratio
            assert (
                view.untyped_storage().nbytes()
                == layout.num_pages * layout.worker_bytes * 2
            )
            # Different rows, workers and logical index bytes are sentinels.
            store.index_payloads[0].fill_(99)
            key = torch.full((4, 1, 8), rank + 3, dtype=dtype)
            value = key + 1
            slots = torch.tensor([ratio * 16 + (ratio - 1) * 16, 2 * ratio * 16, 0, -1])
            layer.write_encoded(key, value, slots)
            assert torch.equal(store.host[6, ratio - 1, 1, 0, 0], key[0])
            assert torch.equal(store.host[6, ratio - 1, 1, 1, 0], value[0])
            assert not store.host[0].any() and not store.host[2].any()
            assert store.index_payloads[0].eq(99).all()
            assert store.ready[2 * ratio * 16] == 0
            table = torch.tensor([[1 * ratio + ratio - 1]], dtype=torch.int32)
            selection = torch.tensor([[0, -1, 0, -1]], dtype=torch.int32)
            hot = layer.gather(
                selection,
                torch.tensor([0], dtype=torch.int32),
                table,
                torch.tensor([1], dtype=torch.int32),
            )
            token = int(hot[0, 0])
            assert torch.equal(layer.key_cache[token // 4, token % 4], key[0])
        assert not stores[0].host.is_contiguous()
    finally:
        del layer, view
        for store in stores:
            store.close()
        assert Path(scheduler.mmap_path).exists()  # workers never unlink
        scheduler.cleanup()
    assert not Path(scheduler.mmap_path).exists()


def test_seal_packs_subpage_padding_into_shared_logical_index_payload():
    shape = HostHistoryShape(
        num_blocks=4,
        kernel_blocks_per_block=1,
        kernel_block_tokens=16,
        num_layers=1,
        kv_heads=1,
        head_dim=8,
        dtype=torch.uint8,
        hot_slots=32,
        max_requests=1,
        selection_width=4,
        max_write_rows=4,
        row_groups=2,
    )
    # A scheduler page containing two padded logical subpages, viewed as
    # one index page. Every gap and the adjacent physical pages are checked.
    raw = torch.full((4, 128), 217, dtype=torch.uint8)
    logical = raw.as_strided((4, 2, 48), (128, 64, 1))
    logical[1].copy_(torch.arange(96, dtype=torch.uint8).reshape(2, 48))
    layout = HostRegionLayout(uuid4().hex, 4, 1, shape.page_bytes, (96,))
    owner = layout.open(None)
    store = HostHistoryStore(shape, ["l"], device="cpu", region_layout=layout)
    try:
        store.bind_pages([(1, 3)])
        store.index_payloads[0].fill_(53)
        ticket = SealTicket(1, 1, 1, PageHandle(3, 0), b"key", 1)
        store.seal([ticket], [logical])
        assert torch.equal(store.index_payloads[0][3], pack_page_payload(logical, 1))
        assert store.index_payloads[0][:3].eq(53).all()
        assert raw[:, 48:64].eq(217).all() and raw[:, 112:].eq(217).all()
    finally:
        store.close()
        owner.cleanup()


def test_whole_region_registration_and_scheduler_only_unlink(monkeypatch):
    from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region

    layout = HostRegionLayout(uuid4().hex, 3, 2, 256, (96,))
    region = layout.open(1)
    owner = layout.open(None)
    calls: list[tuple[int, ...]] = []

    def register(ptr, size, flags):
        calls.append((ptr, size, flags))
        return SimpleNamespace(value=0)

    def unregister(ptr):
        calls.append((ptr,))
        return SimpleNamespace(value=0)

    cudart = SimpleNamespace(
        cudaHostRegister=register,
        cudaHostUnregister=unregister,
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    pin_mmap_region(region, flags=3)
    assert calls == [(region._base.data_ptr(), 3 * 2 * 352, 3)]
    path = region.mmap_path
    region.cleanup()
    assert len(calls) == 2 and Path(path).exists()
    owner.cleanup()
    assert not Path(path).exists()


def test_canonical_handler_keeps_strides_without_host_page_remapping(
    monkeypatch,
):
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
    )
    from vllm.v1.kv_offload.cpu import gpu_worker

    monkeypatch.setattr(gpu_worker, "is_pin_memory_available", lambda: False)
    backing = torch.zeros((7, 160), dtype=torch.int8)
    main = backing[:, 32:96]
    handlers = gpu_worker.CpuGpuOffloadingHandlers(
        CanonicalKVCaches(
            [CanonicalKVCacheTensor(main, 64)], [[CanonicalKVCacheRef(0, 64)]]
        ),
        block_size_factor=1,
        num_cpu_blocks=3,
    )
    try:
        tensor = handlers.gpu_to_cpu_handler.src_tensors[0]
        assert not hasattr(tensor, "_host_page_of")
        assert tensor.data_ptr() == main.data_ptr() and tensor.stride(0) == 160
        assert tensor.untyped_storage()._cdata == backing.untyped_storage()._cdata
        ptrs = np.empty(2, dtype=np.int64)
        gpu_worker.compute_sub_block_ptrs(np.array([1, 3]), 1, ptrs, tensor)
        assert ptrs.tolist() == [main.data_ptr() + 160, main.data_ptr() + 3 * 160]
        gpu_worker.compute_sub_block_ptrs(np.array([1]), 1, ptrs[:1], tensor)
        assert ptrs[0] == main.data_ptr() + 160
    finally:
        handlers.gpu_to_cpu_handler.shutdown()
        handlers.cpu_to_gpu_handler.shutdown()
