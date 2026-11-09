# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU address/descriptor tests; CUDA registration and DMA are separate gates."""

from unittest.mock import Mock
from uuid import uuid4

import pytest
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.unified.layout import transfer_layout

pytestmark = pytest.mark.cpu_test


def caches():
    return CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(torch.zeros((2, 8), dtype=torch.int8), 8)
            for _ in range(2)
        ],
        group_data_refs=[
            [],
            [
                CanonicalKVCacheRef(0, 6, dtypes=("fp16", "fp32")),
                CanonicalKVCacheRef(1, 6, dtypes=("fp16", "fp32")),
            ],
            [CanonicalKVCacheRef(0, 8, subdivision=2, dtypes=("fp16",), scale="none")],
        ],
    )


def test_overlapping_views_do_not_consume_main_cursor_or_map_again(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "vllm.v1.kv_offload.cpu.gpu_worker.is_pin_memory_available", lambda: False
    )

    def handler(**kwargs):
        captured.append(kwargs)
        return Mock()

    monkeypatch.setattr(
        "vllm.v1.kv_offload.cpu.gpu_worker.SingleDirectionOffloadingHandler", handler
    )
    regions = []
    name = "integration requirement-layout-" + uuid4().hex
    try:
        for rank in range(4):
            r = SharedOffloadRegion(name, 2 * 4 * 32, 2, rank, 4, 32)
            regions.append(r)
            main = r.create_next_view(16)
            index = r.create_next_view(16)
            old_cursor = r._worker_offset
            h = CpuGpuOffloadingHandlers(caches(), 1, 2, unified_region=r)
            assert r._worker_offset == old_cursor
            tensors = captured[-2]["cpu_tensors"]
            assert tensors[0].data_ptr() == main.data_ptr()
            assert tensors[1].data_ptr() == main.data_ptr() + 8
            assert tensors[0].stride(0) == 128
            assert [d.offset for d in h.transfer_layout] == [0, 8, 0]
            assert [d.length for d in h.transfer_layout] == [6, 6, 8]
            assert h.transfer_layout[2].subdivision == 2
            tensors[0][1].fill_(rank + 1)
            assert main[1, :8].tolist() == [rank + 1] * 8
            assert index[1].tolist() == [0] * 16
        assert regions[0].create_view_at(0, 8)[1].tolist() == [1] * 8
    finally:
        captured.clear()
        h = None
        main = index = tensors = None
        for r in reversed(regions):
            r.cleanup()


def test_descriptor_rejects_layout_outside_registered_frame():
    with pytest.raises(ValueError, match="exceed"):
        transfer_layout(caches(), 32, 8)
    c = caches()
    c.group_data_refs[1][0].member_offsets = (0, 4)
    c.group_data_refs[1][0].member_lengths = (4, 2)
    desc = transfer_layout(c, 32, 16)[0]
    assert desc.member_offsets == (0, 4) and desc.member_lengths == (4, 2)
    c.group_data_refs[1][0].page_size_bytes = 9
    with pytest.raises(ValueError, match="payload"):
        transfer_layout(c, 32, 16)


def test_worker_rank_bytes_split_by_kind_and_duplicate_aggregate(monkeypatch):
    from types import SimpleNamespace

    from tests.v1.kv_offload.test_unified_units import manager
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
        TransferJob,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.kv_offload.base import GPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
    from vllm.v1.kv_offload.unified.manager import ComponentTransferSpec

    m = manager()
    spec = CPUOffloadingSpec.__new__(CPUOffloadingSpec)
    spec.kv_cache_config = m.spec.kv_cache_config
    desc = transfer_layout(caches(), 128, 32)
    spec._handlers = SimpleNamespace(transfer_layout=desc)
    worker = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    worker.spec = spec
    worker._unified_generations = {}
    worker._unified_payload_bytes = {}
    worker._connector_worker_meta = OffloadingWorkerMetadata()
    monkeypatch.setattr(
        "vllm.distributed.get_world_group", lambda: SimpleNamespace(rank_in_group=2)
    )
    job = TransferJob(
        "r",
        (
            GPULoadStoreSpec([1, 2, 3], group_sizes=[0, 1, 2], block_indices=[0, 0, 0]),
            ComponentTransferSpec((), 41),
        ),
        generation=41,
    )
    worker._remember_unified_job(3, job)
    # The two STATE tensors contribute 12 bytes; two DRAFT pages contribute 16.
    worker._receive_receipt(3, True, 28)
    meta = worker._connector_worker_meta
    assert meta.job_results == {(3, 41): (4, 0)}
    assert meta.job_bytes == {
        (3, 41, 2, "state"): 12,
        (3, 41, 2, "draft"): 16,
        (3, 41, 2, "main"): 0,
    }
    twice = meta.aggregate(meta)
    assert twice.job_results == meta.job_results and twice.job_bytes == meta.job_bytes


def test_real_worker_mamba_members_match_production_transport():
    from dataclasses import replace
    from types import SimpleNamespace

    from tests.v1.core.long_context_kv_utils import (
        SPAN,
        production_config,
        production_specs,
    )
    from tests.v1.core.test_long_context_host_main_kv import host_settings
    from tests.v1.core.test_long_context_split_pool import _layer_specs
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )
    from vllm.v1.attention.kv_cache_view import packed_kv_members
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    config = production_config(
        max_model_len=4 * SPAN,
        max_num_seqs=1,
        long_context=host_settings(host_pool_tokens=8 * SPAN, hot_pool_gib=0.2),
    )
    config.model_config.original_max_model_len = 4 * SPAN
    config.instance_id = "integration requirement-real-worker-members"
    config.parallel_config.world_size = 4
    config.cache_config.block_size = SPAN
    config.cache_config.hash_block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_cache_retention_interval = 0
    config.kv_transfer_config.kv_connector_extra_config = {}
    specs = production_specs(speculative_blocks=0)
    # Frozen reference layout Qwen4Exp TP4 shape, including seven speculative conv
    # slots.
    shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
        4, 16, 48, 128, 128, 4, 7
    )
    dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.float16, "auto", "auto"
    )
    ple_shapes = MambaStateShapeCalculator.short_conv_state_shape(1, 2560 * 4, 10, 7)
    for name, size in specs.items():
        if name.endswith(".linear_attn"):
            specs[name] = replace(size, shapes=shapes, dtypes=dtypes)
        elif name.endswith(".ple"):
            specs[name] = replace(size, shapes=ple_shapes, dtypes=(torch.float16,))
        elif name.endswith(".compressed"):
            specs[name] = replace(size, block_size=SPAN // 2, dcp_sharded=True)
    (kv,) = get_kv_cache_configs(config, [specs], [int(6.05 * (1 << 30))])
    spec = CPUOffloadingSpec(config, kv)
    kv.kv_cache_tensors = [
        replace(t, size=t.size // t.num_blocks * 2, num_blocks=2)
        for t in kv.kv_cache_tensors
    ]
    kv.num_blocks = 2
    layer_specs = _layer_specs(kv)
    packed = packed_kv_members(kv)
    caches = {}
    for tensor in kv.kv_cache_tensors:
        raw = torch.empty(tensor.size, dtype=torch.uint8)
        for name in tensor.shared_by:
            size = layer_specs[name][1]
            if isinstance(size, MambaSpec):
                offset = 0
                members = []
                for shape, dtype in zip(size.shapes, size.dtypes):
                    item = torch.empty((), dtype=dtype).element_size()
                    shape = (2, *shape)
                    stride = torch.empty(shape, device="meta").stride()
                    members.append(
                        torch.as_strided(
                            raw.view(dtype),
                            shape,
                            (size.page_size_bytes // item, *stride[1:]),
                            offset // item,
                        )
                    )
                    offset += stride[0] * item
                caches[name] = members
            else:
                member, _ = packed.get(name, (0, 1))
                caches[name] = raw[member * size.page_size_bytes :]
                if getattr(size, "sliding_window", None) is not None:
                    caches[name] = raw.view(size.dtype).reshape(2, 2, -1)
    store = type("Store", (), {"region": object()})()
    config.compilation_config = SimpleNamespace(
        static_forward_context={
            name: SimpleNamespace(host_history=SimpleNamespace(store=store))
            for name in kv.host_kv_layers
        }
    )
    worker = OffloadingConnectorWorker(spec)
    captured: list[CanonicalKVCaches] = []
    worker._register_handlers = captured.append
    worker.register_kv_caches(caches)
    (canonical,) = captured
    desc = transfer_layout(canonical, 23_003_136, 5_750_784)
    assert len(canonical.tensors) == 6 and len(desc) == 42
    gdn = [d for d in desc if 2 <= d.group <= 7]
    assert len(gdn) == 36
    for group in range(2, 8):
        refs = [d for d in gdn if d.group == group]
        assert [d.offset for d in refs] == [i * 851_968 for i in range(6)]
        assert sum(d.length for d in refs) == 5_025_792
    assert all(d.member_offsets == (d.offset, d.offset + 51_200) for d in gdn)
    assert all(d.member_lengths == (51_200, 786_432) for d in gdn)
    assert all(d.dtypes == ("torch.float16", "torch.float32") for d in gdn)
    assert all(
        d.subdivision == 1 and d.gpu_stride == 851_968 and d.cpu_stride == 23_003_136
        for d in desc
    )
    assert sum(d.length for d in desc if d.group == 8) == 327_680
    assert sum(d.length for d in desc if d.group == 9) == 4_259_840
