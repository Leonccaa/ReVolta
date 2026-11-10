# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-preserving staged writes over the real multi-worker, multi-layer ABI."""

from dataclasses import replace
from uuid import uuid4

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
)
from vllm.models.qwen4_exp.nvidia.host_kv.staged_write import plan_write, prefill_slots
from vllm.v1.kv_offload.host.region import HostRegionLayout
from vllm.v1.kv_offload.tiering.fs.identity import _kernel_identity, canonical_digest


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("chunk", [7, 31])
def test_staged_host_hot_and_neighbor_bytes(dtype, rank, chunk):
    shape = HostHistoryShape(
        num_blocks=6,
        kernel_blocks_per_block=3,
        kernel_block_tokens=8,
        num_layers=2,
        kv_heads=2,
        head_dim=16,
        dtype=dtype,
        hot_slots=64,
        max_requests=2,
        selection_width=4,
        max_write_rows=128,
        row_groups=2,
        dcp_world_size=2,
        dcp_rank=rank,
        interleave=2,
    )
    stores = []
    owners = []
    try:
        for enabled in (False, True):
            layout = HostRegionLayout(uuid4().hex, 8, 2, shape.page_bytes, (96,))
            owners.append(layout.open(None))
            store = HostHistoryStore(
                replace(shape, staged_writethrough=enabled),
                ["a", "b"],
                device="cpu",
                region_layout=layout,
                rank=rank,
            )
            stores.append(store)
            store.region._base.view(torch.uint8).fill_(0xA5)
            store.bind_pages([(1, 6), (2, 2), (3, 5), (4, -1)])
            store.begin_blocks([1, 2, 3, 4])
            for layer in store.layers:
                layer.group_to_slot[:] = torch.arange(
                    shape.num_groups, dtype=torch.int32
                )
                layer.hot.fill_(17)
        tables = {0: [3, 4, 5, 6, 7, 8, 9, 10, 11]}
        all_slots = prefill_slots(tables, [0], [139], [3], [True], 139, 8, 2, rank, 2)
        for start in range(0, len(all_slots), chunk):
            slots = all_slots[start : start + chunk]
            slot_tensor = torch.tensor(slots, dtype=torch.int64)
            key = torch.arange(len(slots) * 32, dtype=torch.int32).reshape(-1, 2, 16)
            key = ((key + start) % 251).to(dtype)
            value = (key + 1).to(dtype)
            stores[1].prepare_write(slots)
            for layer in range(2):
                for store in stores:
                    store.layers[layer].write_encoded(key, value, slot_tensor)
            assert torch.equal(stores[0].region._base, stores[1].region._base)
            assert torch.equal(stores[0].ready, stores[1].ready)
            for a, b in zip(stores[0].layers, stores[1].layers):
                assert torch.equal(a.hot, b.hot)
        # Entire region comparison includes the other worker, index bytes,
        # untouched layers/subpages, null page, unbound pages and tail canaries.
        expected = shape.fixed_device_memory_bytes()
        actual = stores[1].fixed_device_memory_bytes()
        assert actual["write_stage"] == expected["staging"]
        assert actual["write_targets"] == 16 * shape.max_write_rows
        for a, b in zip(stores[0].layers, stores[1].layers):
            # No step plan selects vector UVA for decode, including resident hot groups.
            stores[1].write_plan = None
            slots = torch.tensor([24, 25, -1, 48], dtype=torch.int64)
            key = torch.arange(128, dtype=torch.int32).reshape(4, 2, 16).to(dtype)
            a.write_encoded(key, key, slots)
            b.write_encoded(key, key, slots)
        assert torch.equal(stores[0].region._base, stores[1].region._base)
        assert torch.equal(stores[0].hot[0], stores[1].hot[0])
    finally:
        for store in stores:
            store.close()
        for owner in owners:
            owner.cleanup()


def test_descriptors_merge_by_subpage_and_have_no_token_calls():
    slots = list(range(24, 72))
    plan = plan_write(slots, [0, 6, 2], 3, 8, 16, 8192, 512, 128)
    assert plan.payload_bytes == 2 * len(slots) * 16
    assert len(plan.copies) == 2
    assert [copy.height for copy in plan.copies] == [3, 3]
    assert all(copy.width == 256 for copy in plan.copies)
    with pytest.raises(ValueError, match="Duplicate"):
        plan_write([24, 24], [0, 1], 3, 8, 16, 8192, 512, 128)


def test_dcp_cpu_slot_partition_and_decode_rows():
    args = ({0: [5, 3, 6], 1: [4]}, [0, 1], [39, 2], [1, 0], [True, False], 44, 8)
    for il in (1, 2, 4):
        left = prefill_slots(*args, 2, 0, il)
        right = prefill_slots(*args, 2, 1, il)
        assert all((a >= 0) != (b >= 0) for a, b in zip(left[:39], right[:39]))
        assert left[39:] == right[39:] == [-1] * 5


def test_switch_keeps_legacy_compile_and_ssd_identity():
    a = KernelConfig(long_context_kv={"enabled": True, "host_store_gib": 4})
    b = KernelConfig(
        long_context_kv={
            "enabled": True,
            "host_store_gib": 4,
            "staged_writethrough": False,
        }
    )
    assert a.long_context_kv.staged_writethrough is True
    assert a.compute_hash() == b.compute_hash()
    assert canonical_digest(_kernel_identity(a)) == canonical_digest(
        _kernel_identity(b)
    )
    identity = repr(_kernel_identity(a))
    assert "staged_writethrough" not in identity


def test_cpu_block_mirror_append_overwrite_and_padding():
    from types import SimpleNamespace

    import numpy as np

    from vllm.v1.worker.gpu.block_table import BlockTables

    tables = BlockTables.__new__(BlockTables)
    tables.num_kv_cache_groups = 1
    tables.blocks_per_kv_block = [3]
    tables.num_blocks = SimpleNamespace(np=np.zeros((1, 2), dtype=np.int32))
    tables.cpu_mirrors = {0: {}}
    tables.block_tables = [SimpleNamespace(stage_write=lambda *_: None)]
    tables.append_block_ids(0, ([3, 1],), overwrite=True)
    tables.append_block_ids(0, ([5],), overwrite=False)
    assert tables.cpu_mirrors[0][0] == [9, 10, 11, 3, 4, 5, 15, 16, 17]
    tables.append_block_ids(0, ([2],), overwrite=True)
    assert tables.cpu_mirrors[0][0] == [6, 7, 8]


def test_mixed_batch_stages_prefill_and_keeps_decode_uva():
    shape = HostHistoryShape(
        num_blocks=4,
        kernel_blocks_per_block=1,
        kernel_block_tokens=8,
        num_layers=1,
        kv_heads=1,
        head_dim=16,
        dtype=torch.uint8,
        hot_slots=32,
        max_requests=2,
        selection_width=4,
        max_write_rows=8,
        row_groups=2,
    )
    stores = [
        HostHistoryStore(replace(shape, staged_writethrough=flag), ["a"], device="cpu")
        for flag in (False, True)
    ]
    try:
        slots = torch.tensor([8, 9, 16, 17, -1], dtype=torch.int64)
        key = torch.arange(80).reshape(5, 1, 16).to(torch.uint8)
        stores[1].prepare_write([-1, -1, 16, 17, -1])
        for store in stores:
            store.layers[0].write_encoded(key, key, slots)
        assert torch.equal(stores[0].host, stores[1].host)
        assert torch.equal(stores[0].ready, stores[1].ready)
    finally:
        for store in stores:
            store.close()
