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
    # A last full subpage followed by a far physical page stays two 1D copies.
    far = plan_write(list(range(40, 56)), [0, 1, 1 << 20], 3, 8, 16, 8192, 512, 128)
    assert len(far.copies) == 2
    assert all(copy.height == 1 for copy in far.copies)

    with pytest.raises(ValueError, match="Duplicate"):
        plan_write([24, 24], [0, 1], 3, 8, 16, 8192, 512, 128)


def test_dcp_cpu_slot_partition_and_decode_rows():
    args = ({0: [5, 3, 6], 1: [4]}, [0, 1], [39, 2], [1, 0], [True, False], 44, 8)
    for il in (1, 2, 4):
        left = prefill_slots(*args, 2, 0, il)
        right = prefill_slots(*args, 2, 1, il)
        assert all((a >= 0) != (b >= 0) for a, b in zip(left[:39], right[:39]))
        assert left[39:] == right[39:] == [-1] * 5


def test_switch_keeps_legacy_compile_identity():
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
    for dim in (8, 16):
        shape = HostHistoryShape(
            num_blocks=4,
            kernel_blocks_per_block=1,
            kernel_block_tokens=8,
            num_layers=1,
            kv_heads=1,
            head_dim=dim,
            dtype=torch.uint8,
            hot_slots=32,
            max_requests=2,
            selection_width=4,
            max_write_rows=8,
            row_groups=2,
        )
        stores = [
            HostHistoryStore(
                replace(shape, staged_writethrough=flag), ["a"], device="cpu"
            )
            for flag in (False, True)
        ]
        try:
            slots = torch.tensor([8, 9, 16, 17, -1], dtype=torch.int64)
            key = torch.arange(5 * dim).reshape(5, 1, dim).to(torch.uint8)
            stores[1].prepare_write([-1, -1, 16, 17, -1])
            for store in stores:
                store.layers[0].write_encoded(key, key, slots)
            assert torch.equal(stores[0].host, stores[1].host)
            assert torch.equal(stores[0].ready, stores[1].ready)
        finally:
            for store in stores:
                store.close()


def _planar_prefill_roundtrip(device, rank):
    from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD as codec

    shape = HostHistoryShape(
        num_blocks=6,
        kernel_blocks_per_block=3,
        kernel_block_tokens=8,
        num_layers=2,
        kv_heads=2,
        head_dim=256,
        dtype=torch.uint8,
        hot_slots=64,
        max_requests=2,
        selection_width=4,
        max_write_rows=128,
        row_groups=2,
        dcp_world_size=2,
        dcp_rank=rank,
        interleave=2,
        layout_cache_dtype=codec.name,
        staged_writethrough=True,
    )
    layout = HostRegionLayout(uuid4().hex, 8, 2, shape.page_bytes, (96,))
    owner = layout.open(None)
    store = HostHistoryStore(
        shape, ["a", "b"], device=device, region_layout=layout, rank=rank
    )
    try:
        store.region._base.view(torch.uint8).fill_(0xA5)
        expected_region = store.region._base.view(torch.uint8).clone()
        store.bind_pages([(1, 4), (2, 2)])
        store.begin_blocks([1, 2])
        for layer in store.layers:
            layer.group_to_slot[:] = torch.arange(shape.num_groups, device=device)
        tables = {0: [3, 4, 5, 6, 7, 8]}
        for step, mode in enumerate(("runs", "mixed", "fallback")):
            count, start = (90, 3) if mode != "mixed" else (60, 16)
            slots = prefill_slots(
                tables, [0], [count], [start], [True], count + 2, 8, 2, rank, 2
            )
            if mode == "mixed":
                # Two decode rows retain UVA in the staged scalar kernel.
                slots[-2:] = [24, 25]
                store.prepare_write(slots[:-2] + [-1, -1])
            else:
                store.prepare_prefill_write(
                    tables, [0], [count], [start], [True], len(slots), 2, rank, 2
                )
                fast = store.write_plan
                assert store.write_plan_kind == "runs"
                store.prepare_write(slots)
                assert store.write_plan.copies == fast.copies
                assert store.write_plan.payload_bytes == fast.payload_bytes
                if mode == "runs":
                    store.prepare_prefill_write(
                        tables, [0], [count], [start], [True], len(slots), 2, rank, 2
                    )
            assert store.write_plan_all_prefill == (mode == "runs")
            copies = store.write_plan.copies
            payload, scales = copies[: len(copies) // 2], copies[len(copies) // 2 :]
            assert any(c.width == 8192 and c.height > 1 for c in payload)
            assert any(c.width < 8192 and c.height == 2 for c in payload)
            for data, scale in zip(payload, scales):
                assert scale.source == 131072 + data.source // 64
                assert scale.width == data.width // 64
                assert scale.source_pitch == data.source_pitch // 64
                assert scale.destination_pitch == (
                    data.destination_pitch if data.width == 8192 else 64
                )
            slots_tensor = torch.tensor(slots, dtype=torch.int64, device=device)
            source = torch.arange(len(slots) * 512).reshape(-1, 2, 256)
            key = ((source % 997 - 498).float() * (step + 1) / 17).half().to(device)
            value = (key.float() * -0.375 + step).half()
            key[0].zero_()
            encoded = [*codec.encode(key.cpu()), *codec.encode(value.cpu())]
            for layer in store.layers:
                assert layer._word_writer_eligible(
                    layer.staging_cache.key[:, 0], layer.staging_cache.value[:, 0]
                )
                layer.write(
                    key,
                    value,
                    slots_tensor,
                    codec.name,
                    torch.tensor(9.0, device=device),
                    torch.tensor(11.0, device=device),
                )
                event = store.fence()
                if event is not None:
                    event.synchronize()
                expected_host = expected_region.as_strided(
                    layer.host_view.shape,
                    layer.host_view.stride(),
                    layer.host_view.storage_offset(),
                )
                page_layout = shape.page_layout(8)
                for row, slot in enumerate(slots):
                    if slot < 0:
                        continue
                    kb, off = divmod(slot, 8)
                    data = (
                        encoded[0][row],
                        encoded[2][row],
                        encoded[1][row],
                        encoded[3][row],
                    )
                    for offset, tensor in zip(
                        (
                            off * 512,
                            page_layout.value_offset + off * 512,
                            page_layout.key_scale_offset + off * 8,
                            page_layout.value_scale_offset + off * 8,
                        ),
                        data,
                    ):
                        raw = tensor.contiguous().view(torch.uint8).flatten()
                        expected_host[
                            int(store.page_of_cpu[kb // 3]),
                            kb % 3,
                            offset : offset + raw.numel(),
                        ] = raw
                    for cache, tensor in zip(
                        (
                            layer.key_cache,
                            layer.value_cache,
                            layer.key_scale_cache,
                            layer.value_scale_cache,
                        ),
                        data,
                    ):
                        assert torch.equal(cache[slot // 4, slot % 4].cpu(), tensor)
                    assert store.ready[slot] == 1
                assert torch.equal(
                    store.region._base.view(torch.uint8), expected_region
                )
        # Use the actual final-host bytes for both required negative checks.
        actual = store.region._base.view(torch.uint8)

        def compare(data):
            assert data.shape == expected_region.shape, "INCOMPLETE"
            assert torch.equal(data, expected_region), "FAIL"

        byte = (
            layer.host_view.data_ptr()
            - actual.data_ptr()
            + 4 * layer.host_view.stride(0)
            + page_layout.key_scale_offset
        )
        actual[byte] ^= 1
        with pytest.raises(AssertionError, match="FAIL"):
            compare(actual)
        actual[byte] ^= 1
        with pytest.raises(AssertionError, match="INCOMPLETE"):
            compare(actual[:byte])
    finally:
        store.close()
        owner.cleanup()


@pytest.mark.parametrize("rank", [0, 1])
def test_planar_staged_vector_and_mixed_prefill(rank):
    _planar_prefill_roundtrip("cpu", rank)
