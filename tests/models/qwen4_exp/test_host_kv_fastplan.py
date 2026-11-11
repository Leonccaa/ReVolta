# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run-plan equivalence, fallback and CPU buffer ownership checks."""

import random
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
)
from vllm.models.qwen4_exp.nvidia.host_kv.planning import plan_passes
from vllm.models.qwen4_exp.nvidia.host_kv.staged_write import (
    plan_prefill_write,
    plan_write,
    prefill_slots,
)
from vllm.v1.kv_offload.host.region import HostRegionLayout


@pytest.mark.parametrize(
    "world,rank,interleave",
    [
        (1, 0, 1),
        (2, 0, 1),
        (2, 1, 1),
        (2, 0, 2),
        (2, 1, 2),
        (2, 0, 4),
        (2, 1, 4),
        (2, 0, 8),
        (2, 1, 8),
        (4, 3, 2),
    ],
)
def test_random_run_plan_fields(world, rank, interleave):
    rng = random.Random(901 + world * 10 + rank + interleave)
    for _ in range(100):
        kbs = rng.choice([8, 16, 32])
        ratio = rng.choice([1, 3, 7, 49])
        row_bytes = rng.choice([16, 512, 1024])
        sub_stride = 2 * kbs * row_bytes + rng.choice([0, 128, 4096])
        page_stride = ratio * sub_stride + 512
        args = (ratio, kbs, row_bytes, page_stride, sub_stride, kbs * row_bytes)
        table = rng.sample(range(1, ratio * 70), 60)
        pages = rng.sample(range(1, 150), 71)
        start = rng.randrange(kbs * world * 20)
        count = rng.randrange(1, kbs * world * 25)
        padded = count + rng.randrange(0, 12)
        targets = np.empty((padded, 2), dtype=np.int64)
        metadata = ({0: table}, [0], [count], [start], [True])
        old = plan_write(
            prefill_slots(*metadata, padded, kbs, world, rank, interleave),
            pages,
            *args,
        )
        new = plan_prefill_write(
            *metadata,
            targets,
            np.arange(kbs, dtype=np.int64),
            pages,
            *args,
            world,
            rank,
            interleave,
        )
        assert new is not None
        assert np.array_equal(new.targets, old.targets)
        assert new.copies == old.copies
        assert new.payload_bytes == old.payload_bytes


@pytest.mark.parametrize(
    "case", ["multi", "mixed", "duplicate", "empty", "missing", "geometry"]
)
def test_explicit_fallback(case):
    metadata = [{0: [3, 4, 5]}, [0], [20], [0], [True]]
    pages = [0, 3, 1]
    interleave = 1
    if case in ("multi", "mixed"):
        metadata = [
            {0: [3, 4], 1: [6]},
            [0, 1],
            [12, 3],
            [0, 0],
            [True, case == "multi"],
        ]
    elif case == "duplicate":
        metadata[0] = {0: [3, 3, 5]}
    elif case == "empty":
        pages = [0, 0, 1]
    elif case == "missing":
        metadata[0] = {}
    else:
        interleave = 3
    targets = np.empty((32, 2), dtype=np.int64)
    assert (
        plan_prefill_write(
            *metadata,
            targets,
            np.arange(8, dtype=np.int64),
            pages,
            3,
            8,
            16,
            4096,
            512,
            128,
            2,
            0,
            interleave,
        )
        is None
    )


def test_closed_passes_match_greedy_and_errors():
    rng = random.Random(912)
    for _ in range(3000):
        rows = rng.randrange(0, 8193)
        groups = rng.randrange(0, 30000)
        avail = rng.randrange(0, 30000)
        cap = rng.randrange(1, 1025)
        kw = dict(row_groups=rng.randrange(1, 520), tail_groups=rng.randrange(0, 20))
        try:
            old = plan_passes([rows], [groups], avail, cap, **kw)
        except ValueError as error:
            with pytest.raises(ValueError, match=str(error)):
                plan_passes(
                    [rows], [groups], avail, cap, single_request_fast=True, **kw
                )
        else:
            assert old == plan_passes(
                [rows], [groups], avail, cap, single_request_fast=True, **kw
            )
    # More than one request, including padding, retains the greedy algorithm.
    for _ in range(100):
        batch_rows = [rng.randrange(10), rng.randrange(10), 0]
        batch_groups = [rng.randrange(200), rng.randrange(200), 0]
        assert plan_passes(batch_rows, batch_groups, 1000, 7) == plan_passes(
            batch_rows,
            batch_groups,
            1000,
            7,
            single_request_fast=True,
        )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("shared", [False, True])
def test_runtime_fastplan_bytes_and_fallback(rank, mixed, dtype, shared):
    shape = HostHistoryShape(
        num_blocks=6,
        kernel_blocks_per_block=3,
        kernel_block_tokens=8,
        num_layers=2,
        kv_heads=1,
        head_dim=16,
        dtype=dtype,
        hot_slots=64,
        max_requests=2,
        selection_width=4,
        max_write_rows=40,
        row_groups=2,
        dcp_world_size=2,
        dcp_rank=rank,
        interleave=2,
    )
    stores, owners = [], []
    try:
        for flag in (False, True):
            layout = None
            if shared:
                layout = HostRegionLayout(uuid4().hex, 8, 2, shape.page_bytes, (96,))
                owners.append(layout.open(None))
            stores.append(
                HostHistoryStore(
                    replace(shape, staged_writethrough=flag),
                    ["a", "b"],
                    device="cpu",
                    region_layout=layout,
                    rank=rank,
                )
            )
            if shared:
                stores[-1].region._base.view(torch.uint8).fill_(0xA5)
        for store in stores:
            store.bind_pages([(1, 4), (2, 2), (3, 5)])
            store.begin_blocks([1, 2, 3])
        ptrs = [t.data_ptr() for t in stores[1].write_target_buffers]
        # Three successive non-dividing chunks force both buffers to be reused.
        for start in (3, 34, 65):
            metadata = (
                {0: [3, 4, 5, 6, 7, 8, 9], 1: [6]},
                [0, 1] if mixed else [0],
                [31, 2] if mixed else [31],
                [start, 0] if mixed else [start],
                [True, False] if mixed else [True],
            )
            padded = 36
            stores[1].prepare_prefill_write(*metadata, padded, 2, rank, 2)
            assert stores[1].write_plan_kind == ("fallback" if mixed else "runs")
            slots = prefill_slots(*metadata, padded, 8, 2, rank, 2)
            # Decode positions are GPU-owned: supply them only to the writer.
            if mixed:
                slots[31:33] = [80, 81]
            key = torch.arange(padded * 16).reshape(padded, 1, 16).to(dtype)
            for store in stores:
                for layer in store.layers:
                    layer.write_encoded(key, key + 1, torch.tensor(slots))
            assert torch.equal(stores[0].host, stores[1].host)
            assert torch.equal(stores[0].ready, stores[1].ready)
            if shared:
                assert torch.equal(stores[0].region._base, stores[1].region._base)
            assert ptrs == [t.data_ptr() for t in stores[1].write_target_buffers]
    finally:
        for store in stores:
            store.close()
        for owner in owners:
            owner.cleanup()


def test_pinned_buffer_waits_only_for_its_own_copy():
    pending, other = Mock(), Mock()
    pending.query.return_value = False
    store = SimpleNamespace(
        write_target_index=0,
        write_target_events=[pending, other],
        write_target_arrays=[np.empty((10, 2)), np.empty((10, 2))],
    )
    assert HostHistoryStore._write_target_array(store, 3).shape == (3, 2)
    pending.synchronize.assert_called_once()
    other.query.assert_not_called()
    pending.reset_mock()
    pending.query.return_value = True
    HostHistoryStore._write_target_array(store, 3)
    pending.synchronize.assert_not_called()


def test_target_copy_event_is_recorded_after_htod(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.host_kv import history

    actions = []
    target, buffer, event = Mock(), Mock(), Mock()
    target.copy_.side_effect = lambda *a, **kw: actions.append("copy")
    event.record.side_effect = lambda *a: actions.append("record")
    stream = object()
    monkeypatch.setattr(history.torch.cuda, "current_stream", lambda device: stream)

    class Slice:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, key):
            return self.value

    store = SimpleNamespace(
        write_targets=Slice(target),
        write_target_index=0,
        write_target_buffers=[Slice(buffer), None],
        write_target_events=[event, None],
        device=SimpleNamespace(type="cuda"),
    )
    HostHistoryStore._upload_write_targets(store, 4)
    target.copy_.assert_called_once_with(buffer, non_blocking=True)
    event.record.assert_called_once_with(stream)
    assert actions == ["copy", "record"]
    assert store.write_target_index == 1


@pytest.mark.parametrize("mode", ["off", "decode", "capture"])
def test_runner_does_not_plan_or_upload_for_legacy_decode_capture(mode):
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    store = SimpleNamespace(
        shape=SimpleNamespace(staged_writethrough=mode != "off"),
        write_plan=None,
        prepare_prefill_write=Mock(),
    )
    batch = SimpleNamespace(is_prefilling_np=np.array([mode != "decode"]))
    graph = CUDAGraphMode.FULL if mode == "capture" else CUDAGraphMode.NONE
    GPUModelRunner._set_host_write_plan(
        SimpleNamespace(host_history=store), batch, graph
    )
    store.prepare_prefill_write.assert_not_called()
    assert store.write_plan is None


def test_disjoint_aliases_use_the_reference_run_join():
    metadata = ({0: [3, 3]}, [0], [8], [4], [True])
    targets = np.empty((8, 2), dtype=np.int64)
    args = ([0, 2], 3, 8, 16, 4096, 512, 128)
    assert (
        plan_prefill_write(
            *metadata,
            targets,
            np.arange(8, dtype=np.int64),
            *args,
            1,
            0,
            1,
        )
        is None
    )
    # Slots are unique, so the reference correctly packs these two fragments
    # into one complete subpage rather than rejecting the alias as a duplicate.
    plan = plan_write(prefill_slots(*metadata, 8, 8, 1, 0, 1), *args)
    assert plan.payload_bytes == 256
    assert len(plan.copies) == 1


@pytest.mark.parametrize("cuda_metadata", [False, True])
def test_pure_decode_retains_legacy_writer(monkeypatch, cuda_metadata):
    from vllm.models.qwen4_exp.nvidia.host_kv import history

    shape = HostHistoryShape(
        num_blocks=3,
        kernel_blocks_per_block=1,
        kernel_block_tokens=8,
        num_layers=1,
        kv_heads=1,
        head_dim=16,
        dtype=torch.uint8,
        hot_slots=64,
        max_requests=4,
        selection_width=4,
        max_write_rows=32,
        staged_writethrough=True,
        row_groups=2,
    )
    store = HostHistoryStore(shape, ["a"], device="cpu")
    store.begin_blocks([1, 2])
    if cuda_metadata:
        # Keep all tensors and interpreter kernels on CPU, but exercise the
        # CUDA metadata branch without allowing a capture-state API query.
        store.device = torch.device("cuda")
    monkeypatch.setattr(
        history.torch.cuda,
        "is_current_stream_capturing",
        Mock(side_effect=AssertionError("decode capture query")),
    )
    calls = []
    original = history.hk.write_slots

    class Writer:
        def __getitem__(self, grid):
            def call(*args):
                calls.append(grid)
                return original[grid](*args)

            return call

    monkeypatch.setattr(history.hk, "write_slots", Writer())
    monkeypatch.setattr(
        store.layers[0],
        "_write_words",
        Mock(side_effect=AssertionError("decode vector")),
    )
    try:
        key = torch.ones((4, 1, 16), dtype=torch.uint8)
        store.layers[0].write_encoded(key, key, torch.tensor([8, 9, 16, 17]))
        assert calls == [(4,)]
    finally:
        store.close()
