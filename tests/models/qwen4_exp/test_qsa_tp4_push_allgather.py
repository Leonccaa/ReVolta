# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import vllm._custom_ops as ops
import vllm.distributed as distributed
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.models.qwen4_exp.nvidia.ops import qsa
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CapturedAttentionState,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import PrefillEagleCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from .test_qsa_index_dcp_shard import _reference_select

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("rows", "world", "push_buffer", "capture", "push"),
    [
        (0, 4, True, True, False),
        (8, 4, True, True, True),
        (16, 4, True, True, True),
        (24, 4, True, True, True),
        (40, 4, True, True, True),
        (64, 4, True, True, True),
        (80, 4, True, True, True),
        (81, 4, True, True, False),
        (8, 2, True, True, False),
        (8, 4, False, True, False),
        (8, 4, True, False, False),
    ],
)
def test_push_admission(monkeypatch, rows, world, push_buffer, capture, push):
    ca = object.__new__(CustomAllreduce)
    ca.disabled = False
    ca.world_size = world
    ca.sm70_tp4_push_buffer_ptrs = [1] * world if push_buffer else None
    ca._ptr = 1
    ca.close = lambda: None
    inp = torch.full((rows, 1024), 0x7F7F0001, dtype=torch.int32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)

    def push_bytes(ptr, source, out):
        assert ptr == ca._ptr and source.dtype == out.dtype == torch.uint8
        out.copy_(source.unsqueeze(0).expand_as(out))

    monkeypatch.setattr(ops, "sm70_tp4_push_allgather", push_bytes)
    out = ca.try_sm70_tp4_push_allgather(inp)
    if push:
        assert out is not None and torch.equal(out, inp.expand(4, -1, -1))
    else:
        assert out is None


@pytest.mark.parametrize("decode", [False, True])
def test_qsa_rank_major_push_matches_original_merge(monkeypatch, decode):
    # Real pack, merge and map in TRITON_INTERPRET, exact selector on CPU.
    k, rows, world = 512, 2, 4
    blocks = torch.arange(k, dtype=torch.int32).repeat(rows, 1)
    blocks[1, 127:] = -1
    scores = [torch.full((rows, k), rank, dtype=torch.float32) for rank in range(world)]
    for score in scores:
        score.view(torch.int32)[0, :5] = torch.tensor(
            [-(1 << 31), 0, 0x7F800000, 0x7F800001, 0x7F7F7F7F]
        )
    packed = [
        qsa._pack_index_shard_candidates(blocks, score, world, rank)
        for rank, score in enumerate(scores)
    ]
    reference = torch.stack(packed)
    calls = []

    def push(inp):
        calls.append("push")
        assert torch.equal(inp, packed[0])
        return reference.clone()

    def nccl(inp, dim):
        calls.append("nccl")
        assert dim == 0 and torch.equal(inp[0], packed[0])
        return reference

    group = SimpleNamespace(
        all_gather=nccl,
        device_communicator=SimpleNamespace(
            ca_comm=SimpleNamespace(try_sm70_tp4_push_allgather=push)
        ),
    )
    monkeypatch.setattr(distributed, "get_tp_group", lambda: group)
    kwargs = dict(select=_reference_select, use_triton=True)
    visible = (blocks >= 0).sum(1, dtype=torch.int32)
    expected = qsa._merge_sharded_index_blocks(
        blocks,
        scores[0],
        visible,
        world,
        0,
        k,
        all_gather=lambda _: reference,
        **kwargs,
    )
    with set_forward_context(None, _forward_config(), is_uniform_decode=decode):
        actual = qsa._merge_sharded_index_blocks(
            blocks, scores[0], visible, world, 0, k, **kwargs
        )
    assert calls == ["push" if decode else "nccl"]
    assert torch.equal(actual, expected)


def _forward_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        ),
    )


def test_mtp_first_forward_gets_capture_phase(monkeypatch):
    manager = object.__new__(PrefillEagleCudaGraphManager)
    manager.max_num_reqs = 1
    manager.dp_size = 1
    spec = object.__new__(EagleSpeculator)
    spec.supports_mm_inputs = False
    spec.vllm_config = _forward_config()
    spec.hidden_states = torch.empty(5)
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.empty(5), positions=torch.empty(5)
    )
    observed = []

    def model(**kwargs):
        context = get_forward_context()
        observed.append((context.is_uniform_decode, context.cudagraph_runtime_mode))
        return kwargs["hidden_states"]

    spec.model = model
    # Mixed/prefill and decode descriptors have identical token and request counts.
    descs = [BatchExecutionDescriptor(CUDAGraphMode.FULL, 5, 1, u) for u in (None, 5)]

    def capture(self, create_forward_fn, progress_bar_desc):
        for desc in descs:
            forward, _ = create_forward_fn(desc)
            forward(CUDAGraphMode.NONE)

    monkeypatch.setattr(CudaGraphManager, "capture", capture)
    manager.capture(
        lambda *args, **kwargs: spec.prefill(*args, **kwargs, skip_sample=True),
        {desc: CapturedAttentionState(None, {}) for desc in descs},
    )
    spec.prefill(1, 5, None, {}, None, skip_sample=True)
    assert observed == [
        (False, CUDAGraphMode.NONE),
        (True, CUDAGraphMode.NONE),
        (False, CUDAGraphMode.NONE),
    ]


def _tp4_worker(rank: int, rendezvous: str) -> None:
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    nccl = dist.new_group(backend="nccl", timeout=timedelta(seconds=60))
    group = dist.group.WORLD
    assert group is not None
    ca = CustomAllreduce(group, rank)
    try:
        if ca.sm70_tp4_push_buffer_ptrs is None:
            # In particular, a SYS diagonal cannot open the TP4 push mailbox.
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                assert (
                    ca.try_sm70_tp4_push_allgather(
                        torch.zeros(32, dtype=torch.uint8, device="cuda")
                    )
                    is None
                )
            return
        dist.all_reduce(torch.zeros(1, device="cuda"), group=nccl)
        torch.accelerator.synchronize()
        fixtures = {}
        for size in (4096, 4097, 8 * 4096, 40 * 4096, 80 * 4096, 81 * 4096):
            inp = torch.zeros(size, dtype=torch.uint8, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = ca.try_sm70_tp4_push_allgather(inp)
                if out is None:
                    assert size > 320 * 1024
                    out = torch.empty((4, size), dtype=inp.dtype, device=inp.device)
                    dist.all_gather_into_tensor(out.view(-1), inp, group=nccl)
            fixtures[size] = inp, out, graph
        # Consecutive generations with shrink, growth, fallback and a byte tail.
        for step, size in enumerate((80, 1, 81, 8, 40, 80, 1, 8, 80, 40, 80)):
            byte_count = size * 4096 + (step == 6)
            inp, out, graph = fixtures[byte_count]
            inp.copy_(
                (torch.arange(byte_count, device="cuda") + 17 * rank + step) % 256
            )
            graph.replay()
            expected = torch.empty_like(out)
            dist.all_gather_into_tensor(expected.view(-1), inp, group=nccl)
            assert torch.equal(out, expected), (rank, step, byte_count)
    finally:
        ca.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="needs four SM70 GPUs")
def test_tp4_push_cuda_graph_matches_nccl(tmp_path: Path) -> None:
    if any(torch.cuda.get_device_capability(i) != (7, 0) for i in range(4)):
        pytest.skip("needs SM70 GPUs")
    torch.multiprocessing.spawn(
        _tp4_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4
    )
