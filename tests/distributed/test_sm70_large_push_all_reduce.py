# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm import _custom_ops as ops


def _check_large_push(rank: int, rendezvous: str) -> None:
    from ctypes import c_void_p

    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=180),
    )
    bytes_ = ops.sm70_tp4_push_m64_buffer_size()
    payload_bytes = 2 * 4 * 64 * 5120 * 2
    signal_bytes = bytes_ - payload_bytes
    pointers = CustomAllreduce.create_shared_buffer(bytes_)
    try:
        rt = CudaRTLibrary()
        rt.cudaMemset(c_void_p(pointers[rank]), 0, signal_bytes)
        rt.cudaMemset(c_void_p(pointers[rank] + signal_bytes), 0x7F, payload_bytes)
        torch.accelerator.synchronize()
        shapes = [(rows, 5120) for rows in (33, 40, 48, 51, 52, 56, 63, 64)]
        inputs = [
            torch.empty(shape, dtype=torch.float16, device="cuda") for shape in shapes
        ]
        storage = [
            torch.full((x.numel() + 16,), 123, dtype=x.dtype, device=x.device)
            for x in inputs
        ]
        outputs = [buf[8:-8].view_as(x) for buf, x in zip(storage, inputs)]
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            for _ in range(4):
                for x, y in zip(inputs, outputs):
                    ops.sm70_tp4_push_m64_out(y, x, pointers, rank)
        for cycle in range(12):
            torch.manual_seed(317 + rank + 11 * cycle)
            reference = []
            for x in inputs:
                if cycle == 5:
                    x.fill_((65504.0, 0.0001, -65504.0, 0.03125)[rank])
                    x.flatten()[1::2].mul_(-1)
                else:
                    x.normal_().mul_(0.03 * (cycle + 1))
                peers = [torch.empty_like(x, device="cpu") for _ in range(4)]
                dist.all_gather(peers, x.cpu())
                chunks = [peer.flatten().float().chunk(4) for peer in peers]
                reduced = []
                for part in range(4):
                    start = part if x.numel() * x.element_size() >= 512 * 1024 else 0
                    v = chunks[start][part]
                    for i in range(1, 4):
                        v = v + chunks[(start + i) % 4][part]
                    reduced.append(v)
                reference.append(torch.cat(reduced).half().view_as(x))
            dist.barrier()
            graph.replay()
            for actual, expected in zip(outputs, reference):
                assert torch.equal(
                    actual.cpu().view(torch.int16), expected.view(torch.int16)
                )
            for buf in storage:
                assert torch.all(buf[:8] == 123) and torch.all(buf[-8:] == 123)
        torch.accelerator.synchronize()
        dist.barrier()
    finally:
        for peer, ptr in enumerate(pointers):
            if peer != rank:
                ops.sm70_ring_close_mem_handle(ptr)
        CustomAllreduce.free_shared_buffer(pointers, rank=rank)
        dist.destroy_process_group()


def test_large_push_preserves_order_tail_and_graph_replay(tmp_path):
    if torch.accelerator.device_count() != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        pytest.skip("requires four V100 GPUs")
    if not all(
        torch.cuda.can_device_access_peer(i, j)
        for i in range(4)
        for j in range(4)
        if i != j
    ):
        pytest.skip("requires peer access between all four GPUs")
    rendezvous = (tmp_path / "large-push-rendezvous").as_uri()
    mp.spawn(_check_large_push, args=(rendezvous,), nprocs=4, join=True)
