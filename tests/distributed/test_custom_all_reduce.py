# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float32, True),
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.int8, False),
        (torch.float8_e4m3fn, False),
    ],
)
def test_custom_allreduce_filters_dtype(
    dtype: torch.dtype,
    expected: bool,
) -> None:
    communicator = CustomAllreduce.__new__(CustomAllreduce)
    communicator.disabled = False
    communicator.tp8_hierarchical = False
    communicator.world_size = 2
    communicator.fully_connected = True
    communicator.dispatch_max_size = 1024
    communicator._ptr = 0

    assert communicator.should_custom_ar(torch.empty(16, dtype=dtype)) is expected


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for i in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1)
                torch.testing.assert_close(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1
        sz = 1024
        fa = get_tp_group().device_communicator.ca_comm
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)


def _qsa_gather_bytes_status(source: bytes, final: bytes) -> str:
    if len(final) != len(source):
        return "INCOMPLETE"
    return "PASS" if final == source else "FAIL"


@ray.remote(num_gpus=1, max_calls=1)
def qsa_candidate_allgather(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE", "1")
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        tp = get_tp_group()
        ca = tp.device_communicator.ca_comm
        assert not ca.disabled and ca.sm70_nvlink_tp4
        assert ca.sm70_tp4_push_buffer_ptrs is not None
        select = qsa._sm70_qsa_lexicographic_topk_op()
        moe = torch.ones(1, 2560, dtype=torch.float16, device=device)

        def make_step(rows):
            scores = torch.empty(rows, 2051, device=device).random_(0, 4)
            lengths = torch.tensor(
                [[2051, 0, 512, 1025, 300][r % 5] for r in range(rows)],
                dtype=torch.int32,
                device=device,
            )
            blocks = torch.empty(rows, 512, dtype=torch.int32, device=device)

            def step():
                select(scores, lengths, blocks, 512)
                source = qsa._pack_index_shard_candidates(
                    blocks, scores, 4, rank
                ).unsqueeze(0)
                outputs, reductions = [], []
                for _ in range(13):
                    final = ca.sm70_qsa_index_allgather(source)
                    if rows <= 40:
                        assert final is not None
                    else:
                        assert final is None
                        final = tp.all_gather(source, dim=0)
                    outputs.append(final)
                    reductions.append(tensor_model_parallel_all_reduce(moe))
                merged = qsa._merge_sharded_index_blocks(
                    blocks,
                    scores,
                    lengths,
                    4,
                    rank,
                    512,
                    all_gather=lambda _: outputs[-1],
                    use_triton=True,
                )
                return source, outputs, reductions, merged

            def check(result):
                source, outputs, reductions, merged = result
                reference = tp.all_gather(source, dim=0)
                expected = qsa._merge_sharded_index_blocks(
                    blocks,
                    scores,
                    lengths,
                    4,
                    rank,
                    512,
                    all_gather=lambda _: reference,
                    use_triton=True,
                )
                assert torch.equal(merged, expected)
                raw_source = reference.cpu().numpy().tobytes()
                for final, reduction in zip(outputs, reductions):
                    assert torch.equal(
                        final.view(torch.uint8), reference.view(torch.uint8)
                    )
                    assert torch.equal(reduction, moe * 4)
                    raw_final = final.cpu().numpy().tobytes()
                    assert _qsa_gather_bytes_status(raw_source, raw_final) == "PASS"
                # D68: mutate only copies of the real final; retain real source/final.
                changed = bytearray(raw_final)
                changed[0] ^= 1
                assert _qsa_gather_bytes_status(raw_source, bytes(changed)) == "FAIL"
                assert (
                    _qsa_gather_bytes_status(raw_source, raw_final[:-4096])
                    == "INCOMPLETE"
                )

            return scores, step, check

        graphs = {}
        for rows in (1, 5, 10, 40, 41):
            scores, step, check = make_step(rows)
            check(step())  # Real pack/merge compilation and NCCL warmup before capture.
            with graph_capture(device=device) as context:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=context.stream):
                    result = step()
            graphs[rows] = (scores, graph, result, check)
        # Odd replay counts, changed candidates, and shared-channel shrink/grow.
        for rows in (10, 1, 5, 10, 40, 41):
            scores, graph, result, check = graphs[rows]
            for _ in range(3):
                scores.random_(0, 4)
                graph.replay()
                check(result)


def test_qsa_candidate_allgather_tp4(monkeypatch):
    from vllm.platforms import current_platform

    if torch.accelerator.device_count() < 4:
        pytest.skip("Not enough GPUs to run the test.")
    if not current_platform.is_cuda() or not current_platform.is_device_capability(70):
        pytest.skip("Needs four fully connected NVLink SM70 GPUs.")
    multi_process_parallel(monkeypatch, 4, 1, qsa_candidate_allgather)
