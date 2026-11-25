# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("rows", [*range(1, 10), 17, 32, 33, 40, 64])
@pytest.mark.parametrize("vocab", [62080, 124160])
def test_masked_dense_head_graph_replay(rows, vocab):
    torch.manual_seed(49071)
    weight = torch.randn(vocab, 5120, device="cuda", dtype=torch.float16)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16)
    flags = torch.zeros(rows, device="cuda", dtype=torch.int32)
    scores = torch.empty(rows, vocab, device="cuda", dtype=torch.float16)
    operation = ops.sm70_fp16_masked_head_out
    operation(scores, x, weight, flags)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation(scores, x, weight, flags)
    for cycle in range(16):
        x.normal_().mul_(0.1 if cycle < 8 else 2)
        flags.copy_((torch.arange(rows, device="cuda") + cycle) % 2)
        scores.fill_(111)
        graph.replay()
        reference = torch.nn.functional.linear(x, weight)
        active = flags == 0
        assert torch.equal(
            scores[active].view(torch.int16), reference[active].view(torch.int16)
        )
        assert bool((scores[~active] == 111).all())
    flags.fill_(1)
    scores.fill_(42)
    graph.replay()
    assert bool((scores == 42).all())
