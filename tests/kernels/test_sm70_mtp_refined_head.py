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
def test_fp16_candidate_scores_preserve_precision_and_dense_fallback(rows, vocab):
    operation = ops.sm70_fp16_candidate_scores_out
    torch.manual_seed(71328)
    weight = torch.randn(vocab, 5120, device="cuda", dtype=torch.float16)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, vocab, (rows, 32), device="cuda", dtype=torch.int32)
    scores = torch.empty(rows, 32, device="cuda", dtype=torch.float16)
    operation(scores, x, weight, ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation(scores, x, weight, ids)
    for cycle in range(32):
        x.normal_().mul_(0.1 if cycle < 16 else 2)
        graph.replay()
        assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        assert not torch.backends.cuda.matmul.allow_fp16_accumulation
        dense = torch.nn.functional.linear(x, weight)
        expected = dense.gather(1, ids.long())
        assert torch.equal(scores.view(torch.int16), expected.view(torch.int16))


def test_fp16_candidate_scores_invalid_ids_are_bounded():
    operation = ops.sm70_fp16_candidate_scores_out
    x = torch.ones(1, 5120, device="cuda", dtype=torch.float16)
    weight = torch.ones(32, 5120, device="cuda", dtype=torch.float16)
    ids = torch.tensor([[-1, 0, 31, 32]], device="cuda", dtype=torch.int32)
    scores = torch.empty_like(ids, dtype=torch.float16)
    operation(scores, x, weight, ids)
    assert torch.isnan(scores[0, [0, 3]]).all()
    assert torch.equal(scores[0, 1:3], torch.full_like(scores[0, 1:3], 5120))
