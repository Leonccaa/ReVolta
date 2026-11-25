# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("rows", [2, 8, 40, 64])
@pytest.mark.parametrize("width", [2048, 4096, 9216])
def test_qsa_indexer_and_compact_topk_match_original(rows, width):
    # Adapt the shared-key replay fixture to exercise both new native entries.
    torch.manual_seed(921 + rows)
    dev = "cuda"
    q = torch.randn(rows, 4, 128, device=dev, dtype=torch.float16)
    cache = torch.randn(width // 16, 16, 1, 128, device=dev, dtype=torch.float16)
    table = torch.randperm(width // 16, device=dev, dtype=torch.int32)[None, :]
    requests = torch.zeros(rows, device=dev, dtype=torch.int32)
    positions = torch.full((rows,), width * 4 - 1, device=dev, dtype=torch.int64)
    lengths = torch.full((1,), width * 4, device=dev, dtype=torch.int32)
    args = q, cache, table, requests, positions, lengths, 4.0, 4
    scores = torch.empty(rows, width, device=dev, dtype=torch.float32)
    visible = torch.empty(rows, device=dev, dtype=torch.int32)
    expected_scores = torch.empty_like(scores)
    expected_visible = torch.empty_like(visible)
    actual = torch.empty(rows, 512, device=dev, dtype=torch.int32)
    expected = torch.empty_like(actual)
    workspace = [
        torch.empty(rows, cols, device=dev, dtype=torch.int32)
        for cols in (512, 512, 8192, 8192, 8)
    ]

    def candidate():
        ops.sm70_qsa_indexer_run_topk(actual, scores, visible, *workspace, *args)

    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for ties in (False, True):
        q.normal_()
        if ties:
            q.zero_()
        graph.replay()
        torch.ops.vllm_sm70_qsa_indexer.run(expected_scores, expected_visible, *args)
        torch.ops._C.qsa_lexicographic_topk(
            expected_scores, expected_visible, expected, 512, True
        )
        torch.testing.assert_close(visible, expected_visible, rtol=0, atol=0)
        torch.testing.assert_close(scores, expected_scores, rtol=2e-6, atol=2e-6)
        assert torch.equal(actual, expected)
        ops.qsa_lexicographic_topk_v154(scores, visible, actual, 512, True)
        assert torch.equal(actual, expected)
