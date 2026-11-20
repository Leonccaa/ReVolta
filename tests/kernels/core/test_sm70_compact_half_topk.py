# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


def compact_half_topk(logits):
    rows, width = logits.shape
    if width <= 65535:
        partial = torch.empty(
            (rows, ((width + 1023) // 1024) * 64),
            dtype=torch.uint32,
            device=logits.device,
        )
    else:
        w1 = ((width + 1023) // 1024) * 64
        w2 = ((w1 + 1023) // 1024) * 64
        partial = torch.empty(
            (rows, w1 + w2),
            dtype=torch.int64,
            device=logits.device,
        )
    values = torch.empty((rows, 64), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, 64), dtype=torch.int64, device=logits.device)
    ops.sm70_compact_half_topk_out(logits, partial, values, ids, 64)
    return values, ids


@pytest.mark.parametrize(
    "rows,width",
    [
        (8, 62080),
        (32, 65535),
        (40, 62080),
        (64, 65535),
        (8, 1025),
        (8, 65536),
        (8, 124160),
        (16, 124160),
        (8, 248320),
        (5, 524288),
    ],
)
@torch.inference_mode()
def test_half_source_values_and_live_graph(rows, width):
    torch.manual_seed(1530)
    logits = torch.randn(rows, width, dtype=torch.float16, device="cuda")
    assert compact_half_topk(logits) is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = compact_half_topk(logits)
    for case in ("random", "ties", "zeros", "nonfinite"):
        logits.normal_()
        if case == "ties":
            logits.round_()
        elif case == "zeros":
            logits.zero_()
            logits[:, ::2] = -0.0
        elif case == "nonfinite":
            logits[:, 0] = float("nan")
            logits[:, 1] = float("inf")
            logits[:, 2] = -float("inf")
        graph.replay()
        values, ids = selected
        expected = logits.float().topk(64, dim=-1).values
        torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
        original = logits.gather(1, ids).float()
        torch.testing.assert_close(values, original, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(values)
        assert torch.equal(
            values.view(torch.int32)[finite], original.view(torch.int32)[finite]
        )
        assert all(len(set(row)) == 64 for row in ids.cpu().tolist())
