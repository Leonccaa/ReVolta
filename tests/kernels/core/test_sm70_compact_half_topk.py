# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.sm70_compact_topk import (
    sm70_compact_half_topk_ids_partial_spec,
    sm70_compact_half_topk_partial_spec,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


def compact_half_topk(logits, k=64):
    rows, width = logits.shape
    shape, dtype = sm70_compact_half_topk_partial_spec(rows, width, k)
    partial = torch.empty(shape, dtype=dtype, device=logits.device)
    values = torch.empty((rows, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, k), dtype=torch.int64, device=logits.device)
    ops.sm70_compact_half_topk_out(logits, partial, values, ids, k)
    return values, ids


@pytest.mark.parametrize(
    "rows,width,k",
    [
        (8, 62080, 64),
        (32, 65535, 64),
        (40, 62080, 64),
        (64, 65535, 64),
        (8, 1025, 64),
        (8, 65536, 64),
        (8, 124160, 64),
        (16, 124160, 64),
        (8, 248320, 64),
        (5, 524288, 64),
        (8, 62080, 20),
        (16, 62080, 20),
        (32, 62080, 20),
        (64, 62080, 20),
        (128, 62080, 20),
        (8, 62080, 21),
        (8, 248320, 20),
        (16, 248320, 20),
        (8, 62016, 20),
    ],
)
@torch.inference_mode()
def test_half_source_values_and_live_graph(rows, width, k):
    if (k != 64 or rows > 64) and ops.sm70_compact_half_topk_wide_revision() < 2:
        pytest.skip("requires compact_half_topk revision >= 2")
    torch.manual_seed(1530)
    logits = torch.randn(rows, width, dtype=torch.float16, device="cuda")
    assert compact_half_topk(logits, k) is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = compact_half_topk(logits, k)
    for case in ("distinct", "random", "ties", "zeros", "nonfinite"):
        logits.normal_()
        if case == "distinct":
            cols = 17 + 5 * torch.arange(k, device=logits.device)
            logits[:, cols] = 16.0 + torch.arange(
                k, dtype=logits.dtype, device=logits.device
            )
        elif case == "ties":
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
        fp32 = logits.float()
        reference_values, reference_ids = fp32.topk(k, dim=-1)
        torch.testing.assert_close(
            values, reference_values, rtol=0, atol=0, equal_nan=True
        )
        if case == "distinct":
            assert torch.equal(ids, reference_ids)
        # Stable sort gives ascending token IDs at ties. Normalize signed zeros
        # and rank NaNs above infinity to match the native operator contract.
        limit = torch.finfo(torch.float32).max
        keys = torch.where(fp32.isnan(), float("inf"), fp32.clamp(-limit, limit))
        keys = keys.masked_fill(keys == 0, 0.0)
        expected_ids = torch.argsort(keys, descending=True, stable=True)[:, :k]
        assert torch.equal(ids, expected_ids)
        expected_values = fp32.gather(1, expected_ids)
        non_nan = ~expected_values.isnan()
        assert torch.equal(
            values.view(torch.int32)[non_nan],
            expected_values.view(torch.int32)[non_nan],
        )
        eager_values, eager_ids = compact_half_topk(logits, k)
        assert torch.equal(ids, eager_ids)
        assert torch.equal(values.view(torch.int32), eager_values.view(torch.int32))


@torch.inference_mode()
def test_qpn8_screening_ids_match_torch_at_tied_cutoff():
    rows, width, k = 8, 62080, 64
    logits = torch.zeros(rows, width + 32, dtype=torch.float16, device="cuda")[:, 32:]
    logits[:, 19::1024] = 2.0
    logits[:, 113::512] = 1.0
    shape, dtype = sm70_compact_half_topk_ids_partial_spec(rows, width, k)
    partial = torch.empty(shape, dtype=dtype, device=logits.device)
    ids = torch.empty((rows, k), dtype=torch.int32, device=logits.device)
    ops.sm70_compact_half_topk_ids_out(logits, partial, ids, k)
    expected = logits.topk(k, sorted=False).indices
    assert torch.equal(ids.long().sort().values, expected.sort().values)
