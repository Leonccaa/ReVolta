# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("rows", [31, 32, 33, 64, 65])
def test_prefill_workspace_reuse_preserves_tiled_output_and_graph(rows):
    torch.manual_seed(9683)
    history = torch.randn(16, 2, 256, 1, 256, dtype=torch.float16, device="cuda")
    step = 16
    elements = 64 * 6 * 33
    old_workspace = (
        torch.empty(elements * 256, dtype=torch.float32, device="cuda"),
        torch.empty(2 * elements, dtype=torch.float32, device="cuda"),
    )
    workspace = tuple(t[: t.numel() // (64 // step)] for t in old_workspace)
    workspace = tuple(torch.empty_like(t) for t in workspace)
    query = torch.randn(rows, 6, 256, dtype=torch.float16, device="cuda")
    gate = torch.randn_like(query)
    table = torch.arange(16, dtype=torch.int32, device="cuda").repeat(2, 1)
    requests = torch.arange(rows, dtype=torch.int32, device="cuda") % 2
    requests[-1] = -1
    positions = torch.full((rows,), 3790, dtype=torch.int64, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    groups = torch.randperm(947, device="cuda")[:512].int().sort().values
    selected = (groups[:, None] * 4 + torch.arange(4, device="cuda")).flatten()
    indices = torch.cat([selected, torch.tensor([3788, 3789, 3790], device="cuda")])
    indices = indices.int().repeat(rows, 1)
    scales = torch.ones(2, dtype=torch.float32, device="cuda")
    actual, expected = torch.empty_like(query), torch.empty_like(query)

    def run():
        for output, scratch, step in (
            (actual, workspace, 16),
            (expected, old_workspace, 64),
        ):
            ops.sm70_qsa_device_run_tiled(
                query,
                history,
                scales,
                indices,
                table,
                requests,
                positions,
                lengths,
                output,
                gate,
                *scratch,
                step,
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        query.mul_(0.75)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected)
