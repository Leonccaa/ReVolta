# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops

_GEOMETRY = {
    (1280, 5120): (16, 16),
    (1536, 5120): (32, 16),
    (5120, 1024): (32, 8),
    (8704, 5120): (32, 16),
    (5120, 4352): (32, 8),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "n,k,input_scale",
    [
        (1280, 5120, 1.0),
        (1536, 5120, 1.0),
        (5120, 1024, 1 / 256),
        (8704, 5120, 1 / 32),
        (5120, 4352, 0.1),
    ],
)
@torch.inference_mode()
def test_fp16_m8_replays_live_inputs(n: int, k: int, input_scale: float) -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("DFlash2 M8 is SM70 only")
    if not hasattr(torch.ops._C, "sm70_dflash2_fp16_m8_out"):
        pytest.fail("Build the source-complete SM70 extension")
    torch.manual_seed(20261004)
    layer = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.float16)
    # Use BF16 weight values within FP16 transport's range.
    layer.weight.copy_(layer.weight.bfloat16().half())
    original = layer.weight.clone()
    weight = layer.weight
    tile, warps = _GEOMETRY[n, k]
    slots = torch.arange(tile, device=weight.device)
    columns = (
        (slots & 3) | ((slots & 12) << 1) | ((slots & 16) >> 2)
        if tile == 32
        else (slots & 3) | ((slots & 4) << 1) | ((slots & 8) >> 1)
    )
    packed = (
        weight.view(n // tile, tile, k // 16, 16)
        .permute(0, 2, 1, 3)
        .index_select(2, columns)
        .contiguous()
    )
    x = torch.randn(8, k, device="cuda", dtype=torch.float16) * input_scale
    output = x.new_empty((8, n))
    ops.sm70_dflash2_fp16_m8_out(output, x, packed, tile, warps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.sm70_dflash2_fp16_m8_out(output, x, packed, tile, warps)
    for amplitude in [1.0, -0.5, 0.0, 2.0]:
        x.copy_(torch.randn_like(x) * (input_scale * amplitude))
        graph.replay()
        oracle = F.linear(x.float(), original.float())
        torch.testing.assert_close(output.float(), oracle, atol=2e-3, rtol=2e-3)
        assert torch.isfinite(output).all()
