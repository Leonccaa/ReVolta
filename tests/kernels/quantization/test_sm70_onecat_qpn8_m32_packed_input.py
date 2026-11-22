# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops


@pytest.mark.parametrize("rows", [9, 15, 16, 17, 24, 31, 32, 33, 35, 39, 40])
@pytest.mark.parametrize(
    "k,n,split",
    [(1536, 5120, 12), (4096, 2048, 16), (4352, 5120, 16), (5120, 3584, 16)],
)
def test_qpn8_m32_changed_input_graph(rows, k, n, split):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")

    torch.manual_seed(263)
    weight = (torch.randn(n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    scales = torch.rand(n, 1, device="cuda") * 0.005 + 0.002
    codes, packed_scales = torch.ops._C.fp8_qpn8_prepare_sm70(weight, scales)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty(rows, n, device="cuda", dtype=x.dtype)
    reference = torch.empty_like(out)

    def call():
        ops.fp8_qpn8_gemm_v154_sm70_out(
            out, x, codes, packed_scales, split, 2, True, False
        )

    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for magnitude in (0.1, 0.3, 0.01):
        x.normal_(0, magnitude)
        for start in range(0, rows, 8):
            torch.ops._C.fp8_qpn8_gemm_sm70_out(
                reference[start : start + 8],
                x[start : start + 8],
                codes,
                packed_scales,
                split,
                2,
                True,
                False,
            )
        # Both eager and a replay must consume the new input, including tail
        # rows. Compare FP16 bits, so signed zero differences are visible too.
        for invoke in (call, graph.replay):
            out.fill_(float("nan"))
            invoke()
            assert torch.isfinite(out).all()
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
