# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("rows", [9, 16, 17, 24, 32, 40, 48, 64])
def test_qpn2_batched_gated_and_k16_handoff_match_eager(rows):
    # Reuse the native layout fixture from 1Cat's QPN2 MTP tests. The second
    # projection consumes the gated result's K16 twin inside the graph.
    torch.manual_seed(62147 + rows)
    dev = "cuda"
    width, hidden, cols = 2560, 1024, 1056
    scale = 0.0001

    def prepare(k: int, n: int):
        packed = torch.randint(0, 256, (n, k // 2), device=dev, dtype=torch.uint8)
        scale_bytes = torch.randint(0, 127, (n, k // 16), device=dev, dtype=torch.uint8)
        return torch.ops._C.nvfp4_qpn2_prepare_sm70(
            packed, scale_bytes.view(torch.float8_e4m3fn)
        )

    gate_codes, gate_scales = prepare(width, 2 * hidden)
    out_codes, out_scales = prepare(hidden, cols)
    x = torch.randn(rows, width, device=dev, dtype=torch.float16) * 0.1
    mid = x.new_empty((rows, hidden))
    out = x.new_empty((rows, cols))
    ref_mid, reference = torch.empty_like(mid), torch.empty_like(out)

    def run():
        ops.nvfp4_qpn2_batch_sm70_out(mid, x, gate_codes, gate_scales, scale, True)
        ops.nvfp4_qpn2_batch_sm70_out(out, mid, out_codes, out_scales, scale, False)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for amplitude in (0.1, 0.04, 0.26):
        x.normal_().mul_(amplitude)
        graph.replay()
        for first in range(0, rows, 8):
            block = slice(first, first + 8)
            torch.ops._C.nvfp4_qpn2_gated_sm70_out(
                ref_mid[block], x[block], gate_codes, gate_scales, scale, 8, 2
            )
            torch.ops._C.nvfp4_qpn2_gemm_sm70_out(
                reference[block], ref_mid[block], out_codes, out_scales, scale, 8, 2
            )
        torch.testing.assert_close(mid, ref_mid, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(out, reference, rtol=2e-3, atol=2e-3)
