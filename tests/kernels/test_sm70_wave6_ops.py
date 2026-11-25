# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("rows", (2, 5, 10, 20, 40, 64))
@pytest.mark.parametrize("row_major", (False, True))
def test_qsa_input_batch_cudagraph_parity(rows: int, row_major: bool):
    torch.manual_seed(6100 + rows * 3 + int(row_major))
    dev = torch.device("cuda")
    qw = torch.randn(3584, 2560, device=dev, dtype=torch.float16) * 0.02
    iw = torch.randn(640, 2560, device=dev, dtype=torch.float16) * 0.02

    def pack(weight: torch.Tensor) -> torch.Tensor:
        return weight.view(-1, 32, 160, 2, 8).permute(0, 2, 3, 1, 4).contiguous()

    pq, pi = pack(qw), pack(iw)
    x = torch.randn(rows, 2560, device=dev, dtype=torch.float16)

    s_q = torch.full((rows * 3584 + 16,), 19, device=dev, dtype=torch.float16)
    s_i = torch.full((rows * 640 + 16,), 23, device=dev, dtype=torch.float16)
    out_q = s_q[8:-8].view(rows, 3584)
    out_i = s_i[8:-8].view(rows, 640)
    w_q, w_i = (qw, iw) if row_major else (pq, pi)

    ops.qwen38_qsa_input_batch_sm70_out(out_q, out_i, x, w_q, w_i)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ops.qwen38_qsa_input_batch_sm70_out(out_q, out_i, x, w_q, w_i)
    x.normal_(0, 0.7)
    out_q.fill_(float("nan"))
    out_i.fill_(float("nan"))
    g.replay()
    assert torch.all(s_q[:8] == 19) and torch.all(s_q[-8:] == 19)
    assert torch.all(s_i[:8] == 23) and torch.all(s_i[-8:] == 23)
    q_ref = (x.float() @ qw.float().T).half()
    i_ref = (x.float() @ iw.float().T).half()
    torch.testing.assert_close(out_q, q_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out_i, i_ref, rtol=1e-3, atol=1e-3)
