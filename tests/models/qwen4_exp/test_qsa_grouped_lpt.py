# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LPT launch order for the QSA DCP grouped page4 route must not change any
value: permute -> per-group kernel -> restore equals the kernel in row order.

The real kernel is CUDA-only; this uses a CPU stand-in that, like the real
one, reads only its own group's q rows, pages, masks and length and writes
only its own group's output and LSE rows.
"""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa

Q = qsa._SM70_QSA_GROUPED_PAGE4_QUERIES


def fake_kernel(q, pages, masks, lens):
    groups = lens.shape[0]
    out = torch.empty(q.shape, dtype=torch.float16)
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32)
    for g in range(groups):
        n = int(lens[g]) // 4
        bits = masks[g, :n].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        w = (pages[g, :n].double() * 1e-3 + bits.double() * 1e-10).sum()
        rows = slice(g * Q, (g + 1) * Q)
        out[rows] = (q[rows].double() * (1.0 + w)).to(torch.float16)
        lse[rows] = torch.where(
            torch.tensor(n == 0),
            torch.tensor(-1e30),
            (q[rows].double().sum(-1) + w).float(),
        )
    return out, lse


@pytest.mark.parametrize("groups", [1, 2, 7, 206])
@pytest.mark.parametrize("ties", [False, True])
def test_lpt_roundtrip_is_bit_identical(groups, ties):
    g = torch.Generator().manual_seed(groups * 2 + ties)
    rows, heads, dim, cap = groups * Q, 12, 16, 64
    q = torch.randn(rows, heads, dim, generator=g).to(torch.float16)
    pages = torch.randint(0, 1000, (groups, cap), generator=g, dtype=torch.int32)
    masks = torch.randint(0, 2**32, (groups, cap), generator=g, dtype=torch.int64).to(
        torch.uint32
    )
    if ties:
        lens = torch.full((groups,), 32, dtype=torch.int32)
        lens[::3] = 0
    else:
        lens = (torch.randint(0, cap // 8 + 1, (groups,), generator=g) * 32).to(
            torch.int32
        )
    order, pp, mm, ll, qp = qsa._qsa_grouped_lpt_permute(pages, masks, lens, q)
    assert mm.dtype == torch.uint32
    assert torch.equal(mm.view(torch.int32), masks.view(torch.int32)[order])
    assert torch.equal(ll, lens.sort(descending=True, stable=True).values)
    for head in (0, 6):
        ref_out, ref_lse = fake_kernel(
            q[:, head : head + 6].contiguous(), pages, masks, lens
        )
        got_out, got_lse = fake_kernel(qp[:, head : head + 6].contiguous(), pp, mm, ll)
        res_out, res_lse = qsa._qsa_grouped_lpt_restore(order, got_out, got_lse)
        assert torch.equal(res_out, ref_out)
        assert torch.equal(res_lse, ref_lse)
