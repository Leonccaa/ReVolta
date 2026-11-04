# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HC and PLE Triton kernels must form row addresses in int64.

`row * stride` computed in int32 wraps once it passes 2**31 elements (for
the HC residual stride of 10240 that is row 209716; for a 12800-wide PLE row
it is row 167773). These tests run the kernels in the Triton CPU interpreter
on a strided view whose row stride puts row 2 past 2**31 elements; the
backing storage is allocated lazily, so only the touched pages are real.

Run with: TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= pytest <this file>
"""

import os

import pytest

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "Triton CPU interpreter test (set TRITON_INTERPRET=1)",
        allow_module_level=True,
    )

import torch

from vllm.models.qwen4_exp.nvidia import ple_layer, ple_nvfp4
from vllm.models.qwen4_exp.nvidia.ops import hc

ROWS = 3
# Row 2 starts at 2 * (2**30 + 64) = 2**31 + 128 elements: past int32.
BIG_STRIDE = 2**30 + 64


def _big_rows(values: torch.Tensor) -> torch.Tensor:
    """A [ROWS, D] view with row stride BIG_STRIDE holding `values`."""
    rows, dim = values.shape
    assert rows == ROWS
    storage = torch.empty((ROWS - 1) * BIG_STRIDE + dim, dtype=values.dtype)
    view = storage.as_strided((rows, dim), (BIG_STRIDE, 1))
    view.copy_(values)
    return view


def _rand(rows: int, dim: int, dtype=torch.float16) -> torch.Tensor:
    gen = torch.Generator().manual_seed(rows * 1000 + dim)
    return torch.randn(rows, dim, generator=gen).to(dtype)


def test_grouped_gemma_rmsnorm_big_row_offset():
    dim, groups = 64, 4
    x = _rand(ROWS, dim)
    w = _rand(1, dim // groups).reshape(-1).contiguous()

    def run(src):
        y = torch.empty(ROWS, dim, dtype=torch.float16)
        hc._grouped_gemma_rmsnorm_kernel[(ROWS * groups,)](
            src,
            w,
            y,
            src.stride(0),
            y.stride(0),
            dim,
            groups,
            W_SHARED=True,
            EPS=1e-6,
            launch_pdl=False,
        )
        return y

    torch.testing.assert_close(run(_big_rows(x)), run(x), rtol=0, atol=0)


def test_hc_silu_big_row_offset():
    dim = 64
    x = _rand(ROWS, dim)

    def run(src):
        y = torch.empty(ROWS, dim, dtype=torch.float16)
        hc._hc_silu_kernel[(ROWS,)](
            src, y, src.stride(0), y.stride(0), DIM=dim, HC=4, launch_pdl=False
        )
        return y

    torch.testing.assert_close(run(_big_rows(x)), run(x), rtol=0, atol=0)


def test_hc_gate_mix_big_row_offset():
    dim, count, block = 64, 4, 16
    x, g = _rand(ROWS, dim), _rand(ROWS, dim)

    def run(xs, gs):
        y = torch.empty(ROWS, dim // count, dtype=torch.float16)
        hc._hc_gate_mix_kernel[(ROWS, 1)](
            xs,
            gs,
            y,
            xs.stride(0),
            gs.stride(0),
            y.stride(0),
            dim,
            count,
            block,
            launch_pdl=False,
        )
        return y

    torch.testing.assert_close(
        run(_big_rows(x), _big_rows(g)), run(x, g), rtol=0, atol=0
    )


@pytest.mark.parametrize("fused_norm", [False, True])
def test_hc_combine_big_row_offset(fused_norm):
    count, hc_dim, block = 4, 16, 16
    dim = count * hc_dim
    res, blk, inj = _rand(ROWS, dim), _rand(ROWS, hc_dim), _rand(ROWS, count)
    w = _rand(1, hc_dim).reshape(-1).contiguous()

    def run(r):
        out = torch.empty(ROWS, dim, dtype=torch.float16)
        if not fused_norm:
            hc._hc_combine_kernel[(ROWS, 1)](
                blk,
                r,
                inj,
                out,
                blk.stride(0),
                r.stride(0),
                inj.stride(0),
                out.stride(0),
                hc_dim,
                count,
                block,
                launch_pdl=False,
            )
            return out
        y = torch.empty(ROWS, dim, dtype=torch.float16)
        hc._hc_combine_norm_kernel[(ROWS, count)](
            blk,
            r,
            inj,
            w,
            out,
            y,
            blk.stride(0),
            r.stride(0),
            inj.stride(0),
            out.stride(0),
            y.stride(0),
            hc_dim,
            count,
            W_SHARED=True,
            EPS=1e-6,
            BLOCK_SIZE=block,
            launch_pdl=False,
        )
        return torch.cat([out, y], dim=1)

    torch.testing.assert_close(run(_big_rows(res)), run(res), rtol=0, atol=0)


def _big_table(dim: int) -> tuple[torch.Tensor, list[int]]:
    """A lazily allocated uint8 table past 2**31 bytes and row ids around it."""
    rows = 2**31 // dim + 2
    table = torch.empty(rows * dim, dtype=torch.uint8)
    ids = [0, 2**31 // dim - 1, 2**31 // dim + 1]
    gen = torch.Generator().manual_seed(dim)
    for i in ids:
        table[i * dim : (i + 1) * dim] = torch.randint(
            0, 120, (dim,), generator=gen, dtype=torch.uint8
        )
    return table, ids


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_ple_fp8_gather_big_table_offset(ids_dtype):
    dim = 128
    table, ids = _big_table(dim)
    scale = torch.tensor([0.5], dtype=torch.float32)
    out = torch.empty(len(ids), dim, dtype=torch.float32)
    ple_layer._gather_ple_fp8_from_pinned_kernel[(len(ids),)](
        table,
        torch.tensor(ids, dtype=ids_dtype),
        scale,
        out,
        embedding_dim=dim,
        BLOCK_D=dim,
    )
    for row, i in enumerate(ids):
        raw = table[i * dim : (i + 1) * dim].view(torch.float8_e4m3fn)
        torch.testing.assert_close(out[row], raw.float() * 0.5, rtol=0, atol=0)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_ple_nvfp4_gather_big_table_offset(ids_dtype):
    row_bytes = 72
    table, ids = _big_table(row_bytes)
    out = torch.empty(len(ids), row_bytes, dtype=torch.uint8)
    ple_nvfp4._gather_rows[(len(ids),)](
        table, torch.tensor(ids, dtype=ids_dtype), out, row_bytes, 128
    )
    for row, i in enumerate(ids):
        assert torch.equal(out[row], table[i * row_bytes : (i + 1) * row_bytes])
