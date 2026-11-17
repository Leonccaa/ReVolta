# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""Sparse QSA attention must not recompile per decode row count.

``num_rows`` is the query-row count of the batch and used to enter the
specialization key of the split-K partial kernel, which also indexes its
workspace by ``(split * num_rows + row)``; the merge kernel adds the same
count again. Each new padded decode size therefore paid a cold compile inside
inference. Both kernels now take the count as an unspecialized runtime scalar.

The gates run the partial kernel in the Triton CPU interpreter over row counts
from the three specialization classes (one, a multiple of 16, neither) and
check the result against a torch attention reference. The merge kernel itself
cannot run under the interpreter (numpy rejects its scalar-and-vector mask),
so the split layout it consumes is verified by merging the partials in torch
and comparing against the single-split run.
"""

import math
import os

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.triton_utils import triton

pytestmark = pytest.mark.skip_global_cleanup

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreter = pytest.mark.skipif(
    not INTERPRETED, reason="Triton kernels need CUDA or TRITON_INTERPRET=1"
)
COUNTS = (1, 2, 3, 15, 16, 17, 33)
compiled_only = pytest.mark.skipif(
    INTERPRETED, reason="reads the compiled JIT parameter metadata"
)
DIM, GROUP, PAGE, TOPK, REQUESTS = 32, 16, 16, 32, 2


@compiled_only
def test_row_counts_are_unspecialized_runtime_scalars():
    for kernel, names in (
        (
            qsa_ops._qsa_sparse_paged_gqa_splitk_kernel,
            ["num_requests", "num_rows"],
        ),
        (qsa_ops._qsa_merge_splitk_kernel, ["num_rows"]),
    ):
        params = {p.name: p for p in kernel.params}
        for name in names:
            assert not params[name].is_constexpr, f"{kernel.fn.__name__}.{name}"
            assert params[name].do_not_specialize, f"{kernel.fn.__name__}.{name}"


def _inputs(rows, seed=31):
    generator = torch.Generator().manual_seed(seed)
    table = torch.arange(REQUESTS * 4, dtype=torch.int32).reshape(REQUESTS, 4)

    def values(shape):
        return (torch.randn(shape, generator=generator) * 0.3).half()

    q = values((rows, GROUP, DIM))
    key = values((REQUESTS * 4, PAGE, 1, DIM))
    value = values((REQUESTS * 4, PAGE, 1, DIM))
    indices = torch.randint(
        0, PAGE * 4, (rows, TOPK), generator=generator, dtype=torch.int32
    )
    token_to_req = torch.arange(rows, dtype=torch.int32) % REQUESTS
    return q, key, value, indices, table, token_to_req


def _launch(rows, splits, tensors, block_n=16):
    q, key, value, indices, table, token_to_req = tensors
    heads = q.shape[1]
    tiles = triton.cdiv(TOPK, block_n)
    out = torch.zeros(rows, heads, DIM, dtype=torch.float16)
    lse = torch.full((rows, heads), float("nan"), dtype=torch.float32)
    if splits == 1:
        partial_output = partial_lse = out
    else:
        partial_output = torch.zeros(splits, rows, heads, DIM, dtype=torch.float32)
        partial_lse = torch.zeros(splits, rows, heads, dtype=torch.float32)
    qsa_ops._qsa_sparse_paged_gqa_splitk_kernel[(rows, key.shape[2], splits)](
        q,
        key,
        value,
        indices,
        table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        lse,
        None,
        q.stride(0),
        q.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        indices.stride(0),
        table.stride(0),
        out.stride(0),
        out.stride(1),
        0,
        0,
        rows,
        key.shape[0],
        table.shape[0],
        1.0,
        1.0,
        TOPK=TOPK,
        PAGE_SIZE=PAGE,
        PAGE_TABLE_WIDTH=table.shape[1],
        GROUP_SIZE=GROUP,
        HEAD_DIM=DIM,
        NUM_QUERY_HEADS=heads,
        NUM_SPLITS=splits,
        NUM_TILES=tiles,
        BLOCK_M=triton.next_power_of_2(GROUP),
        BLOCK_N=block_n,
        KV_E4M3=False,
        num_warps=4,
    )
    return out, lse, partial_output, partial_lse


def _torch_attention(tensors):
    q, key, value, indices, table, token_to_req = tensors
    rows, heads, dim = q.shape
    keys = key.float().reshape(-1, DIM)
    values = value.float().reshape(-1, DIM)
    out = torch.zeros(rows, heads, dim, dtype=torch.float32)
    lse = torch.zeros(rows, heads, dtype=torch.float32)
    for row in range(rows):
        request = int(token_to_req[row])
        slots = [
            int(table[request, column // PAGE]) * PAGE + column % PAGE
            for column in indices[row].tolist()
        ]
        # The kernel scales by 1/sqrt(d) and works in the base-2 exponent.
        logits = torch.einsum("hd,cd->hc", q[row].float(), keys[slots]) * (
            1.4426950408889634 / math.sqrt(dim)
        )
        base = logits.max(dim=-1, keepdim=True).values
        weights = torch.exp2(logits - base)
        total = weights.sum(dim=-1, keepdim=True)
        out[row] = torch.einsum("hc,cd->hd", weights / total, values[slots])
        lse[row] = base.squeeze(-1) + torch.log2(total.squeeze(-1))
    return out, lse


@interpreter
@pytest.mark.parametrize("rows", COUNTS)
def test_sparse_single_split_matches_torch(rows):
    tensors = _inputs(rows)
    out, lse, _, _ = _launch(rows, 1, tensors)
    expected, expected_lse = _torch_attention(tensors)
    assert torch.allclose(out.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, expected_lse, atol=2e-2, rtol=2e-2)


@interpreter
@pytest.mark.parametrize("rows", COUNTS)
@pytest.mark.parametrize("splits", (2,))
def test_sparse_split_workspace_merges_to_the_single_split_result(rows, splits):
    tensors = _inputs(rows)
    direct, direct_lse, _, _ = _launch(rows, 1, tensors)
    _, _, partial_output, partial_lse = _launch(rows, splits, tensors)
    assert partial_output.shape == (splits, rows, GROUP, DIM)
    assert partial_lse.shape == (splits, rows, GROUP)
    # A split with no tile of its own reports the empty-split LSE, -inf; no
    # split may produce a NaN.
    assert not torch.isnan(partial_lse).any()
    assert (partial_output.abs() < 100).all()
    maximum = partial_lse.nan_to_num(-float("inf")).max(dim=0, keepdim=True).values
    empty = ~torch.isfinite(partial_lse)
    weights = torch.where(
        empty, torch.zeros_like(partial_lse), torch.exp2(partial_lse - maximum)
    )
    total = weights.sum(0, keepdim=True)
    merged = (partial_output * weights.unsqueeze(-1)).sum(0) / total.squeeze(
        0
    ).unsqueeze(-1)
    merged_lse = (maximum + torch.log2(total)).squeeze(0)  # fmt: skip
    assert torch.allclose(merged, direct.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(merged_lse, direct_lse, atol=2e-2, rtol=2e-2)
