# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-layer attention comparison through the original kernels on CPU."""

import os

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv import (
    ActiveHostKV,
    HostKVConfig,
    LayerScales,
    UnsupportedHostKVConfig,
)
from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="The original QSA Triton kernels require the CPU interpreter here",
)


@pytest.fixture(autouse=True)
def _interpreter_boolean_comparison(monkeypatch):
    # Triton 3.6's interpreter binary_op preserves the floating operand dtype
    # for fcmpOGT. Scalar comparisons then contain FP32 0/1 instead of bool,
    # and the unchanged QSA merge's `split_mask & has_values` raises in NumPy.
    # Repair only the interpreter's comparison result, not kernel arithmetic.
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    from vllm.triton_utils import tl

    def greater(_builder, lhs, rhs):
        return TensorHandle(np.greater(lhs.data, rhs.data), tl.int1)

    monkeypatch.setattr(InterpreterBuilder, "create_fcmpOGT", greater)


def _case(e4m3, width, *, rows=3):
    r, t, h, d = 2, 512, 2, 16
    dtype = torch.uint8 if e4m3 else torch.float16
    scales = LayerScales(0.375, 1.75)
    config = HostKVConfig(
        r, t, rows, width, t, hot_groups=min(r * (t // 4), rows * width + r * 9)
    )
    # Packed member stride and HND payload. Physical page zero is legal and
    # appears at the end of request one's permuted block table.
    layout = torch.empty_strided(
        (32, 2, 32, h, d),
        (32 * h * d * 4, 32 * h * d, d, 32 * d, 1),
        device="meta",
        dtype=dtype,
    )
    table = torch.arange(31, -1, -1, dtype=torch.int32).reshape(r, -1)
    pool = ActiveHostKV(config, layout, device="cpu", scales=scales, block_table=table)
    source = torch.empty_strided(layout.shape, layout.stride(), dtype=dtype)
    generator = torch.Generator().manual_seed(2104)
    for req in range(r):
        k = torch.randn((t, h, d), generator=generator).half()
        v = torch.randn((t, h, d), generator=generator).half()
        if e4m3:
            k = (k.float() / scales.key).to(torch.float8_e4m3fn).view(torch.uint8)
            v = (v.float() / scales.value).to(torch.float8_e4m3fn).view(torch.uint8)
        pool.write_encoded(
            k,
            v,
            torch.full((t,), req, dtype=torch.int32),
            torch.arange(t, dtype=torch.int32),
        )
        for pos in range(t):
            page = int(table[req, pos // 32])
            source[page, 0, pos % 32] = k[pos]
            source[page, 1, pos % 32] = v[pos]
    query = torch.randn((rows, h * 2, d), generator=generator).half()
    selection = torch.randint(
        0, t, (rows, width), generator=generator, dtype=torch.int32
    )
    requests = torch.tensor([0, 1, 0], dtype=torch.int32)[:rows].contiguous()
    qpos = torch.tensor([511, 511, 61], dtype=torch.int64)[:rows].contiguous()
    if width > 1:
        selection[0, :2] = 7  # Repeated attention columns remain repeated.
    if width > 1 and rows > 1:
        selection[1].fill_(-1)  # Entirely masked query, including its LSE.
    lengths = torch.tensor([512, 509], dtype=torch.int32)
    plan = SparseAttentionPlan(
        rows=rows,
        query_heads=h * 2,
        kv_heads=h,
        head_dim=d,
        selection_width=width,
        device="cpu",
        e4m3=e4m3,
        scales=scales,
    )
    return pool, source, query, selection, requests, lengths, qpos, plan


@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize("width", [1, 17, 69])
@pytest.mark.parametrize("gated", [False, True])
def test_original_attention_and_host_hot_attention_are_bitwise_equal(
    e4m3, width, gated
):
    pool, source, query, selection, requests, lengths, qpos, plan = _case(e4m3, width)
    logical = selection.clone()
    valid = (logical >= 0) & (logical < lengths[requests.long(), None])
    valid &= logical <= qpos[:, None]
    logical[~valid] = -1
    gate = (
        torch.linspace(-1, 1, query.numel()).reshape_as(query).half() if gated else None
    )
    source_k, source_v = source.unbind(1)
    expected = plan.run(
        query, source_k, source_v, logical, pool.block_table, requests, output_gate=gate
    ).clone()
    expected_lse = plan.lse.clone()
    mapped = pool.gather(selection, requests, lengths, qpos)
    hot_k, hot_v = pool.hot.unbind(1)
    actual = plan.run(
        query, hot_k, hot_v, mapped, pool.hot_block_table, requests, output_gate=gate
    )
    # Compare bytes, including signed zero. torch.equal alone allows +/-0.
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(expected_lse.view(torch.uint8), plan.lse.view(torch.uint8))
    assert torch.isfinite(actual).all()
    if width > 1:
        assert (actual[1] == 0).all()
        assert torch.isneginf(plan.lse[1]).all()
    assert plan.splits == {1: 1, 17: 2, 69: 4}[width]
    # Independently check logical reconstruction, not only the arithmetic.
    for row in range(pool.config.rows):
        for col in range(width):
            physical = int(mapped[row, col])
            if not valid[row, col]:
                assert physical == -1
            else:
                gid = int(pool.slot_owner[physical // 4])
                assert gid // pool.config.groups_per_request == int(requests[row])
                recovered = (gid % pool.config.groups_per_request) * 4 + physical % 4
                assert recovered == int(selection[row, col])


@pytest.mark.parametrize("e4m3", [False, True])
def test_original_kernel_matches_independent_torch_attention(e4m3):
    pool, source, query, selection, requests, lengths, qpos, plan = _case(e4m3, 17)
    mapped = pool.gather(selection, requests, lengths, qpos)
    k, v = pool.hot.unbind(1)
    actual = plan.run(query, k, v, mapped, pool.hot_block_table, requests).clone()
    expected = torch.zeros_like(actual)
    for row in range(query.shape[0]):
        req = int(requests[row])
        tokens = selection[row]
        tokens = tokens[(tokens >= 0) & (tokens < lengths[req]) & (tokens <= qpos[row])]
        for head in range(query.shape[1]):
            if not tokens.numel():
                continue
            pages = pool.block_table[req, tokens.long() // 32].long()
            within = tokens.long() % 32
            keys = source[pages, 0, within, head // 2]
            values = source[pages, 1, within, head // 2]
            if e4m3:
                keys = keys.view(torch.float8_e4m3fn).float() * pool.scales.key
                values = values.view(torch.float8_e4m3fn).float() * pool.scales.value
            scores = keys.float() @ query[row, head].float() * query.shape[-1] ** -0.5
            expected[row, head] = scores.softmax(0) @ values.float()
    # torch uses a different dot/softmax rounding plan. Bitwise identity is
    # required above only for the two arms using the identical QSA kernels.
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)


def test_eviction_append_and_new_query_still_match_original_kernel():
    pool, source, query, _, requests, lengths, qpos, plan = _case(False, 1, rows=1)
    k, v = source.unbind(1)
    for token in range(0, 64, 4):
        selection = torch.tensor([[token]], dtype=torch.int32)
        expected = plan.run(query, k, v, selection, pool.block_table, requests).clone()
        mapped = pool.gather(selection, requests, lengths, qpos)
        hk, hv = pool.hot.unbind(1)
        actual = plan.run(query, hk, hv, mapped, pool.hot_block_table, requests)
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    # Force overwrite of an existing resident row, then compare another query.
    row_k = torch.full((1, pool.heads, pool.dim), 0.375, dtype=torch.float16)
    row_v = -row_k
    pool.write(row_k, row_v, requests, torch.tensor([60], dtype=torch.int32))
    page = int(pool.block_table[0, 1])
    source[page, 0, 28] = row_k[0]
    source[page, 1, 28] = row_v[0]
    selection = torch.tensor([[60]], dtype=torch.int32)
    query = -query
    expected = plan.run(query, k, v, selection, pool.block_table, requests).clone()
    mapped = pool.gather(selection, requests, lengths, qpos)
    actual = plan.run(query, hk, hv, mapped, pool.hot_block_table, requests)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))


@pytest.mark.parametrize("e4m3", [False, True])
def test_growing_visibility_matches_original_output_and_lse(e4m3):
    pool, source, query, selection, requests, lengths, qpos, plan = _case(
        e4m3, 17, rows=1
    )
    k, v = source.unbind(1)
    hk, hv = pool.hot.unbind(1)
    selection.fill_(-1)
    for length in (61, 62, 64, 65):
        lengths.fill_(length)
        qpos.fill_(length - 1)
        selection[0, :3] = torch.tensor([0, length - 1, length - 1], dtype=torch.int32)
        expected = plan.run(query, k, v, selection, pool.block_table, requests).clone()
        expected_lse = plan.lse.clone()
        mapped = pool.gather(selection, requests, lengths, qpos)
        actual = plan.run(query, hk, hv, mapped, pool.hot_block_table, requests)
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        assert torch.equal(expected_lse.view(torch.uint8), plan.lse.view(torch.uint8))


def test_attention_workspace_product_rejection_precedes_allocation():
    with pytest.raises(UnsupportedHostKVConfig, match="partial elements"):
        SparseAttentionPlan(
            rows=2**20,
            query_heads=4096,
            kv_heads=1,
            head_dim=16,
            selection_width=1,
            device="cpu",
            e4m3=False,
        )
