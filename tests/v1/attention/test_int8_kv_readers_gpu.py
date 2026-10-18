# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-wheel qualification; run in the coordinated SM70 GPU window."""

import math

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_int8 as reader
from vllm.v1.attention.kv_cache_view import KVCacheView, bind_planar_kv_cache
from vllm.v1.attention.kv_codecs import INT8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM70")


def fixture(rows, member=1, padding=0):
    reader.require_int8_abi()
    torch.manual_seed(42)
    raw = torch.zeros(3 * (848640 + 2 * padding), dtype=torch.int8, device="cuda")
    carrier = bind_planar_kv_cache(
        raw,
        num_blocks=3,
        scheduler_block_size=816,
        kernel_block_size=16,
        num_heads=1,
        head_size=256,
        page_size_bytes=424320 + padding,
        member_index=member,
        member_count=2,
    )
    view = KVCacheView.bind(carrier, INT8, 256)
    view.key.random_(-128, 128)
    view.value.random_(-128, 128)
    view.key_scale.uniform_(0.002, 0.008)
    view.value_scale.copy_(
        torch.logspace(-6, -1.7, 2448, device="cuda").view(153, 16, 1)
    )
    q = torch.randn(rows, 6, 256, dtype=torch.float16, device="cuda") * 0.3
    indices = torch.arange(2051, device="cuda", dtype=torch.int32).repeat(rows, 1)
    table = torch.stack(
        [torch.randperm(153, device="cuda", dtype=torch.int32) for _ in range(8)]
    )
    req = torch.arange(rows, device="cuda", dtype=torch.int32) // 5
    pos = 2046 + torch.arange(rows, device="cuda", dtype=torch.int64) % 5
    indices.copy_(torch.where(indices <= pos[:, None], indices, -1))
    lens = torch.full((8,), 2051, device="cuda", dtype=torch.int32)
    return raw, view, q, indices, table, req, pos, lens


def oracle(q, view, indices, table, requests):
    q, indices, table, requests = [x.cpu() for x in (q, indices, table, requests)]
    k, v = view.key.cpu().double(), view.value.cpu().double()
    sk, sv = view.key_scale.cpu().double(), view.value_scale.cpu().double()
    output = torch.zeros_like(q, dtype=torch.float64)
    lse = torch.full(q.shape[:2], -math.inf, dtype=torch.float64)
    for row in range(q.shape[0]):
        r = int(requests[row])
        if r < 0:
            continue
        ids = indices[row]
        ids = ids[(ids >= 0) & (ids < table.shape[1] * 16)].long()
        pages = table[r, ids // 16].long()
        offsets = ids % 16
        keys = k[pages, offsets, 0]
        values = v[pages, offsets, 0] * sv[pages, offsets, 0, None]
        scores = (q[row].double() @ keys.T) * sk[pages, offsets, 0][None, :] / 16
        output[row] = torch.softmax(scores, -1) @ values
        lse[row] = torch.logsumexp(scores, -1) / math.log(2)
    return output.float(), lse.float()


@pytest.mark.parametrize("rows", [1, 5, 8, 17, 20, 40])
@pytest.mark.parametrize("route", ["decode", "grouped"])
@pytest.mark.parametrize("padding", [0, 816])
def test_native_readers_and_graph(rows, route, padding):
    raw, view, q, ids, table, req, pos, lens = fixture(rows, padding=padding)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")
    call = reader.native_decode if route == "decode" else reader.native_grouped

    def run():
        args = (
            q,
            view.key,
            view.value,
            ids,
            table,
            req,
            pos,
            lens,
            out,
            view.key_scale,
            view.value_scale,
        )
        return call(*args, **({"lse": lse} if route == "grouped" else {}))

    saved_read = raw.clone()
    run()
    ref, ref_lse = oracle(q, view, ids, table, req)
    torch.testing.assert_close(out.cpu().float(), ref, atol=0.004, rtol=0.015)
    if route == "grouped":
        torch.testing.assert_close(lse.cpu(), ref_lse, atol=0.002, rtol=0.002)
    triton_out = torch.empty_like(q)
    reader.triton_attention(
        q,
        view.key,
        view.value,
        ids,
        table,
        req,
        triton_out,
        view.key_scale,
        view.value_scale,
    )
    torch.testing.assert_close(out, triton_out, atol=0.004, rtol=0.015)
    assert torch.equal(saved_read, raw)
    saved = raw.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for change in (0.8, 1.1):
        q.mul_(change)
        view.key_scale.mul_(change)
        view.value_scale.mul_(change)
        view.value.neg_()
        table.copy_(table.roll(1, dims=1))
        run()
        eager = out.clone()
        graph.replay()
        assert torch.equal(eager, out)
        ref, _ = oracle(q, view, ids, table, req)
        torch.testing.assert_close(out.cpu().float(), ref, atol=0.004, rtol=0.015)
    assert not torch.equal(saved, raw)


def test_masked_poison_and_empty_rows():
    _, view, q, ids, table, req, pos, lens = fixture(17)
    # Entire invalid owner page; masks must keep its NaN scale out of P@V.
    table.copy_(table[:1].clone().expand_as(table))
    poison = int(table[0, -1].item())
    view.key_scale[poison].fill_(math.nan)
    view.value_scale[poison].fill_(math.nan)
    ids[-1].fill_(-1)
    req[-1] = -1
    pos[-1] = -1
    out = torch.empty_like(q)
    reader.native_grouped(
        q,
        view.key,
        view.value,
        ids,
        table,
        req,
        pos,
        lens,
        out,
        view.key_scale,
        view.value_scale,
    )
    assert torch.isfinite(out).all()
    assert not out[-1].count_nonzero()
