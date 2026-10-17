# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V100-only W3 A/B tests. Written on CPU; NOT executed or compiled there."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.test_sm70_awq_grouped_decode import (
    ACTIVE,
    _routing,
    _run,
)
from vllm import _sm70_ops as ops
from vllm.model_executor.layers.quantization import sm70_awq_grouped_reference as ref
from vllm.model_executor.layers.quantization.awq_int4_zero import pack_int4_zero

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires rebuilt W3 SM70 extension on V100",
)


@pytest.fixture(scope="module")
def banks():
    assert hasattr(torch.ops._C, "awq_sm70_int4_zero_supported")
    w13, s13, _ = ref.random_bank(2560, 320, ACTIVE, True, seed=11, device="cuda")
    w2, s2, _ = ref.random_bank(160, 2560, ACTIVE, True, seed=12, device="cuda")
    old = SimpleNamespace(w13=w13, s13=s13, w2=w2, s2=s2)
    new = SimpleNamespace(
        w13=w13, s13=pack_int4_zero(s13), w2=w2, s2=pack_int4_zero(s2)
    )
    return old, new


def _bits_equal(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("m", list(range(1, 21)) + [21, 24, 25, 32, 39, 40, 48])
@pytest.mark.parametrize("split", [4, 8, 16])
@pytest.mark.parametrize("pack_rows", [0, 24])
def test_grouped_bitwise(banks, m, split, pack_rows):
    torch.manual_seed(m + split)
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    ids, topk = _routing(m, seed=m, hot=511, invalid=2)
    old, new = [_run(ops, b, x, ids, topk, split, pack_rows) for b in banks]
    _bits_equal(old[0], new[0])
    valid = ((ids >= 0) & (ids < 512)).view(-1)
    _bits_equal(old[1][valid], new[1][valid])
    expected_plan = ref.grouped_plan_reference(ids.cpu(), ref.pack_cap(m, pack_rows))
    mask = ref.plan_prefix_mask(expected_plan)
    assert torch.equal(old[2].cpu()[mask], new[2].cpu()[mask])


@pytest.mark.parametrize("index", [None, 0, 31, 32, 2559])
def test_m1_bitwise_and_graph(banks, index):
    x = torch.randn(1, 2560, device="cuda", dtype=torch.float16)
    if index is not None:
        x.zero_()
        x[0, index] = 1
    ids, topk = _routing(1, seed=13)
    results = []
    for b in banks:
        out = torch.empty_like(x)
        mid = torch.empty((10, 160), dtype=x.dtype, device="cuda")

        def run(out=out, mid=mid, b=b):
            ops.awq_moe_qpn_m1_sm70_out(
                out, mid, x, b.w13, b.s13, b.w2, b.s2, ids, topk
            )

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        results.append((out.clone(), mid.clone()))
    for old, new in zip(*results):
        _bits_equal(old, new)


@pytest.mark.parametrize("k,n", [(2560, 320), (160, 2560)])
@pytest.mark.parametrize("indexed", [False, True])
def test_turbomind_expansion_bitwise_and_graph(banks, monkeypatch, k, n, indexed):
    # Compare with the W3 switch OFF baseline as well: default path untouched.
    old, new = banks
    w, s0, s1 = (old.w13, old.s13, new.s13) if k == 2560 else (old.w2, old.s2, new.s2)
    selected = [511, 0, 257, 1, 511, 32, 500, 0] if indexed else list(range(512))
    num = len(selected)
    x = torch.randn(num, k, dtype=torch.float16, device="cuda") * 0.1
    offsets = torch.arange(num + 1, dtype=torch.int32, device="cuda")
    ids = torch.tensor(selected, dtype=torch.int32, device="cuda")
    original_x = x.clone()
    changed_x = (x.float() + 0.125).half()
    results = []
    for metadata, enabled in ((s0, "0"), (s0, "1"), (s1, "1")):
        monkeypatch.setenv("VLLM_SM70_AWQ_MOE_INT4_ZERO", enabled)
        pw, ps = ops.awq_moe_build_strided_ptrs(w, metadata, k * 32, -n, 512)
        out = torch.empty((num, n), dtype=torch.float16, device="cuda")

        def run(out=out, pw=pw, ps=ps):
            if indexed:
                ops.awq_moe_single_token_indexed_dense_stage_sm70_out(
                    out, x, offsets, ids, pw, ps, num, k, n, 32
                )
            else:
                ops.awq_moe_gemm_sm70_per_expert_dispatch_out(
                    out, x, offsets, pw, ps, num, k, n, 32, False
                )

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.copy_(changed_x)
        run()
        eager = out.clone()
        graph.replay()
        _bits_equal(eager, out)
        results.append(out.clone())
        x.copy_(original_x)
    _bits_equal(results[0], results[1])
    _bits_equal(results[0], results[2])


def test_grouped_changed_routing_graph(banks):
    m = 40
    x = torch.randn(m, 2560, dtype=torch.float16, device="cuda")
    ids, topk = _routing(m, seed=137, hot=511, invalid=3)
    changed_ids, changed_topk = _routing(m, seed=291, hot=0, invalid=4)
    results = []
    for b in banks:
        out = torch.empty_like(x)
        mid = torch.empty((m * 10, 160), dtype=x.dtype, device="cuda")
        plan = torch.empty(
            ops.AWQ_GROUPED_DECODE_PLAN_INTS, dtype=torch.int32, device="cuda"
        )
        local_ids, local_topk = ids.clone(), topk.clone()

        def run(
            out=out, mid=mid, plan=plan, b=b, local_ids=local_ids, local_topk=local_topk
        ):
            ops.awq_moe_grouped_decode_sm70_out(
                out,
                mid,
                plan,
                x,
                b.w13,
                b.s13,
                b.w2,
                b.s2,
                local_ids,
                local_topk,
                8,
                24,
            )

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        local_ids.copy_(changed_ids)
        local_topk.copy_(changed_topk)
        run()
        eager_out, eager_mid = out.clone(), mid.clone()
        graph.replay()
        valid = ((changed_ids >= 0) & (changed_ids < 512)).view(-1)
        _bits_equal(eager_out, out)
        _bits_equal(eager_mid[valid], mid[valid])
        results.append((out.clone(), mid[valid].clone()))
    for old, new in zip(*results):
        _bits_equal(old, new)
