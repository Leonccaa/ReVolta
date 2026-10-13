# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 grouped small-M AWQ MoE decode (awq_moe_grouped_decode_sm70_out).

Checks, for M = 1..20 rows with random routings:
* the routing plan against the pure-torch plan reference (bitwise);
* W13 intermediate and final output against the FP64 decoded-bank reference
  (relative 1e-2 of the reference max, FP16 boundaries);
* every row against a separate M=1 QPN call (bitwise with split=16, the
  QPN M1 arithmetic; tolerance with split=8) and batch invariance;
* changed-input CUDA graph replay, invalid ids, input validation, fake op;
* the full AWQSM70MoEMethod.apply() route against the existing TurboMind
  active-grouped (M<=8) / per-expert-dispatch (M>8) route on the same
  prepared banks.

Run on a V100: pytest -s -v tests/kernels/test_sm70_awq_grouped_decode.py
(-s prints the GROUPED_DIFF lines with max abs differences).
"""

import json
import os
import random
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import sm70_awq_grouped_reference as ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70 and its native extension",
)

ACTIVE = tuple(
    sorted(
        {0, 1, 2, 5, 31, 32, 63, 64, 100, 127, 128, 200, 255, 256}
        | {257, 300, 333, 384, 400, 450, 480, 500, 509, 510, 511}
        | set(range(11, 31))
    )
)
ALL_M = list(range(1, 21))
REPORT = os.environ.get("SM70_AWQ_GROUPED_REPORT")


def _report(**fields):
    line = "GROUPED_DIFF " + " ".join(f"{k}={v}" for k, v in fields.items())
    print(line)
    if REPORT:
        with open(REPORT, "a") as handle:
            handle.write(json.dumps(fields) + "\n")


def _ops():
    from vllm import _sm70_ops as ops

    if not hasattr(torch.ops._C, "awq_moe_grouped_decode_sm70_out"):
        pytest.skip("build lacks awq_moe_grouped_decode_sm70_out")
    return ops


@pytest.fixture(scope="module", params=[True, False], ids=["compact", "4byte"])
def banks(request):
    compact = request.param
    w13, s13, d13 = ref.random_bank(2560, 320, ACTIVE, compact, seed=11, device="cuda")
    w2, s2, d2 = ref.random_bank(160, 2560, ACTIVE, compact, seed=12, device="cuda")
    return SimpleNamespace(w13=w13, s13=s13, d13=d13, w2=w2, s2=s2, d2=d2)


def _routing(m, seed, invalid=0, pool=ACTIVE):
    """Distinct experts per token; correlated like MTP verify rows (5/request)."""
    rng = random.Random(seed)
    rows = []
    previous = None
    for token in range(m):
        if token % 5 == 0 or previous is None:
            chosen = rng.sample(pool, 10)
        else:
            keep = rng.sample(previous, rng.randint(3, 8))
            rest = [e for e in pool if e not in keep]
            chosen = keep + rng.sample(rest, 10 - len(keep))
            rng.shuffle(chosen)
        previous = chosen
        rows.append(list(chosen))
    for _ in range(invalid):
        t, s = rng.randrange(m), rng.randrange(10)
        rows[t][s] = rng.choice((-1, 512, 4096))
    ids = torch.tensor(rows, dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(m, 10, device="cuda"), dim=-1)
    return ids, weights.float().contiguous()


def _run(ops, b, x, ids, topk, split):
    m = x.shape[0]
    out = torch.full((m, 2560), float("nan"), dtype=torch.float16, device="cuda")
    inter = torch.full((m * 10, 160), float("nan"), dtype=torch.float16, device="cuda")
    plan = torch.full(
        (ops.AWQ_GROUPED_DECODE_PLAN_INTS,), -7, dtype=torch.int32, device="cuda"
    )
    ops.awq_moe_grouped_decode_sm70_out(
        out, inter, plan, x, b.w13, b.s13, b.w2, b.s2, ids, topk, split
    )
    return out, inter, plan


def _diff(actual, expected):
    diff = (actual.double() - expected.double()).abs()
    scale = expected.double().abs().max().item()
    return diff.max().item(), scale


@pytest.mark.parametrize("split", [16, 8])
@pytest.mark.parametrize("m", ALL_M)
def test_grouped_against_reference_and_qpn_m1(banks, m, split):
    ops = _ops()
    torch.manual_seed(1000 + m)
    x = (torch.randn(m, 2560, device="cuda") * 0.5).half()
    ids, topk = _routing(m, seed=m * 7 + split, invalid=1 if m % 4 == 0 else 0)
    out, inter, plan = _run(ops, banks, x, ids, topk, split)
    torch.cuda.synchronize()

    cap = 8 * ((m + 7) // 8)
    expected_plan = ref.grouped_plan_reference(ids.cpu(), cap)
    mask = ref.plan_prefix_mask(expected_plan)
    assert torch.equal(plan.cpu()[mask], expected_plan[mask])

    ref_out, ref_inter = ref.awq_grouped_decode_reference(
        x, ids, topk, banks.d13.__getitem__, banks.d2.__getitem__
    )
    valid = ((ids >= 0) & (ids < 512)).view(-1)
    mid_diff, mid_scale = _diff(inter[valid], ref_inter[valid])
    out_diff, out_scale = _diff(out, ref_out)

    # Per-token M=1 QPN M1 calls on the same banks.
    qpn_out = torch.empty_like(out)
    qpn_inter = torch.empty_like(inter)
    for t in range(m):
        ops.awq_moe_qpn_m1_sm70_out(
            qpn_out[t : t + 1],
            qpn_inter[t * 10 : (t + 1) * 10],
            x[t : t + 1].contiguous(),
            banks.w13,
            banks.s13,
            banks.w2,
            banks.s2,
            ids[t : t + 1].contiguous(),
            topk[t : t + 1].contiguous(),
        )
    torch.cuda.synchronize()
    qpn_out_diff, _ = _diff(out, qpn_out)
    qpn_mid_diff, _ = _diff(inter[valid], qpn_inter[valid])
    _report(
        test="reference",
        compact=banks.s13.dtype == torch.uint8,
        m=m,
        split=split,
        mid_max_abs=mid_diff,
        mid_ref_max=mid_scale,
        out_max_abs=out_diff,
        out_ref_max=out_scale,
        vs_qpn_m1_out_max_abs=qpn_out_diff,
        vs_qpn_m1_mid_max_abs=qpn_mid_diff,
        distinct_experts=int(expected_plan[0]),
    )
    assert torch.isfinite(out.float()).all()
    assert mid_diff <= 1e-2 * mid_scale + 2**-24
    assert out_diff <= 1e-2 * out_scale + 2**-24
    if split == 16:
        # Same K split, MMA chain, partial order and FP16 boundaries as QPN M1.
        assert torch.equal(out, qpn_out)
        assert torch.equal(inter[valid], qpn_inter[valid])
    else:
        assert qpn_out_diff <= 1e-2 * out_scale + 2**-24


@pytest.mark.parametrize("split", [16, 8])
def test_batch_invariance(banks, split):
    """A row's output does not depend on M or on its co-routed rows."""
    ops = _ops()
    torch.manual_seed(77)
    x = (torch.randn(20, 2560, device="cuda") * 0.5).half()
    ids, topk = _routing(20, seed=78)
    full, full_mid, _ = _run(ops, banks, x, ids, topk, split)
    for start, stop in ((0, 1), (0, 5), (5, 10), (3, 16), (10, 20), (0, 9)):
        part, part_mid, _ = _run(
            ops,
            banks,
            x[start:stop].contiguous(),
            ids[start:stop].contiguous(),
            topk[start:stop].contiguous(),
            split,
        )
        assert torch.equal(part, full[start:stop]), (start, stop)
        assert torch.equal(part_mid, full_mid[start * 10 : stop * 10]), (start, stop)


@pytest.mark.parametrize("m", [2, 5, 8, 9, 16, 17, 20])
def test_graph_replay_with_changed_routing(banks, m):
    ops = _ops()
    x = torch.zeros(m, 2560, dtype=torch.float16, device="cuda")
    ids = torch.zeros(m, 10, dtype=torch.int32, device="cuda")
    topk = torch.zeros(m, 10, dtype=torch.float32, device="cuda")
    out = torch.empty(m, 2560, dtype=torch.float16, device="cuda")
    inter = torch.empty(m * 10, 160, dtype=torch.float16, device="cuda")
    plan = torch.empty(
        ops.AWQ_GROUPED_DECODE_PLAN_INTS, dtype=torch.int32, device="cuda"
    )

    def run():
        ops.awq_moe_grouped_decode_sm70_out(
            out, inter, plan, x, banks.w13, banks.s13, banks.w2, banks.s2, ids, topk, 16
        )

    def load(seed):
        torch.manual_seed(seed)
        x.copy_((torch.randn(m, 2560, device="cuda") * 0.5).half())
        new_ids, new_topk = _routing(m, seed=seed, invalid=seed % 3)
        ids.copy_(new_ids)
        topk.copy_(new_topk)

    load(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for seed in range(2, 8):
        load(seed)
        run()
        torch.cuda.synchronize()
        eager_out, eager_mid = out.clone(), inter.clone()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        valid = ((ids >= 0) & (ids < 512)).view(-1)
        assert torch.equal(out, eager_out), seed
        assert torch.equal(inter[valid], eager_mid[valid]), seed
    ids.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
    ids.fill_(512)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0


def test_rejects_unsupported_inputs_and_traces(banks):
    ops = _ops()
    m = 5
    x = (torch.randn(m, 2560, device="cuda") * 0.5).half()
    ids, topk = _routing(m, seed=5)
    out = torch.empty(m, 2560, dtype=torch.float16, device="cuda")
    inter = torch.empty(m * 10, 160, dtype=torch.float16, device="cuda")
    plan = torch.empty(
        ops.AWQ_GROUPED_DECODE_PLAN_INTS, dtype=torch.int32, device="cuda"
    )
    valid = [
        out,
        inter,
        plan,
        x,
        banks.w13,
        banks.s13,
        banks.w2,
        banks.s2,
        ids,
        topk,
        16,
    ]

    def run():
        ops.awq_moe_grouped_decode_sm70_out(*valid)

    torch.compile(run, backend="eager", fullgraph=True)()
    unaligned = torch.empty(m * 2560 + 1, dtype=x.dtype, device="cuda")[1:].view_as(x)
    big = torch.zeros(21, 2560, dtype=torch.float16, device="cuda")
    for index, replacement in (
        (3, x.float()),
        (3, unaligned),
        (3, big),
        (2, plan[:100]),
        (8, ids.long()),
        (9, topk.half()),
        (1, inter[:-1]),
        (10, 4),
    ):
        args = list(valid)
        args[index] = replacement
        with pytest.raises(RuntimeError):
            ops.awq_moe_grouped_decode_sm70_out(*args)


# --------------------------------------------------------------------------
# Full method route: grouped vs the existing TurboMind route on real prepared
# banks (random AWQ checkpoint tensors through awq_sm70_prepare_compact).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def awq_layer():
    if not hasattr(torch.ops._C, "awq_moe_grouped_decode_sm70_out"):
        pytest.skip("build lacks awq_moe_grouped_decode_sm70_out")
    saved = dict(os.environ)
    os.environ.update(
        VLLM_SM70_AWQ_QWEN38_GROUPED_QPN="1",
        VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE="1",
        VLLM_SM70_AWQ_MOE_COMPACT_METADATA="1",
        VLLM_SM70_AWQ_TUNE_SMALL_SHAPES="0",
    )
    try:
        layer = ref.build_random_awq_layer()
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert layer.sm70_awq_qwen38_grouped_qpn
    assert layer.sm70_awq_moe_compact_metadata
    return layer


def _apply(layer, x, ids, topk, grouped):
    layer.sm70_awq_qwen38_grouped_qpn = grouped
    out = layer._r2_method.apply(layer, x, topk, ids, None, None)
    torch.cuda.synchronize()
    return out.clone()


@pytest.mark.parametrize("m", ALL_M)
def test_method_route_against_turbomind(awq_layer, m):
    pool = tuple(range(512))
    torch.manual_seed(5000 + m)
    x = (torch.randn(m, 2560, device="cuda") * 0.5).half()
    ids, topk = _routing(m, seed=900 + m, pool=pool)
    turbomind = _apply(awq_layer, x, ids, topk, grouped=False)
    grouped = _apply(awq_layer, x, ids, topk, grouped=True)
    awq_layer.sm70_awq_qwen38_grouped_qpn = True

    cache13, cache2 = {}, {}

    def d13(e):
        if e not in cache13:
            cache13[e] = ref.decode_tm_bank(
                awq_layer.w13_tm_weight[e], awq_layer.w13_tm_scales[e]
            )
        return cache13[e]

    def d2(e):
        if e not in cache2:
            cache2[e] = ref.decode_tm_bank(
                awq_layer.w2_tm_weight[e], awq_layer.w2_tm_scales[e]
            )
        return cache2[e]

    reference, _ = ref.awq_grouped_decode_reference(x, ids, topk, d13, d2)
    tm_diff, scale = _diff(grouped, turbomind)
    ref_diff, _ = _diff(grouped, reference)
    tm_ref_diff, _ = _diff(turbomind, reference)
    _report(
        test="method",
        m=m,
        route_turbomind=(
            "qpn_m1" if m == 1 else "active_grouped" if m <= 8 else "per_expert"
        ),
        grouped_vs_turbomind_max_abs=tm_diff,
        grouped_vs_ref_max_abs=ref_diff,
        turbomind_vs_ref_max_abs=tm_ref_diff,
        ref_max=scale,
    )
    assert torch.isfinite(grouped.float()).all()
    if m == 1:
        # M=1 stays on QPN M1 (not admitted to the grouped route).
        assert torch.equal(grouped, turbomind)
    assert tm_diff <= 1e-2 * scale + 2**-24
    assert ref_diff <= 1e-2 * scale + 2**-24
