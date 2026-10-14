# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 GDN input on row-major weights: fp16_skinny_gemm2_sm70_out (V100).

The VLLM_SM70_QWEN38_GDN_NO_PACK route (qkvz [4096, 2560] + ba [24, 2560] in
one launch, plan SkinnyConfig(1, 1, 2, 1), 2..48 rows). GPU checks (skipped
without SM70 and the native op):
* M in {2, 5, 8, 9, 15, 16, 17, 20, 24, 25, 32, 33, 40, 41, 48} against an FP64
  reference (max abs error <= 2e-3 of the reference max: one FP16 rounding of
  an FP32 sum) and against the F.linear + split fallback (FP32 reductions);
  the helper equals the raw native call with the plan bitwise;
* row invariance: every row of an M=40 call is bitwise equal to the same row
  in 8-row chunks, alone (M=1), inside an M=48 call and next to different rows;
* determinism over repeated calls and CUDA-graph capture/replay of the opaque
  GDN input op with changed inputs;
* the route: env set -> helper bitwise, rejected rows/policies -> unchanged
  fallback; env unset -> unchanged; loader skips the packed copy only when set.

Run on a V100: pytest -s -v tests/kernels/test_sm70_gdn_input_rowmajor.py
(-s prints GDN_ROWMAJOR_DIFF lines).
"""

import pytest
import torch
import torch.nn.functional as F

import vllm.envs as envs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70 and its native extension",
)

ENV = "VLLM_SM70_QWEN38_GDN_NO_PACK"
M_LIST = [2, 5, 8, 9, 15, 16, 17, 20, 24, 25, 32, 33, 40, 41, 48]
REL_TOL = 2e-3
COLS = (2560, 1536, 12, 12)


def _modules():
    from vllm import _sm70_ops  # noqa: F401  (loads vllm._C, registers fakes)
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
    from vllm.models.qwen4_exp.nvidia import sm70_gdn_input_rowmajor as rowmajor

    if not rowmajor.native_available():
        pytest.skip("build lacks fp16_skinny_gemm2_sm70_out")
    return rowmajor, gemv


@pytest.fixture(autouse=True)
def _fp32_policy(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    matmul = torch.backends.cuda.matmul
    saved = (
        matmul.allow_fp16_reduced_precision_reduction,
        getattr(matmul, "allow_fp16_accumulation", False),
    )
    matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(matmul, "allow_fp16_accumulation"):
        matmul.allow_fp16_accumulation = False
    envs.disable_envs_cache()
    yield
    matmul.allow_fp16_reduced_precision_reduction = saved[0]
    if hasattr(matmul, "allow_fp16_accumulation"):
        matmul.allow_fp16_accumulation = saved[1]
    envs.disable_envs_cache()


@pytest.fixture(scope="module")
def weights():
    out = []
    for n in (4096, 24):
        g = torch.Generator(device="cuda").manual_seed(n * 7919 + 2560)
        w = (torch.randn(n, 2560, device="cuda", generator=g) * 0.02).half()
        out.append((w, w.double()))
    return out


def _input(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, 2560, device="cuda", generator=g).half()


def _native(x, wq, wb, config):
    """The raw dual op into NaN-filled outputs (any m in 1..48)."""
    m = x.shape[0]
    outs = tuple(
        torch.full((m, c), float("nan"), dtype=torch.float16, device="cuda")
        for c in COLS
    )
    ws = torch.empty(
        config.splits * m * 4120 if config.splits > 1 else 0,
        dtype=torch.float32,
        device="cuda",
    )
    torch.ops._C.fp16_skinny_gemm2_sm70_out(*outs, x, wq, wb, ws, *config)
    return outs


def _split_linear(x, wq, wb):
    qkvz, ba = F.linear(x, wq), F.linear(x, wb)
    return qkvz[:, :2560], qkvz[:, 2560:], ba[:, :12], ba[:, 12:]


def _equal(actual, expected):
    return all(
        torch.equal(a.view(torch.int16), e.contiguous().view(torch.int16))
        for a, e in zip(actual, expected, strict=True)
    )


def _cat(outs):
    return torch.cat((outs[0], outs[1]), 1), torch.cat((outs[2], outs[3]), 1)


@pytest.mark.parametrize("m", M_LIST)
def test_against_fp64_and_linear(weights, m):
    rowmajor, _ = _modules()
    (wq, wq64), (wb, wb64) = weights
    x = _input(m, seed=m * 131 + 4120)
    outs = rowmajor.gdn_input_rowmajor(x, wq, wb)
    assert outs is not None
    assert [tuple(o.shape) for o in outs] == [(m, c) for c in COLS]
    assert all(o.is_contiguous() for o in outs)
    assert _equal(outs, _native(x, wq, wb, rowmajor.PLAN))
    lin_q, lin_b = _cat(_split_linear(x, wq, wb))
    for got, w64, lin, name in zip(_cat(outs), (wq64, wb64), (lin_q, lin_b), "qb"):
        ref = x.double() @ w64.t()
        scale = ref.abs().max().item()
        diff = (got.double() - ref).abs().max().item()
        lin_diff = (got.float() - lin.float()).abs().max().item()
        print(
            f"GDN_ROWMAJOR_DIFF m={m} part={name} max_abs={diff} ref_max={scale} "
            f"vs_linear_max_abs={lin_diff} bitwise_linear={torch.equal(got, lin)}"
        )
        assert torch.isfinite(got.float()).all()
        assert diff <= REL_TOL * scale + 2**-24, (diff, scale)
        assert lin_diff <= REL_TOL * scale + 2**-24, (lin_diff, scale)


def test_row_invariance(weights):
    rowmajor, _ = _modules()
    (wq, _), (wb, _) = weights
    x48 = _input(48, seed=4242)
    full = rowmajor.gdn_input_rowmajor(x48[:40], wq, wb)
    for j in range(5):
        chunk = rowmajor.gdn_input_rowmajor(x48[8 * j : 8 * j + 8], wq, wb)
        assert _equal(chunk, [o[8 * j : 8 * j + 8] for o in full]), j
    for i in (0, 3, 7, 8, 17, 30, 39):
        # M=1 is outside the route; the native op takes it with the same plan.
        row = _native(x48[i : i + 1].contiguous(), wq, wb, rowmajor.PLAN)
        assert _equal(row, [o[i : i + 1] for o in full]), i
        pair = rowmajor.gdn_input_rowmajor(x48[i : i + 2].contiguous(), wq, wb)
        assert _equal([p[:1] for p in pair], [o[i : i + 1] for o in full]), i
    wide = rowmajor.gdn_input_rowmajor(x48, wq, wb)
    assert _equal([o[:40] for o in wide], full)
    # Rows are independent of their neighbours (different co-batched rows).
    mixed = x48[:40].clone()
    mixed[1::2] = _input(20, seed=99)
    mixed_out = rowmajor.gdn_input_rowmajor(mixed, wq, wb)
    assert _equal([o[0::2] for o in mixed_out], [o[0::2] for o in full])


@pytest.mark.parametrize("m", (2, 17, 48))
def test_determinism(weights, m):
    rowmajor, _ = _modules()
    (wq, _), (wb, _) = weights
    x = _input(m, seed=7 + m)
    first = rowmajor.gdn_input_rowmajor(x, wq, wb)
    for _ in range(5):
        assert _equal(rowmajor.gdn_input_rowmajor(x, wq, wb), first)


@pytest.mark.parametrize("m", (2, 5, 16, 17, 40, 48))
def test_graph_replay_of_route(weights, monkeypatch, m):
    rowmajor, gemv = _modules()
    (wq, _), (wb, _) = weights
    monkeypatch.setenv(ENV, "1")
    static_x = _input(m, seed=10 + m)
    eager = torch.ops.vllm.qwen38_sm70_fp16_gdn_input(static_x, wq, wb, None, None)
    assert _equal(eager, rowmajor.gdn_input_rowmajor(static_x, wq, wb))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.ops.vllm.qwen38_sm70_fp16_gdn_input(
            static_x, wq, wb, None, None
        )
    for seed in (11, 12, 13):
        new_x = _input(m, seed=seed * 100 + m)
        static_x.copy_(new_x)
        for out in captured:
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _equal(captured, rowmajor.gdn_input_rowmajor(new_x, wq, wb)), seed
        assert _equal(captured, _native(new_x, wq, wb, rowmajor.PLAN)), seed


@pytest.mark.parametrize("m", (2, 9, 16, 33, 48))
def test_route_when_set_and_unset(weights, monkeypatch, m):
    rowmajor, gemv = _modules()
    (wq, _), (wb, _) = weights
    x = _input(m, seed=500 + m)
    unset = gemv._qwen38_sm70_fp16_gdn_input(x, wq, wb)
    assert _equal(unset, _split_linear(x, wq, wb))
    monkeypatch.setenv(ENV, "1")
    routed = gemv._qwen38_sm70_fp16_gdn_input(x, wq, wb)
    assert _equal(routed, rowmajor.gdn_input_rowmajor(x, wq, wb))


def test_rejected_rows_and_policies_fall_back(weights, monkeypatch):
    rowmajor, gemv = _modules()
    (wq, _), (wb, _) = weights
    x1 = _input(1, seed=1)
    x49 = _input(49, seed=49)
    x20 = _input(20, seed=20)
    m1_default = gemv._qwen38_sm70_fp16_gdn_input(x1, wq, wb)
    monkeypatch.setenv(ENV, "1")
    assert rowmajor.gdn_input_rowmajor(x1, wq, wb) is None
    assert rowmajor.gdn_input_rowmajor(x49, wq, wb) is None
    assert _equal(gemv._qwen38_sm70_fp16_gdn_input(x1, wq, wb), m1_default)
    assert _equal(
        gemv._qwen38_sm70_fp16_gdn_input(x49, wq, wb), _split_linear(x49, wq, wb)
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert rowmajor.gdn_input_rowmajor(x20, wq, wb) is None
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", True)
        assert rowmajor.gdn_input_rowmajor(x20, wq, wb) is None
        monkeypatch.setattr(
            torch.backends.cuda.matmul, "allow_fp16_accumulation", False
        )
    # A strided, misaligned input is copied into an aligned buffer first.
    g = torch.Generator(device="cuda").manual_seed(21)
    big = torch.randn(20, 2560 + 24, device="cuda", generator=g).half()
    strided = big[:, 4 : 4 + 2560]  # 8-byte offset
    assert strided.data_ptr() % 16 != 0
    assert _equal(
        rowmajor.gdn_input_rowmajor(strided, wq, wb),
        rowmajor.gdn_input_rowmajor(strided.contiguous(), wq, wb),
    )


def test_loader_skips_pack_only_when_set(weights, monkeypatch):
    _, gemv = _modules()
    (wq, _), _ = weights
    for no_pack in (False, True):
        monkeypatch.setenv(ENV, "1" if no_pack else "0")
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight", torch.nn.Parameter(wq.clone(), requires_grad=False)
        )
        layer._sm70_qwen38_prepare_gdn_batch = True
        gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)
        assert hasattr(layer, "_sm70_qwen38_gdn_packed") is not no_pack
