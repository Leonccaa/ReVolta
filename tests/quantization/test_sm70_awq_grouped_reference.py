# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the grouped small-M AWQ reference and its admission.

The native op itself is covered by tests/kernels/test_sm70_awq_grouped_decode.py
(SM70 only). These tests validate the pure-torch oracle that the GPU test and
benchmark rely on, and the load/runtime admission logic, without a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import awq_sm70_moe as moe
from vllm.model_executor.layers.quantization import sm70_awq_grouped_reference as ref

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("k,n", [(64, 64), (160, 96)])
def test_bank_pack_decode_round_trip(compact, k, n):
    generator = torch.Generator().manual_seed(k + n + compact)
    codes = torch.randint(0, 16, (k, n), generator=generator)
    zeros = torch.randint(0, 16, (k // 32, n), generator=generator)
    scales = (torch.rand((k // 32, n), generator=generator) * 0.02 + 1e-4).half()
    weight, metadata = ref.pack_tm_bank(codes, zeros, scales, compact)
    assert weight.shape == (k, n // 8) and weight.dtype == torch.int32
    group = torch.arange(k) // 32
    bias = (-zeros.half() * scales).half()
    expected = (codes.double() * scales[group].double() + bias[group].double()).half()
    decoded = ref.decode_tm_bank(weight, metadata)
    assert torch.equal(decoded.view(torch.int16), expected.view(torch.int16))


def test_bank_layout_matches_qpn_word_addressing():
    """Word (tile*(K/8) + kblock)*32 + col holds K8 codes of one column."""
    k, n = 64, 64
    codes = torch.zeros((k, n), dtype=torch.int64)
    codes[17, 37] = 9  # kblock 2, logical 1; tile 1, col 5
    weight, _ = ref.pack_tm_bank(
        codes,
        torch.zeros((k // 32, n), dtype=torch.int64),
        torch.ones((k // 32, n), dtype=torch.float16),
    )
    flat = weight.reshape(-1)
    index = (1 * (k // 8) + 2) * 32 + 5
    assert int(flat[index]) == 9 << (4 * ref.NIBBLE_ORDER[1])
    assert torch.count_nonzero(flat) == 1


def test_plan_reference_groups_by_expert_and_drops_invalid():
    ids = torch.tensor(
        [[5, 3, 511, -1, 7, 8, 9, 10, 11, 12], [3, 5, 512, 0, 1, 2, 4, 6, 13, 14]],
        dtype=torch.int32,
    )
    plan = ref.grouped_plan_reference(ids, cap=8)
    packs = int(plan[0])
    experts = plan[ref.PLAN_EXPERTS : ref.PLAN_EXPERTS + packs].tolist()
    assert experts == sorted(set(experts))
    assert -1 not in experts and 512 not in experts
    routes = []
    for pack in range(packs):
        count = int(plan[ref.PLAN_COUNTS + pack])
        rows = plan[
            ref.PLAN_ROWS + pack * ref.ROW_CAP : ref.PLAN_ROWS
            + pack * ref.ROW_CAP
            + count
        ].tolist()
        assert rows == sorted(rows)
        assert all(int(ids.view(-1)[r]) == experts[pack] for r in rows)
        routes += rows
    valid = [r for r, e in enumerate(ids.view(-1).tolist()) if 0 <= e < 512]
    assert sorted(routes) == valid
    mask = ref.plan_prefix_mask(plan)
    assert torch.all(plan[mask] >= 0)


def test_plan_reference_splits_packs_at_capacity():
    # Duplicate ids inside a token are invalid routing; the plan still keeps
    # every route by splitting the expert into packs of at most `cap` rows.
    ids = torch.full((3, 10), 42, dtype=torch.int32)
    plan = ref.grouped_plan_reference(ids, cap=8)
    assert int(plan[0]) == 4
    counts = plan[ref.PLAN_COUNTS : ref.PLAN_COUNTS + 4].tolist()
    assert counts == [8, 8, 8, 6]


def test_reference_one_hot_matches_decoded_weights():
    experts = (0, 3, 511)
    w13, s13, d13 = ref.random_bank(2560, 320, experts, seed=1)
    w2, s2, d2 = ref.random_bank(160, 2560, experts, seed=2)
    assert w13.shape == (512, 2560, 40) and s13.shape == (512, 80, 320, 3)
    assert w2.shape == (512, 160, 320) and s2.shape == (512, 5, 2560, 3)
    x = torch.zeros((2, 2560), dtype=torch.float16)
    x[0, 31] = 1
    x[1, 2559] = 1
    ids = torch.tensor([[0, 3, 511, -1] + [0] * 6, [511] * 10], dtype=torch.int32)
    topk = torch.full((2, 10), 0.1)
    out, inter = ref.awq_grouped_decode_reference(
        x, ids, topk, d13.__getitem__, d2.__getitem__
    )
    for route, expert in enumerate(ids.view(-1).tolist()):
        if expert < 0:
            assert torch.count_nonzero(inter[route]) == 0
            continue
        row = d13[expert][31 if route < 10 else 2559].float()
        silu = (row[0::2] / (1 + torch.exp(-row[0::2]))).half()
        expected = (silu.float() * row[1::2].half().float()).half()
        assert torch.equal(inter[route], expected)
    assert out.shape == (2, 2560) and out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()


def test_reference_reduction_is_router_order_fma():
    experts = (1, 2)
    _, _, d13 = ref.random_bank(2560, 320, experts, seed=3)
    _, _, d2 = ref.random_bank(160, 2560, experts, seed=4)
    torch.manual_seed(5)
    x = (torch.randn(1, 2560) * 0.05).half()
    ids = torch.tensor([[1, 2] + [-1] * 8], dtype=torch.int32)
    topk = torch.tensor([[0.7, 0.3] + [0.0] * 8])
    out, inter = ref.awq_grouped_decode_reference(
        x, ids, topk, d13.__getitem__, d2.__getitem__
    )
    r0 = (inter[0].double() @ d2[1].double()).float().half().float()
    r1 = (inter[1].double() @ d2[2].double()).float().half().float()
    w0, w1 = topk[0, 0].double(), topk[0, 1].double()  # FP32 router weights
    acc = (w0 * r0.double()).float()  # fmaf(w0, v0, 0)
    acc = (w1 * r1.double() + acc.double()).float()  # fmaf(w1, v1, acc)
    assert torch.equal(out[0], acc.half())


def _layer(**changes):
    layer = SimpleNamespace(
        sm70_awq_moe_batched_gemm=True,
        sm70_awq_moe_w13_interleaved=True,
        sm70_awq_checkpoint_group_size=32,
        sm70_awq_qwen38_grouped_qpn=True,
        w13_tm_weight=torch.empty((512, 2560, 40), dtype=torch.int32, device="meta"),
        w2_tm_weight=torch.empty((512, 160, 320), dtype=torch.int32, device="meta"),
    )
    for key, value in changes.items():
        setattr(layer, key, value)
    return layer


def test_initialize_default_off_never_inspects_layer(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_AWQ_QWEN38_GROUPED_QPN", raising=False)
    assert not moe._initialize_qwen38_grouped_qpn(object(), True, 10)


def test_initialize_rejects_bad_split(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_GROUPED_QPN", "1")
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT", "2")
    with pytest.raises(ValueError):
        moe._initialize_qwen38_grouped_qpn(_layer(), True, 10)


@pytest.mark.parametrize(
    "contract,top_k,changes",
    [
        (False, 10, {}),
        (True, 8, {}),
        (True, 10, {"sm70_awq_moe_batched_gemm": False}),
        (True, 10, {"sm70_awq_moe_w13_interleaved": False}),
        (True, 10, {"sm70_awq_checkpoint_group_size": 128}),
        (
            True,
            10,
            {
                "w2_tm_weight": torch.empty(
                    (512, 160, 160), dtype=torch.int32, device="meta"
                )
            },
        ),
    ],
)
def test_initialize_falls_back_on_unsupported_layers(
    monkeypatch, contract, top_k, changes
):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_GROUPED_QPN", "1")
    monkeypatch.delenv("VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT", raising=False)
    assert not moe._initialize_qwen38_grouped_qpn(_layer(**changes), contract, top_k)


def test_initialize_requires_native_op(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_GROUPED_QPN", "1")
    expected = hasattr(torch.ops._C, "awq_moe_grouped_decode_sm70_out")
    assert moe._initialize_qwen38_grouped_qpn(_layer(), True, 10) == expected


def _runtime_inputs(m, dtype=torch.float16):
    x = torch.zeros((m, 2560), dtype=dtype)
    weights = torch.zeros((m, 10), dtype=torch.float32)
    ids = torch.zeros((m, 10), dtype=torch.int32)
    return x, weights, ids


@pytest.mark.parametrize("m", [1, 2, 5, 16, 17, 20, 21, 32])
def test_runtime_admission_row_band(m):
    admitted = moe._use_qwen38_grouped_qpn(_layer(), *_runtime_inputs(m))
    assert admitted == (2 <= m <= 20)


def test_runtime_admission_rejects_layout_and_dtype():
    layer = _layer()
    x, weights, ids = _runtime_inputs(5)
    assert moe._use_qwen38_grouped_qpn(layer, x, weights, ids)
    assert not moe._use_qwen38_grouped_qpn(layer, x.float(), weights, ids)
    assert not moe._use_qwen38_grouped_qpn(layer, x, weights.half(), ids)
    assert not moe._use_qwen38_grouped_qpn(layer, x, weights, ids.long())
    wide = torch.zeros((5, 20), dtype=torch.int32)
    assert not moe._use_qwen38_grouped_qpn(layer, x, weights, wide[:, ::2])
    assert not moe._use_qwen38_grouped_qpn(
        _layer(sm70_awq_qwen38_grouped_qpn=False), x, weights, ids
    )


@pytest.mark.parametrize("m", [1, 5, 8, 9, 10, 15, 20, 21, 32])
@pytest.mark.parametrize("flag", [False, True])
def test_grouped_decode20_runtime_band(m, flag):
    layer = SimpleNamespace(sm70_awq_qwen38_grouped_decode20=flag)
    assert moe._use_qwen38_grouped_decode20(layer, m, 10) == (flag and 9 <= m <= 20)
    assert not moe._use_qwen38_grouped_decode20(layer, m, 8)


@pytest.mark.parametrize("grouped", [False, True])
def test_grouped_decode20_route_plan(grouped):
    from vllm.model_executor.layers.quantization.sm70_moe_router import (
        Sm70MoeStageRoute,
        select_sm70_quantized_moe_route,
    )

    plan = select_sm70_quantized_moe_route(
        batched_enabled=True,
        num_tokens=20,
        total_slots=200,
        w13_per_expert_dispatch=not grouped,
        w2_per_expert_dispatch=not grouped,
    )
    expected = (
        Sm70MoeStageRoute.BATCHED if grouped else Sm70MoeStageRoute.PER_EXPERT_DISPATCH
    )
    assert plan.w13 == expected and plan.w2 == expected
