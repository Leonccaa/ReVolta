# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe

ref = pytest.importorskip(
    "vllm.model_executor.layers.quantization.sm70_nvfp4_grouped_reference"
)
HAS_ROUTE = hasattr(moe, "_use_qwen38_raw_grouped_decode")


def layer_contract():
    return NS(
        sm70_nvfp4_raw_grouped_decode=True,
        sm70_nvfp4_qwen38_raw_scale=True,
        moe_config=NS(tp_size=4),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
        swiglu_limit=None,
    )


def test_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE", raising=False)
    envs.disable_envs_cache()
    assert not envs.VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE


@pytest.fixture
def m48_layer(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE", "1")
    monkeypatch.setenv("VLLM_SM70_NVFP4_RAW_GROUPED_M48_V154", "1")
    monkeypatch.delenv("VLLM_SM70_NVFP4_RAW_GROUPED_NARROW_V154", raising=False)
    envs.disable_envs_cache()
    monkeypatch.setattr(
        moe.sm70_ops, "has_nvfp4_raw_grouped_decode_dispatch", lambda: True
    )
    layer = layer_contract()
    layer.moe_config = NS(tp_size=4, dp_size=1, pcp_size=1)
    layer.sm70_nvfp4_qwen38_fused_swiglu_prefill = True
    layer.w13_raw_global_scales = torch.ones(512, 2)
    for name in (
        "w13_tm_weight",
        "w13_raw_scale_codes",
        "w2_tm_weight",
        "w2_raw_scale_codes",
        "w2_raw_global_scales",
    ):
        setattr(layer, name, torch.empty(0))
    return layer


@pytest.mark.parametrize(
    "blocked_by", [None, "default", "global", "layout", "tp", "dp"]
)
@pytest.mark.parametrize("narrow", [False, True])
def test_raw_v154_prepare(monkeypatch, m48_layer, blocked_by, narrow):
    layer = m48_layer
    if narrow:
        monkeypatch.delenv("VLLM_SM70_NVFP4_RAW_GROUPED_M48_V154")
        if blocked_by != "default":
            monkeypatch.setenv("VLLM_SM70_NVFP4_RAW_GROUPED_NARROW_V154", "1")
    if blocked_by == "default":
        monkeypatch.delenv("VLLM_SM70_NVFP4_RAW_GROUPED_M48_V154", raising=False)
    elif blocked_by == "global":
        layer.w13_raw_global_scales[-1, 1] = 1.25
    elif blocked_by == "layout":
        layer.sm70_nvfp4_qwen38_fused_swiglu_prefill = False
    elif blocked_by == "tp":
        layer.moe_config.tp_size = 2
    elif blocked_by == "dp":
        layer.moe_config.dp_size = 2

    moe.ModelOptNvFp4SM70MoEMethod._prepare_raw_grouped_decode(layer)
    if narrow:
        plan = layer._nvfp4_raw_grouped_narrow_plan
        assert layer._nvfp4_raw_grouped_m48_plan is None
    else:
        plan = layer._nvfp4_raw_grouped_m48_plan
        assert layer._nvfp4_raw_grouped_narrow_plan is None
    if blocked_by == "tp":
        assert not layer.sm70_nvfp4_raw_grouped_decode
        assert plan is None
        return
    assert layer.sm70_nvfp4_raw_grouped_decode
    rows = 64 if blocked_by is None and not narrow else 40
    assert layer._nvfp4_raw_grouped_output.shape == (rows, 2560)
    assert layer._nvfp4_raw_grouped_intermediate.shape == (rows * 10, 160)
    assert layer._nvfp4_raw_grouped_plan.numel() == ref.PLAN_INTS
    if blocked_by is None:
        assert plan is not None
        routes = 50 if narrow else 640
        assert [tuple(t.shape) for t in plan] == [
            (routes, 8),
            (routes,),
            (routes,),
            (1,),
        ]
        if narrow:
            base = layer._nvfp4_raw_grouped_plan.untyped_storage().data_ptr()
            assert all(t.untyped_storage().data_ptr() == base for t in plan)
    else:
        assert plan is None


def test_wide_raw_v154_disabled_keeps_ordinary_route(monkeypatch, m48_layer):
    rows = 48
    monkeypatch.delenv("VLLM_SM70_NVFP4_RAW_GROUPED_M48_V154")
    layer = m48_layer
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    method._prepare_raw_grouped_decode(layer)
    assert layer._nvfp4_raw_grouped_m48_plan is None
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **kw: True)

    def ordinary(*_):
        raise RuntimeError("ordinary dispatch")

    monkeypatch.setattr(moe, "_use_grouped_mtp5", ordinary)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=None,
        slot_mapping={},
        additional_kwargs={"sm70_nvfp4_decode_only": True},
    )
    with (
        override_forward_context(context),
        pytest.raises(RuntimeError, match="ordinary dispatch"),
    ):
        method.apply(
            layer,
            torch.empty(rows, 2560, dtype=torch.float16),
            torch.empty(rows, 10, dtype=torch.float32),
            torch.empty(rows, 10, dtype=torch.int32),
            None,
            None,
        )


@pytest.mark.parametrize(
    "rows,decode,offset",
    [
        (48, True, 0),
        (48, False, 0),
        (48, True, 1),
        (40, True, 0),
        (41, True, 0),
        (47, True, 0),
        (49, True, 0),
        (64, True, 0),
        (65, True, 0),
        (2, True, 0),
        (3, True, 0),
        (4, True, 0),
        (5, True, 0),
        (6, True, 0),
        (4, False, 0),
        (4, True, 1),
    ],
)
def test_raw_v154_apply(monkeypatch, m48_layer, rows, decode, offset):
    layer = m48_layer
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    monkeypatch.setenv("VLLM_SM70_NVFP4_RAW_GROUPED_NARROW_V154", "1")
    method._prepare_raw_grouped_decode(layer)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **kw: True)
    x = torch.zeros(rows * 2560 + offset, dtype=torch.float16)[offset:].view(rows, 2560)
    weights = torch.ones(rows, 10, dtype=torch.float32)
    ids = torch.zeros(rows, 10, dtype=torch.int32)
    calls = []
    expected_plan = (
        layer._nvfp4_raw_grouped_m48_plan
        if 41 <= rows <= 64
        else layer._nvfp4_raw_grouped_narrow_plan
    )
    mid_ptr = layer._nvfp4_raw_grouped_intermediate.data_ptr()
    out_ptr = layer._nvfp4_raw_grouped_output.data_ptr()

    def w13(mid, x_arg, w, codes, globals_, ids_arg, *plan_and_options):
        assert mid.shape == (rows * 10, 160) and mid.data_ptr() == mid_ptr
        assert x_arg is x and ids_arg is ids
        assert globals_ is layer.w13_raw_global_scales
        assert plan_and_options[:4] == expected_plan
        assert plan_and_options[4:] == (4, True)
        calls.append("w13")

    def w2(out, routed, mid, w, codes, globals_, weights_arg, *plan):
        assert out.shape == (rows, 2560) and out.data_ptr() == out_ptr
        assert routed.numel() == 0
        assert mid.shape == (rows * 10, 160) and mid.data_ptr() == mid_ptr
        assert weights_arg is weights
        assert plan == expected_plan
        calls.append("w2")

    def legacy(*_):
        raise RuntimeError("ordinary dispatch")

    monkeypatch.setattr(moe, "_use_grouped_mtp5", legacy)
    monkeypatch.setattr(moe.sm70_ops, "nvfp4_raw_grouped_decode_sm70_out", legacy)
    monkeypatch.setattr(
        torch.ops._C, "nvfp4_grouped_w13_raw_sm70_v154_out", w13, raising=False
    )
    monkeypatch.setattr(
        torch.ops._C,
        "nvfp4_grouped_w2_batch_reduce_raw_sm70_v154_out",
        w2,
        raising=False,
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=None,
        slot_mapping={},
        additional_kwargs={"sm70_nvfp4_decode_only": True} if decode else {},
    )
    with override_forward_context(context):
        if (rows in (3, 4, 5) or 41 <= rows <= 64) and decode and offset == 0:
            result = method.apply(layer, x, weights, ids, None, None)
            assert result.shape == (rows, 2560) and result.data_ptr() == out_ptr
            assert calls == ["w13", "w2"]
        else:
            with pytest.raises(RuntimeError, match="ordinary dispatch"):
                method.apply(layer, x, weights, ids, None, None)
            assert not calls


@pytest.mark.parametrize("m", [0, 1, *range(2, 41), 41, 48, 128])
def test_admission(m):
    assert moe._use_qwen38_raw_grouped_decode(
        layer_contract(),
        torch.empty(m, 2560, dtype=torch.float16),
        torch.empty(m, 10, dtype=torch.int32),
    ) == (2 <= m <= 40)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sm70_nvfp4_raw_grouped_decode", False),
        ("sm70_nvfp4_qwen38_raw_scale", False),
        ("sm70_nvfp4_num_experts", 256),
        ("sm70_nvfp4_hidden_size", 2048),
        ("sm70_nvfp4_intermediate_size", 320),
        ("sm70_nvfp4_top_k", 8),
        ("swiglu_limit", 7),
        ("moe_config", NS(tp_size=2)),
    ],
)
def test_other_contract_falls_back(field, value):
    layer = layer_contract()
    setattr(layer, field, value)
    assert not moe._use_qwen38_raw_grouped_decode(
        layer,
        torch.empty(40, 2560, dtype=torch.float16),
        torch.empty(40, 10, dtype=torch.int32),
    )


def test_e4m3_all_encodings():
    codes = torch.arange(256, dtype=torch.uint8)
    expected = codes.view(torch.float8_e4m3fn).float()
    actual = ref.decode_e4m3(codes)
    finite = ~expected.isnan()
    assert torch.equal(
        actual[finite].view(torch.int32), expected[finite].view(torch.int32)
    )
    assert torch.equal(actual.isnan(), expected.isnan())


@pytest.mark.parametrize("interleaved", [False, True])
def test_global_slot_and_fp32_rounding(interleaved):
    codes = torch.full((160, 320), 73, dtype=torch.uint8)  # 4.5
    globals_ = torch.tensor([1.00047, 0.00123456])
    expanded = ref.effective_scales(codes, globals_, interleaved)
    slots = (
        (torch.arange(320) // 32) % 2
        if interleaved
        else (torch.arange(320) >= 160).long()
    )
    assert torch.equal(expanded, (4.5 * globals_[slots]).repeat(160, 1).half())
    assert not torch.equal(expanded, (4.5 * globals_[slots].half()).repeat(160, 1))


def test_packed_nibble_decode():
    codes = torch.arange(16).repeat(32, 2)
    scales = torch.full((2, 32), 0.125, dtype=torch.float16)
    decoded = ref.decode_tm_bank(ref.pack_tm_codes(codes), scales)
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    assert torch.equal(decoded, (values[codes] * 0.125).half())


@pytest.mark.parametrize("m", [1, 5, 8, 16, 20, 40])
@pytest.mark.parametrize("pattern", ["hot", "spread", "invalid"])
def test_plan_coverage(m, pattern):
    ids = torch.arange(m * 10).reshape(m, 10).int()
    if pattern == "hot":
        ids %= 3
    if pattern == "invalid":
        ids[:, 0] = -1
        ids[:, 1] = 512
    plan = ref.grouped_plan_reference(ids)
    found = []
    previous_expert = -1
    for p in range(int(plan[0])):
        count, start = int(plan[ref.PLAN_COUNTS + p]), int(plan[ref.PLAN_STARTS + p])
        expert = int(plan[ref.PLAN_EXPERTS + p])
        rows = plan[ref.PLAN_ROWS + start : ref.PLAN_ROWS + start + count].tolist()
        assert 1 <= count <= 8 and rows == sorted(rows)
        assert expert >= previous_expert
        assert all(int(ids.flatten()[r]) == expert for r in rows)
        found.extend(rows)
        previous_expert = expert
    assert sorted(found) == [
        r for r, e in enumerate(ids.flatten().tolist()) if 0 <= e < 512
    ]


@pytest.mark.parametrize("m", [1, 5, 8, 16, 20, 40])
def test_cpu_moe_against_per_route_scalar_reference(m):
    gen = torch.Generator().manual_seed(m)
    x = (torch.randn(m, 32, generator=gen) * 0.1).half()
    ids = (torch.arange(m * 10).reshape(m, 10) % 3).int()
    ids[:, 0] = -1
    topk = torch.rand(m, 10, generator=gen).float() / 10
    w13 = [(torch.randn(32, 32, generator=gen) * 0.1).half() for _ in range(3)]
    w2 = [(torch.randn(16, 32, generator=gen) * 0.1).half() for _ in range(3)]
    actual, _ = ref.moe_reference(x, ids, topk, w13.__getitem__, w2.__getitem__)
    out = torch.zeros_like(x)
    # Separate token/slot traversal, no plan or batched expert implementation.
    for token in range(m):
        acc = torch.zeros(32, dtype=torch.float32)
        for slot in range(10):
            e = int(ids[token, slot])
            if e < 0:
                continue
            gu = (x[token].double() @ w13[e].double()).float().half()
            g, u = gu.chunk(2)
            act = (
                (g.float() / (1 + (-g.float()).exp())).half().float() * u.float()
            ).half()
            y = (act.double() @ w2[e].double()).float().half()
            acc = (acc.double() + y.double() * topk[token, slot].double()).float()
        out[token] = acc.half()
    assert torch.equal(out, actual)


def test_apply_uses_40_row_persistent_buffers(monkeypatch):
    layer = layer_contract()
    layer._nvfp4_raw_grouped_narrow_plan = None
    layer._nvfp4_raw_grouped_output = torch.empty(40, 2560, dtype=torch.float16)
    layer._nvfp4_raw_grouped_intermediate = torch.empty(400, 160, dtype=torch.float16)
    layer._nvfp4_raw_grouped_plan = torch.empty(ref.PLAN_INTS, dtype=torch.int32)
    for name in (
        "w13_tm_weight",
        "w13_raw_scale_codes",
        "w13_raw_global_scales",
        "w2_tm_weight",
        "w2_raw_scale_codes",
        "w2_raw_global_scales",
    ):
        setattr(layer, name, torch.empty(0))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **kw: True)
    calls = []

    def op(out, intermediate, plan, *args):
        calls.append((out.data_ptr(), intermediate.shape, plan.numel()))
        out.fill_(3)

    monkeypatch.setattr(moe.sm70_ops, "nvfp4_raw_grouped_decode_sm70_out", op)
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    for m in (2, 5, 20, 40):
        out = method.apply(
            layer,
            torch.zeros(m, 2560, dtype=torch.float16),
            torch.zeros(m, 10),
            torch.zeros(m, 10, dtype=torch.int32),
            None,
            None,
        )
        assert torch.all(out == 3)
        assert calls[-1] == (
            layer._nvfp4_raw_grouped_output.data_ptr(),
            (m * 10, 160),
            ref.PLAN_INTS,
        )
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **kw: False)
    with pytest.raises(RuntimeError, match="restricted to CUDA SM70"):
        method.apply(
            layer,
            torch.zeros(40, 2560, dtype=torch.float16),
            torch.zeros(40, 10),
            torch.zeros(40, 10, dtype=torch.int32),
            None,
            None,
        )


def test_weight_rounding_bound_against_unrounded_format():
    codes = torch.arange(1, 127, dtype=torch.uint8).repeat(16, 1)
    global_ = torch.tensor([0.00081317])
    scales = ref.effective_scales(codes, global_)
    # Product of two FP16 RNE boundaries compared to the exact format value.
    fp4 = torch.tensor([0.5, 1, 1.5, 2, 3, 4, 6])[:, None, None]
    actual = (fp4 * scales.float()).half().double()
    exact = fp4.double() * ref.decode_e4m3(codes).double() * global_.double()
    u = 2**-11
    bound = (2 * u + u * u + 2**-23) * exact.abs()
    bound += (fp4.double() * (1 + u) + 1) * 2**-25
    assert torch.all((actual - exact).abs() <= bound)


def test_fake_native_op_when_built():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm import _sm70_ops as ops

    if not ops.has_nvfp4_raw_grouped_decode_dispatch():
        pytest.skip("new native op absent from reused W3 binary")
    with FakeTensorMode():
        x = torch.empty(40, 2560, dtype=torch.float16)
        ops.nvfp4_raw_grouped_decode_sm70_out(
            torch.empty_like(x),
            torch.empty(400, 160, dtype=torch.float16),
            torch.empty(ref.PLAN_INTS, dtype=torch.int32),
            x,
            torch.empty(512, 2560, 40, dtype=torch.int32),
            torch.empty(512, 160, 320, dtype=torch.uint8),
            torch.empty(512, 2),
            torch.empty(512, 160, 320, dtype=torch.int32),
            torch.empty(512, 10, 2560, dtype=torch.uint8),
            torch.empty(512, 1),
            torch.empty(40, 10, dtype=torch.int32),
            torch.empty(40, 10),
        )


@pytest.mark.parametrize("m", [1, 5, 8, 16, 20, 40])
def test_real_shape_cpu_oracle(m):
    generator = torch.Generator().manual_seed(114)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    matrices = []
    for k, n, globals_ in [
        (2560, 320, torch.tensor([0.00013017, 0.00025631])),
        (160, 2560, torch.tensor([0.00017139])),
    ]:
        codes = torch.randint(0, 16, (k, n), generator=generator)
        raw = torch.randint(48, 112, (k // 16, n), generator=generator).byte()
        scales = ref.effective_scales(raw, globals_)
        decoded = ref.decode_tm_bank(ref.pack_tm_codes(codes), scales)
        # Separate built-in FP8 conversion and direct logical-nibble formula.
        factors = globals_.repeat_interleave(n // globals_.numel())
        prepared = (raw.view(torch.float8_e4m3fn).float() * factors).half()
        direct = (
            lut[codes & 7]
            * torch.where((codes & 8) != 0, -1, 1)
            * prepared.float().repeat_interleave(16, 0)
        ).half()
        assert torch.equal(decoded.view(torch.int16), direct.view(torch.int16))
        matrices.append(decoded)
    x = (torch.randn(m, 2560, generator=generator) * 0.1).half()
    ids = torch.zeros(m, 10, dtype=torch.int32)
    weights = torch.rand(m, 10, generator=generator) / 10
    actual, _ = ref.moe_reference(
        x, ids, weights, lambda _: matrices[0], lambda _: matrices[1]
    )
    gate, up = (x.float() @ matrices[0].float()).half().chunk(2, -1)
    act = (
        (gate.float() / (1 + (-gate.float()).exp())).half().float() * up.float()
    ).half()
    route = (act.float() @ matrices[1].float()).half()
    expected = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(10):
        expected = (
            expected.double() + route.double() * weights[:, slot : slot + 1].double()
        ).float()
    error = (actual.float() - expected.half().float()).norm()
    assert error / expected.norm().clamp_min(1e-12) < 5e-3
