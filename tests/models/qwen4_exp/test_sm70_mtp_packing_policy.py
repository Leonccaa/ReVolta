# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU loader/dispatch regressions; no GPU arithmetic or capacity claim."""

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
from vllm.sm70_profiles.acceleration import loaded_sm70_preparations


@pytest.fixture(autouse=True)
def _r2_opt_in(monkeypatch):
    # The release configuration keeps 1CatAI #919 opt-in; exercise the new path here.
    monkeypatch.setenv("VLLM_SM70_SKIP_REJECTED_BATCH_PACKS", "1")


@pytest.fixture(autouse=True)
def restore_precision():
    backend = torch.backends.cuda.matmul
    names = (
        "allow_fp16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    )
    before = [getattr(backend, name) for name in names]
    envs.disable_envs_cache()
    yield
    for name, value in zip(names, before):
        setattr(backend, name, value)
    envs.disable_envs_cache()


def make_layer(role):
    layer = torch.nn.Module()
    rows = 512 if role == "router" else 320
    # Exactly representable values let packing tests inspect every FP16 bit.
    weight = (torch.arange(rows * 2560) % 257 - 128).to(torch.float16)
    layer.weight = torch.nn.Parameter(weight.reshape(rows, 2560), requires_grad=False)
    setattr(layer, f"_sm70_mtp_prepare_{role}_batch", True)
    if role == "shared":
        layer.forward_fused_silu_and_mul = MethodType(
            gemv._forward_shared_batch_silu, layer
        )
    return layer


def prepare_on_cpu(layer, monkeypatch, *, native_available=True):
    # Exercise the real loader and packing on CPU storage. Only placement and
    # the unrelated base linear preparation are stubbed; no native op executes.
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        m.setattr(
            UnquantizedLinearMethod, "process_weights_after_loading", lambda *args: None
        )
        if not native_available:
            m.setattr(torch.ops, "_C", SimpleNamespace())
        for name in (
            "qwen38_router_batch_sm70_out",
            "qwen38_shared_up_batch_sm70_out",
        ):
            if native_available:
                m.setattr(torch.ops._C, name, lambda *args: None, raising=False)
        gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)


@pytest.mark.parametrize(
    "reduced,accumulation,reasons",
    [
        # Shared up keeps FP32 partials without reduced-precision reduction.
        (False, False, {"router": "fp16_reduced_precision_reduction_disabled"}),
        (True, True, dict.fromkeys(("router", "shared"), "fp16_accumulation_enabled")),
        (True, False, {}),
    ],
)
def test_precision_skip_info_once_per_role(reduced, accumulation, reasons, monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
    torch.backends.cuda.matmul.allow_fp16_accumulation = accumulation
    # A dedicated logger exercises the real info_once deduplication without
    # depending on earlier loader tests or an initialized distributed group.
    logger = init_logger(f"test_mtp_packing_policy.{reduced}.{accumulation}")
    messages = []
    monkeypatch.setattr(
        logger, "info", lambda msg, *args, **kwargs: messages.append(msg % args)
    )
    monkeypatch.setattr(gemv, "logger", logger)
    for role in ("router", "shared"):
        # Target and draft layers share the process/role precision policy.
        reason = reasons.get(role)
        for _ in range(2):
            layer = torch.nn.Module()
            assert gemv._mtp_batch_packing_allowed(layer, role) == (reason is None)
            assert getattr(layer, f"_sm70_mtp_{role}_batch_reason") == reason
    expected = [
        f"Skipping SM70 MTP {role} packed weights due to precision policy: {reason}."
        for role, reason in reasons.items()
    ]
    assert messages == expected


@pytest.mark.parametrize("role", ["router", "shared"])
@pytest.mark.parametrize(
    "reduced,accumulation", [(False, False), (False, True), (True, False), (True, True)]
)
def test_loader_skips_only_precision_rejected_packs(
    role, reduced, accumulation, monkeypatch
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
    torch.backends.cuda.matmul.allow_fp16_accumulation = accumulation
    layer = make_layer(role)
    weight = layer.weight
    before = weight.detach().clone()
    hook = getattr(layer, "forward_fused_silu_and_mul", None)
    prepare_on_cpu(layer, monkeypatch)

    admitted = (reduced or role == "shared") and not accumulation
    buffer_name = f"_sm70_mtp_{role}_packed"
    assert hasattr(layer, buffer_name) == admitted
    assert layer.weight is weight and torch.equal(layer.weight, before)
    assert getattr(layer, f"_sm70_mtp_prepare_{role}_batch")
    assert getattr(layer, "forward_fused_silu_and_mul", None) is hook
    reason_name = f"_sm70_mtp_{role}_batch_reason"
    report = loaded_sm70_preparations(torch.nn.Sequential(layer))["variants"]["0"]
    if admitted:
        packed = getattr(layer, buffer_name)
        if role == "router":
            restored = packed.permute(0, 4, 3, 1, 2, 5).reshape_as(weight)
        else:
            restored = packed.permute(0, 3, 1, 2, 4).reshape_as(weight)
        assert torch.equal(restored.view(torch.int16), before.view(torch.int16))
        assert buffer_name not in layer.state_dict()
        assert report["prepared_buffers"] == [buffer_name]
        assert reason_name not in report["reasons"]
    else:
        expected = (
            "fp16_reduced_precision_reduction_disabled"
            if not reduced and role == "router"
            else "fp16_accumulation_enabled"
        )
        assert report["reasons"][reason_name] == expected
        assert report["prepared_buffers"] == []


@pytest.mark.parametrize("role", ["router", "shared"])
def test_rejected_pack_does_not_require_native_operator(role, monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    layer = make_layer(role)
    prepare_on_cpu(layer, monkeypatch, native_available=False)
    assert not hasattr(layer, f"_sm70_mtp_{role}_packed")


@pytest.mark.parametrize("role", ["router", "shared"])
def test_admitted_pack_still_requires_native_operator(role, monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    with pytest.raises(RuntimeError, match="Rebuild the SM70 extension"):
        prepare_on_cpu(make_layer(role), monkeypatch, native_available=False)


@pytest.mark.parametrize("role", ["router", "shared"])
def test_admitted_pack_remains_reloadable(role, monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    layer = make_layer(role)
    prepare_on_cpu(layer, monkeypatch)
    old = getattr(layer, f"_sm70_mtp_{role}_packed").clone()
    layer.weight.data.mul_(0.5)
    prepare_on_cpu(layer, monkeypatch)
    assert torch.equal(getattr(layer, f"_sm70_mtp_{role}_packed"), old * 0.5)


@pytest.mark.parametrize("rows", [5, 10])
def test_rejected_policy_fallback_is_unchanged_with_or_without_pack(rows, monkeypatch):
    # Model the pre-fix resident pack, then compare against the skipped pack at
    # the same precision policy. Assert the original F.linear fallback executes.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    layer = make_layer("router")
    prepare_on_cpu(layer, monkeypatch)
    packed = layer._sm70_mtp_router_packed
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    monkeypatch.setenv("VLLM_SM70_MTP_ROUTER_BATCH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    x = torch.linspace(-0.01, 0.01, rows * 2560, dtype=torch.float16).reshape(
        rows, 2560
    )
    linear = F.linear
    calls = []

    def record_linear(*args, **kwargs):
        calls.append(args[0].shape)
        return linear(*args, **kwargs)

    monkeypatch.setattr(F, "linear", record_linear)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        assert not gemv._router_batch_runtime_ok(x, packed)
        before = gemv._qwen38_sm70_fp16_gemv(
            x, layer.weight, "model.layers.0.mlp.gate", packed
        )
        after = gemv._qwen38_sm70_fp16_gemv(
            x, layer.weight, "model.layers.0.mlp.gate", None
        )
    assert calls == [x.shape, x.shape]
    assert torch.equal(before.view(torch.int16), after.view(torch.int16))


def _shared_up_fp32_reference(out, partial, x, packed):
    # Read the pack exactly as shared_up_fused_silu_kernel indexes it:
    # packed[n32_block, k16_group, k8_half, row_in_block, k].
    assert partial.dtype == torch.float32 and partial.numel() == 0
    assert packed.shape == (10, 160, 2, 32, 8)
    weight = packed.permute(0, 3, 1, 2, 4).reshape(320, 2560).float()
    gate_up = x.float() @ weight.t()
    gate = gate_up[:, :160].half().float()
    up = gate_up[:, 160:].half()
    out.copy_((F.silu(gate).half() * up).to(out.dtype))


@pytest.mark.parametrize("rows", [1, 2, 4, 16, 64, 65])
def test_shared_fp32_batch_routes_mtp_rows(rows, monkeypatch):
    # Exact policy (no reduced-precision reduction) now packs shared up and
    # uses the FP32-partial kernel for verify rows 2..64.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    layer = make_layer("shared")
    layer.weight.data.mul_(1 / 4096)
    prepare_on_cpu(layer, monkeypatch)
    packed = layer._sm70_mtp_shared_packed
    monkeypatch.setenv("VLLM_SM70_MTP_SHARED_BATCH", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    x = torch.linspace(-0.5, 0.5, rows * 2560).reshape(rows, 2560).half()
    native = []
    linear = F.linear

    def record_native(*args):
        native.append(args[2].shape)
        _shared_up_fp32_reference(*args)

    def cpu_silu_and_mul(out, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        out.copy_(F.silu(gate.float()).half() * up)

    monkeypatch.setattr(
        torch.ops._C,
        "qwen38_shared_up_batch_fp32_sm70_out",
        record_native,
        raising=False,
    )
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", cpu_silu_and_mul, raising=False)
    monkeypatch.setattr(F, "linear", lambda a, w: linear(a.float(), w.float()).half())
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        actual = gemv._qwen38_sm70_shared_up(x, layer.weight, packed)
        fallback = gemv._qwen38_sm70_shared_up(x, layer.weight, None)
    assert native == ([x.shape] if 2 <= rows <= 64 else [])
    assert actual.shape == (rows, 160)
    torch.testing.assert_close(actual, fallback, atol=0, rtol=0)


def _shared_model(monkeypatch):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    layer = make_layer("shared")
    prepare_on_cpu(layer, monkeypatch)
    model = torch.nn.Module()
    model.gate_up_proj = layer
    return model, layer


def _assert_pack_refreshed(layer, packed, ptr):
    assert layer._sm70_mtp_shared_packed is packed
    assert packed.data_ptr() == ptr
    expected = layer.weight.reshape(10, 32, 160, 2, 8).permute(0, 2, 3, 1, 4)
    assert torch.equal(packed, expected)


def test_kernel_format_reload_refreshes_shared_pack(monkeypatch):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    model, layer = _shared_model(monkeypatch)
    packed = layer._sm70_mtp_shared_packed
    ptr = packed.data_ptr()
    runner = SimpleNamespace(
        get_model=lambda: model, model_config=SimpleNamespace(quantization=None)
    )
    new_weight = torch.flip(layer.weight.detach(), dims=(1,)).clone()
    GPUModelRunner.reload_weights(
        runner,
        weights_iterator=[("gate_up_proj.weight", new_weight)],
        is_checkpoint_format=False,
    )
    assert torch.equal(layer.weight, new_weight)
    _assert_pack_refreshed(layer, packed, ptr)


def test_kernel_format_weight_update_refreshes_shared_pack(monkeypatch):
    from vllm.v1.worker.gpu_worker import Worker

    model, layer = _shared_model(monkeypatch)
    packed = layer._sm70_mtp_shared_packed
    ptr = packed.data_ptr()
    new_weight = torch.flip(layer.weight.detach(), dims=(1,)).clone()
    engine = SimpleNamespace(
        parse_update_info=lambda info: info,
        receive_weights=lambda info, load_weights: load_weights(info),
    )
    worker = SimpleNamespace(
        weight_transfer_engine=engine,
        _check_weight_transfer_engine=lambda: None,
        _weight_update_active=True,
        _is_checkpoint_format=False,
        model_runner=SimpleNamespace(model=model),
        device="cpu",
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    Worker.update_weights(worker, [("gate_up_proj.weight", new_weight)])
    assert torch.equal(layer.weight, new_weight)
    _assert_pack_refreshed(layer, packed, ptr)
