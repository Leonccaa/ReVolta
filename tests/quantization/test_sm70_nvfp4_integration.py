# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm import _sm70_ops as ops
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe
from vllm.model_executor.models import qwen2_moe


def test_graph_safe_decode_scratch_sharing(monkeypatch):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)

    # Exercise CUDA ownership with real CPU tensors and a native size stub.
    empty, arange = torch.empty, torch.arange
    monkeypatch.setattr(
        torch, "empty", lambda *args, dtype, device: empty(*args, dtype=dtype)
    )
    monkeypatch.setattr(
        torch, "arange", lambda *args, dtype, device: arange(*args, dtype=dtype)
    )
    monkeypatch.setattr(
        torch.ops._moe_C,
        "moe_permute_sort_workspace_size",
        Mock(return_value=16),
        raising=False,
    )

    def allocate(intermediate):
        layer = NS(
            w13_tm_weight=NS(device=torch.device("cuda:0")),
            sm70_nvfp4_top_k=10,
            sm70_nvfp4_num_experts=512,
            sm70_nvfp4_hidden_size=2560,
            sm70_nvfp4_intermediate_size=intermediate,
            global_num_experts=512,
        )
        method._allocate_graph_safe_decode_buffers(layer)
        return layer

    with set_current_vllm_config(config):
        first, second, other = allocate(160), allocate(160), allocate(320)
        for name in method._persistent_buffers(first, 5):
            if name != "output":
                attr = f"_nvfp4_sm70_{name}"
                scratch = getattr(first, attr)
                assert scratch is getattr(second, attr)
                assert scratch.data_ptr() != getattr(other, attr).data_ptr()
        assert first._nvfp4_sm70_output.data_ptr() != (
            second._nvfp4_sm70_output.data_ptr()
        )

        registry = config.compilation_config.static_forward_context
        pool = registry["sm70_workspace:pool:nvfp4_moe_decode_scratch"]
        assert len(pool) == 2
        moe.clear_sm70_nvfp4_moe_workspaces()
        assert not pool


def test_raw_scale_shared_workspace_rejects_microbatching(monkeypatch):
    monkeypatch.setattr(
        moe,
        "get_current_vllm_config_or_none",
        lambda: NS(parallel_config=NS(use_ubatching=True)),
    )
    allocate = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(NotImplementedError, match="DBO or microbatching"):
        moe._get_qwen38_raw_scale_workspace(torch.device("cuda", 0))
    allocate.assert_not_called()


@pytest.mark.parametrize("raw_scale", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_m1_fusions_never_consume_raw_scale_workspace(monkeypatch, raw_scale, fused):
    # Execute host dispatch on CPU tensors; every GPU operator is intercepted.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **k: True)
    for name in (
        "_use_qwen38_indexed_prefill",
        "_use_grouped_decode",
        "_use_qwen38_qpn_batch_decode",
        "_use_qwen38_qpn_mtp5_decode",
        "_use_qwen38_qpn_batch_fused_w13",
        "_use_qwen38_qpn_batch_fused_w2",
    ):
        monkeypatch.setattr(moe, name, lambda *a: False)
    monkeypatch.setattr(moe, "_use_qwen38_qpn_m1_decode", lambda *a: True)
    monkeypatch.delenv("VLLM_SM70_QWEN38_QPN_ROUTE_DEBUG", raising=False)
    layer = NS(
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_qwen38_fused_swiglu_prefill=True,
        sm70_nvfp4_qwen38_fused_swiglu_decode=fused,
        sm70_nvfp4_qwen38_w2_direct_reduce=fused,
        sm70_nvfp4_qwen38_raw_scale=raw_scale,
        w13_tm_weight=object(),
        w2_tm_weight=object(),
        w13_tm_scales=object(),
        w2_tm_scales=object(),
        w13_raw_scale_codes=object(),
        w2_raw_scale_codes=object(),
        w13_raw_global_scales=object(),
        w2_raw_global_scales=object(),
    )
    buffers = {
        key: torch.empty(1)
        for key in ("output", "intermediate", "gate_up", "sorted_output")
    }
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    method._get_buffers = Mock(return_value=buffers)
    method._apply_swiglu = Mock()
    calls = {}
    for name in (
        "nvfp4_moe_qpn_m1_sm70_out",
        "nvfp4_moe_qpn_raw_scale_sm70_out",
        "nvfp4_qwen38_w13_fused_swiglu_out",
        "nvfp4_qwen38_w2_direct_reduce_out",
    ):
        calls[name] = Mock()
        monkeypatch.setattr(ops, name, calls[name])
    reduce = Mock()
    monkeypatch.setattr(moe, "_single_token_weighted_reduce", reduce)
    result = method.apply(
        layer,
        torch.zeros(1, 2560, dtype=torch.float16),
        torch.ones(1, 10),
        torch.zeros(1, 10, dtype=torch.int32),
        None,
        None,
    )
    assert result is buffers["output"]
    use_fused = fused and not raw_scale
    assert calls["nvfp4_qwen38_w13_fused_swiglu_out"].call_count == int(use_fused)
    assert calls["nvfp4_qwen38_w2_direct_reduce_out"].call_count == int(use_fused)
    assert calls["nvfp4_moe_qpn_raw_scale_sm70_out"].call_count == (
        2 if raw_scale else 0
    )
    assert calls["nvfp4_moe_qpn_m1_sm70_out"].call_count == (
        0 if raw_scale or fused else 2
    )
    assert reduce.call_count == int(not use_fused)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("available", [False, True])
def test_shared_gate_m1_kernel_is_not_called_for_batch(monkeypatch, rows, available):
    monkeypatch.setattr(
        qwen2_moe, "_sm70_dump_qwen_mlp_tensor", lambda label, idx, x: x
    )
    monkeypatch.setattr(ops, "has_qwen38_shared_gate_exact", lambda: available)
    exact = Mock()
    monkeypatch.setattr(ops, "qwen38_shared_gate_exact_out", exact)
    gate = Mock(return_value=(torch.ones(rows, 1, dtype=torch.float16), None))
    gate.weight = torch.zeros(1, 2560, dtype=torch.float16)
    layer = NS(
        layer_idx=0,
        _sm70_exact_shared_expert_gate=True,
        _sm70_batch_shared_expert_gate=False,
        gate_up_proj=NS(forward_fused_silu_and_mul=lambda x: x),
        down_proj=lambda x: (x, None),
        expert_gate=gate,
    )
    x = torch.ones(rows, 2560, dtype=torch.float16)
    result = qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    use_fused = rows == 1 and available
    assert exact.call_count == int(use_fused)
    assert gate.call_count == int(not use_fused)
    if not use_fused:
        torch.testing.assert_close(result, torch.sigmoid(torch.ones_like(x)))


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 17])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("decode_phase", [False, True])
def test_batch_shared_gate_preserves_linear(
    monkeypatch, rows, enabled, available, decode_phase
):
    monkeypatch.setattr(
        qwen2_moe, "_sm70_dump_qwen_mlp_tensor", lambda label, idx, x: x
    )
    monkeypatch.setattr(
        qwen2_moe, "use_sm70_decode_graph_semantics", lambda: decode_phase
    )
    monkeypatch.setattr(ops, "has_qwen38_shared_gate_exact", lambda: False)
    monkeypatch.setattr(ops, "has_qwen38_shared_gate_sigmoid_mul", lambda: available)
    fused = Mock(side_effect=lambda out, gate: out.mul_(torch.sigmoid(gate)))
    monkeypatch.setattr(ops, "qwen38_shared_gate_sigmoid_mul_out", fused)
    logits = torch.linspace(-3, 3, rows, dtype=torch.float16).reshape(rows, 1)
    gate = Mock(return_value=(logits, None))
    layer = NS(
        layer_idx=0,
        _sm70_exact_shared_expert_gate=True,
        _sm70_batch_shared_expert_gate=enabled,
        gate_up_proj=NS(forward_fused_silu_and_mul=lambda x: x.clone()),
        down_proj=lambda x: (x, None),
        expert_gate=gate,
    )
    x = torch.ones(rows, 2560, dtype=torch.float16)
    result = qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    gate.assert_called_once_with(x)
    assert fused.call_count == int(
        enabled and available and decode_phase and 2 <= rows <= 16
    )
    torch.testing.assert_close(result, x * torch.sigmoid(logits), atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 2, 8, 9, 64, 65])
@pytest.mark.parametrize("mtp", [False, True])
def test_mtp_shared_down_gated_route(monkeypatch, rows, mtp):
    # Registers the vllm shared-expert ops used by the MTP fallback.
    import vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv  # noqa: F401

    monkeypatch.setattr(
        qwen2_moe, "_sm70_dump_qwen_mlp_tensor", lambda label, idx, x: x
    )
    monkeypatch.setattr(qwen2_moe, "use_sm70_decode_graph_semantics", lambda: True)
    monkeypatch.setattr(ops, "has_qwen38_shared_gate_exact", lambda: False)
    monkeypatch.setattr(ops, "has_qwen38_shared_gate_sigmoid_mul", lambda: False)
    torch.manual_seed(0)
    x = (torch.randn(rows, 2560) * 0.1).half()
    act = (torch.randn(rows, 160) * 0.1).half()
    down_w = (torch.randn(2560, 160) * 0.05).half()
    gate_w = (torch.randn(1, 2560) * 0.05).half()

    def matmul(a, w):
        return (a.float() @ w.float().t()).half()

    def fused_op(x, act, dw, gw):
        return (matmul(act, dw).float() * torch.sigmoid(matmul(x, gw).float())).half()

    fused = Mock(side_effect=fused_op)
    monkeypatch.setattr(
        torch.ops.vllm, "qwen38_sm70_shared_down_gated", fused, raising=False
    )
    down = Mock(side_effect=lambda a: (matmul(a, down_w), None))
    down.weight = down_w
    gate = Mock(side_effect=lambda a: (matmul(a, gate_w), None))
    gate.weight = gate_w
    layer = NS(
        layer_idx=0,
        _sm70_exact_shared_expert_gate=True,
        _sm70_batch_shared_expert_gate=True,
        gate_up_proj=NS(
            forward_fused_silu_and_mul=lambda x: act,
            _sm70_mtp_prepare_shared_batch=mtp,
        ),
        down_proj=down,
        expert_gate=gate,
    )
    result = qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    use_fused = mtp and 2 <= rows <= 64
    assert fused.call_count == int(use_fused)
    assert down.call_count == int(not use_fused)
    assert result.shape == (rows, 2560)
    torch.testing.assert_close(result, fused_op(x, act, down_w, gate_w))


@pytest.mark.parametrize("rows", [2, 8, 9, 64])
def test_mtp_shared_down_gated_op_dispatch(monkeypatch, rows):
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv

    fused, dot = Mock(), Mock()
    monkeypatch.setattr(
        torch.ops._C, "qwen38_shared_down_gated_sm70_out", fused, raising=False
    )
    monkeypatch.setattr(
        torch.ops._C, "qwen38_shared_gate_dot_mul_sm70_out", dot, raising=False
    )
    x = torch.zeros(rows, 2560, dtype=torch.float16)
    act = torch.zeros(rows, 160, dtype=torch.float16)
    down_w = torch.zeros(2560, 160, dtype=torch.float16)
    gate_w = torch.zeros(1, 2560, dtype=torch.float16)
    out = gemv._qwen38_sm70_shared_down_gated(x, act, down_w, gate_w)
    assert out.shape == (rows, 2560)
    # 1Cat v1.5.4: fused down+gate up to M8, then down GEMM + gate dot/mul.
    assert fused.call_count == int(rows <= 8)
    assert dot.call_count == int(rows > 8)
