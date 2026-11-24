# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CompilationMode, KernelConfig
from vllm.model_executor import parameter
from vllm.model_executor.kernels.linear.scaled_mm import sm70_fp8
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    sm70_mtp_draft_qpn8 as draft_qpn8,
)
from vllm.model_executor.layers.quantization import sm70_online_qpn8 as online_qpn8
from vllm.models.qwen4_exp.nvidia import mtp, sm70_fp16_gemv


@pytest.fixture(autouse=True)
def single_tp_rank(monkeypatch):
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)


def _linear(prefix, *, bias=False):
    return ReplicatedLinear(
        128, 64, prefix=prefix, bias=bias, params_dtype=torch.half, disable_tp=True
    )


def test_mark_only_unquantized_bias_free_draft_projections():
    admitted = [
        _linear(f"mtp.layers.0.{suffix}")
        for suffix in (
            "fc",
            "fc_embedding",
            "fc_hidden",
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        )
    ]
    quantized = _linear("mtp.layers.0.mlp.down_proj")
    quantized.quant_method = object()
    not_linear = nn.Identity()
    not_linear.prefix = "mtp.fc"
    skipped = [
        _linear("mtp.layers.0.self_attn.o_proj", bias=True),
        _linear("mtp.layers.0.self_attn.extra_qkv_proj"),
        quantized,
        not_linear,
    ]
    assert draft_qpn8.mark_mtp_draft_qpn8(nn.ModuleList(admitted + skipped)) == 7
    assert all(getattr(layer, draft_qpn8._MARK, False) for layer in admitted)
    assert not any(getattr(layer, draft_qpn8._MARK, False) for layer in skipped)


@pytest.mark.parametrize("enabled", [True, False])
def test_draft_constructor_respects_switch_and_fp16_gemv(monkeypatch, enabled):
    projection = _linear("mtp.layers.0.self_attn.qkv_proj")
    predictor = nn.Module()
    predictor.layers = nn.ModuleList([projection])
    predictor.embed_tokens = nn.Identity()
    predictor.make_empty_intermediate_tensors = lambda *a: {}
    monkeypatch.setattr(mtp, "Qwen4ExpMultiTokenPredictor", lambda **kw: predictor)
    monkeypatch.setattr(mtp, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(mtp.Qwen4ExpMTP, "set_moe_parameters", lambda *a: None)
    monkeypatch.setattr(mtp, "enable_qwen4_exp_low_latency_gemm", lambda *a: None)
    monkeypatch.setattr(mtp, "enable_qwen38_sm70_fp16_fused_hc", lambda *a: None)
    monkeypatch.setattr(sm70_fp16_gemv, "_exact_runtime_contract", lambda _: True)
    monkeypatch.setattr(
        sm70_fp16_gemv,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda _: True),
    )
    monkeypatch.setattr(sm70_fp16_gemv.envs, "VLLM_SM70_QWEN38_FP16_GEMV", True)
    monkeypatch.setattr(sm70_fp16_gemv.envs, "VLLM_SM70_QWEN38_BATCH_FASTPATH", False)
    monkeypatch.setattr(sm70_fp16_gemv.envs, "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(tie_word_embeddings=True, vocab_size=256),
            dtype=torch.float16,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        quant_config=None,
        speculative_config=object(),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        kernel_config=KernelConfig()
        if enabled
        else KernelConfig(sm70_mtp_draft_qpn8=False),
    )

    assert mtp.Qwen4ExpMTP(vllm_config=config).model is predictor
    assert getattr(projection, draft_qpn8._MARK, False) is enabled
    assert type(projection.quant_method) is (
        UnquantizedLinearMethod
        if enabled
        else sm70_fp16_gemv.Qwen38SM70FP16LinearMethod
    )


@pytest.mark.parametrize("sm70", [False, True], ids=["non-sm70", "missing-op"])
def test_prepare_falls_back_to_fp16(monkeypatch, sm70):
    layer = _linear("mtp.fc")
    original = layer.weight
    with torch.no_grad():
        original.fill_(1)
    draft_qpn8.mark_mtp_draft_qpn8(layer)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda t: t is original))
    monkeypatch.setattr(
        draft_qpn8,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda cap: sm70),
    )
    monkeypatch.setattr(
        torch.ops, "_C", SimpleNamespace(fp8_qpn8_prepare_sm70=object())
    )
    monkeypatch.setattr(online_qpn8.envs, "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", False)
    assert not online_qpn8.maybe_prepare_online_qpn8(layer)
    assert layer.weight is original
    x = torch.ones((1, 128), dtype=torch.float16)
    assert online_qpn8.maybe_apply_online_qpn8(layer, x, None) is None
    torch.testing.assert_close(layer(x)[0], torch.full((1, 64), 128, dtype=x.dtype))


def test_prepare_and_dispatch_draft_gate_up(monkeypatch):
    # Adapted from the 1Cat QPN8 dispatch and fused gate-up tests.
    layer = MergedColumnParallelLinear(
        128,
        [32, 32],
        bias=False,
        params_dtype=torch.float16,
        prefix="mtp.layers.0.mlp.gate_up_proj",
        disable_tp=True,
    )
    original = layer.weight
    draft_qpn8.mark_mtp_draft_qpn8(layer)
    codes = torch.zeros((128, 64), dtype=torch.uint8)
    scales = torch.ones((1, 64), dtype=torch.float16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda t: t is original))
    monkeypatch.setattr(
        draft_qpn8,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda cap: cap == 70),
    )
    monkeypatch.setattr(
        torch.ops, "_C", SimpleNamespace(**dict.fromkeys(draft_qpn8._REQUIRED_OPS))
    )
    monkeypatch.setattr(
        sm70_fp8,
        "_get_sm70_fp8_prefill_exact_dense_workspace",
        lambda _: torch.empty(1),
    )
    monkeypatch.setattr(
        online_qpn8, "prepare_channel_qpn8_weight", lambda _: (codes, scales)
    )
    monkeypatch.setattr(online_qpn8.envs, "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", False)

    assert online_qpn8.maybe_prepare_online_qpn8(layer)
    assert layer.weight.dtype == torch.uint8 and layer.weight is not original
    assert layer._sm70_mtp_draft_qpn8_scales is scales
    calls = []

    def dispatch(out, x, weight, packed_scales, split_k, nacc, prefetch, gated):
        assert x.shape == (6, 128) and x.is_contiguous()
        assert weight is layer.weight and packed_scales is scales
        calls.append((split_k, nacc, prefetch, gated))
        out.fill_(2)

    monkeypatch.setattr(torch.ops.vllm, "sm70_ct_fp8_qpn8_dispatch", dispatch)
    x = torch.ones((2, 3, 128), dtype=torch.float16)
    torch.testing.assert_close(layer(x)[0], torch.full((2, 3, 64), 2, dtype=x.dtype))
    torch.testing.assert_close(
        layer.forward_fused_silu_and_mul(x), torch.full((2, 3, 32), 2, dtype=x.dtype)
    )
    assert calls == [(8, 2, False, False), (8, 2, False, True)]
