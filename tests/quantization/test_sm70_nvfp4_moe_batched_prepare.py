# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU bitwise checks for the batched SM70 NVFP4 MoE weight preparation.

The chunked prepare must hand the per-expert TurboMind prepare op and the
raw E4M3 scale tables bitwise-identical results compared with the
pre-batching per-expert loop. The native ops do not exist on CPU, so they
are stubbed and the surrounding torch arithmetic is compared.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as nvfp4
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    NVFP4_GROUP_SIZE,
    unpack_mxfp4_weight,
)

pytestmark = pytest.mark.skip_global_cleanup


def _make_layer(num_experts, hidden, intermediate, *, tp_size=1, seed=0):
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(
        tp_size=tp_size,
        hidden_dim=hidden,
        intermediate_size_per_partition=intermediate,
        experts_per_token=10,
    )
    layer.local_num_experts = layer.global_num_experts = num_experts
    layer.activation = nvfp4.MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.expert_map = layer.swiglu_limit = None
    generator = torch.Generator().manual_seed(seed)
    e, h, i = num_experts, hidden, intermediate

    def parameter(shape, dtype):
        if dtype == torch.uint8:
            value = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
        else:
            value = (torch.randn(shape, generator=generator).abs() + 0.25).to(dtype)
        return torch.nn.Parameter(value, requires_grad=False)

    layer.w13_weight = parameter((e, 2 * i, h // 2), torch.uint8)
    layer.w13_weight_scale = parameter(
        (e, 2 * i, h // NVFP4_GROUP_SIZE), torch.float8_e4m3fn
    )
    layer.w13_weight_scale_2 = parameter((e, 2), torch.float32)
    layer.w13_input_scale = parameter((e,), torch.float32)
    layer.w2_weight = parameter((e, h, i // 2), torch.uint8)
    layer.w2_weight_scale = parameter(
        (e, h, i // NVFP4_GROUP_SIZE), torch.float8_e4m3fn
    )
    layer.w2_weight_scale_2 = parameter((e,), torch.float32)
    layer.w2_input_scale = parameter((e,), torch.float32)
    return layer


def _install_stubs(monkeypatch, calls, *, raw_scale=False, fused_swiglu=False):
    class EnvFlags:
        def __init__(self, **overrides):
            self.__dict__.update(overrides)

        def __getattr__(self, name):
            return False

    env = EnvFlags(
        VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=raw_scale,
        VLLM_SM70_NVFP4_QWEN38_MOE_FUSED_SWIGLU_PREFILL=fused_swiglu,
    )
    monkeypatch.setattr(nvfp4, "envs", env)
    for owner, name in (
        (torch.ops._C, "nvfp4_sm70_prepare"),
        (torch.ops._C, "nvfp4_moe_dense_stage_sm70_out"),
        (torch.ops._C, "awq_moe_build_strided_ptrs"),
        (torch.ops._C, "nvfp4_moe_indexed_fused_swiglu_sm70_out"),
        (torch.ops._moe_C, "moe_permute_with_scratch"),
    ):
        monkeypatch.setattr(owner, name, Mock(), raising=False)
    monkeypatch.setattr(nvfp4, "validate_nvfp4_sm70_moe_contract", lambda _: None)
    monkeypatch.setattr(
        nvfp4.sm70_ops, "has_nvfp4_qpn_raw_scale_dispatch", lambda: raw_scale
    )

    def prepare(weight, scales, group_size, **kwargs):
        # Stand-in for the native repack: the test compares the captured
        # inputs, so a deterministic clone preserves bitwise intent.
        assert group_size == NVFP4_GROUP_SIZE
        calls.append((weight, scales))
        return weight.clone(), scales.clone(), torch.tensor([64, 64])

    monkeypatch.setattr(nvfp4.sm70_ops, "nvfp4_sm70_prepare", prepare)
    monkeypatch.setattr(
        nvfp4.sm70_ops,
        "awq_moe_build_strided_ptrs",
        lambda weight, scales, k_ld, q_ld, experts: (
            torch.zeros(experts, 8, dtype=torch.uint8),
            torch.zeros(experts, 8, dtype=torch.uint8),
        ),
    )
    if raw_scale:
        # The expansion check runs a CUDA-only op; the test compares the
        # raw code tables themselves against the per-expert reference.
        workspace = torch.zeros(
            nvfp4._QWEN38_RAW_SCALE_WORKSPACE_ELEMENTS, dtype=torch.float16
        )
        monkeypatch.setattr(nvfp4, "_raw_scales_match_prepared", lambda *_: True)
        monkeypatch.setattr(
            nvfp4, "_get_qwen38_raw_scale_workspace", lambda device: workspace
        )
    method = SimpleNamespace(
        moe=SimpleNamespace(has_bias=False),
        _prepare_grouped_decode=lambda layer: None,
        _prepare_raw_grouped_decode=lambda layer: None,
        _allocate_graph_safe_decode_buffers=lambda layer: None,
    )
    return nvfp4.ModelOptNvFp4SM70MoEMethod.process_weights_after_loading, method


def _reference_per_expert(
    layer, num_experts, intermediate, *, fused_swiglu=False, raw_scale=False
):
    """The pre-batching preparation loop, verbatim except for stubbed ops."""
    result: dict[str, list[torch.Tensor]] = {
        f"{projection}_{kind}": []
        for projection in ("w13", "w2")
        for kind in ("inputs", "raw_codes", "raw_globals")
    }

    def raw_codes(scales_half, physical_global_1d):
        return (
            (scales_half.float() / physical_global_1d[None, :].float())
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .contiguous()
        )

    for e in range(num_experts):
        w13_packed = unpack_mxfp4_weight(layer.w13_weight[e].data)
        w13_scales = layer.w13_weight_scale[e].float().clone()
        w13_global = layer.w13_weight_scale_2[e].float()
        w13_scales[:intermediate].mul_(w13_global[0])
        w13_scales[intermediate:].mul_(w13_global[1])
        w13_half = w13_scales.half().t().contiguous()
        result["w13_inputs"].append((w13_packed, w13_half))
        w2_packed = unpack_mxfp4_weight(layer.w2_weight[e].data)
        w2_global = layer.w2_weight_scale_2[e].float().reshape(())
        w2_half = (layer.w2_weight_scale[e].float() * w2_global).half().t().contiguous()
        result["w2_inputs"].append((w2_packed, w2_half))
        if raw_scale:
            if fused_swiglu:
                physical_global = w13_global.repeat(intermediate)
            else:
                physical_global = w13_global.repeat_interleave(intermediate)
            result["w13_raw_codes"].append(raw_codes(w13_half, physical_global))
            result["w13_raw_globals"].append(w13_global.contiguous())
            result["w2_raw_codes"].append(raw_codes(w2_half, w2_global.reshape(1)))
            result["w2_raw_globals"].append(w2_global.reshape(1))
    return result


def _compare(layer, calls, reference, raw_scale=False):
    # The stubbed prepare clones its inputs, so comparing the captured
    # call inputs and the raw scale tables covers every prepared tensor.
    def bitwise_equal(actual, expected):
        # Values are positive finite or uint8 codes, so torch.equal here
        # is a bitwise comparison with no -0.0/NaN ambiguity.
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)

    assert len(calls) == 2 * len(reference["w13_inputs"])
    for projection, captured in (("w13", calls[0::2]), ("w2", calls[1::2])):
        for actual, (packed, scales_half) in zip(
            captured, reference[f"{projection}_inputs"]
        ):
            bitwise_equal(actual[0], packed)
            bitwise_equal(actual[1], scales_half)
    if raw_scale:
        for attr, key in (
            ("w13_raw_scale_codes", "w13_raw_codes"),
            ("w13_raw_global_scales", "w13_raw_globals"),
            ("w2_raw_scale_codes", "w2_raw_codes"),
            ("w2_raw_global_scales", "w2_raw_globals"),
        ):
            bitwise_equal(getattr(layer, attr), torch.stack(reference[key]))


def test_batched_prepare_matches_per_expert_loop(monkeypatch):
    # 70 experts exercise one full 64-expert chunk plus a partial tail.
    num_experts, hidden, intermediate = 70, 64, 32
    layer = _make_layer(num_experts, hidden, intermediate, seed=1)
    calls: list = []
    load, method = _install_stubs(monkeypatch, calls)
    reference = _reference_per_expert(layer, num_experts, intermediate)
    load(method, layer)
    _compare(layer, calls, reference)
    assert bool(layer.sm70_nvfp4_qwen38_fused_swiglu_prefill) is False
    assert bool(layer.sm70_nvfp4_qwen38_raw_scale) is False


def test_perturbed_weight_is_detected(monkeypatch):
    # Reverse check: flip one packed byte, run the batched prepare, and
    # confirm the same _compare rejects the mismatch against the
    # reference made from the untouched weights.
    num_experts, hidden, intermediate = 4, 64, 32
    layer = _make_layer(num_experts, hidden, intermediate, seed=2)
    reference = _reference_per_expert(layer, num_experts, intermediate)
    layer.w13_weight.data[0, 3, 5] ^= 0x10
    calls: list = []
    load, method = _install_stubs(monkeypatch, calls)
    load(method, layer)
    with pytest.raises(AssertionError):
        _compare(layer, calls, reference)


def test_batched_prepare_matches_qwen38_raw_scale_layer(monkeypatch):
    # The raw-scale and fused-SwiGLU routes only engage at the exact
    # Qwen3.8-Flash-Next TP4 shape, so this case uses the real dimensions.
    num_experts, hidden, intermediate = 512, 2560, 160
    layer = _make_layer(num_experts, hidden, intermediate, tp_size=4, seed=3)
    calls: list = []
    load, method = _install_stubs(monkeypatch, calls, raw_scale=True, fused_swiglu=True)
    reference = _reference_per_expert(
        layer, num_experts, intermediate, fused_swiglu=True, raw_scale=True
    )
    load(method, layer)
    _compare(layer, calls, reference, raw_scale=True)
    assert bool(layer.sm70_nvfp4_qwen38_fused_swiglu_prefill) is True
    assert bool(layer.sm70_nvfp4_qwen38_raw_scale) is True
