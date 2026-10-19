# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for vision-only NVFP4 expansion across TP group boundaries."""

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm.config import MultiModalConfig
from vllm.config.mm_encoder_tp import (
    qwen3_vision_tp_linears,
    resolve_mm_encoder_tp_mode,
)
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.quantization import modelopt, sm70_turbomind

pytestmark = pytest.mark.skip_global_cleanup
PREFIX = "model.visual.blocks.0.mlp.linear_fc2"


def quant_config(algo="W4A16_NVFP4", group=16, exclude=()):
    return modelopt.ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "group_size": group,
            "ignore": list(exclude),
            "quantized_layers": {
                f"model.visual.blocks.{i}.mlp.linear_fc{j}": {
                    "quant_algo": algo,
                    "group_size": group,
                }
                for i in range(27)
                for j in (1, 2)
            },
        }
    )


def config(mode="weights", explicit=False, **kwargs):
    vision = SimpleNamespace(
        hidden_size=1152,
        intermediate_size=4304,
        depth=27,
        spatial_merge_size=2,
        out_hidden_size=2560,
        deepstack_visual_indexes=[],
    )
    return SimpleNamespace(
        quant_config=quant_config(**kwargs),
        is_speculative_draft=False,
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            architecture="Qwen4ExpForConditionalGeneration",
            hf_config=SimpleNamespace(vision_config=vision),
            multimodal_config=MultiModalConfig(mm_encoder_tp_mode=mode),
            _mm_encoder_tp_mode_explicit=explicit,
            _model_info=SimpleNamespace(supports_multimodal_encoder_tp_data=True),
        ),
    )


@pytest.fixture(autouse=True)
def sm70(monkeypatch):
    monkeypatch.setattr(sm70_turbomind, "is_exact_sm70_cuda_platform", lambda: True)
    # Select production quantizers without claiming a CPU can run Marlin.
    # Its GEMM is never invoked in this suite.
    monkeypatch.setattr(
        modelopt.MarlinNvFp4LinearKernel, "is_supported", lambda *a: (True, None)
    )
    monkeypatch.setattr(
        parameter,
        "get_tensor_model_parallel_rank",
        lambda: linear.get_tensor_model_parallel_rank(),
    )
    monkeypatch.setattr(
        parameter,
        "get_tensor_model_parallel_world_size",
        lambda: linear.get_tensor_model_parallel_world_size(),
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_sm70_tp4_automatically_keeps_weights(explicit):
    cfg = config(explicit=explicit)
    resolve_mm_encoder_tp_mode(cfg)
    assert cfg.model_config.multimodal_config.mm_encoder_tp_mode == "weights"


def test_data_is_explicitly_available():
    cfg = config(mode="data", explicit=True)
    resolve_mm_encoder_tp_mode(cfg)
    assert cfg.model_config.multimodal_config.mm_encoder_tp_mode == "data"


@pytest.mark.parametrize("case", ["w4a4", "group128", "bf16", "tp8", "non_sm70"])
def test_unsupported_combinations_keep_fallback(case, monkeypatch):
    cfg = config(
        algo="NVFP4" if case == "w4a4" else "W4A16_NVFP4",
        group=128 if case == "group128" else 16,
    )
    if case == "bf16":
        cfg.model_config.dtype = torch.bfloat16
    if case == "tp8":
        cfg.parallel_config.tensor_parallel_size = 8
    if case == "non_sm70":
        monkeypatch.setattr(
            sm70_turbomind, "is_exact_sm70_cuda_platform", lambda: False
        )
    # g128 is already invalid before TP (4304 % 128 != 0).
    if case == "group128":
        with pytest.raises(ValueError, match="Invalid quantized vision layer"):
            resolve_mm_encoder_tp_mode(cfg)
    else:
        resolve_mm_encoder_tp_mode(cfg)
        assert cfg.model_config.multimodal_config.mm_encoder_tp_mode == "data"


def test_shape_inventory_includes_merger_and_deepstack():
    vision = config().model_config.hf_config.vision_config
    vision.deepstack_visual_indexes = [8, 16, 24]
    shapes = list(qwen3_vision_tp_linears(vision))
    assert len(shapes) == 27 * 4 + 8
    failures = [s for s in shapes if s.invalid_reason(4, 16)]
    assert len(failures) == 27
    assert all(s.prefix.endswith(".mlp.linear_fc2") for s in failures)
    assert [s.input_size for s in shapes[-8:]] == [4608] * 8


def _row(quant, rank=0, tp=4, prefix=PREFIX, dtype=torch.float16):
    with (
        patch.object(linear, "get_tensor_model_parallel_rank", return_value=rank),
        patch.object(linear, "get_tensor_model_parallel_world_size", return_value=tp),
    ):
        return linear.RowParallelLinear(
            4304,
            32,
            params_dtype=dtype,
            quant_config=quant,
            prefix=prefix,
            return_bias=False,
        )


def tensors(rows=32, width=4304, seed=19):
    rng = torch.Generator().manual_seed(seed)
    return {
        "weight": torch.randint(
            0, 256, (rows, width // 2), generator=rng, dtype=torch.uint8
        ),
        "weight_scale": torch.randint(1, 125, (rows, width // 16), generator=rng)
        .byte()
        .view(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.0012345),
        "bias": torch.randn(rows, generator=rng).half(),
    }


def reference(data):
    # Independent FP64 E2M1 decoding and scale multiplication, then one FP16
    # rounding. Deliberately avoid the production dequantizer/LUT construction.
    packed = data["weight"]
    codes = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.int64)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    exponent, fraction = (codes & 7) >> 1, codes & 1
    magnitude = torch.where(
        exponent == 0,
        fraction.double() / 2,
        (1 + fraction.double() / 2) * 2.0 ** (exponent - 1),
    )
    values = torch.where(codes >= 8, -magnitude, magnitude)
    scales = data["weight_scale"].double().repeat_interleave(16, -1)
    return (values * scales * data["weight_scale_2"].double()).half()


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("reverse", [False, True])
def test_real_row_loader_expands_before_slicing(rank, reverse):
    row = _row(quant_config(), rank)
    assert type(row.quant_method).__name__ == "ModelOptNvFp4VisionFC2Method"
    data = tensors()
    items = list(data.items())
    for name, value in reversed(items) if reverse else items:
        param = getattr(row, name)
        param.weight_loader(param, value)
    row.quant_method.process_weights_after_loading(row)
    expected = reference(data)[:, rank * 1076 : (rank + 1) * 1076]
    assert row.weight.shape == (32, 1076)
    assert row.weight.is_contiguous()
    assert torch.equal(row.weight.view(torch.int16), expected.view(torch.int16))
    assert not hasattr(row, "weight_scale")
    assert not hasattr(row, "weight_scale_2")


def test_row_partials_and_bias_match_reference():
    data = tensors()
    x = torch.randn(7, 4304, generator=torch.Generator().manual_seed(42)).half()
    partials = []
    for rank in range(4):
        row = _row(quant_config(), rank)
        for name, value in data.items():
            param = getattr(row, name)
            param.weight_loader(param, value)
        row.quant_method.process_weights_after_loading(row)
        # Real forward owns rank-zero bias; only the collective is replaced.
        with patch.object(
            linear, "tensor_model_parallel_all_reduce", side_effect=lambda y: y
        ):
            partials.append(row(x[:, rank * 1076 : (rank + 1) * 1076]))
    actual = sum(p.double() for p in partials)
    expected = F.linear(x.double(), reference(data).double(), data["bias"].double())
    # Bound four FP16 partial roundings by u * sum(abs(exact partial)), plus
    # the subnormal half-ULP terms. This remains valid near cancellation.
    dense = reference(data).double()
    exact_parts = [
        F.linear(
            x[:, rank * 1076 : (rank + 1) * 1076].double(),
            dense[:, rank * 1076 : (rank + 1) * 1076],
            data["bias"].double() if rank == 0 else None,
        )
        for rank in range(4)
    ]
    bound = sum(p.abs() for p in exact_parts) * 2**-11 + 4 * 2**-25
    assert bool(((actual - expected).abs() <= bound * 1.01).all())


def test_fc1_column_loader_preserves_weights_scales_and_bias():
    data = tensors(rows=4304, width=1152)
    for rank in range(4):
        with (
            patch.object(linear, "get_tensor_model_parallel_rank", return_value=rank),
            patch.object(
                linear, "get_tensor_model_parallel_world_size", return_value=4
            ),
        ):
            fc1 = linear.ColumnParallelLinear(
                1152,
                4304,
                params_dtype=torch.float16,
                quant_config=quant_config(),
                prefix=PREFIX.replace("fc2", "fc1"),
            )
        assert type(fc1.quant_method) is modelopt.ModelOptNvFp4W4A16LinearMethod
        for name, value in data.items():
            param = getattr(fc1, name)
            param.weight_loader(param, value)
        shard_data = {name: getattr(fc1, name) for name in data}
        assert torch.equal(
            reference(shard_data), reference(data)[rank * 1076 : (rank + 1) * 1076]
        )
        assert torch.equal(fc1.bias, data["bias"][rank * 1076 : (rank + 1) * 1076])


@pytest.mark.parametrize(
    "prefix",
    [
        "model.language_model.layers.0.mlp.linear_fc2",
        "mtp.layers.0.mlp.linear_fc2",
        "model.visual.merger.linear_fc2",
        "model.visual.deepstack_merger_list.0.linear_fc2",
    ],
)
def test_other_prefixes_keep_original_method(prefix):
    quant = quant_config()
    quant.quantized_layers[prefix] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    # Ask dispatch with the real RowParallelLinear type but no allocation:
    # these synthetic unaligned shapes must still be rejected by the old path.
    with pytest.raises(ValueError, match="multiple of 16"):
        _row(quant, prefix=prefix)


def test_explicit_data_and_excluded_vision_keep_original_method():
    row = _row(quant_config(), tp=1)
    assert type(row.quant_method) is modelopt.ModelOptNvFp4W4A16LinearMethod
    row = _row(quant_config(exclude=["model.visual.*"]))
    assert type(row.quant_method) is linear.UnquantizedLinearMethod


def test_expansion_rejects_nonfinite_weights():
    module = importlib.import_module(
        "vllm.model_executor.layers.quantization.modelopt_vision"
    )
    row = _row(quant_config())
    data = tensors()
    data["weight_scale_2"] = torch.tensor(float("inf"))
    for name, value in data.items():
        param = getattr(row, name)
        param.weight_loader(param, value)
    assert isinstance(row.quant_method, module.ModelOptNvFp4VisionFC2Method)
    with pytest.raises(ValueError, match="not finite"):
        row.quant_method.process_weights_after_loading(row)


def test_all_vision_loading_methods_and_unchanged_awq_standard_nvfp4():
    from vllm.model_executor.layers.quantization.awq import AWQConfig

    cfg = config()
    vision = cfg.model_config.hf_config.vision_config
    quant = cfg.quant_config
    awq = AWQConfig(4, 32, True, modules_to_not_convert=["model.visual"])
    standard = modelopt.ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True, exclude_modules=["model.visual.*"]
    )
    for shape in qwen3_vision_tp_linears(vision):
        if ".attn." in shape.prefix:
            quant.quantized_layers[shape.prefix] = {
                "quant_algo": "MXFP8",
                "group_size": 32,
            }
        elif ".merger." in shape.prefix:
            quant.quantized_layers[shape.prefix] = {
                "quant_algo": "W4A16_NVFP4",
                "group_size": 16,
            }
        cls = (
            linear.RowParallelLinear
            if shape.split == "row"
            else linear.ColumnParallelLinear
        )
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        layer.input_size, layer.tp_size = shape.input_size, 4
        layer.params_dtype = torch.float16
        assert (
            type(awq.get_quant_method(layer, shape.prefix))
            is linear.UnquantizedLinearMethod
        )
        assert (
            type(standard.get_quant_method(layer, shape.prefix))
            is linear.UnquantizedLinearMethod
        )
        method = quant.get_quant_method(layer, shape.prefix)
        if ".attn." in shape.prefix:
            assert type(method) is modelopt.ModelOptMxFp8SM70LinearMethod
        elif ".blocks." in shape.prefix and shape.prefix.endswith("fc2"):
            assert type(method).__name__ == "ModelOptNvFp4VisionFC2Method"
        else:
            assert type(method) is modelopt.ModelOptNvFp4W4A16LinearMethod
        # Explicit data uses tp_size=1 and keeps every quantizer unchanged.
        layer.tp_size = 1
        original = quant.get_quant_method(layer, shape.prefix)
        expected = (
            modelopt.ModelOptMxFp8SM70LinearMethod
            if ".attn." in shape.prefix
            else modelopt.ModelOptNvFp4W4A16LinearMethod
        )
        assert type(original) is expected


def test_fc1_native_preparation_already_pads_1076_output_rows(monkeypatch):
    from vllm import _sm70_ops

    layer = torch.nn.Module()
    data = tensors(rows=1076, width=1152)
    layer.weight = torch.nn.Parameter(data["weight"], requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(data["weight_scale"], requires_grad=False)
    layer.weight_global_scale = data["weight_scale_2"]
    captured: dict[str, Any] = {}

    def prepare(weight, scale, group, gated):
        captured.update(weight=weight, scale=scale, group=group, gated=gated)
        return weight, scale, None

    monkeypatch.setattr(_sm70_ops, "nvfp4_sm70_prepare", prepare)
    sm70_turbomind.prepare_nvfp4_linear(layer)
    assert captured["weight"].shape == (1152, 1088)
    assert captured["scale"].shape == (72, 1088)
    assert captured["group"] == 16
    assert not captured["gated"]
    assert not bool(captured["weight"][:, 1076:].any())
    assert not bool(captured["scale"][:, 1076:].any())
    state = getattr(layer, sm70_turbomind.STATE_ATTR)
    assert (state.output_size, state.padded_output_size) == (1076, 1088)


def test_production_weight_mapper_keeps_new_vision_dispatch():
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration

    quant = quant_config()
    mapper = Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper
    quant.apply_vllm_mapper(mapper)
    row = _row(quant, prefix=PREFIX.removeprefix("model."))
    assert type(row.quant_method).__name__ == "ModelOptNvFp4VisionFC2Method"
    assert quant._resolve_quant_algo("visual.blocks.0.mlp.linear_fc1") == "W4A16_NVFP4"
