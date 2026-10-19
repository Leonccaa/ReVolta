# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

fp4 = pytest.importorskip("vllm.model_executor.layers.quantization.utils.fp4_local")
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import (
    sm70_turbomind as sm70_tm,
)
from vllm.model_executor.model_loader.csf_scales import (
    decode_nvfp4_csf_scale,
)

spec = importlib.util.spec_from_file_location(
    "oracle", Path(__file__).resolve().parents[2] / "tools/fp4_local_reference.py"
)
assert spec is not None and spec.loader is not None
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def test_all_nvfp4_codes_and_scale_values():
    weight = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    scale = torch.arange(32, dtype=torch.uint8).reshape(16, 2).view(torch.float8_e4m3fn)
    global_scale = torch.tensor(0.037)
    actual = fp4.dequantize_nvfp4_cpu(weight, scale, global_scale)
    expected = ref.nvfp4_reference(weight, scale.view(torch.uint8), global_scale.item())
    assert torch.equal(actual, expected)


def test_mxfp8_e8m0_reference_and_invalid():
    weight = (
        torch.arange(64, dtype=torch.float32).reshape(2, 32).to(torch.float8_e4m3fn)
    )
    scales = torch.tensor([[125], [130]], dtype=torch.uint8)
    expected = weight.float() * torch.tensor([[0.25], [8.0]])
    assert torch.equal(fp4.dequantize_mxfp8_cpu(weight, scales), expected.half())
    scales[0, 0] = 255
    with pytest.raises(ValueError, match="NaN"):
        fp4.dequantize_mxfp8_cpu(weight, scales)


def test_sm70_mxfp8_converts_and_discards_scales():
    method = modelopt.ModelOptMxFp8SM70LinearMethod()
    layer = torch.nn.Module()
    import vllm.model_executor.parameter as parameters

    with (
        patch.object(parameters, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(
            parameters, "get_tensor_model_parallel_world_size", return_value=1
        ),
    ):
        method.create_weights(
            layer, 32, [16], 32, 16, torch.float16, weight_loader=lambda *a: None
        )
    layer.weight.data.fill_(1)
    layer.weight_scale.data.fill_(125)
    with patch.object(method.unquantized, "process_weights_after_loading"):
        method.process_weights_after_loading(layer)
    assert not hasattr(layer, "weight_scale")
    assert torch.equal(layer.weight, torch.full((16, 32), 0.25, dtype=torch.float16))


def test_mixed_dispatch_mxfp8_and_nvfp4():
    cfg = modelopt.ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt",
            "quantized_layers": {
                "q": {"quant_algo": "MXFP8"},
                "v": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            },
        }
    )
    from vllm.model_executor.layers.linear import LinearBase

    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    with patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True):
        assert isinstance(
            cfg.get_quant_method(layer, "q"), modelopt.ModelOptMxFp8SM70LinearMethod
        )
    # Vision/MTP W4A16 retains the established NVFP4 adapter.
    with patch.object(modelopt, "MarlinNvFp4LinearKernel"):
        assert isinstance(
            cfg.get_quant_method(layer, "v"), modelopt.ModelOptNvFp4W4A16LinearMethod
        )


def test_checkpoint_samples_independent_dequant():
    sample_dir = os.getenv("FP4_LOCAL_SAMPLE_DIR")
    if not sample_dir:
        pytest.skip("set FP4_LOCAL_SAMPLE_DIR to the read-only audit sample directory")
    assert sample_dir is not None
    from safetensors.torch import load_file

    for path in sorted(Path(sample_dir).glob("*.safetensors")):
        data = load_file(str(path))
        if "fixed" in data:
            scales = decode_nvfp4_csf_scale(data["fixed"], data["exceptions"])
            ref_scale = ref.csf_reference_bytes(data["fixed"], data["exceptions"])
            assert torch.equal(scales.view(torch.uint8), ref_scale)
        else:
            scales = data["scale"]
            ref_scale = scales.view(torch.uint8)
        actual = fp4.dequantize_nvfp4_cpu(data["weight"], scales, data["global"])
        expected = ref.nvfp4_reference(data["weight"], ref_scale, data["global"].item())
        assert torch.equal(actual, expected), path.name


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("kind", ["merged", "row"])
def test_mxfp8_tp_loads_before_fp16_conversion(rank, kind):
    from types import SimpleNamespace

    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameters

    method = modelopt.ModelOptMxFp8SM70LinearMethod()
    config = SimpleNamespace(get_quant_method=lambda *a, **kw: method)
    weight = (
        (torch.arange(96 * 128).reshape(96, 128) % 15 - 7)
        .float()
        .to(torch.float8_e4m3fn)
    )
    scale = (torch.arange(96 * 4).reshape(96, 4) % 5 + 124).byte()
    with (
        patch.object(linear, "get_tensor_model_parallel_rank", return_value=rank),
        patch.object(linear, "get_tensor_model_parallel_world_size", return_value=4),
        patch.object(parameters, "get_tensor_model_parallel_rank", return_value=rank),
        patch.object(
            parameters, "get_tensor_model_parallel_world_size", return_value=4
        ),
    ):
        if kind == "merged":
            layer = linear.MergedColumnParallelLinear(
                128,
                [32, 64],
                bias=False,
                params_dtype=torch.float16,
                quant_config=config,
            )
            for shard, (lo, hi) in enumerate([(0, 32), (32, 96)]):
                layer.weight.weight_loader(layer.weight, weight[lo:hi], shard)
                layer.weight_scale.weight_loader(
                    layer.weight_scale, scale[lo:hi], shard
                )
            w = torch.cat(
                (
                    weight[rank * 8 : rank * 8 + 8],
                    weight[32 + rank * 16 : 32 + rank * 16 + 16],
                ),
                0,
            )
            s = torch.cat(
                (
                    scale[rank * 8 : rank * 8 + 8],
                    scale[32 + rank * 16 : 32 + rank * 16 + 16],
                ),
                0,
            )
        else:
            layer = linear.RowParallelLinear(
                128, 96, bias=False, params_dtype=torch.float16, quant_config=config
            )
            layer.weight.weight_loader(layer.weight, weight)
            layer.weight_scale.weight_loader(layer.weight_scale, scale)
            w, s = weight[:, rank * 32 : (rank + 1) * 32], scale[:, rank : rank + 1]
        with patch.object(method.unquantized, "process_weights_after_loading"):
            method.process_weights_after_loading(layer)
        assert torch.equal(layer.weight, fp4.dequantize_mxfp8_cpu(w, s))


def test_mtp_nvfp4_recipe_remap():
    from vllm.models.qwen4_exp.nvidia.mtp import _remap_quantized_layers

    recipe = {
        "mtp.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    }
    assert _remap_quantized_layers(recipe, 48) == {
        "mtp.layers.48.mlp.experts": recipe["mtp.layers.0.mlp.experts"]
    }
