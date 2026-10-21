# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small safetensors packages through the real default and module loaders."""

import time
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vllm.config import LoadConfig
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8SM70LinearMethod,
    ModelOptNvFp4FusedMoE,
    ModelOptNvFp4W4A16LinearMethod,
)
from vllm.model_executor.model_loader.csf_scales import decode_csf_scale_streams
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.utils import AutoWeightsLoader

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
KINDS = ("weight", "weight_scale", "weight_scale_2", "input_scale")


class Dense(nn.Module):
    def __init__(self, mxfp8=False):
        super().__init__()
        cls = ModelOptMxFp8SM70LinearMethod if mxfp8 else ModelOptNvFp4W4A16LinearMethod
        self.quant_method = object.__new__(cls)
        self.quant_method.quant_config = SimpleNamespace(quant_method="W4A16_NVFP4")
        for name in KINDS[:2] if mxfp8 else KINDS:
            param = nn.Parameter(torch.full((16, 2), -1.0), requires_grad=False)
            param.weight_loader = self.weight_loader
            self.register_parameter(name, param)

    def weight_loader(self, param, value):
        param.data.copy_(value.float())


class Experts(nn.Module):
    # Exercise the real mapping, per-expert dispatch and success protocol.
    load_weights = RoutedExperts.load_weights

    def __init__(self):
        super().__init__()
        self.layer_name = "model.layers.0.mlp.experts"
        self.quant_method = object.__new__(ModelOptNvFp4FusedMoE)
        self.quant_method.quant_config = SimpleNamespace(quant_method="NVFP4")
        self.expert_mapping = [
            (
                f"experts.{'w2' if shard == 'w2' else 'w13'}_",
                f"experts.{expert}.{proj}.",
                expert,
                shard,
            )
            for expert in range(3)
            for proj, shard in (
                ("gate_proj", "w1"),
                ("up_proj", "w3"),
                ("down_proj", "w2"),
            )
        ]
        for proj in ("w13", "w2"):
            for kind in KINDS:
                param = nn.Parameter(torch.full((2, 2), -1.0), requires_grad=False)
                param.weight_loader = self.weight_loader
                self.register_parameter(f"{proj}_{kind}", param)

    def _map_global_expert_id_to_local_expert_id(self, expert):
        return expert if expert < 2 else -1

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if self._map_global_expert_id_to_local_expert_id(expert_id) < 0:
            return False
        param.data[expert_id, int(shard_id == "w3")].copy_(loaded_weight.float().mean())
        return True if return_success else None


class Model(nn.Module):
    def __init__(self, mixed):
        super().__init__()
        self.experts = Experts()
        if mixed:
            self.dense = Dense()
            self.mxfp8 = Dense(mxfp8=True)

    def load_weights(self, weights):
        return AutoWeightsLoader(
            self, ignore_unexpected_suffixes=["_weight_scale", "_input_scale"]
        ).load_weights(weights)


def checkpoint(mixed):
    weights = {}
    for expert in range(3):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"experts.{expert}.{proj}."
            for kind in KINDS:
                if mixed and kind == "weight_scale":
                    # A valid CSF pair decodes to [16, 2] E4M3 scales.
                    weights[prefix + kind + ".nvfp4_csf_fixed"] = torch.full(
                        (1, 32), 0, dtype=torch.uint8
                    )
                    weights[prefix + kind + ".nvfp4_csf_exceptions"] = torch.empty(
                        0, dtype=torch.uint32
                    )
                else:
                    weights[prefix + kind] = torch.tensor(float(expert + 2))
    if mixed:
        for prefix, kinds in (("dense", KINDS[:3]), ("mxfp8", KINDS[:2])):
            for kind in kinds:
                weights[f"{prefix}.{kind}"] = torch.full((16, 2), 2.0)
    return weights


def load(
    model, weights, tmp_path, monkeypatch, mixed=True, quantization=None, algo="NVFP4"
):
    paths = []
    # Components in different files: validate only once the whole stream ends.
    for i, (name, value) in enumerate(weights.items()):
        path = tmp_path / f"part-{i}.safetensors"
        save_file({name: value}, path)
        paths.append(path)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    monkeypatch.setattr(loader, "_init_ep_weight_filter", lambda _: None)
    monkeypatch.setattr(
        loader,
        "get_all_weights",
        lambda *_: decode_csf_scale_streams(
            item for path in paths for item in load_file(path).items()
        ),
    )
    loader.counter_before_loading_weights = time.perf_counter()
    config = SimpleNamespace(
        quantization_config={"quantized_layers": {"experts": {"quant_algo": algo}}}
    )
    loader.load_weights(
        model,
        SimpleNamespace(
            quantization=quantization
            or ("modelopt_mixed" if mixed else "modelopt_fp4"),
            hf_config=config,
        ),
    )


@pytest.mark.parametrize("mixed", [False, True], ids=["nvfp4", "fp4-local"])
@pytest.mark.parametrize("missing", KINDS)
def test_missing_expert_component(tmp_path, monkeypatch, mixed, missing):
    weights = checkpoint(mixed)
    key = f"experts.1.up_proj.{missing}"
    for name in list(weights):
        if name == key or name.startswith(key + "."):
            del weights[name]
    with pytest.raises(ValueError, match="NVFP4.*coverage.*w13_" + missing):
        load(Model(mixed), weights, tmp_path, monkeypatch, mixed)


@pytest.mark.parametrize(
    "prefix,missing",
    [("dense", kind) for kind in KINDS[:3]] + [("mxfp8", kind) for kind in KINDS[:2]],
)
def test_missing_dense_component(tmp_path, monkeypatch, prefix, missing):
    weights = checkpoint(True)
    del weights[f"{prefix}.{missing}"]
    with pytest.raises(ValueError, match="NVFP4.*coverage.*" + prefix):
        load(Model(True), weights, tmp_path, monkeypatch)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("missing", ["expert", "layer"])
def test_missing_entire_expert_or_layer(tmp_path, monkeypatch, mixed, missing):
    prefix = "experts.1." if missing == "expert" else "experts."
    weights = {k: v for k, v in checkpoint(mixed).items() if not k.startswith(prefix)}
    with pytest.raises(ValueError, match="NVFP4.*coverage"):
        load(Model(mixed), weights, tmp_path, monkeypatch, mixed)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "unknown", ["experts.0.unknown_proj.weight_scale", "dense.unknown_input_scale"]
)
def test_unknown_quantized_tensor(tmp_path, monkeypatch, mixed, unknown):
    if not mixed and unknown.startswith("dense"):
        pytest.skip("standard fixture has no quantized dense layer")
    weights = checkpoint(mixed)
    weights[unknown] = torch.ones(1)
    with pytest.raises(
        ValueError, match="NVFP4.*unexpected.*" + unknown.split(".")[-1]
    ):
        load(Model(mixed), weights, tmp_path, monkeypatch, mixed)


@pytest.mark.parametrize("mixed", [False, True])
def test_complete_package_preserves_values_and_callbacks(tmp_path, monkeypatch, mixed):
    model = Model(mixed)
    params = dict(model.named_parameters())
    callbacks = {name: p.weight_loader for name, p in params.items()}
    load(model, checkpoint(mixed), tmp_path, monkeypatch, mixed)
    for name, param in model.named_parameters():
        assert param is params[name]
        assert param.weight_loader == callbacks[name]
    for module in model.modules():
        assert "load_weights" not in module.__dict__
        assert "weight_loader" not in module.__dict__
    assert torch.equal(model.experts.w13_weight, torch.tensor([[2.0, 2.0], [3.0, 3.0]]))
    if mixed:
        # W4A16 placeholder input_scale is optional, MXFP8 needs no global scale.
        assert torch.all(model.dense.input_scale == -1)
        assert torch.all(model.mxfp8.weight == 2)


def test_loader_exception_restores_callbacks(tmp_path, monkeypatch):
    model = Model(True)
    callbacks = {name: p.weight_loader for name, p in model.named_parameters()}
    weights = checkpoint(True)
    weights["dense.weight"] = torch.ones(3)
    with pytest.raises(RuntimeError):
        load(model, weights, tmp_path, monkeypatch)
    assert all(p.weight_loader == callbacks[n] for n, p in model.named_parameters())
    assert all("load_weights" not in m.__dict__ for m in model.modules())


def test_fp8_only_mixed_checkpoint_bypasses_new_checks(tmp_path, monkeypatch):
    # Reuse the fixture's missing components to detect accidental activation.
    from vllm.model_executor.model_loader import default_loader

    def unexpected(_):
        pytest.fail("NVFP4 coverage must not run for an FP8-only draft")

    monkeypatch.setattr(
        default_loader, "nvfp4_checkpoint_coverage", unexpected, raising=False
    )
    load(
        Model(False),
        {},
        tmp_path,
        monkeypatch,
        quantization="modelopt_mixed",
        algo="FP8",
    )


@pytest.mark.parametrize("qkv", [False, True], ids=["merged", "qkv"])
@pytest.mark.parametrize("missing", [None, *KINDS[:3]])
def test_real_fused_parameters(tmp_path, monkeypatch, qkv, missing):
    import vllm.model_executor.parameter as parameters
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )
    from vllm.model_executor.models.utils import WeightsMapper

    monkeypatch.setattr(parameters, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_world_size", lambda: 1)
    cls = QKVParallelLinear if qkv else MergedColumnParallelLinear
    layer = object.__new__(cls)
    nn.Module.__init__(layer)
    layer.prefix = "projection"
    layer.tp_rank, layer.tp_size = 0, 1
    layer.output_sizes = [16] * (3 if qkv else 2)
    if qkv:
        layer.num_heads = layer.num_kv_heads = layer.num_kv_head_replicas = 1
        layer.head_size = layer.v_head_size = 16
    method = object.__new__(ModelOptNvFp4W4A16LinearMethod)
    method.quant_config = SimpleNamespace(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True, group_size=16
    )
    layer.quant_method = method
    method.create_weights(
        layer,
        32,
        layer.output_sizes,
        32,
        sum(layer.output_sizes),
        torch.float16,
        weight_loader=layer.weight_loader_v2,
    )
    shard_ids = ("q", "k", "v") if qkv else (0, 1)
    mapper = WeightsMapper(
        orig_to_new_stacked={
            f"part{i}.": ("projection.", shard) for i, shard in enumerate(shard_ids)
        }
    )

    class FusedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = layer

        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    weights = {}
    for i in range(len(shard_ids)):
        weights[f"part{i}.weight"] = torch.full((16, 16), i + 1, dtype=torch.uint8)
        weights[f"part{i}.weight_scale"] = torch.full((16, 2), i + 1).to(
            torch.float8_e4m3fn
        )
        weights[f"part{i}.weight_scale_2"] = torch.tensor(float(i + 1))
    if missing:
        del weights[f"part1.{missing}"]
        with pytest.raises(ValueError, match="NVFP4.*coverage.*" + missing):
            load(FusedModel(), weights, tmp_path, monkeypatch)
    else:
        load(FusedModel(), weights, tmp_path, monkeypatch)
        assert layer.weight_scale_2.tolist() == list(range(1, len(shard_ids) + 1))
        for i in range(len(shard_ids)):
            assert torch.all(layer.weight[i * 16 : (i + 1) * 16] == i + 1)


@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("missing", [None, *KINDS])
def test_real_moe_parameters_and_copies(tmp_path, monkeypatch, tp_rank, missing):
    import vllm.model_executor.parameter as parameters
    from tests.quantization.test_sm70_awq_indexed_prefill import (
        _FakeRoutedExpertsLoader,
    )

    monkeypatch.setattr(parameters, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_world_size", lambda: 4)

    class RealExperts(nn.Module, _FakeRoutedExpertsLoader):
        load_weights = RoutedExperts.load_weights
        _load_per_tensor_weight_scale = RoutedExperts._load_per_tensor_weight_scale
        _load_combined_w13_weight_scale = RoutedExperts._load_combined_w13_weight_scale
        _load_single_value = RoutedExperts._load_single_value

        def __init__(self):
            nn.Module.__init__(self)
            _FakeRoutedExpertsLoader.__init__(self, tp_rank)
            self.tp_size = 4
            self.layer_name = "model.layers.0.mlp.experts"
            self.expert_mapping = Experts().expert_mapping
            self.quant_method = object.__new__(ModelOptNvFp4FusedMoE)
            self.quant_method.quant_config = SimpleNamespace(
                quant_method="NVFP4",
                is_checkpoint_nvfp4_serialized=True,
                group_size=16,
                get_name=lambda: "modelopt_fp4",
            )
            self.quant_method.use_global_sf = False
            self.quant_method.moe = self.moe_config
            self.quant_method.create_weights(
                self,
                num_experts=2,
                hidden_size=32,
                intermediate_size_per_partition=32,
                params_dtype=torch.float16,
                weight_loader=self.weight_loader,
            )

    model = Model(False)
    model.experts = RealExperts()
    weights = {}
    for expert in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            down = proj == "down_proj"
            for kind in KINDS:
                if kind in ("input_scale", "weight_scale_2"):
                    value = torch.tensor(float(expert + 2))
                else:
                    shape = (
                        (32, (8 if kind == "weight_scale" else 64))
                        if down
                        else (128, (2 if kind == "weight_scale" else 16))
                    )
                    value = torch.full(shape, expert + 2).to(
                        torch.float8_e4m3fn if kind == "weight_scale" else torch.uint8
                    )
                weights[f"experts.{expert}.{proj}.{kind}"] = value
    if missing:
        del weights[f"experts.1.up_proj.{missing}"]
        with pytest.raises(ValueError, match="NVFP4.*coverage.*w13_" + missing):
            load(model, weights, tmp_path, monkeypatch, mixed=False)
    else:
        load(model, weights, tmp_path, monkeypatch, mixed=False)
        for param in model.experts.parameters():
            for expert in range(2):
                assert torch.all(param[expert].float() == expert + 2)
