# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from vllm.config import DeviceConfig
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import Mxfp4SM70MoEMethod
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
    _validate_weight_layout,
    validate_nvfp4_sm70_moe_contract,
)
from vllm.model_executor.model_loader.nvfp4_coverage import nvfp4_checkpoint_coverage
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashModel
from vllm.models.hy_v4_flash.nvidia.moe import HYV4FlashMoE
from vllm.platforms import current_platform
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skip_global_cleanup

_HIDDEN = 1536
_LOCAL_INTERMEDIATE = 384
_TP_RANK = 2
_LOADED_EXPERTS = (7, 511)
_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _rtn(weight):
    blocks = weight.unflatten(-1, (-1, 16))
    global_scale = weight.abs().amax() / (6 * 448)
    block_scale = (blocks.abs().amax(-1) / (6 * global_scale)).to(torch.float8_e4m3fn)
    effective = (block_scale.float() * global_scale).unsqueeze(-1)
    scaled = blocks.abs() / effective
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    magnitude = torch.bucketize(scaled, boundaries)
    # E2M1 ties round to even (codes 2, 4 and 6 for these three midpoints).
    magnitude += (scaled == 0.75) | (scaled == 1.75) | (scaled == 3.5)
    signed = torch.where(blocks < 0, -1, 1)
    reference = (torch.tensor(_MAGNITUDES)[magnitude] * signed * effective).flatten(1)
    codes = (magnitude + 8 * (blocks < 0)).flatten(1).to(torch.uint8)
    return (codes[:, ::2] | (codes[:, 1::2] << 4)), block_scale, global_scale, reference


@pytest.fixture
def sample(tmp_path):
    generator = torch.Generator().manual_seed(70)
    tensors = {}
    reference = {}
    tp_shard = slice(
        _TP_RANK * _LOCAL_INTERMEDIATE, (_TP_RANK + 1) * _LOCAL_INTERMEDIATE
    )
    for expert in _LOADED_EXPERTS:
        for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            source = torch.randn(_HIDDEN, _HIDDEN, generator=generator)
            source.mul_((0.16, 0.12, 0.007)[index])
            packed, scale, global_scale, dq = _rtn(source)
            prefix = f"layers.1.mlp.experts.{expert}.{projection}"
            tensors[f"{prefix}.weight"] = packed
            tensors[f"{prefix}.weight_scale"] = scale
            tensors[f"{prefix}.weight_scale_2"] = global_scale.reshape(1)
            tensors[f"{prefix}.input_scale"] = torch.tensor([0.01 * (index + 1)])
            reference[expert, projection] = (
                dq[:, tp_shard] if projection == "down_proj" else dq[tp_shard]
            ).contiguous()
    path = tmp_path / "experts.safetensors"
    save_file(tensors, path)
    return path, reference


def _load(sample, monkeypatch, vllm_config, device):
    model = HYV4FlashModel.__new__(HYV4FlashModel)
    nn.Module.__init__(model)
    model.config = HYV4FlashConfig.from_dict(
        {
            "text_config": {"num_hidden_layers": 2},
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "producer": {"name": "modelopt"},
                "ignore": ["*.mlp.shared_experts.*", "*.mlp.routed_expert_*"],
            },
        }
    )
    metadata = ModelArchConfigConvertorBase(model.config, model.config)
    quant = metadata.get_quantization_config()
    assert (
        ModelOptNvFp4Config.override_quantization_method(quant, None) == "modelopt_fp4"
    )
    quant_config = ModelOptNvFp4Config.from_config(quant)
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=4, rank_in_group=_TP_RANK)
    )
    if device == "cpu":
        monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
        monkeypatch.setattr(sm70_tm, "should_use_nvfp4_moe_turbomind", lambda: True)
    else:
        vllm_config.device_config = DeviceConfig(device=device)
    model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    with torch.device(device), set_default_torch_dtype(torch.float16):
        mlp = HYV4FlashMoE(model.config, quant_config, "model.layers.1.mlp")
    model.layers[1].mlp = mlp
    model.start_layer, model.end_layer = 0, 2
    model.layers[0].mlp = nn.Identity()
    model._init_expert_mapping()
    assert isinstance(
        mlp.shared_experts.gate_up_proj.quant_method, UnquantizedLinearMethod
    )
    assert isinstance(mlp.routed_expert_down_proj.quant_method, UnquantizedLinearMethod)
    layer = mlp.experts
    assert isinstance(layer.quant_method, ModelOptNvFp4SM70MoEMethod)
    assert layer.quant_method.use_a16
    assert layer.swiglu_limit == 10.0
    assert (layer.top_k, layer.local_num_experts) == (16, 512)
    assert (layer.hidden_size, layer.intermediate_size_per_partition) == (1536, 384)
    if device == "cuda":
        # CUDA repacks all 512 experts. Only the two sampled experts are nonzero.
        for name, param in layer.named_parameters(recurse=False):
            param.fill_(0 if name.endswith("_weight") else 1)
    path, _ = sample
    # The coverage hook must observe all 24 copies from this partial sample.
    with (
        pytest.raises(ValueError, match="missing 6120 tensor components"),
        nvfp4_checkpoint_coverage(model),
        safe_open(path, framework="pt") as f,
    ):
        keys = f.keys()
        loaded = model.load_weights((key, f.get_tensor(key)) for key in keys)
    assert loaded == {
        f"layers.1.mlp.experts.{proj}_{suffix}"
        for proj in ("w13", "w2")
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }
    return layer


def _dequant(packed, scale, global_scale):
    codes = sm70_tm.unpack_mxfp4_weight(packed).T.long()
    magnitudes = torch.tensor(_MAGNITUDES)[codes % 8]
    scale = (scale.float() * global_scale).repeat_interleave(16, dim=-1)
    return torch.where(codes < 8, magnitudes, -magnitudes) * scale


@torch.inference_mode()
def test_hy4_nvfp4_loader_tp4_cpu(sample, monkeypatch, tp1):
    layer = _load(sample, monkeypatch, tp1, device="cpu")
    _validate_weight_layout(layer)
    _, reference = sample
    for expert in _LOADED_EXPERTS:
        for index, projection in enumerate(("gate_proj", "up_proj")):
            rows = slice(index * _LOCAL_INTERMEDIATE, (index + 1) * _LOCAL_INTERMEDIATE)
            actual = _dequant(
                layer.w13_weight[expert, rows],
                layer.w13_weight_scale[expert, rows],
                layer.w13_weight_scale_2[expert, index],
            )
            torch.testing.assert_close(
                actual, reference[expert, projection], atol=0, rtol=0
            )
        actual = _dequant(
            layer.w2_weight[expert],
            layer.w2_weight_scale[expert],
            layer.w2_weight_scale_2[expert],
        )
        torch.testing.assert_close(
            actual, reference[expert, "down_proj"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            layer.w13_input_scale[expert], torch.tensor([0.01, 0.02])
        )
        torch.testing.assert_close(layer.w2_input_scale[expert], torch.tensor(0.03))


def test_hy4_nvfp4_contract_for_single_gpu_standin():
    moe = SimpleNamespace(
        num_experts=512,
        experts_per_token=16,
        hidden_dim=1536,
        intermediate_size_per_partition=1536,
        tp_size=1,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    validate_nvfp4_sm70_moe_contract(moe)

    moe.tp_size = 2
    moe.intermediate_size_per_partition = 768
    with pytest.raises(NotImplementedError, match="shape is not validated"):
        validate_nvfp4_sm70_moe_contract(moe)


@pytest.mark.parametrize("method", [ModelOptNvFp4SM70MoEMethod, Mxfp4SM70MoEMethod])
def test_hy4_fp4_clamp_cpu(monkeypatch, method):
    def clamp(out, gate_up, limit):
        assert limit == 10.0
        gate, up = gate_up.chunk(2, dim=-1)
        out.copy_(F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit))

    monkeypatch.setattr(torch.ops._C, "silu_and_mul_with_clamp", clamp, raising=False)
    gate_up = torch.tensor([[20.0, 4.0, -20.0, 15.0, -30.0, -10.0]])
    output = torch.empty(1, 3)
    method._apply_swiglu(SimpleNamespace(swiglu_limit=10.0), output, gate_up)
    expected = F.silu(torch.tensor([[10.0, 4.0, -20.0]])) * torch.tensor(
        [[10.0, -10.0, -10.0]]
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="requires SM70 TurboMind CUDA",
)
@torch.inference_mode()
def test_hy4_nvfp4_sm70_against_torch(sample, monkeypatch, tp1):
    layer = _load(sample, monkeypatch, tp1, device="cuda")
    method = layer.quant_method
    method.process_weights_after_loading(layer)
    assert not layer.sm70_nvfp4_qwen38_fused_swiglu_prefill
    x = (
        torch.randn(7, _HIDDEN, generator=torch.Generator().manual_seed(97))
        .mul_(2)
        .half()
    )
    ids = torch.tensor([*_LOADED_EXPERTS, *range(8, 22)], dtype=torch.int32)
    weights = torch.arange(1, 17, dtype=torch.float32)
    weights *= 3.998 / weights.sum()
    _, reference = sample
    expected = torch.zeros_like(x, dtype=torch.float32)
    for i, expert in enumerate(_LOADED_EXPERTS):
        gate, up, down = (
            reference[expert, proj].half().float()
            for proj in ("gate_proj", "up_proj", "down_proj")
        )
        gate = F.linear(x.float(), gate).half().float()
        up = F.linear(x.float(), up).half().float()
        assert (gate > 10).any() and (up > 10).any() and (up < -10).any()
        activated = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).half().float()
        expected += weights[i] * F.linear(activated, down).half().float()
    for tokens in (1, 4, 7):
        actual = (
            method.apply(
                layer,
                x[:tokens].cuda(),
                weights.repeat(tokens, 1).cuda(),
                ids.repeat(tokens, 1).cuda(),
                None,
                None,
            )
            .cpu()
            .float()
        )
        target = expected[:tokens]
        assert (actual - target).norm() / target.norm() < 5e-3
        torch.testing.assert_close(actual, target, atol=0.02, rtol=1e-2)
