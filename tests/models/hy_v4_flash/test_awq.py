# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    AWQSM70MoEMethod,
    _silu_and_mul_w13,
    _supports_compact_metadata_shape,
)
from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashModel
from vllm.platforms import current_platform

pytestmark = [pytest.mark.skip_global_cleanup, pytest.mark.usefixtures("tp1")]

_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
_SHIFTS = [0, 4, 8, 12, 16, 20, 24, 28]


def _pack(values):
    shifts = torch.tensor(_SHIFTS, device=values.device)
    blocks = values.to(torch.int64).unflatten(-1, (-1, 8))
    return (blocks[..., _PACK_ORDER] << shifts).sum(-1).int().contiguous()


def _unpack(values):
    shifts = torch.tensor(_SHIFTS, device=values.device)
    nibbles = (values.long().unsqueeze(-1) >> shifts) & 15
    return nibbles[..., [0, 4, 1, 5, 2, 6, 3, 7]].flatten(-2)


def _rtn(weight):
    # Quantize per input group and output channel, rounding scale before zero.
    w = weight.T.contiguous().unflatten(0, (-1, 32)).float()
    lo, hi = w.amin(dim=1), w.amax(dim=1)
    scale = ((hi - lo) / 15).clamp_min(1e-5).half()
    zero = (-lo / scale.float()).round().clamp(0, 15)
    codes = (w / scale.float()[:, None] + zero[:, None]).round().clamp(0, 15)
    dq = ((codes - zero[:, None]) * scale.float()[:, None]).flatten(0, 1).T
    return dict(
        qweight=_pack(codes.flatten(0, 1)), qzeros=_pack(zero), scales=scale
    ), dq


def _dequant(qweight, qzeros, scales):
    zeros = _unpack(qzeros).repeat_interleave(32, dim=0)
    return ((_unpack(qweight) - zeros) * scales.float().repeat_interleave(32, 0)).T


def _make_experts(monkeypatch, device):
    config = AWQConfig(4, 32, True, [])
    monkeypatch.setattr(
        config,
        "get_quant_method",
        lambda layer, prefix: AWQSM70MoEMethod(4, 32, True, layer),
    )
    with torch.device(device):
        model = HYV4FlashModel.__new__(HYV4FlashModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=4)
        model.layers = nn.ModuleList([nn.Module()])
        model.layers[0].mlp = nn.Module()
        experts = FusedMoE(
            4,
            2,
            64,
            64,
            quant_config=config,
            params_dtype=torch.float16,
            swiglu_limit=10.0,
            prefix="model.layers.0.mlp.experts",
        )
        model.layers[0].mlp.experts = experts
        model.start_layer, model.end_layer = 0, 1
        model._init_expert_mapping()
        assert isinstance(experts.quant_method, AWQSM70MoEMethod)
        weights = []
        refs = []
        for i in range(4):
            matrices = []
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weight = torch.randn(64, 64) * (0.2 if projection == "down_proj" else 2)
                packed, dq = _rtn(weight)
                matrices.append(dq)
                for suffix, tensor in packed.items():
                    weights.append(
                        (f"layers.0.mlp.experts.{i}.{projection}.{suffix}", tensor)
                    )
            refs.append(matrices)
        assert model.load_weights(weights) == set(dict(model.named_parameters()))
    return experts, refs


def _reference(x, weights):
    gate, up, down = weights
    gate = F.linear(x, gate.to(x.dtype)).clamp(max=10)
    up = F.linear(x, up.to(x.dtype)).clamp(-10, 10)
    return F.linear(F.silu(gate) * up, down.to(x.dtype))


@torch.inference_mode()
def test_awq_rtn_loader_and_clamp_cpu(monkeypatch):
    torch.manual_seed(32)
    layer, refs = _make_experts(monkeypatch, "cpu")
    x = torch.randn(8, 64)
    calls = []

    def clamp(out, gate_up, limit):
        calls.append(limit)
        gate, up = gate_up.chunk(2, dim=-1)
        out.copy_(F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit))

    monkeypatch.setattr(torch.ops._C, "silu_and_mul_with_clamp", clamp, raising=False)
    for i, ref in enumerate(refs):
        w13 = _dequant(layer.w13_qweight[i], layer.w13_qzeros[i], layer.w13_scales[i])
        w2 = _dequant(layer.w2_qweight[i], layer.w2_qzeros[i], layer.w2_scales[i])
        torch.testing.assert_close(w13, torch.cat(ref[:2]), atol=0, rtol=0)
        torch.testing.assert_close(w2, ref[2], atol=0, rtol=0)
        gate_up = F.linear(x, w13)
        gate, up = gate_up.chunk(2, dim=-1)
        assert gate.min() < -10 < 10 < gate.max()
        assert up.min() < -10 < 10 < up.max()
        intermediate = torch.empty_like(x)
        _silu_and_mul_w13(layer, intermediate, gate_up)
        actual = F.linear(intermediate, w2)
        torch.testing.assert_close(actual, _reference(x, ref), atol=0, rtol=0)
        assert not torch.allclose(actual, F.linear(F.silu(gate) * up, w2))
    assert calls == [10.0] * 4


def test_hy4_tp4_compact_geometry():
    layer = SimpleNamespace(
        w13_qweight=torch.empty((512, 1536, 96), device="meta"),
        w2_qweight=torch.empty((512, 384, 192), device="meta"),
    )
    assert _supports_compact_metadata_shape(layer, 32)
    assert not _supports_compact_metadata_shape(layer, 64)
    layer.w2_qweight = layer.w2_qweight[:, :-1]
    assert not _supports_compact_metadata_shape(layer, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 7, 67])
@torch.inference_mode()
def test_sm70_awq_rtn_experts_against_torch(monkeypatch, tokens):
    if not current_platform.is_device_capability(70):
        pytest.skip("requires SM70 TurboMind")
    torch.manual_seed(19)
    layer, refs = _make_experts(monkeypatch, "cuda")
    method = layer.quant_method
    method.process_weights_after_loading(layer)
    assert not layer.sm70_awq_moe_w13_interleaved
    x = torch.randn(tokens, 64, dtype=torch.float16, device="cuda")
    ids = torch.tensor([[0, 3]], dtype=torch.int32, device="cuda").repeat(tokens, 1)
    weights = torch.tensor([[0.9, 3.098]], device="cuda").repeat(tokens, 1)
    actual = method.apply(layer, x, weights, ids, None, None)
    cpu = x.cpu()
    expected = (
        0.9 * _reference(cpu, [w.cpu() for w in refs[0]]).float()
        + 3.098 * _reference(cpu, [w.cpu() for w in refs[3]]).float()
    )
    assert (actual.cpu().float() - expected).norm() / expected.norm() < 5e-3
    torch.testing.assert_close(actual.cpu().float(), expected, atol=1, rtol=1e-2)
