# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.config.kernel import KernelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.quantization.sm70_dflash2_fp16 import (
    _GEOMETRY,
    pack_dflash2_fp16_weight,
)
from vllm.model_executor.models import qwen3_dflash2 as draft


@pytest.mark.parametrize(
    "opt_in,method,tp,async_scheduling,drafts,expected",
    [
        (False, "dflash", 4, False, 7, 0),
        (True, "dflash", 4, False, 7, 6),
        (True, "dflash", 4, True, 7, 0),
        (True, "dflash", 2, False, 7, 0),
        (True, "dflash", 4, False, 3, 0),
        (True, "dflash_ddtree", 4, False, 7, 0),
        (True, "mtp", 4, False, 4, 0),
    ],
)
def test_only_sync_mrv2_dflash2_query_projections_are_marked(
    opt_in, method, tp, async_scheduling, drafts, expected, monkeypatch
):
    def init_decoder(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Identity()
        self.self_attn.o_proj = nn.Identity()
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Identity()
        self.mlp.down_proj = nn.Identity()

    monkeypatch.setattr(draft.DFlashQwen3DecoderLayer, "__init__", init_decoder)
    monkeypatch.setattr(draft, "_use_sm70_bf16_emulation", lambda config: True)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: tp)
    spec = object.__new__(SpeculativeConfig)
    spec.method = method
    spec.num_speculative_tokens = drafts
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float16),
        kernel_config=KernelConfig(sm70_dflash2_fp16_m8=opt_in),
        speculative_config=spec,
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    decoder = draft.DFlash2Qwen3DecoderLayer(
        config,
        config=SimpleNamespace(
            hidden_size=2560,
            rms_norm_eps=1e-6,
            dflash_config={"conv_kernel_size": 2, "conv_group_size": 16},
        ),
        layer_idx=0,
    )
    count = sum(
        bool(getattr(module, "_sm70_dflash2_fp16_m8", False))
        for module in decoder.modules()
    )
    assert count == expected
    assert decoder.attention_conv.kernel_projection.weight.shape == (640, 2560)


def test_packed_draft_projection_is_opt_in_and_changes_compile_key():
    default = KernelConfig()
    assert not default.sm70_dflash2_fp16_m8
    assert (
        default.compute_hash() != KernelConfig(sm70_dflash2_fp16_m8=True).compute_hash()
    )


@pytest.mark.parametrize("n,k", _GEOMETRY)
def test_pack_is_a_bitwise_permutation(n, k):
    tile, warps = _GEOMETRY[n, k]
    assert n % tile == k % (16 * warps) == 0
    # Exercise signed zeros and arbitrary FP16 payloads via an integer view.
    weight = torch.arange(n * k, dtype=torch.int32).short().view(torch.float16)
    weight = weight.view(n, k)
    packed = pack_dflash2_fp16_weight(weight, tile)
    cols = torch.arange(tile)
    slots = (cols & 3) | ((cols & 24) >> 1) | ((cols & 4) << 2)
    restored = packed.index_select(2, slots).permute(0, 2, 1, 3).reshape(n, k)
    assert torch.equal(restored.view(torch.int16), weight.view(torch.int16))


def test_packed_linear_eager_dispatch_and_fallback(monkeypatch):
    monkeypatch.setattr(linear, "dispatch_unquantized_gemm", lambda: dense)
    monkeypatch.setattr(linear.envs, "VLLM_BATCH_INVARIANT", False)
    calls = []

    def dense(layer, x, w, bias):
        return F.linear(x, w, bias)

    def packed(out, x, layout, weight, tile, warps):
        calls.append(x.shape[0])
        out.copy_(F.linear(x, weight))

    monkeypatch.setattr(ops, "sm70_dflash2_fp16_dispatch_out", packed)
    layer = nn.Linear(32, 32, bias=False, dtype=torch.float16)
    method = linear.UnquantizedLinearMethod()
    x = torch.randn(8, 32, dtype=torch.float16)
    torch.testing.assert_close(method.apply(layer, x), layer(x), rtol=0, atol=0)
    assert not calls

    layer._sm70_dflash2_fp16_geometry = (32, 8)
    layer.register_buffer("_sm70_dflash2_fp16_packed", layer.weight)
    torch.testing.assert_close(method.apply(layer, x), layer(x), rtol=0, atol=0)
    assert calls == [8]
    for x_other in (x[:4], x.t().contiguous().t(), x.unsqueeze(0)):
        torch.testing.assert_close(
            method.apply(layer, x_other), layer(x_other), rtol=0, atol=0
        )
    bias = torch.ones(32, dtype=torch.float16)
    torch.testing.assert_close(
        method.apply(layer, x, bias), F.linear(x, layer.weight, bias), rtol=0, atol=0
    )
    assert calls == [8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_packed_draft_linear_dynamic_trace_and_cudagraph(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    assert hasattr(torch.ops._C, "sm70_dflash2_fp16_dispatch_out")
    monkeypatch.setattr(linear.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(linear.envs, "VLLM_SM70_ENABLE_DENSE_F16_FASTPATH", False)
    layer = nn.Linear(2560, 640, bias=False, device="cuda", dtype=torch.float16)
    layer._sm70_dflash2_fp16_m8 = True
    method = linear.UnquantizedLinearMethod()
    method.process_weights_after_loading(layer)
    assert "_sm70_dflash2_fp16_packed" in dict(layer.named_buffers())
    assert "_sm70_dflash2_fp16_packed" not in layer.state_dict()

    compiled = torch.compile(
        lambda x: method.apply(layer, x),
        backend="aot_eager",
        dynamic=True,
        fullgraph=True,
    )
    x16 = torch.randn(16, 2560, device="cuda", dtype=torch.float16)
    x8 = x16[:8].clone()
    torch.testing.assert_close(compiled(x16), layer(x16), rtol=0, atol=0)
    native = x8.new_empty((8, 640))
    ops.sm70_dflash2_fp16_m8_out(native, x8, layer._sm70_dflash2_fp16_packed, 32, 8)
    torch.testing.assert_close(compiled(x8), native, rtol=0, atol=0)
    # The native M8 path uses 16-byte vector loads. This offset view must
    # follow the ordinary dense result even though its strides are contiguous.
    offset = x16.flatten()[1 : 1 + 8 * 2560].view_as(x8)
    torch.testing.assert_close(
        method.apply(layer, offset), layer(offset), rtol=0, atol=0
    )

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(x8)
    for _ in range(3):
        x8.normal_()
        graph.replay()
        torch.testing.assert_close(captured, method.apply(layer, x8), rtol=0, atol=0)
        ref = (x8.double() @ layer.weight.double().T).half()
        assert torch.isfinite(captured).all()
        relative_l2 = (captured.float() - ref).norm() / ref.float().norm()
        assert relative_l2 <= 1e-3
