# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.fused_moe import MoEActivation, RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.models.qwen4_exp.nvidia import mtp_q8_experts as q8
from vllm.models.qwen4_exp.nvidia.mtp_fp8_experts import (
    MTPCheckpointFp8SM70MoEMethod,
    MTPExpertQuantConfig,
)
from vllm.v1.worker.workspace import WorkspaceManager


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


def make_layer(rank, tp, experts=2, hidden=256, width=160):
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    parallel = SimpleNamespace(tp_size=tp, tp_rank=rank)
    layer.moe_parallel_config = parallel
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=parallel,
        experts_per_token=10,
        is_act_and_mul=True,
        has_bias=False,
        ep_size=1,
        dp_size=1,
        pcp_size=1,
    )
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.quant_config = None
    layer.expert_map_manager = SimpleNamespace(map_global_to_local=lambda e: e)
    layer.layer_name = "mtp.layers.48.mlp.experts"
    layer.expert_mapping = [
        (".w13_weight", ".gate_up_proj", 0, "w1"),
        (".w13_weight", ".gate_up_proj", 1, "w3"),
        (".w2_weight", ".down_proj", 0, "w2"),
    ]
    method = q8.MTPQ80MoEMethod(layer.moe_config)
    layer.quant_method = method
    method.create_weights(layer, experts, hidden, width, torch.float16)
    return layer, method


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_checkpoint_load_shards_before_packing(rank, dtype):
    layer, method = make_layer(rank, tp=4)
    torch.manual_seed(27)
    w13 = torch.randn(2, 1280, 256).to(dtype)
    w2 = torch.randn(2, 256, 640).to(dtype)
    original = w2.clone()
    assert set(layer.load_weights([("gate_up_proj", w13), ("down_proj", w2)])) == {
        "w13_weight",
        "w2_weight",
    }
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.shape == (2, 320, 272)
    assert layer.w2_weight.shape == (2, 256, 170)
    start = rank * 160
    expected_w13 = torch.cat(
        (w13[:, start : start + 160], w13[:, 640 + start : 800 + start]), dim=1
    )
    assert torch.equal(layer.w13_weight, q8.quantize_q8_0(expected_w13))
    assert torch.equal(
        layer.w2_weight, q8.quantize_q8_0(original[:, :, start : start + 160])
    )
    assert torch.equal(w2, original)


def test_named_floating_checkpoint_uses_the_standalone_mtp_loader():
    from vllm.models.qwen4_exp.nvidia import mtp

    prefix = "model.layers.0.mlp.experts"
    layer, method = make_layer(rank=3, tp=4, experts=1)
    layer.expert_mapping = [
        ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
        ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
    ]
    model = mtp.Qwen4ExpMTP.__new__(mtp.Qwen4ExpMTP)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = layer
    model.model.config = SimpleNamespace(num_experts=1)
    model.model.fp8_mtp_checkpoint_prefixes = set()
    model.model.fp8_mtp_tp_size = 4
    model.model._kv_cache_dtype = "auto"
    torch.manual_seed(18)
    gate = torch.randn(640, 256).bfloat16()
    up = torch.randn(640, 256).bfloat16()
    down = torch.randn(256, 640).bfloat16()
    weights = [
        (f"mtp.layers.0.mlp.experts.0.{projection}.weight", weight)
        for projection, weight in [
            ("gate_proj", gate),
            ("up_proj", up),
            ("down_proj", down),
        ]
    ]
    assert model.load_weights(weights) == {
        f"{prefix}.w13_weight",
        f"{prefix}.w2_weight",
    }
    method.process_weights_after_loading(layer)
    assert torch.equal(
        layer.w13_weight[0], q8.quantize_q8_0(torch.cat((gate[480:], up[480:])))
    )
    assert torch.equal(layer.w2_weight[0], q8.quantize_q8_0(down[:, 480:]))


def test_q8_payload_rounds_in_floating_point_before_storing_half_scale():
    weight = torch.cat((torch.zeros(32), torch.linspace(-2, 2, 32))).reshape(2, 32)
    packed = q8.quantize_q8_0(weight.to(torch.bfloat16))
    scales = packed[:, :2].contiguous().view(torch.float16).float()
    codes = packed[:, 2:].contiguous().view(torch.int8)
    assert torch.count_nonzero(packed[0]) == 0
    assert (codes[1, 0].item(), codes[1, -1].item()) == (-127, 127)
    assert (
        (codes.float() * scales - weight.bfloat16().float()).abs() <= scales * 0.52
    ).all()


def test_expert_format_selection_and_compile_identity(monkeypatch):
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(
            is_cuda=lambda: True, is_device_capability=lambda cap: cap == (7, 0)
        ),
    )
    hf_config = SimpleNamespace(architectures=["Qwen4ExpMTP"])
    config = SimpleNamespace(
        method="mtp",
        rejection_sample_method="standard",
        draft_model_config=SimpleNamespace(hf_config=hf_config),
        use_local_argmax_reduction=False,
        sm70_dflash2=SimpleNamespace(resolved=False),
        gdn_spec_state_replay=False,
        use_dflash_family=lambda: False,
        use_dspark=lambda: False,
        use_dflash_ddtree=lambda: False,
    )
    identities = set()
    for fmt in (None, "fp8", "q8_0"):
        config.mtp_expert_quantization = fmt
        SpeculativeConfig._verify_mtp_expert_quantization(config)
        identities.add(SpeculativeConfig.compute_hash(config))
    assert len(identities) == 3

    layer = Mock(spec=RoutedExperts)
    layer.moe_config = SimpleNamespace(has_bias=False)
    prefix = "mtp.layers.48.mlp.experts"
    fallback = Mock()
    assert isinstance(
        MTPExpertQuantConfig(fallback, {prefix}).get_quant_method(layer, prefix),
        MTPCheckpointFp8SM70MoEMethod,
    )
    fallback.get_quant_method.return_value = Mock(spec=UnquantizedFusedMoEMethod)
    for source in (None, fallback):
        q8_config = MTPExpertQuantConfig(source, expert_format="q8_0")
        assert isinstance(q8_config.get_quant_method(layer, prefix), q8.MTPQ80MoEMethod)
    assert (
        q8_config.get_quant_method(layer, "model.layers.0.mlp.experts")
        is fallback.get_quant_method.return_value
    )


@pytest.mark.parametrize(
    "rows,expected", [(0, []), (33, [33]), (64, [64]), (65, [64, 1])]
)
def test_dispatch_boundary_and_workspace(monkeypatch, rows, expected):
    layer, method = make_layer(rank=0, tp=4)
    workspace = WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(q8, "current_workspace_manager", lambda: workspace)
    grouped, gguf = [], []

    def grouped_kernel(out, scratch, x, w13, w2, weights, ids):
        grouped.append(x.shape[0])
        assert scratch.dtype == torch.uint8
        assert scratch.numel() == 64 * (256 + 10 * 160) // 32 * 36
        assert x.is_contiguous() and ids.is_contiguous()
        assert weights.dtype == torch.float32
        out.copy_(x)

    def gguf_kernel(x, w13, w2, weights, ids, q1, q2, act):
        gguf.append(x.shape[0])
        assert w13 is layer.w13_weight and w2 is layer.w2_weight
        assert (q1, q2, act) == (8, 8, "silu")
        return x

    monkeypatch.setattr(q8.ops, "sm70_mtp_q8_0_moe_out", grouped_kernel)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.gguf",
        SimpleNamespace(fused_moe_gguf=gguf_kernel),
    )
    x = torch.randn(256, rows).half().T
    weights = torch.ones(rows, 10).half()
    ids = torch.zeros(rows, 10, dtype=torch.int32)
    assert torch.equal(method.apply(layer, x, weights, ids, None, None), x)
    assert grouped == (expected if rows <= 64 else [])
    assert gguf == (expected if rows > 64 else [])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Requires SM70 native operators",
)
def test_q8_method_cuda_graph_replays_high_expert_ids(monkeypatch):
    from vllm.model_executor.layers.quantization.gguf import fused_moe_gguf

    workspace = WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(q8, "current_workspace_manager", lambda: workspace)
    torch.manual_seed(8)
    with torch.device("cuda"):
        layer, method = make_layer(rank=0, tp=1, experts=512)
        # Reuse two different packed experts across the bank. Route only to
        # experts above 255, which the legacy MMQ would silently omit.
        for name in ("w13_weight", "w2_weight"):
            bank = getattr(layer, name)
            shape = (2, bank.shape[1], bank.shape[-1] // 34 * 32)
            weights = torch.randn(shape, dtype=torch.float16) * 0.02
            bank[::2].copy_(q8.quantize_q8_0(weights[0]))
            bank[1::2].copy_(q8.quantize_q8_0(weights[1]))
        for rows in (32, 40, 64, 65):
            x = torch.randn(rows, 256, dtype=torch.float16)
            ids = (
                torch.arange(501, 511, dtype=torch.int32).expand(rows, -1).contiguous()
            )
            weights = torch.rand(rows, 10).softmax(-1)
            eager = method.apply(layer, x, weights, ids, None, None)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = method.apply(layer, x, weights, ids, None, None)
            graph.replay()
            torch.testing.assert_close(actual, eager, rtol=0, atol=0)

            x.mul_(0.7)
            ids.add_(1)
            graph.replay()
            ref = torch.cat(
                [
                    fused_moe_gguf(
                        x[s],
                        layer.w13_weight,
                        layer.w2_weight,
                        weights[s],
                        ids[s],
                        8,
                        8,
                        "silu",
                    )
                    for s in (slice(0, 64), slice(64, None))
                    if x[s].numel()
                ]
            )
            assert ref.float().norm() > 0
            if rows > 64:
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            else:
                assert (actual.float() - ref.float()).norm() / ref.float().norm() < 2e-3
