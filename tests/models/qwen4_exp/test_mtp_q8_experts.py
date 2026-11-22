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
    model.model.mtp_expert_format = "q8_0"
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
    "mmq,rows,expected",
    [
        (False, 0, []),
        (False, 33, [33]),
        (False, 64, [64]),
        (False, 65, [64, 1]),
        (True, 64, [64]),
        (True, 65, [65]),
        (True, 2048, [2048]),
        (True, 2049, [2048, 1]),
    ],
)
def test_dispatch_boundary_and_workspace(monkeypatch, mmq, rows, expected):
    if mmq:
        monkeypatch.setenv("VLLM_SM70_MTP_Q8_PREFILL_MMQ", "1")
    else:
        monkeypatch.delenv("VLLM_SM70_MTP_Q8_PREFILL_MMQ", raising=False)
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
@pytest.mark.parametrize("mmq", [False, True])
def test_q8_method_cuda_graph_replays_high_expert_ids(monkeypatch, mmq):
    from vllm.model_executor.layers.quantization.gguf import fused_moe_gguf

    workspace = WorkspaceManager(torch.device("cuda"))
    monkeypatch.setenv("VLLM_SM70_MTP_Q8_PREFILL_MMQ", str(int(mmq)))
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
            if rows > 64 and not mmq:
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            else:
                assert (actual.float() - ref.float()).norm() / ref.float().norm() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_gguf_q8_moe_reads_last_expert_partial_k_tile():
    from vllm.model_executor.layers.quantization.gguf import (
        GGUFConfig,
        GGUFMoEMethod,
        fused_moe_gguf,
    )

    torch.manual_seed(17)
    with torch.device("cuda"):
        layer, _ = make_layer(rank=0, tp=4, experts=512, hidden=160)
        experts = torch.tensor([255, 256, 511], dtype=torch.int32)
        for bank in (layer.w13_weight, layer.w2_weight):
            selected = torch.randn(3, bank.shape[1], 160, dtype=torch.half) * 0.2
            bank[experts.long()] = q8.quantize_q8_0(selected)
            assert bank.untyped_storage().nbytes() == bank.numel()
        regular = SimpleNamespace(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_qweight_type=SimpleNamespace(weight_type=8),
            w2_qweight_type=SimpleNamespace(weight_type=8),
            apply_router_weight_on_input=False,
            activation=MoEActivation.SILU,
        )
        x = torch.randn(65, 160, dtype=torch.half) * 0.2
        ids = experts.repeat(22)[-65:, None].contiguous()
        weights = torch.ones_like(ids, dtype=torch.float32)
        method = GGUFMoEMethod(GGUFConfig(), layer.moe_config)
        output = method.apply(regular, x, weights, ids, None, None)
        reference = torch.cat(
            [
                fused_moe_gguf(t, layer.w13_weight, layer.w2_weight, w, i, 8, 8, "silu")
                for t, w, i in zip(
                    x.split(64), weights.split(64), ids.split(64), strict=True
                )
            ]
        )
        assert ids[-1].item() == 511
        for expert in experts:
            actual = output[ids[:, 0] == expert].float()
            expected = reference[ids[:, 0] == expert].float()
            assert (actual - expected).norm() < expected.norm() * 2e-3
        torch.testing.assert_close(output[-1], reference[-1], rtol=0.02, atol=4e-4)


def test_block_fp8_conversion_preserves_unpadded_tp_offsets(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import mtp

    prefix = "model.layers.0.mlp.experts"
    layer, method = make_layer(rank=3, tp=4, experts=1)
    projections = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    layer.expert_mapping = [
        (
            "experts.w2_" if shard == "w2" else "experts.w13_",
            f"experts.0.{projection}.",
            0,
            shard,
        )
        for projection, shard in projections.items()
    ]
    model = mtp.Qwen4ExpMTP.__new__(mtp.Qwen4ExpMTP)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = layer
    model.model.config = SimpleNamespace(num_experts=1)
    model.model.fp8_mtp_checkpoint_prefixes = {prefix}
    model.model.fp8_mtp_tp_size = 4
    model.model.mtp_expert_format = "q8_0"
    model.model._kv_cache_dtype = "auto"
    log = Mock()
    monkeypatch.setattr(mtp.logger, "info_once", log)
    states = {}
    torch.manual_seed(18)
    for projection, shard in projections.items():
        down = shard == "w2"
        shape = (256, 640) if down else (640, 256)
        weight = torch.randint(-8, 9, shape).to(torch.float8_e4m3fn)
        scale_shape = tuple(dim // 128 for dim in shape)
        scale = (torch.arange(1.0, 11.0).reshape(scale_shape) / 128).bfloat16()
        base = f"mtp.layers.0.mlp.experts.0.{projection}"
        states[f"{base}.weight_scale_inv"] = scale
        states[f"{base}.weight"] = weight
    assert model.load_weights(states.items()) == {
        f"{prefix}.w13_weight",
        f"{prefix}.w2_weight",
    }
    assert "secondary quantization" in log.call_args.args[0]
    for projection, shard in projections.items():
        base = f"mtp.layers.0.mlp.experts.0.{projection}"
        weight, scale = states[base + ".weight"], states[base + ".weight_scale_inv"]
        scales = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        dequantized = weight.float() * scales
        param_name = "w2_weight" if shard == "w2" else "w13_weight"
        param = getattr(layer, param_name)
        if shard == "w2":
            assert torch.equal(param[0], q8.quantize_q8_0(dequantized[:, 480:]))
        else:
            offset = 160 if shard == "w3" else 0
            assert torch.equal(
                param[0, offset : offset + 160], q8.quantize_q8_0(dequantized[480:])
            )
    method.process_weights_after_loading(layer)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("float", "q8_0"),
        ("mixed_fp8", "q8_0"),
        ("mixed_nvfp4", None),
        ("awq_excluded", "q8_0"),
        ("awq_packed", None),
    ],
)
def test_auto_expert_format_uses_draft_checkpoint_source(monkeypatch, source, expected):
    from vllm.model_executor.layers.quantization.awq import AWQConfig
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )
    from vllm.models.qwen4_exp.nvidia import mtp

    quant_config = None
    if source.startswith("mixed"):
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "mtp.layers.0.mlp.experts": {
                            "quant_algo": (
                                "FP8_BLOCK_SCALES" if source == "mixed_fp8" else "NVFP4"
                            ),
                            "group_size": 128 if source == "mixed_fp8" else 16,
                        }
                    },
                }
            }
        )
    elif source.startswith("awq"):
        quant_config = AWQConfig(
            4, 32, True, ["mtp"] if source == "awq_excluded" else []
        )
    cfg = _auto_expert_format_config()
    monkeypatch.setattr(mtp, "get_draft_quant_config", lambda _: quant_config)
    mtp.resolve_mtp_expert_default(cfg)
    assert cfg.speculative_config.mtp_expert_quantization == expected


@pytest.mark.parametrize("exclude_mtp", [True, False])
def test_auto_q8_respects_nvfp4_checkpoint_exclusions(monkeypatch, exclude_mtp):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    from vllm.models.qwen4_exp.nvidia import mtp

    # Match the HF ModelOpt metadata used by the NVFP4 checkpoint. Only
    # the excluded MTP experts are floating-point source tensors.
    quant_config = ModelOptNvFp4Config.from_config(
        {
            "producer": {"name": "modelopt", "version": "0.46.0"},
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "ignore": ["mtp.*", "model.mtp.*"] if exclude_mtp else [],
        }
    )
    cfg = _auto_expert_format_config()
    monkeypatch.setattr(mtp, "get_draft_quant_config", lambda _: quant_config)

    mtp.resolve_mtp_expert_default(cfg)
    assert cfg.speculative_config.mtp_expert_quantization == (
        "q8_0" if exclude_mtp else None
    )
    if exclude_mtp:
        layer = Mock(spec=RoutedExperts)
        layer.moe_config = SimpleNamespace(has_bias=False)
        quant_method = MTPExpertQuantConfig(
            quant_config, expert_format="q8_0"
        ).get_quant_method(layer, "mtp.layers.48.mlp.experts")
        assert isinstance(quant_method, q8.MTPQ80MoEMethod)


def _auto_expert_format_config():
    draft = SimpleNamespace(
        dtype=torch.float16,
        hf_text_config=SimpleNamespace(
            hidden_size=2560,
            moe_intermediate_size=640,
            num_experts_per_tok=10,
            num_hidden_layers=48,
        ),
    )
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=draft,
            mtp_expert_quantization=None,
            rejection_sample_method="standard",
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=4,
            enable_expert_parallel=False,
        ),
    )
