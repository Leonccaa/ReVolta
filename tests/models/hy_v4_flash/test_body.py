# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashForCausalLM
from vllm.models.hy_v4_flash.nvidia.moe import HYV4FlashMoE
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def reference():
    return pytest.importorskip("transformers.models.hy_v4_flash.modeling_hy_v4_flash")


@pytest.fixture
def torch_experts(monkeypatch):
    # Replace only the GPU expert GEMMs/activation/scatter. Routing, both
    # latent projections, the shared experts and the norm use the candidate.
    def apply(self, layer, x, topk_weights, topk_ids, **kwargs):
        w13 = layer.w13_weight[topk_ids.long()]
        w2 = layer.w2_weight[topk_ids.long()]
        gate, up = torch.einsum("tkih,th->tki", w13, x).chunk(2, dim=-1)
        limit = layer.swiglu_limit
        activations = F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
        routed = torch.einsum("tkhi,tki->tkh", w2, activations)
        return (routed * topk_weights.unsqueeze(-1)).sum(1).to(x.dtype)

    monkeypatch.setattr(UnquantizedFusedMoEMethod, "is_monolithic", False)
    monkeypatch.setattr(UnquantizedFusedMoEMethod, "apply", apply)


def _config(**overrides):
    return dict(
        dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            moe_intermediate_size=32,
            routed_expert_hidden_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            swa_num_attention_heads=4,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            swa_qk_nope_head_dim=8,
            swa_qk_rope_head_dim=8,
            swa_v_head_dim=16,
            sliding_window=3,
            rope_parameters={
                kind: {
                    "rope_type": "default",
                    "rope_theta": theta,
                    "mrope_section": [2, 1, 1],
                    "mrope_interleaved": True,
                }
                for kind, theta in (
                    ("indexed_attention", 1e7),
                    ("sliding_attention", 1e4),
                )
            },
            compressed_indexer_compress_ratio=2,
            compressed_indexer_num_attention_heads=2,
            compressed_indexer_head_dim=8,
            compressed_indexer_group_topk=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        ),
        **overrides,
    )


def _install_module(monkeypatch, name, **symbols):
    path = f"vllm.models.hy_v4_flash.{name}"
    module = ModuleType(path)
    module.__dict__.update(symbols)
    monkeypatch.setitem(sys.modules, path, module)


def _checkpoint_name(name):
    for old, new in (
        ("attn_hc.fn", "hc_attn_layer.hc_pre.hc_fn"),
        ("attn_hc.base", "hc_attn_layer.hc_pre.hc_base"),
        ("attn_hc.scale", "hc_attn_layer.hc_pre.hc_scale"),
        ("ffn_hc.fn", "hc_mlp_layer.hc_pre.hc_fn"),
        ("ffn_hc.base", "hc_mlp_layer.hc_pre.hc_base"),
        ("ffn_hc.scale", "hc_mlp_layer.hc_pre.hc_scale"),
        ("hc_head.hc_", "hc_head.hc_head_"),
    ):
        name = name.replace(old, new)
    return name


def _record_error(record_property, name, actual, expected):
    diff = (actual.float() - expected.float()).flatten()
    norm = expected.float().norm().clamp_min(1e-12)
    record_property(f"{name}_relative_l2", (diff.norm() / norm).item())
    record_property(f"{name}_max_abs", diff.abs().max().item())


@torch.inference_mode()
def test_three_layer_body(
    reference, cpu_vllm, torch_experts, monkeypatch, record_property
):
    torch.manual_seed(117)
    cfg = _config()
    ref_config = reference.HYV4FlashTextConfig(**cfg)
    ref_config._attn_implementation = "eager"
    ref = reference.HYV4FlashTextModel(ref_config).eval()
    for parameter in ref.parameters():
        parameter.uniform_(-0.2, 0.2)
    for layer in ref.layers[1:]:
        layer.mlp.gate.e_score_correction_bias.uniform_(-0.2, 0.2)

    class ReferenceAttention(nn.Module):
        def __init__(self, vllm_config, config, layer_idx, topk_indices_buffer, prefix):
            super().__init__()
            self.idx = layer_idx
            self.attn = SimpleNamespace(prefix=f"{prefix}.attn")

        def forward(self, positions, hidden_states):
            layer = ref.layers[self.idx]
            x = hidden_states.unsqueeze(0)
            pos = positions.view(1, 1, -1).expand(3, 1, -1)
            delta = positions[:, None] - positions[None, :]
            masked = delta < 0
            if layer.attention_type == "sliding_attention":
                masked |= delta >= ref_config.sliding_window
            mask = torch.zeros_like(delta, dtype=x.dtype).masked_fill(
                masked, -torch.inf
            )
            out, _ = layer.self_attn(
                x,
                ref.rotary_emb(x, pos, layer_type=layer.attention_type),
                mask[None, None],
                position_ids=pos,
            )
            return out.squeeze(0)

    class IdentityEngram(nn.Module):
        def __init__(self, vllm_config, layout, layer_id, prefix):
            super().__init__()
            assert layer_id == 2

        def forward(self, hidden_states, hashes):
            assert hidden_states.shape == (len(ids), 4, cfg["hidden_size"])
            torch.testing.assert_close(hashes, torch.full_like(hashes, 17))
            return hidden_states

    class StubHash:
        def __init__(self, config, dummy):
            self.layout = None
            self.offsets = torch.empty(1, 16)

        def prepare_hashes_with_output(self, input_ids, lookback, layer_name, out):
            assert input_ids is ids and lookback is history
            assert layer_name == "model.layers.0.self_attn.attn"
            out.fill_(17)

    _install_module(
        monkeypatch, "sm70.attention", HYV4FlashAttention=ReferenceAttention
    )
    _install_module(
        monkeypatch,
        "nvidia.engram",
        FlashEngram=IdentityEngram,
        FlashNgramHashState=StubHash,
    )
    config = HYV4FlashConfig(**cfg, engram_layer_ids=[2])
    monkeypatch.setattr(HYV4FlashForCausalLM, "_load_engram_tables", lambda self: None)
    cpu_vllm.model_config = SimpleNamespace(hf_text_config=config, dtype=torch.float32)
    candidate = HYV4FlashForCausalLM(vllm_config=cpu_vllm)
    weights = [
        ("model." + _checkpoint_name(name), param)
        for name, param in ref.named_parameters()
        if ".self_attn." not in name
    ]
    weights += [
        (
            f"model.layers.{i}.mlp.gate.e_score_correction_bias",
            layer.mlp.gate.e_score_correction_bias,
        )
        for i, layer in enumerate(ref.layers)
        if i > 0
    ]
    head = torch.randn(config.vocab_size, config.hidden_size)
    weights += [
        ("lm_head.weight", head),
        ("vit.unused", head),
        ("dfly_layers.unused", head),
    ]
    assert candidate.load_weights(weights) == set(dict(candidate.named_parameters()))

    ids = torch.tensor([2, 5, 18, 3, 7, 1])
    history = torch.tensor([[0, 0]], dtype=torch.int32)
    with set_forward_context(None, cpu_vllm):
        actual = candidate(ids, torch.arange(ids.numel()), lookback_token_ids=history)
    expected = ref(input_ids=ids[None], use_cache=False).last_hidden_state.squeeze(0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    _record_error(record_property, "hidden", actual, expected)
    logits = candidate.compute_logits(actual)
    assert logits is not None and logits.dtype == torch.float32
    expected_logits = F.linear(expected, head)
    torch.testing.assert_close(logits, expected_logits, atol=2e-6, rtol=2e-5)
    _record_error(record_property, "logits", logits, expected_logits)
    torch.testing.assert_close(
        candidate.compute_logits(actual.half()),
        F.linear(actual.half().float(), head),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_latent_moe_and_router(
    reference, cpu_vllm, torch_experts, record_property, dtype
):
    torch.manual_seed(9)
    cfg = _config(n_routed_experts=512, num_experts_per_tok=16)
    with set_default_torch_dtype(dtype):
        candidate = HYV4FlashMoE(HYV4FlashConfig(**cfg), None, "mlp")
        ref = reference.HYV4FlashMoE(reference.HYV4FlashTextConfig(**cfg))
    ref.gate.float()
    for param in candidate.parameters():
        param.uniform_(-0.7, 0.7)
    ref.gate.weight.copy_(candidate.gate.weight)
    ref.gate.e_score_correction_bias.copy_(candidate.expert_bias)
    ref.experts.gate_up_proj.copy_(candidate.experts.w13_weight)
    ref.experts.down_proj.copy_(candidate.experts.w2_weight)
    for projection in (
        "routed_expert_down_proj",
        "routed_expert_up_proj",
        "combine_output_norm",
    ):
        getattr(ref, projection).weight.copy_(getattr(candidate, projection).weight)
    gate, up = candidate.shared_experts.gate_up_proj.weight.chunk(2)
    ref.shared_experts.gate_proj.weight.copy_(gate)
    ref.shared_experts.up_proj.weight.copy_(up)
    ref.shared_experts.down_proj.weight.copy_(candidate.shared_experts.down_proj.weight)

    x = torch.randn(9, cfg["hidden_size"], dtype=dtype) * 2
    with set_forward_context(None, cpu_vllm):
        logits, _ = candidate.gate(x)
        w, ids = candidate.experts.router.select_experts(x, logits)
        got = candidate(x)
    ref_logits, ref_w, ref_ids = ref.gate(x)
    assert logits.dtype == w.dtype == torch.float32
    torch.testing.assert_close(logits, ref_logits, atol=0, rtol=0)
    torch.testing.assert_close(ids.long(), ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(w, ref_w, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        w.sum(-1), torch.full((len(x),), 3.998), atol=1e-6, rtol=0
    )
    expected = ref(x)
    if dtype == torch.float32:
        torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)
    else:
        # The reference casts each weighted contribution before summing.
        assert (got.float() - expected.float()).norm() / expected.float().norm() < 3e-3
        torch.testing.assert_close(got, expected, atol=3e-3, rtol=3e-3)
    _record_error(record_property, "routed_weights", w, ref_w)
    _record_error(record_property, "latent_moe", got, expected)
