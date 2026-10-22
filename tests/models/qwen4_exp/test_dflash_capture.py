# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp target features for DFlash-family drafters.

The incoai/Qwen3.8-Flash-Next-DFlash2 drafter was trained on the completed,
zero-based outputs of layers 5/15/25/35/45, averaged over the four
hyper-connection streams. vLLM maps DFlash ``target_layer_ids`` to layer
boundaries with ``+1``; boundary ``b`` is the state entering layer ``b``.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.nvidia import model as qwen4_exp_model
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpModel,
    hc_stream_mean,
)

INCOAI_DFLASH_CONFIG = {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248077,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 15, 25, 35, 45],
}


# ---------------------------------------------------------------- config gate


def _vllm_config(method: str, dflash_config: dict | None = None):
    draft = None
    if dflash_config is not None:
        draft = SimpleNamespace(hf_config=SimpleNamespace(dflash_config=dflash_config))
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hc_count=4, ple_layer_ids=[2], indexer_n_heads=4
            ),
            multimodal_config=None,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            ubatch_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        speculative_config=SimpleNamespace(method=method, draft_model_config=draft),
    )


@pytest.fixture
def _no_base_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Qwen3_5ForConditionalGenerationConfig,
        "verify_and_update_config",
        lambda _config: None,
    )


@pytest.mark.usefixtures("_no_base_verify")
def test_dflash_with_hc_mean_contract_is_admitted() -> None:
    config = dict(INCOAI_DFLASH_CONFIG, target_feature_reduction="hc_mean")
    Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
        _vllm_config("dflash", config)
    )


@pytest.mark.usefixtures("_no_base_verify")
@pytest.mark.parametrize(
    "dflash_config",
    [
        None,
        dict(INCOAI_DFLASH_CONFIG),
        dict(INCOAI_DFLASH_CONFIG, target_feature_reduction="hc_learned_mix"),
    ],
    ids=["no-draft-config", "undeclared", "learned-mixer"],
)
def test_dflash_without_a_supported_contract_is_rejected(dflash_config) -> None:
    with pytest.raises(NotImplementedError, match="target_feature_reduction"):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
            _vllm_config("dflash", dflash_config)
        )


@pytest.mark.usefixtures("_no_base_verify")
def test_dflash_with_pipeline_parallelism_fails_early() -> None:
    config = _vllm_config(
        "dflash", dict(INCOAI_DFLASH_CONFIG, target_feature_reduction="hc_mean")
    )
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)


@pytest.mark.usefixtures("_no_base_verify")
@pytest.mark.parametrize("method", ["dspark", "eagle3", "eagle"])
def test_other_proposers_remain_rejected(method: str) -> None:
    with pytest.raises(NotImplementedError, match="supports only its native MTP"):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
            _vllm_config(method)
        )


@pytest.mark.usefixtures("_no_base_verify")
@pytest.mark.parametrize("method", ["mtp", "ngram", "ngram_gpu"])
def test_existing_proposers_are_unchanged(method: str) -> None:
    Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
        _vllm_config(method)
    )


# --------------------------------------------------------- feature reduction


def test_hc_stream_mean_matches_reference_contract() -> None:
    torch.manual_seed(0)
    hidden = (torch.randn(7, 4 * 32) * 30).to(torch.float16)
    got = hc_stream_mean(hidden, 4)
    # incoai capture patch: hidden.unflatten(-1, (hc, H)).mean(dim=-2).
    want = hidden.float().unflatten(-1, (4, 32)).mean(dim=-2).to(torch.float16)
    assert got.dtype == torch.float16 and got.shape == (7, 32)
    assert torch.equal(got, want)


_HC_CODE = r"""
import torch
from vllm.models.qwen4_exp.nvidia.ops import hc as hc_ops
from vllm.models.qwen4_exp.nvidia.model import hc_stream_mean
torch.manual_seed(0)
T, HC, H = 3, 4, 64
h = torch.randn(T, HC * H, dtype=torch.float16)
b = torch.randn(T, H, dtype=torch.float16)
inj = torch.randn(T, HC, dtype=torch.float16)
w = torch.randn(HC * H, dtype=torch.float16)
materialized = hc_ops._hc_combine(h, b, inj, HC)
fused_state, _ = hc_ops._hc_combine_norm(h, b, inj, w, 1e-6, HC)
assert torch.equal(materialized, fused_state), (materialized - fused_state).abs().max()
assert torch.equal(hc_stream_mean(fused_state, HC), hc_stream_mean(materialized, HC))
print("HC_TAP_EQUAL")
"""


def test_fused_combine_state_equals_materialized_layer_output() -> None:
    """The state read inside layer b is layer b-1's materialized output."""
    env = dict(os.environ, TRITON_INTERPRET="1")
    result = subprocess.run(
        [sys.executable, "-c", _HC_CODE], env=env, capture_output=True, text=True
    )
    print(result.stdout[-2000:], result.stderr[-4000:])
    assert result.returncode == 0 and "HC_TAP_EQUAL" in result.stdout


# --------------------------------------------------------- decoder-layer tap


class _CombiningHyperConnection:
    """Stand-in HC: the pending combine adds the block output to every stream."""

    def __init__(self, hc_count: int) -> None:
        self.hc_count = hc_count

    def materialize(self, hidden, block_output, injection):
        return hidden + block_output.repeat(
            1, self.hc_count
        ) * injection.repeat_interleave(hidden.shape[-1] // self.hc_count, dim=-1)

    def combine(self, hidden, block_output, injection):
        return self.materialize(hidden, block_output, injection)

    def combine_and_mix(self, hidden, block_output, injection):
        state = self.materialize(hidden, block_output, injection)
        return state, hc_stream_mean(state, self.hc_count) * 2, injection

    def mix(self, hidden):
        injection = torch.ones(hidden.shape[0], self.hc_count, dtype=hidden.dtype)
        return hidden, hc_stream_mean(hidden, self.hc_count) * 2, injection


def _layer(tap: bool, *, ple=None) -> Qwen4ExpDecoderLayer:
    layer = object.__new__(Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.ple = ple
    layer.layer_type = "full_attention"
    layer.attn_hyper_connection = _CombiningHyperConnection(4)
    layer.mlp_hyper_connection = _CombiningHyperConnection(4)
    layer.self_attn = lambda *, hidden_states, positions, output: hidden_states + 1
    layer.mlp = lambda hidden: hidden * 3
    layer.hc_count = 4
    layer.aux_hidden_state_tap = tap
    return layer


def _inputs():
    torch.manual_seed(1)
    hidden = torch.randn(5, 4 * 8, dtype=torch.float16)
    block = torch.randn(5, 8, dtype=torch.float16)
    injection = torch.randn(5, 4, dtype=torch.float16)
    return hidden, block, injection


def _run(layer, hidden, block, injection, **kwargs):
    return Qwen4ExpDecoderLayer.forward(
        layer,
        hidden,
        block,
        injection,
        torch.arange(hidden.shape[0]),
        input_ids=torch.zeros(hidden.shape[0], dtype=torch.long),
        query_start_loc=torch.tensor([0, hidden.shape[0]]),
        ngram_context=torch.zeros(1),
        **kwargs,
    )


def test_tap_layer_appends_previous_layer_output_and_keeps_main_path() -> None:
    hidden, block, injection = _inputs()
    plain = _run(_layer(tap=False), hidden, block, injection)
    aux: list[torch.Tensor] = []
    tapped = _run(_layer(tap=True), hidden, block, injection, aux_hidden_states=aux)

    completed = _CombiningHyperConnection(4).materialize(hidden, block, injection)
    assert len(aux) == 1
    assert torch.equal(aux[0], hc_stream_mean(completed, 4))
    for a, b in zip(plain, tapped):
        assert torch.equal(a, b)


def test_non_tap_layer_appends_nothing() -> None:
    hidden, block, injection = _inputs()
    aux: list[torch.Tensor] = []
    _run(_layer(tap=False), hidden, block, injection, aux_hidden_states=aux)
    assert aux == []


def test_ple_layer_taps_before_its_ple_term() -> None:
    hidden, block, injection = _inputs()
    ple_term = torch.full_like(hidden, 0.5)
    aux: list[torch.Tensor] = []
    _run(
        _layer(tap=True, ple=lambda *_args: ple_term),
        hidden,
        block,
        injection,
        aux_hidden_states=aux,
    )
    completed = _CombiningHyperConnection(4).materialize(hidden, block, injection)
    assert len(aux) == 1
    assert torch.equal(aux[0], hc_stream_mean(completed, 4))


# ------------------------------------------------------------ backbone wiring


def _model(monkeypatch: pytest.MonkeyPatch, num_layers: int = 8) -> Qwen4ExpModel:
    calls: list[dict] = []

    class FakeEmbedding(nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

    class FakeLayer(nn.Module):
        def __init__(self, _vllm_config, layer_type: str, prefix: str = "", **_kw):
            super().__init__()
            self.layer_idx = int(prefix.rsplit(".", 1)[-1])
            self.aux_hidden_state_tap = False

        def forward(
            self,
            *,
            hidden_states,
            prev_block_output,
            prev_injection,
            positions,
            input_ids,
            query_start_loc,
            ngram_context,
            **kw,
        ):
            calls.append(dict(layer=self.layer_idx, kwargs=sorted(kw)))
            if self.aux_hidden_state_tap:
                kw["aux_hidden_states"].append(
                    torch.full((hidden_states.shape[0], 1), float(self.layer_idx))
                )
            return hidden_states + 1, None, None

    class FakeMixer(nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

        def combine_and_mix(self, hidden, _block, _injection):
            return hidden, hidden[:, :2], None

    def fake_make_layers(n, get_layer, prefix):
        return 0, n, nn.ModuleList(get_layer(f"{prefix}.{i}") for i in range(n))

    monkeypatch.setattr(qwen4_exp_model, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(qwen4_exp_model, "Qwen4ExpDecoderLayer", FakeLayer)
    monkeypatch.setattr(qwen4_exp_model, "GatedResidual", FakeMixer)
    monkeypatch.setattr(qwen4_exp_model, "make_layers", fake_make_layers)
    monkeypatch.setattr(
        qwen4_exp_model,
        "make_empty_intermediate_tensors_factory",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        qwen4_exp_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=2,
        hc_count=2,
        hc_lowrank=2,
        rms_norm_eps=1e-6,
        num_hidden_layers=num_layers,
        layer_types=["linear_attention"] * num_layers,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, dtype=torch.float32),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(mode=0),
        speculative_config=None,
    )
    model = Qwen4ExpModel(vllm_config=vllm_config)
    model._test_calls = calls
    return model


def _forward(model):
    embeds = torch.zeros(3, 2)
    return Qwen4ExpModel.forward(model, None, torch.arange(3), inputs_embeds=embeds)


def test_backbone_without_aux_layers_is_unchanged(monkeypatch) -> None:
    model = _model(monkeypatch)
    out = _forward(model)
    assert isinstance(out, torch.Tensor)
    # The MTP / no-speculation call passes no extra keyword to the layers.
    assert all(call["kwargs"] == [] for call in model._test_calls)


def test_backbone_returns_aux_in_boundary_order(monkeypatch) -> None:
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    model = _model(monkeypatch, num_layers=48)
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config=INCOAI_DFLASH_CONFIG)
        )
    )
    boundaries = get_eagle3_aux_layers_from_config(spec)
    assert boundaries == (6, 16, 26, 36, 46)
    model._set_aux_hidden_state_layers(boundaries)

    taps = [i for i, layer in enumerate(model.layers) if layer.aux_hidden_state_tap]
    assert taps == [6, 16, 26, 36, 46]
    sample, aux = _forward(model)
    assert sample.shape == (3, 2)
    assert [int(a[0, 0]) for a in aux] == [6, 16, 26, 36, 46]


def test_backbone_final_boundary_reads_the_final_mixer_state(monkeypatch) -> None:
    model = _model(monkeypatch, num_layers=4)
    model._set_aux_hidden_state_layers((2, 4))
    _, aux = _forward(model)
    assert len(aux) == 2
    # Layers add 1 per layer to a zero input, so the final state is 4 everywhere.
    assert torch.equal(aux[1], torch.full((3, 2), 4.0))


@pytest.mark.parametrize("layers", [(16, 6), (6, 6), (0,), (49,)])
def test_backbone_rejects_invalid_boundaries(monkeypatch, layers) -> None:
    model = _model(monkeypatch, num_layers=48)
    with pytest.raises(ValueError):
        model._set_aux_hidden_state_layers(layers)


def test_vl_shell_routes_aux_layers_to_the_text_backbone(monkeypatch) -> None:
    from vllm.model_executor.models.interfaces import SupportsEagle3, supports_eagle3
    from vllm.models.qwen4_exp.nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )

    assert supports_eagle3(Qwen4ExpForConditionalGeneration)
    assert supports_eagle3(Qwen4ExpForCausalLM)
    backbone = _model(monkeypatch, num_layers=48)
    shell = SimpleNamespace(get_language_model=lambda: SimpleNamespace(model=backbone))
    SupportsEagle3.set_aux_hidden_state_layers(shell, (6, 16, 26, 36, 46))
    assert backbone.aux_hidden_state_layers == (6, 16, 26, 36, 46)


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (torch.float32, torch.float16, torch.float16),  # shrink: applied
        (torch.float16, torch.float16, torch.float16),  # same width: kept
        (torch.bfloat16, torch.float16, torch.bfloat16),  # same width: kept
        (torch.float16, torch.float32, torch.float16),  # never widened
        (torch.float16, None, torch.float16),
    ],
)
def test_aux_state_compaction_only_shrinks(source, target, expected) -> None:
    from vllm.models.qwen4_exp.nvidia.model import compact_aux_hidden_state

    value = torch.ones(2, 8, dtype=source)
    assert compact_aux_hidden_state(value, target).dtype == expected
