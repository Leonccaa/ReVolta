# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Outer rope parameters of the Qwen4Exp multimodal config (issue #2)."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from vllm.models.qwen4_exp import config as qwen4_exp_config
from vllm.models.qwen4_exp.config import Qwen4ExpConfig

_DEFAULT_ROPE = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default",
}
_YARN = {
    "rope_type": "yarn",
    "factor": 2,
    "original_max_position_embeddings": 262144,
}

# The Flash-Next checkpoint config.json without quantization_config. It has no
# outer rope_parameters; the text config carries the default (M)RoPE.
_CHECKPOINT_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "image_token_id": 248056,
    "language_model_only": False,
    "model_type": "qwen4_exp",
    "tie_word_embeddings": False,
    "video_token_id": 248057,
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
    "dtype": "float16",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 248044,
        "dtype": "float16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "heads_per_ngram": 8,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 128,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 4,
        "initializer_range": 0.02,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "make_ngram_vocab_size_divisible_by": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen4_exp_text",
        "moe_intermediate_size": 640,
        "mtp": {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "mtp_use_hidden_state_from_layer": None,
            "num_hidden_layers": 1,
            "rope_theta": 10000000,
        },
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "ngram_size": 3,
        "ngram_vocab_size_base": 20000000,
        "num_attention_heads": 24,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "num_hidden_layers": 48,
        "num_key_value_heads": 2,
        "output_gate_type": "sigmoid",
        "output_router_logits": False,
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "ple_conv_kernel_size": 4,
        "ple_embed_dim": 2560,
        "ple_layer_ids": [2],
        "rms_norm_eps": 1e-06,
        "rope_parameters": copy.deepcopy(_DEFAULT_ROPE),
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": 640,
        "split_ngram_parts": 128,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 248320,
        "ple_embedding_dtype": "float8_e4m3fn",
    },
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen4_exp",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}


def _yarn_text_only() -> dict[str, Any]:
    """A 512K view that writes YaRN into text_config only."""
    config = copy.deepcopy(_CHECKPOINT_CONFIG)
    text_rope = {**_DEFAULT_ROPE, **_YARN}
    config["text_config"]["rope_parameters"] = copy.deepcopy(text_rope)
    config["text_config"]["rope_scaling"] = copy.deepcopy(text_rope)
    return config


def _production_view() -> dict[str, Any]:
    """A 512K view that also writes the outer default rope explicitly."""
    config = _yarn_text_only()
    config["rope_parameters"] = copy.deepcopy(_DEFAULT_ROPE)
    return config


def _build(config: dict[str, Any]) -> Qwen4ExpConfig:
    kwargs = copy.deepcopy(config)
    kwargs.pop("model_type", None)
    return Qwen4ExpConfig(**kwargs)


def _canonical(config: Qwen4ExpConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True, default=str)


def _build_with_previous_inheritance(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> Qwen4ExpConfig:
    """Build with the outer config inheriting the text rope parameters as is."""
    with monkeypatch.context() as m:
        # raising=False: the helper does not exist before the issue #2 fix.
        m.setattr(
            qwen4_exp_config,
            "_outer_rope_parameters",
            lambda rope: rope,
            raising=False,
        )
        return _build(config)


def test_yarn_only_in_text_config_keeps_the_outer_default_rope() -> None:
    config = _build(_yarn_text_only())

    assert config.rope_parameters == _DEFAULT_ROPE
    assert config.rope_parameters is not config.text_config.rope_parameters
    text_rope = config.text_config.rope_parameters
    assert text_rope["rope_type"] == "yarn"
    assert text_rope["factor"] == 2
    assert text_rope["original_max_position_embeddings"] == 262144
    assert config.text_config.max_position_embeddings == 262144

    # The saved config now names the outer default rope and loads the same.
    reloaded = Qwen4ExpConfig(
        **{k: v for k, v in config.to_dict().items() if k != "model_type"}
    )
    assert reloaded.rope_parameters == _DEFAULT_ROPE
    assert reloaded.text_config.rope_parameters == text_rope


def test_yarn_only_in_text_config_view_loads_through_model_config(
    tmp_path: Path,
) -> None:
    from vllm.config import ModelConfig

    (tmp_path / "config.json").write_text(json.dumps(_yarn_text_only()))
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, dtype="float16"
    )

    assert model_config.hf_config.rope_parameters["rope_type"] == "default"
    assert model_config.hf_text_config.rope_parameters["rope_type"] == "yarn"
    assert model_config.max_model_len == 2 * 262144


def test_yarn_only_in_text_config_matches_the_production_view() -> None:
    implicit = _build(_yarn_text_only())
    explicit = _build(_production_view())

    assert _canonical(implicit) == _canonical(explicit)


def test_explicit_outer_rope_parameters_are_left_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _canonical(
        _build_with_previous_inheritance(monkeypatch, _production_view())
    )

    def _not_consulted(rope: Any) -> Any:
        raise AssertionError("an explicit outer value must be used as is")

    monkeypatch.setattr(
        qwen4_exp_config, "_outer_rope_parameters", _not_consulted, raising=False
    )
    config = _build(_production_view())

    assert config.rope_parameters == _DEFAULT_ROPE
    assert config.text_config.rope_parameters == {**_DEFAULT_ROPE, **_YARN}
    assert _canonical(config) == previous


def test_unscaled_text_rope_is_inherited_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _build_with_previous_inheritance(monkeypatch, _CHECKPOINT_CONFIG)
    config = _build(_CHECKPOINT_CONFIG)

    assert config.rope_parameters is config.text_config.rope_parameters
    assert config.rope_parameters == _DEFAULT_ROPE
    assert _canonical(config) == _canonical(previous)


@pytest.mark.parametrize(
    "text_rope",
    [
        {**_DEFAULT_ROPE, **_YARN, "beta_fast": 32, "beta_slow": 1},
        {**_DEFAULT_ROPE, "type": "yarn", "factor": 2},
        {**_DEFAULT_ROPE, "rope_type": "linear", "factor": 2.0},
    ],
)
def test_outer_rope_parameters_drop_every_scaling_key(
    text_rope: dict[str, Any],
) -> None:
    original = copy.deepcopy(text_rope)

    assert qwen4_exp_config._outer_rope_parameters(text_rope) == _DEFAULT_ROPE
    assert text_rope == original
