# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash context guard using configuration-only local checkpoints."""

import copy

import pytest
from transformers import Qwen3Config

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skip_global_cleanup

NATIVE = 262144
YARN = {
    "rope_type": "yarn",
    "rope_theta": 1000000.0,
    "factor": 2.0,
    "original_max_position_embeddings": NATIVE,
}
DEFAULT = {"rope_type": "default", "rope_theta": 1000000.0}


def build(
    tmp_path,
    monkeypatch,
    target_rope,
    draft_rope,
    target_len,
    draft_len=NATIVE,
    spec_len=None,
    target_rope_update=None,
    method="dflash",
):
    for name, rope, positions, arch in (
        (
            "target",
            target_rope,
            NATIVE if target_rope == YARN else target_len,
            "Qwen3ForCausalLM",
        ),
        ("draft", draft_rope, draft_len, "DFlash2DraftModel"),
    ):
        Qwen3Config(
            architectures=[arch],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            max_position_embeddings=positions,
            rope_parameters=rope,
            dflash_config={"selector_top_k": 16, "block_size": 8},
        ).save_pretrained(tmp_path / name)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    target = ModelConfig(
        model=str(tmp_path / "target"),
        dtype="float16",
        max_model_len=target_len,
        skip_tokenizer_init=True,
    )
    if target_rope_update:
        target.hf_text_config.rope_parameters.update(target_rope_update)
    return SpeculativeConfig(
        method=method,
        model=str(tmp_path / "draft"),
        num_speculative_tokens=7,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        max_model_len=spec_len,
    )


def test_inherits_yarn_for_extended_dflash(tmp_path, monkeypatch):
    spec = build(tmp_path, monkeypatch, YARN, DEFAULT, 524288)
    draft = spec.draft_model_config.hf_text_config
    assert (
        draft.rope_parameters == spec.target_model_config.hf_text_config.rope_parameters
    )
    assert (
        draft.rope_parameters
        is not spec.target_model_config.hf_text_config.rope_parameters
    )
    assert draft.max_position_embeddings == NATIVE
    assert spec.draft_model_config.max_model_len == 524288
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
        YaRNScalingRotaryEmbedding,
    )

    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            head_size=8, max_position=NATIVE, rope_parameters=draft.rope_parameters
        )
    assert isinstance(rope, YaRNScalingRotaryEmbedding)
    assert rope.cos_sin_cache.shape[0] == 524288


def test_extended_dflash_without_yarn_fails(tmp_path, monkeypatch):
    # The target has a native extended limit but no scaling.
    with pytest.raises(ValueError, match="DFlash.*YaRN"):
        build(tmp_path, monkeypatch, DEFAULT, DEFAULT, 524288, draft_len=NATIVE)


def test_rejects_mismatched_yarn_original_limit(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="DFlash.*to match the drafter"):
        build(tmp_path, monkeypatch, YARN, DEFAULT, 524288, draft_len=131072)


@pytest.mark.parametrize("draft_len", [NATIVE, 524288])
def test_unextended_dflash_unchanged(tmp_path, monkeypatch, draft_len):
    spec = build(tmp_path, monkeypatch, YARN, DEFAULT, draft_len, draft_len=draft_len)
    assert spec.draft_model_config.hf_text_config.rope_parameters == DEFAULT
    assert spec.draft_model_config.hf_text_config.max_position_embeddings == draft_len


def test_target_multimodal_layout_does_not_replace_draft_layout(tmp_path, monkeypatch):
    draft_rope = dict(DEFAULT, rope_theta=10000.0, partial_rotary_factor=0.5)
    spec = build(
        tmp_path,
        monkeypatch,
        YARN,
        draft_rope,
        524288,
        target_rope_update={"mrope_section": [11, 11, 10], "mrope_interleaved": True},
    )
    rope = spec.draft_model_config.hf_text_config.rope_parameters
    assert rope["rope_type"] == "yarn" and rope["factor"] == 2.0
    assert rope["rope_theta"] == 10000.0 and rope["partial_rotary_factor"] == 0.5
    assert "mrope_section" not in rope and "mrope_interleaved" not in rope


@pytest.mark.parametrize("spec_len", [131072, 524288])
def test_explicit_draft_limit(tmp_path, monkeypatch, spec_len):
    spec = build(tmp_path, monkeypatch, YARN, DEFAULT, 524288, spec_len=spec_len)
    assert spec.draft_model_config.hf_text_config.rope_parameters["rope_type"] == "yarn"
    assert spec.draft_model_config.max_model_len == spec_len


def test_detected_dflash_explicit_extension(tmp_path, monkeypatch):
    spec = build(
        tmp_path, monkeypatch, YARN, DEFAULT, 524288, spec_len=524288, method=None
    )
    assert spec.method == "dflash"
    assert spec.draft_model_config.hf_text_config.rope_parameters["rope_type"] == "yarn"


@pytest.mark.parametrize("draft_rope", [DEFAULT, YARN])
def test_explicit_limit_validation_outside_guard(tmp_path, monkeypatch, draft_rope):
    with pytest.raises(ValueError, match="User-specified max_model_len"):
        build(
            tmp_path,
            monkeypatch,
            YARN,
            draft_rope,
            NATIVE,
            spec_len=1048576,
        )


def test_existing_yarn_explicit_limit_validation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="User-specified max_model_len"):
        build(tmp_path, monkeypatch, YARN, YARN, 524288, spec_len=1048576)


def test_explicit_extension_without_yarn_fails(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="DFlash.*YaRN"):
        build(tmp_path, monkeypatch, DEFAULT, DEFAULT, 524288, spec_len=524288)


def test_explicit_draft_yarn_is_preserved(tmp_path, monkeypatch):
    draft_rope = dict(YARN, factor=3.0)
    before = copy.deepcopy(draft_rope)
    spec = build(tmp_path, monkeypatch, YARN, draft_rope, 524288)
    assert spec.draft_model_config.hf_text_config.rope_parameters == before
