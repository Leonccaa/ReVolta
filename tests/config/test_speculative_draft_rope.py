# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json
from types import SimpleNamespace

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.config import speculative as impl


def _config(rope, positions):
    text = SimpleNamespace(rope_parameters=rope, max_position_embeddings=positions)
    return SimpleNamespace(hf_text_config=text)


YARN = {
    "rope_type": "yarn",
    "rope_theta": 1e7,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}


def test_draft_adopts_extended_target_rope():
    target = _config(YARN, 1048576)
    draft = _config({"rope_type": "default", "rope_theta": 1e7}, 262144)
    assert impl.match_target_rope_scaling(target, draft)
    assert draft.hf_text_config.rope_parameters == YARN
    assert draft.hf_text_config.max_position_embeddings == 1048576
    # Idempotent: an already scaled draft is left alone.
    assert not impl.match_target_rope_scaling(target, draft)


def test_draft_keeps_rope_for_native_or_mismatched_targets():
    native = _config({"rope_type": "default", "rope_theta": 1e7}, 262144)
    draft = _config({"rope_type": "default", "rope_theta": 1e7}, 262144)
    assert not impl.match_target_rope_scaling(native, draft)
    other_theta = _config({"rope_type": "default", "rope_theta": 1e6}, 262144)
    assert not impl.match_target_rope_scaling(_config(YARN, 1048576), other_theta)
    assert other_theta.hf_text_config.max_position_embeddings == 262144


@pytest.mark.parametrize("draft_type", ["yarn", "linear"])
def test_explicit_draft_scaling_is_preserved(draft_type):
    draft = _config(dict(YARN, rope_type=draft_type, factor=2.0), 262144)
    before = copy.deepcopy(draft)
    assert not impl.match_target_rope_scaling(_config(YARN, 1048576), draft)
    assert draft == before


@pytest.mark.parametrize("missing", ["target", "draft"])
def test_missing_config_is_unchanged(missing):
    target = _config(YARN, 1048576) if missing != "target" else None
    draft = _config({"rope_type": "default", "rope_theta": 1e7}, 262144)
    if missing == "draft":
        draft = None
    before = copy.deepcopy((target, draft))
    assert not impl.match_target_rope_scaling(target, draft)
    assert (target, draft) == before


@pytest.fixture
def checkpoint(tmp_path):
    # Configuration-only checkpoint: no weights, tokenizer or network required.
    def write(name, rope):
        path = tmp_path / name
        path.mkdir()
        config = {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "dtype": "float16",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560,
                "num_hidden_layers": 48,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "max_position_embeddings": 262144,
                "rope_parameters": rope,
                "mtp_num_hidden_layers": 1,
                "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]},
            },
        }
        (path / "config.json").write_text(json.dumps(config))
        return str(path)

    return write


SERVING_ROPE = dict(YARN, factor=2, partial_rotary_factor=0.25)
NATIVE_ROPE = {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25}


def _speculative(target_path, draft_path, **kwargs):
    target = ModelConfig(
        model=target_path,
        dtype="half",
        max_model_len=524288,
        hf_overrides={"text_config": {"rope_parameters": SERVING_ROPE}},
    )
    return SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        model=draft_path,
        method="mtp",
        num_speculative_tokens=4,
        draft_sample_method="greedy",
        **kwargs,
    )


@pytest.mark.parametrize("same_checkpoint", [False, True])
@pytest.mark.parametrize("draft_limit", [None, 131072])
def test_mtp_native_draft_follows_target(checkpoint, same_checkpoint, draft_limit):
    target_path = checkpoint("target", NATIVE_ROPE)
    draft_path = target_path if same_checkpoint else checkpoint("draft", NATIVE_ROPE)
    spec = _speculative(target_path, draft_path, max_model_len=draft_limit)
    draft = spec.draft_model_config
    assert (
        draft.hf_text_config.rope_parameters
        == spec.target_model_config.hf_text_config.rope_parameters
    )
    assert draft.max_model_len == (draft_limit or 524288)
    assert draft.hf_text_config.max_position_embeddings == 262144


@pytest.mark.parametrize(
    "rope", [dict(NATIVE_ROPE, rope_theta=1e6), dict(SERVING_ROPE, factor=1.5)]
)
def test_mtp_preserves_independent_draft(checkpoint, rope):
    spec = _speculative(checkpoint("target", NATIVE_ROPE), checkpoint("draft", rope))
    assert spec.draft_model_config.hf_text_config.rope_parameters == rope
    expected_len = 262144 * rope.get("factor", 1)
    assert spec.draft_model_config.max_model_len == expected_len


def test_scaled_pair_keeps_every_model_field_and_hash(checkpoint, monkeypatch):
    original = impl.match_target_rope_scaling
    calls = []

    def checked_match(target, draft):
        before = copy.deepcopy((vars(target), vars(draft)))
        hashes = (target.compute_hash(), draft.compute_hash())
        assert original(target, draft) is False
        assert (vars(target), vars(draft)) == before
        assert (target.compute_hash(), draft.compute_hash()) == hashes
        calls.append(True)
        return False

    monkeypatch.setattr(impl, "match_target_rope_scaling", checked_match)
    target_path = checkpoint("target", SERVING_ROPE)
    draft_path = checkpoint("draft", SERVING_ROPE)
    spec = _speculative(target_path, draft_path)
    assert calls == [True]
    assert spec.draft_model_config.max_model_len == 524288
    assert spec.draft_model_config.hf_text_config.rope_parameters == SERVING_ROPE


@pytest.mark.parametrize("same_checkpoint", [False, True])
def test_unvalidated_explicit_draft_extension_is_still_rejected(
    checkpoint, same_checkpoint
):
    target_path = checkpoint("target", NATIVE_ROPE)
    draft_path = target_path if same_checkpoint else checkpoint("draft", NATIVE_ROPE)
    # ModelConfig validates an explicit limit before the new matching helper.
    with pytest.raises(ValueError, match="User-specified max_model_len"):
        _speculative(target_path, draft_path, max_model_len=524288)
