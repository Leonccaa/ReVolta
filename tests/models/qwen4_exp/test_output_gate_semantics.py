# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do not reinterpret HF's hidden_act fallback as a sigmoid gate."""

import pytest

from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig


@pytest.mark.parametrize("activation", [None, "sigmoid", "silu"])
def test_explicit_gate_metadata_is_preserved(activation):
    config = Qwen4ExpTextConfig(output_gate_type=activation)
    assert config.output_gate_type == activation
    assert Qwen4ExpTextConfig.from_dict(config.to_dict()).output_gate_type == activation


def test_legacy_omitted_gate_default_is_unchanged():
    assert Qwen4ExpTextConfig().output_gate_type == "sigmoid"


@pytest.mark.parametrize("hidden_act", ["silu", "sigmoid"])
def test_transformers_null_gate_uses_hidden_act(hidden_act):
    hf_config = pytest.importorskip(
        "transformers.models.qwen4_exp.configuration_qwen4_exp"
    )
    hf_model = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config = hf_config.Qwen4ExpTextConfig(
        hidden_size=32,
        hidden_act=hidden_act,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    assert config.output_gate_type is None
    layer = hf_model.Qwen4ExpTextGatedDeltaNet(config, layer_idx=0)
    assert layer.norm.activation == hidden_act
