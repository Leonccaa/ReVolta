# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of Qwen4Exp checkpoint configuration spellings."""

import pytest

from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig


@pytest.mark.parametrize(
    "layer_type", ["full_attention", "qwen_sparse_attention", "indexed_attention"]
)
def test_qsa_layer_alias_normalizes_and_round_trips(layer_type):
    config = Qwen4ExpTextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", layer_type],
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    assert config.layer_types == ["linear_attention", "full_attention"]
    restored = Qwen4ExpTextConfig.from_dict(config.to_dict())
    assert restored.layer_types == config.layer_types


def test_unknown_layer_type_is_still_rejected():
    with pytest.raises(ValueError, match="unknown_attention"):
        Qwen4ExpTextConfig(num_hidden_layers=1, layer_types=["unknown_attention"])
