# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject partial GDN heads before allocating any projection weights."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vllm.model_executor.layers.mamba.gdn import base
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig


def initialize(monkeypatch, key_heads, value_heads, tp):
    monkeypatch.setattr(base, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(base, "get_tensor_model_parallel_rank", lambda: 0)
    config = Qwen4ExpTextConfig(
        linear_num_key_heads=key_heads, linear_num_value_heads=value_heads
    )
    vllm_config = NS(
        model_config=None, cache_config=None, quant_config=None, speculative_config=None
    )
    return gdn.QwenGatedDeltaNetAttention(
        config, vllm_config, prefix="model.layers.0.linear_attn"
    )


@pytest.mark.parametrize(
    "key_heads,value_heads,tp,field,heads",
    [
        (2, 8, 4, "linear_num_key_heads", 2),
        (4, 6, 4, "linear_num_value_heads", 6),
        (3, 8, 2, "linear_num_key_heads", 3),
        (8, 10, 8, "linear_num_value_heads", 10),
    ],
)
def test_bad_head_count_fails_before_projections(
    monkeypatch, key_heads, value_heads, tp, field, heads
):
    projection = Mock(side_effect=AssertionError("projection allocation attempted"))
    monkeypatch.setattr(gdn, "ColumnParallelLinear", projection)
    with pytest.raises(
        ValueError, match=rf"GDN {field}={heads}.*tensor_parallel_size={tp}"
    ):
        initialize(monkeypatch, key_heads, value_heads, tp)
    projection.assert_not_called()


@pytest.mark.parametrize("key_heads,value_heads,tp", [(2, 2, 1), (16, 48, 4)])
def test_supported_heads_reach_projection_initialization(
    monkeypatch, key_heads, value_heads, tp
):
    class ProjectionReached(Exception):
        pass

    projection = Mock(side_effect=ProjectionReached)
    monkeypatch.setattr(gdn, "ColumnParallelLinear", projection)
    with pytest.raises(ProjectionReached):
        initialize(monkeypatch, key_heads, value_heads, tp)
    projection.assert_called_once()
