# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp vision exports omit deepstack unless explicitly configured."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
from torch import nn

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models import qwen3_vl as vision
from vllm.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpVisionConfig
from vllm.models.qwen4_exp.nvidia import model


@pytest.mark.parametrize("indexes", [None, [], [0, 1]])
def test_vision_merger_weights_follow_explicit_indexes(monkeypatch, indexes):
    kwargs = {} if indexes is None else {"deepstack_visual_indexes": indexes}
    config = Qwen4ExpConfig(
        vision_config=dict(
            depth=0,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            patch_size=2,
            temporal_patch_size=1,
            num_position_embeddings=16,
            **kwargs,
        )
    )
    expected = [] if indexes is None else indexes
    assert config.vision_config.deepstack_visual_indexes == expected
    restored = Qwen4ExpConfig.from_dict(config.to_dict())
    assert restored.vision_config.deepstack_visual_indexes == expected

    monkeypatch.setattr(
        vision.parallel_state, "get_tp_group", lambda: NS(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(vision, "get_vit_attn_backend", lambda **_: None)
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(vllm_config):
        tower = vision.Qwen3_VisionTransformer(config.vision_config)
    assert len(tower.deepstack_merger_list) == len(expected)
    keys = tower.state_dict().keys()
    assert any(key.startswith("merger.") for key in keys)
    assert any(key.startswith("deepstack_merger_list.") for key in keys) == bool(
        expected
    )
    assert tower.out_hidden_size == 16 * (1 + len(expected))


def test_qwen4_vision_default_does_not_inherit_qwen3_deepstack():
    assert Qwen4ExpVisionConfig().deepstack_visual_indexes == []


def test_multimodal_wrapper_accepts_vision_config_without_deepstack(monkeypatch):
    hf_config = pytest.importorskip(
        "transformers.models.qwen4_exp.configuration_qwen4_exp"
    )
    config = Qwen4ExpConfig(vision_config=hf_config.Qwen4ExpVisionConfig())
    assert not hasattr(config.vision_config, "deepstack_visual_indexes")
    cls = model.Qwen4ExpForConditionalGeneration
    monkeypatch.setattr(cls, "_mark_tower_model", lambda *a: nullcontext())
    monkeypatch.setattr(cls, "_mark_language_model", lambda *a: nullcontext())
    monkeypatch.setattr(cls, "set_moe_parameters", lambda *a: None)
    monkeypatch.setattr(model, "cached_tokenizer_from_config", lambda _: None)
    monkeypatch.setattr(model, "get_pp_group", lambda: NS(is_first_rank=True))
    monkeypatch.setattr(model, "Qwen3_VisionTransformer", lambda *a, **kw: nn.Module())
    monkeypatch.setattr(
        model,
        "Qwen4ExpForCausalLM",
        lambda **kw: NS(model=NS(layers=[]), make_empty_intermediate_tensors=None),
    )
    vllm_config = NS(
        quant_config=None,
        model_config=NS(
            hf_config=config,
            multimodal_config=NS(
                language_model_only=False,
                mm_encoder_tp_mode="data",
                video_pruning_rate=0.0,
                is_multimodal_pruning_enabled=lambda: False,
            ),
        ),
    )
    wrapper = cls(vllm_config=vllm_config)
    assert not wrapper.use_deepstack
    assert wrapper.deepstack_num_level == 0
    assert wrapper.multiscale_dim == 0
