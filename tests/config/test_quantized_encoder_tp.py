# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No checkpoint weights or device required for quantized vision TP admission."""

import json
from types import SimpleNamespace

import pytest

from vllm.config import ModelConfig, MultiModalConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("mode", [None, "weights", "data"])
def test_cli_preserves_whether_encoder_tp_was_explicit(mode):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    options = [] if mode is None else ["--mm-encoder-tp-mode", mode]
    parsed = parser.parse_args(
        options
        + [
            "--speculative-config",
            json.dumps(
                {
                    "method": "mtp",
                    "num_speculative_tokens": 4,
                    "kv_cache_dtype": "float16",
                }
            ),
        ]
    )
    args = EngineArgs.from_cli_args(parsed)
    assert args.mm_encoder_tp_mode == mode
    assert args.speculative_config == {
        "method": "mtp",
        "num_speculative_tokens": 4,
        "kv_cache_dtype": "float16",
    }


def _quant(group_size=16, quantized_vision=True, excluded=False):
    layers = {
        "model.layers.0.mlp.experts.gate_proj": {
            "quant_algo": "NVFP4",
            "group_size": group_size,
        }
    }
    if quantized_vision:
        layers["model.visual.blocks.0.mlp.linear_fc2"] = {
            "quant_algo": "W4A16_NVFP4",
            "group_size": group_size,
        }
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "group_size": group_size,
            "quantized_layers": layers,
            "ignore": ["model.visual.*"] if excluded else [],
        }
    )


def _config(mode=None, intermediate=4304, tp=4, **quant_kwargs):
    vision = SimpleNamespace(
        hidden_size=1152,
        intermediate_size=intermediate,
        depth=1,
        spatial_merge_size=2,
        out_hidden_size=2560,
        deepstack_visual_indexes=[],
    )
    return SimpleNamespace(
        is_speculative_draft=False,
        quant_config=_quant(**quant_kwargs),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        model_config=SimpleNamespace(
            architecture="Qwen4ExpForConditionalGeneration",
            hf_config=SimpleNamespace(vision_config=vision),
            multimodal_config=MultiModalConfig(mm_encoder_tp_mode=mode or "weights"),
            _mm_encoder_tp_mode_explicit=mode is not None,
            _model_info=SimpleNamespace(supports_multimodal_encoder_tp_data=True),
        ),
    )


def _resolve(config):
    module = pytest.importorskip("vllm.config.mm_encoder_tp")
    module.resolve_mm_encoder_tp_mode(config)
    return config.model_config.multimodal_config.mm_encoder_tp_mode


def test_fp4_local_defaults_to_replicated_encoder():
    config = _config()
    assert _resolve(config) == "data"
    assert config.model_config._mm_encoder_tp_mode_explicit is False


def test_explicit_weights_has_an_actionable_config_error():
    with pytest.raises(
        ValueError, match=r"local input 1076.*alignment 16.*--mm-encoder-tp-mode data"
    ):
        _resolve(_config(mode="weights"))


@pytest.mark.parametrize("mode", [None, "weights"])
def test_unquantized_vision_keeps_weights(mode):
    assert _resolve(_config(mode=mode, quantized_vision=False)) == "weights"


def test_excluded_vision_keeps_weights():
    assert _resolve(_config(excluded=True)) == "weights"


def test_awq_unquantized_vision_keeps_weights():
    from vllm.model_executor.layers.quantization.awq import AWQConfig

    config = _config()
    config.quant_config = AWQConfig(
        weight_bits=4,
        group_size=32,
        zero_point=True,
        modules_to_not_convert=["model.visual"],
    )
    assert _resolve(config) == "weights"


def test_valid_quantized_partition_keeps_weights():
    assert _resolve(_config(intermediate=4352)) == "weights"


def test_alignment_uses_the_resolved_group_size():
    # 4352/TP4 = 1088 is g16-aligned but not g128-aligned.
    assert _resolve(_config(intermediate=4352, group_size=128)) == "data"


@pytest.mark.parametrize("mode,tp", [("data", 4), (None, 1)])
def test_replication_and_legal_tp_need_no_fallback(mode, tp):
    assert _resolve(_config(mode=mode, tp=tp)) == (mode or "weights")


def test_invalid_unsplit_weights_are_not_hidden_by_replication():
    with pytest.raises(ValueError, match="Invalid quantized vision layer"):
        _resolve(_config(intermediate=4300))


def test_encoder_without_data_mode_fails_before_weight_allocation():
    config = _config()
    config.model_config._model_info.supports_multimodal_encoder_tp_data = False
    with pytest.raises(ValueError, match="does not support data TP mode"):
        _resolve(config)


@pytest.mark.parametrize("mode", [None, "weights"])
def test_model_config_preserves_explicitness_without_changing_hash(tmp_path, mode):
    from vllm.models.qwen4_exp.config import Qwen4ExpConfig

    checkpoint = Qwen4ExpConfig(architectures=["Qwen4ExpForConditionalGeneration"])
    checkpoint.save_pretrained(tmp_path)
    kwargs = dict(model=str(tmp_path), skip_tokenizer_init=True, dtype="float16")
    automatic = ModelConfig(**kwargs)
    configured = ModelConfig(**kwargs, mm_encoder_tp_mode=mode)
    if not hasattr(configured, "_mm_encoder_tp_mode_explicit"):
        pytest.skip("base has no explicit encoder TP provenance")
    assert configured._mm_encoder_tp_mode_explicit == (mode is not None)
    assert configured.multimodal_config.mm_encoder_tp_mode == "weights"
    assert configured.compute_hash() == automatic.compute_hash()
