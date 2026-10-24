# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission checks for Flash-V100 DFlash draft KV storage."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.config.speculative_kv import (
    dflash_draft_cache_dtype,
    validate_speculative_kv_cache_dtype,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.skip_global_cleanup


def _config(target_dtype="int8", draft_dtype=None, method="dflash"):
    hf_config = SimpleNamespace(
        dflash_config={"selector_top_k": 16, "use_swa": True, "swa_window_size": 256},
        layer_types=None,
    )
    model = SimpleNamespace(dtype=torch.float16, hf_config=hf_config)
    return SimpleNamespace(
        is_speculative_draft=False,
        cache_config=CacheConfig(cache_dtype=target_dtype),
        speculative_config=SimpleNamespace(
            method=method,
            kv_cache_dtype=draft_dtype,
            attention_backend=AttentionBackendEnum.FLASH_ATTN_V100,
            draft_model_config=model,
        ),
        model_config=model,
    )


@pytest.fixture
def sm70(monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(vllm.platforms.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        vllm.platforms.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )


@pytest.mark.parametrize("backend", [None, AttentionBackendEnum.FLASH_ATTN_V100])
@pytest.mark.parametrize(
    "target_dtype,draft_dtype",
    [("int8", None), ("int8", "int8"), ("fp8_e4m3", "int8")],
)
@pytest.mark.parametrize("feature", ["swa", "layer_types", "sink", "cache_window"])
def test_dflash_int8_rejected_before_loading_weights(
    backend, target_dtype, draft_dtype, feature
):
    config = _config(target_dtype, draft_dtype)
    spec = config.speculative_config
    spec.attention_backend = backend
    hf_config = spec.draft_model_config.hf_config
    if feature != "swa":
        hf_config.dflash_config["use_swa"] = False
        if feature == "layer_types":
            hf_config.layer_types = ["full_attention", "sliding_attention"]
        elif feature == "cache_window":
            config.cache_config.sliding_window = 256
        else:
            hf_config.dflash_config["attention_sink_bias"] = True

    with pytest.raises(ValueError) as exc:
        validate_speculative_kv_cache_dtype(config)

    message = str(exc.value)
    for keyword in (
        "DFlash2",
        "INT8",
        "FLASH_ATTN_V100",
        "fp8_e4m3",
        "float16",
        "auto",
        "mtp",
    ):
        assert keyword in message


@pytest.mark.parametrize(
    "target_dtype,draft_dtype,method",
    [
        ("fp8_e4m3", None, "dflash"),
        ("float16", None, "dflash"),
        ("auto", None, "dflash"),
        ("int8", "float16", "dflash"),
        ("int8", "auto", "dflash"),
        ("int8", None, "mtp"),
        ("int8", "int8", "mtp"),
        ("fp8_e4m3", None, "mtp"),
        ("fp8_e4m3", "fp8_e4m3", "mtp"),
    ],
)
def test_compatible_kv_choices_are_unchanged(target_dtype, draft_dtype, method):
    config = _config(target_dtype, draft_dtype, method)
    before = deepcopy(config)
    validate_speculative_kv_cache_dtype(config)
    assert config == before


def test_int8_without_speculation_is_unchanged():
    config = _config()
    config.speculative_config = None
    before = deepcopy(config)
    validate_speculative_kv_cache_dtype(config)
    assert config == before


def test_full_attention_int8_dflash_keeps_the_existing_dense_route():
    config = _config()
    config.speculative_config.draft_model_config.hf_config.dflash_config = {
        "selector_top_k": 16,
    }
    validate_speculative_kv_cache_dtype(config)


def test_top_level_sink_bias_is_also_rejected():
    config = _config()
    hf_config = config.speculative_config.draft_model_config.hf_config
    hf_config.dflash_config = {"selector_top_k": 16}
    hf_config.add_swa_attention_sink_bias = True
    with pytest.raises(ValueError, match="DFlash2.*INT8"):
        validate_speculative_kv_cache_dtype(config)


def test_explicit_sink_setting_overrides_top_level_default():
    config = _config()
    hf_config = config.speculative_config.draft_model_config.hf_config
    hf_config.dflash_config = {"attention_sink_bias": False}
    hf_config.add_swa_attention_sink_bias = True
    validate_speculative_kv_cache_dtype(config)


@pytest.mark.parametrize(
    "target_dtype,expected", [("int8", "int8"), ("fp8_e4m3", "auto")]
)
def test_dflash_resolves_the_same_dtype_as_its_loader(sm70, target_dtype, expected):
    assert dflash_draft_cache_dtype(_config(target_dtype)) == expected


@pytest.mark.parametrize(
    "method,target_dtype,draft_dtype,reject",
    [
        ("dflash", "int8", None, True),
        ("dflash", "fp8_e4m3", None, False),
        ("dflash", "float16", None, False),
        ("dflash", "auto", None, False),
        ("dflash", "int8", "float16", False),
        (None, "int8", None, False),
    ],
)
def test_engine_args_validates_kv_before_model_loading(
    tmp_path, method, target_dtype, draft_dtype, reject
):
    from transformers import Qwen3Config

    from vllm.engine.arg_utils import EngineArgs

    for name, architecture in (
        ("target", "Qwen3ForCausalLM"),
        ("draft", "DFlash2DraftModel"),
    ):
        checkpoint = Qwen3Config(
            architectures=[architecture],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            max_position_embeddings=512,
            dflash_config={
                "selector_top_k": 16,
                "block_size": 8,
                "use_swa": True,
                "swa_window_size": 256,
            },
        )
        checkpoint.save_pretrained(tmp_path / name)

    # These directories contain config.json only: weights cannot be loaded.
    args = EngineArgs(
        model=str(tmp_path / "target"),
        dtype="float16",
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        kv_cache_dtype=target_dtype,
        speculative_config=(
            {
                "method": method,
                "model": str(tmp_path / "draft"),
                "num_speculative_tokens": 7,
                "attention_backend": "FLASH_ATTN_V100",
                "kv_cache_dtype": draft_dtype,
            }
            if method is not None
            else None
        ),
    )
    if reject:
        with pytest.raises(ValueError, match="DFlash2.*INT8"):
            args.create_engine_config()
    else:
        config = args.create_engine_config()
        assert config.cache_config.cache_dtype == target_dtype


@pytest.mark.parametrize("feature", [None, "sliding_window", "sinks"])
def test_dense_int8_reader_boundary_without_gpu(monkeypatch, feature):
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.ops import int8_dense

    monkeypatch.setattr(int8_dense, "require_int8_abi", lambda: None)
    impl = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        alibi_slopes=None,
        logits_soft_cap=0,
        sinks=None,
        sliding_window=(-1, -1),
    )
    output = torch.ones((1, 1, 64))
    if feature is not None:
        setattr(impl, feature, (255, 0) if feature == "sliding_window" else output)
        with pytest.raises(NotImplementedError, match="planar INT8 dense feature"):
            int8_dense.forward_int8_dense(impl, output, None, None, output)
    else:
        result = int8_dense.forward_int8_dense(impl, output, None, None, output)
        assert result is output
        assert not output.any()
