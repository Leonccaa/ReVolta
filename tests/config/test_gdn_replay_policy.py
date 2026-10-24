# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay/packed precedence through real model and engine configuration (CPU)."""

import json
import os
from dataclasses import asdict

import pytest

from vllm import envs
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.sm70_dflash2 import Sm70DFlash2Config

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture
def engine_config(monkeypatch, tmp_path):
    import vllm.config.vllm as config_module
    from vllm.platforms import current_platform

    monkeypatch.setattr(config_module.logger, "info_once", config_module.logger.info)
    monkeypatch.setattr(config_module.logger, "propagate", True)
    import logging

    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)

    # Only device discovery is simulated; policy qualification and both config
    # constructors run normally, with local configuration-only checkpoints.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name)
    for name in envs.environment_variables:
        monkeypatch.delitem(vars(envs), name, raising=False)
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", "0")
    monkeypatch.setenv("VLLM_SM70_FUSED_SIGMOID_MIXED_QKV", "1")
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "device_count", lambda: 4)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability in (70, (7, 0)),
    )

    def build(hidden_size=5120, *, method="dflash", replay=None, packed=None):
        target = tmp_path / "target"
        target.mkdir(exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(
                {
                    "architectures": [
                        "Qwen4ExpForConditionalGeneration"
                        if hidden_size == 2560
                        else "Qwen3_5ForConditionalGeneration"
                    ],
                    "model_type": "qwen4_exp" if hidden_size == 2560 else "qwen3_5",
                    "text_config": {
                        "model_type": "qwen4_exp_text"
                        if hidden_size == 2560
                        else "qwen3_5_text",
                        "hidden_size": hidden_size,
                        "num_hidden_layers": 4,
                        "num_attention_heads": 24,
                        "num_key_value_heads": 2 if hidden_size == 2560 else 4,
                        "head_dim": 256,
                        "max_position_embeddings": 524288,
                        "vocab_size": 128,
                        "mtp_num_hidden_layers": 1,
                        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                    },
                }
            )
        )
        draft = tmp_path / "draft"
        draft.mkdir(exist_ok=True)
        (draft / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DFlash2DraftModel"],
                    "model_type": "qwen3",
                    "hidden_size": hidden_size,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "max_position_embeddings": 524288,
                    "vocab_size": 128,
                    "dflash_config": {
                        "selector_top_k": 16,
                        "target_feature_reduction": "hc_mean",
                    },
                }
            )
        )
        model = ModelConfig(model=str(target), dtype="float16", max_model_len=524288)
        parallel = ParallelConfig(
            tensor_parallel_size=4,
            decode_context_parallel_size=2 if hidden_size == 2560 else 1,
            distributed_executor_backend="mp",
        )
        spec = (
            None
            if method is None
            else SpeculativeConfig(
                target_model_config=model,
                target_parallel_config=parallel,
                model=str(draft) if method == "dflash" else None,
                method=method,
                num_speculative_tokens=7 if method == "dflash" else 4,
                draft_sample_method="greedy",
                sm70_dflash2=Sm70DFlash2Config(
                    spec_state_replay=replay, fused_gdn_verify=packed
                ),
            )
        )
        return VllmConfig(
            model_config=model,
            speculative_config=spec,
            parallel_config=parallel,
            device_config=DeviceConfig(device="cpu"),
            cache_config=CacheConfig(
                enable_prefix_caching=True, mamba_cache_mode="align"
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=8,
                max_num_batched_tokens=8192,
                max_model_len=524288,
                is_encoder_decoder=model.is_encoder_decoder,
                async_scheduling=False,
            ),
        )

    yield build
    envs.disable_envs_cache()


def test_explicit_replay_overrides_qualified_packed_default(engine_config):
    config = engine_config(replay=True)
    spec = config.speculative_config
    assert spec is not None
    assert spec.sm70_dflash2.qualified
    assert spec.gdn_spec_state_replay
    assert spec.sm70_dflash2.fused_gdn_verify is False


@pytest.mark.parametrize(
    "hidden_size,qualified,replay,packed",
    [
        (2560, False, True, False),
        (5120, True, False, True),
    ],
)
def test_default_replay_yields_only_to_enabled_packed_verifier(
    engine_config, caplog, hidden_size, qualified, replay, packed
):
    spec = engine_config(hidden_size).speculative_config
    assert spec is not None
    assert spec.sm70_dflash2.resolved
    assert spec.sm70_dflash2.qualified is qualified
    assert spec.gdn_spec_state_replay is replay
    assert spec.sm70_dflash2.fused_gdn_verify is packed
    if packed:
        assert "spec_state_replay=None gives precedence" in caplog.text


def test_explicit_replay_logs_why_the_packed_default_was_disabled(
    engine_config, caplog
):
    engine_config(replay=True)
    assert "Disabling the automatic sm70_dflash2.fused_gdn_verify" in caplog.text
    assert "spec_state_replay=True was explicitly requested" in caplog.text


@pytest.mark.parametrize("legacy", [False, True])
def test_explicit_replay_and_explicit_packed_are_a_configuration_error(
    engine_config, monkeypatch, legacy
):
    if legacy:
        monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY", "1")
    with pytest.raises(
        ValueError, match=r"spec_state_replay=True.*fused_gdn_verify=True"
    ):
        engine_config(replay=True, packed=None if legacy else True)


@pytest.mark.parametrize(
    "replay,packed,expected",
    [
        (False, None, False),
        (False, False, False),
        (False, True, False),
        (True, False, True),
        (None, False, True),
        (None, True, False),
    ],
)
def test_explicit_choices_are_respected(engine_config, replay, packed, expected):
    spec = engine_config(replay=replay, packed=packed).speculative_config
    assert spec is not None
    assert spec.gdn_spec_state_replay is expected
    if packed is not None:
        assert spec.sm70_dflash2.fused_gdn_verify is packed


@pytest.mark.parametrize("method", [None, "mtp"])
@pytest.mark.parametrize("replay", [None, False, True])
def test_other_methods_keep_their_compilation_configuration(
    engine_config, method, replay
):
    config = engine_config(
        2560, method=method, replay=replay, packed=True if replay is True else None
    )
    spec = config.speculative_config
    assert spec is None or not spec.gdn_spec_state_replay
    if spec is not None:
        assert spec.sm70_dflash2.fused_gdn_verify is (replay is True)
    # Optional machine-readable evidence for paired baseline/branch runs.
    receipt_dir = os.environ.get("REPLAY_POLICY_RECEIPTS")
    if receipt_dir:
        from pathlib import Path

        payload = {
            "compilation": asdict(config.compilation_config),
            "graph_options": spec.sm70_dflash2.graph_options() if spec else None,
            "spec_hash": spec.compute_hash() if spec else None,
            "config_hash": config.compute_hash(),
            "environment": {
                k: v for k, v in os.environ.items() if k.startswith("VLLM_")
            },
        }
        Path(receipt_dir).mkdir(parents=True, exist_ok=True)
        Path(receipt_dir, f"{method}-{replay}.json").write_text(
            json.dumps(payload, sort_keys=True, default=str, indent=2) + "\n"
        )


def test_explicit_packed_false_wins_over_legacy_packed_true(engine_config, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY", "1")
    spec = engine_config(replay=True, packed=False).speculative_config
    assert spec is not None
    assert spec.gdn_spec_state_replay
    assert spec.sm70_dflash2.fused_gdn_verify is False
