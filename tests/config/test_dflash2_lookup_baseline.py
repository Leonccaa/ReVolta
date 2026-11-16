# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-main compatibility fingerprints for normalized and serialized configs."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from transformers import Qwen3Config

from vllm import envs
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.config.utils import normalize_value

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def controlled_lookup_environment(monkeypatch):
    envs.disable_envs_cache()
    for name in list(os.environ):
        if name.startswith("VLLM_DFLASH2_LOOKUP_"):
            monkeypatch.delenv(name)
    yield
    envs.disable_envs_cache()


def test_registered_lookup_defaults_match_fixed_main():
    fixture = Path(__file__).with_name("data") / "dflash2_lookup_baseline.json"
    expected = json.loads(fixture.read_text())["lookup_env_defaults"]
    actual = {
        name: value
        for name, value in envs.compile_factors().items()
        if name.startswith("VLLM_DFLASH2_LOOKUP_")
    }
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _serialization_fallback(value):
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return normalize_value(value)


def compatibility_snapshot(tmp_path, new_config=None):
    for name, arch in (("target", "Qwen3ForCausalLM"), ("draft", "DFlash2DraftModel")):
        Qwen3Config(
            architectures=[arch],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            max_position_embeddings=512,
            dflash_config={"selector_top_k": 16, "block_size": 8},
        ).save_pretrained(tmp_path / name)
    target = ModelConfig(
        model=str(tmp_path / "target"),
        dtype="float16",
        max_model_len=512,
        skip_tokenizer_init=True,
    )
    parallel = ParallelConfig()
    adapter = TypeAdapter(SpeculativeConfig)
    snapshots = {}
    for name, extra in (
        ("off", {}),
        ("legacy_host", {"ngram_assist": True}),
        ("legacy_gpu", {"ngram_assist": True, "num_speculative_tokens": 15}),
        ("legacy_prompt", {"prompt_lookup_min": 3, "prompt_lookup_max": 16}),
        (
            "ngram",
            {
                "method": "ngram",
                "model": "ngram",
                "prompt_lookup_min": 3,
                "prompt_lookup_max": 16,
            },
        ),
    ):
        values = (
            dict(
                method="dflash",
                model=str(tmp_path / "draft"),
                num_speculative_tokens=7,
                target_model_config=target,
                target_parallel_config=parallel,
            )
            | extra
        )
        if new_config is not None:
            values["dflash2_lookup"] = new_config
        spec = SpeculativeConfig(**values)
        # Include the complete diagnostic serialization, including target and
        # draft configs. Canonicalize only the per-run temporary checkpoint root.
        serialized = json.dumps(
            adapter.dump_python(spec), default=_serialization_fallback, sort_keys=True
        ).replace(str(tmp_path), "<checkpoint>")
        factors = (
            envs.compile_factors(spec)
            if hasattr(spec, "resolved_dflash2_lookup")
            else envs.compile_factors()
        )
        snapshots[name] = {
            "compile_factors_sha256": hashlib.sha256(
                json.dumps(factors, sort_keys=True).encode()
            ).hexdigest(),
            "serialization_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "speculative_hash": spec.compute_hash(),
            "lookup_env_factors": {
                name: value
                for name, value in factors.items()
                if name.startswith("VLLM_DFLASH2_LOOKUP_")
            },
        }
    return snapshots


@pytest.mark.parametrize(
    "new_config",
    [
        None,
        {},
        {"enabled": False},
        {"enabled": False, "policy": "longest", "nmin": 3, "nmax": 16},
    ],
)
def test_existing_configuration_matches_fixed_main(tmp_path, new_config):
    fixture = Path(__file__).with_name("data") / "dflash2_lookup_baseline.json"
    expected = json.loads(fixture.read_text())
    actual = compatibility_snapshot(tmp_path, new_config)
    # Pin lookup defaults independently of candidate getters. Other environment
    # factors remain host-dependent and are compared within the same process.
    live_env_hash = hashlib.sha256(
        json.dumps(envs.compile_factors(), sort_keys=True).encode()
    ).hexdigest()
    for name, snapshot in actual.items():
        assert json.dumps(snapshot["lookup_env_factors"], sort_keys=True) == json.dumps(
            expected["lookup_env_defaults"], sort_keys=True
        )
        assert snapshot["compile_factors_sha256"] == live_env_hash
        assert (
            snapshot["serialization_sha256"]
            == expected["snapshots"][name]["serialization_sha256"]
        )
        assert (
            snapshot["speculative_hash"]
            == expected["snapshots"][name]["speculative_hash"]
        )


def test_fixed_default_fingerprint_detects_boolean_zero(monkeypatch):
    monkeypatch.setitem(
        envs.environment_variables, "VLLM_DFLASH2_LOOKUP_AGREE", lambda: False
    )
    with pytest.raises(AssertionError):
        test_registered_lookup_defaults_match_fixed_main()
