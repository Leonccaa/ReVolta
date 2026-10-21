# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 profile defaults, weight-format admission, and worker provenance."""

import importlib.util
import json
import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs

NVFP4 = {
    "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE": "1",
    "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE": "1",
    "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE": "1",
    "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "0",
    "VLLM_SM70_NVFP4_TURBOMIND": "1",
    "VLLM_SM70_FP8_TURBOMIND": "1",
    "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL": "1",
    "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES": "1",
    "VLLM_V2_CUDAGRAPH_MEM_MIB": "700",
}


def load_profile():
    spec = importlib.util.find_spec("vllm.revolta_profile")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model(algo, nested=False):
    quant = {
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.layers.0.mlp.experts.0.gate_proj": {"quant_algo": algo}
        },
    }
    if nested:
        quant = {"quantization": quant}
    return NS(
        quantization="modelopt_mixed",
        dtype=torch.float16,
        hf_config=NS(quantization_config=quant),
        hf_text_config=NS(
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            layer_types=["full_attention"],
        ),
    )


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    envs.disable_envs_cache()
    yield load_profile()
    envs.disable_envs_cache()


@pytest.mark.parametrize("algo", ["NVFP4", "W4A16_NVFP4"])
@pytest.mark.parametrize("nested", [False, True])
def test_clean_nvfp4_defaults_and_worker_sources(profile, algo, nested):
    config = model(algo, nested)
    assert profile.apply_revolta_profile(config, 4, is_sm70=True)
    for module in (profile, load_profile()):
        audit = json.loads(
            module.profile_audit(enabled=True, quantization="modelopt_mixed")
        )
        for name, value in NVFP4.items():
            assert os.environ.get(name) == value
            assert audit["values"][name] == {"value": value, "source": "profile"}


@pytest.mark.parametrize("algo", ["NVFP4", "W4A16_NVFP4"])
@pytest.mark.parametrize("name,value", NVFP4.items())
def test_nvfp4_explicit_settings_win(profile, monkeypatch, algo, name, value):
    override = "0" if value == "1" else "1"
    monkeypatch.setenv(name, override)
    assert profile.apply_revolta_profile(model(algo), 4, is_sm70=True)
    for module in (profile, load_profile()):
        audit = json.loads(
            module.profile_audit(enabled=True, quantization="modelopt_mixed")
        )
        assert audit["values"][name] == {"value": override, "source": "environment"}


def test_mixed_fp8_does_not_enable_nvfp4(profile):
    assert profile.apply_revolta_profile(model("MXFP8"), 4, is_sm70=True)
    assert not set(NVFP4) & set(os.environ)


@pytest.mark.parametrize("tp,sm70", [(1, True), (4, False)])
def test_nvfp4_respects_profile_scope(profile, tp, sm70):
    assert not profile.apply_revolta_profile(model("NVFP4"), tp, is_sm70=sm70)
    assert not set(NVFP4) & set(os.environ)


def test_resolved_modelopt_fp4(profile):
    config = model("NVFP4")
    config.quantization = "modelopt_fp4"
    config.hf_config = NS()
    assert profile.apply_revolta_profile(config, 4, is_sm70=True)
    assert all(os.environ.get(name) == value for name, value in NVFP4.items())
