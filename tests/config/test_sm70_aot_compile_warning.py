# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The regular torch.compile warning names only an operator's explicit choice."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.config.vllm as config_module
import vllm.revolta_profile as profile
from vllm import envs

_WARNING = "explicit VLLM_USE_AOT_COMPILE=0 is a diagnostic-only"


def _model():
    return SimpleNamespace(
        quantization="awq",
        dtype=torch.float16,
        hf_text_config=SimpleNamespace(
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            layer_types=["linear_attention", "full_attention"],
        ),
    )


@pytest.fixture
def logger(monkeypatch):
    # Fresh profile state in an empty environment; a Mock avoids *_once dedup.
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(profile, "_PROFILE_VALUES", {})
    monkeypatch.setattr(profile, "_INITIAL_ENV", {})
    monkeypatch.setattr(profile, "_INHERITED_PROVENANCE", {})
    envs.disable_envs_cache()
    mock = Mock()
    monkeypatch.setattr(config_module, "logger", mock)
    yield mock
    envs.disable_envs_cache()


def _warned(logger: Mock) -> bool:
    return any(_WARNING in call.args[0] for call in logger.warning_once.mock_calls)


def test_profile_set_aot_compile_off_does_not_warn(logger):
    assert profile.apply_revolta_profile(_model(), 4, is_sm70=True)
    assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"
    assert profile.value_source("VLLM_USE_AOT_COMPILE") == "profile"

    config_module._log_sm70_compile_graph_without_aot()

    assert not _warned(logger)
    logger.info_once.assert_called_once()
    assert "ReVolta profile" in logger.info_once.call_args.args[0]


def test_profile_value_inherited_by_a_child_process_does_not_warn(logger):
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"
    profile._INHERITED_PROVENANCE["VLLM_USE_AOT_COMPILE"] = {
        "value": "0",
        "source": "profile",
    }

    config_module._log_sm70_compile_graph_without_aot()

    assert not _warned(logger)


@pytest.mark.parametrize("profile_enabled", [True, False])
def test_explicit_aot_compile_off_warns(logger, profile_enabled):
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"
    tp_size = 4 if profile_enabled else 2
    assert (
        profile.apply_revolta_profile(_model(), tp_size, is_sm70=True)
        == profile_enabled
    )
    assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"
    assert profile.value_source("VLLM_USE_AOT_COMPILE") != "profile"

    config_module._log_sm70_compile_graph_without_aot()

    assert _warned(logger)
    logger.info_once.assert_not_called()


def test_post_init_routes_the_regular_compile_case_through_the_helper():
    assert config_module.__file__ is not None
    source = Path(config_module.__file__).read_text()
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VllmConfig"
    )
    post_init = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
    )
    calls = [
        node.func.id
        for node in ast.walk(post_init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("_log_sm70_compile_graph_without_aot") == 1
    assert source.count(_WARNING) == 1
