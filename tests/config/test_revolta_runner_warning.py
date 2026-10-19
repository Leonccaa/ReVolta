# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tests.utils import set_lazy_env
from vllm.config import VllmConfig
from vllm.config import vllm as config_module


@pytest.mark.parametrize(
    "reason",
    ["ec_transfer_config", "logprobs_mode=processed_logits", "Triton (unavailable)"],
)
def test_profile_fallback_explains_qwen4exp_risk_and_explicit_opt_in(
    monkeypatch, reason
):
    import vllm.revolta_profile as profile

    set_lazy_env(monkeypatch, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(profile, "all_visible_devices_sm70", lambda: True)
    monkeypatch.setattr(config_module, "HAS_TRITON", reason != "Triton (unavailable)")
    model = NS(
        architectures=["Qwen4ExpForConditionalGeneration"],
        dtype=torch.float16,
        hf_text_config=NS(
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            layer_types=["full_attention"],
        ),
    )
    config = NS(
        model_config=model,
        parallel_config=NS(tensor_parallel_size=4),
        speculative_config=None,
        _is_default_v2_model_runner_model=lambda: True,
        _get_v2_model_runner_unsupported_features=lambda: [reason],
    )
    warning = Mock()
    monkeypatch.setattr(config_module.logger, "warning_once", warning)
    assert VllmConfig.use_v2_model_runner.fget(config) is False
    messages = [call.args[0] % call.args[1:] for call in warning.call_args_list]
    assert any(
        reason in msg
        and "QSA ring cache" in msg
        and "PLE" in msg
        and "VLLM_USE_V2_MODEL_RUNNER=1" in msg
        for msg in messages
    )


def test_supported_automatic_qwen4exp_still_selects_v2(monkeypatch):
    set_lazy_env(monkeypatch, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(config_module, "HAS_TRITON", True)
    config = NS(
        model_config=NS(architectures=["Qwen4ExpForConditionalGeneration"]),
        speculative_config=None,
        _is_default_v2_model_runner_model=lambda: True,
        _get_v2_model_runner_unsupported_features=lambda: [],
    )
    assert VllmConfig.use_v2_model_runner.fget(config) is True
