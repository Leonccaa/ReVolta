# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU configuration regressions for the AWQ MoE WNA16 fallback."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
from vllm.platforms import current_platform

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    "values",
    [
        {"quant_method": "awq", "w_bit": 4, "q_group_size": 128, "zero_point": True},
        {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": False},
    ],
)
def test_config_accepts_both_key_families(monkeypatch, values):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(to_int=lambda: 80),
    )
    config = MoeWNA16Config.from_config(values)
    assert config.weight_bits == 4
    assert config.group_size == 128
    assert config.has_zp is True
    assert config.full_config == values
    assert MoeWNA16Config.is_moe_wna16_compatible(values)


def test_config_prefers_gptq_keys(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(to_int=lambda: 80),
    )
    values = {
        "quant_method": "gptq",
        "bits": 8,
        "group_size": 64,
        "w_bit": 4,
        "q_group_size": 128,
        "sym": True,
    }
    config = MoeWNA16Config.from_config(values)
    assert config.weight_bits == 8
    assert config.group_size == 64
    assert MoeWNA16Config.is_moe_wna16_compatible(values)
    values["bits"] = 3
    assert not MoeWNA16Config.is_moe_wna16_compatible(values)
