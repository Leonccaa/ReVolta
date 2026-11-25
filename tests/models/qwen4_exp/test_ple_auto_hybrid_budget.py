# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.utils import set_lazy_env
from vllm.models.qwen4_exp.nvidia import ple_layer as ple


@pytest.mark.parametrize(
    ("cap", "cascade", "rows"),
    [
        (None, False, (0, 100, 0)),
        (400, False, (0, 100, 0)),
        (120, False, (70, 30, 0)),
        (None, True, (50, 20, 30)),
        (120, True, (50, 30, 20)),
    ],
)
def test_ple_placement_with_and_without_host_cap(monkeypatch, cap, cascade, rows):
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", "0")
    monkeypatch.setattr(ple, "ple_host_budget_bytes", lambda: cap)
    monkeypatch.setattr(ple, "ple_cascade_configured", lambda: cascade)
    monkeypatch.setattr(ple, "_long_context_host_placement", lambda: False)
    table = SimpleNamespace(
        _meta_weight_shape=(100, 4),
        embedding_dim=4,
        _device_spill_bytes=Mock(return_value=200),
        _cap_derived_host_budget=Mock(return_value=80),
    )
    plan = ple.Qwen4ExpPinnedHostEmbedding._plan_placement(table, None)
    assert plan == ple.PLEPlacement(*rows)
    assert table._device_spill_bytes.call_count == int(cascade)
    assert table._cap_derived_host_budget.call_count == int(cascade and cap is None)
