# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A8: exercise actual CPU interpreter hot eviction and index receive paths."""

import os

import pytest

from tests.models.qwen4_exp.test_host_kv_shadow import (
    _drive,
    _engine,
    _interpreter_boolean_comparison,
)
from tests.v1.core.test_value_arbiter import make
from vllm.v1.core.value_arbiter import HOT_FORCED_RECENT, HOT_REFETCH

__all__ = ["_interpreter_boolean_comparison"]
pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="CPU interpreter required"
)


def test_hot_pressure_refetch_reaches_p4_window():
    engine = _engine()
    _drive(engine, 12, jumps={5, 9})
    values = engine.store.shadow.counters.tolist()
    assert values[HOT_FORCED_RECENT] > 0 and values[HOT_REFETCH] > 0
    arb, _ = make()
    arb.begin_step(1)
    arb.add_rank_counters(1, 1, values, 1)
    assert arb.windows[0].hot[HOT_REFETCH] == values[HOT_REFETCH]
    assert arb.windows[0].hot[HOT_FORCED_RECENT] == values[HOT_FORCED_RECENT]
