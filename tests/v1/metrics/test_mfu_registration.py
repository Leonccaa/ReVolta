# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MFU registration follows its own existing configuration switch."""

from functools import partial
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter

from vllm.v1.metrics.perf import PerfMetricsProm, PerfStats


@pytest.mark.parametrize("enabled", [False, True])
def test_mfu_registration_and_observe_follow_switch(enabled):
    config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_mfu_metrics=enabled)
    )
    registry = CollectorRegistry()

    class Isolated(PerfMetricsProm):
        _counter_cls = staticmethod(partial(Counter, registry=registry))

    metrics = Isolated(config, ["engine"], {0: ["0"]})
    assert bool(list(registry.collect())) == enabled
    metrics.observe(PerfStats())
