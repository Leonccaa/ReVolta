# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReVolta does not send usage stats to the upstream vLLM server."""

import pytest

from vllm.usage import usage_lib
from vllm.v1 import utils as v1_utils


@pytest.fixture
def no_opt_out(monkeypatch, tmp_path):
    for name in ("VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK", "DO_NOT_TRACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_CONFIG_ROOT", str(tmp_path))


def test_usage_stats_off_without_opt_out(no_opt_out):
    assert usage_lib.is_usage_stats_enabled() is False


def test_report_usage_stats_sends_nothing(no_opt_out, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("usage stats must not be reported")

    monkeypatch.setattr(usage_lib.usage_message, "report_usage", fail)
    monkeypatch.setattr(v1_utils.usage_message, "report_usage", fail)
    # The config is never read: the function returns before touching it.
    v1_utils.report_usage_stats(vllm_config=None)
