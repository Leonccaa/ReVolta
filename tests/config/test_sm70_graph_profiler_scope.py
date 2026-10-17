# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A speculative draft config must not switch off the target's graph reserve."""

import pytest

from vllm.config.vllm import _should_disable_legacy_sm70_graph_profiler

ENV = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    # Pure config logic: no distributed state or device memory to clean up.
    return False


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_legacy_target_disables_profiler():
    assert _should_disable_legacy_sm70_graph_profiler(False, False)


def test_v2_target_keeps_profiler():
    assert not _should_disable_legacy_sm70_graph_profiler(True, False)


@pytest.mark.parametrize("use_v2", [False, True])
def test_draft_never_disables_profiler(use_v2):
    # The MTP draft config of a V2 target resolves use_v2_model_runner=False
    # (its architecture is not in the V2 list); it must leave the env alone.
    assert not _should_disable_legacy_sm70_graph_profiler(use_v2, True)


@pytest.mark.parametrize("value", ["0", "1"])
def test_explicit_env_is_respected(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert not _should_disable_legacy_sm70_graph_profiler(False, False)
