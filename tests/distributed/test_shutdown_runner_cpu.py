# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public runner retirement contracts with CPU tensors and stubbed CUDA calls."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.shutdown import ShutdownError, ShutdownIncompleteError
from vllm.v1.spec_decode import lookup_quarantine
from vllm.v1.worker.gpu import model_runner

GRAPH_NAMES = (
    "_context_kv_graphs",
    "_context_compute_graphs",
    "_context_store_graphs",
    "_draft_metadata_graphs",
)
MANAGER_NAMES = (
    "query_cudagraph_manager",
    "prefill_cudagraph_manager",
    "decode_cudagraph_manager",
)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(lookup_quarantine, "_retained", {})
    monkeypatch.setattr(lookup_quarantine, "_closing", {})
    monkeypatch.setattr(lookup_quarantine, "_reason", None)
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    monkeypatch.setattr(torch.accelerator, "empty_cache", Mock())
    monkeypatch.setattr(model_runner, "free_before_shutdown", Mock())
    result = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    result.cudagraph_manager = object()
    result._ple_offload_connector = SimpleNamespace(close=Mock())
    result.model = torch.nn.Linear(2, 2)
    result.speculator = SimpleNamespace(
        model=torch.nn.Linear(2, 2),
        **{name: {"shape": object()} for name in GRAPH_NAMES},
        **{name: object() for name in MANAGER_NAMES},
    )
    result.kv_caches = [torch.ones(2)]
    result.attn_groups = [object()]
    result.host_history = SimpleNamespace(close=Mock())
    result.kv_connector = SimpleNamespace(close_host_kv=Mock())
    result.vllm_config = SimpleNamespace()
    result.model_state = object()
    return result


@pytest.mark.parametrize("failure_stage", [None, "ple", "history", "free"])
def test_runner_releases_graphs_after_sync_even_when_cleanup_fails(
    monkeypatch, runner, failure_stage
):
    target_ref = weakref.ref(runner.model)
    draft_ref = weakref.ref(runner.speculator.model)
    # External ownership must not keep directly owned speculative graphs alive.
    speculator = runner.speculator
    if failure_stage is not None:
        fail = Mock(side_effect=RuntimeError(failure_stage + " cleanup failed"))
        if failure_stage == "ple":
            runner._ple_offload_connector.close = fail
        elif failure_stage == "history":
            runner.host_history.close = fail
        else:
            monkeypatch.setattr(model_runner, "free_before_shutdown", fail)

    def synchronized():
        assert runner.cudagraph_manager is not None
        assert all(getattr(speculator, name) for name in GRAPH_NAMES)
        assert not runner._shutdown_gpu_retired

    monkeypatch.setattr(torch.accelerator, "synchronize", synchronized)
    # Keep the raised error alive while checking that its traceback does not
    # retain the target model through resource-cleanup frames.
    failure = None
    try:
        runner.shutdown()
    except ShutdownError as error:
        failure = error
    assert (failure is not None) == (failure_stage is not None)
    if failure is not None:
        assert failure_stage + " cleanup failed" in str(failure)
    gc.collect()
    assert runner._shutdown_gpu_retired
    assert not lookup_quarantine.context_isolated()
    assert runner.cudagraph_manager is None and not runner.kv_caches
    assert runner.host_history is None and runner.speculator is None
    assert not runner.attn_groups and not hasattr(runner, "model_state")
    assert all(not getattr(speculator, name) for name in GRAPH_NAMES)
    assert all(getattr(speculator, name) is None for name in MANAGER_NAMES)
    assert target_ref() is None
    # An external speculator may retain its model, but no captured graphs.
    assert draft_ref() is speculator.model
    torch.accelerator.empty_cache.assert_called_once()


def test_failed_sync_retains_owners_and_rejects_retry(monkeypatch, runner):
    runner._shutdown_gpu_retired = True  # Stale proof must be reset.
    model = runner.model
    speculator = runner.speculator
    release = Mock()
    monkeypatch.setattr(runner, "_release_model_resources", release)
    sync = Mock(side_effect=RuntimeError("synchronize failed"))
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)
    with pytest.raises(ShutdownIncompleteError, match="synchronize failed"):
        runner.shutdown()
    assert not runner._shutdown_gpu_retired
    assert lookup_quarantine.isolated(runner)
    assert lookup_quarantine.context_isolated()
    assert runner.model is model and runner.speculator is speculator
    assert runner.cudagraph_manager is not None
    assert all(getattr(speculator, name) for name in GRAPH_NAMES)
    with pytest.raises(ShutdownIncompleteError, match="quarantined"):
        runner.shutdown()
    sync.assert_called_once()
    release.assert_not_called()
    torch.accelerator.empty_cache.assert_not_called()


def test_preexisting_quarantine_does_not_touch_gpu(runner):
    lookup_quarantine.quarantine("existing failure", runner)
    runner.shutdown()
    assert not runner._shutdown_gpu_retired
    assert runner.cudagraph_manager is not None
    torch.accelerator.synchronize.assert_not_called()
    torch.accelerator.empty_cache.assert_not_called()
