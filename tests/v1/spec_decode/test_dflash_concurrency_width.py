# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash verifies fewer drafts per request at high concurrency."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.speculative import (
    check_dflash_concurrency_drafts,
    dflash_drafts_for_requests,
)
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("policy", "num_reqs", "drafts"),
    [
        (None, 8, 7),
        ({5: 4}, 1, 7),
        ({5: 4}, 4, 7),
        ({5: 4}, 5, 4),
        ({5: 4}, 8, 4),
        ({3: 5, 5: 3}, 3, 5),
        ({3: 5, 5: 3}, 6, 3),
    ],
)
def test_drafts_follow_the_running_request_count(policy, num_reqs, drafts) -> None:
    spec = SimpleNamespace(num_speculative_tokens=7, dflash_concurrency_drafts=policy)
    assert dflash_drafts_for_requests(spec, num_reqs) == drafts


@pytest.mark.parametrize(
    ("policy", "dflash", "ngram_assist", "ok"),
    [
        (None, False, False, True),
        ({5: 4}, True, False, True),
        ({5: 4}, False, False, False),  # not DFlash
        ({5: 4}, True, True, False),  # lookup owns the width
        ({1: 4}, True, False, False),  # a single request keeps the full width
        ({5: 7}, True, False, False),  # not narrower
        ({5: 0}, True, False, False),
    ],
)
def test_policy_validation(policy, dflash, ngram_assist, ok) -> None:
    def check():
        check_dflash_concurrency_drafts(
            policy, dflash=dflash, ngram_assist=ngram_assist, num_speculative_tokens=7
        )

    if ok:
        check()
    else:
        with pytest.raises(ValueError, match="dflash_concurrency_drafts"):
            check()


def _manager(monkeypatch, policy, tails=True):
    # A V100 worker without the long-context attention operator.
    monkeypatch.setattr(
        "vllm.v1.attention.ops.sm70_e4m3_long.long_attention_enabled", lambda: False
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils._worker_device_is_pre_ampere", lambda: True
    )
    monkeypatch.setenv("VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS", str(int(tails)))
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_global_graph_pool",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_pp_group",
        lambda: MagicMock(is_first_rank=True, is_last_rank=True),
    )
    sizes = [8, 16, 24, 32, 40, 48, 56, 64]
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(tensor_parallel_size=4, data_parallel_size=1),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=sizes,
            max_cudagraph_capture_size=64,
        ),
        speculative_config=SimpleNamespace(
            method="dflash",
            ngram_assist=False,
            num_speculative_tokens=7,
            dflash_concurrency_drafts=policy,
        ),
    )
    manager = ModelCudaGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, decode_query_len=8
    )
    manager._graphs_captured = True
    return manager


@pytest.mark.parametrize("tails", [True, False])
def test_narrow_target_graphs_from_the_threshold_up(monkeypatch, tails) -> None:
    manager = _manager(monkeypatch, {5: 4}, tails)
    assert manager.decode_query_lens[0] == 8 and 5 in manager.decode_query_lens
    for num_reqs in range(5, 9):
        assert num_reqs * 5 in manager._capture_sizes
        desc = manager.dispatch(num_reqs, num_reqs * 5, 5)
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.uniform_token_count == 5 and desc.num_reqs == num_reqs
    # The full width keeps every batch size; the narrow width stays above 4.
    assert manager.dispatch(8, 64, 8).cg_mode == CUDAGraphMode.FULL
    assert manager.dispatch(2, 16, 8).cg_mode == CUDAGraphMode.FULL
    assert manager.dispatch(2, 10, 5).uniform_token_count != 5
    # A single-request tail graph of the same length stays with the tail policy.
    assert (manager.dispatch(1, 5, 5).cg_mode == CUDAGraphMode.FULL) is tails


def test_no_policy_keeps_the_capture_plan(monkeypatch) -> None:
    manager = _manager(monkeypatch, None, tails=False)
    assert manager.decode_query_lens == (8,)
    assert not manager._dflash_width_min_reqs
