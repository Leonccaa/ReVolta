# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for acceptance-driven dynamic width contraction in DFlash2."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from tests.v1.core import test_long_context_admission as hybrid
from tests.v1.core.test_mamba_sparse_retention import _r1_sample
from vllm.config import CUDAGraphMode
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.config.vllm import VllmConfig
from vllm.v1.spec_decode.dynamic.adaptive_dflash2 import (
    AWQ_GROUPED_SWEET_SPOT_ROWS,
    MAX_ADAPTIVE_DFLASH2_DRAFTS,
    MIN_ADAPTIVE_DFLASH2_DRAFTS,
    AdaptiveDFlash2Depth,
    choose_batch_dflash2_drafts,
    limit_dflash2_drafts,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_sm70_dflash2_adaptive_width_policy_default_and_graph_options():
    default_cfg = Sm70DFlash2Config()
    assert default_cfg.adaptive_width is False
    assert default_cfg.adaptive_width_max_grouped_rows == 48
    assert "adaptive_width" not in default_cfg.graph_options()

    opt_in = Sm70DFlash2Config(adaptive_width=True)
    assert opt_in.adaptive_width is True
    assert opt_in.graph_options()["adaptive_width"] is True


def test_adaptive_dflash2_depth_contracts_and_recovers():
    tracker = AdaptiveDFlash2Depth(depth=7)
    # Low acceptance (1 accepted out of drafted -> p -> 0.5): contracts down to 3 (q=4)
    for _ in range(15):
        tracker.observe(tracker.depth, 1, cap=7)
    assert tracker.depth == MIN_ADAPTIVE_DFLASH2_DRAFTS
    assert tracker.accept_rate < 0.58

    # Full acceptance ramps back up one step per round to cap=7 (q=8)
    for _ in range(4):
        tracker.observe(tracker.depth, tracker.depth, cap=7)
    assert tracker.depth == MAX_ADAPTIVE_DFLASH2_DRAFTS


def test_adaptive_dflash2_depth_hysteresis_avoids_boundary_jitter():
    # Near the boundary between depths 4 and 5.
    tracker = AdaptiveDFlash2Depth(depth=5, successes=7.0, trials=10.0)
    # Two accepts and one rejection keep p inside the 0.70 hysteresis band.
    tracker.observe(5, 2, cap=7)
    assert tracker.depth == 5


def test_choose_batch_dflash2_drafts_c1_and_c8_awq_sweet_spot():
    # C1 follows request depth directly
    d_low = AdaptiveDFlash2Depth(depth=4, successes=6.5, trials=10.0)
    assert choose_batch_dflash2_drafts(1, 7, [d_low]) == 4

    # C8 with moderate acceptance contracts from 56 to 48 rows.
    depths_mod = [
        AdaptiveDFlash2Depth(depth=6, successes=7.8, trials=10.0) for _ in range(8)
    ]
    assert (
        choose_batch_dflash2_drafts(
            8, 7, depths_mod, max_grouped_rows=AWQ_GROUPED_SWEET_SPOT_ROWS
        )
        == 5
    )

    # A C8 batch with low acceptance uses 40 rows.
    depths_low = [
        AdaptiveDFlash2Depth(depth=4, successes=6.2, trials=10.0) for _ in range(8)
    ]
    assert (
        choose_batch_dflash2_drafts(
            8, 7, depths_low, max_grouped_rows=AWQ_GROUPED_SWEET_SPOT_ROWS
        )
        == 4
    )

    # High acceptance restores full width even before depths have caught up.
    depths_high = [
        AdaptiveDFlash2Depth(depth=4, successes=9.2, trials=10.0) for _ in range(8)
    ]
    assert (
        choose_batch_dflash2_drafts(
            8, 7, depths_high, max_grouped_rows=AWQ_GROUPED_SWEET_SPOT_ROWS
        )
        == 7
    )

    # Disabling grouped row cap (max_grouped_rows=0) keeps depth=6 at C8
    assert choose_batch_dflash2_drafts(8, 7, depths_mod, max_grouped_rows=0) == 6
    assert choose_batch_dflash2_drafts(8, 7, depths_mod[:7]) == 7
    assert choose_batch_dflash2_drafts(8, 7, depths_mod, max_grouped_rows=31) == 7


def test_cudagraph_manager_registers_adaptive_dflash2_sweet_spot_descriptors():
    base_full = [
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=n * 8,
            num_reqs=n,
            uniform_token_count=8,
        )
        for n in (1, 2, 4, 8)
    ]
    mgr = NS(
        decode_query_len=8,
        max_num_reqs=8,
        _capture_descs={CUDAGraphMode.FULL: list(base_full)},
        _candidates=[[] for _ in range(65)],
        _graphs_captured=True,
    )
    CudaGraphManager._add_adaptive_dflash2_candidates(mgr, max_grouped_rows=48)
    descs = set(mgr._capture_descs[CUDAGraphMode.FULL])
    # C1 widths q=4, 5, 6, 7
    for q in (4, 5, 6, 7):
        assert (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=q,
                num_reqs=1,
                uniform_token_count=q,
            )
            in descs
        )
    # C8 AWQ grouped sweet spots: q=4 (M=32), q=5 (M=40), q=6 (M=48)
    for q in (4, 5, 6):
        assert (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=8 * q,
                num_reqs=8,
                uniform_token_count=q,
            )
            in descs
        )
    # C8 q=7 (M=56 > 48) should NOT be added under max_grouped_rows=48
    assert (
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=56,
            num_reqs=8,
            uniform_token_count=7,
        )
        not in descs
    )
    for requests in range(1, 9):
        for q in range(4, 8):
            if requests * q > 48:
                continue
            desc = CudaGraphManager.dispatch(mgr, requests, requests * q, q)
            assert desc.cg_mode == CUDAGraphMode.FULL
            assert desc.uniform_token_count == q
            assert requests <= desc.num_reqs and desc.num_tokens <= 48


def test_adaptive_width_admission(monkeypatch):
    import vllm.platforms

    device = NS(is_cuda=lambda: True, is_device_capability=lambda cap: cap == (7, 0))
    monkeypatch.setattr(vllm.platforms, "_current_platform", device)
    spec = NS(
        method="dflash",
        use_dflash=lambda: True,
        num_speculative_tokens=7,
        sm70_dflash2=Sm70DFlash2Config(adaptive_width=True),
        draft_model_config=NS(
            hf_config=NS(dflash_config={"selector_top_k": 20, "block_size": 8})
        ),
    )
    cfg = NS(
        speculative_config=spec,
        model_config=NS(quantization="awq"),
        scheduler_config=NS(async_scheduling=False),
        use_v2_model_runner=True,
        compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE),
    )
    enabled = VllmConfig.adaptive_dflash2_enabled.fget
    assert enabled(cfg)
    cfg.scheduler_config.async_scheduling = True
    assert not enabled(cfg)
    cfg.scheduler_config.async_scheduling = False
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    assert not enabled(cfg)
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    spec.num_speculative_tokens = 15
    spec.ngram_assist = True
    assert enabled(cfg)
    spec.ngram_assist = False
    assert not enabled(cfg)
    cfg.model_config.quantization = "modelopt"
    spec.num_speculative_tokens = 7
    assert not enabled(cfg)


def test_limit_only_the_native_dflash2_width():
    assert limit_dflash2_drafts(5, None) == 5
    assert limit_dflash2_drafts(5, 7) == 5
    assert limit_dflash2_drafts(5, 15) == 15
    assert limit_dflash2_drafts(0, 0) == 0


def test_sync_scheduler_uses_next_proposal_width_and_cleans_up(monkeypatch):
    make_config = hybrid.make_config

    def configure(**kwargs):
        config = make_config(**kwargs)
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        return config

    monkeypatch.setattr(hybrid, "make_config", configure)
    monkeypatch.setattr(
        VllmConfig, "adaptive_dflash2_enabled", property(lambda self: True)
    )
    scheduler = hybrid.build(None, speculative_tokens=7)
    requests = hybrid.make_requests(8, 10, 128)
    for req in requests:
        scheduler.add_request(req)
    prefill = scheduler.schedule()
    assert prefill.num_spec_tokens_to_schedule == 7
    _r1_sample(scheduler, prefill)
    first = scheduler.schedule()
    assert len(first.scheduled_spec_decode_tokens) == 8
    assert {len(ids) for ids in first.scheduled_spec_decode_tokens.values()} == {7}
    _r1_sample(scheduler, first, "none")
    second = scheduler.schedule()
    assert second.num_spec_tokens_to_schedule == 5
    assert {len(ids) for ids in second.scheduled_spec_decode_tokens.values()} == {7}
    _r1_sample(scheduler, second, "none")
    third = scheduler.schedule()
    assert {len(ids) for ids in third.scheduled_spec_decode_tokens.values()} == {5}
    _r1_sample(scheduler, third, "none")
    preempted = requests[0]
    scheduler.running.remove(preempted)
    scheduler._preempt_request(preempted, 0.0)
    assert preempted.request_id not in scheduler._adaptive_dflash2_depths
