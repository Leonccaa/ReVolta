# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for acceptance-driven dynamic width contraction in DFlash2."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from tests.v1.core import test_long_context_admission as hybrid
from tests.v1.core.test_mamba_sparse_retention import _r1_sample
from vllm.config import CUDAGraphMode, SpeculativeConfig
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.config.vllm import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.spec_decode.dynamic.adaptive_dflash2 import (
    AWQ_GROUPED_SWEET_SPOT_ROWS,
    MAX_ADAPTIVE_DFLASH2_DRAFTS,
    MIN_ADAPTIVE_DFLASH2_DRAFTS,
    AdaptiveDFlash2Depth,
    choose_batch_dflash2_drafts,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
pytest_plugins = ("tests.v1.spec_decode.test_dflash2_native_lookup",)


def test_sm70_dflash2_adaptive_width_policy_default_and_graph_options():
    default_cfg = Sm70DFlash2Config()
    assert default_cfg.adaptive_width is False
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
    assert choose_batch_dflash2_drafts(8, 7, depths_mod) == 5

    # A C8 batch with low acceptance uses 40 rows.
    depths_low = [
        AdaptiveDFlash2Depth(depth=4, successes=6.2, trials=10.0) for _ in range(8)
    ]
    assert choose_batch_dflash2_drafts(8, 7, depths_low) == 4

    # High acceptance restores full width even before depths have caught up.
    depths_high = [
        AdaptiveDFlash2Depth(depth=4, successes=9.2, trials=10.0) for _ in range(8)
    ]
    assert choose_batch_dflash2_drafts(8, 7, depths_high) == 7
    # A configured concurrency width below the adaptive floor stays in force.
    assert choose_batch_dflash2_drafts(8, 2, depths_mod) == 2


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
    CudaGraphManager._add_adaptive_dflash2_candidates(mgr)
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
    # C8 q=7 exceeds the grouped budget.
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
            if requests * q > AWQ_GROUPED_SWEET_SPOT_ROWS:
                continue
            desc = CudaGraphManager.dispatch(mgr, requests, requests * q, q)
            assert desc.cg_mode == CUDAGraphMode.FULL
            assert desc.uniform_token_count == q
            assert requests <= desc.num_reqs
            assert desc.num_tokens <= AWQ_GROUPED_SWEET_SPOT_ROWS


def _spec(base, **options):
    return SpeculativeConfig(
        method="dflash",
        model=base.model,
        target_model_config=base.target_model_config,
        target_parallel_config=base.target_parallel_config,
        sm70_dflash2={"adaptive_width": True},
        **({"num_speculative_tokens": 7} | options),
    )


@pytest.mark.parametrize(
    "mode,options",
    [
        ("off", {}),
        (
            "gpu_head",
            {
                "dflash2_lookup": {
                    "enabled": True,
                    "policy": "longest",
                    "nmin": 2,
                    "nmax": 4,
                }
            },
        ),
        ("legacy_host", {"ngram_assist": True}),
        ("legacy_gpu", {"ngram_assist": True, "num_speculative_tokens": 15}),
    ],
)
def test_adaptive_width_admission(native_allocation_config, monkeypatch, mode, options):
    monkeypatch.setattr(current_platform, "is_device_capability", lambda c: c == (7, 0))
    spec = _spec(native_allocation_config.speculative_config, **options)
    assert spec.resolved_dflash2_lookup.mode == mode
    cfg = NS(
        speculative_config=spec,
        model_config=NS(quantization="awq"),
        scheduler_config=NS(async_scheduling=False),
        use_v2_model_runner=True,
        compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE),
    )
    enabled = VllmConfig.adaptive_dflash2_enabled.fget
    assert enabled(cfg) is (mode == "off")
    cfg.scheduler_config.async_scheduling = True
    assert not enabled(cfg)
    cfg.scheduler_config.async_scheduling = False
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    assert not enabled(cfg)
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    cfg.model_config.quantization = "modelopt"
    assert not enabled(cfg)


def _build_scheduler(native_allocation_config, monkeypatch, **options):
    make_config = hybrid.make_config
    production_specs = hybrid.production_specs
    replay = options.get("exact_end_draft_threshold") is not None

    def configure(**kwargs):
        config = make_config(**kwargs)
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        monkeypatch.setattr(
            current_platform, "is_device_capability", lambda c: c == (7, 0)
        )
        config.speculative_config = _spec(
            native_allocation_config.speculative_config, **options
        )
        config.speculative_config.gdn_spec_state_replay = replay
        config.model_config.quantization = "awq"
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        return config

    monkeypatch.setattr(hybrid, "make_config", configure)
    if replay:
        monkeypatch.setattr(
            hybrid,
            "production_specs",
            lambda **kwargs: production_specs(**(kwargs | {"speculative_blocks": 0})),
        )
    scheduler = hybrid.build(None, speculative_tokens=7)
    assert scheduler.vllm_config.adaptive_dflash2_enabled
    return scheduler


@pytest.mark.parametrize("dsd", [None, [(1, 7, 7), (8, 8, 2)]])
def test_sync_scheduler_uses_next_proposal_width_and_cleans_up(
    native_allocation_config, monkeypatch, dsd
):
    scheduler = _build_scheduler(
        native_allocation_config, monkeypatch, num_speculative_tokens_per_batch_size=dsd
    )
    requests = hybrid.make_requests(8, 10, 128)
    for req in requests:
        scheduler.add_request(req)
    prefill = scheduler.schedule()
    initial, narrowed = (7, 5) if dsd is None else (2, 2)
    assert prefill.num_spec_tokens_to_schedule == initial
    _r1_sample(scheduler, prefill)
    first = scheduler.schedule()
    assert len(first.scheduled_spec_decode_tokens) == 8
    assert {len(ids) for ids in first.scheduled_spec_decode_tokens.values()} == {
        initial
    }
    _r1_sample(scheduler, first, "none")
    second = scheduler.schedule()
    assert second.num_spec_tokens_to_schedule == narrowed
    assert {len(ids) for ids in second.scheduled_spec_decode_tokens.values()} == {
        initial
    }
    _r1_sample(scheduler, second, "none")
    third = scheduler.schedule()
    assert {len(ids) for ids in third.scheduled_spec_decode_tokens.values()} == {
        narrowed
    }
    _r1_sample(scheduler, third, "none")
    preempted = requests[0]
    scheduler.running.remove(preempted)
    scheduler._preempt_request(preempted, 0.0)
    assert preempted.request_id not in scheduler._adaptive_dflash2_depths


def test_clipped_terminal_round_does_not_train_width(
    native_allocation_config, monkeypatch
):
    scheduler = _build_scheduler(
        native_allocation_config,
        monkeypatch,
        exact_end_draft_threshold=3,
        num_speculative_tokens_per_batch_size=[(1, 8, 7)],
    )
    assert scheduler.exact_end_draft_threshold == 3
    (request,) = hybrid.make_requests(1, 10, 6)
    scheduler.add_request(request)
    _r1_sample(scheduler, scheduler.schedule())
    clipped = scheduler.schedule()
    assert len(clipped.scheduled_spec_decode_tokens[request.request_id]) == 4
    _r1_sample(scheduler, clipped, "none")
    assert request.num_output_tokens == 2
    assert request.request_id not in scheduler._adaptive_dflash2_depths
