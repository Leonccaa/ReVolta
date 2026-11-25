# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core import test_long_context_admission as hybrid
from tests.v1.core.utils import create_requests
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.dynamic.adaptive_mtp import AdaptiveMTPDepth
from vllm.v1.worker.gpu import cudagraph_utils as graphs
from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import PrefillEagleCudaGraphManager

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def build_scheduler(monkeypatch):
    make_config = hybrid.make_config

    def build(*, k=4, cap=4, method="mtp", schedule=True, async_scheduling=False):
        def flash_next_config(**kwargs):
            config = make_config(**kwargs)
            spec = config.speculative_config
            spec.draft_model_config = SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen4_exp_mtp"),
                max_model_len=config.model_config.max_model_len,
            )
            spec.num_speculative_tokens_per_batch_size = (
                [(1, 1, cap), (2, 8, 3)] if schedule else None
            )
            monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
            return config

        monkeypatch.setattr(hybrid, "make_config", flash_next_config)
        scheduler = hybrid.build(
            None,
            speculative_tokens=k,
            speculative_method=method,
            async_scheduling=async_scheduling,
        )
        assert scheduler.use_v2_model_runner
        return scheduler

    return build


def _run_step(scheduler, *, accept=0, invalid=False):
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    if invalid:
        output.num_invalid_spec_tokens = {req_ids[0]: 1}
    sampled = []
    for req_id in req_ids:
        drafted = len(output.scheduled_spec_decode_tokens.get(req_id, ()))
        sampled.append(
            []
            if scheduler.requests[req_id].is_prefill_chunk
            else list(range(100, 101 + min(accept, drafted)))
        )
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
        ),
    )
    # MRv2 copies the width alone for requests without grammar validation.
    proposal_k = 0 if output.skip_drafts else output.num_spec_tokens_to_schedule
    scheduler.update_draft_token_ids(
        DraftTokenIds(req_ids, [[-1] * proposal_k for _ in req_ids])
    )
    return output


def test_geometric_trials_and_hysteresis():
    partial = AdaptiveMTPDepth(4)
    partial.observe(drafted=4, accepted=1, cap=4)
    assert (partial.successes, partial.trials) == pytest.approx((3.16, 4.16))

    near_boundary = AdaptiveMTPDepth(3, successes=7.5, trials=10)
    near_boundary.observe(drafted=4, accepted=3, cap=4)
    assert near_boundary.depth == 3
    near_boundary.observe(drafted=2, accepted=2, cap=4)
    assert near_boundary.depth == 4
    near_boundary.observe(drafted=4, accepted=4, cap=4)
    assert near_boundary.depth == 4


@pytest.mark.parametrize("cap", [3, 4])
def test_sync_feedback_changes_the_next_proposal(monkeypatch, build_scheduler, cap):
    monkeypatch.setenv("VLLM_1CAT_ADAPTIVE_MTP", "1")
    scheduler = build_scheduler(cap=cap)
    request = create_requests(1, max_tokens=64)[0]
    scheduler.add_request(request)
    first = _run_step(scheduler)
    assert first.num_spec_tokens_to_schedule == cap
    assert not first.scheduled_spec_decode_tokens
    assert not scheduler._adaptive_mtp_depths

    verify_widths, proposal_widths = [], []
    for _ in range(7):
        step = _run_step(scheduler)
        verify_widths.append(len(step.scheduled_spec_decode_tokens[request.request_id]))
        proposal_widths.append(step.num_spec_tokens_to_schedule)
    assert verify_widths == [cap, *proposal_widths[:-1]]
    assert proposal_widths[-1] == min(proposal_widths) == 2
    assert proposal_widths[0] == max(proposal_widths) == cap

    # Invalid padding does not train the estimator.
    before = vars(scheduler._adaptive_mtp_depths[request.request_id]).copy()
    _run_step(scheduler, invalid=True)
    assert vars(scheduler._adaptive_mtp_depths[request.request_id]) == before

    # A new prefill plus the existing decode row keeps the batch on its DSD cap.
    other = create_requests(1, num_tokens=40, max_tokens=64, req_ids=["other"])[0]
    scheduler.add_request(other)
    mixed = _run_step(scheduler)
    assert len(mixed.num_scheduled_tokens) == 2
    assert mixed.num_spec_tokens_to_schedule == 3
    scheduler.finish_requests(other.request_id, RequestStatus.FINISHED_ABORTED)

    _run_step(scheduler)  # Verify the C2 proposal and return to the learned K=2.
    _run_step(scheduler, accept=2)
    after = _run_step(scheduler)
    assert after.num_spec_tokens_to_schedule == 3
    assert len(after.scheduled_spec_decode_tokens[request.request_id]) == 2

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id not in scheduler._adaptive_mtp_depths
    restarted = create_requests(1, max_tokens=64, req_ids=[request.request_id])[0]
    scheduler.add_request(restarted)
    assert _run_step(scheduler).num_spec_tokens_to_schedule == cap


@pytest.mark.parametrize(
    "mode,k,cap,method,schedule,async_scheduling,sm70,enabled",
    [
        (None, 4, 4, "mtp", True, False, True, False),
        ("false", 4, 4, "mtp", True, False, True, False),
        (" YeS ", 4, 4, "mtp", True, False, False, True),
        ("auto", 4, 4, "mtp", True, False, True, False),
        ("auto", 5, 4, "mtp", True, False, False, False),
        ("auto", 5, 4, "mtp", True, False, True, True),
        ("on", 4, 4, "mtp", False, False, True, False),
        ("on", 4, 4, "mtp", True, True, True, False),
        ("on", 7, 7, "dflash", True, False, True, False),
    ],
)
def test_admission(
    monkeypatch,
    build_scheduler,
    mode,
    k,
    cap,
    method,
    schedule,
    async_scheduling,
    sm70,
    enabled,
):
    if mode is None:
        monkeypatch.delenv("VLLM_1CAT_ADAPTIVE_MTP", raising=False)
    else:
        monkeypatch.setenv("VLLM_1CAT_ADAPTIVE_MTP", mode)
    scheduler = build_scheduler(
        k=k,
        cap=cap,
        method=method,
        schedule=schedule,
        async_scheduling=async_scheduling,
    )
    monkeypatch.setattr(current_platform, "is_cuda", lambda: sm70)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: sm70
    )
    assert scheduler.vllm_config.adaptive_mtp_enabled is enabled


@pytest.mark.parametrize("cap", [0, 1, 2])
def test_small_dsd_caps_do_not_train_or_raise_depth(monkeypatch, build_scheduler, cap):
    monkeypatch.setenv("VLLM_1CAT_ADAPTIVE_MTP", "on")
    scheduler = build_scheduler(cap=cap)
    scheduler.add_request(create_requests(1, max_tokens=64)[0])
    for _ in range(3):
        output = _run_step(scheduler, accept=cap)
        assert output.num_spec_tokens_to_schedule == cap
    assert not scheduler._adaptive_mtp_depths


def test_feedback_excludes_tail_and_clears_on_finish(monkeypatch, build_scheduler):
    monkeypatch.setenv("VLLM_1CAT_ADAPTIVE_MTP", "on")
    scheduler = build_scheduler()
    request = create_requests(1, max_tokens=7)[0]
    scheduler.add_request(request)
    _run_step(scheduler)
    _run_step(scheduler)
    before = vars(scheduler._adaptive_mtp_depths[request.request_id]).copy()
    full_tail = _run_step(scheduler)
    assert len(full_tail.scheduled_spec_decode_tokens[request.request_id]) == 4
    assert vars(scheduler._adaptive_mtp_depths[request.request_id]) == before
    _run_step(scheduler, accept=4)
    assert request.is_finished()
    assert not scheduler._adaptive_mtp_depths


@pytest.mark.parametrize("enabled", [False, True])
def test_target_and_draft_first_step_share_narrow_c1_descriptors(
    monkeypatch, build_scheduler, enabled
):
    monkeypatch.setenv("VLLM_1CAT_ADAPTIVE_MTP", str(int(enabled)))
    config = build_scheduler().vllm_config
    mode = CUDAGraphMode.FULL_DECODE_ONLY
    config.compilation_config = CompilationConfig(
        cudagraph_mode=mode,
        cudagraph_capture_sizes=[5, 10, 20, 40],
        max_cudagraph_capture_size=40,
    )
    monkeypatch.setattr(
        graphs,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: None)
    target, draft = (
        cls(config, torch.device("cpu"), mode, decode_query_len=5)
        for cls in (graphs.ModelCudaGraphManager, PrefillEagleCudaGraphManager)
    )
    for manager in (target, draft):
        manager._graphs_captured = True
        assert manager.dispatch(1, 3, 3).cg_mode == (
            CUDAGraphMode.FULL if enabled else CUDAGraphMode.NONE
        )
        assert manager.dispatch(1, 3, None).cg_mode == CUDAGraphMode.NONE
        assert manager.dispatch(1, 4, 4).cg_mode == CUDAGraphMode.FULL
        assert manager.dispatch(1, 5, 5).cg_mode == CUDAGraphMode.FULL
        assert manager.dispatch(2, 6, 3).cg_mode == CUDAGraphMode.NONE
        assert manager.dispatch(2, 8, 4).cg_mode == CUDAGraphMode.FULL
        assert len(manager._capture_descs[CUDAGraphMode.FULL]) == len(
            set(manager._capture_descs[CUDAGraphMode.FULL])
        )
    assert target._capture_descs == draft._capture_descs
