# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic SD scheduling regressions adapted from vLLM #32374."""

from types import SimpleNamespace

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager


def _make_lookup(
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]],
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> list[int]:
    return build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )


def _make_scheduler_with_dynamic_sd(
    schedule: list[tuple[int, int, int]],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 3,
    async_scheduling: bool = False,
    method="mtp",
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
    )

    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.method = method
    speculative_config.num_speculative_tokens_per_batch_size = schedule

    scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    return scheduler_cls(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _add_requests_and_schedule(
    scheduler: Scheduler, num_requests: int, *, num_tokens: int = 10
):
    requests = create_requests(num_requests=num_requests, num_tokens=num_tokens)
    for request in requests:
        scheduler.add_request(request)
    return scheduler.schedule()


def test_mtp_zero_drafts_preserves_prefill_forward(monkeypatch):
    from unittest.mock import Mock

    import numpy as np
    import torch

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle

    monkeypatch.setenv("VLLM_SM70_MTP_KV_ONLY_PREFILL", "0")
    spec = object.__new__(eagle.EagleSpeculator)
    spec.method = "mtp"
    spec.num_speculative_steps = 2
    spec.max_num_reqs = 1
    spec.dp_size, spec.dp_rank = 1, 0
    spec.temperature = torch.ones(1)
    spec.seeds = torch.zeros(1, dtype=torch.int64)
    spec.idx_mapping = torch.zeros(1, dtype=torch.int32)
    spec.last_token_indices = torch.tensor([3])
    spec.current_draft_step = torch.tensor(0)
    spec.input_buffers = InputBuffers(1, 4, torch.device("cpu"))
    spec.share_mtp_topk_indices = False
    spec.draft_logits = None
    spec.draft_tokens = torch.zeros(1, 2, dtype=torch.int64)
    spec.hidden_states = torch.zeros(4, 3)
    target_hidden = torch.arange(12, dtype=torch.float32).view(4, 3)
    draft_hidden = target_hidden + 1
    spec.run_model = Mock(return_value=(draft_hidden, draft_hidden))
    spec._sample_draft = Mock(return_value=torch.tensor([17]))
    spec.multi_step_decode = Mock()
    spec.prefill_cudagraph_manager = Mock()
    spec.prefill_cudagraph_manager.dispatch.return_value = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=4, num_reqs=1
    )
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=4,
        num_tokens_after_padding=4,
        num_scheduled_tokens=np.array([4]),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        positions=torch.arange(4),
        is_prefilling_np=np.array([True]),
    )
    num_sampled = torch.ones(1, dtype=torch.int32)
    num_rejected = torch.zeros(1, dtype=torch.int32)
    last_sampled = torch.tensor([13])
    next_prefill_tokens = torch.tensor([0])

    def propose(k):
        return spec.propose(
            batch,
            {},
            {},
            target_hidden,
            None,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            spec.temperature,
            spec.seeds,
            num_speculative_tokens=k,
        )

    propose(0)
    spec.run_model.assert_called_once()
    zero_draft_forward = spec.run_model.call_args
    assert zero_draft_forward.args[0] == 4
    spec._sample_draft.assert_not_called()
    spec.multi_step_decode.assert_not_called()

    spec.run_model.reset_mock()
    assert propose(1)[0, 0] == 17
    spec.run_model.assert_called_once_with(
        *zero_draft_forward.args, **zero_draft_forward.kwargs
    )
    spec._sample_draft.assert_called_once()


def test_dynamic_sd_uses_batch_size_schedule():
    dynamic_sd_lookup = _make_lookup(
        [
            (1, 16, 3),
            (32, 128, 2),
            (256, 2048, 0),
        ]
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[16] == 3
    assert dynamic_sd_lookup[17] == 3
    assert dynamic_sd_lookup[31] == 3
    assert dynamic_sd_lookup[32] == 2
    assert dynamic_sd_lookup[128] == 2
    assert dynamic_sd_lookup[129] == 2
    assert dynamic_sd_lookup[255] == 2
    assert dynamic_sd_lookup[256] == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_lookup([(2, 16, 3)])


def test_dynamic_sd_clamps_k_to_runtime_max():
    dynamic_sd_lookup = _make_lookup(
        [(1, 256, 4)],
        runtime_num_speculative_tokens=3,
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[256] == 3


def test_dynamic_sd_rejects_invalid_schedule_entry():
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16, 3), (32, 64)])  # type: ignore[list-item]


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_lookup([(1, 16, 3), (16, 32, 2)])


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_lookup([(1, 16, -1)])


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_lookup([])


def test_scheduler_initializes_dynamic_sd_lookup_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_lookup is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 256, 5)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_falls_back_to_static_k_when_dsd_not_configured():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=40,
        num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 4)

    assert scheduler.dynamic_sd_lookup is None
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd([(2, 16, 3)])


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler.dynamic_sd_lookup is not None
    assert len(scheduler.dynamic_sd_lookup) == 17
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_scheduled_count_after_budget_truncates_verification():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 3, 4), (4, 4, 3)], runtime_num_speculative_tokens=4
    )
    first = _add_requests_and_schedule(scheduler, 4)
    assert first.num_spec_tokens_to_schedule == 3
    req_ids = list(first.num_scheduled_tokens)
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[[100]] * 4,
        ),
    )
    scheduler.update_draft_token_ids(DraftTokenIds(req_ids, [[101, 102, 103]] * 4))
    scheduler.max_num_scheduled_tokens = 3

    next_output = scheduler.schedule()

    assert len(scheduler.running) == 4
    assert next_output.num_scheduled_tokens == {req_ids[0]: 3}
    assert next_output.scheduled_spec_decode_tokens == {req_ids[0]: [101, 102]}
    assert next_output.num_spec_tokens_to_schedule == 4


@pytest.mark.parametrize("method", ["mtp", "dflash"])
def test_async_scheduler_reserves_new_width_and_counts_previous_width(method):
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 1, 4), (2, 2, 3), (3, 3, 0)],
        runtime_num_speculative_tokens=4,
        async_scheduling=True,
        method=method,
    )
    reqs = create_requests(3)
    scheduler.add_request(reqs[0])
    assert scheduler.schedule().num_spec_tokens_to_schedule == 4
    assert reqs[0].spec_token_ids == [-1] * 4

    scheduler.add_request(reqs[1])
    second = scheduler.schedule()
    assert second.num_spec_tokens_to_schedule == 3
    assert second.scheduled_spec_decode_tokens == {reqs[0].request_id: [-1] * 4}
    assert reqs[0].num_output_placeholders == 6
    assert reqs[1].num_output_placeholders == 1
    assert reqs[0].spec_token_ids == reqs[1].spec_token_ids == [-1] * 3
    assert reqs[0].spec_token_ids is not reqs[1].spec_token_ids

    scheduler.add_request(reqs[2])
    third = scheduler.schedule()
    assert third.num_spec_tokens_to_schedule == 0
    assert third.scheduled_spec_decode_tokens == {
        req.request_id: [-1] * 3 for req in reqs[:2]
    }
    assert all(req.spec_token_ids == [] for req in reqs)

    scheduler.finish_requests(
        [req.request_id for req in reqs[1:]], RequestStatus.FINISHED_ABORTED
    )
    fourth = scheduler.schedule()
    assert fourth.num_scheduled_tokens == {reqs[0].request_id: 1}
    assert fourth.scheduled_spec_decode_tokens == {}
    assert fourth.num_spec_tokens_to_schedule == 4
    assert reqs[0].spec_token_ids == [-1] * 4
