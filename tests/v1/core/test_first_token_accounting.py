# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confirm first-token metrics only after ordinary output filtering."""

import pytest

from tests.v1.core.test_async_scheduler import _make_model_runner_output
from tests.v1.core.utils import create_requests, create_scheduler


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_first_token_accounting_after_reset(async_scheduling, monkeypatch):
    scheduler = create_scheduler(
        async_scheduling=async_scheduling, enable_prefix_caching=True
    )
    (request,) = create_requests(num_requests=1, num_tokens=64, max_tokens=3)

    assert scheduler.vllm_config.observability_config.enable_store_metrics
    metrics = scheduler.store_metrics
    assert metrics is not None
    metrics.next_export = float("inf")
    now = 100.0
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.monotonic", lambda: now)
    scheduler.add_request(request)
    output = scheduler.schedule()
    if async_scheduling:
        assert request.num_output_placeholders == 1
        assert scheduler.reset_prefix_cache(reset_running_requests=True)
        assert request.async_tokens_to_discard == 1
        now = 110.0
        scheduler.update_from_output(output, _make_model_runner_output(output))
        assert request.async_tokens_to_discard == 0
        assert request.num_output_tokens == 0
        assert not request.store_attempt.first_token
        assert not metrics.observations
        assert not any(
            count
            for name, count in metrics.counters.items()
            if name.startswith("prompt_tokens_")
        )
        output = scheduler.schedule()

    now = 120.0
    scheduler.update_from_output(output, _make_model_runner_output(output))
    assert request.num_output_tokens == 1
    assert request.store_attempt.first_token
    assert [item for item in metrics.observations if item[0] == "ttft"] == [
        ("ttft", "compute", 20.0)
    ]
    assert (
        sum(
            count
            for name, count in metrics.counters.items()
            if name.startswith("prompt_tokens_")
        )
        == request.num_prompt_tokens
    )
    first_counters = metrics.counters.copy()
    first_observations = list(metrics.observations)
    while scheduler.has_unfinished_requests():
        now += 10.0
        output = scheduler.schedule()
        scheduler.update_from_output(output, _make_model_runner_output(output))
    assert request.num_output_tokens == 3
    assert metrics.counters == first_counters
    assert list(metrics.observations) == first_observations


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_first_token_accounting_precedes_streaming_turn_end(
    async_scheduling, monkeypatch
):
    scheduler = create_scheduler(
        async_scheduling=async_scheduling, enable_prefix_caching=True
    )
    (request,) = create_requests(num_requests=1, num_tokens=64, max_tokens=1)
    request.resumable = True
    metrics = scheduler.store_metrics
    assert metrics is not None
    metrics.next_export = float("inf")
    end_turn = scheduler._end_streaming_turn
    ended = []

    def checked_end_turn(request, status_before_stop):
        assert request.store_attempt.first_token
        assert metrics.counters["prompt_tokens_compute"] == request.num_prompt_tokens
        assert sum(phase == "ttft" for phase, _, _ in metrics.observations) == 1
        ended.append(request.request_id)
        end_turn(request, status_before_stop)

    monkeypatch.setattr(scheduler, "_end_streaming_turn", checked_end_turn)
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(output, _make_model_runner_output(output))
    assert ended == [request.request_id]
    assert request.num_output_tokens == 1
