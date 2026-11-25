# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _start_decode_request(scheduler) -> None:
    request = create_requests(
        num_requests=1,
        num_tokens=8,
        max_tokens=8,
        req_ids=["decode"],
    )[0]
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["decode"],
            req_id_to_index={"decode": 0},
            sampled_token_ids=[[1]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_prefill_budget_is_inactive_without_decode():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
    )
    request = create_requests(
        num_requests=1,
        num_tokens=200,
        max_tokens=4,
        req_ids=["prefill"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"prefill": 96}


def test_prefill_budget_caps_mixed_step_and_preserves_decode():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
    )
    _start_decode_request(scheduler)
    requests = create_requests(
        num_requests=2,
        num_tokens=200,
        max_tokens=4,
        req_ids=["prefill-0", "prefill-1"],
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["decode"] == 1
    assert output.num_scheduled_tokens["prefill-0"] == 32
    assert "prefill-1" not in output.num_scheduled_tokens
    assert sum(output.num_scheduled_tokens.values()) == 33

    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["decode", "prefill-0"],
            req_id_to_index={"decode": 0, "prefill-0": 1},
            sampled_token_ids=[[2], []],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    next_output = scheduler.schedule()

    assert next_output.num_scheduled_tokens["decode"] == 1
    assert next_output.num_scheduled_tokens["prefill-0"] == 32
    assert "prefill-1" not in next_output.num_scheduled_tokens


def test_prefill_budget_is_aggregate_across_requests():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
    )
    _start_decode_request(scheduler)
    requests = create_requests(
        num_requests=2,
        num_tokens=20,
        max_tokens=4,
        req_ids=["prefill-0", "prefill-1"],
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["decode"] == 1
    assert output.num_scheduled_tokens["prefill-0"] == 20
    assert output.num_scheduled_tokens["prefill-1"] == 12
    assert (
        sum(
            tokens
            for request_id, tokens in output.num_scheduled_tokens.items()
            if request_id != "decode"
        )
        == 32
    )


def test_prefill_pressure_threshold_keeps_low_pressure_path_unchanged():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        prefill_pressure_threshold=2,
        max_model_len=512,
    )
    _start_decode_request(scheduler)
    request = create_requests(
        num_requests=1,
        num_tokens=200,
        max_tokens=4,
        req_ids=["prefill"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["decode"] == 1
    assert output.num_scheduled_tokens["prefill"] == 95


def test_prefill_pressure_budget_requires_chunked_prefill():
    with pytest.raises(ValueError, match="Chunked prefill must be enabled"):
        create_scheduler(
            max_num_batched_tokens=512,
            enable_chunked_prefill=False,
            prefill_pressure_token_budget=32,
            max_model_len=512,
        )


MAMBA_BLOCK_SIZE = 816


def _align_pressure_scheduler(budget: int, *, align: bool = True):
    decode = SimpleNamespace(num_computed_tokens=9, num_prompt_tokens=8)
    prefills = [
        SimpleNamespace(
            num_computed_tokens=0,
            num_prompt_tokens=8 * MAMBA_BLOCK_SIZE,
            num_tokens=8 * MAMBA_BLOCK_SIZE,
        )
        for _ in range(2)
    ]
    coordinator = SimpleNamespace(eagle_group_ids=set())
    coordinator.get_replay_boundaries = lambda request, block: (
        KVCacheCoordinator.get_replay_boundaries(coordinator, request, block)
    )
    scheduler = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            prefill_pressure_token_budget=budget,
            prefill_pressure_threshold=1,
            long_prefill_token_threshold=0,
        ),
        running=[decode, *prefills],
        waiting=[],
        skipped_waiting=[],
        max_num_scheduled_tokens=8192,
        need_mamba_block_aligned_split=align,
        mamba_state_block_size=MAMBA_BLOCK_SIZE if align else None,
        cache_config=SimpleNamespace(block_size=16),
        use_eagle=False,
        mamba_state_retention_interval=0 if align else None,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
    )
    return scheduler, prefills


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(768, 768), (816, 816), (1000, 816), (1700, 1632), (3300, 3264)],
)
def test_align_pressure_budget_keeps_whole_state_blocks(budget, expected):
    scheduler, _ = _align_pressure_scheduler(budget)

    assert Scheduler._get_prefill_pressure_token_budget(scheduler) == expected


def test_pressure_budget_is_unchanged_without_align_split():
    scheduler, _ = _align_pressure_scheduler(1700, align=False)

    assert Scheduler._get_prefill_pressure_token_budget(scheduler) == 1700


def test_align_pressure_budget_leaves_no_short_remainder_chunk():
    scheduler, (first, second) = _align_pressure_scheduler(1700)
    budget = Scheduler._get_prefill_pressure_token_budget(scheduler)

    # The scheduler caps each prefill by the remaining budget before the
    # align split, then charges what the split actually scheduled.
    first_chunk = Scheduler._mamba_block_aligned_split(scheduler, first, budget)
    budget -= first_chunk

    assert first_chunk == 2 * MAMBA_BLOCK_SIZE
    assert budget == 0
    # A raw 1700-token budget would hand the second prefill a 68-token chunk.
    raw_remainder = 1700 - first_chunk
    assert (
        Scheduler._mamba_block_aligned_split(scheduler, second, raw_remainder)
        == raw_remainder
    )


@pytest.mark.parametrize(
    ("with_decode", "num_prefills", "expected"),
    [
        # Alone: neither the threshold nor the pressure budget applies.
        (False, 1, {"prefill-0": 96}),
        # Prefills only: the threshold splits the budget, no pressure cap.
        (False, 2, {"prefill-0": 40, "prefill-1": 40}),
        # With a decode the tighter aggregate pressure budget wins.
        (True, 2, {"decode": 1, "prefill-0": 32}),
    ],
)
def test_long_prefill_threshold_combines_with_pressure_budget(
    with_decode, num_prefills, expected
):
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        long_prefill_token_threshold=40,
        prefill_pressure_token_budget=32,
        max_model_len=512,
    )
    if with_decode:
        _start_decode_request(scheduler)
    for request in create_requests(
        num_requests=num_prefills,
        num_tokens=200,
        max_tokens=4,
        req_ids=[f"prefill-{i}" for i in range(num_prefills)],
    ):
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == expected


def _complete_pressure_step(scheduler, output, sampled):
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[sampled.get(req_id, []) for req_id in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


@pytest.mark.parametrize("short_ids", [(), ("first",), ("first", "second")])
def test_cached_short_prefill_gets_pressure_budget_before_running_long_prefill(
    short_ids,
):
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
        enable_prefix_caching=True,
    )
    primer = create_requests(1, 65, max_tokens=1, req_ids=["primer"])[0]
    scheduler.add_request(primer)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"primer": [1]})
    _start_decode_request(scheduler)

    long, older_waiting = create_requests(2, 200, req_ids=["long", "older"])
    scheduler.add_request(long)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"decode": [2]})
    scheduler.add_request(older_waiting)
    for name in short_ids:
        scheduler.add_request(create_requests(1, 80, req_ids=[name])[0])

    output = scheduler.schedule()

    expected = {"decode": 1, **dict.fromkeys(short_ids, 16)}
    if len(short_ids) < 2:
        expected["long"] = 32 - 16 * len(short_ids)
    assert output.num_scheduled_tokens == expected
    assert [r.req_id for r in output.scheduled_new_reqs] == list(short_ids)
    assert all(r.num_computed_tokens == 64 for r in output.scheduled_new_reqs)

    _complete_pressure_step(
        scheduler, output, {"decode": [3], **dict.fromkeys(short_ids, [1])}
    )
    next_output = scheduler.schedule()
    assert next_output.num_scheduled_tokens["long"] == 32
    assert "older" not in next_output.num_scheduled_tokens
