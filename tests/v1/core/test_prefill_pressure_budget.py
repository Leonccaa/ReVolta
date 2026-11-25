# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import LoRAConfig
from vllm.lora.request import LoRARequest
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, create_scheduler, mock_kv

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


def test_rejected_short_prefill_leaves_full_budget_for_running_long_prefill():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
        enable_prefix_caching=True,
    )
    scheduler.lora_config = LoRAConfig(max_loras=1)
    resident_adapter = LoRARequest("resident", 1, "/unused")
    waiting_adapter = LoRARequest("waiting", 2, "/unused")
    (primer,) = create_requests(1, 65, max_tokens=1, req_ids=["primer"])
    (decode,) = create_requests(1, 8, req_ids=["decode"])
    (long,) = create_requests(1, 200, req_ids=["long"])
    (short,) = create_requests(1, 80, req_ids=["short"])
    for request, adapter in (
        (primer, waiting_adapter),
        (decode, resident_adapter),
        (long, resident_adapter),
        (short, waiting_adapter),
    ):
        request.lora_request = adapter
        request.block_hashes.clear()
        request.update_block_hashes()

    scheduler.add_request(primer)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"primer": [1]})
    scheduler.add_request(decode)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"decode": [1]})
    scheduler.add_request(long)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"decode": [2]})
    assert scheduler.kv_cache_manager.get_computed_blocks(short)[1] == 64
    scheduler.add_request(short)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"decode": 1, "long": 32}
    assert not output.scheduled_new_reqs


def test_short_prefill_does_not_admit_waiting_long_on_alignment_remainder():
    block = MAMBA_BLOCK_SIZE
    spec = MambaSpec(
        block_size=block,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    initial = create_scheduler(
        max_num_batched_tokens=3 * block,
        prefill_pressure_token_budget=2 * block,
        max_model_len=8 * block,
        enable_prefix_caching=True,
        block_size=block,
        kv_cache_spec=spec,
    )
    config = initial.vllm_config
    config.cache_config.mamba_cache_mode = "align"
    config.model_config.max_model_len = 8 * block
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=KVCacheConfig(64, [], [KVCacheGroupSpec(["mamba"], spec)]),
        structured_output_manager=StructuredOutputManager(config),
        block_size=block,
        hash_block_size=block,
    )
    (primer,) = create_requests(
        1, block + 1, max_tokens=1, block_size=block, req_ids=["primer"]
    )
    scheduler.add_request(primer)
    _complete_pressure_step(scheduler, scheduler.schedule(), {})
    _complete_pressure_step(scheduler, scheduler.schedule(), {"primer": [1]})
    _start_decode_request(scheduler)

    _, long, older = create_requests(
        3, 7 * block, block_size=block, req_ids=["unused", "long", "older"]
    )
    scheduler.add_request(long)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"decode": 1, "long": 2 * block}
    _complete_pressure_step(scheduler, first, {"decode": [2]})
    scheduler.add_request(older)
    (short,) = create_requests(1, block + 16, block_size=block, req_ids=["short"])
    scheduler.add_request(short)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"decode": 1, "long": block, "short": 16}
    assert [r.req_id for r in output.scheduled_new_reqs] == ["short"]
    assert output.scheduled_new_reqs[0].num_computed_tokens == block


def test_async_first_output_placeholders_leave_room_for_short_prefill():
    scheduler = create_scheduler(
        max_num_batched_tokens=33,
        prefill_pressure_token_budget=32,
        max_model_len=512,
        enable_prefix_caching=True,
        async_scheduling=True,
    )
    (primer,) = create_requests(1, 33, max_tokens=1, req_ids=["primer"])
    scheduler.add_request(primer)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"primer": [1]})
    _start_decode_request(scheduler)

    _, long = create_requests(2, 200, req_ids=["unused", "long"])
    scheduler.add_request(long)
    (pending,) = create_requests(1, 3, req_ids=["pending"])
    scheduler.add_request(pending)
    started = scheduler.schedule()
    assert started.num_scheduled_tokens == {"decode": 1, "long": 29, "pending": 3}
    assert pending.num_output_tokens == 0
    assert pending.num_output_placeholders == 1

    (short,) = create_requests(1, 48, req_ids=["short"])
    scheduler.add_request(short)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {
        "decode": 1,
        "pending": 1,
        "long": 15,
        "short": 16,
    }
    assert output.scheduled_new_reqs[0].num_computed_tokens == 32


def test_host_kv_short_prefill_uses_admitted_prefix():
    from .long_context_kv_utils import SPAN
    from .test_long_context_admission import build, make_requests, run, settings
    from .test_scheduler_kv_capacity import step

    scheduler = build(settings(main_kv_placement="host"))
    scheduler.scheduler_config.prefill_pressure_token_budget = SPAN
    assert scheduler.host_history_pool is not None
    assert scheduler.kv_cache_manager.hit_contract is not None

    (primer,) = make_requests(1, 2 * SPAN + 1, 1, same_prompt=True, tag="primer")
    run(scheduler, [primer])
    (decode,) = make_requests(1, 8, 8, tag="decode")
    scheduler.add_request(decode)
    step(scheduler)
    (long,) = make_requests(1, 7 * SPAN, 8, tag="long")
    scheduler.add_request(long)
    started = step(scheduler)
    assert started.num_scheduled_tokens == {decode.request_id: 1, long.request_id: SPAN}
    (short,) = make_requests(1, 2 * SPAN + 16, 8, same_prompt=True, tag="short")
    scheduler.add_request(short)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {
        decode.request_id: 1,
        long.request_id: SPAN - 16,
        short.request_id: 16,
    }
    assert [req.req_id for req in output.scheduled_new_reqs] == [short.request_id]
    assert output.scheduled_new_reqs[0].num_computed_tokens == 2 * SPAN


def test_finished_remote_short_prefill_uses_its_reserved_slot(monkeypatch):
    from .test_async_kv_admission import complete

    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(64, True),
    )
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda req, _: (64, True) if req.request_id == "short" else (0, False),
    )
    _start_decode_request(scheduler)
    _, long = create_requests(2, 200, req_ids=["unused", "long"])
    scheduler.add_request(long)
    complete(scheduler, scheduler.schedule())
    (short,) = create_requests(1, 80, req_ids=["short"])
    scheduler.add_request(short)
    load = scheduler.schedule()
    assert load.num_scheduled_tokens == {"decode": 1, "long": 32}
    assert short.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert short.num_computed_tokens == 64
    assert short in scheduler._inflight_prefills
    assert len(scheduler.running) + 1 == scheduler.max_num_running_reqs
    complete(scheduler, load, received={"short"})
    assert "short" in scheduler.finished_recving_kv_req_ids

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"decode": 1, "long": 16, "short": 16}
    assert [req.req_id for req in output.scheduled_new_reqs] == ["short"]
    assert output.scheduled_new_reqs[0].num_computed_tokens == 64


def test_short_prefill_kv_allocation_refusal_leaves_full_pressure_budget(
    monkeypatch,
):
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
        enable_prefix_caching=True,
        num_blocks=9,
    )
    (primer,) = create_requests(1, 65, max_tokens=1, req_ids=["primer"])
    scheduler.add_request(primer)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"primer": [1]})
    _start_decode_request(scheduler)
    _, long = create_requests(2, 88, req_ids=["unused", "long"])
    scheduler.add_request(long)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"decode": [2]})
    assert long.num_computed_tokens == 32
    (short,) = create_requests(1, 96, req_ids=["short"])
    manager = scheduler.kv_cache_manager
    assert manager.get_computed_blocks(short)[1] == 64
    scheduler.add_request(short)

    # Five free blocks can hold the long chunk, but the short request needs
    # four cached and two new blocks.
    allocate_slots = manager.allocate_slots
    refusals = []

    def observe_allocation(request, *args, **kwargs):
        allocated = allocate_slots(request, *args, **kwargs)
        if allocated is None:
            refusals.append(request.request_id)
        return allocated

    monkeypatch.setattr(manager, "allocate_slots", observe_allocation)
    output = scheduler.schedule()

    assert refusals == ["short"]
    assert not output.scheduled_new_reqs
    assert output.num_scheduled_tokens == {"decode": 1, "long": 32}


@pytest.mark.parametrize(
    ("connector_reply", "status"),
    [
        ((None, False), RequestStatus.WAITING),
        ((16, True), RequestStatus.WAITING_FOR_REMOTE_KVS),
    ],
)
def test_unready_connector_does_not_consume_pressure_budget(
    monkeypatch, connector_reply, status
):
    from .test_async_kv_admission import complete

    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
        use_kv_connector=mock_kv(16, True),
    )
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda req, _: connector_reply if req.request_id == "short" else (0, False),
    )
    _start_decode_request(scheduler)
    _, long = create_requests(2, 200, req_ids=["unused", "long"])
    scheduler.add_request(long)
    complete(scheduler, scheduler.schedule())
    (short,) = create_requests(1, 32, req_ids=["short"])
    scheduler.add_request(short)

    for _ in range(2):
        output = scheduler.schedule()
        assert short.status == status
        assert output.num_scheduled_tokens == {"decode": 1, "long": 32}
        complete(scheduler, output)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("fcfs", {"decode": 1, "long": 16, "short": 16}),
        ("priority", {"decode": 1, "long": 32}),
    ],
)
def test_cold_short_prefill_is_prioritized_only_under_fcfs(policy, expected):
    initial = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        max_model_len=512,
    )
    config = initial.vllm_config
    config.scheduler_config.policy = policy
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=initial.kv_cache_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=16,
    )
    _start_decode_request(scheduler)
    long, older = create_requests(2, 200, req_ids=["long", "older"])
    scheduler.add_request(long)
    _complete_pressure_step(scheduler, scheduler.schedule(), {"decode": [2]})
    (short,) = create_requests(1, 16, req_ids=["short"])
    short.priority = 1
    scheduler.add_request(older)
    scheduler.add_request(short)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == expected
    if "short" in expected:
        (admitted,) = output.scheduled_new_reqs
        assert admitted.req_id == "short"
        assert admitted.num_computed_tokens == 0
    else:
        assert not output.scheduled_new_reqs


def test_no_short_prefill_preserves_running_batches_across_steps():
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        prefill_pressure_token_budget=32,
        long_prefill_token_threshold=48,
        max_model_len=512,
    )
    (long,) = create_requests(1, 350, req_ids=["long"])
    (decode,) = create_requests(1, 8, req_ids=["decode"])
    scheduler.add_request(long)
    scheduler.add_request(decode)
    started = scheduler.schedule()
    assert started.num_scheduled_tokens == {"long": 48, "decode": 8}
    _complete_pressure_step(scheduler, started, {"decode": [1]})
    (older,) = create_requests(1, 200, req_ids=["older"])
    scheduler.add_request(older)

    for step in range(3):
        output = scheduler.schedule()

        assert output.num_scheduled_tokens == {"long": 32, "decode": 1}
        assert list(output.num_scheduled_tokens) == ["long", "decode"]
        assert not output.scheduled_new_reqs
        assert output.scheduled_cached_reqs.req_ids == ["long", "decode"]
        assert output.scheduled_cached_reqs.num_computed_tokens == [
            48 + 32 * step,
            8 + step,
        ]
        _complete_pressure_step(scheduler, output, {"decode": [2 + step]})
