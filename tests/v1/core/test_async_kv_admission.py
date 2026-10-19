# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async loads must leave room for admitted prefills to make progress."""

import time

import pytest
import torch

from tests.v1.kv_connector.unit.utils import MockKVConnector
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test


class HybridMockConnector(MockKVConnector, SupportsHMA):
    def request_finished_all_groups(self, request, block_ids):
        return False, None


def make_scheduler(
    *,
    blocks,
    spans=(16,),
    spec=0,
    mamba=None,
    fixed_states=False,
    token_budget=256,
    **kwargs,
):
    kwargs.setdefault("max_model_len", 256)
    original = create_scheduler(
        block_size=16,
        num_blocks=blocks,
        max_num_batched_tokens=token_budget,
        max_num_seqs=4,
        use_kv_connector=mock_kv(64, True),
        num_speculative_tokens=spec or None,
        **kwargs,
    )
    config = original.vllm_config
    if spec:
        config.speculative_config.method = "mtp"
    groups = [
        KVCacheGroupSpec(
            [f"attention{i}"],
            FullAttentionSpec(
                block_size=span, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        )
        for i, span in enumerate(spans)
    ]
    if fixed_states:
        groups.extend(
            [
                KVCacheGroupSpec(
                    ["ring"],
                    CircularBufferSpec(
                        block_size=4,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
                KVCacheGroupSpec(
                    ["tail"],
                    KpoolTailSpec(
                        block_size=16,
                        sliding_window=4,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
            ]
        )
    if mamba:
        mode, span = mamba
        config.cache_config.mamba_cache_mode = mode
        groups.append(
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=span,
                    shapes=((16,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode=mode,
                    num_speculative_blocks=spec,
                ),
            )
        )
    cache = KVCacheConfig(
        num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=cache,
        structured_output_manager=StructuredOutputManager(config),
        block_size=16,
        hash_block_size=16,
    )
    scheduler.connector = HybridMockConnector(config, KVConnectorRole.SCHEDULER, cache)
    return scheduler


def request(scheduler, name, length, max_tokens=1):
    (req,) = create_requests(
        1, num_tokens=length, req_ids=[name], max_tokens=max_tokens, ignore_eos=True
    )
    scheduler.add_request(req)
    return req


def complete(scheduler, output, *, received=(), invalid=()):
    received = {
        rid
        for rid in received
        if rid in scheduler.requests
        and (
            scheduler.requests[rid].status == RequestStatus.WAITING_FOR_REMOTE_KVS
            or scheduler.requests[rid].is_finished()
        )
    }
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[
                [1]
                if scheduler.requests[rid].num_computed_tokens
                >= scheduler.requests[rid].num_prompt_tokens
                else []
                for rid in ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=KVConnectorOutput(
                finished_recving=set(received), invalid_block_ids=set(invalid)
            ),
        ),
    )


@pytest.mark.parametrize("full_isl", [False, True])
@pytest.mark.parametrize("spans,blocks", [((16,), 10), ((16, 32), 15)])
def test_async_loads_leave_room_for_first_prefill(full_isl, spans, blocks):
    scheduler = make_scheduler(blocks=blocks, spans=spans)
    scheduler.scheduler_reserve_full_isl = full_isl
    a = request(scheduler, "a", 128)
    b = request(scheduler, "b", 80)
    output = scheduler.schedule()
    complete(scheduler, output, received={"a", "b"})
    steps = []
    for _ in range(5):
        output = scheduler.schedule()
        steps.append(dict(output.num_scheduled_tokens))
        complete(scheduler, output)
        if a.is_finished():
            break
    assert any("a" in step for step in steps), steps
    assert a.is_finished()
    assert b.num_preemptions == 0


@pytest.mark.parametrize("spans,blocks", [((16,), 8), ((16, 16), 16), ((32, 16), 13)])
def test_full_hit_mtp_load_keeps_promotion_margin(spans, blocks):
    scheduler = make_scheduler(blocks=blocks, spans=spans, spec=4)
    scheduler.connector.get_num_new_matched_tokens = lambda req, _: (
        req.num_tokens,
        True,
    )
    a = request(scheduler, "a", 64)
    b = request(scheduler, "b", 48)
    output = scheduler.schedule()
    assert a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert b.status == RequestStatus.WAITING
    assert not any(scheduler.kv_cache_manager.get_block_ids("b"))
    complete(scheduler, output, received={"a"})
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["a"] > 0
    complete(scheduler, output)
    assert a.is_finished()


@pytest.mark.parametrize("mode,span,blocks", [("align", 16, 18), ("none", 128, 18)])
def test_mamba_speculative_states_are_reserved_once(mode, span, blocks):
    scheduler = make_scheduler(blocks=blocks, spec=3, mamba=(mode, span))
    scheduler.connector.get_num_new_matched_tokens = lambda req, _: (
        req.num_tokens,
        True,
    )
    a = request(scheduler, "a", 64)
    b = request(scheduler, "b", 48)
    output = scheduler.schedule()
    assert a.status == b.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    complete(scheduler, output, received={"a", "b"})
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {"a", "b"}
    complete(scheduler, output)


def test_fixed_ring_and_tail_do_not_reserve_decode_growth():
    scheduler = make_scheduler(blocks=14, spec=4, fixed_states=True)
    scheduler.connector.get_num_new_matched_tokens = lambda req, _: (
        req.num_tokens,
        True,
    )
    a = request(scheduler, "a", 64)
    b = request(scheduler, "b", 48)
    output = scheduler.schedule()
    assert a.status == b.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    complete(scheduler, output, received={"a", "b"})
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {"a", "b"}
    complete(scheduler, output)


def test_async_load_does_not_displace_chunked_prefill():
    scheduler = make_scheduler(blocks=10, long_prefill_token_threshold=32)
    scheduler.connector.get_num_new_matched_tokens = lambda req, _: (
        (0, False) if req.request_id == "a" else (64, True)
    )
    a = request(scheduler, "a", 128)
    b = request(scheduler, "b", 80)
    for _ in range(4):
        output = scheduler.schedule()
        assert b.status == RequestStatus.WAITING
        complete(scheduler, output)
    assert a.is_finished() and a.num_preemptions == 0
    assert scheduler.schedule().kv_connector_metadata is not None
    assert b.status == RequestStatus.WAITING_FOR_REMOTE_KVS


@pytest.mark.parametrize("action", ["cancel", "fail", "recompute", "preempt"])
@pytest.mark.parametrize("hybrid", [False, True])
def test_reservation_released_on_lifecycle_changes(action, hybrid):
    scheduler = make_scheduler(
        blocks=16 if hybrid else 10,
        mamba=("align", 32) if hybrid else None,
        token_budget=32,
        long_prefill_token_threshold=32,
    )
    a = request(scheduler, "a", 128)
    output = scheduler.schedule()
    if action == "cancel":
        scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
        # Transfer destinations are retained until the worker acknowledges.
        complete(scheduler, output, received={"a"})
    elif action in ("fail", "recompute"):
        scheduler.recompute_kv_load_failures = action == "recompute"
        invalid = {scheduler.kv_cache_manager.get_block_ids("a")[0][0]}
        complete(scheduler, output, received={"a"}, invalid=invalid)
        if action == "recompute":
            scheduler.connector.get_num_new_matched_tokens = lambda req, _: (0, False)
            output = scheduler.schedule()
            assert "a" in output.num_scheduled_tokens
            complete(scheduler, output)
            scheduler.running.remove(a)
            scheduler._preempt_request(a, time.monotonic())
    else:
        complete(scheduler, output, received={"a"})
        output = scheduler.schedule()
        complete(scheduler, output)
        scheduler.running.remove(a)
        scheduler._preempt_request(a, time.monotonic())
    assert a not in scheduler._inflight_prefills
    assert scheduler._inflight_prefill_reserved_blocks() == 0
    if action in ("preempt", "recompute"):
        scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    if action == "recompute":
        scheduler.connector.get_num_new_matched_tokens = lambda req, _: (64, True)
    b = request(scheduler, "b", 80)
    scheduler.schedule()
    assert b.status == RequestStatus.WAITING_FOR_REMOTE_KVS


@pytest.mark.parametrize("failed_group", [0, 1])
@pytest.mark.parametrize("with_running_peer", [False, True])
def test_hybrid_failed_load_recomputes_all_state_groups(
    failed_group, with_running_peer
):
    scheduler = make_scheduler(
        blocks=30, mamba=("align", 32), token_budget=64 if with_running_peer else 32
    )
    scheduler.recompute_kv_load_failures = True
    scheduler.connector.get_num_new_matched_tokens = lambda req, _: (
        (64, True) if req.request_id == "a" else (0, False)
    )
    if with_running_peer:
        request(scheduler, "peer", 16, max_tokens=16)
        complete(scheduler, scheduler.schedule())
        assert scheduler.running
    a = request(scheduler, "a", 128)
    output = scheduler.schedule()
    null = scheduler.kv_cache_manager.block_pool.null_block.block_id
    block_ids = scheduler.kv_cache_manager.get_block_ids("a")[failed_group]
    invalid = {next(bid for bid in block_ids if bid != null)}
    complete(scheduler, output, received={"a"}, invalid=invalid)
    assert a.num_computed_tokens == 0
    scheduler.connector.get_num_new_matched_tokens = lambda req, _: (0, False)
    for _ in range(5):
        output = scheduler.schedule()
        complete(scheduler, output)
        if a.is_finished():
            break
    assert a.is_finished()
    assert not scheduler._inflight_prefills


def test_remaining_mtp_requirement_includes_lookahead():
    scheduler = make_scheduler(blocks=30, spans=(16, 32), spec=4)
    a = request(scheduler, "a", 64)
    scheduler.schedule()
    # The full-hit load already owns the prompt; only promotion remains.
    assert scheduler._request_remaining_blocks(a) == 2
