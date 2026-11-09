# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler wiring of partial prefix-cache hits on the Qwen4Exp layout:
chunk stops at boundaries, block copies in the scheduler output, and the
KV-connector lookup from the block-aligned hit."""

from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.core.partial_prefix_trace import BLOCK, make_manager, make_request
from vllm.v1.kv_cache_interface import MambaSpec

from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    # The CPU test image has no accelerator platform; the scheduler never
    # touches a device.
    from vllm.platforms import current_platform

    if not current_platform.device_type:
        monkeypatch.setattr(current_platform, "device_type", "cpu")


PROMPT = [(7 * i + 3) % 1000 for i in range(700)]
P = 576


def _scheduler(connector=None):
    scheduler = create_scheduler(
        block_size=BLOCK,
        num_blocks=96,
        max_num_batched_tokens=1024,
        max_model_len=8192,
        enable_prefix_caching=True,
        kv_cache_spec=MambaSpec(
            block_size=BLOCK,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
    )
    scheduler.cache_config.mamba_cache_mode = "align"
    scheduler.need_mamba_block_aligned_split = True
    scheduler.mamba_state_block_size = BLOCK
    scheduler.mamba_state_retention_interval = 0
    scheduler.kv_cache_manager = make_manager(partial_prefix_cache=True)
    scheduler.connector = connector
    return scheduler


def _run_prefill(scheduler, request):
    """Schedule until the prompt is computed; return (chunk, copies) per step."""
    steps = []
    while request.num_computed_tokens < request.num_prompt_tokens:
        output = scheduler.schedule()
        steps.append(
            (
                output.num_scheduled_tokens.get(request.request_id, 0),
                output.kv_block_copies,
            )
        )
    return steps


def test_first_turn_stops_at_the_boundary_and_snapshots_it():
    scheduler = _scheduler()
    first = make_request("first", PROMPT, boundaries=[P])
    scheduler.add_request(first)
    steps = _run_prefill(scheduler, first)
    # 1024-token budget: 512 (block aligned), then the boundary at 576.
    assert [n for n, _ in steps] == [512, 64, 124]
    # The step after the boundary carries the snapshot copies (7 pages).
    assert steps[0][1] is None and steps[1][1] is None
    assert steps[2][1] is not None and len(steps[2][1]) == 7


def test_second_turn_resumes_at_the_boundary():
    scheduler = _scheduler()
    first = make_request("first", PROMPT, boundaries=[P])
    scheduler.add_request(first)
    _run_prefill(scheduler, first)
    scheduler.schedule()  # entry becomes ready
    second = make_request("second", PROMPT[:P] + [999] * 300, boundaries=[800])
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert second.num_computed_tokens == P + output.num_scheduled_tokens["second"]
    # Resumed at P; the chunk ends at the next block boundary (768).
    assert output.num_scheduled_tokens["second"] == 3 * BLOCK - P
    # Copy-on-write pages for the five copied groups.
    assert output.kv_block_copies is not None and len(output.kv_block_copies) == 5


def _connector(ext_tokens: int):
    connector = MagicMock()
    connector.get_num_new_matched_tokens.return_value = (ext_tokens, False)
    connector.build_connector_meta.return_value = MagicMock()
    connector.take_events.return_value = ()
    return connector


def test_connector_is_asked_from_the_aligned_hit_and_partial_hit_kept():
    connector = _connector(0)
    scheduler = _scheduler(connector)
    first = make_request("first", PROMPT, boundaries=[P])
    scheduler.add_request(first)
    _run_prefill(scheduler, first)
    scheduler.schedule()
    second = make_request("second", PROMPT[:P] + [999] * 300)
    scheduler.add_request(second)
    connector.get_num_new_matched_tokens.reset_mock()
    connector.update_state_after_alloc.reset_mock()
    scheduler.schedule()
    [call] = connector.get_num_new_matched_tokens.call_args_list
    aligned = call.args[1]
    assert aligned % BLOCK == 0 and aligned < P
    assert second.num_computed_tokens > P
    # No external tokens for a kept partial hit.
    assert connector.update_state_after_alloc.call_args.args[2] == 0


def test_longer_connector_hit_replaces_the_partial_hit():
    scheduler = _scheduler(_connector(0))
    first = make_request("first", PROMPT, boundaries=[P])
    scheduler.add_request(first)
    _run_prefill(scheduler, first)
    scheduler.schedule()
    # The host tier reaches 768 (> P) from the aligned hit.
    second = make_request("second", PROMPT[:P] + [999] * 300)
    aligned_holder = {}

    def matched(request, num_computed):
        aligned_holder["aligned"] = num_computed
        return 3 * BLOCK - num_computed, False

    scheduler.connector.get_num_new_matched_tokens.side_effect = matched
    scheduler.add_request(second)
    scheduler.schedule()
    args = scheduler.connector.update_state_after_alloc.call_args.args
    assert args[2] == 3 * BLOCK - aligned_holder["aligned"]
    assert second.partial_prefix_entry is None
    assert second.num_computed_tokens >= 3 * BLOCK


def test_switched_off_scheduler_ignores_boundaries():
    scheduler = _scheduler()
    scheduler.kv_cache_manager = make_manager()
    first = make_request("first", PROMPT, boundaries=[P])
    scheduler.add_request(first)
    assert [n for n, _ in _run_prefill(scheduler, first)] == [512, 188]
    scheduler.schedule()
    second = make_request("second", PROMPT[:P] + [999] * 300)
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert output.kv_block_copies is None
    assert second.partial_prefix_fallback is None
    assert (second.num_computed_tokens - output.num_scheduled_tokens["second"]) % (
        BLOCK
    ) == 0


@pytest.mark.parametrize("override,hold", [(0, 3), (4, 5)])
def test_copy_hold_follows_async_queue_depth(monkeypatch, override, hold):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_SM70_ASYNC_SCHEDULING_QUEUE_DEPTH", override)
    # The manager built by Scheduler.__init__ (not the test replacement).
    scheduler = create_scheduler(enable_prefix_caching=True)
    assert scheduler.kv_cache_manager._copy_hold_steps == hold
