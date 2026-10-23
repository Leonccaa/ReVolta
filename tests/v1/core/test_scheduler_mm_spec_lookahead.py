# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler progress with EAGLE/MTP lookahead and multimodal items.

With EAGLE/MTP the drafter reads the token one position ahead, so the
scheduler treats an encoder input as needed one token early
(``shift_computed_tokens=1``). When an item cannot be scheduled the request
is rolled back to stop before it. If the request already stands one token
before that item, the rollback used to produce zero tokens, and the same
state repeated on every step:

* two adjacent items that do not fit in the encoder cache together: the
  first one stays referenced until its last token is computed, so the
  second one can never be allocated;
* ``disable_chunked_mm_input`` with a token budget equal to the item size:
  the budget can never cover one extra token plus the whole item.

The request must instead compute the remaining token before the item (the
drafter's lookahead embedding for that one position falls back to the
placeholder token), so it can then make progress. This is the same class of
problem as lmdeploy#5061.
"""

import pytest

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

ITEM = 8


def make_scheduler(
    use_eagle: bool,
    cache_size: int = ITEM,
    max_num_batched_tokens: int = 64,
    disable_chunked_mm_input: bool = False,
) -> Scheduler:
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=128,
        disable_chunked_mm_input=disable_chunked_mm_input,
    )
    # The OPT config has no encoder; give the scheduler a small encoder
    # budget and cache so it schedules the image items itself.
    scheduler.max_num_encoder_input_tokens = ITEM
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=cache_size)
    # Only the lookahead shift matters here (EAGLE / EAGLE3 / MTP).
    scheduler.use_eagle = use_eagle
    return scheduler


def make_request(req_id, num_tokens, items):
    (request,) = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        mm_hashes_list=[[name for name, _, _ in items]],
        mm_positions=[[PlaceholderRange(o, n) for _, o, n in items]],
        max_tokens=1,
        req_ids=[req_id],
    )
    return request


def step(scheduler: Scheduler, trace: list) -> None:
    output = scheduler.schedule()
    trace.append(
        (
            dict(output.num_scheduled_tokens),
            {r: list(ids) for r, ids in output.scheduled_encoder_inputs.items()},
        )
    )
    ids = list(output.num_scheduled_tokens)
    sampled = []
    for r in ids:
        # schedule() has already advanced num_computed_tokens.
        req = scheduler.requests[r]
        sampled.append([1] if req.num_computed_tokens >= req.num_tokens else [])
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def run_prefill(scheduler, request, limit=10):
    """Step until the prompt is computed; every step must make progress."""
    scheduler.add_request(request)
    trace: list = []
    history = [request.num_computed_tokens]
    for _ in range(limit):
        if request.is_finished():
            break
        step(scheduler, trace)
        if request.is_finished():
            break
        history.append(request.num_computed_tokens)
        assert history[-1] > history[-2], (
            f"no progress: num_computed_tokens history={history}, trace={trace}"
        )
    assert request.is_finished(), f"history={history}, trace={trace}"
    manager = scheduler.encoder_cache_manager
    assert all(not refs for refs in manager.cached.values())
    return history, trace


@pytest.mark.parametrize("use_eagle", [False, True])
def test_adjacent_items_larger_than_cache_make_progress(use_eagle):
    # X = [8, 16) and Y = [16, 24) touch; the cache holds one of them.
    scheduler = make_scheduler(use_eagle)
    request = make_request("r", 32, [("imgX", 8, ITEM), ("imgY", 16, ITEM)])
    history, trace = run_prefill(scheduler, request)
    if use_eagle:
        # Stop one token early so the drafter does not reach Y, then compute
        # the last token of X alone, then encode Y once X is freed.
        assert history[:3] == [0, 15, 16]
        assert trace[0][1] == {"r": [0]}
        assert trace[1] == ({"r": 1}, {})
        assert trace[2][1] == {"r": [1]}
    else:
        assert history[:2] == [0, 16]
        assert trace[1][1] == {"r": [1]}


@pytest.mark.parametrize("use_eagle", [False, True])
def test_step_ending_one_token_before_adjacent_item(use_eagle):
    # The first chunk ends at X_end - 1 because of the token budget.
    scheduler = make_scheduler(
        use_eagle, cache_size=ITEM + ITEM // 2, max_num_batched_tokens=15
    )
    request = make_request("r", 32, [("imgX", 8, ITEM), ("imgY", 16, ITEM)])
    history, _ = run_prefill(scheduler, request)
    assert history[1] == 15


@pytest.mark.parametrize("use_eagle", [False, True])
def test_unchunked_item_with_budget_equal_to_item_size(use_eagle):
    scheduler = make_scheduler(
        use_eagle, max_num_batched_tokens=ITEM, disable_chunked_mm_input=True
    )
    request = make_request("r", 24, [("imgY", 12, ITEM)])
    history, trace = run_prefill(scheduler, request)
    # The item itself is always scheduled in one step.
    (encoded_step,) = [i for i, t in enumerate(trace) if t[1]]
    assert history[encoded_step] == 12
    assert history[encoded_step + 1] == 20


def test_other_requests_are_not_affected():
    # A text-only request next to the multimodal one keeps its schedule.
    scheduler = make_scheduler(True)
    mm = make_request("mm", 32, [("imgX", 8, ITEM), ("imgY", 16, ITEM)])
    (text,) = create_requests(num_requests=1, num_tokens=20, req_ids=["text"])
    scheduler.add_request(text)
    trace: list = []
    scheduler.add_request(mm)
    for _ in range(4):
        step(scheduler, trace)
    assert trace[0][0] == {"text": 20, "mm": 15}
    assert trace[1][0].get("mm") == 1
