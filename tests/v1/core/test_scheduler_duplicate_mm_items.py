# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler regressions for an image repeated within one request
(upstream #54284).

The encoder cache keeps one reference per request. Freeing the first
occurrence once it is computed must not release the entry while the second
occurrence is still being computed, or another request can evict it between
the two chunks and the image is encoded a second time. The hold must not
span a different item either, or a request whose items do not all fit in the
cache would wait on itself forever.
"""

import pytest

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

CACHE_SIZE = 8


def make_scheduler() -> Scheduler:
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=28,
        max_model_len=128,
        long_prefill_token_threshold=14,
    )
    # The OPT config has no encoder; give the scheduler a small encoder
    # budget and cache so it schedules the image items itself.
    scheduler.max_num_encoder_input_tokens = CACHE_SIZE
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=CACHE_SIZE)
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


def step(scheduler: Scheduler, encoded: list[tuple[str, str]]) -> None:
    output = scheduler.schedule()
    for req_id, input_ids in output.scheduled_encoder_inputs.items():
        request = scheduler.requests[req_id]
        encoded.extend((req_id, request.mm_features[i].identifier) for i in input_ids)
    ids = list(output.num_scheduled_tokens)
    if not ids:
        return
    sampled = [
        [1]
        if scheduler.requests[r].num_computed_tokens >= scheduler.requests[r].num_tokens
        else []
        for r in ids
    ]
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


def run_to_completion(scheduler, requests, encoded, limit=60):
    for _ in range(limit):
        if all(r.is_finished() for r in requests):
            return
        step(scheduler, encoded)
    raise AssertionError("requests did not finish")


def assert_all_references_released(scheduler: Scheduler) -> None:
    manager = scheduler.encoder_cache_manager
    assert all(not refs for refs in manager.cached.values())
    assert manager.num_freeable_slots == CACHE_SIZE


def test_repeated_image_is_encoded_once_when_evicted_between_chunks():
    scheduler = make_scheduler()
    # "other" runs first in every step and reaches image B in step 2.
    other = make_request("other", 40, [("imgB", 20, CACHE_SIZE)])
    # Step 1 computes [0, 14): the first copy of A, and the start of the
    # second copy. Step 2 computes the rest of the second copy.
    repeat = make_request("repeat", 30, [("imgA", 2, 4), ("imgA", 12, 4)])
    scheduler.add_request(other)
    scheduler.add_request(repeat)
    encoded: list[tuple[str, str]] = []

    step(scheduler, encoded)
    assert repeat.num_computed_tokens == 14
    assert encoded == [("repeat", "imgA")]

    run_to_completion(scheduler, [other, repeat], encoded)
    # A is encoded once for both copies; B waits until A's second copy is
    # done instead of evicting it.
    assert encoded.count(("repeat", "imgA")) == 1
    assert encoded.count(("other", "imgB")) == 1
    assert_all_references_released(scheduler)


def test_repeated_image_does_not_block_a_different_item_in_between():
    scheduler = make_scheduler()
    # A and C do not fit in the cache together: C must evict A, and the
    # request must still finish (A is encoded again for its last copy).
    request = make_request(
        "acA", 40, [("imgA", 2, 4), ("imgC", 16, 6), ("imgA", 30, 4)]
    )
    scheduler.add_request(request)
    encoded: list[tuple[str, str]] = []

    run_to_completion(scheduler, [request], encoded)
    assert encoded == [("acA", "imgA"), ("acA", "imgC"), ("acA", "imgA")]
    assert_all_references_released(scheduler)
