# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder outputs stay cached until their tokens are confirmed (#1485)."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = pytest.mark.cpu_test


class _EncoderCache:
    def __init__(self):
        self.cached = {0}
        self.freed: list[int] = []

    def get_cached_input_ids(self, request):
        return self.cached

    def free_encoder_input(self, request, input_id):
        self.cached.discard(input_id)
        self.freed.append(input_id)


def _free(num_computed_tokens: int, num_in_flight_tokens: int) -> list[int]:
    cache = _EncoderCache()
    scheduler = SimpleNamespace(encoder_cache_manager=cache, is_encoder_decoder=False)
    image = SimpleNamespace(mm_position=SimpleNamespace(offset=10, length=4096))
    request = SimpleNamespace(
        mm_features=[image],
        num_computed_tokens=num_computed_tokens,
        num_in_flight_tokens=num_in_flight_tokens,
    )
    Scheduler._free_encoder_inputs(scheduler, request)
    return cache.freed


def test_in_flight_speculative_step_keeps_the_encoder_output():
    # The image ends at 4106. A step scheduled ahead with 1 + 4 draft tokens
    # moved the optimistic count past it, but those drafts may be rejected and
    # the next draft pass re-gathers the image tail.
    assert _free(num_computed_tokens=4108, num_in_flight_tokens=5) == []


def test_confirmed_tokens_release_the_encoder_output():
    assert _free(num_computed_tokens=4108, num_in_flight_tokens=0) == [0]
    assert _free(num_computed_tokens=4111, num_in_flight_tokens=5) == [0]


def _real_free(image_len, computed, in_flight, other_image_len):
    """Real EncoderCacheManager: can another image request evict the entry?"""
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

    mgr = EncoderCacheManager(cache_size=image_len)
    image = SimpleNamespace(
        mm_position=SimpleNamespace(offset=10, length=image_len),
        identifier="a",
    )
    req = SimpleNamespace(
        request_id="r1",
        mm_features=[image],
        num_computed_tokens=0,
        num_in_flight_tokens=0,
        get_num_encoder_embeds=lambda i: image_len,
    )
    mgr.allocate(req, 0)
    req.num_computed_tokens = computed
    req.num_in_flight_tokens = in_flight
    sched = SimpleNamespace(encoder_cache_manager=mgr, is_encoder_decoder=False)
    Scheduler._free_encoder_inputs(sched, req)
    other = SimpleNamespace(
        request_id="r2",
        mm_features=[SimpleNamespace(identifier="b")],
        get_num_encoder_embeds=lambda i: other_image_len,
    )
    can = mgr.can_allocate(other, 0, other_image_len, 0)
    if can:
        mgr.allocate(other, 0)
    return mgr.get_freed_mm_hashes()


def test_other_request_cannot_evict_unconfirmed_image():
    # Image [10, 110); chunk boundary inside it is never freeable.
    assert _real_free(100, computed=60, in_flight=0, other_image_len=100) == []
    # Whole image scheduled, last step (1+k=3 tokens) in flight: draft may be
    # rejected and re-gathers the tail -> must not be evictable.
    assert _real_free(100, computed=112, in_flight=3, other_image_len=100) == []
    # Confirmed (sync, or in-flight step completed): evictable.
    assert _real_free(100, computed=112, in_flight=0, other_image_len=100) == ["a"]
