# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CachedRequestData.all_token_ids is MRV1-only (upstream vLLM #45840).

The V2 model runner keeps its own token history and receives resumed
requests as new requests with their full token ids, so the scheduler must
not copy every token list for requests that skipped a step. The V1 model
runner still gets the copy.
"""

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler


def _make_cached(scheduler, running, resumed, num_tokens=1):
    empty_blocks = KVCacheBlocks(blocks=((),))
    reqs = list(running) + list(resumed)
    return scheduler._make_cached_request_data(
        running_reqs=list(running),
        resumed_reqs=list(resumed),
        num_scheduled_tokens={r.request_id: num_tokens for r in reqs},
        spec_decode_tokens={},
        req_to_new_blocks={r.request_id: empty_blocks for r in reqs},
    )


def test_cached_request_data_resumed_all_token_ids_mrv1_only():
    """all_token_ids carries a resumed request's token ids to the connector
    for the V1 model runner, but is skipped entirely for the V2 model runner.
    """
    scheduler = create_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=8)
    req.append_output_token_ids([101, 102, 103])
    assert req.request_id not in scheduler.prev_step_scheduled_req_ids

    assert not scheduler.use_v2_model_runner
    cached = _make_cached(scheduler, [], [req])
    assert req.request_id in cached.resumed_req_ids
    assert cached.all_token_ids[req.request_id] == list(req.all_token_ids)

    scheduler.use_v2_model_runner = True
    cached = _make_cached(scheduler, [], [req])
    assert req.request_id in cached.resumed_req_ids
    assert cached.all_token_ids == {}


@pytest.mark.parametrize("v2", [False, True])
def test_running_request_that_skipped_a_step(v2):
    """A running request not scheduled in the previous step gets its token ids
    only under MRV1; one scheduled in the previous step never does."""
    scheduler = create_scheduler()
    scheduler.use_v2_model_runner = v2
    skipped, kept = create_requests(num_requests=2, num_tokens=8)
    scheduler.prev_step_scheduled_req_ids = {kept.request_id}
    cached = _make_cached(scheduler, [skipped, kept], [])
    assert cached.resumed_req_ids == set()
    if v2:
        assert cached.all_token_ids == {}
    else:
        assert set(cached.all_token_ids) == {skipped.request_id}
        assert cached.all_token_ids[skipped.request_id] == list(skipped.all_token_ids)
    assert cached.num_computed_tokens == [
        skipped.num_computed_tokens,
        kept.num_computed_tokens,
    ]


def _run_one_step(scheduler, outputs_for):
    from vllm.v1.outputs import ModelRunnerOutput

    out = scheduler.schedule()
    req_ids = list(out.num_scheduled_tokens)
    mro = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[outputs_for(r)] for r in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(out, mro)
    return out


@pytest.mark.parametrize("v2", [False, True])
def test_preempt_and_resume_keeps_token_ids(v2):
    """End to end through schedule(): after a preemption the resumed request
    carries the same token ids and computed-token count it had before."""
    scheduler = create_scheduler()
    scheduler.use_v2_model_runner = v2
    (req,) = create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(req)
    for step in range(3):
        _run_one_step(scheduler, lambda r, s=step: 500 + s)
    expected_ids = list(req.all_token_ids)
    assert expected_ids[-3:] == [500, 501, 502]

    scheduler._preempt_request(req, 0.0)
    scheduler.running.remove(req)
    assert req.status == RequestStatus.PREEMPTED
    # A real preemption happens inside schedule(), so the preempted request is
    # never in that step's scheduled set.
    scheduler.prev_step_scheduled_req_ids.discard(req.request_id)
    computed_before = req.num_computed_tokens

    out = scheduler.schedule()
    if v2:
        # V2: resumed requests are sent as new requests with all token ids.
        (new,) = out.scheduled_new_reqs
        assert new.req_id == req.request_id
        assert new.prefill_token_ids == expected_ids
        assert new.num_computed_tokens == computed_before
        assert out.scheduled_cached_reqs.resumed_req_ids == set()
        assert out.scheduled_cached_reqs.all_token_ids == {}
        assert scheduler.prev_step_scheduled_req_ids == set()
    else:
        assert out.scheduled_new_reqs == []
        cached = out.scheduled_cached_reqs
        assert req.request_id in cached.resumed_req_ids
        assert cached.all_token_ids[req.request_id] == expected_ids
        idx = cached.req_ids.index(req.request_id)
        assert cached.num_computed_tokens[idx] == computed_before
        assert scheduler.prev_step_scheduled_req_ids == {req.request_id}
