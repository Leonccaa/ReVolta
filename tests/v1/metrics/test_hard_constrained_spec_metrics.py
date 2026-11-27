# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec-decode metrics split rounds with active target-side hard masks.

The draft path (MTP / DFlash2 argmax) never sees target-side hard masks,
so masked draft tokens are always rejected; the scheduler counts the
constrained subset per verification round. CPU-only checks.
"""

import pytest

from tests.v1.core.utils import EOS_TOKEN_ID, create_requests, create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.spec_decode.metrics import has_hard_token_mask

pytestmark = pytest.mark.cpu_test


def _params(**kwargs) -> SamplingParams:
    params = SamplingParams(max_tokens=16, **kwargs)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    if kwargs.get("bad_words"):
        # What the engine's tokenizer pass would produce.
        params._bad_words_token_ids = [[7]]
    return params


@pytest.mark.parametrize(
    "kwargs,num_output_tokens,expected",
    [
        ({}, 0, False),
        ({"bad_words": ["nope"]}, 0, True),
        ({"allowed_token_ids": [1, 2]}, 0, True),
        ({"min_tokens": 3}, 2, True),  # EOS/stop still masked
        ({"min_tokens": 3}, 3, False),  # min_tokens met
        ({"presence_penalty": 1.5, "repetition_penalty": 1.2}, 0, False),
        ({"stop_token_ids": [42]}, 0, False),
    ],
)
def test_mask_classification(kwargs, num_output_tokens, expected):
    params = _params(**kwargs)
    assert has_hard_token_mask(params, num_output_tokens) is expected
    assert has_hard_token_mask(None, num_output_tokens) is False


def test_constrained_subset_counted_end_to_end():
    scheduler = create_scheduler(num_speculative_tokens=2)
    hard, plain = create_requests(num_requests=2, num_tokens=1)
    hard.sampling_params = _params(allowed_token_ids=[1, 2])
    plain.sampling_params = _params()
    req_ids = [hard.request_id, plain.request_id]
    req_to_index = {rid: i for i, rid in enumerate(req_ids)}
    for request in (hard, plain):
        scheduler.add_request(request)

    output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0], [0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)
    scheduler.update_draft_token_ids(DraftTokenIds(req_ids, [[1, 2], [3, 4]]))

    output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1, 2, 20], [3, 21]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    outputs = scheduler.update_from_output(output, model_output)
    stats = outputs[0].scheduler_stats.spec_decoding_stats
    # Two drafts of two tokens each; hard accepted 2, plain accepted 1.
    assert stats.num_drafts == 2
    assert stats.num_draft_tokens == 4
    assert stats.num_accepted_tokens == 3
    assert stats.num_hard_constrained_draft_tokens == 2
    assert stats.num_hard_constrained_accepted_tokens == 2
