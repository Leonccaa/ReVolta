# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-side sampling defaults reach the batch chat endpoint.

Ported from 1CatAI/1Cat-vLLM#1622: BatchChatCompletionRequest used to
hard-code temperature/top_p/min_p/repetition_penalty, so
``/v1/chat/completions/batch`` ignored the server-side default sampling
params (generation config / --override-generation-config).  The fields
are now None, mirroring ChatCompletionRequest, so unset values fall
back to the server defaults exactly like the single-conversation path.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
)

_SERVER_DEFAULTS = {
    "temperature": 0.3,
    "top_p": 0.5,
    "min_p": 0.1,
    "repetition_penalty": 1.1,
}


_MESSAGES = [{"role": "user", "content": "x"}]


def _batch_request(**kwargs) -> BatchChatCompletionRequest:
    return BatchChatCompletionRequest(
        model="test-model",
        messages=[_MESSAGES],
        **kwargs,
    )


def _single_params(request: BatchChatCompletionRequest, defaults: dict):
    # Same path as OpenAIServingChatBatch.create_batch_chat_completion.
    return request.to_chat_completion_request(_MESSAGES).to_sampling_params(
        10, defaults
    )


@pytest.mark.parametrize(
    ("field", "default"),
    [
        ("temperature", 1.0),
        ("top_p", 1.0),
        ("min_p", 0.0),
        ("repetition_penalty", 1.0),
    ],
)
def test_server_defaults_apply_when_unset(field, default):
    # Without server defaults, SamplingParams' own neutral defaults apply.
    assert getattr(_single_params(_batch_request(), {}), field) == default
    # With server defaults configured, they take effect on the batch path.
    params = _single_params(_batch_request(), _SERVER_DEFAULTS)
    assert getattr(params, field) == _SERVER_DEFAULTS[field]


def test_explicit_request_values_win_over_server_defaults():
    params = _single_params(
        _batch_request(temperature=0.9, top_p=0.2, min_p=0.0), _SERVER_DEFAULTS
    )
    assert params.temperature == 0.9
    assert params.top_p == 0.2
    assert params.min_p == 0.0
