# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Penalty fields on /v1/messages reach SamplingParams.

Ported from 1CatAI/1Cat-vLLM#1622: presence/frequency/repetition penalty
fields on AnthropicMessagesRequest are forwarded to the OpenAI chat
request and thus to the engine's sampling parameters.
"""

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

_convert = AnthropicServingMessages._convert_anthropic_to_openai_request


def _messages_request(**kwargs) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=128,
        messages=[{"role": "user", "content": "x"}],
        **kwargs,
    )


def test_penalties_forwarded_to_sampling_params():
    req = _messages_request(
        presence_penalty=0.5,
        frequency_penalty=0.25,
        repetition_penalty=1.2,
    )
    chat_req = _convert(req)
    assert chat_req.presence_penalty == 0.5
    assert chat_req.frequency_penalty == 0.25
    assert chat_req.repetition_penalty == 1.2
    params = chat_req.to_sampling_params(128, {})
    assert params.presence_penalty == 0.5
    assert params.frequency_penalty == 0.25
    assert params.repetition_penalty == 1.2
