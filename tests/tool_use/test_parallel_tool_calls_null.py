# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An explicit null parallel_tool_calls keeps the default (parallel calls
allowed), as in the OpenAI API."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.utils import maybe_filter_parallel_tool_calls

pytestmark = pytest.mark.cpu_test

TOOL = {
    "type": "function",
    "function": {
        "name": "f",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
    },
}


def _request(**kwargs):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "call f"}],
        tools=[TOOL],
        **kwargs,
    )


def _choice(n):
    calls = [
        ToolCall(id=f"call_{i}", function=FunctionCall(name="f", arguments="{}"))
        for i in range(n)
    ]
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="", tool_calls=calls),
        finish_reason="tool_calls",
    )


@pytest.mark.parametrize("parallel, kept", [(False, 1), (True, 2), (None, 2)])
def test_post_filter_treats_null_as_default(parallel, kept):
    request = _request(tool_choice="auto", parallel_tool_calls=parallel)
    choice = maybe_filter_parallel_tool_calls(_choice(2), request)
    assert len(choice.message.tool_calls) == kept


def test_responses_echoes_default_for_null_parallel_tool_calls():
    from vllm.entrypoints.openai.responses.protocol import (
        ResponsesRequest,
        ResponsesResponse,
    )
    from vllm.sampling_params import SamplingParams

    request = ResponsesRequest(model="test", input="hi", parallel_tool_calls=None)
    response = ResponsesResponse.from_request(
        request,
        SamplingParams(max_tokens=8),
        model_name="test",
        created_time=0,
        output=[],
        status="completed",
        usage=None,
    )
    assert response.parallel_tool_calls is True
