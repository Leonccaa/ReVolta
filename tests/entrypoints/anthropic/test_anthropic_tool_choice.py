# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Anthropic tool_choice handling in /v1/messages.

- A forced tool (tool_choice type "tool") maps to an OpenAI named tool choice,
  which finishes with "stop"; Anthropic reports any tool call as "tool_use".
- disable_parallel_tool_use maps to parallel_tool_calls=false.
- A named call cut short has arguments that are not JSON. It used to raise in
  the converter and fail the whole response with a 500.
"""

import asyncio
import json

import pytest

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
    UsageInfo,
)

TOOLS = [
    {
        "name": "get_weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


@pytest.fixture
def serving():
    serving = object.__new__(AnthropicServingMessages)
    serving.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    return serving


def _convert(**tool_choice):
    request = AnthropicMessagesRequest(
        model="test",
        max_tokens=64,
        messages=[{"role": "user", "content": "weather?"}],
        tools=TOOLS,
        **({"tool_choice": tool_choice} if tool_choice else {}),
    )
    return AnthropicServingMessages._convert_anthropic_to_openai_request(request)


@pytest.mark.parametrize("choice_type", ["auto", "any", "tool"])
@pytest.mark.parametrize("disable", [None, False, True])
def test_disable_parallel_tool_use(choice_type, disable):
    choice = {"type": choice_type}
    if choice_type == "tool":
        choice["name"] = "get_weather"
    if disable is not None:
        choice["disable_parallel_tool_use"] = disable
    request = _convert(**choice)
    assert request.parallel_tool_calls is (not disable)


def test_no_tool_choice_keeps_parallel_default():
    assert _convert().parallel_tool_calls is True


def _response(finish_reason, arguments=None, content=None):
    tool_calls = []
    if arguments is not None:
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(name="get_weather", arguments=arguments),
            )
        ]
    return ChatCompletionResponse(
        id="chatcmpl-test",
        model="test",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant", content=content, tool_calls=tool_calls
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.mark.parametrize(
    "finish_reason, arguments, content, stop_reason, blocks",
    [
        # forced (named) tool choice
        ("stop", '{"city": "Paris"}', None, "tool_use", ["tool_use"]),
        # auto / required
        ("tool_calls", '{"city": "Paris"}', None, "tool_use", ["tool_use"]),
        # plain answer
        ("stop", None, "Sunny.", "end_turn", ["text"]),
        # named call cut by max_tokens
        ("length", '{"city": "Par', None, "max_tokens", []),
        ("stop", '{"city": "Par', None, "end_turn", []),
    ],
)
def test_full_converter_stop_reason(
    serving, finish_reason, arguments, content, stop_reason, blocks
):
    result = serving.messages_full_converter(
        _response(finish_reason, arguments, content)
    )
    assert result.stop_reason == stop_reason
    assert [block.type for block in result.content] == blocks
    if "tool_use" in blocks:
        assert result.content[0].input == {"city": "Paris"}


def _chunk(delta=None, finish_reason=None, usage=False):
    choices = []
    if delta is not None:
        choices = [
            ChatCompletionResponseStreamChoice(
                index=0, delta=delta, finish_reason=finish_reason
            )
        ]
    chunk = ChatCompletionStreamResponse(
        id="chatcmpl-test",
        model="test",
        choices=choices,
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        if usage
        else None,
    )
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"


def _stream_stop_reason(serving, deltas, finish_reason):
    async def source():
        yield _chunk(DeltaMessage(role="assistant", content=""))
        for i, delta in enumerate(deltas):
            last = i == len(deltas) - 1
            yield _chunk(delta, finish_reason if last else None)
        yield _chunk(usage=True)
        yield "data: [DONE]\n\n"

    async def convert():
        return [item async for item in serving.message_stream_converter(source())]

    events = [json.loads(item.split("data: ", 1)[1]) for item in asyncio.run(convert())]
    (delta,) = [e["delta"] for e in events if e["type"] == "message_delta"]
    return delta["stop_reason"], [
        e["content_block"]["type"] for e in events if e["type"] == "content_block_start"
    ]


def _tool_deltas():
    return [
        DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="call_1",
                    type="function",
                    function=DeltaFunctionCall(name="get_weather", arguments=""),
                )
            ]
        ),
        DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0, function=DeltaFunctionCall(arguments='{"city": "Paris"}')
                )
            ]
        ),
    ]


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
def test_stream_tool_call_reports_tool_use(serving, finish_reason):
    stop_reason, blocks = _stream_stop_reason(serving, _tool_deltas(), finish_reason)
    assert stop_reason == "tool_use"
    assert blocks == ["tool_use"]


@pytest.mark.parametrize(
    "finish_reason, stop_reason", [("stop", "end_turn"), ("length", "max_tokens")]
)
def test_stream_text_stop_reason_unchanged(serving, finish_reason, stop_reason):
    deltas = [DeltaMessage(content="Sunny.")]
    assert _stream_stop_reason(serving, deltas, finish_reason) == (
        stop_reason,
        ["text"],
    )
