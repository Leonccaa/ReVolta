# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen finish propagation through the shared parser entry point.

The finish flag follows vllm-project/vllm#44017 (9affc17a05a4).
The trailing '<' case adapts TestAdapterFinishOnStreamEnd from upstream
tests/parser/engine/test_parser_engine.py at 3213796cd32d. Our legacy Qwen
parser also restores uncommitted markup, including the #47562 cutoff inputs.
"""

import asyncio

import pytest
from openai.types.responses import FunctionTool

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
    _output,
    _request,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

TOOLS = [FunctionTool(type="function", name="f", parameters={"type": "object"})]


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)
    return _WrappedParser(_byte_tokenizer())


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize(
    "tail",
    [
        "<",
        "<tool_",
        "<tool_call>",
        "<tool_call>\n",
        "<tool_call>\n<",
        "<tool_call>\n<function",
        "<tool_call>` DONE",
        "<tool_call>garbage</tool_call> B",
    ],
)
@pytest.mark.parametrize("empty_terminal", [False, True])
def test_finished_releases_uncommitted_text(parser, responses, tail, empty_terminal):
    request = _request(tool_choice="auto")
    if responses:
        request = ResponsesRequest(
            model="test", input="hi", tools=TOOLS, tool_choice="auto"
        )
    tok = parser.model_tokenizer
    # The explicit boundary also covers a finish delta shared with reasoning.
    first = "reasoning</think>Hello world"
    ids = tok.encode(first, add_special_tokens=False)
    delta = parser.parse_delta(first, ids, request, finished=False)
    assert delta.reasoning == "reasoning"
    content = delta.content or ""
    tail_ids = tok.encode(tail, add_special_tokens=False)
    for i, tid in enumerate(tail_ids):
        last = i == len(tail_ids) - 1
        delta = parser.parse_delta(
            tok.decode([tid]),
            [tid],
            request,
            finished=last and not empty_terminal,
        )
        if delta:
            content += delta.content or ""
            assert not delta.tool_calls
    if empty_terminal:
        delta = parser.parse_delta("", [], request, finished=True)
        if delta:
            content += delta.content or ""
    assert content == "Hello world" + tail
    again = parser.parse_delta("", [], request, finished=True)
    assert not again or not again.content


def test_default_finished_keeps_legacy_callers(parser):
    request = _request(tool_choice="auto")
    tok = parser.model_tokenizer
    ids = tok.encode("</think>Hello<", add_special_tokens=False)
    delta = parser.parse_delta("</think>Hello<", ids, request)
    assert delta.content == "Hello"
    assert parser.parse_delta("", [], request, finished=True).content == "<"


def test_reasoning_only_termination_does_not_flush_tool_text(parser, monkeypatch):
    request = _request()
    monkeypatch.setattr(
        parser.tool_parser,
        "finish_streaming",
        lambda: pytest.fail("tool parser was never entered"),
    )
    delta = parser.parse_delta("Still thinking", [], request, finished=True)
    assert delta == DeltaMessage(reasoning="Still thinking")


@pytest.mark.parametrize("finish", ["stop", "length", "repetition"])
@pytest.mark.parametrize("empty_terminal", [False, True])
def test_responses_serving_propagates_finish(parser, finish, empty_terminal):
    tok = parser.model_tokenizer
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.parser = _WrappedParser
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.chat_template_kwargs = {"enable_thinking": False}
    request = ResponsesRequest(
        model="test", input="hi", tools=TOOLS, tool_choice="auto"
    )
    text = "START `<tool_call>` DONE"
    ids = tok.encode(text, add_special_tokens=False)
    outputs = [_output(tok.decode([tid]), [tid]) for tid in ids]
    if empty_terminal:
        outputs.append(_output("", [], finish))
    else:
        outputs[-1].outputs[0].finish_reason = finish
    prompt = tok.encode("<think></think>", add_special_tokens=False)

    async def generate():
        for output in outputs:
            output.prompt_token_ids = prompt
            context = SimpleContext()
            context.last_output = output
            yield context

    async def collect():
        return [
            event
            async for event in serving._process_simple_streaming_events(
                request,
                None,
                generate(),
                SimpleContext(),
                "test",
                tok,
                RequestResponseMetadata(request_id="test"),
                0,
                lambda event: event,
            )
        ]

    events = asyncio.run(collect())
    content = "".join(
        event.delta for event in events if event.type == "response.output_text.delta"
    )
    assert content == text
    assert not any(
        event.type == "response.function_call_arguments.delta" for event in events
    )
