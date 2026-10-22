# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tool_choice="none" must not produce tool calls in streaming either.

The non-streaming path never runs the tool parser for tool_choice="none" and
returns any tool-call markup the model writes as content. Streaming used to
run the tool parser on every delta regardless of tool_choice, so the same
output became a tool call with finish_reason "tool_calls".
"""

import asyncio

import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
    _choices,
    _generate,
    _output,
    _serving,
    _stream,
    _text,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import RequestOutput
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }
]
XML_CALL = (
    "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n"
    "</tool_call>"
)


@pytest.fixture
def tok():
    return _byte_tokenizer()


@pytest.fixture(autouse=True)
def wrapped_parser(monkeypatch):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)


def _request(stream, with_tools):
    kwargs = {"tools": TOOLS, "tool_choice": "none"} if with_tools else {}
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "call f"}],
        stream=stream,
        **kwargs,
    )


def _with_prompt(output: RequestOutput, tok, thinking: bool) -> RequestOutput:
    # The chat template ends the prompt with an open <think> when thinking is
    # on, and with a closed empty block when it is off.
    prompt = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    output.prompt_token_ids = tok.encode(prompt, add_special_tokens=False)
    return output


def _steps(tok, text, thinking=False, finish="stop", eos_alone=False):
    """One engine step per token; optionally a final text-less EOS step."""
    ids = tok.encode(text, add_special_tokens=False)
    outputs = [
        _output(
            tok.decode([tid]), [tid], None if eos_alone or i < len(ids) - 1 else finish
        )
        for i, tid in enumerate(ids)
    ]
    if eos_alone:
        outputs.append(_output("", [0], finish))
    return [_with_prompt(output, tok, thinking) for output in outputs]


def _full(tok, request, text, thinking):
    ids = tok.encode(text, add_special_tokens=False)
    kwargs = {"enable_thinking": thinking}
    response = asyncio.run(
        _serving(_WrappedParser).chat_completion_full_generator(
            request,
            _generate([_with_prompt(_output(text, ids, "stop"), tok, thinking)]),
            "test",
            "test",
            [],
            tok,
            RequestResponseMetadata(request_id="test"),
            reasoning_parser=Qwen3ReasoningParser(tok, chat_template_kwargs=kwargs),
        )
    )
    return response.choices[0]


@pytest.mark.parametrize("with_tools", [True, False])
def test_streamed_tool_markup_stays_content(tok, with_tools):
    request = _request(stream=True, with_tools=with_tools)
    chunks = _stream(
        _serving(_WrappedParser),
        request,
        _steps(tok, XML_CALL),
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": False},
    )
    choices = _choices(chunks)[0]
    assert not any(choice["delta"].get("tool_calls") for choice in choices)
    assert _text(choices, "content") == XML_CALL
    assert choices[-1]["finish_reason"] == "stop"

    full = _full(tok, _request(stream=False, with_tools=with_tools), XML_CALL, False)
    assert full.message.content == XML_CALL
    assert not full.message.tool_calls
    assert full.finish_reason == "stop"


def test_streamed_tool_markup_after_reasoning_stays_content(tok):
    text = "Think first.</think>\n\n" + XML_CALL
    chunks = _stream(
        _serving(_WrappedParser),
        _request(stream=True, with_tools=True),
        _steps(tok, text, thinking=True),
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": True},
    )
    choices = _choices(chunks)[0]
    assert not any(choice["delta"].get("tool_calls") for choice in choices)
    assert _text(choices, "reasoning") == "Think first."
    assert _text(choices, "content") == "\n\n" + XML_CALL
    assert choices[-1]["finish_reason"] == "stop"

    full = _full(tok, _request(stream=False, with_tools=True), text, True)
    assert full.message.content == "\n\n" + XML_CALL
    assert full.message.reasoning == "Think first."


@pytest.mark.parametrize("with_tools", [True, False])
def test_plain_text_keeps_final_empty_content_chunk(tok, with_tools):
    # A text-less EOS step still ends with an empty content delta, as before.
    text = "Plain answer: a < b and c > d."
    chunks = _stream(
        _serving(_WrappedParser),
        _request(stream=True, with_tools=with_tools),
        _steps(tok, text, eos_alone=True),
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": False},
    )
    choices = _choices(chunks)[0]
    assert _text(choices, "content") == text
    assert choices[-1]["delta"] == {"content": ""}
    assert choices[-1]["finish_reason"] == "stop"
