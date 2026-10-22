# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU replay: whitespace-only content next to auto tool calls.

The non-streaming response drops content that is only whitespace when tool
calls are returned, such as the line breaks a Qwen3 model writes after
</think> before <tool_call>. Streaming must not send it either, while content
that turns out to be text keeps its leading whitespace.
"""

import asyncio

import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
    _choices,
    _generate,
    _request,
    _serving,
    _stream,
    _text,
)
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

CALL = (
    "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n"
    "</tool_call>"
)
# (thinking, completion, content of the non-streaming response, tool names)
CASES = {
    "think-call": (True, "Plan.</think>\n\n" + CALL, None, ["f"]),
    "think-two-calls": (
        True,
        "Plan.</think>\n\n" + CALL + "\n" + CALL,
        None,
        ["f", "f"],
    ),
    "no-think-call": (False, "\n\n" + CALL, None, ["f"]),
    "think-text": (True, "Plan.</think>\n\nHello.", "\n\nHello.", []),
    "think-whitespace-only": (True, "Plan.</think>\n\n", "\n\n", []),
    "think-text-then-call": (
        True,
        "Plan.</think>\n\nI will.\n" + CALL,
        "\n\nI will.\n",
        ["f"],
    ),
    "no-think-text": (False, "\n\nHello.", "\n\nHello.", []),
}


@pytest.fixture
def tok():
    return _byte_tokenizer()


@pytest.fixture(autouse=True)
def wrapped_parser(monkeypatch):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)


def _prompt(tok, thinking):
    # Qwen3.5 templates open <think> in the prompt and, with thinking
    # disabled, close it there too.
    text = "q<think>\n" if thinking else "q<think>\n\n</think>\n\n"
    return tok.encode(text, add_special_tokens=False)


def _out(prompt, text, ids, finish):
    return RequestOutput(
        request_id="test",
        prompt="x",
        prompt_token_ids=prompt,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=ids,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish,
            )
        ],
        finished=finish is not None,
    )


def _kwargs(thinking):
    return {"enable_thinking": thinking}


def _full(tok, thinking, completion):
    ids = tok.encode(completion, add_special_tokens=False)
    response = asyncio.run(
        _serving(_WrappedParser).chat_completion_full_generator(
            _request(tool_choice="auto"),
            _generate([_out(_prompt(tok, thinking), completion, ids, "stop")]),
            "test",
            "test",
            [],
            tok,
            RequestResponseMetadata(request_id="test"),
            reasoning_parser=Qwen3ReasoningParser(
                tok, chat_template_kwargs=_kwargs(thinking)
            ),
        )
    )
    message = response.choices[0].message
    return message.content, [call.function.name for call in message.tool_calls]


def _streamed(tok, thinking, completion, chunk, offset):
    ids = tok.encode(completion, add_special_tokens=False)
    bounds = sorted({0, len(ids), *range(offset or chunk, len(ids), chunk)})
    outputs = [
        _out(
            _prompt(tok, thinking),
            tok.decode(ids[start:end]),
            ids[start:end],
            "stop" if end == len(ids) else None,
        )
        for start, end in zip(bounds, bounds[1:])
    ]
    chunks = _stream(
        _serving(_WrappedParser),
        _request(stream=True, tool_choice="auto"),
        outputs,
        tok,
        chat_template_kwargs=_kwargs(thinking),
    )
    choices = _choices(chunks)[0]
    names = [
        call["function"]["name"]
        for choice in choices
        for call in choice["delta"].get("tool_calls", [])
        if call["function"].get("name")
    ]
    return choices, names


@pytest.mark.parametrize("case", CASES)
def test_full_response(tok, case):
    thinking, completion, content, names = CASES[case]
    assert _full(tok, thinking, completion) == (content, names)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    ("chunk", "offset"), [(k, o) for k in range(1, 6) for o in range(k)]
)
def test_stream_matches_full_response(tok, case, chunk, offset):
    thinking, completion, content, names = CASES[case]
    choices, streamed_names = _streamed(tok, thinking, completion, chunk, offset)
    assert streamed_names == names
    assert _text(choices, "content") == (content or "")
    # No chunk carries whitespace-only content alongside a tool-call response.
    if names and content is None:
        assert not any(choice["delta"].get("content") for choice in choices)
    assert choices[-1]["finish_reason"] == ("tool_calls" if names else "stop")
