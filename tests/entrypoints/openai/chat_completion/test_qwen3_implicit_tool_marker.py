# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU replay: an implicit <tool_call> reasoning end under a JSON grammar.

With thinking enabled, a Qwen3 model may open <tool_call> before closing
</think>. The structured-output manager then ends reasoning at that token and
applies the tool_choice="required" (or named) JSON grammar to the tokens after
it, so the output is ``reasoning<tool_call>[...]``. The marker must not be
part of the JSON content handed to the tool-call parsing.
"""

import asyncio
import json

import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
    _choices,
    _generate,
    _output,
    _request,
    _serving,
    _stream,
    _text,
    _tool_calls,
)
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

REASONING = "Call f.\n"
GRAMMAR_OUTPUT = {
    "required": '[{"name": "f", "parameters": {"x": "v"}}]',
    "named": '{"x": "v"}',
}
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


def _tool_request(tok, choice, stream):
    tool_choice = (
        {"type": "function", "function": {"name": "f"}} if choice == "named" else choice
    )
    request = _request(stream=stream, tool_choice=tool_choice)
    # Install the same grammar the server does before generation.
    request = Qwen3CoderToolParser(tok, request.tools).adjust_request(request)
    if choice in GRAMMAR_OUTPUT:
        assert request.structured_outputs.json is not None
    else:
        assert request.structured_outputs is None
    return request


def _full(tok, request, text):
    ids = tok.encode(text, add_special_tokens=False)
    response = asyncio.run(
        _serving(_WrappedParser).chat_completion_full_generator(
            request,
            _generate([_output(text, ids, "stop")]),
            "test",
            "test",
            [],
            tok,
            RequestResponseMetadata(request_id="test"),
            reasoning_parser=Qwen3ReasoningParser(tok),
        )
    )
    choice = response.choices[0]
    calls = [
        (call.function.name, json.loads(call.function.arguments))
        for call in choice.message.tool_calls
    ]
    return choice, calls


def _streamed(tok, request, text, chunk):
    ids = tok.encode(text, add_special_tokens=False)
    outputs = []
    for start in range(0, len(ids), chunk):
        piece = ids[start : start + chunk]
        last = start + chunk >= len(ids)
        outputs.append(
            _output(tok.decode(piece), piece, "stop" if last else None),
        )
    chunks = _stream(_serving(_WrappedParser), request, outputs, tok)
    return _choices(chunks)[0]


@pytest.mark.parametrize("choice", ["required", "named"])
def test_full_json_tool_call_after_implicit_end(tok, choice):
    text = REASONING + "<tool_call>" + GRAMMAR_OUTPUT[choice]
    message_choice, calls = _full(tok, _tool_request(tok, choice, False), text)
    assert calls == [("f", {"x": "v"})]
    assert message_choice.message.reasoning == REASONING
    expected = "tool_calls" if choice == "required" else "stop"
    assert message_choice.finish_reason == expected


@pytest.mark.parametrize("choice", ["required", "named"])
@pytest.mark.parametrize("chunk", [1, 3, 5])
def test_stream_json_tool_call_after_implicit_end(tok, choice, chunk):
    text = REASONING + "<tool_call>" + GRAMMAR_OUTPUT[choice]
    choices = _streamed(tok, _tool_request(tok, choice, True), text, chunk)
    names, arguments = _tool_calls(choices)
    assert names == ["f"]
    assert arguments == {"x": "v"}
    assert _text(choices, "reasoning") == REASONING
    assert _text(choices, "content") == ""


@pytest.mark.parametrize("choice", ["required", "named"])
def test_explicit_end_is_unchanged(tok, choice):
    text = REASONING + "</think>" + GRAMMAR_OUTPUT[choice]
    _, calls = _full(tok, _tool_request(tok, choice, False), text)
    assert calls == [("f", {"x": "v"})]
    choices = _streamed(tok, _tool_request(tok, choice, True), text, 1)
    assert _tool_calls(choices) == (["f"], {"x": "v"})


@pytest.mark.parametrize("stream", [False, True])
def test_auto_keeps_marker_for_xml_tool_parser(tok, stream):
    # Without a grammar, the implicit end still hands <tool_call> to the parser.
    request = _tool_request(tok, "auto", stream)
    text = REASONING + XML_CALL
    if stream:
        choices = _streamed(tok, request, text, 1)
        assert _tool_calls(choices) == (["f"], {"x": "v"})
        assert _text(choices, "reasoning") == REASONING
    else:
        _, calls = _full(tok, request, text)
        assert calls == [("f", {"x": "v"})]


def test_parser_split_depends_only_on_json_constraint(tok):
    parser = Qwen3ReasoningParser(tok)
    text = REASONING + "<tool_call>" + GRAMMAR_OUTPUT["required"]
    ids = tok.encode(text, add_special_tokens=False)
    constrained = _tool_request(tok, "required", False)
    free = _tool_request(tok, "auto", False)
    for split in (
        lambda request: parser.extract_reasoning(text, request),
        lambda request: parser.extract_reasoning_with_token_ids(text, ids, request),
    ):
        assert split(constrained) == (REASONING, GRAMMAR_OUTPUT["required"])
        assert split(free) == (REASONING, "<tool_call>" + GRAMMAR_OUTPUT["required"])


def test_json_response_format_content_after_implicit_end(tok):
    request = _request(response_format={"type": "json_object"})
    request.to_sampling_params(64, {})
    assert request.structured_outputs.json_object
    text = REASONING + "<tool_call>" + GRAMMAR_OUTPUT["named"]
    split = Qwen3ReasoningParser(tok).extract_reasoning(text, request)
    assert split == (REASONING, GRAMMAR_OUTPUT["named"])
