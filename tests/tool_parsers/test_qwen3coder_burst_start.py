# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consume opener, function bodies and subsequent calls in the same delta."""

import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
    _choices,
    _serving,
    _stream,
)
from tests.tool_parsers.test_qwen3coder_burst_after_header import (
    NOARG,
    SHORT,
    _non_streaming_calls,
    _request,
    _steps,
    _streamed_calls,
)
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


@pytest.mark.parametrize("text", [NOARG, SHORT, SHORT + NOARG, SHORT + SHORT])
@pytest.mark.parametrize("prefix", ["", "Ready.\n"])
@pytest.mark.parametrize("empty_terminal", [False, True])
def test_entire_burst_matches_full(text, prefix, empty_terminal):
    tok = _byte_tokenizer()
    request = _request(True)
    text = prefix + text
    expected = Qwen3CoderToolParser(tok, request.tools).extract_tool_calls(
        text, request
    )
    parser = Qwen3CoderToolParser(tok, request.tools)
    ids = tok.encode(text, add_special_tokens=False)
    delta = parser.extract_tool_calls_streaming("", text, text, [], ids, ids, request)
    assert delta is not None
    assert delta.content == expected.content
    assert [(c.function.name, c.function.arguments) for c in delta.tool_calls] == [
        (c.function.name, c.function.arguments) for c in expected.tool_calls
    ]
    assert [c.index for c in delta.tool_calls] == list(range(len(expected.tool_calls)))
    assert len({c.id for c in delta.tool_calls}) == len(expected.tool_calls)
    if empty_terminal:
        terminal = parser.extract_tool_calls_streaming(
            text, text, "", ids, ids, [], request
        )
        assert terminal is None or not terminal.tool_calls
    assert parser.finish_streaming() is None


@pytest.mark.parametrize("text", [SHORT, SHORT + NOARG, SHORT + SHORT])
@pytest.mark.parametrize("prefix", ["Ready.\n", "<think>Plan.</think>"])
def test_chat_terminal_burst_keeps_content_reasoning_and_all_calls(
    monkeypatch, text, prefix
):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)
    tok = _byte_tokenizer()
    thinking = prefix.startswith("<think>")
    outputs = _steps(tok, [prefix + text], eos_alone=False)
    if thinking:
        outputs[0].prompt_token_ids = tok.encode("<think>", add_special_tokens=False)
    choices = _choices(
        _stream(
            _serving(_WrappedParser),
            _request(True),
            outputs,
            tok,
            chat_template_kwargs={"enable_thinking": thinking},
        )
    )[0]
    assert _streamed_calls(choices) == _non_streaming_calls(tok, text)
    assert "".join(c["delta"].get("content") or "" for c in choices) == (
        "" if thinking else prefix
    )
    assert "".join(c["delta"].get("reasoning") or "" for c in choices) == (
        "Plan." if thinking else ""
    )
    assert choices[-1]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("cut", [1, 11, 28, 43, len(SHORT) - 2])
def test_partial_burst_continues_without_duplicate_headers(cut):
    tok = _byte_tokenizer()
    request = _request(True)
    parser = Qwen3CoderToolParser(tok, request.tools)
    expected = Qwen3CoderToolParser(tok, request.tools).extract_tool_calls(
        SHORT, request
    )
    previous = ""
    previous_ids: list[int] = []
    names = []
    arguments = ""
    for current in [SHORT[:cut], SHORT]:
        ids = tok.encode(current, add_special_tokens=False)
        delta = parser.extract_tool_calls_streaming(
            previous,
            current,
            current[len(previous) :],
            previous_ids,
            ids,
            ids[len(previous_ids) :],
            request,
        )
        for call in delta.tool_calls if delta else []:
            if call.id:
                names.append(call.function.name)
            arguments += call.function.arguments or ""
        previous, previous_ids = current, ids
    assert names == ["f"]
    assert arguments == expected.tool_calls[0].function.arguments
    assert parser.finish_streaming() is None


def test_partial_body_is_emitted_in_the_opener_delta():
    tok = _byte_tokenizer()
    request = _request(True)
    parser = Qwen3CoderToolParser(tok, request.tools)
    text = "<tool_call><function=f><parameter=x>value"
    ids = tok.encode(text, add_special_tokens=False)
    delta = parser.extract_tool_calls_streaming("", text, text, [], ids, ids, request)
    assert len(delta.tool_calls) == 1
    call = delta.tool_calls[0]
    assert call.function.name == "f"
    assert call.function.arguments == '{"x": "value'
    # A committed partial call keeps the existing incremental protocol.
    assert parser.finish_streaming() is None


@pytest.mark.parametrize("preceding", ["", SHORT])
def test_terminal_partial_header_keeps_real_prefix(monkeypatch, preceding):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)
    tok = _byte_tokenizer()
    text = preceding + "<tool_call><function=f>\n"
    outputs = _steps(tok, [text], eos_alone=False)
    outputs[-1].outputs[0].finish_reason = "length"
    choices = _choices(
        _stream(
            _serving(_WrappedParser),
            _request(True),
            outputs,
            tok,
            chat_template_kwargs={"enable_thinking": False},
        )
    )[0]
    calls = _streamed_calls(choices)
    assert calls[-1] == ("f", "{")
    assert len(calls) == (2 if preceding else 1)
    assert choices[-1]["finish_reason"] == "length"


def test_header_only_event_opens_arguments_on_a_later_delta():
    tok = _byte_tokenizer()
    request = _request(True)
    parser = Qwen3CoderToolParser(tok, request.tools)
    previous = ""
    previous_ids: list[int] = []
    deltas = []
    for piece in ["<tool_call><function=f", ">\n", "\n"]:
        current = previous + piece
        ids = tok.encode(current, add_special_tokens=False)
        delta = parser.extract_tool_calls_streaming(
            previous,
            current,
            piece,
            previous_ids,
            ids,
            ids[len(previous_ids) :],
            request,
        )
        if delta and delta.tool_calls:
            deltas.append(delta.tool_calls[0])
        previous, previous_ids = current, ids
    assert deltas[0].function.name == "f"
    assert deltas[0].function.arguments == ""
    assert deltas[1].id is None
    assert deltas[1].function.arguments == "{"
    assert parser.prev_tool_call_arr[0]["arguments"] == "{"


@pytest.mark.parametrize("text", [NOARG, SHORT, SHORT + NOARG, SHORT + SHORT])
@pytest.mark.parametrize("prefix", ["", "Ready.\n"])
def test_finish_parses_valid_buffer_before_literal_fallback(text, prefix):
    tok = _byte_tokenizer()
    request = _request(True)
    expected = Qwen3CoderToolParser(tok, request.tools).extract_tool_calls(
        prefix + text, request
    )
    parser = Qwen3CoderToolParser(tok, request.tools)
    # Exercise finish independently of the normal per-delta draining path.
    parser.streaming_request = request
    parser._fallback_text = prefix + text
    parser._content_chars_sent = len(prefix)
    delta = parser.finish_streaming()
    assert delta is not None
    assert delta.content is None
    assert [(c.function.name, c.function.arguments) for c in delta.tool_calls] == [
        (c.function.name, c.function.arguments) for c in expected.tool_calls
    ]
    assert parser.streamed_args_for_tool == [
        c.function.arguments for c in expected.tool_calls
    ]
    assert parser.finish_streaming() is None
