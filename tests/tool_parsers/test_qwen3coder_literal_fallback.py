# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU replay of generated Qwen markers, including tokens hidden by parsing.

The fixture pairs actual completion IDs with logprob spellings. EOS remains in
the IDs, but is excluded from detokenized text as in the default API path.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
    _choices,
    _output,
    _request,
    _serving,
    _stream,
    _text,
    _tool_calls,
)
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.v1.core.sched.utils import check_stop

CAPTURES = json.loads(
    (Path(__file__).parent / "fixtures/qwen_template_tags.json").read_text()
)
TOOL_IDS = [21864, 1510, 248058, 63, 52847, 248046]


def _captured_tokenizer():
    spellings = {248068: "<think>", 248059: "</tool_call>"}
    for capture in CAPTURES:
        for tid, token in zip(capture["ids"], capture["tokens"]):
            assert spellings.get(tid, token) == token
            spellings[tid] = token
    tok = MagicMock()
    tok.get_vocab.return_value = {s: i for i, s in spellings.items()}
    tok.decode.side_effect = lambda ids, **kw: "".join(
        spellings[i]
        for i in ids
        if not (kw.get("skip_special_tokens", False) and i == 248046)
    )
    return tok


@pytest.fixture(autouse=True)
def wrapped(monkeypatch):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)


@pytest.mark.parametrize("burst", [1, 2, 6])
@pytest.mark.parametrize("finish", ["stop", "length", "repetition"])
@pytest.mark.parametrize("empty_terminal", [False, True])
def test_captured_tool_opener_is_not_lost(burst, finish, empty_terminal):
    tok = _captured_tokenizer()
    outputs = []
    for start in range(0, len(TOOL_IDS), burst):
        ids = TOOL_IDS[start : start + burst]
        outputs.append(_output(tok.decode(ids, skip_special_tokens=True), ids))
    # Length and stop-string termination need not carry an EOS token or text.
    if empty_terminal:
        outputs.append(_output("", [], finish))
    else:
        outputs[-1].outputs[0].finish_reason = finish
        outputs[-1].finished = True
    request = _request(stream=True, tool_choice="auto", return_token_ids=True)
    outputs[0].prompt_token_ids = [248068, 248069]
    chunks = _stream(
        _serving(_WrappedParser),
        request,
        outputs,
        tok,
        chat_template_kwargs={"enable_thinking": False},
    )
    choices = _choices(chunks)[0]
    assert _text(choices, "content") == "START `<tool_call>` DONE"
    assert not any(c["delta"].get("tool_calls") for c in choices)
    assert choices[-1]["finish_reason"] == finish
    assert [tid for c in choices for tid in c.get("token_ids", [])] == TOOL_IDS


@pytest.mark.parametrize(
    "text",
    [
        "<",
        "START <tool_",
        "<tool_call>",
        "<tool_call>not a call DONE",
        '<tool_call>{"name": 7}</tool_call> DONE',
        "START <tool_call><function=unfinished",
    ],
)
@pytest.mark.parametrize("burst", [1, 5, 1000])
def test_uncommitted_tool_markup_round_trips(text, burst, monkeypatch):
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", None)
    tok = _byte_tokenizer()
    request = _request(stream=True, tool_choice="auto")
    parser = Qwen3CoderToolParser(tok, request.tools)
    full = parser.extract_tool_calls(text, request)
    assert not full.tools_called
    assert full.content == text
    ids = tok.encode(text, add_special_tokens=False)
    outputs = [
        _output(tok.decode(ids[i : i + burst]), ids[i : i + burst])
        for i in range(0, len(ids), burst)
    ]
    outputs.append(_output("", [], "stop"))
    choices = _choices(_stream(_serving(_WrappedParser), request, outputs, tok))[0]
    assert _text(choices, "content") == text


@pytest.mark.parametrize("capture", CAPTURES, ids=lambda c: c["name"])
@pytest.mark.parametrize("burst", [1, 5, 256])
def test_captured_reasoning_boundaries(capture, burst):
    tok = _captured_tokenizer()
    thinking = capture["name"].split("_")[1] == "1"
    parser = Qwen3ReasoningParser(
        tok, chat_template_kwargs={"enable_thinking": thinking}
    )
    request = _request()
    ids = capture["ids"]
    text = tok.decode(ids, skip_special_tokens=True)
    full = parser.extract_reasoning_with_token_ids(text, ids, request)
    if not thinking:
        assert full == (None, text)
        assert parser.extract_reasoning(text, request) == (None, text)
        return
    reasoning, content = "", ""
    for i in range(0, len(ids), burst):
        before = tok.decode(ids[:i], skip_special_tokens=True)
        current = tok.decode(ids[: i + burst], skip_special_tokens=True)
        delta = parser.extract_reasoning_streaming(
            before,
            current,
            current[len(before) :],
            ids[:i],
            ids[: i + burst],
            ids[i : i + burst],
        )
        if delta:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
    assert (reasoning or None, content or None) == (full[0] or None, full[1] or None)
    # Only the first actual boundary is structural. Later markers survive.
    marker = "<tool_call>" if "tool_start" in capture["name"] else "</think>"
    if marker in text:
        assert marker in (full[1] or "")


@pytest.mark.parametrize("capture", CAPTURES, ids=lambda c: c["name"])
@pytest.mark.parametrize("burst", [1, 5, 256])
def test_captured_serving_stream_matches_full_split(capture, burst):
    tok = _captured_tokenizer()
    thinking = capture["name"].split("_")[1] == "1"
    kwargs = {"enable_thinking": thinking}
    request = _request(stream=True, tool_choice="auto")
    ids = capture["ids"]
    text = tok.decode(ids, skip_special_tokens=True)
    reasoning, content = Qwen3ReasoningParser(
        tok, chat_template_kwargs=kwargs
    ).extract_reasoning_with_token_ids(text, ids, request)
    full = Qwen3CoderToolParser(tok, request.tools).extract_tool_calls(
        content or "", request
    )
    assert not full.tools_called
    outputs = [
        _output(
            tok.decode(ids[i : i + burst], skip_special_tokens=True), ids[i : i + burst]
        )
        for i in range(0, len(ids), burst)
    ]
    outputs.append(_output("", [], "stop"))
    for output in outputs:
        output.prompt_token_ids = [248068] if thinking else [248068, 248069]
    choices = _choices(
        _stream(
            _serving(_WrappedParser),
            request,
            outputs,
            tok,
            chat_template_kwargs=kwargs,
        )
    )[0]
    assert _text(choices, "content") == (full.content or "")
    assert _text(choices, "reasoning") == (reasoning or "")
    assert not any(c["delta"].get("tool_calls") for c in choices)


def test_normal_call_remains_a_tool_call(monkeypatch):
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", None)
    tok = _byte_tokenizer()
    request = _request(stream=True, tool_choice="auto")
    text = "<tool_call><function=f><parameter=x>v</parameter></function></tool_call>"
    ids = tok.encode(text, add_special_tokens=False)
    outputs = [_output(tok.decode([i]), [i]) for i in ids]
    outputs.append(_output("", [], "stop"))
    choices = _choices(_stream(_serving(_WrappedParser), request, outputs, tok))[0]
    assert _tool_calls(choices) == (["f"], {"x": "v"})
    assert _text(choices, "content") == ""


def test_fallback_is_flushed_only_once():
    tok = _captured_tokenizer()
    request = _request()
    parser = Qwen3CoderToolParser(tok, request.tools)
    text = tok.decode(TOOL_IDS, skip_special_tokens=True)
    first = parser.extract_tool_calls_streaming(
        "", text, text, [], TOOL_IDS, TOOL_IDS, request
    )
    pending = parser.finish_streaming()
    assert (first.content if first else "") + pending.content == text
    assert parser.finish_streaming() is None


def test_thinking_off_preserves_literal_thinking_tags():
    parser = Qwen3ReasoningParser(
        _captured_tokenizer(), chat_template_kwargs={"enable_thinking": False}
    )
    for text in ("START `</think>` DONE", "START `<think>` DONE"):
        assert parser.extract_reasoning(text, _request()) == (None, text)


def test_ordinary_token_spellings_do_not_end_reasoning():
    tok = _byte_tokenizer()
    text = "Reason about `</think>` and `<tool_call>` literally."
    # Encode one character at a time so the tokenizer cannot select a marker.
    ids = [tid for char in text for tid in tok.encode(char, add_special_tokens=False)]
    assert tok.decode(ids) == text
    parser = Qwen3ReasoningParser(tok)
    assert parser.extract_reasoning_with_token_ids(text, ids, _request()) == (
        text,
        None,
    )


def test_captured_im_end_stops_before_parser_extraction():
    capture = next(c for c in CAPTURES if c["name"] == "matrix_1_0_im_end")
    assert capture["ids"][-1] == 248046
    request = SimpleNamespace(
        pooling_params=None,
        sampling_params=SimpleNamespace(min_tokens=0, eos_token_id=248046),
        num_output_tokens=len(capture["ids"]),
        output_token_ids=capture["ids"],
    )
    assert check_stop(request, max_model_len=524288)
    assert "DONE" not in "".join(capture["tokens"])
