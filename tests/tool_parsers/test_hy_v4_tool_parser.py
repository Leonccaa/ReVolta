# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HYV4 tool call parser, using the real Hy4-flash tokenizer."""

import json
from typing import Any

import pytest

from tests.tool_parsers.hy_v4_utils import (
    ARG_KEY_BEGIN,
    ARG_KEY_END,
    ARG_VALUE_BEGIN,
    ARG_VALUE_END,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALLS_BEGIN,
    TOOL_CALLS_END,
    TOOLS,
    encode,
    load_tokenizer,
    make_request,
    stream_deltas,
    tool,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolParser

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()[0]


@pytest.fixture
def parser(tokenizer):
    return HYV4ToolParser(tokenizer, TOOLS)


def call(name: str, **args: str) -> str:
    body = "".join(
        f"{ARG_KEY_BEGIN}{k}{ARG_KEY_END}{ARG_VALUE_BEGIN}{v}{ARG_VALUE_END}"
        for k, v in args.items()
    )
    return f"{TOOL_CALL_BEGIN}{name}{body}{TOOL_CALL_END}"


def block(*calls: str) -> str:
    return TOOL_CALLS_BEGIN + "".join(calls) + TOOL_CALLS_END


def stream(parser, tokenizer, text: str, chunk: int = 1):
    """Drive the streaming parser per token; return (content, {idx: call})."""
    request = make_request([])
    ids = encode(tokenizer, text)
    content = ""
    calls: dict[int, dict[str, Any]] = {}
    prev_text = ""
    prev_ids: list[int] = []
    for delta_text, delta_ids in stream_deltas(tokenizer, [], ids, chunk):
        cur_text, cur_ids = prev_text + delta_text, prev_ids + delta_ids
        msg = parser.extract_tool_calls_streaming(
            prev_text, cur_text, delta_text, prev_ids, cur_ids, delta_ids, request
        )
        prev_text, prev_ids = cur_text, cur_ids
        if msg is None:
            continue
        content += msg.content or ""
        for tc in msg.tool_calls:
            entry = calls.setdefault(tc.index, {"id": None, "name": None, "args": ""})
            if tc.id:
                assert entry["id"] is None
                entry["id"] = tc.id
            if tc.function and tc.function.name:
                entry["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                entry["args"] += tc.function.arguments
    return content, calls


def test_registered():
    assert ToolParserManager.get_tool_parser("hy_v4") is HYV4ToolParser


class TestNonStreaming:
    def test_no_tool_call(self, parser):
        text = "just an answer with a < b and <b>bold</b>"
        r = parser.extract_tool_calls(text, make_request([]))
        assert not r.tools_called
        assert r.content == text
        assert r.tool_calls == []

    def test_single_call_typed_args(self, parser):
        text = block(
            call(
                "search",
                query="cats and dogs",
                filters='{"year": 2024}',
                tags='["a", "b"]',
                strict="true",
                limit="2.5",
                note="null",
            )
        )
        r = parser.extract_tool_calls(text, make_request([]))
        assert r.tools_called
        assert r.content is None
        (tc,) = r.tool_calls
        assert tc.type == "function"
        assert tc.function.name == "search"
        assert json.loads(tc.function.arguments) == {
            "query": "cats and dogs",
            "filters": {"year": 2024},
            "tags": ["a", "b"],
            "strict": True,
            "limit": 2.5,
            "note": None,
        }

    def test_int_enum_and_string_args(self, parser):
        text = block(call("get_weather", city="北京", unit="c", days="3"))
        r = parser.extract_tool_calls(text, make_request([]))
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "city": "北京",
            "unit": "c",
            "days": 3,
        }

    def test_string_arg_is_not_coerced(self, parser):
        text = block(call("get_weather", city="123", unit="true"))
        r = parser.extract_tool_calls(text, make_request([]))
        assert json.loads(r.tool_calls[0].function.arguments) == {
            "city": "123",
            "unit": "true",
        }

    def test_content_before_calls(self, parser):
        text = "Let me look.\n" + block(call("get_time"))
        r = parser.extract_tool_calls(text, make_request([]))
        assert r.tools_called
        assert r.content == "Let me look.\n"
        assert r.tool_calls[0].function.name == "get_time"
        assert json.loads(r.tool_calls[0].function.arguments) == {}

    def test_multiple_calls(self, parser):
        text = block(
            call("get_weather", city="Paris"),
            call("get_time"),
            call("get_weather", city="Rome", days="2"),
        )
        r = parser.extract_tool_calls(text, make_request([]))
        assert [tc.function.name for tc in r.tool_calls] == [
            "get_weather",
            "get_time",
            "get_weather",
        ]
        assert [json.loads(tc.function.arguments) for tc in r.tool_calls] == [
            {"city": "Paris"},
            {},
            {"city": "Rome", "days": 2},
        ]

    def test_value_with_markup_like_text_and_newlines(self, parser):
        value = 'line1\nline2 "quoted" <not a tag> {braces}'
        r = parser.extract_tool_calls(
            block(call("search", query=value)), make_request([])
        )
        assert json.loads(r.tool_calls[0].function.arguments) == {"query": value}

    def test_bare_tool_call_without_wrapper(self, parser):
        text = "ok " + call("get_weather", city="Oslo")
        r = parser.extract_tool_calls(text, make_request([]))
        assert r.tools_called
        assert r.content == "ok "
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Oslo"}

    @pytest.mark.parametrize(
        "text",
        [
            # no closing wrapper
            TOOL_CALLS_BEGIN + call("get_time"),
            # tool_call not closed
            TOOL_CALLS_BEGIN + f"{TOOL_CALL_BEGIN}get_time" + TOOL_CALLS_END,
            # unclosed argument value
            TOOL_CALLS_BEGIN
            + f"{TOOL_CALL_BEGIN}search{ARG_KEY_BEGIN}query{ARG_KEY_END}"
            + f"{ARG_VALUE_BEGIN}x{TOOL_CALL_END}{TOOL_CALLS_END}",
            # empty function name
            block(f"{TOOL_CALL_BEGIN}{TOOL_CALL_END}"),
        ],
    )
    def test_malformed_falls_back_to_content(self, parser, text):
        r = parser.extract_tool_calls(text, make_request([]))
        assert not r.tools_called
        assert r.content == text


class TestStreaming:
    @pytest.mark.parametrize("chunk", [1, 2, 5])
    def test_plain_content(self, parser, tokenizer, chunk):
        text = "no tools here, only a < b comparison."
        content, calls = stream(parser, tokenizer, text, chunk)
        assert content == text
        assert calls == {}
        assert parser.prev_tool_call_arr == []

    @pytest.mark.parametrize("chunk", [1, 3])
    def test_single_call(self, parser, tokenizer, chunk):
        text = "Checking.\n" + block(
            call("get_weather", city="Paris", unit="c", days="3")
        )
        content, calls = stream(parser, tokenizer, text, chunk)
        assert content == "Checking.\n"
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["id"]
        assert json.loads(calls[0]["args"]) == {
            "city": "Paris",
            "unit": "c",
            "days": 3,
        }

    @pytest.mark.parametrize("chunk", [1, 7])
    def test_multiple_calls(self, parser, tokenizer, chunk):
        text = block(
            call("get_weather", city="Paris"),
            call("get_time"),
            call("search", query="x", limit="3"),
        )
        _, calls = stream(parser, tokenizer, text, chunk)
        assert [c["name"] for _, c in sorted(calls.items())] == [
            "get_weather",
            "get_time",
            "search",
        ]
        assert all(c["id"] for c in calls.values())
        assert len({c["id"] for c in calls.values()}) == 3
        assert [json.loads(c["args"]) for _, c in sorted(calls.items())] == [
            {"city": "Paris"},
            {},
            {"query": "x", "limit": 3},
        ]
        # Serving-layer state mirrors what was streamed.
        assert [c["name"] for c in parser.prev_tool_call_arr] == [
            "get_weather",
            "get_time",
            "search",
        ]
        assert parser.streamed_args_for_tool == [
            c["args"] for _, c in sorted(calls.items())
        ]

    def test_string_argument_split_across_deltas(self, parser, tokenizer):
        """A long string value is emitted in several argument deltas."""
        value = "a fairly long query string that spans many tokens, 含中文和标点！"
        text = block(call("search", query=value))
        request = make_request([])
        ids = encode(tokenizer, text)
        prev_text = ""
        prev_ids: list[int] = []
        arg_pieces: list[str] = []
        for delta_text, delta_ids in stream_deltas(tokenizer, [], ids, 1):
            cur_text, cur_ids = prev_text + delta_text, prev_ids + delta_ids
            msg = parser.extract_tool_calls_streaming(
                prev_text, cur_text, delta_text, prev_ids, cur_ids, delta_ids, request
            )
            prev_text, prev_ids = cur_text, cur_ids
            for tc in msg.tool_calls if msg else []:
                if tc.function and tc.function.arguments:
                    arg_pieces.append(tc.function.arguments)
        assert len(arg_pieces) > 3
        assert json.loads("".join(arg_pieces)) == {"query": value}
        # No piece before the closing one may contain the final brace.
        assert not any(p.endswith("}") for p in arg_pieces[:-1])

    def test_non_string_argument_is_withheld_until_closed(self, parser, tokenizer):
        text = block(call("search", filters='{"a": {"b": [1, 2, 3]}}'))
        _, calls = stream(parser, tokenizer, text, 1)
        assert json.loads(calls[0]["args"]) == {"filters": {"a": {"b": [1, 2, 3]}}}

    def test_streaming_matches_nonstreaming(self, tokenizer):
        text = "pre " + block(
            call("search", query="q <x>", tags='["t"]', strict="false"),
            call("get_weather", city="X", days="9"),
        )
        for chunk in (1, 4):
            content, calls = stream(
                HYV4ToolParser(tokenizer, TOOLS), tokenizer, text, chunk
            )
            whole = HYV4ToolParser(tokenizer, TOOLS).extract_tool_calls(
                text, make_request([])
            )
            assert content == whole.content
            assert [
                (c["name"], json.loads(c["args"])) for _, c in sorted(calls.items())
            ] == [
                (t.function.name, json.loads(t.function.arguments))
                for t in whole.tool_calls
            ]

    def test_bare_tool_call_without_wrapper(self, parser, tokenizer):
        text = "go " + call("get_time") + call("get_weather", city="Rome")
        content, calls = stream(parser, tokenizer, text, 1)
        assert content == "go "
        assert [c["name"] for _, c in sorted(calls.items())] == [
            "get_time",
            "get_weather",
        ]
        assert json.loads(calls[1]["args"]) == {"city": "Rome"}

    def test_unknown_tool_args_are_strings(self, tokenizer):
        parser = HYV4ToolParser(tokenizer, [tool("other")])
        _, calls = stream(
            parser, tokenizer, block(call("get_weather", city="1", days="3")), 1
        )
        assert json.loads(calls[0]["args"]) == {"city": "1", "days": "3"}


class TestAdjustRequest:
    def test_auto_is_untouched(self, parser):
        request = make_request([])
        assert parser.adjust_request(request).structured_outputs is None

    @pytest.mark.parametrize(
        "tool_choice",
        [
            "required",
            {"type": "function", "function": {"name": "get_time"}},
        ],
    )
    def test_required_and_named_are_not_json_constrained(self, parser, tool_choice):
        """JSON guided decoding would hide the calls from this parser."""
        request = make_request([], tool_choice=tool_choice)
        assert parser.adjust_request(request).structured_outputs is None
        assert HYV4ToolParser.supports_required_and_named is False

    def test_base_class_would_constrain_required(self, parser):
        request = make_request([], tool_choice="required")
        assert ToolParser.adjust_request(parser, request).structured_outputs
