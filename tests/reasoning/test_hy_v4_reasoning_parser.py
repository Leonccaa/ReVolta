# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HYV4 reasoning parser, using the real Hy4-flash tokenizer."""

import pytest

from tests.tool_parsers.hy_v4_utils import (
    THINK_BEGIN,
    THINK_END,
    encode,
    load_tokenizer,
    make_request,
    stream_deltas,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.hy_v4_reasoning_parser import HYV4ReasoningParser

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()[0]


def _stream(parser, tokenizer, output, chunk=1):
    """Feed an output the way the serving layer does; return (reasoning, content)."""
    ids = encode(tokenizer, output)
    reasoning, content = "", ""
    prev_text = ""
    prev_ids: list[int] = []
    for text, delta_ids in stream_deltas(tokenizer, [], ids, chunk):
        cur_text, cur_ids = prev_text + text, prev_ids + delta_ids
        msg = parser.extract_reasoning_streaming(
            prev_text, cur_text, text, prev_ids, cur_ids, delta_ids
        )
        prev_text, prev_ids = cur_text, cur_ids
        if msg is None:
            continue
        assert isinstance(msg, DeltaMessage)
        reasoning += msg.reasoning or ""
        content += msg.content or ""
    return reasoning or None, content or None


def test_registered():
    assert ReasoningParserManager.get_reasoning_parser("hy_v4") is HYV4ReasoningParser


# (kwargs given to the parser, expected thinking?)
MODES = [
    ({}, True),  # template default is "high"
    ({"chat_template_kwargs": None}, True),
    ({"chat_template_kwargs": {}}, True),
    ({"chat_template_kwargs": {"reasoning_effort": "high"}}, True),
    ({"chat_template_kwargs": {"reasoning_effort": "max"}}, True),
    ({"chat_template_kwargs": {"reasoning_effort": "no_think"}}, False),
    ({"reasoning_effort": "no_think"}, False),
    (
        {
            "chat_template_kwargs": {"reasoning_effort": "high"},
            "reasoning_effort": "no_think",
        },
        True,
    ),
]


@pytest.mark.parametrize("kwargs,thinking", MODES)
def test_mode_selection(tokenizer, kwargs, thinking):
    parser = HYV4ReasoningParser(tokenizer, **kwargs)
    reasoning, content = parser.extract_reasoning(
        f"why{THINK_END}answer", make_request([], tools=False)
    )
    if thinking:
        assert (reasoning, content) == ("why", "answer")
    else:
        assert (reasoning, content) == (None, f"why{THINK_END}answer")
    assert parser.is_reasoning_end([]) is (not thinking)


THINK = {"chat_template_kwargs": {"reasoning_effort": "high"}}
NO_THINK = {"chat_template_kwargs": {"reasoning_effort": "no_think"}}

THINKING_CASES = [
    # (output, reasoning, content)
    (f"step one{THINK_END}the answer", "step one", "the answer"),
    (f"{THINK_BEGIN}step one{THINK_END}the answer", "step one", "the answer"),
    (f"step one{THINK_END}", "step one", None),
    ("truncated reasoning", "truncated reasoning", None),
    (f"{THINK_END}answer only", "", "answer only"),
    ("先想一想\n再想想" + THINK_END + "答案：42", "先想一想\n再想想", "答案：42"),
]


@pytest.mark.parametrize("output,reasoning,content", THINKING_CASES)
def test_thinking_nonstreaming(tokenizer, output, reasoning, content):
    parser = HYV4ReasoningParser(tokenizer, **THINK)
    assert parser.extract_reasoning(output, make_request([], tools=False)) == (
        reasoning,
        content,
    )


@pytest.mark.parametrize("chunk", [1, 4])
@pytest.mark.parametrize("output,reasoning,content", THINKING_CASES)
def test_thinking_streaming(tokenizer, output, reasoning, content, chunk):
    parser = HYV4ReasoningParser(tokenizer, **THINK)
    got_reasoning, got_content = _stream(parser, tokenizer, output, chunk)
    assert got_reasoning == (reasoning or None)
    assert got_content == content


@pytest.mark.parametrize("chunk", [1, 4])
def test_no_think(tokenizer, chunk):
    output = "plain answer, no reasoning\nsecond line"
    parser = HYV4ReasoningParser(tokenizer, **NO_THINK)
    assert parser.extract_reasoning(output, make_request([], tools=False)) == (
        None,
        output,
    )
    assert _stream(parser, tokenizer, output, chunk) == (None, output)


def test_token_id_queries(tokenizer):
    vocab = tokenizer.get_vocab()
    begin, end = vocab[THINK_BEGIN], vocab[THINK_END]
    thinking = HYV4ReasoningParser(tokenizer, **THINK)
    no_think = HYV4ReasoningParser(tokenizer, **NO_THINK)

    assert thinking.reasoning_start_str == THINK_BEGIN
    assert thinking.reasoning_end_str == THINK_END
    # The engine-wide config reads the delimiters from a parser built without
    # request kwargs.
    default = HYV4ReasoningParser(tokenizer)
    assert (default.reasoning_start_str, default.reasoning_end_str) == (
        THINK_BEGIN,
        THINK_END,
    )

    # A prompt that ends in the injected start token is still reasoning.
    assert not thinking.is_reasoning_end([5, 6, begin])
    assert not thinking.is_reasoning_end([5, end, 6, begin])  # earlier turn closed
    assert thinking.is_reasoning_end([5, begin, 6, end, 7])
    assert no_think.is_reasoning_end([5, begin])

    assert thinking.is_reasoning_end_streaming([1, 2], [2, end])
    assert not thinking.is_reasoning_end_streaming([1, 2], [2])

    assert thinking.extract_content_ids([1, 2, end, 3, 4]) == [3, 4]
    assert thinking.extract_content_ids([1, 2, end]) == []
    assert thinking.extract_content_ids([1, 2]) == []
    assert no_think.extract_content_ids([1, 2]) == [1, 2]

    assert thinking.count_reasoning_tokens([1, 2, end, 3]) == 2
    assert thinking.count_reasoning_tokens([begin, 1, 2, end, 3]) == 2
    assert thinking.count_reasoning_tokens([1, 2, 3]) == 3  # truncated
    assert no_think.count_reasoning_tokens([1, 2, end, 3]) == 0
