# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for incremental streaming of long qwen3_coder string args.

Field finding (2026-10-18): with ``--tool-call-parser qwen3_coder`` in
streaming mode, a long string argument (an agent writing a whole file) is held
until its closing ``</parameter>`` plus one structural lookahead, then sent in
one delta. The client sees no bytes for the whole value; behind a proxy with a
300 s streaming idle timeout the request was cut off.

Tests:

* ``test_long_string_value_streams_before_close`` -- the bug. Expected to FAIL
  on the current parser (0 bytes of the value before the close tag) and to
  pass once string values are streamed incrementally.
* ``test_streamed_args_match_non_streaming`` -- regression guard for any fix:
  the concatenated deltas must be byte-identical to the non-streaming
  ``arguments`` (the serving layer's end-of-stream ``remaining_call`` check
  relies on this) and every intermediate concatenation must be a prefix of it.
  Covers escapes, unicode, CRLF, trailing newlines, literal ``</parameter>``
  and inline ``<parameter=`` text inside the value. Passes today.
* ``test_non_string_values_are_atomic`` -- integer/object/array values must
  still be emitted only once complete. Passes today; must keep passing.

Feeding is token-by-token (``burst=1``) and in 5-token bursts (``burst=5``,
like an MTP4 step that accepts all drafts). Drop into ``tests/tool_parsers/``
of a ReVolta checkout to run; needs only a cached tokenizer, no GPU.
"""

import json
import os

import pytest
from transformers import PreTrainedTokenizerFast

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

# Same tokenizer as tests/tool_parsers/test_qwen3coder_tool_parser.py.
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

WRITE_TOOL = ChatCompletionToolsParam(
    type="function",
    function={
        "name": "write",
        "description": "Write a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
)

TYPED_TOOL = ChatCompletionToolsParam(
    type="function",
    function={
        "name": "configure",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "meta": {"type": "object"},
                "tags": {"type": "array"},
            },
        },
    },
)


@pytest.fixture(scope="module")
def tokenizer():
    if directory := os.environ.get("QWEN3_TEST_TOKENIZER"):
        return PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(directory, "tokenizer.json")
        )
    return get_tokenizer(tokenizer_name=MODEL)


def _long_markdown(lines: int = 120) -> str:
    rows = []
    for i in range(lines):
        rows.append(
            f'## Section {i}\nPCIe switch "P2P" path {i}: a\\b\ttab, '
            f"中文说明 {i}, emoji ✅, value={i * 7}."
        )
    return "\n".join(rows)


def _write_call(path: str, content: str) -> str:
    return (
        "<tool_call>\n<function=write>\n"
        f"<parameter=path>\n{path}\n</parameter>\n"
        f"<parameter=content>\n{content}\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def _stream(parser, tokenizer, model_output, request, burst):
    """Feed ``model_output`` like the serving loop does.

    Returns a list of (current_text, args_delta, n_tokens_fed) for every
    parser call, ``args_delta`` being the concatenated argument text of that
    call ("" when the parser emitted nothing for arguments).
    """
    ids = tokenizer.encode(model_output, add_special_tokens=False)
    prev_text, prev_tokens, prefix_offset, read_offset = "", None, 0, 0
    events = []
    i = 0
    while i < len(ids):
        step = ids[i : i + burst]
        delta_text = ""
        # Detokenize token by token (as the engine does), then hand the
        # parser one delta for the whole burst.
        for j in range(len(step)):
            cur_ids = ids[: i + j + 1]
            new_tokens, piece, prefix_offset, read_offset = detokenize_incrementally(
                tokenizer=tokenizer,
                all_input_ids=cur_ids,
                prev_tokens=prev_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
            prev_tokens = prev_tokens + new_tokens if prev_tokens else new_tokens
            delta_text += piece
        cur_text = prev_text + delta_text
        msg = parser.extract_tool_calls_streaming(
            prev_text,
            cur_text,
            delta_text,
            ids[:i],
            ids[: i + len(step)],
            step,
            request=request,
        )
        args = ""
        if msg is not None and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function is not None and tc.function.arguments:
                    args += tc.function.arguments
        events.append((cur_text, args, i + len(step)))
        prev_text = cur_text
        i += len(step)
    return events


def _non_streaming_args(tokenizer, tools, model_output):
    parser = Qwen3CoderToolParser(tokenizer, tools=tools)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    info = parser.extract_tool_calls(model_output, request=request)
    assert info.tools_called and len(info.tool_calls) == 1
    return info.tool_calls[0].function.arguments


@pytest.mark.parametrize("burst", [1, 5])
def test_long_string_value_streams_before_close(tokenizer, burst):
    content = _long_markdown()
    model_output = _write_call("/tmp/report.md", content)
    tools = [WRITE_TOOL]
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    parser = Qwen3CoderToolParser(tokenizer, tools=tools)
    events = _stream(parser, tokenizer, model_output, request, burst)

    value_start = model_output.index("<parameter=content>\n") + len(
        "<parameter=content>\n"
    )
    close_at = model_output.index("\n</parameter>", value_start)

    streamed = ""
    streamed_before_close = None
    last_emit_tokens = None
    max_gap_tokens = 0
    for cur_text, args, n_fed in events:
        inside_value = value_start < len(cur_text) <= close_at
        if args:
            streamed += args
            if inside_value and last_emit_tokens is not None:
                max_gap_tokens = max(max_gap_tokens, n_fed - last_emit_tokens)
            last_emit_tokens = n_fed
        if streamed_before_close is None and len(cur_text) > close_at:
            streamed_before_close = streamed

    final = _non_streaming_args(tokenizer, tools, model_output)
    escaped_value = json.dumps(content, ensure_ascii=False)
    assert streamed_before_close is not None
    assert final.startswith(streamed_before_close)
    # Bytes of the "content" value that reached the client before the close
    # tag was even generated. The current parser sends none of them.
    value_prefix = streamed_before_close.split('"content": ', 1)
    sent_value_bytes = len(value_prefix[1]) if len(value_prefix) == 2 else 0
    assert sent_value_bytes >= 0.9 * len(escaped_value), (
        f"only {sent_value_bytes} of {len(escaped_value)} value bytes "
        "were streamed before </parameter>"
    )
    # No long silent stretch while the value is being generated.
    assert max_gap_tokens <= 16, f"max silent gap {max_gap_tokens} tokens"


CASES = {
    "long_markdown": _long_markdown(),
    "quotes_backslash_unicode": 'say "hi" \\n not-newline \\\\ 中文 ✅ \x07bell',
    "crlf_and_blank_lines": "line1\r\n\r\nline3\n\n\nline6",
    "trailing_newlines": "body\n\n",
    "leading_newline_and_spaces": "\n   indented\n",
    "literal_close_inside": 'x = "before</parameter>after"\nprint(x)',
    "literal_close_then_newline_text": "a</parameter>\nnot a tag\nb",
    "inline_param_open": 'tag = "<parameter=path>inline"',
    "lt_heavy": "if a < b and b <= c: return a<b<c </ /p </param",
}


@pytest.mark.parametrize("burst", [1, 5])
@pytest.mark.parametrize("case", sorted(CASES))
def test_streamed_args_match_non_streaming(tokenizer, case, burst):
    content = CASES[case]
    model_output = _write_call("/tmp/out.txt", content)
    tools = [WRITE_TOOL]
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    parser = Qwen3CoderToolParser(tokenizer, tools=tools)
    events = _stream(parser, tokenizer, model_output, request, burst)

    final = _non_streaming_args(tokenizer, tools, model_output)
    streamed = ""
    for _, args, _ in events:
        streamed += args
        assert final.startswith(streamed), (streamed, final)
    assert streamed == final
    assert json.loads(final)["content"] == content


@pytest.mark.parametrize("burst", [1, 5])
def test_non_string_values_are_atomic(tokenizer, burst):
    model_output = (
        "<tool_call>\n<function=configure>\n"
        "<parameter=name>\nalpha beta gamma\n</parameter>\n"
        "<parameter=count>\n1234567\n</parameter>\n"
        '<parameter=meta>\n{"k": [1, 2, {"z": "w"}], "s": "long text"}\n'
        "</parameter>\n"
        '<parameter=tags>\n["a", "b", "c"]\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    tools = [TYPED_TOOL]
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
    parser = Qwen3CoderToolParser(tokenizer, tools=tools)
    events = _stream(parser, tokenizer, model_output, request, burst)

    final = _non_streaming_args(tokenizer, tools, model_output)
    streamed = "".join(args for _, args, _ in events)
    assert streamed == final
    expected = {
        '"count": ': '"count": 1234567',
        '"meta": ': '"meta": {"k": [1, 2, {"z": "w"}], "s": "long text"}',
        '"tags": ': '"tags": ["a", "b", "c"]',
    }
    for _, args, _ in events:
        for key, whole in expected.items():
            if key in args:
                assert whole in args, f"partial non-string value: {args!r}"
