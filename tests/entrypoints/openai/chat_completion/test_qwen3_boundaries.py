# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU replay of token-origin boundaries, stop guards and tool streams."""

import asyncio
import inspect
import json
import os
from unittest.mock import patch

import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
    _generate,
    _output,
    _request,
    _serving,
    _stream,
)
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.sampling_params import SamplingParams
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import IncrementalDetokenizer


@pytest.fixture
def tok():
    return _byte_tokenizer()


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def literal(tok, text):
    # Encode each character separately so added vocabulary cannot match.
    return [i for c in text for i in encode(tok, c)]


def detok(tok, thinking=True, reported_ended=None, **kwargs):
    req = EngineCoreRequest(
        request_id="test",
        external_req_id="test",
        prompt_token_ids=encode(
            tok, "q<think>\n" if thinking else "q<think>\n</think>"
        ),
        mm_features=None,
        sampling_params=SamplingParams(**kwargs),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        reasoning_ended=not thinking if reported_ended is None else reported_ended,
        reasoning_parser_kwargs={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    factory = IncrementalDetokenizer.from_new_request
    # The baseline has no frontend boundary configuration yet.
    options = (
        {"reasoning_parser_name": "qwen3"}
        if "reasoning_parser_name" in inspect.signature(factory).parameters
        else {}
    )
    return factory(tok, req, **options)


def split(parser, text, ids):
    method = getattr(parser, "extract_reasoning_with_token_ids", None)
    return (
        method(text, ids, _request())
        if method
        else parser.extract_reasoning(text, _request())
    )


@pytest.mark.parametrize("with_end", [False, True])
def test_literal_reasoning_boundary(tok, with_end):
    ids = literal(tok, "引用 </think> 仍是思考")
    if with_end:
        ids += encode(tok, "</think>答案")
    text = tok.decode(ids)
    assert split(Qwen3ReasoningParser(tok), text, ids) == (
        "引用 </think> 仍是思考",
        "答案" if with_end else None,
    )


def test_literal_and_real_end_in_same_delta(tok):
    ids = literal(tok, "</think>") + encode(tok, "</think>OK")
    text = tok.decode(ids)
    out = Qwen3ReasoningParser(tok).extract_reasoning_streaming(
        "", text, text, [], ids, ids
    )
    assert out.reasoning == "</think>"
    assert out.content == "OK"


@pytest.mark.parametrize("burst", [1, 5, 1000])
@pytest.mark.parametrize("include", [False, True])
def test_stop_only_after_reasoning(tok, burst, include):
    ids = encode(tok, "复述 STOP 然后继续</think>答案 STOP 尾巴")
    d = detok(tok, stop=["STOP"], include_stop_str_in_output=include)
    p = Qwen3ReasoningParser(tok)
    previous = ""
    prev_ids: list[int] = []
    reasoning, content = "", ""
    hit = None
    for offset in range(0, len(ids), burst):
        hit = d.update(ids[offset : offset + burst], False)
        delta = d.get_next_output_text(
            hit is not None or offset + burst >= len(ids), True
        )
        current = previous + delta
        cur_ids = d.output_token_ids
        out = p.extract_reasoning_streaming(
            previous, current, delta, prev_ids, cur_ids, cur_ids[len(prev_ids) :]
        )
        if out:
            reasoning += out.reasoning or ""
            content += out.content or ""
        previous, prev_ids = current, list(cur_ids)
        if hit:
            break
    assert hit == "STOP"
    assert reasoning == "复述 STOP 然后继续"
    assert content == "答案 " + ("STOP" if include else "")
    assert split(Qwen3ReasoningParser(tok), d.output_text, d.output_token_ids) == (
        reasoning,
        content,
    )


@pytest.mark.parametrize("burst", [1, 5])
def test_stop_lookback_keeps_reasoning_tail(tok, burst):
    d = detok(tok, stop=["NEVERMATCH"])
    ids = encode(tok, "想清楚✅</think>正文")
    p = Qwen3ReasoningParser(tok)
    previous, reasoning, content = "", "", ""
    prev_ids: list[int] = []
    for offset in range(0, len(ids), burst):
        assert d.update(ids[offset : offset + burst], False) is None
        delta = d.get_next_output_text(offset + burst >= len(ids), True)
        current = previous + delta
        out = p.extract_reasoning_streaming(
            previous,
            current,
            delta,
            prev_ids,
            d.output_token_ids,
            d.output_token_ids[len(prev_ids) :],
        )
        if out:
            reasoning += out.reasoning or ""
            content += out.content or ""
        previous, prev_ids = current, list(d.output_token_ids)
    assert (reasoning, content) == ("想清楚✅", "正文")


def test_stop_cannot_span_boundary(tok):
    d = detok(tok, stop=["x</think>y"])
    assert d.update(encode(tok, "x</think>yOK"), False) is None
    assert d.output_text == "x</think>yOK"


@pytest.mark.parametrize("thinking", [False, True])
def test_stop_min_tokens(tok, thinking):
    prefix = "STOP xx" + ("</think>" if thinking else "")
    d = detok(tok, thinking, stop=["STOP"], min_tokens=len(encode(tok, prefix)))
    assert d.update(encode(tok, prefix + "a STOPtail"), False) == "STOP"
    assert d.output_text == prefix + "a "


def test_thinking_disabled_stop_unchanged(tok):
    d = detok(tok, False, stop=["STOP"])
    assert d.update(encode(tok, "first STOP tail"), False) == "STOP"
    assert d.output_text == "first "


def tool_text(value="first", duplicate=False):
    extra = "\n<parameter=x>second</parameter>" if duplicate else ""
    return (
        f"<tool_call>\n<function=f>\n<parameter=x>{value}</parameter>{extra}"
        "\n</function>\n</tool_call>"
    )


def replay_tools(tok, text, burst, req=None):
    req = req or _request(stream=True)
    p = Qwen3CoderToolParser(tok, tools=req.tools)
    ids = encode(tok, text)
    events, previous, sent = [], "", ""
    decoder = detok(tok, False)
    for offset in range(0, len(ids), burst):
        decoder.update(ids[offset : offset + burst], False)
        current = decoder.output_text
        out = p.extract_tool_calls_streaming(
            previous,
            current,
            current[len(previous) :],
            ids[:offset],
            ids[: offset + burst],
            ids[offset : offset + burst],
            req,
        )
        fragment = (
            ""
            if not out
            else "".join(
                tc.function.arguments or ""
                for tc in out.tool_calls or []
                if tc.function
            )
        )
        sent += fragment
        events.append((current, fragment, sent))
        previous = current
    return p, events


@pytest.mark.parametrize("burst", [1, 5])
def test_duplicate_parameters_first_value(tok, burst):
    text = tool_text(duplicate=True)
    req = _request()
    with patch("vllm.tool_parsers.qwen3coder_tool_parser.logger.warning") as warning:
        full = (
            Qwen3CoderToolParser(tok, tools=req.tools)
            .extract_tool_calls(text, req)
            .tool_calls[0]
            .function.arguments
        )
        p, events = replay_tools(tok, text, burst)
        assert full == '{"x": "first"}'
        assert events[-1][2] == full
        assert p.streamed_args_for_tool == [full]
        assert any("duplicate" in call.args[0] for call in warning.call_args_list)


@pytest.mark.parametrize("burst", [1, 5])
def test_string_streams_before_close(tok, burst):
    value = '文字 "quoted" \\ tab\t✅\n' * 400
    text = tool_text(value)
    p, events = replay_tools(tok, text, burst)
    expected = json.dumps({"x": value[:-1]}, ensure_ascii=False)
    before_close = [
        sent for current, _, sent in events if "</parameter>" not in current
    ]
    assert len(before_close[-1]) > len(expected) * 0.9
    assert all(expected.startswith(sent) for _, _, sent in events)
    assert events[-1][2] == expected
    assert p.prev_tool_call_arr[0]["arguments"] == expected


@pytest.mark.parametrize("burst", [1, 5])
def test_real_service_duplicate_finish(tok, burst, monkeypatch):
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", None)
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    text = tool_text(duplicate=True)
    ids = encode(tok, text)
    outputs, previous = [], ""
    for offset in range(0, len(ids), burst):
        current = tok.decode(ids[: offset + burst])
        outputs.append(
            _output(
                current[len(previous) :],
                ids[offset : offset + burst],
                "stop" if offset + burst >= len(ids) else None,
            )
        )
        previous = current
    chunks = _stream(_serving(_WrappedParser), _request(stream=True), outputs, tok)
    args = "".join(
        tc.get("function", {}).get("arguments", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        for tc in choice["delta"].get("tool_calls", [])
    )
    assert json.loads(args) == {"x": "first"}
    assert args == '{"x": "first"}'


@pytest.mark.parametrize("burst", [1, 5])
@pytest.mark.parametrize("ending", ["", "\n", "</para", "</parameter>   "])
def test_truncated_string_state(tok, burst, ending):
    text = '<tool_call>\n<function=f>\n<parameter=x>value "中文\\' + ending
    p, events = replay_tools(tok, text, burst)
    sent = events[-1][2]
    assert sent.startswith('{"x": "value ')
    assert not sent.endswith('"}')
    assert p.streamed_args_for_tool == [sent]
    assert p.prev_tool_call_arr[0]["arguments"] == sent


@pytest.mark.parametrize("burst", [1, 5])
@pytest.mark.parametrize(
    "value",
    [
        'quote " slash \\ control\t\r\n中文✅',
        "a</parameter>\n   ordinary text\nb",
        "a</parameter>\n   <parameter=unknown>literal\nb",
        'x = "<parameter=x>inline"',
        "a\n  <parameter=unknown>literal\nb",
        "\n\nbody\n\n",
        "dense <" * 300,
        "a</parameter>" + " " * 1000 + "ordinary",
    ],
)
def test_string_prefix_and_final_object(tok, burst, value):
    text = tool_text(value)
    req = _request()
    expected = (
        Qwen3CoderToolParser(tok, tools=req.tools)
        .extract_tool_calls(text, req)
        .tool_calls[0]
        .function.arguments
    )
    _, events = replay_tools(tok, text, burst)
    assert all(expected.startswith(sent) for _, _, sent in events)
    assert events[-1][2] == expected


@pytest.mark.parametrize("burst", [1, 5])
@pytest.mark.parametrize(
    "schema,value",
    [
        ({"type": "integer"}, "12345"),
        ({"type": "object"}, '{"k": 2}'),
        ({"type": "array"}, "[1, 2]"),
        ({"type": ["string", "null"]}, "null"),
        ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "12345"),
        ({"enum": ["hello", 1]}, "1"),
    ],
)
def test_non_string_schema_atomic(tok, burst, schema, value):
    req = _request(stream=True)
    req.tools[0].function.parameters["properties"]["x"] = schema
    p = Qwen3CoderToolParser(tok, tools=req.tools)
    text = tool_text(value)
    ids = encode(tok, text)
    previous = ""
    for offset in range(0, len(ids), burst):
        current = tok.decode(ids[: offset + burst])
        out = p.extract_tool_calls_streaming(
            previous,
            current,
            current[len(previous) :],
            ids[:offset],
            ids[: offset + burst],
            ids[offset : offset + burst],
            req,
        )
        if out:
            for tc in out.tool_calls or []:
                fragment = tc.function.arguments if tc.function else ""
                if fragment and '"x"' in fragment:
                    assert "</parameter>" in current
        previous = current


def test_service_flush_rejects_non_prefix(tok, monkeypatch):
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaFunctionCall,
        DeltaMessage,
        DeltaToolCall,
    )

    class InconsistentParser(Qwen3CoderToolParser):
        def extract_tool_calls_streaming(self, previous_text, *args, **kwargs):
            fragment = "}" if previous_text else '{"x": "old"'
            self.prev_tool_call_arr = [{"name": "f", "arguments": '{"x": "new"}'}]
            self.streamed_args_for_tool = [
                '{"x": "old"' + ("}" if previous_text else "")
            ]
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id="call_test" if not previous_text else None,
                        type="function",
                        function=DeltaFunctionCall(
                            name="f" if not previous_text else None, arguments=fragment
                        ),
                    )
                ]
            )

    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", None)
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", InconsistentParser)
    chunks = _stream(
        _serving(_WrappedParser),
        _request(stream=True),
        [_output("a", [1]), _output("b", [2], "stop")],
        tok,
    )
    args = "".join(
        tc.get("function", {}).get("arguments", "")
        for chunk in chunks
        for choice in chunk["choices"]
        for tc in choice["delta"].get("tool_calls", [])
    )
    assert args == '{"x": "old"}'
    assert json.loads(args) == {"x": "old"}


@pytest.mark.parametrize("burst", [1, 5])
@pytest.mark.parametrize("repeated", ["x", "y", "z"])
def test_repeated_first_middle_last_and_two_calls(tok, burst, repeated):
    req = _request(stream=True)
    req.tools[0].function.parameters["properties"] = {
        "x": {"type": "string"},
        "y": {"type": "integer"},
        "z": {"type": "string"},
    }
    params = [("x", "first"), ("y", "1"), ("z", "last"), (repeated, "2")]
    call = (
        "<tool_call>\n<function=f>\n"
        + "\n".join(f"<parameter={key}>{value}</parameter>" for key, value in params)
        + "\n</function>\n</tool_call>"
    )
    text = call + "\n" + call
    p, events = replay_tools(tok, text, burst, req)
    expected = '{"x": "first", "y": 1, "z": "last"}'
    assert p.streamed_args_for_tool == [expected, expected]
    assert events[-1][2] == expected * 2


def test_real_tokenizer_five_token_literal_boundary():
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    directory = os.environ.get("QWEN3_TEST_TOKENIZER")
    if not directory:
        pytest.skip("optional local tokenizer not provided")
    assert directory is not None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(directory, "tokenizer.json")
    )
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend["added_tokens"] = []
    plain = Tokenizer.from_str(json.dumps(backend))
    ids = plain.encode("</think>", add_special_tokens=False).ids
    ids += encode(tokenizer, "</think>OK")
    assert len(ids) == 5
    text = tokenizer.decode(ids)
    parser = Qwen3ReasoningParser(tokenizer)
    out = parser.extract_reasoning_streaming("", text, text, [], ids, ids)
    assert (out.reasoning, out.content) == ("</think>", "OK")
    assert split(Qwen3ReasoningParser(tokenizer), text, ids) == ("</think>", "OK")


def test_hidden_reasoning_still_guards_stop(tok):
    decoder = detok(tok, reported_ended=True, stop=["STOP"])
    assert (
        decoder.update(
            encode(tok, "STOP inside thought</think>answer STOP tail"), False
        )
        == "STOP"
    )
    assert decoder.output_text == "STOP inside thought</think>answer "


@pytest.mark.xfail(
    strict=True,
    reason="literal function closer needs a separate structural recovery policy",
)
@pytest.mark.parametrize("streaming", [False, True])
def test_literal_function_closer_known_limit(tok, streaming):
    value = 'example = "<function=x></function>"\nkeep this text'
    req = _request()
    parsed = Qwen3CoderToolParser(tok, tools=req.tools).extract_tool_calls(
        tool_text(value), req
    )
    actual = parsed.tool_calls[0].function.arguments
    if streaming:
        _, events = replay_tools(tok, tool_text(value), 5)
        actual = events[-1][2]
    assert json.loads(actual) == {"x": value}


@pytest.mark.parametrize("text", ["", "</think>answer", "<think></think>answer"])
def test_empty_reasoning_non_streaming_unchanged(tok, text):
    parser = Qwen3ReasoningParser(tok)
    assert split(parser, text, encode(tok, text)) == parser.extract_reasoning(
        text, _request()
    )


@pytest.mark.parametrize(
    "generated,mutate",
    [
        pytest.param("思考</think>答案", lambda text: "✅" + text, id="extra_prefix"),
        pytest.param("ab思考</think>答案", lambda text: text[1:], id="dropped_char"),
        pytest.param("想<tool_call>x", lambda text: "✅" + text, id="implicit_tool"),
        pytest.param("<think>A</think>B", lambda text: "x" + text, id="start_token"),
    ],
)
def test_token_split_falls_back_on_decode_mismatch(tok, generated, mutate):
    # The token-id split decodes token prefixes and applies their lengths to
    # the detokenized text. When the two disagree, the offsets no longer
    # bracket the markers and the text-based split must be used instead.
    ids = encode(tok, generated)
    text = mutate(tok.decode(ids))
    parser = Qwen3ReasoningParser(tok)
    expected = parser.extract_reasoning(text, _request())
    assert expected[1] is not None
    assert split(parser, text, ids) == expected


def test_reasoning_delta_preserves_unset_content(tok):
    ids = encode(tok, "ordinary reasoning")
    parser = Qwen3ReasoningParser(tok)
    out = parser.extract_reasoning_streaming(
        "", "ordinary reasoning", "ordinary reasoning", [], ids, ids
    )
    assert out.model_dump(exclude_unset=True) == {"reasoning": "ordinary reasoning"}


@pytest.mark.parametrize("thinking", [False, True])
def test_stop_in_tool_parameter_keeps_existing_semantics(tok, thinking):
    prefix = (
        "reasoning</think>" if thinking else ""
    ) + "<tool_call><function=f><parameter=x>before "
    decoder = detok(tok, thinking, stop=["STOP"])
    assert (
        decoder.update(
            encode(tok, prefix + "STOP after</parameter></function></tool_call>"), False
        )
        == "STOP"
    )
    assert decoder.output_text == prefix


def test_non_streaming_service_uses_token_origins(tok):
    from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata

    ids = literal(tok, "quote </think> still reasoning") + encode(tok, "</think>answer")
    text = tok.decode(ids)
    request = _request(stream=False, tool_choice="none")
    response = asyncio.run(
        _serving().chat_completion_full_generator(
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
    message = response.choices[0].message
    assert (message.reasoning, message.content) == (
        "quote </think> still reasoning",
        "answer",
    )


@pytest.mark.parametrize("encoded_arguments", [False, True])
def test_json_alternate_duplicate_parameters_first_value(tok, encoded_arguments):
    arguments = '{"x":"first","x":"second","meta":{"nested":1,"nested":2}}'
    body = json.dumps(arguments) if encoded_arguments else arguments
    text = '<tool_call>{"name":"f","arguments":' + body + "}</tool_call>"
    request = _request()
    full = Qwen3CoderToolParser(tok, tools=request.tools).extract_tool_calls(
        text, request
    )
    args = full.tool_calls[0].function.arguments
    assert full.tool_calls[0].function.name == "f"
    assert json.loads(args) == {"x": "first", "meta": {"nested": 2}}
    _, events = replay_tools(tok, text, 5)
    assert events[-1][2] == args
