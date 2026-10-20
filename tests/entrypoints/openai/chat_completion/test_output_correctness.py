# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from copy import copy
from unittest.mock import MagicMock

import numpy as np
import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    RequestResponseMetadata,
)
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.outputs import LogprobsLists


def _serving(parser_factory=None):
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.use_harmony = False
    serving.enable_auto_tools = True
    serving.tool_parser = Qwen3CoderToolParser
    serving.response_role = "assistant"
    serving.tool_call_id_type = "random"
    serving.parser_cls = parser_factory
    serving.enable_force_include_usage = False
    serving.enable_log_outputs = False
    serving.enable_prompt_tokens_details = True
    serving.system_fingerprint = None
    serving.return_tokens_as_token_ids = False
    return serving


def _output(text, ids, finish=None, logprobs=False, index=0):
    return RequestOutput(
        request_id="test",
        prompt="x",
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text=text,
                token_ids=ids,
                cumulative_logprob=-0.5 * len(ids),
                logprobs=[{tid: Logprob(-0.5, 1, text)} for tid in ids]
                if logprobs
                else None,
                finish_reason=finish,
            )
        ],
        finished=finish is not None,
    )


async def _generate(outputs):
    for output in outputs:
        yield output


def _request(**kwargs):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "call f"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
        ],
        **kwargs,
    )


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<think>": 10,
        "</think>": 11,
        "<tool_call>": 12,
        "</tool_call>": 13,
    }
    return tokenizer


def _stream(serving, request, outputs, tokenizer=None, chat_template_kwargs=None):
    async def collect():
        lines = [
            line
            async for line in serving.chat_completion_stream_generator(
                request,
                _generate(outputs),
                "test",
                "test",
                [],
                tokenizer or _tokenizer(),
                RequestResponseMetadata(request_id="test"),
                chat_template_kwargs=chat_template_kwargs,
            )
        ]
        assert lines[-1] == "data: [DONE]\n\n"
        chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        assert not any("error" in chunk for chunk in chunks), chunks
        return chunks

    return asyncio.run(collect())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize("finish", ["stop", "length", "repetition"])
def test_tool_finish_reason_preserves_truncation(stream, choice, finish, monkeypatch):
    # Exercise the real qwen3_coder parser and the required/named bypasses.
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", None)
    serving = _serving(_WrappedParser)
    tool_choice = (
        {"type": "function", "function": {"name": "f"}} if choice == "named" else choice
    )
    request = _request(stream=stream, tool_choice=tool_choice)
    if choice == "auto":
        text = "<tool_call>\n<function=f>\n<parameter=x>value"
        if finish == "stop":
            text += "</parameter>\n</function>\n</tool_call>"
    elif choice == "required":
        text = '[{"name":"f","parameters":{"x":"value'
        if finish == "stop":
            text += '"}}]'
    else:
        text = '{"x":"value' + ('"}' if finish == "stop" else "")
    output = _output(text, [30, 31], finish)
    if stream:
        if choice == "auto":
            pieces = ["<tool_call>\n", "<function=f>\n", "<parameter=x>value"]
            if finish == "stop":
                pieces.append("</parameter>\n</function>\n</tool_call>")
            outputs = [
                _output(
                    piece,
                    [12] if i == 0 else [30 + i],
                    finish if i == len(pieces) - 1 else None,
                )
                for i, piece in enumerate(pieces)
            ]
        else:
            outputs = [output]
        chunks = _stream(serving, request, outputs)
        reasons = [
            c["finish_reason"]
            for chunk in chunks
            for c in chunk["choices"]
            if c.get("finish_reason")
        ]
        assert any(
            c["delta"].get("tool_calls") for chunk in chunks for c in chunk["choices"]
        )
        actual = reasons[0]
    else:
        response = asyncio.run(
            serving.chat_completion_full_generator(
                request,
                _generate([output]),
                "test",
                "test",
                [],
                _tokenizer(),
                RequestResponseMetadata(request_id="test"),
            )
        )
        if choice != "required" or finish == "stop":
            assert response.choices[0].message.tool_calls
        actual = response.choices[0].finish_reason
    expected = "tool_calls" if finish == "stop" and choice != "named" else finish
    assert actual == expected


@pytest.mark.parametrize("return_token_ids", [False, True])
@pytest.mark.parametrize("include_reasoning", [False, True])
def test_parser_suppressed_logprobs_are_preserved(return_token_ids, include_reasoning):
    parser = MagicMock()
    parser.tool_parser = None
    parser.parse_delta.side_effect = [None, None, DeltaMessage(content="answer")]
    serving = _serving(MagicMock(return_value=parser))
    chunks = _stream(
        serving,
        _request(
            stream=True,
            logprobs=True,
            return_token_ids=return_token_ids,
            include_reasoning=include_reasoning,
        ),
        [
            _output("<think>", [10], logprobs=True),
            _output("</think>", [11], logprobs=True),
            _output("answer", [20], "stop", logprobs=True),
        ],
    )
    tokens = [
        lp["token"]
        for chunk in chunks
        for c in chunk["choices"]
        if c.get("logprobs")
        for lp in c["logprobs"]["content"]
    ]
    assert tokens == ["<think>", "</think>", "answer"]


@pytest.mark.parametrize("parser_kind", ["reasoning", "tool", "both"])
def test_real_qwen_parser_control_token_logprobs(parser_kind, monkeypatch):
    monkeypatch.setattr(
        _WrappedParser,
        "tool_parser_cls",
        Qwen3CoderToolParser if parser_kind in ("tool", "both") else None,
    )
    monkeypatch.setattr(
        _WrappedParser,
        "reasoning_parser_cls",
        Qwen3ReasoningParser if parser_kind in ("reasoning", "both") else None,
    )
    marker, token_id = ("<tool_call>", 12) if parser_kind == "tool" else ("<think>", 10)
    parser = _WrappedParser(_tokenizer())
    request = _request(stream=True, logprobs=True, tool_choice="auto")
    assert parser.parse_delta(marker, [token_id], request, prompt_token_ids=[1]) is None
    serving = _serving(_WrappedParser)
    chunks = _stream(
        serving,
        request,
        [
            _output(marker, [token_id], logprobs=True),
            _output("", [], "length", logprobs=True),
        ],
    )
    tokens = [
        lp["token"]
        for chunk in chunks
        for c in chunk["choices"]
        if c.get("logprobs")
        for lp in c["logprobs"]["content"]
    ]
    assert tokens == [marker]


def test_suppressed_logprobs_are_isolated_per_choice():
    parsers = [MagicMock(), MagicMock()]
    for index, parser in enumerate(parsers):
        parser.tool_parser = None
        parser.parse_delta.side_effect = [None, DeltaMessage(content=f"answer{index}")]
    serving = _serving(MagicMock(side_effect=parsers))
    first = _output("marker0", [10], logprobs=True)
    first.outputs += _output("marker1", [11], logprobs=True, index=1).outputs
    last = _output("answer0", [20], "stop", logprobs=True)
    last.outputs += _output("answer1", [21], "stop", logprobs=True, index=1).outputs
    chunks = _stream(serving, _request(stream=True, n=2, logprobs=True), [first, last])
    tokens: dict[int, list[str]] = {0: [], 1: []}
    for chunk in chunks:
        for choice in chunk["choices"]:
            if choice.get("logprobs"):
                tokens[choice["index"]].extend(
                    lp["token"] for lp in choice["logprobs"]["content"]
                )
    assert tokens == {0: ["marker0", "answer0"], 1: ["marker1", "answer1"]}


# Steps with empty delta text. The tests below do not hand-write the step
# texts: the sampled token ids go through the real OutputProcessor (incremental
# detokenizer, stop-string check and logprobs), one token per engine step, and
# its RequestOutputs feed the streaming generator. A character split across
# byte tokens and the stop-string look-back then leave some steps without text,
# which the Qwen3 parsers answer with no delta.

_STOP = "STOP-HERE"


def _byte_tokenizer():
    # Byte-level BPE without merges: every UTF-8 byte is its own token, so a
    # CJK character or an emoji spans three or four tokens, as rare characters
    # do in the Qwen vocabularies.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens(
        [
            AddedToken(token, special=False, normalized=False)
            for token in ("<think>", "</think>", "<tool_call>", "</tool_call>")
        ]
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def _encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _engine_outputs(request, tokenizer, prompt, completions):
    params = request.to_sampling_params(64, {})
    engine_request = EngineCoreRequest(
        request_id="test",
        external_req_id="test",
        prompt_token_ids=_encode(tokenizer, prompt),
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    processor = OutputProcessor(tokenizer, log_stats=False)
    queue = RequestOutputCollector(params.output_kind, "test")
    parent = ParentRequest(engine_request) if params.n > 1 else None
    sampled = {}
    for index, completion in enumerate(completions):
        child = copy(engine_request)
        if parent is not None:
            child.request_id, child.sampling_params = parent.get_child_info(index)
        processor.add_request(child, "x", parent, index, queue)
        sampled[child.request_id] = _encode(tokenizer, completion)
    outputs = []
    step = 0
    while processor.has_unfinished_requests():
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=[ids[step]],
                    new_logprobs=None
                    if params.logprobs is None
                    else LogprobsLists(
                        np.array([[ids[step]]]), np.array([[-0.5]]), np.array([1])
                    ),
                    # The last token hits max_tokens unless a stop string matched.
                    finish_reason=FinishReason.LENGTH if step == len(ids) - 1 else None,
                )
                for request_id, ids in sampled.items()
                if request_id in processor.request_states
            ]
        )
        outputs.append(queue.get_nowait())
        step += 1
    return outputs


def _qwen_stream(parser_kind, thinking, completions, monkeypatch, **kwargs):
    monkeypatch.setattr(
        _WrappedParser,
        "tool_parser_cls",
        Qwen3CoderToolParser if parser_kind in ("tool", "both") else None,
    )
    monkeypatch.setattr(
        _WrappedParser,
        "reasoning_parser_cls",
        Qwen3ReasoningParser if parser_kind in ("reasoning", "both") else None,
    )
    tokenizer = _byte_tokenizer()
    kwargs.setdefault("logprobs", True)
    request = _request(
        stream=True, tool_choice="auto", return_tokens_as_token_ids=True, **kwargs
    )
    # Qwen3.5 templates open <think> in the prompt; with thinking disabled
    # they close it there as well.
    prompt = "q<think>\n" if thinking else "q<think>\n\n</think>\n\n"
    outputs = _engine_outputs(request, tokenizer, prompt, completions)
    chunks = _stream(
        _serving(_WrappedParser),
        request,
        outputs,
        tokenizer,
        chat_template_kwargs=None if thinking else {"enable_thinking": False},
    )
    return chunks, outputs, [_encode(tokenizer, c) for c in completions]


def _choices(chunks):
    choices: dict[int, list[dict]] = {}
    for chunk in chunks:
        for choice in chunk["choices"]:
            if "role" not in choice["delta"]:
                choices.setdefault(choice["index"], []).append(choice)
    return choices


def _logprob_ids(choice):
    return [
        int(lp["token"].removeprefix("token_id:"))
        for lp in choice["logprobs"]["content"]
    ]


def _text(choices, field):
    return "".join(choice["delta"].get(field) or "" for choice in choices)


def _tool_calls(choices):
    calls = [tc for choice in choices for tc in choice["delta"].get("tool_calls", [])]
    names = [tc["function"]["name"] for tc in calls if tc["function"].get("name")]
    arguments = "".join(tc["function"].get("arguments") or "" for tc in calls)
    return names, json.loads(arguments) if arguments else None


def _assert_empty_steps_keep_logprobs(
    chunks, outputs, sampled, finish_reasons, return_token_ids=False
):
    steps: dict[int, list[CompletionOutput]] = {}
    for output in outputs:
        for completion in output.outputs:
            steps.setdefault(completion.index, []).append(completion)
    choices = _choices(chunks)
    assert choices.keys() == steps.keys() == set(range(len(sampled)))
    for index, ids in enumerate(sampled):
        assert any(not step.text for step in steps[index])
        # One chunk per engine step, carrying exactly that step's tokens, so
        # every sampled token is streamed once and in order.
        assert [_logprob_ids(c) for c in choices[index]] == [
            list(step.token_ids) for step in steps[index]
        ]
        assert [i for c in choices[index] for i in _logprob_ids(c)] == ids
        for choice, step in zip(choices[index], steps[index]):
            if not step.text:
                assert not choice["delta"].get("content")
                assert not choice["delta"].get("reasoning")
                assert not choice["delta"].get("tool_calls")
            if return_token_ids:
                assert choice["token_ids"] == list(step.token_ids)
            else:
                assert choice.get("token_ids") is None
        assert [c.get("finish_reason") for c in choices[index]] == [None] * (
            len(steps[index]) - 1
        ) + [finish_reasons[index]]
    return choices


@pytest.mark.parametrize("include_reasoning", [False, True])
@pytest.mark.parametrize("return_token_ids", [False, True])
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("parser_kind", ["reasoning", "both"])
def test_split_character_logprobs_are_preserved(
    parser_kind, thinking, return_token_ids, include_reasoning, monkeypatch
):
    # The detokenizer returns no text until the last byte of a character.
    completion = "想🙂</think>\n\n好😀" if thinking else "好😀"
    chunks, outputs, sampled = _qwen_stream(
        parser_kind,
        thinking,
        [completion],
        monkeypatch,
        return_token_ids=return_token_ids,
        include_reasoning=include_reasoning,
    )
    choices = _assert_empty_steps_keep_logprobs(
        chunks, outputs, sampled, ["length"], return_token_ids
    )
    assert _text(choices[0], "reasoning") == ("想🙂" if thinking else "")
    assert _text(choices[0], "content") == ("\n\n好😀" if thinking else "好😀")


@pytest.mark.parametrize(
    ("parser_kind", "thinking", "stop"),
    [
        ("tool", False, None),
        ("tool", False, [_STOP]),
        ("both", False, None),
        ("both", False, [_STOP]),
        ("both", True, None),
    ],
)
def test_split_character_tool_call_logprobs_are_preserved(
    parser_kind, thinking, stop, monkeypatch
):
    call = (
        "<tool_call>\n<function=f>\n<parameter=x>\n值🙂\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    completion = ("想</think>\n\n" if thinking else "") + call + (_STOP if stop else "")
    chunks, outputs, sampled = _qwen_stream(
        parser_kind, thinking, [completion], monkeypatch, stop=stop
    )
    choices = _assert_empty_steps_keep_logprobs(
        chunks, outputs, sampled, ["tool_calls" if stop else "length"]
    )
    assert _tool_calls(choices[0]) == (["f"], {"x": "值🙂"})
    assert _text(choices[0], "reasoning") == ("想" if thinking else "")
    assert _text(choices[0], "content") == ("\n\n" if thinking else "")


@pytest.mark.parametrize("include_reasoning", [False, True])
@pytest.mark.parametrize("return_token_ids", [False, True])
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("parser_kind", ["reasoning", "both"])
def test_stop_string_lookback_logprobs_are_preserved(
    parser_kind, thinking, return_token_ids, include_reasoning, monkeypatch
):
    # The detokenizer holds back len(stop) - 1 characters, so the first eight
    # steps stream no text.
    completion = ("planning</think>\n\n" if thinking else "") + f"answer {_STOP}"
    chunks, outputs, sampled = _qwen_stream(
        parser_kind,
        thinking,
        [completion],
        monkeypatch,
        stop=[_STOP],
        return_token_ids=return_token_ids,
        include_reasoning=include_reasoning,
    )
    choices = _assert_empty_steps_keep_logprobs(
        chunks, outputs, sampled, ["stop"], return_token_ids
    )
    held_back = len(_STOP) - 1
    assert [step.outputs[0].text for step in outputs[:held_back]] == [""] * held_back
    # With thinking on, the text held back when </think> is sampled arrives in
    # later steps, which the reasoning parser does not split correctly; only
    # the logprobs are checked then.
    if not thinking:
        assert _text(choices[0], "content") == "answer "


@pytest.mark.parametrize("stop", [None, [_STOP]])
def test_empty_step_logprobs_are_isolated_per_choice(stop, monkeypatch):
    if stop:
        thinking, completions = False, [f"好😀 {_STOP}", "🙂ok"]
        finish_reasons = ["stop", "length"]
        reasoning, content = ["", ""], ["好😀 ", "🙂ok"]
    else:
        thinking, completions = True, ["想🙂</think>\n\n好😀", "🙂</think>\n\nok"]
        finish_reasons = ["length", "length"]
        reasoning, content = ["想🙂", "🙂"], ["\n\n好😀", "\n\nok"]
    chunks, outputs, sampled = _qwen_stream(
        "both", thinking, completions, monkeypatch, n=2, stop=stop
    )
    choices = _assert_empty_steps_keep_logprobs(
        chunks, outputs, sampled, finish_reasons
    )
    for index in (0, 1):
        assert _text(choices[index], "reasoning") == reasoning[index]
        assert _text(choices[index], "content") == content[index]


@pytest.mark.parametrize(
    ("parser_kind", "thinking", "stop"),
    [
        ("reasoning", True, None),
        ("both", True, None),
        ("both", False, None),
        ("both", False, [_STOP]),
    ],
)
def test_empty_steps_without_logprobs_are_skipped(
    parser_kind, thinking, stop, monkeypatch
):
    completion = "想🙂</think>\n\n好😀" if thinking else "好😀"
    if stop:
        completion += f" {_STOP}"
    streams = {}
    for logprobs in (False, True):
        chunks, _, _ = _qwen_stream(
            parser_kind,
            thinking,
            [completion],
            monkeypatch,
            logprobs=logprobs,
            stop=stop,
        )
        streams[logprobs] = [
            {k: v for k, v in choice.items() if k != "logprobs"}
            for chunk in chunks
            for choice in chunk["choices"]
        ]
    # Without logprobs, a step the parser suppresses sends no chunk. With
    # logprobs, it sends an empty delta and nothing else changes.
    assert all(c["delta"] or c.get("finish_reason") for c in streams[False])
    kept = [c for c in streams[True] if c["delta"] or c.get("finish_reason")]
    assert streams[False] == kept
    assert len(kept) < len(streams[True])
