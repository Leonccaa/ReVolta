# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import numpy as np
import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _generate,
    _request,
    _serving,
    _stream,
    _tokenizer,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.outputs import LogprobsLists


class _Detokenizer(BaseIncrementalDetokenizer):
    def decode_next(self, token_id):
        return chr(token_id)


@pytest.mark.parametrize("api", ["chat", "completion"])
@pytest.mark.parametrize("stream", [False, True])
def test_stop_aligns_api_tokens_logprobs_and_usage(api, stream):
    ids = list(map(ord, "abcde"))
    request = EngineCoreRequest(
        request_id="test",
        external_req_id="test",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(
            stop=["cd"],
            logprobs=0,
            output_kind=RequestOutputKind.DELTA
            if stream
            else RequestOutputKind.FINAL_ONLY,
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    processor = OutputProcessor(None, log_stats=False)
    processor.add_request(request, prompt="x")
    processor.request_states["test"].detokenizer = _Detokenizer(request)
    output = processor.process_outputs(
        [
            EngineCoreOutput(
                request_id="test",
                new_token_ids=ids,
                new_logprobs=LogprobsLists(
                    np.array(ids).reshape(-1, 1),
                    np.full((5, 1), -0.5),
                    np.ones(5, dtype=int),
                ),
            )
        ]
    ).request_outputs[0]
    # LogprobsProcessor intentionally had no tokenizer; supply decoded strings
    # here so the API layer cannot hide a surplus row behind token rendering.
    for token_id, probs in zip(output.outputs[0].token_ids, output.outputs[0].logprobs):
        probs[token_id].decoded_token = chr(token_id)
    metadata = RequestResponseMetadata(request_id="test")
    if api == "chat":
        serving = _serving()
        serving.enable_auto_tools = False
        api_request = _request(
            stream=stream,
            tool_choice="none",
            logprobs=True,
            return_token_ids=True,
            stream_options={"include_usage": True} if stream else None,
        )
        if stream:
            chunks = _stream(serving, api_request, [output])
        else:
            response = asyncio.run(
                serving.chat_completion_full_generator(
                    api_request,
                    _generate([output]),
                    "test",
                    "test",
                    [],
                    _tokenizer(),
                    metadata,
                )
            )
    else:
        serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
        serving.enable_force_include_usage = False
        serving.enable_prompt_tokens_details = True
        serving.system_fingerprint = None
        serving.return_tokens_as_token_ids = False
        api_request = CompletionRequest(
            model="test",
            prompt="x",
            stream=stream,
            logprobs=0,
            return_token_ids=True,
            stream_options={"include_usage": True} if stream else None,
        )
        if stream:

            async def collect():
                async def indexed():
                    yield 0, output

                lines = [
                    line
                    async for line in serving.completion_stream_generator(
                        api_request,
                        [],
                        indexed(),
                        "test",
                        0,
                        "test",
                        1,
                        _tokenizer(),
                        metadata,
                    )
                ]
                return [
                    json.loads(line.removeprefix("data: "))
                    for line in lines
                    if "[DONE]" not in line
                ]

            chunks = asyncio.run(collect())
        else:
            response = serving.request_output_to_completion_response(
                [output],
                api_request,
                "test",
                0,
                "test",
                _tokenizer(),
                metadata,
            )
    if stream:
        choices = [choice for chunk in chunks for choice in chunk["choices"]]
        tokens = [tid for choice in choices for tid in choice.get("token_ids", [])]
        probs = [choice["logprobs"] for choice in choices if choice.get("logprobs")]
        assert (
            sum(len(p["content"] if api == "chat" else p["tokens"]) for p in probs) == 4
        )
        assert chunks[-1]["usage"]["completion_tokens"] == 4
        text = "".join(
            (c["delta"].get("content") or "") if api == "chat" else c["text"]
            for c in choices
        )
    else:
        choice = response.choices[0]
        tokens = choice.token_ids
        probs = choice.logprobs
        assert len(probs.content if api == "chat" else probs.tokens) == 4
        assert response.usage.completion_tokens == 4
        text = choice.message.content if api == "chat" else choice.text
    assert tokens == list(map(ord, "abcd"))
    assert text == "ab"
