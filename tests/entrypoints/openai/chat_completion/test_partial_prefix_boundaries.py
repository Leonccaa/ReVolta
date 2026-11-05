# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End to end through OpenAIServingChat with ``--partial-prefix-cache``.

Every chat request must reach the engine whatever the client puts in
``vllm_xargs``; derived boundaries (end of the system prompt and of the last
message) are passed on together with the valid client ones. Uses a local
word-level tokenizer, so it runs offline.
"""

import json
from contextlib import suppress
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import (
    OpenAIServingChat,
    _with_prefix_cache_boundaries,
)
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.renderers.hf import HfRenderer
from vllm.sampling_params import SamplingParams
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.request import MAX_PREFIX_CACHE_BOUNDARIES, Request

MODEL = "partial-prefix-test"
CHAT_TEMPLATE = (
    "{% for m in messages %}<s> {{ m['role'] }} {{ m['content'] }} </s> "
    "{% endfor %}{% if add_generation_prompt %}<s> assistant {% endif %}"
)
WORDS = [
    "<s>",
    "</s>",
    "<unk>",
    "system",
    "user",
    "assistant",
    "be",
    "brief",
    "please",
    "hello",
    "there",
    "what",
    "is",
    "one",
    "plus",
    "two",
]


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    model = tokenizers.models.WordLevel(
        {word: i for i, word in enumerate(WORDS)}, unk_token="<unk>"
    )
    backend = tokenizers.Tokenizer(model)
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        chat_template=CHAT_TEMPLATE,
    )
    path = tmp_path_factory.mktemp("tokenizer")
    tokenizer.save_pretrained(path)
    (path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    return str(path)


@dataclass
class _HFConfig:
    model_type: str = "any"


@dataclass
class _ModelConfig:
    model: str
    tokenizer: str
    task = "generate"
    runner_type = "generate"
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 4096
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = _HFConfig()
    hf_text_config = _HFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _serving_chat(tokenizer_dir: str, switch_on: bool):
    model_config = _ModelConfig(model=MODEL, tokenizer=tokenizer_dir)
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = model_config
    engine.input_processor = MagicMock()
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(partial_prefix_cache=switch_on),
        kv_transfer_config=None,
    )
    engine.renderer = HfRenderer(
        SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(_api_process_rank=0),
        ),
        cached_tokenizer_from_config(model_config),
    )
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=MODEL, model_path=MODEL)],
    )
    render = OpenAIServingRender(
        model_config=model_config,
        renderer=engine.renderer,
        model_registry=models.registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        openai_serving_render=render,
        chat_template=None,
        chat_template_content_format="auto",
        request_logger=None,
    )
    return serving, engine


MESSAGES = [
    {"role": "system", "content": "be brief please"},
    {"role": "user", "content": "what is one plus two"},
]
# Token positions under the template: system ends after 6 tokens, the last
# message after 6 + 8, and the prompt adds "<s> assistant".
SYSTEM_END, LAST_MESSAGE_END, PROMPT_LEN = 6, 14, 16


async def _engine_call(serving, engine, **kwargs):
    engine.generate.reset_mock()
    request = ChatCompletionRequest(model=MODEL, messages=MESSAGES, **kwargs)
    with suppress(Exception):
        # The mock engine yields nothing; the request has been handed over.
        await serving.create_chat_completion(request)
    assert engine.generate.call_count == 1, "request did not reach the engine"
    prompt, params = engine.generate.call_args.args[:2]
    return prompt, params


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "xargs,expected",
    [
        # Plain request (finding 1: this used to fail with int(())).
        (None, [SYSTEM_END, LAST_MESSAGE_END]),
        ({"other": 1}, [SYSTEM_END, LAST_MESSAGE_END]),
        ({"prefix_cache_boundaries": []}, [SYSTEM_END, LAST_MESSAGE_END]),
        ({"prefix_cache_boundaries": [3, 9]}, [3, SYSTEM_END, 9, LAST_MESSAGE_END]),
        ({"prefix_cache_boundaries": ["x", -1, 10**6, 1e309, 2.5]}, None),
    ],
)
async def test_switch_on_every_request_reaches_the_engine(
    tokenizer_dir, xargs, expected
):
    serving, engine = _serving_chat(tokenizer_dir, switch_on=True)
    kwargs = {} if xargs is None else {"vllm_xargs": xargs}
    prompt, params = await _engine_call(serving, engine, **kwargs)
    token_ids = prompt["prompt_token_ids"]
    assert len(token_ids) == PROMPT_LEN
    boundaries = params.extra_args["prefix_cache_boundaries"]
    assert boundaries == (expected or [SYSTEM_END, LAST_MESSAGE_END])
    if xargs and "other" in xargs:
        assert params.extra_args["other"] == 1


@pytest.mark.asyncio
async def test_switch_off_leaves_requests_untouched(tokenizer_dir):
    serving, engine = _serving_chat(tokenizer_dir, switch_on=False)
    _, params = await _engine_call(serving, engine)
    assert not params.extra_args
    _, params = await _engine_call(
        serving, engine, vllm_xargs={"prefix_cache_boundaries": [1e309]}
    )
    # Passed through unparsed; the engine never reads it with the switch off.
    assert params.extra_args == {"prefix_cache_boundaries": [1e309]}


@pytest.mark.parametrize(
    "extra_args",
    [
        None,
        {},
        {"x": 1},
        {"prefix_cache_boundaries": None},
        {"prefix_cache_boundaries": 5},
    ],
)
def test_merge_handles_absent_and_scalar_client_boundaries(extra_args):
    merged = _with_prefix_cache_boundaries(extra_args, [7, 12], 20)
    expected = (
        [5, 7, 12]
        if extra_args and extra_args.get("prefix_cache_boundaries") == 5
        else [7, 12]
    )
    assert merged["prefix_cache_boundaries"] == expected
    if extra_args and "x" in extra_args:
        assert merged["x"] == 1


def test_merge_keeps_derived_boundaries_within_the_engine_limit():
    # A client can fill the whole allowance; the engine reads only the first
    # MAX_PREFIX_CACHE_BOUNDARIES values of the merged list.
    client = list(range(64, 64 + MAX_PREFIX_CACHE_BOUNDARIES))
    merged = _with_prefix_cache_boundaries(
        {"prefix_cache_boundaries": client}, [576, 640], 700
    )["prefix_cache_boundaries"]
    assert len(merged) == MAX_PREFIX_CACHE_BOUNDARIES
    assert {576, 640} <= set(merged)
    assert merged == sorted(merged)
    request = Request(
        request_id="r",
        prompt_token_ids=list(range(700)),
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"prefix_cache_boundaries": merged}
        ),
        pooling_params=None,
    )
    assert {576, 640} <= set(request.prefix_cache_boundaries)
