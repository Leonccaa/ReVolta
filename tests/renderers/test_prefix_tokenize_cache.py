# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from vllm.renderers.base import BaseRenderer, _PrefixTokenizeCache
from vllm.renderers.params import TokenizeParams

DELIMITER = "<|im_start|>"


def make_tokenizer(*, lstrip: bool = False) -> PreTrainedTokenizerFast:
    vocab = {char: idx for idx, char in enumerate(sorted(ByteLevel.alphabet()))}
    backend = Tokenizer(BPE(vocab, []))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.add_special_tokens([AddedToken(DELIMITER, special=True, lstrip=lstrip)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, truncation_side="left")


@pytest.mark.parametrize("async_path", [False, True], ids=["sync", "async"])
def test_renderer_reuses_prefix_and_preserves_token_ids(async_path):
    tokenizer = make_tokenizer()
    with (
        patch.object(BaseRenderer, "__abstractmethods__", set()),
        patch(
            "vllm.renderers.base.mm_registry.supports_multimodal_inputs",
            return_value=False,
        ),
    ):
        renderer = BaseRenderer(
            SimpleNamespace(
                model_config=SimpleNamespace(renderer_num_workers=1),
                parallel_config=SimpleNamespace(_api_process_rank=0),
            ),
            tokenizer,
        )

    prefix = f"{DELIMITER}user\n" + "日志 ABC🙂\n" * 400
    first_turn = f"{prefix}{DELIMITER}assistant\n"
    next_turn = f"{first_turn}done\n{DELIMITER}tool\nnext result \t\n"
    rewritten = f"{prefix}{DELIMITER}assistant\nupdated\n{DELIMITER}tool\nnew"
    params = TokenizeParams(max_total_tokens=20_000, add_special_tokens=False)
    limited = TokenizeParams(max_total_tokens=20, add_special_tokens=False)
    truncated = TokenizeParams(
        max_total_tokens=20_000,
        add_special_tokens=False,
        truncate_prompt_tokens=20,
        truncation_side="right",
    )

    async def run():
        cache = renderer._prefix_tok_cache
        for text, options, expected_hits in (
            (first_turn, params, 0),
            (next_turn, params, 1),
            (rewritten, params, 2),
            (next_turn, limited, 3),
            (next_turn, truncated, 3),
        ):
            if async_path:
                result = await renderer._tokenize_prompt_async(
                    {"prompt": text}, options
                )
            else:
                result = renderer._tokenize_prompt({"prompt": text}, options)
            assert result["prompt_token_ids"] == tokenizer.encode(
                text, **options.get_encode_kwargs()
            )
            assert cache._entries.stat().hits == expected_hits

    try:
        asyncio.run(run())
    finally:
        renderer.shutdown()


def test_prefix_cache_excludes_whitespace_stripping_markers():
    cache = _PrefixTokenizeCache(make_tokenizer(lstrip=True))
    assert not cache.enabled


def test_prefix_cache_evicts_by_token_budget(monkeypatch):
    monkeypatch.setattr(_PrefixTokenizeCache, "_MAX_TOKENS", 3)
    cache = _PrefixTokenizeCache(make_tokenizer())
    cache._put(b"first", [1, 2])
    cache._put(b"second", [3, 4])
    cache._put(b"too_large", [0, 1, 2, 3])
    assert list(cache._entries) == [b"second"]
    assert cache._entries[b"second"].tolist() == [3, 4]


def test_chat_prefix_survives_eviction_during_tail_encode():
    tokenizer = make_tokenizer()
    other_tokenizer = make_tokenizer()
    cache = _PrefixTokenizeCache(tokenizer, max_entries=1)
    text = f"{DELIMITER}user\n" + "history " * 300 + f"{DELIMITER}assistant\n"
    other = text.replace("history", "rewrite")
    opts = {"add_special_tokens": False}
    cache.encode_chat_fragment(tokenizer.encode, text, **opts)
    [stored] = cache._entries
    tail_started, evicted = Event(), Event()

    def encode_tail(tail, **kwargs):
        assert tail == f"{DELIMITER}assistant\nnext"
        tail_started.set()
        assert evicted.wait(timeout=5)
        return tokenizer.encode(tail, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        live = executor.submit(
            cache.encode_chat_fragment, encode_tail, text + "next", **opts
        )
        try:
            assert tail_started.wait(timeout=5)
            cache.encode_chat_fragment(other_tokenizer.encode, other, **opts)
            assert stored not in cache._entries
        finally:
            evicted.set()
        assert live.result() == tokenizer.encode(text + "next", **opts)


def test_chat_prefix_delegates_marked_encode_options():
    tokenizer = make_tokenizer()
    cache = _PrefixTokenizeCache(tokenizer)
    text = f"{DELIMITER}user\n" + "history " * 300 + f"{DELIMITER}assistant\n"
    cache.encode_chat_fragment(tokenizer.encode, text, add_special_tokens=False)
    marked = {"add_special_tokens": False, "split_special_tokens": True}
    expected = tokenizer.encode(text, **marked)
    with patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode:
        assert cache.encode_chat_fragment(encode, text, **marked) == expected
        encode.assert_called_once_with(text, **marked)
    assert cache._entries.stat().hits == 0
