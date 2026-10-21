# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo

TOKENS = {
    "image_token_id": "<|image_pad|>",
    "video_token_id": "<|video_pad|>",
    "vision_start_token_id": "<|vision_start|>",
    "vision_end_token_id": "<|vision_end|>",
}


@pytest.mark.parametrize("field", TOKENS)
@pytest.mark.parametrize("missing", [False, True])
def test_mismatched_visual_token_ids_fail_at_initialization(field, missing):
    ids = dict(zip(TOKENS, range(10, 14)))
    vocab = {token: ids[name] for name, token in TOKENS.items()}
    if missing:
        del vocab[TOKENS[field]]
    else:
        vocab[TOKENS[field]] += 100
    ctx = SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: vocab))
    with (
        patch.object(
            Qwen3VLProcessingInfo, "get_hf_config", return_value=SimpleNamespace(**ids)
        ),
        pytest.raises(ValueError, match=field),
    ):
        Qwen3VLProcessingInfo(ctx)


def test_matching_visual_token_ids_initialize():
    ids = dict(zip(TOKENS, range(10, 14)))
    vocab = {token: ids[name] for name, token in TOKENS.items()}
    ctx = SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: vocab))
    with patch.object(
        Qwen3VLProcessingInfo, "get_hf_config", return_value=SimpleNamespace(**ids)
    ):
        assert Qwen3VLProcessingInfo(ctx).ctx is ctx
