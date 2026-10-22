# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp asks for a block that a DFlash drafter cache can split into slots."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skip_global_cleanup


def _config(method: str | None = "dflash", draft_kv: str | None = None):
    # Flash-Next target: one 256-dim KV head per rank in FP8 (512 B/token).
    # incoai DFlash2 drafter: two 128-dim KV heads per rank (TP4 of 8).
    draft = SimpleNamespace(
        dtype=torch.float16,
        get_num_kv_heads=lambda parallel_config: 2,
        get_head_size=lambda: 128,
    )
    spec = None
    if method is not None:
        spec = SimpleNamespace(
            use_dflash=lambda: method == "dflash",
            kv_cache_dtype=draft_kv,
            draft_model_config=draft,
            draft_parallel_config=None,
        )
    return SimpleNamespace(
        speculative_config=spec,
        model_config=SimpleNamespace(
            dtype=torch.float16,
            get_num_kv_heads=lambda parallel_config: 1,
            get_head_size=lambda: 256,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
    )


@pytest.fixture
def sm70(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda capability: capability == 70
    )


@pytest.mark.parametrize(
    "model_cls", [Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration]
)
def test_fp16_drafter_on_fp8_target_needs_two_slot_halves(sm70, model_cls) -> None:
    # The SM70 drafter keeps an FP16 cache: 1,024 B/token against 512 B/token,
    # so its block is half the target block and the target block is 32-aligned.
    assert model_cls.get_kv_block_size_alignment(_config()) == 32


def test_drafter_with_the_target_cache_dtype_needs_no_extra_split(sm70) -> None:
    assert (
        Qwen4ExpForCausalLM.get_kv_block_size_alignment(_config(draft_kv="fp8_e4m3"))
        == 16
    )


@pytest.mark.parametrize("method", [None, "mtp"])
def test_no_alignment_without_dflash(method) -> None:
    assert Qwen4ExpForCausalLM.get_kv_block_size_alignment(_config(method)) is None
