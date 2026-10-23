# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import ModelRegistry
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import MambaSpec

# FP16, one KV head of size 64: 2 * 64 * 2 bytes of K and V per token.
ATTN_BYTES_PER_TOKEN = 256
KERNEL_BLOCK = 16


class _LooseBackend:
    """A backend that accepts smaller kernel blocks than the sharded layers."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(8)]


def _vllm_config(mamba_tokens: int, dcp: int):
    class _Model:
        @classmethod
        def get_mamba_specs_from_config(cls, vllm_config):
            elements = mamba_tokens * ATTN_BYTES_PER_TOKEN // 2
            return (
                MambaSpec(
                    shapes=((elements,),), dtypes=(torch.float16,), block_size=-1
                ),
            )

        @classmethod
        def get_kv_block_size_multiple(cls, vllm_config):
            return dcp

    model_config = SimpleNamespace(
        use_mla=False,
        dtype=torch.float16,
        architecture="FakeHybridForCausalLM",
        get_num_kv_heads=lambda parallel_config: 1,
        get_head_size=lambda: 64,
        hf_text_config=SimpleNamespace(),
    )
    cache_config = SimpleNamespace(
        cache_dtype="auto",
        block_size=KERNEL_BLOCK,
        mamba_cache_mode="align",
        mamba_block_size=None,
        user_specified_mamba_block_size=False,
        mamba_page_size_padded=None,
    )
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
    )
    return config, _Model


# The mamba page spans 1,610 attention tokens (MTP4-like, where a 16-token
# alignment yields 1,616) or 1,590 tokens (MTP3-like, 1,600 either way).
@pytest.mark.parametrize("mamba_tokens", [1610, 1590])
@pytest.mark.parametrize("dcp", [1, 2])
def test_dcp_share_keeps_the_block_alignment(monkeypatch, mamba_tokens, dcp) -> None:
    config, model_cls = _vllm_config(mamba_tokens, dcp)
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda *args, **kwargs: (model_cls, "FakeHybridForCausalLM"),
    )

    Platform._align_hybrid_block_size(config, _LooseBackend)

    block_size = config.cache_config.block_size
    assert block_size % dcp == 0
    assert (block_size // dcp) % KERNEL_BLOCK == 0, block_size
    assert block_size * ATTN_BYTES_PER_TOKEN >= mamba_tokens * ATTN_BYTES_PER_TOKEN
    # No larger than the next aligned size: existing geometries do not move.
    assert block_size - KERNEL_BLOCK * dcp < mamba_tokens
    assert config.cache_config.mamba_block_size == block_size


def _with_alignment(model_cls, align):
    return type(
        "_Aligned",
        (model_cls,),
        {"get_kv_block_size_alignment": classmethod(lambda cls, vllm_config: align)},
    )


# A DFlash drafter that needs a 32-token block multiple (two main-KV tokens of
# bytes per drafter token): MTP4-like and seven-token-draft-like mamba pages.
@pytest.mark.parametrize(
    ("mamba_tokens", "dcp", "expected"),
    [(1610, 1, 1632), (1636, 1, 1664), (1610, 2, 1632), (1636, 2, 1664)],
)
def test_model_alignment_rounds_the_automatic_block(
    monkeypatch, mamba_tokens, dcp, expected
) -> None:
    config, model_cls = _vllm_config(mamba_tokens, dcp)
    aligned = _with_alignment(model_cls, 32)
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda *args, **kwargs: (aligned, "FakeHybridForCausalLM"),
    )

    Platform._align_hybrid_block_size(config, _LooseBackend)

    assert config.cache_config.block_size == expected
    assert config.cache_config.mamba_block_size == expected


@pytest.mark.parametrize("align", [None, 32])
def test_model_alignment_keeps_an_explicit_block(monkeypatch, align) -> None:
    config, model_cls = _vllm_config(1610, 1)
    config.cache_config.block_size = 1632
    aligned = _with_alignment(model_cls, align)
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_model_cls",
        lambda *args, **kwargs: (aligned, "FakeHybridForCausalLM"),
    )

    Platform._align_hybrid_block_size(config, _LooseBackend)

    assert config.cache_config.block_size == 1632
