# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A smaller M-RoPE YaRN cache must keep every frequency and cached row."""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

# Qwen3.8-Flash-Next text rope, with a short original length to keep the
# 4x reference cache small on the CPU. The correction range depends on it,
# so the full-length case is checked separately for inv_freq only.
HEAD, ROT, THETA, FACTOR = 256, 64, 10_000_000, 2.0
SECTION = [11, 11, 10]


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def _mrope(original: int, positions_factor: int) -> MRotaryEmbedding:
    return MRotaryEmbedding(
        HEAD,
        ROT,
        original,
        THETA,
        True,
        torch.float16,
        mrope_section=SECTION,
        mrope_interleaved=True,
        scaling_factor=FACTOR,
        positions_factor=positions_factor,
    )


@pytest.mark.usefixtures("default_vllm_config")
def test_1x_cache_is_a_prefix_of_the_4x_cache():
    legacy, small = _mrope(8192, 4), _mrope(8192, 1)
    rows = int(8192 * FACTOR)
    assert legacy.cos_sin_cache.shape[0] == 4 * rows
    assert small.cos_sin_cache.shape[0] == rows
    assert small.cos_sin_cache.dtype == legacy.cos_sin_cache.dtype
    assert torch.equal(small.cos_sin_cache, legacy.cos_sin_cache[:rows])


def test_inv_freq_keeps_the_4x_correction_range_at_full_length():
    # Production: original_max_position_embeddings 262144. Only inv_freq is
    # compared here; building the 4x cache at this length is too large.
    original = 262_144
    legacy = MRotaryEmbedding.__new__(MRotaryEmbedding)
    small = MRotaryEmbedding.__new__(MRotaryEmbedding)
    for emb, factor in ((legacy, 4), (small, 1)):
        emb.scaling_factor = FACTOR
        emb.extrapolation_factor = 1
        emb.beta_fast, emb.beta_slow, emb.truncate = 32, 1, True
        emb.base, emb.rotary_dim = THETA, ROT
        emb.max_position_embeddings = original * factor
        emb.yarn_correction_max_position = original * 4
    assert torch.equal(
        small._compute_inv_freq(FACTOR), legacy._compute_inv_freq(FACTOR)
    )


@pytest.mark.usefixtures("default_vllm_config")
def test_get_rope_keys_by_positions_factor():
    params = {
        "rope_type": "yarn",
        "factor": FACTOR,
        "original_max_position_embeddings": 4096,
        "rope_theta": THETA,
        "partial_rotary_factor": ROT / HEAD,
        "mrope_section": SECTION,
        "mrope_interleaved": True,
    }
    _ROPE_DICT.clear()
    try:
        default = get_rope(HEAD, 8192, rope_parameters=params, dtype=torch.float16)
        small = get_rope(
            HEAD,
            8192,
            rope_parameters=params,
            dtype=torch.float16,
            mrope_positions_factor=1,
        )
        assert default is not small
        assert default.cos_sin_cache.shape[0] == 4 * small.cos_sin_cache.shape[0]
        assert (
            get_rope(HEAD, 8192, rope_parameters=params, dtype=torch.float16) is default
        )
    finally:
        _ROPE_DICT.clear()
