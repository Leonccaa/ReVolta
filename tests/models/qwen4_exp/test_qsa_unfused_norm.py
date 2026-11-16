# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise both unfused indexer norms on CPU without FlashInfer kernels."""

import sys
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.models.qwen4_exp.nvidia import indexer_qsa
from vllm.models.qwen4_exp.nvidia.ops import qsa


@pytest.mark.parametrize("head_dim", [8, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_unfused_query_and_pooled_key_norm_match_reference(
    monkeypatch, head_dim, dtype
):
    generator = torch.Generator().manual_seed(37)
    rows, heads = 4, 2
    projected = torch.randn(
        rows, (heads + 1) * head_dim, generator=generator, dtype=dtype
    )
    # Include a zero row to exercise epsilon, and nonzero, nonuniform weights
    # to distinguish Gemma's (1 + weight) from ordinary RMSNorm.
    projected[0].zero_()
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        q_norm = GemmaRMSNorm(head_dim, eps=1e-6).to(dtype)
        k_norm = GemmaRMSNorm(head_dim, eps=1e-5).to(dtype)
    with torch.no_grad():
        q_norm.weight.copy_(torch.linspace(-0.7, 0.8, head_dim))
        k_norm.weight.copy_(torch.linspace(0.5, -0.4, head_dim))
    raw_keys = projected[:, heads * head_dim :].reshape(rows, 1, head_dim)
    pooled = raw_keys.reshape(2, 2, 1, head_dim).float().mean(1).to(dtype)
    raw_metadata = NS(
        num_actual_tokens=rows,
        logical_positions=torch.arange(rows),
        slot_mapping=torch.arange(rows),
        block_table=None,
        token_to_req=None,
        query_start_loc=None,
    )
    compressed_metadata = NS(slot_mapping=torch.arange(2))
    raw_cache, compressed_cache = torch.empty(0), torch.empty(0)
    indexer = NS(
        _metadata=lambda: (raw_metadata, compressed_metadata),
        index_qk_proj=lambda _: (projected, None),
        index_n_heads=heads,
        index_kv_heads=1,
        index_head_dim=head_dim,
        use_fused_pre_indexer=False,
        q_layernorm=q_norm,
        k_layernorm=k_norm,
        rotary_emb=NS(),
        raw_key_cache=NS(key_cache=raw_cache, rope_position_cache=None),
        compressed_key_cache=NS(kv_cache=compressed_cache),
        compress_ratio=2,
        skip_topk=True,
    )
    normalized = []
    stored = []

    def record_rope(_rope, _positions, tensor):
        normalized.append(tensor)
        return tensor

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.norm",
        NS(gemma_rmsnorm=lambda *a: pytest.fail("CuTe RMSNorm is unsupported")),
    )
    monkeypatch.setattr(indexer_qsa, "apply_qsa_rope", record_rope)
    monkeypatch.setattr(
        qsa,
        "qsa_compress_groups_with_ratio",
        lambda *a: (pooled, torch.tensor([[0, 0, 0], [2, 2, 2]])),
    )
    monkeypatch.setattr(qsa, "qsa_store_cache_rows", lambda *args: stored.append(args))
    out = torch.empty(0, dtype=torch.int32)
    assert (
        indexer_qsa.QSAIndexer.forward(
            indexer, torch.empty(rows, 1), torch.arange(rows), out
        )
        is out
    )

    def reference(x, norm):
        x = x.double()
        return (
            x
            / (x.square().mean(-1, keepdim=True) + norm.variance_epsilon).sqrt()
            * (1 + norm.weight.double())
        ).to(dtype)

    queries = projected[:, : heads * head_dim].reshape(rows, heads, head_dim)
    torch.testing.assert_close(normalized[0], reference(queries, q_norm))
    torch.testing.assert_close(normalized[1], reference(pooled, k_norm))
    assert stored[0][0] is compressed_cache
    assert stored[1][0] is raw_cache
    torch.testing.assert_close(stored[0][2], normalized[1])
    torch.testing.assert_close(stored[1][2], raw_keys.reshape(rows, head_dim))


def test_production_shape_retains_fused_pre_indexer():
    rope = NS(
        rotary_dim=64,
        is_neox_style=True,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    )
    assert indexer_qsa._supports_fused_pre_indexer(rope, 128, 1, 4)
    assert not indexer_qsa._supports_fused_pre_indexer(rope, 64, 1, 4)
