# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP draft DCP geometry and workspaces derived from existing configuration."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.common.qsa_cache import (
    QSACompressedKeyCache,
    QSAKeyStateCache,
    is_mtp_draft_prefix,
    qsa_dcp_block_geometry,
    qsa_draft_dcp_sharded,
    qsa_index_dcp_sharded,
)
from vllm.models.qwen4_exp.nvidia import mtp as mtp_module
from vllm.models.qwen4_exp.nvidia.qsa import (
    Qwen4ExpQSAAttention,
    Qwen4ExpQSAMetadataBuilder,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import CircularBufferSpec

pytestmark = pytest.mark.skip_global_cleanup

TARGET = "model.layers.3.self_attn"
DRAFT = "mtp.layers.48.self_attn"


def _config(*, method="mtp", m5=True, long_context=True, dcp=2, block_size=1664):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size, cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp, tensor_parallel_size=4
        ),
        kernel_config=SimpleNamespace(
            qsa_dcp_index_shard=m5,
            long_context_kv=SimpleNamespace(enabled=long_context),
        ),
        speculative_config=SimpleNamespace(method=method) if method else None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        num_speculative_tokens=3,
    )


@pytest.mark.parametrize(
    ("method", "m5", "long_context", "dcp", "sharded"),
    [
        ("mtp", True, True, 2, True),
        ("mtp", False, True, 2, False),
        ("mtp", True, False, 2, False),
        ("mtp", False, False, 2, False),
        ("mtp", True, True, 1, False),
        ("dflash", True, True, 2, False),
        (None, True, True, 2, False),
    ],
)
def test_geometry_decision_table(method, m5, long_context, dcp, sharded):
    config = _config(method=method, m5=m5, long_context=long_context, dcp=dcp)
    assert qsa_draft_dcp_sharded(config) is sharded
    assert qsa_dcp_block_geometry(config, TARGET) == (1664 // dcp, 1664, True)
    assert qsa_dcp_block_geometry(config, DRAFT) == (
        1664 // dcp if sharded else 1664,
        1664,
        sharded,
    )
    if method == "mtp" and m5 and not long_context and dcp > 1:
        with pytest.raises(ValueError, match="does not support MTP"):
            qsa_index_dcp_sharded(config, TARGET, 4)
    else:
        assert qsa_index_dcp_sharded(config, TARGET, 4) is (m5 and dcp > 1)
    assert qsa_index_dcp_sharded(config, DRAFT, 4) is sharded

    for prefix in (TARGET, DRAFT):
        side = SimpleNamespace(
            prefix=prefix, compress_ratio=4, head_size=128, dtype=torch.float16
        )
        if prefix == TARGET and method == "mtp" and m5 and not long_context:
            continue
        spec = QSACompressedKeyCache.get_kv_cache_spec(side, config)
        index_sharded = m5 and dcp > 1 and (prefix == TARGET or sharded)
        assert spec.dcp_sharded is index_sharded
        assert spec.block_size == (1664 // dcp if index_sharded else 1664)
        assert spec.storage_block_size == spec.block_size // 4

    ring = QSAKeyStateCache.get_kv_cache_spec(
        SimpleNamespace(
            compress_ratio=4,
            cache_config=config.cache_config,
            head_size=128,
            dtype=torch.float16,
        ),
        config,
    )
    assert isinstance(ring, CircularBufferSpec)
    assert ring.block_size == 8 and not ring.dcp_sharded


def test_mtp_prefix_matches_a_complete_path_segment():
    assert is_mtp_draft_prefix(DRAFT)
    assert is_mtp_draft_prefix("model.mtp.layers.48.self_attn")
    assert not is_mtp_draft_prefix("model.mtp_projection.self_attn")


@pytest.mark.parametrize("prefix", [TARGET, DRAFT])
def test_sharded_index_requires_whole_compression_groups(prefix):
    with pytest.raises(ValueError, match="divisible"):
        qsa_index_dcp_sharded(_config(block_size=1668), prefix, 4)


@pytest.mark.parametrize(
    ("prefix", "long_context", "replicated"),
    [(DRAFT, True, False), (DRAFT, False, True), (TARGET, True, False)],
)
def test_main_builder_uses_group_metadata_when_sharded(
    prefix, long_context, replicated, monkeypatch
):
    monkeypatch.setattr(FlashAttentionMetadataBuilder, "__init__", lambda *a: None)
    metadata = SimpleNamespace(
        block_table=torch.tensor([[7, 2]], dtype=torch.int32),
        slot_mapping=torch.tensor([112, -1, 113, -1]),
    )
    monkeypatch.setattr(FlashAttentionMetadataBuilder, "build", lambda *a: metadata)
    builder = Qwen4ExpQSAMetadataBuilder(
        None, [prefix], _config(long_context=long_context), torch.device("cpu")
    )
    assert builder.replicated_draft is replicated
    if not replicated:
        assert builder.build(0, metadata) is metadata


@pytest.mark.parametrize("sharded", [True, False])
def test_predictor_allocates_independent_dcp_buffers(sharded, monkeypatch):
    config = _config(long_context=sharded)
    config.compilation_config = SimpleNamespace(mode=0)
    config.quant_config = None
    config.model_config = SimpleNamespace(
        dtype=torch.float16,
        hf_text_config=SimpleNamespace(
            vocab_size=32,
            hidden_size=16,
            hc_count=2,
            rms_norm_eps=1e-6,
            hc_lowrank=4,
            mtp_num_hidden_layers=1,
            num_hidden_layers=48,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            num_attention_heads=24,
            head_dim=256,
        ),
    )
    built = []

    class Layer(nn.Module):
        def __init__(self, cfg, layer_type, prefix, **buffers):
            super().__init__()
            built.append(buffers)
            # Exercise the production top-k allocation separately from DCP.
            self.self_attn = object.__new__(Qwen4ExpQSAAttention)
            nn.Module.__init__(self.self_attn)
            self.self_attn.indexer = SimpleNamespace(output_width=2051)
            self.self_attn._set_topk_indices_buffer(
                max_tokens=16, topk_indices_buffer=None
            )

    class Passthrough(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(mtp_module, "Qwen4ExpDecoderLayer", Layer)
    for name in (
        "ColumnParallelLinear",
        "VocabParallelEmbedding",
        "GemmaRMSNorm",
        "GatedResidual",
    ):
        monkeypatch.setattr(mtp_module, name, Passthrough)
    monkeypatch.setattr(mtp_module, "HyperConnectionConfig", lambda **kw: None)
    monkeypatch.setattr(
        mtp_module, "make_empty_intermediate_tensors_factory", lambda *a, **kw: None
    )
    monkeypatch.setattr(mtp_module, "_make_draft_vllm_config", lambda *a: config)
    monkeypatch.setattr(
        mtp_module, "set_current_vllm_config", lambda *a, **kw: contextlib.nullcontext()
    )
    predictor = mtp_module.Qwen4ExpMultiTokenPredictor(vllm_config=config, prefix="mtp")
    assert len(built) == 1
    if not sharded:
        assert all(buffer is None for buffer in built[0].values())
        return
    local = built[0]["dcp_local_indices_buffer"]
    output = built[0]["dcp_partial_output_buffer"]
    lse = built[0]["dcp_partial_lse_buffer"]
    assert local is predictor.dcp_local_indices_buffer
    assert output is predictor.dcp_partial_output_buffer
    assert lse is predictor.dcp_partial_lse_buffer
    assert local.shape == (16, 2051) and local.dtype == torch.int32
    assert output.shape == (16, 12, 256) and output.dtype == torch.float32
    assert lse.shape == (16, 12) and lse.dtype == torch.float32
    topk = predictor.layers[0].self_attn.topk_indices_buffer
    assert (
        len({t.untyped_storage().data_ptr() for t in (local, output, lse, topk)}) == 4
    )
    topk.fill_(7)
    local.fill_(3)
    assert torch.all(topk == 7)
