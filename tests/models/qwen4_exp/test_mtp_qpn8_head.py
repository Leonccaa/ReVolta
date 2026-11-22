# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import SpeculativeConfig
from vllm.model_executor.layers import logits_processor as lp
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.models.qwen4_exp.nvidia import mtp, sm70_mtp_head
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle


@pytest.mark.parametrize("enabled", [False, True])
def test_qpn8_draft_head_config_defaults_off(enabled):
    config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=4, draft_qpn8_head=enabled
    )
    assert config.draft_qpn8_head is enabled
    assert (
        SpeculativeConfig(method="ngram", num_speculative_tokens=4).draft_qpn8_head
        is False
    )


def _make_admitted_draft():
    spec = SimpleNamespace(
        draft_qpn8_head=True,
        method="mtp",
        num_speculative_tokens=5,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        use_local_argmax_reduction=True,
        parallel_drafting=False,
    )
    config = SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, pipeline_parallel_size=1
        ),
        model_config=SimpleNamespace(
            quantization="modelopt_fp4", enable_cumem_allocator=False
        ),
        weight_transfer_config=None,
        lora_config=None,
    )
    head = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    nn.Module.__init__(head)
    head.quant_method = UnquantizedEmbeddingMethod()
    head.num_embeddings = head.org_vocab_size = 248320
    # Simulate the CUDA allocation at the admission boundary only.
    head.weight = SimpleNamespace(
        shape=(62080, 2560), is_cuda=True, dtype=torch.float16
    )
    head.shard_indices = SimpleNamespace(
        num_org_elements=62080,
        org_vocab_start_index=62080,
        num_org_vocab_padding=0,
    )
    draft = mtp.Qwen4ExpMTP.__new__(mtp.Qwen4ExpMTP)
    nn.Module.__init__(draft)
    draft.vllm_config = config
    draft.config = SimpleNamespace(vocab_size=248320)
    draft.model = nn.Module()
    draft.lm_head = head
    draft.logits_processor = lp.LogitsProcessor(248320)
    return draft


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(mtp.current_platform, "is_device_capability", lambda *a: True)
    monkeypatch.setattr(mtp.envs, "VLLM_BATCH_INVARIANT", False)

    def prepare(weight):
        assert weight.shape == (62080, 2560)
        return torch.empty(1, dtype=torch.uint8), torch.ones(1, dtype=torch.float16)

    def gemm(logits, hidden, codes, scales, *params):
        assert hidden.is_contiguous() and params == (8, 2, True)
        logits.fill_(0)

    calls = []

    def top1(pairs, partial, logits, valid, start):
        assert valid == 62080 and start == 62080
        assert partial.shape == (len(pairs), 122, 2)
        pairs[:, 0] = 6
        pairs[:, 1] = start + 7
        calls.append(pairs)

    for name in ("fp8_qpn8_prepare_sm70", "fp8_qpn8_gemm_sm70_out"):
        monkeypatch.setattr(torch.ops._C, name, object(), raising=False)
    monkeypatch.setattr(
        torch.ops._C, "qwen38_mtp_local_top1_sm70_out", top1, raising=False
    )
    monkeypatch.setattr(sm70_mtp_head, "prepare_channel_qpn8_weight", prepare)
    monkeypatch.setattr(sm70_mtp_head._sm70_ops, "fp8_qpn8_gemm_sm70_out", gemm)
    return calls


@pytest.mark.parametrize("rows", [1, 8])
def test_mrv2_load_and_greedy_draft_send_the_native_pair(monkeypatch, native, rows):
    draft = _make_admitted_draft()
    shared_head = draft.lm_head
    target = SimpleNamespace(lm_head=shared_head)
    draft.lm_head = None

    def share(target_model, config):
        draft.lm_head = target_model.lm_head
        return draft

    monkeypatch.setattr(eagle, "load_eagle_model", share)
    monkeypatch.setattr(eagle, "get_layers_from_vllm_config", lambda *a: {})
    spec = draft.vllm_config.speculative_config
    runner = SimpleNamespace(
        vllm_config=draft.vllm_config,
        speculative_config=spec,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        method="mtp",
        use_local_argmax_reduction=True,
        _static_draft_vocab=None,
        _maybe_initialize_static_draft_vocab=lambda: None,
    )
    runner._validate_local_argmax_reduction = (
        lambda: eagle.EagleSpeculator._validate_local_argmax_reduction(runner)
    )
    monkeypatch.setattr(lp, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        lp.LogitsProcessor, "_maybe_custom_top1_argmax", lambda *a: None
    )

    def gather(pair, *, dim):
        assert pair is native[-1] and dim == -1
        assert pair.dtype == torch.float32 and pair.stride() == (2, 1)
        return torch.cat([pair * 0, pair, pair * 0, pair * 0], dim=-1)

    monkeypatch.setattr(lp, "tensor_model_parallel_all_gather", gather)
    eagle.EagleSpeculator.load_model(runner, target)
    assert target.lm_head is draft.lm_head is shared_head
    hidden = torch.zeros((rows, 2560), dtype=torch.float16)
    actual = eagle.EagleSpeculator._sample_draft(
        runner, hidden, torch.zeros(1), torch.zeros(1), torch.zeros(1), None
    )
    assert actual.dtype == torch.int64 and actual.tolist() == [62087] * rows
    assert len(native) == 1


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("speculative_config", "draft_qpn8_head", False),
        ("speculative_config", "draft_sample_method", "probabilistic"),
        ("speculative_config", "num_speculative_tokens", 3),
        ("model_config", "enable_cumem_allocator", True),
        ("parallel_config", "tensor_parallel_size", 2),
    ],
)
def test_unsupported_admission_retains_original_processor(native, group, field, value):
    draft = _make_admitted_draft()
    original = draft.logits_processor
    setattr(getattr(draft.vllm_config, group), field, value)
    draft.prepare_sm70_mtp_draft_head()
    assert draft.logits_processor is original and not native


def test_missing_cuda_op_retains_original_processor(monkeypatch, native):
    draft = _make_admitted_draft()
    original = draft.logits_processor
    monkeypatch.delattr(torch.ops._C, "qwen38_mtp_local_top1_sm70_out")
    draft.prepare_sm70_mtp_draft_head()
    assert draft.logits_processor is original and not native


def test_reload_preserves_packed_storage_and_m9_falls_back(monkeypatch, native):
    draft = _make_admitted_draft()
    draft.prepare_sm70_mtp_draft_head()
    processor = draft.logits_processor
    codes, scales = processor.codes, processor.scales
    monkeypatch.setattr(
        sm70_mtp_head,
        "prepare_channel_qpn8_weight",
        lambda weight: (torch.full_like(codes, 3), torch.full_like(scales, 4)),
    )
    draft.reload_sm70_mtp_draft_head()
    assert processor.codes is codes and codes.tolist() == [3]
    assert processor.scales is scales and scales.tolist() == [4]

    hidden = torch.zeros((9, 2560), dtype=torch.float16)
    monkeypatch.setattr(draft.lm_head, "maybe_get_sm70_lm_head_top1", lambda *a: None)
    monkeypatch.setattr(
        draft.lm_head.quant_method,
        "apply",
        lambda *a, **kw: torch.tensor([[0, 9, 1]], dtype=torch.float16).repeat(9, 1),
    )
    pairs = processor._get_local_top1_pair(draft.lm_head, hidden, None)
    assert pairs.tolist() == [[9, 62081]] * 9 and not native
