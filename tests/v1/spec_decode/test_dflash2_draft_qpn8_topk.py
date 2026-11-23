# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SM70 DFlash2 draft pure-QPN8 top-k and native FP16 top-k wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.model_executor.layers.sm70_compact_topk import (
    maybe_sm70_compact_half_topk,
    sm70_compact_half_topk_ids_partial_spec,
    sm70_compact_half_topk_partial_spec,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    _maybe_sm70_dflash2_draft_qpn8_topk,
    _prepare_sm70_dflash2_draft_qpn8,
    _sm70_dflash2_draft_qpn8_enabled,
)
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM


def test_sm70_dflash2_draft_qpn8_policy_default_and_graph_options(monkeypatch):
    env_name = "VLLM_SM70_DFLASH2_DRAFT_QPN8_TOPK"
    monkeypatch.delenv(env_name, raising=False)
    default_cfg = Sm70DFlash2Config()
    assert default_cfg.draft_qpn8_topk is False
    assert "draft_qpn8_topk" not in default_cfg.graph_options()

    opt_in_cfg = Sm70DFlash2Config(draft_qpn8_topk=True)
    assert opt_in_cfg.draft_qpn8_topk is True
    assert opt_in_cfg.graph_options()["draft_qpn8_topk"] is True

    monkeypatch.setenv(env_name, "1")
    for policy, enabled in (
        (Sm70DFlash2Config(), True),
        (Sm70DFlash2Config(draft_qpn8_topk=False), False),
    ):
        layer = SimpleNamespace(_sm70_dflash2_policy=policy)
        assert _sm70_dflash2_draft_qpn8_enabled(layer) is enabled
        assert policy.graph_options().get("draft_qpn8_topk", False) is enabled


def test_compact_half_topk_partial_specs():
    shape, dtype = sm70_compact_half_topk_partial_spec(8, 62080, 20)
    assert shape == (8, 61 * 20)
    assert dtype == torch.uint32

    shape, dtype = sm70_compact_half_topk_partial_spec(8, 248320, 20)
    assert shape == (8, 243 * 20 + 5 * 20)
    assert dtype == torch.int64

    shape, dtype = sm70_compact_half_topk_ids_partial_spec(8, 62080, 64)
    assert shape == (8, 61 * 64)
    assert dtype == torch.uint32


def test_maybe_sm70_compact_half_topk_rejects_cpu():
    x = torch.randn(4, 128, dtype=torch.float16)
    assert maybe_sm70_compact_half_topk(x, 20) is None


def test_compute_candidates_prefers_draft_qpn8_topk(monkeypatch):
    calls: list[str] = []

    class DummyHead:
        quant_method = UnquantizedEmbeddingMethod()
        shard_indices = SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=0
        )

        def maybe_get_sm70_dflash2_draft_topk(self, hidden_states, k):
            calls.append(f"draft_topk:{k}")
            vals = (
                torch.linspace(20.0, 1.0, k)
                .unsqueeze(0)
                .expand(hidden_states.shape[0], k)
            )
            ids = (
                torch.arange(k, dtype=torch.int64)
                .unsqueeze(0)
                .expand(hidden_states.shape[0], k)
            )
            return vals.clone(), ids.clone()

        def maybe_get_sm70_dflash2_top20(self, hidden_states, k):
            calls.append(f"rerank_top20:{k}")
            return None

    dummy = SimpleNamespace(
        lm_head=DummyHead(),
        model=SimpleNamespace(candidate_selector=SimpleNamespace(top_k=20)),
        _sm70_dflash2_policy=Sm70DFlash2Config(draft_qpn8_topk=True),
        output_multiplier=1.0,
        final_logit_softcapping=None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash2.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    x = torch.randn(8, 2560, dtype=torch.float16)
    ids, vals = DFlash2Qwen3ForCausalLM.compute_candidates(dummy, x)
    assert calls == ["draft_topk:20"]
    assert ids.shape == (8, 20)
    assert vals.shape == (8, 20)
    assert vals.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)
@pytest.mark.parametrize(
    "rows,hidden,padding,vocab_start",
    [
        (8, 2560, 0, 0),
        (16, 2560, 0, 62080),
        (16, 2560, 64, 124160),
        (16, 5120, 64, 124160),
    ],
)
@torch.inference_mode()
def test_sm70_draft_qpn8_topk_accuracy_and_graph(rows, hidden, padding, vocab_start):
    if not (
        hasattr(torch.ops._C, "fp8_qpn8_prepare_sm70")
        and hasattr(torch.ops._C, "fp8_qpn8_gemm_sm70_out")
        and hasattr(torch.ops._C, "sm70_compact_half_topk_wide_revision")
        and torch.ops._C.sm70_compact_half_topk_wide_revision() >= 2
    ):
        pytest.skip("requires SM70 QPN8 and compact half top-k revision 2")

    torch.manual_seed(2026)
    vocab = 62080
    layer = torch.nn.Module()
    weight = torch.randn(vocab, hidden, dtype=torch.float16, device="cuda") * 0.04
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.shard_indices = SimpleNamespace(
        num_org_vocab_padding=padding,
        org_vocab_start_index=vocab_start,
    )
    layer._sm70_dflash2_policy = Sm70DFlash2Config(draft_qpn8_topk=True)

    assert _prepare_sm70_dflash2_draft_qpn8(layer) is True

    x = torch.randn(rows, hidden, dtype=torch.float16, device="cuda")
    # Warmup and CUDA Graph capture
    out_eager = _maybe_sm70_dflash2_draft_qpn8_topk(layer, x, 20)
    assert out_eager is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = _maybe_sm70_dflash2_draft_qpn8_topk(layer, x, 20)
    assert out_graph is not None
    graph.replay()
    vals, ids = out_graph
    assert vals.shape == (rows, 20)
    assert ids.shape == (rows, 20)
    assert ids.dtype == torch.int64
    assert int(ids.min().item()) >= vocab_start
    assert int(ids.max().item()) < vocab_start + vocab - padding
    assert torch.equal(ids, out_eager[1])
    assert torch.equal(vals.view(torch.int32), out_eager[0].view(torch.int32))

    # Compare against dense FP16 reference on valid vocab
    ref_logits = torch.mm(x, weight[: vocab - padding].t()).float()
    ref_ids = torch.topk(ref_logits, 20, dim=-1).indices
    ref_ids = ref_ids + vocab_start
    # Top-1 match and top-20 recall should be very high with E4M3 channel scales
    top1_match = (ids[:, 0] == ref_ids[:, 0]).float().mean().item()
    recall_20 = (
        (ref_ids[:, :, None] == ids[:, None, :]).any(dim=-1).float().mean().item()
    )
    assert top1_match >= 0.90, f"top1_match={top1_match:.3f}"
    assert recall_20 >= 0.90, f"recall_20={recall_20:.3f}"

    x.neg_()
    graph.replay()
    assert not torch.equal(ids, out_eager[1])
    qpn8_logits = layer._sm70_dflash2_draft_qpn8_logits[:rows, : vocab - padding]
    expected_ids = torch.argsort(qpn8_logits, descending=True, stable=True)[:, :20]
    expected_values = qpn8_logits.gather(1, expected_ids).float()
    assert torch.equal(ids, expected_ids + vocab_start)
    assert torch.equal(vals.view(torch.int32), expected_values.view(torch.int32))

    changed_eager = _maybe_sm70_dflash2_draft_qpn8_topk(layer, x, 20)
    assert changed_eager is not None
    assert torch.equal(ids, changed_eager[1])
    assert torch.equal(vals.view(torch.int32), changed_eager[0].view(torch.int32))
    assert _maybe_sm70_dflash2_draft_qpn8_topk(layer, x[:1].expand(17, -1), 20) is None
