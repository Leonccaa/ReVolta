# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers import logits_processor as lp
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.models.qwen4_exp.nvidia import sm70_bounded_target_head as bth
from vllm.models.qwen4_exp.nvidia import sm70_mtp_head


def _head(rows=64, width=2560, start=64, padding=0):
    head = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    nn.Module.__init__(head)
    head.quant_method = UnquantizedEmbeddingMethod()
    torch.manual_seed(7)
    head.weight = nn.Parameter(
        torch.randn(rows, width).mul_(0.02).half(), requires_grad=False
    )
    head.shard_indices = SimpleNamespace(
        org_vocab_start_index=start,
        num_org_vocab_padding=padding,
        num_added_elements=0,
    )
    return head


def _config(tp=4, method="mtp", draft="greedy", flag=True):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method, draft_sample_method=draft),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        kernel_config=SimpleNamespace(sm70_mtp_bounded_target_head=flag),
        lora_config=None,
    )


@pytest.fixture
def sm70(monkeypatch):
    monkeypatch.setattr(
        bth,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda cap: cap == 70),
    )
    # Keep the pre-pack FP8 codes and scales visible to the emulated GEMM.
    monkeypatch.setattr(
        bth.sm70_ops, "fp8_qpn8_prepare_sm70", lambda q, s: (q, s), raising=False
    )


@pytest.mark.parametrize(
    ("kwargs", "head_kwargs", "admitted"),
    [
        ({}, {}, True),
        ({"tp": 1}, {}, False),
        ({"method": "dflash"}, {}, False),
        ({"draft": "probabilistic"}, {}, False),
        ({}, {"padding": 32}, False),
        ({}, {"width": 2048}, False),
        ({}, {"rows": 48}, False),
    ],
)
def test_admission(sm70, kwargs, head_kwargs, admitted):
    head = _head(**head_kwargs)
    prepared = bth.prepare_mtp_bounded_target_head(head, _config(**kwargs))
    assert (prepared is not None) is admitted


def test_model_prepare_respects_flag_and_routes_top_tokens(sm70):
    # The full model module needs the multimodal stack (torchvision).
    qmodel = pytest.importorskip("vllm.models.qwen4_exp.nvidia.model")
    head = _head()
    for flag, expect in ((False, False), (True, True)):
        target = qmodel.Qwen4ExpForCausalLM.__new__(qmodel.Qwen4ExpForCausalLM)
        nn.Module.__init__(target)
        object.__setattr__(target, "_sm70_target_head", None)
        target.vllm_config = _config(flag=flag)
        target.lm_head = head
        seen: list[nn.Module] = []

        def top_tokens(h, x, seen=seen):
            seen.append(h)
            return x

        target.logits_processor = SimpleNamespace(get_top_tokens=top_tokens)
        target.prepare_sm70_bounded_target_head()
        assert (target._sm70_target_head is not None) is expect
        target.get_top_tokens(torch.zeros(1))
        assert (seen[0] is not head) is expect


def test_draft_reuses_shared_codes(sm70, monkeypatch):
    head = _head()
    prepared = bth.prepare_mtp_bounded_target_head(head, _config())
    calls = []

    def prepare(weight):
        calls.append(weight)
        return None, None

    monkeypatch.setattr(sm70_mtp_head, "prepare_channel_qpn8_weight", prepare)
    draft = sm70_mtp_head.MTPQPN8LogitsProcessor(248320, head.weight)
    assert not calls
    assert draft.codes is prepared.codes and draft.scales is prepared.scales


def test_shared_codes_match_draft_quantization(monkeypatch):
    # The draft head may read the target pack only if both quantize alike.
    from vllm.model_executor.layers.quantization import sm70_online_qpn8

    monkeypatch.setattr(
        sm70_online_qpn8.sm70_ops, "fp8_qpn8_prepare_sm70", lambda q, s: (q, s)
    )
    monkeypatch.setattr(bth.sm70_ops, "fp8_qpn8_prepare_sm70", lambda q, s: (q, s))
    weight = _head().weight
    weight.data[3].zero_()
    draft_q, draft_s = sm70_online_qpn8.prepare_channel_qpn8_weight(weight)
    target_q, target_s, _ = bth._prepare_proposal(weight)
    assert torch.equal(draft_q.view(torch.uint8), target_q.view(torch.uint8))
    assert torch.equal(draft_s, target_s)


def test_local_pair_routing(sm70, monkeypatch):
    # Batches above the row limit keep the dense head.
    processor = lp.LogitsProcessor(256)
    head = bth.prepare_mtp_bounded_target_head(_head(), _config())
    sentinel = torch.tensor([[1.0, 2.0]])
    monkeypatch.setattr(bth, "_bounded_pair", lambda *a: sentinel)
    monkeypatch.setattr(
        processor,
        "get_local_top1",
        lambda *a: (torch.tensor([3.0]), torch.tensor([4])),
    )
    limit = bth._MAX_BOUNDED_ROWS
    x = torch.zeros(limit + 1, 2560)
    assert processor._get_local_top1_pair(head, x[:limit], None) is sentinel
    assert processor._get_local_top1_pair(head, x, None).tolist() == [[3.0, 4.0]]


def _emulate_native(monkeypatch, dense_calls):
    def gemm(out, x, codes, scales, split_k, nacc, fast, prefetch):
        represented = (
            ((codes.float() / 256) * (scales * 256).half().float()).half().float()
        )
        out.copy_((x.float() @ represented.t()).half())

    def candidate_scores(scores, x, weight, ids, tiled):
        rows = weight.float()[ids.long()]
        scores.copy_(torch.einsum("rk,rck->rc", x.float(), rows).half())

    def masked_head(scores, x, weight, flags, tiled):
        dense_calls.append(int((flags == 0).sum()))
        dense = (x.float() @ weight.float().t()).half()
        keep = (flags != 0).view(-1, 1)
        scores.copy_(torch.where(keep, scores, dense))

    monkeypatch.setattr(bth.sm70_ops, "fp8_qpn8_gemm_sm70_out", gemm, raising=False)
    monkeypatch.setattr(
        torch.ops._C, "sm70_fp16_candidate_scores_out", candidate_scores, raising=False
    )
    monkeypatch.setattr(
        torch.ops._C, "sm70_fp16_masked_head_out", masked_head, raising=False
    )
    monkeypatch.setattr(bth, "_gather_bounded_packet", lambda packet: packet)


@pytest.mark.parametrize("weak_bound", [False, True])
def test_bounded_pair_matches_dense_argmax(sm70, monkeypatch, weak_bound):
    pytest.importorskip("triton")
    dense_calls: list[int] = []
    _emulate_native(monkeypatch, dense_calls)
    vocab, width, start = 2048, 2560, 4096
    head = _head(rows=vocab, width=width, start=start)
    # A few clear winners keep the bound conclusive for the first rows.
    head.weight.data[17].mul_(8)
    head.weight.data[1500].mul_(8)
    prepared = bth.prepare_mtp_bounded_target_head(head, _config())
    if weak_bound:
        prepared.coefficients = prepared.coefficients.clone()
        prepared.coefficients[:, 0] = 1e4
    torch.manual_seed(3)
    x = torch.randn(3, width).half()
    x[0] = head.weight[17].float().mul(400).half()
    x[1] = head.weight[1500].float().mul(400).half()
    pairs = prepared.maybe_get_sm70_lm_head_top1_pair(x)
    reference = (x.float() @ head.weight.float().t()).half().float()
    assert pairs[:, 1].long().tolist() == (reference.argmax(-1) + start).tolist()
    assert torch.equal(pairs[:, 0], reference.max(-1).values)
    if weak_bound:
        assert dense_calls == [3]
    else:
        assert dense_calls[0] < 3
