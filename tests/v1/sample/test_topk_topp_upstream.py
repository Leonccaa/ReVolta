# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream dispatch and cutoff contracts, including steady-state sync checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.ops import topk_topp_sampler as sampler
from vllm.v1.worker.gpu.sample import states as states_module

GPU = pytest.mark.skipif(
    not (HAS_TRITON and current_platform.is_cuda()), reason="CUDA Triton required"
)


@pytest.mark.parametrize("rows", [1, 5])
def test_dispatch_forwards_topp_only_hint(monkeypatch, rows):
    seen = []

    def triton_filter(logits, k, p, *, has_topp_only_rows):
        seen.append(has_topp_only_rows)
        return logits

    monkeypatch.setattr(sampler, "HAS_TRITON", True)
    monkeypatch.setattr(
        sampler, "apply_top_k_top_p_triton", triton_filter, raising=False
    )
    logits = torch.ones(rows, 32)
    p = torch.full((rows,), 0.95)
    assert sampler.apply_top_k_top_p(logits, None, p, False) is logits
    assert seen == [False]


@pytest.mark.parametrize("p_only", [False, True])
def test_v2_topp_only_hint_uses_selected_requests(monkeypatch, p_only):
    state = object.__new__(states_module.SamplingStates)
    state.vocab_size = 32
    state.top_k = SimpleNamespace(
        np=np.array([32, 20, 32]), gpu=torch.tensor([32, 20, 32])
    )
    state.top_p = SimpleNamespace(
        np=np.array([0.5, 0.95, 0.5 if p_only else 1.0]),
        gpu=torch.tensor([0.5, 0.95, 0.5 if p_only else 1.0]),
    )
    seen = []

    def filter_logits(logits, k, p, *, has_topp_only_rows):
        seen.append(has_topp_only_rows)
        assert k.tolist() == [20, 20, 32]
        return logits

    monkeypatch.setattr(states_module, "apply_top_k_top_p", filter_logits)
    logits = torch.ones(3, 32)
    assert (
        state.apply_top_k_top_p(logits, torch.tensor([1, 1, 2]), np.array([1, 2]))
        is logits
    )
    assert seen == [p_only]


def _logits(rows, pattern, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(423)
    logits = torch.randn(rows, 248320, device=device, generator=generator)
    if pattern == "ties":
        logits.round_()
    elif pattern == "quantized":
        logits = logits.half().float()
    return logits


def _assert_topk_contract(logits, actual, reference, k):
    kept = torch.isfinite(actual)
    ref_kept = torch.isfinite(reference)
    assert not (kept & ~ref_kept).any()
    assert torch.equal(kept.sum(-1), torch.minimum(ref_kept.sum(-1), k.to(torch.int64)))
    cutoff = logits.sort(-1, descending=True).values.gather(1, (k.long() - 1)[:, None])
    extra = ref_kept & ~kept
    assert torch.equal(logits[extra], cutoff.expand_as(logits)[extra])
    # Upstream gives the cutoff tie budget to the earliest vocabulary indices.
    ties = logits == cutoff
    expected = (logits > cutoff) | (
        ties & (ties.cumsum(-1) <= (k - (logits > cutoff).sum(-1))[:, None])
    )
    assert torch.equal(kept, expected)
    assert torch.equal(actual[kept].view(torch.int32), logits[kept].view(torch.int32))


@GPU
@pytest.mark.parametrize("rows", [1, 5, 10, 20, 40])
@pytest.mark.parametrize("top_k", [1, 20, 127, 128, 248320, None])
@pytest.mark.parametrize("pattern", ["random", "ties", "quantized"])
def test_topk_cutoff_subset_count_and_value_bits(rows, top_k, pattern):
    logits = _logits(rows, pattern)
    k = torch.full((rows,), top_k or logits.shape[1], device="cuda", dtype=torch.int32)
    # Mixed greedy-equivalent rows and sampled rows in the same filter call.
    if top_k is not None:
        k[4::4] = 1
    active_k = k if top_k is not None else None
    actual = sampler.apply_top_k_top_p(logits.clone(), active_k, None)
    reference = sampler.apply_top_k_top_p_pytorch(logits.clone(), active_k, None)
    _assert_topk_contract(logits, actual, reference, k)


@GPU
@pytest.mark.parametrize("mode", ["topk", "topp", "joint", "mixed"])
def test_steady_state_has_no_host_sync(mode):
    logits = _logits(20, "random")
    k = torch.full((20,), 20, device="cuda", dtype=torch.int32)
    p = torch.full((20,), 0.95, device="cuda")
    if mode == "topk":
        p = None
    elif mode == "topp":
        k = None
    elif mode == "mixed":
        k[::2] = logits.shape[1]
    work = logits.clone()
    sampler.apply_top_k_top_p(work, k, p)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        for _ in range(2):
            work.copy_(logits)
            sampler.apply_top_k_top_p(work, k, p)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@GPU
def test_native_filter_graph_capture():
    logits = _logits(5, "random")
    k = torch.tensor([1, 20, 248320, 128, 248320], device="cuda", dtype=torch.int32)
    p = torch.tensor([1.0, 0.95, 0.5, 0.01, 1.0], device="cuda")
    expected = sampler.apply_top_k_top_p(logits.clone(), k, p)
    work = logits.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        work.copy_(logits)
        sampler.apply_top_k_top_p(work, k, p)
    graph.replay()
    assert torch.equal(work, expected)


def test_dummy_warmup_covers_filter_modes_and_split_boundary(monkeypatch):
    seen = set()

    def filter_logits(logits, k, p):
        assert logits.dtype == torch.float32
        seen.add((logits.shape[0], k is not None, p is not None))
        return logits

    monkeypatch.setattr(sampler, "HAS_TRITON", True)
    monkeypatch.setattr(sampler, "apply_top_k_top_p", filter_logits)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16), num_speculative_tokens=4
    )
    sampler.warmup_top_k_top_p(torch.randn(16, 32, dtype=torch.float16), config)
    for rows in (1, 5, 10, 20, 40, 64, 80):
        for has_k, has_p in ((True, False), (False, True), (True, True)):
            assert (rows, has_k, has_p) in seen


@GPU
def test_fp16_warmup_runs_expanded_native_modes():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16), num_speculative_tokens=4
    )
    sampler.warmup_top_k_top_p(_logits(16, "random").half(), config)
    torch.accelerator.synchronize()


def _assert_topp_boundary(logits, actual, reference):
    kept = torch.isfinite(actual)
    ref_kept = torch.isfinite(reference)
    actual_count = kept.sum(-1)
    ref_count = ref_kept.sum(-1)
    assert (actual_count > 0).all()

    # Match TestTritonTopkTopp: at most 3 tokens, or strictly below 0.5%.
    max_diff = (actual_count - ref_count).abs().max().item()
    max_kept = ref_count.max().item()
    if max_diff > 3:
        assert max_diff / max_kept * 100 < 0.5
    tolerance = max(3, int(np.ceil(max_kept * 0.005)) - 1)

    # Count agreement alone must not permit dropping a high-scoring token
    # and retaining an unrelated low-scoring one. Allow only the ranks near
    # the reference nucleus boundary, including equal-valued boundary ties.
    ordered = logits.sort(dim=-1, descending=True).values
    upper_rank = (ref_count - tolerance - 1).clamp(min=0)
    lower_rank = (ref_count + tolerance - 1).clamp(max=logits.shape[1] - 1)
    upper = ordered.gather(1, upper_rank[:, None]).expand_as(logits)
    lower = ordered.gather(1, lower_rank[:, None]).expand_as(logits)
    different = kept ^ ref_kept
    assert (logits[different] <= upper[different]).all()
    assert (logits[different] >= lower[different]).all()


def _upstream_joint_reference(logits, k, p):
    cutoff = logits.sort(-1, descending=True).values.gather(1, (k.long() - 1)[:, None])
    ties = logits == cutoff
    topk_mask = (logits > cutoff) | (
        ties & (ties.cumsum(-1) <= (k - (logits > cutoff).sum(-1))[:, None])
    )
    candidates = logits.masked_fill(~topk_mask, -float("inf"))
    return candidates, sampler.apply_top_k_top_p_pytorch(candidates.clone(), None, p)


def _assert_joint_contract(logits, actual, candidates, reference, p):
    kept = torch.isfinite(actual)
    common = kept & torch.isfinite(reference)
    assert torch.equal(
        actual[common].view(torch.int32), reference[common].view(torch.int32)
    )
    assert torch.equal(actual[kept].view(torch.int32), logits[kept].view(torch.int32))

    topk_mask = torch.isfinite(candidates)
    assert not (kept & ~topk_mask).any()
    no_topp = p == 1.0
    assert torch.equal(kept[no_topp], topk_mask[no_topp])
    _assert_topp_boundary(candidates, actual, reference)


@pytest.mark.parametrize(
    "top_k,pattern",
    [(20, "random"), (20, "ties"), (128, "random"), (256, "quantized")],
)
def test_joint_comparison_reference_on_cpu(top_k, pattern):
    generator = torch.Generator().manual_seed(423)
    logits = torch.randn(5, 256, generator=generator)
    if pattern == "ties":
        logits.round_()
    elif pattern == "quantized":
        logits = logits.half().float()
    k = torch.full((5,), top_k, dtype=torch.int32)
    p = torch.full((5,), 0.95)
    candidates, reference = _upstream_joint_reference(logits, k, p)
    # Independent stable sort applies the upstream top-k index tie-break.
    indices = logits.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    stable_candidates = torch.full_like(logits, -float("inf"))
    stable_candidates.scatter_(1, indices, logits.gather(1, indices))
    assert torch.equal(candidates, stable_candidates)
    actual = sampler.apply_top_k_top_p_pytorch(stable_candidates, None, p)
    _assert_joint_contract(logits, actual, candidates, reference, p)
    retained = torch.isfinite(actual).nonzero()[0]
    changed = actual.clone()
    changed.view(torch.int32)[retained[0], retained[1]] ^= 1
    with pytest.raises(AssertionError):
        _assert_joint_contract(logits, changed, candidates, reference, p)


def test_joint_reference_normalizes_after_index_ordered_topk():
    logits = torch.full((2, 256), -20.0)
    logits[:, :18] = 2.0
    logits[:, 18:24] = 1.0
    k = torch.full((2,), 20, dtype=torch.int32)
    p = torch.full((2,), 0.5)
    candidates, reference = _upstream_joint_reference(logits, k, p)
    old_reference = sampler.apply_top_k_top_p_pytorch(logits.clone(), k, p)
    assert torch.equal(torch.isfinite(candidates).sum(-1), k.long())
    assert torch.isfinite(candidates[:, :20]).all()
    assert torch.isneginf(candidates[:, 20:]).all()
    assert torch.isfinite(old_reference).sum(-1).tolist() == [11, 11]
    assert torch.isfinite(reference).sum(-1).tolist() == [10, 10]


def test_topp_comparison_accepts_boundary_but_rejects_distant_token_on_cpu():
    generator = torch.Generator().manual_seed(423)
    logits = torch.randn(1, 256, generator=generator)
    reference = sampler.apply_top_k_top_p_pytorch(
        logits.clone(), None, torch.tensor([0.5])
    )
    count = int(torch.isfinite(reference).sum())
    indices = logits.argsort(dim=-1, descending=True)[:, : count + 1]
    near = torch.full_like(logits, -float("inf"))
    near.scatter_(1, indices, logits.gather(1, indices))
    _assert_topp_boundary(logits, near, reference)

    distant = reference.clone()
    distant[0, logits.argmax()] = -float("inf")
    distant[0, logits.argmin()] = logits.min()
    with pytest.raises(AssertionError):
        _assert_topp_boundary(logits, distant, reference)


@GPU
@pytest.mark.parametrize("rows", [1, 5, 10, 20, 40])
@pytest.mark.parametrize("top_k", [1, 20, 127, 128, 248320, None])
@pytest.mark.parametrize("top_p", [0.01, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("pattern", ["random", "ties", "quantized"])
def test_joint_cutoff_support_and_value_bits(rows, top_k, top_p, pattern):
    logits = _logits(rows, pattern)
    k = torch.full((rows,), top_k or logits.shape[1], device="cuda", dtype=torch.int32)
    p = torch.full((rows,), top_p, device="cuda")
    if top_k is not None:
        k[4::4] = 1
    p[4::4] = 1.0
    active_k = k if top_k is not None else None
    actual = sampler.apply_top_k_top_p(logits.clone(), active_k, p)
    candidates, reference = _upstream_joint_reference(logits, k, p)
    _assert_joint_contract(logits, actual, candidates, reference, p)
