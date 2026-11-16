# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resampling after a rejection must stay on the target support.

When the target allows a single token (for example under a grammar) and the
draft puts almost all of its mass on that token, a single FP32 logsumexp
rounds the residual max(p - q, 0) to zero on every token, and resampling
used to return token 0. The residual now keeps (max, correction) apart; if
it is still empty, the single-logsumexp residual and then the target
distribution are used.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="rejection sampler kernels need CUDA"
)


def _resample(
    target_row: dict[int, float],
    draft_row: dict[int, float],
    drafted: int,
    vocab: int,
    num_reqs: int,
    seed_base: int = 0,
    fallback_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """One draft step per request; returns the token sampled at step 0."""
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    rows = num_reqs * 2
    target = torch.full((rows, vocab), float("-inf"), device="cuda")
    for token, logit in target_row.items():
        target[:, token] = logit
    draft = torch.full((num_reqs, 1, vocab), float("-inf"), device="cuda")
    for token, logit in draft_row.items():
        draft[:, :, token] = logit
    sampled_in = torch.zeros((num_reqs, 2), device="cuda", dtype=torch.int64)
    sampled_in[:, 1] = drafted
    cu = torch.arange(num_reqs + 1, device="cuda", dtype=torch.int32) * 2
    idx = torch.arange(num_reqs, device="cuda", dtype=torch.int32)
    out, num_sampled = rejection_sample(
        target,
        draft,
        sampled_in.flatten(),
        cu,
        torch.zeros(rows, device="cuda", dtype=torch.int64),
        idx,
        idx.repeat_interleave(2),
        torch.arange(2, device="cuda", dtype=torch.int32).repeat(num_reqs),
        torch.ones(num_reqs, device="cuda"),
        torch.arange(num_reqs, device="cuda", dtype=torch.int64) + seed_base,
        1,
        fallback_count=fallback_count,
    )
    assert torch.all(num_sampled == 1)
    return out[:, 0]


@pytest.mark.parametrize("vocab", [248320, 32000])
@pytest.mark.parametrize("gap", [10.0, 14.0, 17.0, 20.0, 25.0, 60.0])
def test_single_token_support_survives_rejection(gap: float, vocab: int):
    for seed_base in (0, 10_000):
        count = torch.zeros(2, device="cuda", dtype=torch.int64)
        first = _resample(
            {3: 50.0}, {3: 50.0, 1: 50.0 - gap}, 1, vocab, 256, seed_base, count
        )
        assert torch.all(first == 3), torch.unique(first, return_counts=True)
        # The stable residual keeps exp(-gap) without any fallback.
        assert count.tolist() == [0, 0]


def test_single_token_support_in_a_late_block():
    token = 200_001
    first = _resample({token: 50.0}, {token: 50.0, 5: 30.0}, 5, 248320, 256)
    assert torch.all(first == token), torch.unique(first, return_counts=True)


def test_two_tied_support_stays_in_support_and_balanced():
    # Target ties tokens 3 and 4; the draft adds exp(-20) relative mass on
    # token 1 and drafts it. The true residual is (0.5, 0.5) on {3, 4}.
    # (8192 x 2 rows of a 248320 vocab would not fit a 16 GB card.)
    num_reqs = 8192
    count = torch.zeros(2, device="cuda", dtype=torch.int64)
    first = _resample(
        {3: 0.0, 4: 0.0}, {3: 10.0, 4: 10.0, 1: -10.0}, 1, 32000, num_reqs, 0, count
    )
    # The stable residual is empty here; the single-logsumexp one is not.
    assert count.tolist() == [num_reqs, 0]
    assert torch.all((first == 3) | (first == 4)), torch.unique(
        first, return_counts=True
    )
    share = (first == 3).float().mean().item()
    assert 0.45 < share < 0.55, share


def test_residual_with_two_positive_tokens():
    # p = (0.5, 0.3, 0.2) on tokens (3, 5, 9); q = (0.6, 0.1, 0.0) plus 0.3 on
    # the drafted token 7, which the target rejects. Residual: (0, 0.2, 0.2).
    t = torch.tensor([0.5, 0.3, 0.2]).log().tolist()
    d = torch.tensor([0.6, 0.1, 0.3]).log().tolist()
    num_reqs = 8192
    first = _resample(
        {3: t[0], 5: t[1], 9: t[2]}, {3: d[0], 5: d[1], 7: d[2]}, 7, 4096, num_reqs
    )
    assert torch.all((first == 5) | (first == 9)), torch.unique(
        first, return_counts=True
    )
    share = (first == 5).float().mean().item()
    assert 0.45 < share < 0.55, share


def test_empty_residual_falls_back_to_nonuniform_target():
    # The draft equals the target on (3, 5000, 31999), in different vocab
    # blocks, plus exp(-40) relative mass on the drafted token 1, which the
    # target rejects. FP32 log probs agree on the support, the residual is
    # empty, and the true residual is proportional to p = (0.5, 0.3, 0.2).
    support = {3: 0.5, 5000: 0.3, 31999: 0.2}
    logits = {t: float(torch.tensor(pr).log()) for t, pr in support.items()}
    draft = dict(logits)
    draft[1] = max(logits.values()) - 40.0
    num_reqs = 16384
    count = torch.zeros(2, device="cuda", dtype=torch.int64)
    first = _resample(logits, draft, 1, 32000, num_reqs, 0, count)
    assert count.tolist() == [0, num_reqs]
    in_support = torch.zeros_like(first, dtype=torch.bool)
    for token, prob in support.items():
        in_support |= first == token
        share = (first == token).float().mean().item()
        assert abs(share - prob) < 0.02, (token, share)
    assert torch.all(in_support)


def test_ordinary_residual_does_not_fall_back():
    t = torch.tensor([0.5, 0.3, 0.2]).log().tolist()
    d = torch.tensor([0.6, 0.1, 0.3]).log().tolist()
    count = torch.zeros(2, device="cuda", dtype=torch.int64)
    _resample(
        {3: t[0], 5: t[1], 9: t[2]},
        {3: d[0], 5: d[1], 7: d[2]},
        7,
        4096,
        1024,
        0,
        count,
    )
    assert count.tolist() == [0, 0]


def test_asymmetric_near_equal_residual_is_kept():
    # Target and draft agree up to small logit shifts; the true residual on
    # (3, 4) is about (0.488, 0.512). A single FP32 logsumexp keeps only
    # token 3; the stable form keeps both.
    num_reqs = 8192
    count = torch.zeros(2, device="cuda", dtype=torch.int64)
    first = _resample(
        {3: -2.625, 4: -1.375, 5: -4.0},
        {3: 7.375004768371582, 4: 8.625005722045898, 5: 6.000017166137695, 1: -30.0},
        1,
        32000,
        num_reqs,
        0,
        count,
    )
    assert count.tolist() == [0, 0]
    assert torch.all((first == 3) | (first == 4)), torch.unique(
        first, return_counts=True
    )
    share = (first == 4).float().mean().item()
    assert 0.46 < share < 0.56, share


def _graph_inputs(num_reqs: int, vocab: int):
    rows = num_reqs * 2
    return dict(
        target=torch.zeros(rows, vocab, device="cuda"),
        draft=torch.zeros(num_reqs, 1, vocab, device="cuda"),
        sampled_in=torch.zeros(num_reqs * 2, device="cuda", dtype=torch.int64),
    )


def _fill(bufs, target_row, draft_row, drafted, num_reqs):
    bufs["target"].fill_(float("-inf"))
    bufs["draft"].fill_(float("-inf"))
    for token, logit in target_row.items():
        bufs["target"][:, token] = logit
    for token, logit in draft_row.items():
        bufs["draft"][:, :, token] = logit
    bufs["sampled_in"].view(num_reqs, 2)[:, 1] = drafted


def test_fallback_tiers_under_cuda_graph_replay():
    """Capture on an input whose stable residual is non-empty, then replay on
    inputs that need tier 2 (two tied tokens) and tier 3 (p equal to q)."""
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    num_reqs, vocab = 64, 32000
    bufs = _graph_inputs(num_reqs, vocab)
    cu = torch.arange(num_reqs + 1, device="cuda", dtype=torch.int32) * 2
    idx = torch.arange(num_reqs, device="cuda", dtype=torch.int32)
    count = torch.zeros(2, device="cuda", dtype=torch.int64)
    args = (
        bufs["target"],
        bufs["draft"],
        bufs["sampled_in"],
        cu,
        torch.zeros(num_reqs * 2, device="cuda", dtype=torch.int64),
        idx,
        idx.repeat_interleave(2),
        torch.arange(2, device="cuda", dtype=torch.int32).repeat(num_reqs),
        torch.ones(num_reqs, device="cuda"),
        torch.arange(num_reqs, device="cuda", dtype=torch.int64),
        1,
    )
    t = torch.tensor([0.5, 0.3, 0.2]).log().tolist()
    d = torch.tensor([0.6, 0.1, 0.3]).log().tolist()
    _fill(bufs, {3: t[0], 5: t[1], 9: t[2]}, {3: d[0], 5: d[1], 7: d[2]}, 7, num_reqs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            rejection_sample(*args, fallback_count=count)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, num_sampled = rejection_sample(*args, fallback_count=count)

    count.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    assert count.tolist() == [0, 0]
    assert torch.all((out[:, 0] == 5) | (out[:, 0] == 9))

    _fill(bufs, {3: 0.0, 4: 0.0}, {3: 10.0, 4: 10.0, 1: -10.0}, 1, num_reqs)
    count.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    assert count.tolist() == [num_reqs, 0]
    assert torch.all((out[:, 0] == 3) | (out[:, 0] == 4))
    assert torch.all(num_sampled == 1)

    support = {3: 0.5, 5000: 0.3, 31999: 0.2}
    logits = {k: float(torch.tensor(v).log()) for k, v in support.items()}
    draft = dict(logits)
    draft[1] = max(logits.values()) - 40.0
    _fill(bufs, logits, draft, 1, num_reqs)
    count.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    assert count.tolist() == [0, num_reqs]
    first = out[:, 0]
    assert torch.all((first == 3) | (first == 5000) | (first == 31999))
