# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 compact target cutoffs agree with upstream dense index ordering."""

import pytest
import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import (
    apply_top_k_top_p_triton,
    sort_topk_with_vocab_ties,
)
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _compact_target_reference_rows,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _dflash2_compact_target_row,
)


@triton.jit
def _sparse_target_logits_kernel(
    ids,
    values,
    ps,
    output,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.load(ps + row)
    _, logits, _, _ = _dflash2_compact_target_row(
        ids,
        values,
        WIDTH,
        row,
        p,
        TOP_K=WIDTH,
        BLOCK_K=BLOCK,
        LIMIT_K=20,
    )
    col = tl.arange(0, BLOCK)
    tl.store(output + row * WIDTH + col, logits, col < WIDTH)


def _cutoff_input(device):
    generator = torch.Generator().manual_seed(425)
    ids = torch.randperm(32768, generator=generator)[:24].to(device)
    logits = torch.full((2, 32768), -20.0, device=device)
    logits[:, ids[:18]] = 2.0
    logits[:, ids[18:]] = 1.0
    return logits


def _scaled(logits, temperature):
    scaled = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
    apply_temperature(
        scaled,
        torch.arange(logits.shape[0], device=logits.device),
        torch.full((logits.shape[0],), temperature, device=logits.device),
    )
    return scaled


def _sorted_probe(logits, temperature):
    values, ids = logits.topk(64, dim=-1)
    return sort_topk_with_vocab_ties(
        _scaled(values, temperature), ids, vocab_size=logits.shape[1], descending=True
    )


def _sparse_logits(logits, p, temperature):
    values, ids = _sorted_probe(logits, temperature)
    masked = torch.empty_like(values)
    _sparse_target_logits_kernel[(logits.shape[0],)](
        ids, values, p, masked, WIDTH=64, BLOCK=64, num_warps=1
    )
    result = torch.full_like(logits, -float("inf"))
    result.scatter_(1, ids, masked)
    return result, values


def _dense_index_ordered_reference(logits, p):
    ids = logits.argsort(dim=-1, descending=True, stable=True)[:, :20]
    values = logits.gather(1, ids)
    candidates = torch.full_like(logits, -float("inf"))
    candidates.scatter_(1, ids, values)
    reference = apply_top_k_top_p_pytorch(candidates, None, p)
    counts = torch.isfinite(reference).sum(-1)
    # PyTorch supplies the nucleus count; upstream gives its boundary tie
    # budget to the earliest vocabulary IDs in this score-ordered prefix.
    mask = torch.arange(20, device=logits.device)[None, :] < counts[:, None]
    expected = torch.full_like(logits, -float("inf"))
    expected.scatter_(1, ids, values.masked_fill(~mask, -float("inf")))
    return expected


@pytest.mark.parametrize("top_p", [0.5, 0.95, 1.0])
@pytest.mark.parametrize("temperature", [1.0, 1.6, 0.7])
def test_sparse_prefix_matches_dense_topk_and_guard_on_cpu(top_p, temperature):
    logits = _cutoff_input("cpu")
    probe, ids = _sorted_probe(logits, temperature)
    actual = torch.full_like(logits, -float("inf"))
    actual.scatter_(1, ids[:, :20], probe[:, :20])
    expected = _dense_index_ordered_reference(
        _scaled(logits, temperature), torch.ones(2)
    )
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert not _compact_target_reference_rows(
        probe, 1.0, top_p, vocab_ordered=True
    ).any()


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="CUDA Triton required"
)
@pytest.mark.parametrize("top_p", [0.5, 0.95, 1.0])
@pytest.mark.parametrize("temperature", [1.0, 1.6, 0.7])
def test_sparse_and_dense_cutoffs_match_on_gpu(top_p, temperature):
    logits = _cutoff_input("cuda")
    p = torch.full((2,), top_p, device="cuda")
    k = torch.full((2,), 20, dtype=torch.int32, device="cuda")
    sparse, _ = _sparse_logits(logits, p, temperature)
    dense = apply_top_k_top_p_triton(_scaled(logits, temperature), k, p)
    assert torch.equal(torch.isfinite(sparse), torch.isfinite(dense))
    assert torch.equal(sparse.view(torch.int32), dense.view(torch.int32))


def _scaled_tie_input(high_count, device):
    logits = torch.full((2, 32768), -20.0, device=device)
    logits[:, :high_count] = 2.0
    logits[:, 1000] = 1.6000001430511475
    logits[:, 2000] = 1.600000262260437
    return logits


@pytest.mark.parametrize("high_count,top_p", [(19, 1.0), (18, 0.94)])
def test_temperature_created_cutoff_ties_on_cpu(high_count, top_p):
    raw = _scaled_tie_input(high_count, "cpu")
    scaled = _scaled(raw, 1.6)
    assert torch.equal(scaled[:, 1000], scaled[:, 2000])
    values, ids = _sorted_probe(raw, 1.6)
    prefix = values[:, :20]
    probabilities = prefix.softmax(-1)
    keep = probabilities.cumsum(-1) - probabilities < top_p
    actual = torch.full_like(raw, -float("inf"))
    actual.scatter_(1, ids[:, :20], prefix.masked_fill(~keep, -float("inf")))
    expected = _dense_index_ordered_reference(scaled, torch.full((2,), top_p))
    assert torch.isfinite(actual[:, 1000]).all()
    assert torch.isneginf(actual[:, 2000]).all()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="CUDA Triton required"
)
@pytest.mark.parametrize("high_count,top_p", [(19, 1.0), (18, 0.94)])
def test_temperature_created_cutoff_ties_on_gpu(high_count, top_p):
    raw = _scaled_tie_input(high_count, "cuda")
    p = torch.full((2,), top_p, device="cuda")
    actual, _ = _sparse_logits(raw, p, 1.6)
    expected = _dense_index_ordered_reference(_scaled(raw, 1.6), p)
    assert torch.isfinite(actual[:, 1000]).all()
    assert torch.isneginf(actual[:, 2000]).all()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
