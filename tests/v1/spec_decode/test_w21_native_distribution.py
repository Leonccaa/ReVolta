# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selector, fusion and rejection sampling distribution regressions."""

import argparse
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.native_lookup import _native_fuse_kernel
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    dflash2_sparse_topk_rejection_sample,
    rejection_sample,
)

pytestmark = pytest.mark.skip_global_cleanup


def make_selector_trial(n, device, speculator_class=DFlash2Speculator, steps=7):
    spec = object.__new__(speculator_class)
    spec.draft_block = steps
    spec.num_speculative_steps = steps
    spec.selector_top_k = 32
    spec._use_sm70_tail = device.type == "cuda"
    spec.use_fp64_gumbel = False
    spec.proposal_temperature_scale = spec.proposal_top_p = 1.0
    spec.sample_idx_mapping = torch.arange(
        n, dtype=torch.int32, device=device
    ).repeat_interleave(steps)
    spec.sample_pos = torch.arange(101, 101 + steps, device=device).repeat(n)
    spec.temperature = torch.ones(n, device=device)
    spec.seeds = torch.arange(n, dtype=torch.int64, device=device) + 19
    spec._selector_tokens = torch.empty(n, steps, dtype=torch.int64, device=device)
    spec._selector_scores = torch.empty(n, steps, 32, device=device)
    spec._selector_path_state = torch.empty(n, dtype=torch.int32, device=device)
    spec.draft_tokens = torch.empty_like(spec._selector_tokens)
    spec.draft_logits = torch.full((n, steps, 32), -float("inf"), device=device)
    spec._cached_candidate_ids = torch.zeros(
        n, steps, 32, dtype=torch.int64, device=device
    )
    spec._cached_candidate_scores = torch.full(
        (n, steps, 32), -float("inf"), device=device
    )
    ids = torch.arange(32, device=device).repeat(n, steps, 1)
    lattice = torch.full((1, steps, 32, 32), -float("inf"), device=device)
    for step in range(steps):
        for previous in range(32):
            if step == 0:
                q = torch.full((32,), 0.25 / 31, device=device)
                q[0] = 0.75
                lattice[0, step, previous] = q.log()
            else:
                q = (0.9, 0.1) if previous == 0 else (0.15, 0.85)
                lattice[0, step, previous, :2] = torch.tensor(q, device=device).log()
    lattice = lattice.expand(n, -1, -1, -1).contiguous()
    spec._sample_path(ids, lattice, n)
    spec.draft_tokens.copy_(spec._selector_tokens)
    spec._cache_draft_logits(ids, n * steps)
    return spec, ids, lattice


@triton.jit
def _interpreter_log1p(value):
    # The CPU interpreter has no libdevice implementation. GPU uses log1p.
    return tl.log(1.0 + value)


@triton.jit
def _interpreter_expm1(value):
    return tl.exp(value) - 1.0


@triton.jit
def _interpreter_compact_full_support(
    target_ids_ptr,
    target_logits_ptr,
    target_stride,
    row,
    temperature,
    top_p,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LIMIT_K: tl.constexpr,
):
    # The smoke uses temperature=top_p=1 and the complete target shortlist.
    # Avoid the interpreter's float/bool broadcasting bug in nucleus filtering.
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOP_K
    ids = tl.load(
        target_ids_ptr + row * target_stride + offsets, mask=mask, other=0
    ).to(tl.int64)
    logits = tl.load(
        target_logits_ptr + row * target_stride + offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    maximum = tl.max(logits, 0)
    lse = maximum + tl.log(tl.sum(tl.exp(logits - maximum), 0))
    return ids, logits, mask, lse


@contextmanager
def interpreter_math(device):
    if device.type == "cpu":
        with (
            patch.object(tldevice, "log1p", _interpreter_log1p),
            patch.object(tldevice, "expm1", _interpreter_expm1),
            patch.object(
                rejection_sampler_utils,
                "_dflash2_compact_target_row",
                _interpreter_compact_full_support,
            ),
        ):
            yield
    else:
        yield


def run_distribution(n, device, sparse, valid=2, defect=None):
    with interpreter_math(device):
        return _run_distribution(n, device, sparse, valid, defect)


def verify_proposals(spec, device, sparse, target=None):
    n, steps = spec.draft_tokens.shape
    sampled_input = torch.zeros(n, steps + 1, dtype=torch.int64, device=device)
    sampled_input[:, 1:] = spec.draft_tokens
    if target is None:
        target = torch.full((n, steps + 1, 32), -float("inf"), device=device)
        first_target = torch.full((32,), 0.35 / 31, device=device)
        first_target[0] = 0.65
        target[:, 0] = first_target.log()
        first_p = torch.where(spec.draft_tokens == 0, 0.3, 0.8)
        target[:, 1:, 0] = first_p.log()
        target[:, 1:, 1] = (1 - first_p).log()
    cu = torch.arange(n + 1, device=device, dtype=torch.int32) * (steps + 1)
    positions = torch.arange(100, 101 + steps, device=device).repeat(n)
    idx = torch.arange(n, device=device, dtype=torch.int32)
    if sparse:
        target_ids = (
            torch.arange(32, device=device).expand(n * (steps + 1), 32).contiguous()
        )
        out, counts = dflash2_sparse_topk_rejection_sample(
            target_ids,
            target.reshape(-1, 32),
            spec._cached_candidate_ids,
            spec._cached_candidate_scores,
            sampled_input.flatten(),
            cu,
            positions,
            idx,
            spec.temperature,
            torch.ones_like(spec.temperature),
            spec.seeds,
            steps,
            use_fp64=spec.use_fp64_gumbel,
        )
    else:
        out, counts = rejection_sample(
            target.reshape(-1, 32),
            spec.draft_logits,
            sampled_input.flatten(),
            cu,
            positions,
            idx,
            idx.repeat_interleave(steps + 1),
            torch.arange(steps + 1, device=device, dtype=torch.int32).repeat(n),
            spec.temperature,
            spec.seeds,
            steps,
            use_fp64=spec.use_fp64_gumbel,
        )
    return out, counts


def fuse_trial(spec, valid, defect=None):
    n = spec.draft_tokens.shape[0]
    device = spec.draft_tokens.device
    lookup = torch.tensor(
        [0, 1, 1, 0, 1, 0, 1], dtype=torch.int32, device=device
    ).repeat(n, 1)
    match = torch.full((n,), 4, dtype=torch.int32, device=device)
    lengths = torch.full_like(match, valid)
    eligible = torch.ones_like(match)
    use = torch.zeros_like(lookup)
    # Rows are independent repetitions of one admitted request, vectorized for N.
    _native_fuse_kernel[(n,)](
        spec.draft_tokens,
        7,
        lookup,
        7,
        match,
        lengths,
        eligible,
        use,
        NMIN=2,
        num_warps=1,
    )
    if defect == "first_mass":
        use[:, 0] = (spec.draft_tokens[:, 0] == 0).to(torch.int32)
    _point_mass_draft_logits_kernel[(n * 7,)](
        spec.draft_logits,
        spec._cached_candidate_ids,
        spec._cached_candidate_scores,
        spec.draft_tokens,
        7,
        use,
        spec.sample_idx_mapping,
        7,
        spec._cached_candidate_ids.stride(0),
        spec._cached_candidate_ids.stride(1),
        spec.draft_logits.stride(0),
        spec.draft_logits.stride(1),
        num_steps=7,
        top_k=32,
        BLOCK_K=32,
        CACHE_SCORES=True,
        num_warps=1,
    )
    if defect == "conditional_suffix":
        assert valid == 2
        # Keep every sampled token. Only mislabel the retained neural row's q.
        bad_q = torch.full((n, 32), -float("inf"), device=device)
        probability = torch.where(spec.draft_tokens[:, 1] == 0, 0.9, 0.15)
        bad_q[:, 0], bad_q[:, 1] = probability.log(), (1 - probability).log()
        selected = spec.draft_tokens[:, 0] == 0
        for cache in (spec.draft_logits, spec._cached_candidate_scores):
            cache[:, 2] = torch.where(selected[:, None], bad_q, cache[:, 2])
    return spec


def joint_metrics(tokens, expected, *, pool_rare):
    observed = torch.bincount(tokens, minlength=expected.numel()).double()
    n = int(observed.sum())
    assert int(observed[expected == 0].sum()) == 0
    expected_counts = expected * n
    raw_tv = float((observed / n - expected).abs().sum() / 2)
    if pool_rare:
        retained = expected_counts >= 5
        observed_bins = torch.cat((observed[retained], observed[~retained].sum()[None]))
        expected_bins = torch.cat(
            (expected_counts[retained], expected_counts[~retained].sum()[None])
        )
    else:
        observed_bins, expected_bins = observed, expected_counts
    supported = expected_bins > 0
    tv = float((observed_bins / n - expected_bins / n).abs().sum() / 2)
    chi2 = (
        (observed_bins[supported] - expected_bins[supported]) ** 2
        / expected_bins[supported]
    ).sum()
    degrees = int(supported.sum()) - 1
    p_value = float(
        torch.special.gammaincc(
            torch.tensor(degrees / 2, dtype=torch.float64, device=expected.device),
            chi2 / 2,
        )
    )
    assert degrees > 0 and math.isfinite(tv) and math.isfinite(p_value)
    return dict(
        tv=tv,
        raw_tv=raw_tv,
        chi2=float(chi2),
        degrees=degrees,
        p_value=p_value,
        pooled=pool_rare,
        pooled_cells=int((expected_counts < 5).sum()) if pool_rare else 0,
        counts=observed.cpu().to(torch.int64).tolist(),
        target_joint=expected.cpu().tolist(),
        tested_counts=observed_bins[supported].cpu().tolist(),
        tested_expected=expected_bins[supported].cpu().tolist(),
    )


def _run_distribution(n, device, sparse, valid, defect):
    spec, _, lattice = make_selector_trial(n, device)
    del lattice
    fuse_trial(spec, valid, defect)
    out, counts = verify_proposals(spec, device, sparse)
    idx = torch.arange(n, device=device, dtype=torch.int32)
    completed = torch.empty(n, 3, dtype=torch.int64, device=device)
    completed[:, 0] = out[:, 0]
    # Fill missing outputs after a rejection with fresh target RNG positions.
    for step in (1, 2):
        continuation = torch.full((n, 32), -float("inf"), device=device)
        probability = torch.where(completed[:, step - 1] == 0, 0.3, 0.8)
        continuation[:, 0] = probability.log()
        continuation[:, 1] = (1 - probability).log()
        extra = gumbel_sample(
            continuation,
            idx,
            spec.temperature,
            spec.seeds,
            torch.full_like(idx, 9999 + step),
            apply_temperature=False,
            is_drafting=False,
            use_fp64=spec.use_fp64_gumbel,
        )
        completed[:, step] = torch.where(counts > step, out[:, step], extra)
    expected_two = torch.zeros(32 * 32, dtype=torch.float64, device=device)
    expected_three = torch.zeros(32**3, dtype=torch.float64, device=device)
    for first in range(32):
        mass = 0.65 if first == 0 else 0.35 / 31
        second_p = 0.3 if first == 0 else 0.8
        for second in range(2):
            p_two = mass * (second_p if second == 0 else 1 - second_p)
            expected_two[first * 32 + second] = p_two
            third_p = 0.3 if second == 0 else 0.8
            for third in range(2):
                expected_three[first * 1024 + second * 32 + third] = p_two * (
                    third_p if third == 0 else 1 - third_p
                )
    two = joint_metrics(
        completed[:, 0] * 32 + completed[:, 1], expected_two, pool_rare=False
    )
    three = joint_metrics(
        completed[:, 0] * 1024 + completed[:, 1] * 32 + completed[:, 2],
        expected_three,
        pool_rare=True,
    )
    return dict(
        n=n,
        vocab=32,
        valid=valid,
        sparse=sparse,
        defect=defect,
        joint_two=two,
        joint_three=three,
        bonus_count=int((counts == 8).sum()),
        positive_passed=(
            two["tv"] <= 0.01
            and two["p_value"] > 0.001
            and three["tv"] <= 0.02
            and three["p_value"] > 0.001
        ),
        negative_detected=(three["tv"] > 0.03 or three["p_value"] < 1e-6),
        scope="Independent single-request repetitions; test-only q mutations",
    )


def distribution_suite(n, device, output):
    if device.type == "cuda" and n != 100_000:
        raise ValueError("GPU acceptance requires frozen N=100000")
    results = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for sparse in (False, True):
        for valid, defect in (
            (2, None),
            (7, None),
            (2, "first_mass"),
            (2, "conditional_suffix"),
        ):
            result = run_distribution(n, device, sparse, valid, defect)
            results.append(result)
            # Preserve every completed cell even if a later cell fails.
            output.write_text(json.dumps(results, indent=2) + "\n")
    if n >= 100_000:
        for result in results:
            if result["defect"] is None:
                assert result["positive_passed"] and result["bonus_count"] > 0, result
            else:
                assert result["negative_detected"], (
                    "INVALID acceptance test: injected q defect was not detected",
                    result,
                )
    return results


@pytest.mark.parametrize("sparse", [False, True])
def test_cpu_distribution_smoke(sparse):
    correct = run_distribution(64, torch.device("cpu"), sparse)
    for defect in ("first_mass", "conditional_suffix"):
        mutated = run_distribution(64, torch.device("cpu"), sparse, defect=defect)
        print(
            "SMOKE",
            sparse,
            defect,
            "tv3",
            mutated["joint_three"]["tv"],
            "p3",
            mutated["joint_three"]["p_value"],
        )
        assert mutated["negative_detected"]
        assert mutated["joint_three"]["tv"] > correct["joint_three"]["tv"]
        assert mutated["joint_three"]["counts"] != correct["joint_three"]["counts"]
        if defect == "conditional_suffix":
            assert mutated["joint_two"]["counts"] == correct["joint_two"]["counts"]
        assert sum(mutated["joint_three"]["counts"]) == 64


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("valid", [2, 7])
def test_full_accept_and_bonus(sparse, valid):
    device = torch.device(os.environ.get("W21_TEST_DEVICE", "cpu"))
    with interpreter_math(device):
        spec, _, _ = make_selector_trial(4, device)
        fuse_trial(spec, valid)
        target = torch.full((4, 8, 32), -float("inf"), device=device)
        target[:, :7].copy_(spec.draft_logits)
        target[:, 7, 31] = 0
        out, counts = verify_proposals(spec, device, sparse, target=target)
        assert counts.tolist() == [8] * 4
        assert torch.equal(out[:, :7], spec.draft_tokens)
        assert out[:, 7].tolist() == [31] * 4


def test_three_token_metric_pools_rare_cells():
    expected = torch.tensor([0.49, 0.49, 0.01, 0.01], dtype=torch.float64)
    tokens = torch.tensor([0] * 49 + [1] * 49 + [2, 3])
    result = joint_metrics(tokens, expected, pool_rare=True)
    assert result["pooled_cells"] == 2
    assert result["tested_expected"] == [49.0, 49.0, 2.0]
    assert result["degrees"] == 2 and result["tv"] == 0


@pytest.mark.parametrize("bad_cell", ["positive", "negative", "bonus"])
def test_distribution_suite_rejects_failed_acceptance(monkeypatch, tmp_path, bad_cell):
    import sys

    def sample(n, device, sparse, valid, defect):
        return dict(
            n=n,
            defect=defect,
            positive_passed=bad_cell != "positive",
            negative_detected=bad_cell != "negative",
            bonus_count=0 if bad_cell == "bonus" else 1,
        )

    monkeypatch.setattr(sys.modules[__name__], "run_distribution", sample)
    output = tmp_path / "distribution.json"
    with pytest.raises(AssertionError):
        distribution_suite(100_000, torch.device("cpu"), output)
    assert len(json.loads(output.read_text())) == 8


def test_gpu_acceptance_rejects_nonfrozen_sample_count(tmp_path):
    with pytest.raises(ValueError, match="N=100000"):
        distribution_suite(64, torch.device("cuda"), tmp_path / "invalid.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    distribution_suite(args.samples, torch.device(args.device), args.out)


def _tensor_fingerprint(value):
    import hashlib

    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest(),
    }


def compatibility_trial(mode, device, speculator_class=DFlash2Speculator):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.dflash2.ngram_assist import DFlash2NgramAssist

    global_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device.type == "cuda" else None
    with interpreter_math(device):
        spec, candidates, _ = make_selector_trial(2, device, speculator_class)
        spec._native_lookup, spec._lookup_enabled = None, mode == "legacy_gpu"
        spec._req_states = None
        if mode == "legacy_host":
            spec._ngram_assist = DFlash2NgramAssist(2, 4, 7, 128)
            spec._ngram_rounds = spec._ngram_skipped_rounds = 0
            spec._ngram_tokens_cpu_tensor = torch.zeros(2, 7, dtype=torch.int64)
            spec._ngram_lengths_cpu_tensor = torch.zeros(2, dtype=torch.int32)
            spec._ngram_tokens_cpu = spec._ngram_tokens_cpu_tensor.numpy()
            spec._ngram_lengths_cpu = spec._ngram_lengths_cpu_tensor.numpy()
            spec._ngram_tokens = torch.zeros(2, 7, dtype=torch.int64, device=device)
            spec._ngram_lengths = torch.zeros(2, dtype=torch.int32, device=device)
            history = np.zeros((2, 128), dtype=np.int32)
            history[:, :10] = [4, 5, 0, 1, 2, 3, 4, 5, 6, 4]
            batch = SimpleNamespace(
                num_reqs=2,
                has_structured_output_reqs=False,
                num_draft_tokens_per_req=None,
                seq_lens_cpu_upper_bound=torch.tensor([10, 10]),
                is_prefilling_np=np.array([False, False]),
                idx_mapping_np=np.array([0, 1]),
            )
            event = SimpleNamespace(synchronize=lambda: None)
            assert spec._prepare_ngram_assist(
                batch,
                event,
                np.array([[5], [5]], dtype=np.int64),
                np.ones(2, dtype=np.int32),
                history,
            )
            assert spec._ngram_num_hits == 2
            assert spec._ngram_tokens.tolist() == [list(range(7))] * 2
            spec._apply_ngram_assist(2)
        elif mode == "legacy_gpu":
            spec.num_speculative_steps = 15
            spec.draft_tokens = torch.cat(
                (
                    spec.draft_tokens,
                    torch.zeros(2, 8, dtype=torch.int64, device=device),
                ),
                dim=1,
            )
            spec.draft_logits = torch.full((2, 15, 32), -float("inf"), device=device)
            spec._cached_candidate_ids = torch.zeros(
                2, 15, 32, dtype=torch.int64, device=device
            )
            spec._cached_candidate_scores = torch.full_like(
                spec.draft_logits, -float("inf")
            )
            spec._cache_draft_logits(candidates, 14)
            history = torch.tensor(
                [list(range(20)) + [0, 1, 2, 3]] * 2, dtype=torch.int32, device=device
            )
            lengths = torch.full((2,), 24, dtype=torch.int32, device=device)
            spec._req_states = SimpleNamespace(
                all_token_ids=SimpleNamespace(gpu=history),
                total_len=SimpleNamespace(gpu=lengths),
            )
            spec._lookup_tokens = torch.zeros(2, 15, dtype=torch.int32, device=device)
            spec._lookup_use = torch.zeros_like(spec._lookup_tokens)
            spec._lookup_match_len = torch.zeros(2, dtype=torch.int32, device=device)
            spec._lookup_valid = torch.zeros_like(spec._lookup_match_len)
            spec._lookup_eligible = torch.ones_like(spec._lookup_match_len)
            spec._lookup_hits = torch.zeros((), dtype=torch.int64, device=device)
            spec._lookup_take_flags = torch.zeros_like(spec._lookup_match_len)
            spec._lookup_nmin, spec._lookup_nmax, spec._lookup_search = 2, 4, 64
            spec._lookup_nstrong, spec._lookup_agree = 3, 0
            spec._lookup_nmin_tail, spec._lookup_long_min = 2, 3
        # Includes the normal post-selector routing for all three legacy modes.
        rng_before = torch.get_rng_state().clone()
        cuda_before = (
            torch.cuda.get_rng_state().clone() if device.type == "cuda" else None
        )
        spec._apply_lookup(2)
        assert torch.equal(torch.get_rng_state(), rng_before)
        if cuda_before is not None:
            assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
        out, counts = verify_proposals(spec, device, sparse=False)
        active = torch.arange(out.shape[1], device=device)[None, :] < counts[:, None]
        spec.verified_tokens = torch.where(active, out, -1)
        spec.verified_counts = counts
        assert torch.equal(torch.get_rng_state(), global_rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        return {
            name: _tensor_fingerprint(getattr(spec, name))
            for name in (
                "draft_tokens",
                "draft_logits",
                "_selector_tokens",
                "_selector_scores",
                "_cached_candidate_ids",
                "_cached_candidate_scores",
                "sample_pos",
                "seeds",
                "verified_tokens",
                "verified_counts",
            )
        }


@pytest.mark.parametrize("mode", ["off", "legacy_host", "legacy_gpu"])
def test_closed_and_legacy_match_fixed_main(mode):
    import importlib.util
    import os
    import sys

    device = torch.device(os.environ.get("W21_TEST_DEVICE", "cpu"))
    fixture = Path(__file__).parents[2] / "config/data/dflash2_runtime_baseline.json"
    expected = json.loads(fixture.read_text())
    actual = compatibility_trial(mode, device)
    if device.type == "cpu":
        assert actual == expected["cpu"][mode]
    else:
        root = Path(os.environ["W21_BASELINE"])
        modules = {}
        for name in ("lookup", "speculator"):
            spec = importlib.util.spec_from_file_location(
                "w21_fixed_" + name, root / (name + ".py")
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        for name in ("suffix_lookup", "fuse_draft", "_point_mass_draft_logits_kernel"):
            setattr(modules["speculator"], name, getattr(modules["lookup"], name))
        assert actual == compatibility_trial(
            mode, device, modules["speculator"].DFlash2Speculator
        )


if __name__ == "__main__":
    main()
