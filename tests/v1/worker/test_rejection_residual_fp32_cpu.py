# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the dense rejection residual and its fallbacks.

The kernel helpers are taken from the source by AST and executed with FP32
numpy arrays. This checks the arithmetic the kernels run (not Triton
compilation, loads or reduction order). The CUDA regression lives in
tests/v1/spec_decode/test_rejection_residual_underflow.py.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vllm

PATH = (
    Path(vllm.__file__).parent / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
)
F32 = np.float32
HELPERS = (
    "_compute_block_max_and_sumexp_excl",
    "_compute_global_max_and_correction",
    "_compute_single_lse_from_excl",
    "_log_softmax",
    "_stable_residual_logits",
    "_single_lse_residual_logits",
)


class _Ptr:
    def __init__(self, data: np.ndarray, offset=0):
        self.data = data
        self.offset = offset

    def __add__(self, other):
        return _Ptr(self.data, self.offset + other)

    __radd__ = __add__


def _load(ptr: _Ptr, mask=None, other=0.0):
    off = np.asarray(ptr.offset)
    if mask is None:
        return ptr.data[off]
    safe = np.where(mask, off, 0)
    return np.where(mask, ptr.data[safe], F32(other)).astype(F32)


def _max(x, axis=0, return_indices=False):
    m = np.max(x, axis=axis)
    return (m, np.argmax(x, axis=axis)) if return_indices else m


class _Arr(np.ndarray):
    def to(self, dtype):
        return np.asarray(self, dtype=dtype).view(_Arr)


def _wrap(x):
    return np.asarray(x, dtype=F32).view(_Arr)


def _tree() -> ast.Module:
    return ast.parse(PATH.read_text())


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _env():
    tree = _tree()
    body: list[ast.stmt] = []
    for name in HELPERS:
        fn = _func(tree, name)
        fn.decorator_list = []
        body.append(fn)
    env = {
        "tl": SimpleNamespace(
            max=_max,
            sum=lambda x, axis=None: _wrap(np.sum(x, axis=axis, dtype=F32)),
            where=lambda c, a, b: _wrap(np.where(c, a, b)),
            exp=lambda x: _wrap(np.exp(x, dtype=F32)),
            log=lambda x: _wrap(np.log(x, dtype=F32)),
            arange=lambda a, b: np.arange(a, b),
            load=_load,
            constexpr=int,
            float32=F32,
            int32=np.int32,
        ),
        "tldevice": SimpleNamespace(
            log1p=lambda x: _wrap(np.log1p(x, dtype=F32)),
            expm1=lambda x: _wrap(np.expm1(x, dtype=F32)),
        ),
    }
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(PATH), "exec"), env)
    return env


ENV = _env()


def _stats(logits: np.ndarray, block_size: int):
    vocab = logits.shape[0]
    nblocks = (vocab + block_size - 1) // block_size
    padded = 1 << (nblocks - 1).bit_length()
    maxes = np.full(padded, -np.inf, dtype=F32)
    excl = np.zeros(padded, dtype=F32)
    for b in range(nblocks):
        chunk = np.full(block_size, -np.inf, dtype=F32)
        part = logits[b * block_size : (b + 1) * block_size]
        chunk[: part.shape[0]] = part
        maxes[b], excl[b] = ENV["_compute_block_max_and_sumexp_excl"](
            _wrap(chunk), block_size
        )
    gmax, corr = ENV["_compute_global_max_and_correction"](
        _Ptr(maxes), 0, _Ptr(excl), 0, 0, nblocks, padded
    )
    lse = ENV["_compute_single_lse_from_excl"](
        _Ptr(maxes), 0, _Ptr(excl), 0, 0, nblocks, padded
    )
    return F32(gmax), F32(corr), F32(lse)


def _residuals(target: np.ndarray, draft: np.ndarray, block_size: int = 8):
    t_max, t_corr, t_lse = _stats(target, block_size)
    d_max, d_corr, d_lse = _stats(draft, block_size)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        stable = ENV["_stable_residual_logits"](
            _wrap(target), _wrap(draft), t_max, t_corr, d_max, d_corr
        )
        single = ENV["_single_lse_residual_logits"](
            _wrap(target), _wrap(draft), t_lse, d_lse
        )
    return np.asarray(stable), np.asarray(single)


def _reference(target: np.ndarray, draft: np.ndarray) -> np.ndarray:
    """Normalized residual max(p - q, 0) in float64."""

    def probs(x):
        x = x.astype(np.float64)
        e = np.exp(x - np.max(x))
        return e / e.sum()

    r = np.maximum(probs(target) - probs(draft), 0.0)
    return r / r.sum()


def _normalized(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    if not np.isfinite(x).any():
        return np.zeros_like(x)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _row(values: dict[int, float], vocab: int = 16) -> np.ndarray:
    row = np.full(vocab, -np.inf, dtype=F32)
    for token, logit in values.items():
        row[token] = logit
    return row


@pytest.mark.parametrize("gap", [10.0, 14.0, 20.0, 25.0, 60.0])
def test_single_token_support_keeps_stable_residual(gap):
    target = _row({3: 50.0})
    draft = _row({3: 50.0, 1: 50.0 - gap})
    stable, single = _residuals(target, draft)
    assert np.argmax(stable) == 3 and np.isfinite(stable[3])
    assert np.all(np.isneginf(np.delete(stable, 3)))
    assert stable[3] == pytest.approx(-gap, abs=1e-3)
    if gap >= 14.0:
        # The single logsumexp loses it: this was the token 0 bug.
        assert np.all(np.isneginf(single))


def test_two_tied_tokens_need_the_second_tier():
    target = _row({3: 0.0, 4: 0.0})
    draft = _row({3: 10.0, 4: 10.0, 1: -10.0})
    stable, single = _residuals(target, draft)
    assert np.all(np.isneginf(stable))
    assert np.all(np.isfinite(single[[3, 4]]))
    assert single[3] == single[4]


def test_equal_distributions_need_the_target_fallback():
    support = {3: 0.5, 9: 0.3, 14: 0.2}
    target = _row({t: float(np.log(p)) for t, p in support.items()})
    draft = target.copy()
    draft[1] = np.max(target) - 40.0
    stable, single = _residuals(target, draft)
    assert np.all(np.isneginf(stable)) and np.all(np.isneginf(single))


def test_asymmetric_near_equal_residual():
    target = _row({3: -2.625, 4: -1.375, 5: -4.0})
    draft = _row({3: 7.375004768371582, 4: 8.625005722045898, 5: 6.000017166137695})
    ref = _reference(target, draft)
    stable, single = _residuals(target, draft)
    tv_stable = 0.5 * np.abs(_normalized(stable) - ref).sum()
    tv_single = 0.5 * np.abs(_normalized(single) - ref).sum()
    assert tv_single > 0.4
    assert tv_stable < 0.02


@pytest.mark.parametrize("block_size", [8, 64])
@pytest.mark.parametrize("seed", range(5))
def test_matches_float64_reference(block_size, seed):
    rng = np.random.default_rng(seed)
    vocab = 300
    target = (rng.standard_normal(vocab) * 4).astype(F32)
    draft = (rng.standard_normal(vocab) * 4).astype(F32)
    target[rng.choice(vocab, 50, replace=False)] = -np.inf
    draft[rng.choice(vocab, 50, replace=False)] = -np.inf
    target[7] = target[8] = np.max(target) + 1  # tied maxima in one block
    stable, _ = _residuals(target, draft, block_size)
    ref = _reference(target, draft)
    assert not np.any(np.isnan(stable))
    assert np.all(np.isneginf(stable[ref == 0]))
    assert 0.5 * np.abs(_normalized(stable) - ref).sum() < 1e-4


def test_all_masked_row_is_neutral():
    row = np.full(64, -np.inf, dtype=F32)
    with np.errstate(invalid="ignore"):
        gmax, corr, _ = _stats(row, 16)
    assert np.isneginf(gmax)
    assert corr == 0.0


def test_accept_log_ratio_keeps_small_rejection_probability():
    """Accept uses the same stable log probs as the residual. With target
    mass 1 - d on the drafted token (d = exp(-20)) and the draft certain of
    it, the true rejection probability is d. The stable log probs keep
    log p - log q = log(1 - d) < 0; a single FP32 logsumexp rounds it to 0."""
    target = _row({3: 0.0, 4: -20.0})
    draft = _row({3: 0.0})
    t_max, t_corr, t_lse = _stats(target, 8)
    d_max, d_corr, d_lse = _stats(draft, 8)
    stable = ENV["_log_softmax"](_wrap(target), t_max, t_corr) - ENV["_log_softmax"](
        _wrap(draft), d_max, d_corr
    )
    single = (target - t_lse) - (draft - d_lse)
    assert stable[3] == pytest.approx(-np.exp(-20.0), rel=1e-3)
    assert single[3] == 0.0


def test_rejection_kernel_accepts_with_stable_log_probs():
    fn = _func(_tree(), "_rejection_kernel")
    text = ast.unparse(fn)
    assert (
        "target_log_prob = _log_softmax(target_logit, target_max, "
        "target_correction)" in text
    )
    assert (
        "draft_log_prob = _log_softmax(draft_logit, draft_max, draft_correction)"
        in text
    )
    assert "accepted &= target_log_prob > tl.log(u) + draft_log_prob" in text
    calls = [
        c
        for c in _calls(fn, "_compute_global_max_and_correction")
        if ast.unparse(c.args[2]).endswith("local_sumexp_ptr")
    ]
    assert len(calls) == 2


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == name
    ]


def test_resample_kernel_uses_stable_residual():
    fn = _func(_tree(), "_resample_kernel")
    assert _calls(fn, "_stable_residual_logits")
    assert not _calls(fn, "_single_lse_residual_logits")


def test_insert_kernel_tiers():
    fn = _func(_tree(), "_insert_resampled_kernel")
    guard = next(
        n
        for n in fn.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "HAS_FALLBACK"
    )
    text = ast.unparse(guard)
    assert "num_residual_blocks == 0" in text
    calls = _calls(guard, "_row_gumbel_argmax")
    modes = [
        next(ast.unparse(k.value) for k in c.keywords if k.arg == "SINGLE_LSE_RESIDUAL")
        for c in calls
    ]
    # Second tier first, then the target distribution.
    assert modes == ["True", "False"]
    assert calls[0].lineno < calls[1].lineno
    # The token is stored once, after the fallbacks.
    stores = [
        n
        for n in fn.body
        if isinstance(n, ast.Expr) and ast.unparse(n.value).startswith("tl.store")
    ]
    assert stores and stores[-1].lineno > guard.lineno


def test_launcher_enables_fallback_with_draft_logits():
    fn = _func(_tree(), "rejection_sample")
    launches = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Subscript)
        and ast.unparse(n.func.value) == "_insert_resampled_kernel"
    ]
    assert len(launches) == 1
    kwargs = {k.arg: ast.unparse(k.value) for k in launches[0].keywords}
    assert kwargs["HAS_FALLBACK"] == "has_draft_logits"
    assert kwargs["num_warps"] == "1"
