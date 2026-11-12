# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded multi-request admission, row isolation and joint-distribution checks."""

import itertools
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.test_dflash2_native_lookup import (
    native_config,
    neural_q,
    reference_lookup,
    target_p,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.native_lookup import (
    NativeDFlash2Lookup,
    _native_fuse_kernel,
)

DEVICE = torch.device(os.environ.get("W21_TEST_DEVICE", "cpu"))
pytestmark = pytest.mark.skip_global_cleanup


def multi_fixture(n=3, limit=3, padded=4):
    config = native_config(nmin=2, nmax=4).model_copy(
        update={"max_requests": limit, "admission_version": 1 if limit == 1 else 2}
    )
    stats = SimpleNamespace(counters=torch.zeros(18, dtype=torch.uint64, device=DEVICE))
    native = NativeDFlash2Lookup(config, padded, 64, DEVICE, stats=stats)
    history = torch.zeros((8, 64), dtype=torch.int32, device=DEVICE)
    lengths = torch.zeros(8, dtype=torch.int32, device=DEVICE)
    # Batch row, persistent slot and token ID are deliberately different.
    slots = [5, 1, 7, 3][:n]
    for row, slot in enumerate(slots):
        a, b, c, d = 2 + row * 4, 3 + row * 4, 4 + row * 4, 5 + row * 4
        sequence = [a, b, c, d, a, b]
        history[slot, :6] = torch.tensor(sequence, device=DEVICE)
        lengths[slot] = 6
    states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=history),
        total_len=SimpleNamespace(gpu=lengths),
    )
    mapping = torch.full((padded * 7,), -1, dtype=torch.int32, device=DEVICE)
    mapping[: n * 7] = torch.tensor(
        slots, dtype=torch.int32, device=DEVICE
    ).repeat_interleave(7)
    draft = torch.full((padded, 7), 15, dtype=torch.int64, device=DEVICE)
    for row in range(n):
        draft[row, 0] = 4 + row * 4
    ids = torch.arange(16, device=DEVICE).repeat(8, 7, 1)
    weights = torch.arange(1, 17, device=DEVICE).double()
    q = torch.stack([(weights + slot) / (weights + slot).sum() for slot in range(8)])
    logits = q.log()[:, None, :].repeat(1, 7, 1)
    scores = logits.clone()
    batch = SimpleNamespace(
        num_reqs=n,
        has_structured_output_reqs=False,
        is_prefilling_np=np.zeros(n, dtype=bool),
    )
    sampled = torch.ones(padded, dtype=torch.int32, device=DEVICE)
    return native, states, mapping, draft, logits, ids, scores, batch, sampled


def apply_fixture(parts, width=7, real=True):
    native, states, mapping, draft, logits, ids, scores, batch, sampled = parts
    native.prepare(batch, sampled, real_run=real, draft_width=width)
    native.apply(states, mapping, draft, len(draft), logits, ids, scores)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("sparse", [False, True])
def test_multi_row_history_q_and_exact_counts(n, sparse):
    parts = list(multi_fixture(n=n))
    native, states, mapping, draft, logits, ids, scores, batch, sampled = parts
    if not sparse:
        parts[6] = None
    original = [x.clone() for x in (draft, logits, ids, scores)]
    rng = torch.get_rng_state().clone()
    apply_fixture(parts)
    assert native.eligible.tolist() == [1] * n + [0] * (4 - n), "MULTI_ADMISSION"
    assert torch.equal(torch.get_rng_state(), rng), "MULTI_RNG"
    live_slots = []
    for row in range(n):
        slot = int(mapping[row * 7])
        live_slots.append(slot)
        sequence = states.all_token_ids.gpu[slot, :6].tolist()
        tokens, match, valid = reference_lookup(sequence, 2, 4, 64)
        assert native.tokens[row].tolist() == tokens, "MULTI_HISTORY"
        assert int(native.match[row]) == match
        assert int(native.valid[row]) == valid
        assert (
            draft[row].tolist()
            == [original[0][row, 0].item()]
            + tokens[1:valid]
            + original[0][row, valid:].tolist()
        ), "MULTI_FUSION_ROW"
        assert torch.equal(logits[slot, 0], original[1][slot, 0]), "MULTI_FIRST_Q"
        for step in range(1, valid):
            assert torch.isfinite(logits[slot, step]).nonzero().flatten().tolist() == [
                tokens[step]
            ], "MULTI_Q_SLOT"
            assert logits[slot, step, tokens[step]] == 0
            if sparse:
                assert int(ids[slot, step, 0]) == tokens[step], "MULTI_SPARSE_SLOT"
                assert scores[slot, step, 0] == 0
                assert torch.isneginf(scores[slot, step, 1:]).all()
        assert torch.equal(logits[slot, valid:], original[1][slot, valid:])
    for slot in set(range(8)) - set(live_slots):
        assert torch.equal(logits[slot], original[1][slot]), "MULTI_UNUSED_SLOT"
    assert int(native.stats.counters[0]) == n, "MULTI_ROW_COUNTS"
    assert native.stats.counters.tolist()[:4] == [n] * 4, "MULTI_ROW_COUNTS"
    assert sum(native.stats.counters.tolist()[4:17]) == n
    assert int(native.stats.counters[17]) == n * 2
    assert not native.use[n:].any()
    assert not native.valid[n:].any()


@pytest.mark.parametrize(
    "reason",
    [
        "prefill",
        "grammar",
        "no_sample",
        "no_space",
        "short",
        "invalid_slot",
        "width",
        "limit",
        "profile",
    ],
)
def test_mixed_batch_is_all_or_none_and_recovers(reason):
    parts = multi_fixture(n=3)
    native, states, mapping, draft, logits, ids, scores, batch, sampled = parts
    original = [x.clone() for x in (draft, logits, ids, scores)]
    if reason == "prefill":
        batch.is_prefilling_np[1] = True
    elif reason == "grammar":
        batch.has_structured_output_reqs = True
    elif reason == "no_sample":
        sampled[1] = 0
    elif reason == "no_space":
        states.total_len.gpu[1] = 64
    elif reason == "short":
        states.total_len.gpu[1] = 2
    elif reason == "invalid_slot":
        mapping[7:14] = -1
    elif reason == "limit":
        batch.num_reqs = 4
        batch.is_prefilling_np = np.zeros(4, dtype=bool)
    apply_fixture(parts, width=3 if reason == "width" else 7, real=reason != "profile")
    assert native.eligible.tolist() == [0] * 4, "MIXED_BATCH_BYPASS"
    assert not native.use.any()
    assert not native.tokens.any()
    assert native.stats.counters.tolist() == [0] * 18
    for actual, expected in zip((draft, logits, ids, scores), original):
        assert torch.equal(actual, expected)
    # Restore a clean batch without recreating the instance or a cooldown.
    fresh = multi_fixture(n=3)
    batch.num_reqs = 3
    batch.is_prefilling_np = np.zeros(3, dtype=bool)
    batch.has_structured_output_reqs = False
    mapping.copy_(fresh[2])
    states.total_len.gpu.copy_(fresh[1].total_len.gpu)
    sampled.fill_(1)
    apply_fixture(parts)
    assert native.eligible.tolist() == [1, 1, 1, 0], "NO_ADMISSION_HYSTERESIS"


@pytest.mark.parametrize("n", [1, 2, 3])
def test_single_request_fallback_topology_and_c1_parity(n):
    old = multi_fixture(n=n, limit=1)
    new = multi_fixture(n=n, limit=3)
    apply_fixture(old)
    apply_fixture(new)
    assert old[0].control.numel() == 2, "FALLBACK_RESOURCES"
    assert new[0].control.numel() == 3
    if n == 1:
        for i in (3, 4, 5, 6):
            assert torch.equal(old[i], new[i]), "C1_TOKEN_Q_PARITY"
        for name in ("tokens", "use", "match", "valid", "eligible", "sampled"):
            assert torch.equal(getattr(old[0], name), getattr(new[0], name)), (
                "C1_SCRATCH_PARITY"
            )
        assert torch.equal(old[0].stats.counters, new[0].stats.counters), (
            "C1_COUNT_PARITY"
        )
    else:
        assert not old[0].eligible.any(), "FALLBACK_SINGLE_REQUEST"
        assert old[0].stats.counters.tolist() == [0] * 18
        assert new[0].eligible[:n].tolist() == [1] * n


def row_distribution(neural, q, draft):
    """Independent rejection reference; integrate continuation after first rejection."""
    result: dict[tuple[int, ...], float] = {}
    accepted_mass = 1.0
    prefix: tuple[int, ...] = ()
    for step in range(3):
        p = target_p(prefix)
        token = draft[step]
        accept = min(1.0, p[token] / q[step][token])
        residual = [max(p[t] - q[step][t], 0.0) for t in (0, 1)]
        norm = sum(residual)
        if norm > 0:
            for tail in itertools.product((0, 1), repeat=3 - step):
                mass = accepted_mass * (1 - accept) * residual[tail[0]] / norm
                sequence: tuple[int, ...] = prefix + (tail[0],)
                for t in tail[1:]:
                    mass *= target_p(sequence)[t]
                    sequence += (t,)
                result[sequence] = result.get(sequence, 0.0) + mass
        accepted_mass *= accept
        prefix += (token,)
    result[prefix] = result.get(prefix, 0.0) + accepted_mass
    return result


@pytest.mark.parametrize("rows", [2, 3])
@pytest.mark.parametrize("sparse", [False, True])
def test_multi_row_exact_joint_distribution(rows, sparse):
    sequences = list(itertools.product((0, 1), repeat=3))
    joint_paths = list(itertools.product(sequences, repeat=rows))
    originals = [p for trial in joint_paths for p in trial]
    n = len(originals)
    draft = torch.tensor(
        [list(p) + [0] * 4 for p in originals], dtype=torch.int64, device=DEVICE
    )
    q = torch.ones(n, 7, 2, dtype=torch.float64, device=DEVICE) * 0.5
    for i, p in enumerate(originals):
        for step in range(3):
            q[i, step] = torch.tensor(
                neural_q(step, p[step - 1] if step else 0),
                dtype=torch.float64,
                device=DEVICE,
            )
    logits = q.log()
    ids = torch.arange(2, device=DEVICE).repeat(n, 7, 1)
    scores = logits.clone()
    lookup = torch.tensor(
        [[0, 1, 1, 0, 1, 0, 1]], dtype=torch.int32, device=DEVICE
    ).repeat(n, 1)
    match = torch.full((n,), 4, dtype=torch.int32, device=DEVICE)
    valid = torch.tensor(
        [2 if i % rows == 0 else 7 for i in range(n)], dtype=torch.int32, device=DEVICE
    )
    use = torch.zeros_like(draft, dtype=torch.int32)
    _native_fuse_kernel[(n,)](
        draft,
        7,
        lookup,
        7,
        match,
        valid,
        torch.ones_like(match),
        use,
        NMIN=2,
        num_warps=1,
    )
    mapping = torch.arange(n, dtype=torch.int32, device=DEVICE).repeat_interleave(7)
    _point_mass_draft_logits_kernel[(n * 7,)](
        logits,
        ids,
        scores,
        draft,
        7,
        use,
        mapping,
        7,
        ids.stride(0),
        ids.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_steps=7,
        top_k=2,
        BLOCK_K=2,
        CACHE_SCORES=True,
        num_warps=1,
    )
    actual_q = (
        (
            torch.zeros_like(logits).scatter_add_(-1, ids, scores.softmax(-1))
            if sparse
            else logits.exp()
        )
        .cpu()
        .tolist()
    )
    proposals = draft.cpu().tolist()
    joint: dict[tuple[tuple[int, ...], ...], float] = {}
    for trial_id, paths in enumerate(joint_paths):
        distributions = [
            row_distribution(
                p, actual_q[trial_id * rows + r], proposals[trial_id * rows + r]
            )
            for r, p in enumerate(paths)
        ]
        neural_mass = math.prod(
            math.prod(
                neural_q(step, p[step - 1] if step else 0)[t]
                for step, t in enumerate(p)
            )
            for p in paths
        )
        for outcomes in itertools.product(*(d.items() for d in distributions)):
            key = tuple(o[0] for o in outcomes)
            joint[key] = joint.get(key, 0.0) + neural_mass * math.prod(
                o[1] for o in outcomes
            )
    error = 0.0
    for key in itertools.product(sequences, repeat=rows):
        expected = math.prod(
            math.prod(target_p(p[:step])[t] for step, t in enumerate(p)) for p in key
        )
        error = max(error, abs(joint.get(key, 0.0) - expected))
    assert error <= 1e-9, ("MULTI_JOINT_UNBIASED", error)
    assert abs(sum(joint.values()) - 1) <= 1e-9


@pytest.mark.skipif(DEVICE.type != "cuda", reason="real CUDA capture/replay")
@pytest.mark.parametrize("limit", [1, 3])
def test_multi_request_real_capture_replay(limit):
    parts = multi_fixture(n=3 if limit == 3 else 1, limit=limit)
    native, states, mapping, draft, logits, ids, scores, batch, sampled = parts
    native.reset_for_capture()
    # Compile all kernels before graph capture; zero recording suppresses counters.
    native.apply(states, mapping, draft, 4, logits, ids, scores)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.apply(states, mapping, draft, 4, logits, ids, scores)
    native.prepare(batch, sampled, real_run=True)
    graph.replay()
    torch.accelerator.synchronize()
    assert native.eligible.tolist() == [1] * batch.num_reqs + [0] * (4 - batch.num_reqs)
    assert native.stats.counters.cpu().tolist()[:4] == [batch.num_reqs] * 4
    # Invalidate one live row, then restore; no recapture and no stale admission.
    sampled[0] = 0
    native.prepare(batch, sampled, real_run=True)
    graph.replay()
    torch.accelerator.synchronize()
    assert not native.eligible.any()
    assert native.stats.counters.cpu().tolist()[:4] == [batch.num_reqs] * 4


@pytest.mark.parametrize("n", [2, 3])
def test_configured_two_request_limit(n):
    parts = multi_fixture(n=n, limit=2)
    apply_fixture(parts)
    expected = n if n == 2 else 0
    assert parts[0].eligible.tolist() == [int(expected > 0)] * n + [0] * (4 - n)
    assert int(parts[0].stats.counters[0]) == expected


def test_first_token_gate_remains_per_row():
    parts = multi_fixture(n=3)
    parts[3][1, 0] = 0  # One row disagrees; admission of other rows stays valid.
    original_draft = parts[3].clone()
    original_q = parts[4][1].clone()
    apply_fixture(parts)
    assert parts[0].eligible.tolist() == [1, 1, 1, 0]
    assert parts[0].stats.counters.tolist()[:4] == [3, 3, 2, 2]
    assert torch.equal(parts[3][1], original_draft[1])
    assert torch.equal(parts[4][1], original_q)
    assert parts[0].use[0, 1:].any() and parts[0].use[2, 1:].any()
    assert not parts[0].use[1].any()
