# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native lookup components on CPU interpreter or explicitly selected CUDA."""

import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.dflash2_lookup import LookupSource, ResolvedDFlash2Lookup
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (
    _point_mass_draft_logits_kernel,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.native_lookup import (
    NativeDFlash2Lookup,
    _native_fuse_kernel,
)

DEVICE = os.environ.get("W21_TEST_DEVICE", "cpu")
pytestmark = pytest.mark.skip_global_cleanup


def native_config(nmin=2, nmax=4, window=64):
    fixture = Path(__file__).parents[2] / "config/data/dflash2_lookup_baseline.json"
    factors = json.loads(fixture.read_text())["lookup_env_defaults"]
    return ResolvedDFlash2Lookup(
        mode="gpu_head",
        policy="longest",
        nmin=nmin,
        nmax=nmax,
        search_window=window,
        draft_tokens=7,
        verify_tokens=7,
        env_factors=tuple(sorted(factors.items())),
        source=LookupSource(
            root="object",
            keys=("enabled", "nmin", "nmax", "policy"),
            expand_keys=(),
            ngram_assist=None,
            prompt_fields=(),
            environment=(),
        ),
    )


def reference_lookup(sequence, nmin, nmax, window):
    end = len(sequence) - 1
    best = (-1, -1)
    for candidate in range(max(nmin - 1, len(sequence) - window), end):
        length = 0
        while (
            length < nmax
            and candidate - length >= 0
            and sequence[candidate - length] == sequence[end - length]
        ):
            length += 1
        if length >= nmin:
            best = max(best, (length, candidate))
    if best[0] < 0:
        return [0] * 7, 0, 0
    valid = min(7, end - best[1])
    tokens = sequence[best[1] + 1 : best[1] + 1 + valid]
    return tokens + [0] * (7 - valid), best[0], valid


def fixture_components(sequence, *, nmin=2, nmax=4, window=64):
    device = torch.device(DEVICE)
    native = NativeDFlash2Lookup(native_config(nmin, nmax, window), 2, 2048, device)
    history = torch.zeros((3, 2048), dtype=torch.int32, device=device)
    history[2, : len(sequence)] = torch.tensor(sequence, device=device)
    lengths = torch.tensor([0, 0, len(sequence)], dtype=torch.int32, device=device)
    states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=history),
        total_len=SimpleNamespace(gpu=lengths),
    )
    mapping = torch.full((14,), -1, dtype=torch.int32, device=device)
    mapping[:7] = 2
    draft = torch.tensor(
        [[9, 1, 2, 3, 4, 5, 6], [5] * 7], dtype=torch.int64, device=device
    )
    ids = torch.arange(4, device=device).repeat(3, 7, 1)
    logits = torch.full((3, 7, 16), -float("inf"), device=device)
    logits[:, :, :4] = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device).log()
    scores = logits[:, :, :4].clone()
    batch = SimpleNamespace(
        num_reqs=1, has_structured_output_reqs=False, is_prefilling_np=np.array([False])
    )
    sampled = torch.ones(2, dtype=torch.int32, device=device)
    return native, states, mapping, draft, logits, ids, scores, batch, sampled


@pytest.mark.parametrize("window", [2, 4, 64])
@pytest.mark.parametrize(
    "sequence",
    [
        [0, 0],
        [1, 2, 3, 4, 1, 2, 3],
        [4, 5, 9, 8, 4, 5],
        [1, 2, 3, 4, 1, 2, 3, 9, 1, 2, 3],
        [1, 2, 3, 4, 5],
    ],
)
def test_longest_matches_reference_and_clears_scratch(sequence, window):
    native, states, mapping, draft, logits, ids, scores, batch, sampled = (
        fixture_components(sequence, nmin=1, nmax=4, window=window)
    )
    native.tokens.fill_(99)
    native.match.fill_(99)
    native.valid.fill_(99)
    native.use.fill_(99)
    native.prepare(batch, sampled, real_run=True)
    native.apply(states, mapping, draft, 2, logits, ids, scores)
    expected, match, valid = reference_lookup(sequence, 1, 4, window)
    assert native.tokens[0].tolist() == expected
    assert native.match.tolist() == [match, 0]
    assert native.valid.tolist() == [valid, 0]
    assert native.tokens[1].tolist() == [0] * 7
    assert native.use[1].tolist() == [0] * 7


@pytest.mark.parametrize(
    "reason",
    [
        "multi",
        "prefill",
        "grammar",
        "no_sample",
        "negative",
        "out_of_range",
        "too_long",
        "profile",
        "capture",
    ],
)
def test_admission_bypasses_each_ineligible_step_and_recovers(reason):
    native, states, mapping, draft, logits, ids, scores, batch, sampled = (
        fixture_components([4, 5, 9, 8, 4, 5])
    )
    before = draft.clone(), logits.clone(), ids.clone(), scores.clone()
    if reason == "multi":
        batch.num_reqs = 2
    elif reason == "prefill":
        batch.is_prefilling_np[0] = True
    elif reason == "grammar":
        batch.has_structured_output_reqs = True
    elif reason == "no_sample":
        sampled.zero_()
    elif reason == "negative":
        mapping[:7] = -1
    elif reason == "out_of_range":
        mapping[:7] = 99
    elif reason == "too_long":
        states.total_len.gpu[2] = 2048
    native.prepare(batch, sampled, real_run=reason != "profile")
    if reason == "capture":
        native.reset_for_capture()
    native.apply(states, mapping, draft, 2, logits, ids, scores)
    for actual, expected in zip((draft, logits, ids, scores), before):
        assert torch.equal(actual, expected)
    assert native.eligible.tolist() == [0, 0]
    assert not native.use.any()
    assert not native.valid.any()
    batch.num_reqs = 1
    batch.is_prefilling_np[0] = False
    batch.has_structured_output_reqs = False
    sampled.fill_(1)
    mapping[:7] = 2
    states.total_len.gpu[2] = 6
    native.prepare(batch, sampled, real_run=True)
    native.apply(states, mapping, draft, 2, logits, ids, scores)
    assert native.eligible.tolist() == [1, 0]
    assert draft[0].tolist() == [9, 8, 4, 5, 4, 5, 6]
    assert native.use[0].tolist() == [0, 1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("valid", [0, 1, 2, 4, 7])
@pytest.mark.parametrize("agree", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_strict_gate_short_valid_and_q(valid, agree, sparse):
    native, states, mapping, draft, logits, ids, scores, _, _ = fixture_components(
        [0, 0]
    )
    # Independent component trials enter fusion after admission has completed.
    native.eligible[0] = 1
    native.tokens[0] = torch.tensor([0, 7, 0, 8, 9, 10, 11], device=DEVICE)
    native.match[0] = 100  # A strong match still cannot bypass the first token.
    native.valid[0] = valid
    draft[0, 0] = 0 if agree else 1
    before = draft.clone(), logits.clone(), ids.clone(), scores.clone()
    _native_fuse_kernel[(2,)](
        draft,
        draft.stride(0),
        native.tokens,
        7,
        native.match,
        native.valid,
        native.eligible,
        native.use,
        NMIN=2,
        num_warps=1,
    )
    _point_mass_draft_logits_kernel[(14,)](
        logits,
        ids,
        scores,
        draft,
        7,
        native.use,
        mapping,
        7,
        ids.stride(0),
        ids.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_steps=7,
        top_k=4,
        BLOCK_K=4,
        CACHE_SCORES=sparse,
        num_warps=1,
    )
    assert torch.equal(logits[2, 0], before[1][2, 0])
    assert torch.equal(ids[2, 0], before[2][2, 0])
    assert torch.equal(scores[2, 0], before[3][2, 0])
    for step in range(7):
        changed = agree and 0 < step < valid
        assert int(native.use[0, step]) == int(changed)
        if changed:
            token = int(native.tokens[0, step])
            assert int(draft[0, step]) == token
            assert torch.where(torch.isfinite(logits[2, step]))[0].tolist() == [token]
            assert float(logits[2, step, token]) == 0.0
            if sparse:
                assert int(ids[2, step, 0]) == token
                assert float(scores[2, step, 0]) == 0.0
                assert torch.isneginf(scores[2, step, 1:]).all()
        else:
            assert torch.equal(logits[2, step], before[1][2, step])
            assert torch.equal(draft[0, step], before[0][0, step])


def neural_q(step, previous):
    if step == 0:
        return (0.75, 0.25)
    return (0.9, 0.1) if previous == 0 else (0.15, 0.85)


def target_p(prefix):
    if not prefix:
        return (0.65, 0.35)
    return (0.3, 0.7) if prefix[-1] == 0 else (0.8, 0.2)


def extend_target(prefix, mass, horizon, output):
    if len(prefix) == horizon:
        output[prefix] = output.get(prefix, 0.0) + mass
        return
    for token, probability in enumerate(target_p(prefix)):
        extend_target(prefix + (token,), mass * probability, horizon, output)


def exact_distribution(valid, defect=None, sparse=False):
    sequences = list(itertools.product(range(2), repeat=7))
    original = torch.tensor(sequences, dtype=torch.int64, device=DEVICE)
    proposals = original.clone()
    q = torch.tensor(
        [[neural_q(i, row[i - 1] if i else 0) for i in range(7)] for row in sequences],
        dtype=torch.float64,
        device=DEVICE,
    )
    dense = q.log()
    ids = torch.arange(2, device=DEVICE).repeat(len(sequences), 7, 1)
    cached_scores = dense.clone()
    lookup = torch.tensor(
        [0, 1, 1, 0, 1, 0, 1], dtype=torch.int32, device=DEVICE
    ).repeat(len(sequences), 1)
    match = torch.full((len(sequences),), 4, dtype=torch.int32, device=DEVICE)
    lengths = torch.full_like(match, valid)
    use = torch.zeros_like(proposals, dtype=torch.int32)
    eligible = torch.ones_like(match)
    _native_fuse_kernel[(len(sequences),)](
        proposals, 7, lookup, 7, match, lengths, eligible, use, NMIN=2, num_warps=1
    )
    mapping = torch.arange(
        len(sequences), device=DEVICE, dtype=torch.int32
    ).repeat_interleave(7)
    _point_mass_draft_logits_kernel[(len(sequences) * 7,)](
        dense,
        ids,
        cached_scores,
        proposals,
        7,
        use,
        mapping,
        7,
        ids.stride(0),
        ids.stride(1),
        dense.stride(0),
        dense.stride(1),
        num_steps=7,
        top_k=2,
        BLOCK_K=2,
        CACHE_SCORES=True,
        num_warps=1,
    )
    sparse_q = torch.zeros_like(dense).scatter_add_(-1, ids, cached_scores.softmax(-1))
    assert torch.allclose(dense.exp(), sparse_q, atol=1e-15, rtol=0)
    actual_q = (sparse_q if sparse else dense.exp()).cpu().tolist()
    actual_d = proposals.cpu().tolist()
    original_q = q.cpu().tolist()
    output: dict[tuple[int, ...], float] = {}
    for row_index, neural in enumerate(sequences):
        d, qs = actual_d[row_index], actual_q[row_index]
        mass = float(np.prod([original_q[row_index][i][neural[i]] for i in range(7)]))
        if defect == "first_mass" and valid > 0 and neural[0] == 0:
            qs[0] = [1.0, 0.0]
        if defect == "conditional_suffix" and valid > 1 and neural[0] == 0:
            for i in range(valid, 7):
                qs[i] = list(neural_q(i, d[i - 1]))
        prefix: tuple[int, ...] = ()
        for i in range(7):
            p = target_p(prefix)
            accept = min(1.0, p[d[i]] / qs[i][d[i]])
            rejected = mass * (1.0 - accept)
            if rejected > 1e-18:
                residual = [max(p[t] - qs[i][t], 0.0) for t in range(2)]
                norm = sum(residual)
                for token in range(2):
                    if residual[token] > 0:
                        extend_target(
                            prefix + (token,),
                            rejected * residual[token] / norm,
                            8,
                            output,
                        )
            mass *= accept
            prefix += (d[i],)
        extend_target(prefix, mass, 8, output)
    expected: dict[tuple[int, ...], float] = {}
    extend_target((), 1.0, 8, expected)
    error = max(
        abs(output.get(tokens, 0.0) - probability)
        for tokens, probability in expected.items()
    )
    return output, expected, error


@pytest.mark.parametrize("valid", [0, 1, 2, 4, 7])
@pytest.mark.parametrize("sparse", [False, True])
def test_exact_joint_and_conditional_distribution(valid, sparse):
    output, expected, error = exact_distribution(valid, sparse=sparse)
    print(f"EXACT valid={valid} sparse={sparse} max_joint_error={error:.17g}")
    assert error <= 1e-9, error
    assert abs(sum(output.values()) - 1.0) <= 1e-9
    for length in range(1, 8):
        for prefix in itertools.product(range(2), repeat=length):
            actual = [
                sum(
                    p
                    for tokens, p in output.items()
                    if tokens[: length + 1] == prefix + (t,)
                )
                for t in range(2)
            ]
            normalizer = sum(actual)
            assert (
                max(abs(actual[t] / normalizer - target_p(prefix)[t]) for t in range(2))
                <= 1e-9
            )


@pytest.mark.parametrize("defect", ["first_mass", "conditional_suffix"])
def test_exact_enumeration_detects_biased_alternatives(defect):
    _, _, error = exact_distribution(2, defect)
    print(f"NEGATIVE defect={defect} max_joint_error={error:.17g}")
    assert error > 1e-3, error


def test_native_resources_have_no_expansion_controller():
    native = NativeDFlash2Lookup(native_config(), 2, 2048, torch.device(DEVICE))
    assert native.tokens.shape == native.use.shape == (2, 7)
    assert native.eligible.shape == native.valid.shape == native.match.shape == (2,)
    assert not any(
        "stream" in name
        or "event" in name
        or "streak" in name
        or "sticky" in name
        or "hits" in name
        for name in vars(native)
    )


def test_speculator_entry_and_closed_noop_preserve_rng(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    native, states, mapping, draft, logits, ids, scores, batch, sampled = (
        fixture_components([4, 5, 9, 8, 4, 5])
    )
    spec = object.__new__(DFlash2Speculator)
    spec._native_lookup, spec._lookup_enabled = native, False
    spec._req_states, spec.sample_idx_mapping = states, mapping
    spec.draft_tokens, spec.draft_logits = draft, logits
    spec._cached_candidate_ids, spec._cached_candidate_scores = ids, scores
    spec._debug_real_proposal = True
    spec.speculative_config = SimpleNamespace(
        num_speculative_tokens=7, dflash_concurrency_drafts=None
    )
    spec._prepare_proposal_runtime(batch, sampled, sampled)
    spec._apply_lookup(2)
    assert spec.next_num_draft_tokens() == 7
    assert native.use[0].tolist() == [0, 1, 1, 1, 0, 0, 0]
    spec._native_lookup = None
    before = [value.clone() for value in (draft, logits, ids, scores)]
    rng = torch.get_rng_state().clone()
    spec._apply_lookup(2)
    for actual, expected in zip((draft, logits, ids, scores), before):
        assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng)


def test_multi_request_batch_and_reused_slot_are_recomputed():
    native, states, mapping, draft, logits, ids, scores, batch, sampled = (
        fixture_components([4, 5, 9, 8, 4, 5])
    )
    native.prepare(batch, sampled, real_run=True)
    native.apply(states, mapping, draft, 2, logits, ids, scores)
    assert native.valid[0] == 4
    states.all_token_ids.gpu[1].copy_(states.all_token_ids.gpu[2])
    states.total_len.gpu[1] = 6
    mapping[7:] = 1
    batch.num_reqs = 2
    before = draft.clone(), logits.clone(), ids.clone(), scores.clone()
    native.prepare(batch, sampled, real_run=True)
    native.apply(states, mapping, draft, 2, logits, ids, scores)
    assert native.eligible.tolist() == [0, 0]
    for actual, expected in zip((draft, logits, ids, scores), before):
        assert torch.equal(actual, expected)
    # A different request reuses the first row. No stale hit or cooldown remains.
    batch.num_reqs = 1
    mapping[:7] = 1
    mapping[7:] = -1
    states.all_token_ids.gpu[1, :6] = torch.tensor([4, 5, 9, 3, 4, 5], device=DEVICE)
    draft[0, 0] = 9
    native.prepare(batch, sampled, real_run=True)
    native.apply(states, mapping, draft, 2, logits, ids, scores)
    assert native.tokens[0, :4].tolist() == [9, 3, 4, 5]
    assert draft[0, :4].tolist() == [9, 3, 4, 5]


def test_native_target_capture_descriptors_have_no_per_request_q16(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.config import CompilationConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    monkeypatch.setattr(
        "vllm.v1.attention.ops.sm70_e4m3_long.long_attention_enabled", lambda: False
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils._worker_device_is_pre_ampere", lambda: True
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_global_graph_pool",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_pp_group",
        lambda: MagicMock(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setenv("VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS", "1")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=list(range(8, 65, 8)),
            max_cudagraph_capture_size=64,
        ),
        speculative_config=SimpleNamespace(
            method="dflash",
            ngram_assist=False,
            num_speculative_tokens=7,
            dflash_concurrency_drafts=None,
        ),
    )
    manager = ModelCudaGraphManager(
        config, torch.device(DEVICE), CUDAGraphMode.FULL_DECODE_ONLY, decode_query_len=8
    )
    manager._graphs_captured = True
    assert max(manager.decode_query_lens) == 8
    assert 16 not in manager.decode_query_lens
    descriptors = [
        {
            "requests": d.num_reqs,
            "tokens": d.num_tokens,
            "query": d.uniform_token_count,
            "mode": str(d.cg_mode),
        }
        for descs in manager._capture_descs.values()
        for d in descs
    ]
    assert any(d["requests"] == 2 and d["tokens"] == 16 for d in descriptors)
    assert all(d["query"] != 16 for d in descriptors)
    _write_resource("target-capture-plan", {"descriptors": descriptors})


def _write_resource(name, data):
    output = os.environ.get("W21_RESOURCE_OUT")
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")


@pytest.mark.skipif(DEVICE != "cuda", reason="Coordinator single-card capture check")
def test_cuda_graph_replay_recomputes_native_admission():
    native, states, mapping, draft, logits, ids, scores, batch, sampled = (
        fixture_components([4, 5, 9, 8, 4, 5])
    )
    from vllm.v1.worker.gpu.buffer_utils import UvaBuffer

    history = UvaBuffer(states.all_token_ids.gpu.shape, torch.int32)
    history.cpu.copy_(states.all_token_ids.gpu.cpu())
    states.all_token_ids.gpu = history.uva
    baseline = [x.clone() for x in (draft, logits, ids, scores)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        native.prepare(batch, sampled, real_run=True)
        native.apply(states, mapping, draft, 2, logits, ids, scores)
    torch.cuda.current_stream().wait_stream(stream)
    native.reset_for_capture()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.apply(states, mapping, draft, 2, logits, ids, scores)
    for count in (1, 2, 1):
        batch.num_reqs = count
        for x, original in zip((draft, logits, ids, scores), baseline):
            x.copy_(original)
        native.prepare(batch, sampled, real_run=True)
        native.apply(states, mapping, draft, 2, logits, ids, scores)
        expected = [x.clone() for x in (draft, logits, ids, scores, native.use)]
        for x, original in zip((draft, logits, ids, scores), baseline):
            x.copy_(original)
        native.prepare(batch, sampled, real_run=True)
        graph.replay()
        for actual, eager in zip((draft, logits, ids, scores, native.use), expected):
            assert torch.equal(actual, eager)
        assert native.eligible.tolist() == [int(count == 1), 0]
    for _ in range(5):
        native.apply(states, mapping, draft, 2, logits, ids, scores)
    samples = []
    for _ in range(30):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        native.apply(states, mapping, draft, 2, logits, ids, scores)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    _write_resource(
        "native-timing",
        {
            "warmup": 5,
            "samples_ms": samples,
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "scope": "eligibility + suffix + fuse + q kernels, UVA history",
        },
    )
    _write_resource(
        "native-capture",
        {
            "graphs": ["native-lookup padded_requests=2 K=7"],
            "buffers": {
                name: list(value.shape)
                for name, value in vars(native).items()
                if isinstance(value, torch.Tensor)
            },
            "admission_sequence": [1, 2, 1],
        },
    )


@pytest.mark.skipif(DEVICE != "cuda", reason="Coordinator delayed GDN graph pool check")
def test_delayed_gdn_replay_pool_and_real_buffer_geometry():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.worker.gpu.gdn_spec_replay import (
        GDNSpecReplayGraphs,
        GDNSpecReplayLayer,
        GDNSpecReplayStep,
        build_replay_state_indices,
        split_replay_launch,
    )

    # Use the actual layer allocation entry point with small TP1 dimensions.
    layer = object.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.num_v_heads, layer.tp_size = 1, 1
    layer.key_dim, layer.value_dim, layer.head_v_dim = 16, 16, 16
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_state_tokens=lambda: 7),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    layer._init_spec_state_replay(cfg)
    assert layer._replay_slots == 8
    assert layer._replay_qkv.shape == (16, 48)
    assert layer._replay_table.shape == (2, 9)
    actual_table = torch.empty(2, 9, dtype=torch.int32, device=DEVICE)
    actual_cu = torch.empty(3, dtype=torch.int32, device=DEVICE)

    # The callback isolates the graph manager/table kernels from model weights.
    def replay(table, cu, tokens):
        assert tokens == 8
        actual_table.copy_(table)
        actual_cu.copy_(cu)

    layer.replay_spec_states = replay
    base = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    cu = torch.tensor([0, 8], dtype=torch.int32, device=DEVICE)
    rows = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    blocks = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=DEVICE)
    idx = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    accepted = torch.ones(1, dtype=torch.int32, device=DEVICE)
    computed = torch.tensor([13], dtype=torch.int32, device=DEVICE)
    step = GDNSpecReplayStep(
        num_reqs=1,
        num_tokens=8,
        spec_rows=rows,
        layers=[GDNSpecReplayLayer(layer, 0, base, cu)],
        block_tables=(blocks,),
        full_graph=True,
    )
    manager = GDNSpecReplayGraphs(enabled=True)
    snapshots = []
    for run, count in enumerate((1, 4, 7, 8)):
        accepted.fill_(count)
        manager.run(
            step,
            idx_mapping=idx,
            num_sampled=accepted,
            num_computed_tokens=computed,
            block_size=8,
        )
        expected = build_replay_state_indices(
            base_state_ids=base,
            num_reqs=1,
            num_columns=9,
            spec_rows=rows,
            num_accepted=accepted,
            num_computed_tokens=computed,
            block_table=blocks,
            block_size=8,
        )
        table_ref, cu_ref = split_replay_launch(expected, cu, accepted)
        assert torch.equal(actual_table, table_ref)
        assert torch.equal(actual_cu, cu_ref)
        assert len(manager.graphs) == int(run > 0)
        assert (manager._pool is not None) == (run > 0)
        snapshots.append(
            {
                "run": run + 1,
                "accepted": count,
                "graphs": len(manager.graphs),
                "pool": str(manager._pool),
                "reserved_bytes": torch.accelerator.memory_reserved(),
            }
        )
    _write_resource(
        "gdn-delayed-pool",
        {
            "scope": (
                "Real layer buffer allocator and replay manager; component callback"
            ),
            "buffers": {
                name: list(value.shape)
                for name, value in vars(layer).items()
                if name.startswith("_replay_") and isinstance(value, torch.Tensor)
            },
            "snapshots": snapshots,
            "keys": [str(key) for key in manager.graphs],
        },
    )


@pytest.fixture
def native_allocation_config(tmp_path, monkeypatch):
    from transformers import Qwen3Config

    from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
    from vllm.model_executor.models import registry

    # The kit appends installed extension modules to the parent's package path.
    # A fresh registry subprocess loses that path. Register the actual classes
    # eagerly through the registry API so metadata inspection stays in-process.
    model_registry = registry.ModelRegistry
    monkeypatch.setattr(model_registry, "models", model_registry.models.copy())
    for arch in ("Qwen3ForCausalLM", "DFlash2DraftModel"):
        model_registry.register_model(
            arch, model_registry.models[arch].load_model_cls()
        )

    def unexpected_subprocess(fn):
        pytest.fail("Allocation metadata must be inspected in-process")

    monkeypatch.setattr(registry, "_run_in_subprocess", unexpected_subprocess)

    if DEVICE == "cpu":
        from vllm.platforms import current_platform

        # The CPU regression constructs metadata only. Actual allocation below
        # remains CUDA-only and uses the real platform without this override.
        monkeypatch.setattr(current_platform, "is_cuda", lambda: True)

    for name, arch in (("target", "Qwen3ForCausalLM"), ("draft", "DFlash2DraftModel")):
        Qwen3Config(
            architectures=[arch],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            max_position_embeddings=512,
            dflash_config={
                "selector_top_k": 16,
                "block_size": 8,
                "mask_token_id": 127,
            },
        ).save_pretrained(tmp_path / name)
    target = ModelConfig(
        model=str(tmp_path / "target"),
        dtype="float16",
        max_model_len=512,
        skip_tokenizer_init=True,
    )
    spec_config = SpeculativeConfig(
        method="dflash",
        model=str(tmp_path / "draft"),
        num_speculative_tokens=7,
        draft_sample_method="probabilistic",
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        dflash2_lookup={"enabled": True, "policy": "longest", "nmin": 2, "nmax": 4},
    )
    return SimpleNamespace(
        speculative_config=spec_config,
        model_config=target,
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=8),
        parallel_config=ParallelConfig(),
    )


def test_native_allocation_metadata_in_process(native_allocation_config):
    from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

    config = native_allocation_config
    assert config.model_config.architectures == ["Qwen3ForCausalLM"]
    assert config.speculative_config.draft_model_config.architectures == [
        "DFlash2DraftModel"
    ]
    assert config.speculative_config.num_speculative_tokens == 7
    assert config.speculative_config.resolved_dflash2_lookup.mode == "gpu_head"
    hf_config = config.speculative_config.draft_model_config.hf_config
    assert get_parallel_drafting_token_id(hf_config) == 127 < hf_config.vocab_size


@pytest.mark.skipif(DEVICE != "cuda", reason="Coordinator real constructor allocation")
def test_real_speculator_allocates_only_native_width(native_allocation_config):
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    config = native_allocation_config
    spec_config = config.speculative_config
    spec = DFlash2Speculator(config, torch.device(DEVICE))
    spec._allocate_draft_logits()
    assert spec.num_speculative_steps == spec.draft_block == 7
    assert spec.num_query_per_req == 8
    assert spec.draft_tokens.shape == (2, 7)
    assert spec.draft_logits.shape == (2, 7, 128)
    assert spec._cached_candidate_ids.shape == (2, 7, 16)
    assert spec.sample_idx_mapping.numel() == 14
    assert spec.input_buffers.input_ids.numel() == 16
    assert spec._native_lookup is not None
    assert not spec._lookup_enabled and spec._ngram_assist is None
    assert not hasattr(spec, "_lookup_copy_stream")
    _write_resource(
        "speculator-allocations",
        {
            "K": spec.num_speculative_steps,
            "D": spec.draft_block,
            "query": spec.num_query_per_req,
            "buffers": {
                name: list(value.shape)
                for name, value in vars(spec).items()
                if isinstance(value, torch.Tensor)
            },
            "context_graphs": {
                name: list(getattr(spec, name))
                for name in (
                    "_context_kv_graphs",
                    "_context_compute_graphs",
                    "_context_store_graphs",
                    "_draft_metadata_graphs",
                )
            },
            "state_tokens": spec_config.num_speculative_state_tokens(),
            "state_blocks": spec_config.num_speculative_state_blocks(),
            "scope": (
                "Real allocation entries with checkpoint metadata, no model weights"
            ),
        },
    )


@pytest.mark.skipif(DEVICE != "cuda", reason="Coordinator UVA long-history check")
@pytest.mark.parametrize("length", [32768, 131072, 262144, 524288])
@pytest.mark.parametrize("pattern", ["random", "periodic", "far_only", "miss"])
@pytest.mark.parametrize("window_kind", ["64k", "full"])
def test_longest_uva_history_boundaries(length, pattern, window_kind):
    from vllm.v1.worker.gpu.buffer_utils import UvaBuffer
    from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import suffix_lookup

    generator = np.random.default_rng(31)
    if pattern == "random":
        sequence = generator.integers(0, 32, length).tolist()
    elif pattern == "periodic":
        sequence = (np.arange(length) % 4).tolist()
    elif pattern == "far_only":
        sequence = (100 + np.arange(length) % 10000).tolist()
        sequence[:8] = sequence[-8:] = list(range(1, 9))
    else:
        sequence = list(range(length))
    window = 65536 if window_kind == "64k" else length
    history = UvaBuffer((1, length + 16), torch.int32)
    history.cpu.fill_(777)
    history.cpu[0, :length] = torch.tensor(sequence, dtype=torch.int32)
    lengths = torch.tensor([length], dtype=torch.int32, device=DEVICE)
    mapping = torch.zeros(7, dtype=torch.int32, device=DEVICE)
    eligible = torch.ones(1, dtype=torch.int32, device=DEVICE)
    guarded = torch.full((1, 9), 777, dtype=torch.int32, device=DEVICE)
    expected, match, valid = reference_lookup(sequence, 3, 8, window)

    def lookup():
        return suffix_lookup(
            history.uva,
            lengths,
            mapping,
            eligible,
            1,
            7,
            idx_mapping_stride=7,
            nmin=3,
            nmax=8,
            search_max=window,
            out_tokens=guarded[:, 1:8],
            native=True,
        )

    actual, found, available = lookup()
    assert actual[0].tolist() == expected
    assert found.tolist() == [match] and available.tolist() == [valid]
    assert guarded[0, [0, 8]].tolist() == [777, 777]
    torch.accelerator.synchronize()
    assert history.cpu[0, length:].tolist() == [777] * 16
    # Future token storage is outside the committed history and cannot affect it.
    history.cpu[0, length:] = 13
    repeated, _, _ = lookup()
    assert repeated[0].tolist() == expected
    _write_resource(
        f"uva-{length}-{pattern}-{window_kind}",
        {
            "length": length,
            "window": window,
            "pattern": pattern,
            "match": match,
            "valid": valid,
            "tokens": expected,
            "history_storage": "UvaBuffer",
            "red_zone": "passed",
        },
    )


@pytest.mark.parametrize("probabilistic", [False, True])
def test_native_entry_with_dense_or_greedy_cache(probabilistic):
    native, states, mapping, draft, logits, ids, _, batch, sampled = fixture_components(
        [4, 5, 9, 8, 4, 5]
    )
    before = logits.clone()
    native.prepare(batch, sampled, real_run=True)
    native.apply(
        states, mapping, draft, 2, logits if probabilistic else None, ids, None
    )
    assert draft[0, :4].tolist() == [9, 8, 4, 5]
    assert torch.equal(logits[2, 0], before[2, 0])
    if probabilistic:
        assert torch.where(torch.isfinite(logits[2, 1]))[0].tolist() == [8]
    else:
        assert torch.equal(logits, before)


def c64_components(scenario):
    length, capacity = 131072, 131104
    native = NativeDFlash2Lookup(
        native_config(nmin=5, nmax=16, window=65536),
        2,
        capacity,
        torch.device(DEVICE),
    )
    if DEVICE == "cuda":
        from vllm.v1.worker.gpu.buffer_utils import UvaBuffer

        owner = UvaBuffer((3, capacity), torch.int32)
        history_cpu, history = owner.cpu, owner.uva
    else:
        owner = None
        history_cpu = history = torch.empty((3, capacity), dtype=torch.int32)
    history_cpu.fill_(31)
    history_cpu[:, length:] = 99
    pattern = torch.arange(16, dtype=torch.int32)
    location = 4096 if scenario == "far_only" else length - 96
    for slot in (1, 2):
        history_cpu[slot, location : location + 16] = pattern
        history_cpu[slot, location + 16 : location + 23] = torch.arange(17, 24)
        history_cpu[slot, length - 16 : length] = pattern
    lengths = torch.tensor([0, length, length], dtype=torch.int32, device=DEVICE)
    states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=history),
        total_len=SimpleNamespace(gpu=lengths),
        uva_owner=owner,
    )
    mapping = torch.tensor([2] * 7 + [1] * 7, dtype=torch.int32, device=DEVICE)
    draft = torch.tensor(
        [[17, 0, 2, 3, 17, 0, 2]] * 2, dtype=torch.int64, device=DEVICE
    )
    if scenario == "mismatch":
        draft[0, 0] = 0
    ids = torch.tensor([0, 17, 2, 3], device=DEVICE).repeat(3, 7, 1)
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4], device=DEVICE).log().repeat(3, 7, 1)
    logits = torch.full((3, 7, 32), -float("inf"), device=DEVICE)
    logits.scatter_(-1, ids, scores)
    batch = SimpleNamespace(
        num_reqs=1,
        has_structured_output_reqs=False,
        is_prefilling_np=np.array([False, False]),
    )
    sampled = torch.ones(2, dtype=torch.int32, device=DEVICE)
    return native, states, mapping, draft, logits, ids, scores, batch, sampled


def assert_c64_outputs(parts, scenario, count, sparse, baseline):
    native, _, _, draft, logits, ids, scores, _, _ = parts
    active = count == 1
    hit = active and scenario != "far_only"
    fused = hit and scenario != "mismatch"
    assert native.eligible.tolist() == [int(active), 0]
    assert native.match.tolist() == [16 if hit else 0, 0]
    assert native.valid.tolist() == [7 if hit else 0, 0]
    assert native.use[0].tolist() == [0] + [int(fused)] * 6
    assert native.tokens[0].tolist() == (list(range(17, 24)) if hit else [0] * 7)
    assert native.tokens[1].tolist() == [0] * 7
    assert torch.equal(logits[2, 0], baseline[1][2, 0])
    assert torch.equal(ids[2, 0], baseline[2][2, 0])
    assert torch.equal(scores[2, 0], baseline[3][2, 0])
    if not fused:
        for actual, original in zip((draft, logits, ids, scores), baseline):
            assert torch.equal(actual, original)
        return
    assert draft[0].tolist() == list(range(17, 24))
    assert torch.equal(draft[1], baseline[0][1])
    for step in range(1, 7):
        token = 17 + step
        assert torch.where(torch.isfinite(logits[2, step]))[0].tolist() == [token]
        assert logits[2, step, token] == 0
        if sparse:
            assert ids[2, step, 0] == token
            assert scores[2, step, 0] == 0
            assert torch.isneginf(scores[2, step, 1:]).all()
    if not sparse:
        assert torch.equal(scores, baseline[3])
    assert torch.equal(logits[:2], baseline[1][:2])
    assert torch.equal(ids[:2], baseline[2][:2])
    assert torch.equal(scores[:2], baseline[3][:2])


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("scenario", ["near", "far_only", "mismatch", "multi_recover"])
def test_c64_native_chain_and_capture(sparse, scenario):
    parts = c64_components(scenario)
    native, states, mapping, draft, logits, ids, scores, batch, sampled = parts
    assert native.max_model_len - 131072 >= 8
    original = [value.clone() for value in (draft, logits, ids, scores)]

    def run():
        native.apply(states, mapping, draft, 2, logits, ids, scores if sparse else None)

    graph = None
    if DEVICE == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            native.prepare(batch, sampled, real_run=True)
            run()
        torch.cuda.current_stream().wait_stream(stream)
        native.reset_for_capture()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    phases = []
    for count in (1, 2, 1) if scenario == "multi_recover" else (1,):
        batch.num_reqs = count
        for actual, value in zip((draft, logits, ids, scores), original):
            actual.copy_(value)
        native.prepare(batch, sampled, real_run=True)
        run()
        assert_c64_outputs(parts, scenario, count, sparse, original)
        snapshot = [
            value.clone()
            for value in (
                draft,
                logits,
                ids,
                scores,
                native.tokens,
                native.match,
                native.valid,
                native.eligible,
                native.use,
            )
        ]
        if graph is not None:
            for actual, value in zip((draft, logits, ids, scores), original):
                actual.copy_(value)
            native.prepare(batch, sampled, real_run=True)
            graph.replay()
            for actual, eager in zip(
                (
                    draft,
                    logits,
                    ids,
                    scores,
                    native.tokens,
                    native.match,
                    native.valid,
                    native.eligible,
                    native.use,
                ),
                snapshot,
            ):
                assert torch.equal(actual, eager)
            assert_c64_outputs(parts, scenario, count, sparse, original)
        timing = {}
        if graph is not None:
            for mode, execute in (("eager", run), ("replay", graph.replay)):
                for _ in range(5):
                    native.prepare(batch, sampled, real_run=True)
                    execute()
                samples = []
                for _ in range(30):
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    native.prepare(batch, sampled, real_run=True)
                    execute()
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end))
                timing[mode] = dict(
                    warmup=5,
                    samples_ms=samples,
                    p50_ms=float(np.percentile(samples, 50)),
                    p95_ms=float(np.percentile(samples, 95)),
                    scope="prepare + eligibility + suffix + fuse + dense/sparse q",
                )
        phases.append(
            dict(
                requests=count,
                match=native.match.tolist(),
                valid=native.valid.tolist(),
                use=native.use.tolist(),
                tokens=draft.tolist(),
                timing=timing,
            )
        )
    if DEVICE == "cuda":
        torch.accelerator.synchronize()
        assert states.uva_owner.cpu[:, 131072:].eq(99).all()
    _write_resource(
        f"c64-chain-{scenario}-sparse{int(sparse)}",
        dict(
            history_length=131072,
            capacity=131104,
            nmin=5,
            nmax=16,
            search_window=65536,
            storage="UVA" if DEVICE == "cuda" else "CPU interpreter",
            capture_replay_checked=graph is not None,
            sparse=sparse,
            phases=phases,
        ),
    )
