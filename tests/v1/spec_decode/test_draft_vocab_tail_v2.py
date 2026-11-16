# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/interpreter coverage of the V2 global MTP vocabulary tail."""

import ast
import inspect
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers import logits_processor as lp
from vllm.v1.spec_decode.static_draft_vocab import (
    StaticDraftVocabRuntime,
    update_dynamic_draft_vocab_lru_state,
)
from vllm.v1.worker.gpu.spec_decode.eagle import draft_vocab_tail as tail_mod
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle


def _tail(tmp_path, *, T=4, tp=1, rank=0, probabilistic=False, rows=8, steps=4):
    V, B, H = max(32, 2 * T + 16), 8, 4
    V = (V + tp - 1) // tp * tp
    gen = torch.Generator().manual_seed(42)
    weight = torch.randn(V, H, generator=gen, dtype=torch.float16)
    # Token order differs from token IDs, including among the tail seeds.
    ranking = torch.randperm(V - 1, generator=gen)
    ranking = torch.cat((ranking, torch.tensor([V - 1])))
    path = tmp_path / "ranking.pt"
    torch.save(dict(global_ranking=ranking, tokenizer_vocab_size=V - 1), path)
    base = ranking[:B]
    local_base = base[rank * (B // tp) : (rank + 1) * (B // tp)]
    head = SimpleNamespace(weight=weight[local_base])
    processor = Mock(side_effect=lambda head, h: h @ weight[base].T)
    processor.soft_cap = None
    processor.scale = 1.0

    def local_top1(head, h):
        vals, ids = (h @ head.weight.T).max(-1)
        return vals, ids + rank * (B // tp)

    processor.get_local_top1.side_effect = local_top1
    runtime = StaticDraftVocabRuntime(head, processor, base, V, B, None)
    width = V // tp
    source = SimpleNamespace(
        weight=weight[rank * width : (rank + 1) * width],
        num_embeddings_per_partition=width,
    )
    draft_logits = torch.full((rows, steps, V), -torch.inf) if probabilistic else None
    tail = tail_mod.GlobalDraftVocabTail(
        runtime, source, str(path), T, rows, steps, draft_logits, tp, rank
    )
    return tail, weight, ranking


@pytest.mark.parametrize("T", [4, 512, 1024])
@pytest.mark.parametrize("W", [1, 5])
def test_lru_stepwise_1cat_and_torch(tmp_path, T, W):
    tail, _, ranking = _tail(tmp_path, T=T)
    tokens, stamps, clock = (
        tail.slot_token.clone(),
        tail.slot_stamp.clone(),
        tail.clock.clone(),
    )
    lru = OrderedDict.fromkeys(tokens.tolist())
    base_ids = frozenset(tail.runtime.token_id_map.tolist())
    gen = torch.Generator().manual_seed(153)
    for step in range(8):
        sampled = torch.randint(-2, tail.full_vocab_size + 2, (8, W), generator=gen)
        counts = torch.randint(0, W + 1, (8,), generator=gen, dtype=torch.int32)
        counts[0] = 0
        counts[-1] = W
        sampled[-1, 0] = ranking[0] if step % 2 else tokens[0]
        # All ignored columns can change without affecting the state.
        alternate = sampled.clone()
        for row, count in enumerate(counts):
            alternate[row, int(count) :] = 999999
        rt, rs, rc = tokens.clone(), stamps.clone(), clock.clone()
        tail_mod.update_reference(
            tokens,
            stamps,
            clock,
            sampled,
            counts,
            tail.base_mask,
            tail.tokenizer_vocab_size,
        )
        tail_mod.update_reference(
            rt, rs, rc, alternate, counts, tail.base_mask, tail.tokenizer_vocab_size
        )
        torch.testing.assert_close(rt, tokens, rtol=0, atol=0)
        torch.testing.assert_close(rs, stamps, rtol=0, atol=0)
        observed = torch.cat([sampled[r, : int(counts[r])] for r in range(8)])
        observed = observed[observed < tail.tokenizer_vocab_size]
        update_dynamic_draft_vocab_lru_state(
            lru,
            observed,
            torch.empty(0, dtype=torch.int64),
            full_vocab_size=tail.full_vocab_size,
            base_token_ids=base_ids,
            tail_size=T,
        )
        tail.update(sampled, counts, 8, torch.arange(8))
        torch.testing.assert_close(tail.slot_token, tokens, rtol=0, atol=0)
        torch.testing.assert_close(tail.slot_stamp, stamps, rtol=0, atol=0)
        torch.testing.assert_close(tail.clock, clock, rtol=0, atol=0)
        assert tuple(tokens[stamps.argsort()].tolist()) == tuple(lru)
        assert not tail.changed.any()
        assert tail.tail_valid.all()
        torch.testing.assert_close(
            tail.tail_weight, tail.source_weight[tokens], rtol=0, atol=0
        )


@pytest.mark.parametrize("tokenizer_vocab_size", [28, 26])
def test_large_base_seed_skips_padding_and_requires_full_capacity(
    tmp_path, tokenizer_vocab_size
):
    tail, weight, _ = _tail(tmp_path, probabilistic=True)
    base = torch.tensor(list(range(23)) + [28])
    ranking = torch.tensor(list(range(23)) + [28, 29, 23, 30, 24, 31, 25, 26, 27])
    tail.runtime.shortlist_size = 24
    tail.runtime.token_id_map = base
    tail.runtime.lm_head.weight = weight[base]
    tail.runtime.logits_processor.side_effect = lambda head, h: h @ head.weight.T
    path = tmp_path / "ranking.pt"
    torch.save(
        dict(global_ranking=ranking, tokenizer_vocab_size=tokenizer_vocab_size), path
    )
    source = SimpleNamespace(weight=weight, num_embeddings_per_partition=32)
    args = (tail.runtime, source, str(path), 4, 8, 4, tail.draft_logits, 1, 0)
    if tokenizer_vocab_size == 26:
        with pytest.raises(ValueError, match="non-padding seeds"):
            tail_mod.GlobalDraftVocabTail(*args)
        return
    active = tail_mod.GlobalDraftVocabTail(*args)
    # [B:B+T] includes padding; scan farther to preserve the requested capacity.
    assert ranking[24:28].tolist() == [29, 23, 30, 24]
    assert active.slot_token.tolist() == [23, 24, 25, 26]
    assert active.sink == 29  # Padding ID 28 is already in the base.
    assert active.sink not in active.runtime.token_id_map
    assert active.sink not in active.slot_token
    assert (active.slot_token < tokenizer_vocab_size).all()
    torch.testing.assert_close(active.tail_weight, weight[active.slot_token])
    active.sample_probabilistic(
        torch.eye(4, dtype=torch.float16),
        torch.arange(4, dtype=torch.int32),
        torch.ones(8),
        torch.arange(8, dtype=torch.int64),
        torch.arange(4),
        torch.tensor(0),
        active.draft_logits,
        True,
    )
    assert torch.isneginf(active.draft_logits[..., active.sink]).all()


def test_update_ignores_all_padding_ids_in_kernel_and_reference(tmp_path):
    tail, weight, _ = _tail(tmp_path)
    path = tmp_path / "ranking.pt"
    payload = torch.load(path, weights_only=True)
    payload["tokenizer_vocab_size"] = 28
    torch.save(payload, path)
    tail = tail_mod.GlobalDraftVocabTail(
        tail.runtime,
        SimpleNamespace(weight=weight, num_embeddings_per_partition=32),
        str(path),
        4,
        8,
        4,
        None,
        1,
        0,
    )
    tokens, stamps, clock = (
        tail.slot_token.clone(),
        tail.slot_stamp.clone(),
        tail.clock.clone(),
    )
    initial = tokens.clone(), stamps.clone(), clock.clone()
    new = next(i for i in range(28) if not tail.base_mask[i] and i not in tokens)
    for sampled in (
        torch.tensor([[28, 29, 30, 31], [31, 30, 29, 28]]),
        torch.tensor([[28, new, 29, new], [31, 30, 29, 28]]),
    ):
        counts = torch.full((2,), 4, dtype=torch.int32)
        tail_mod.update_reference(
            tokens,
            stamps,
            clock,
            sampled,
            counts,
            tail.base_mask,
            tail.tokenizer_vocab_size,
        )
        tail.update(sampled, counts, 2, torch.arange(2))
        torch.testing.assert_close(tail.slot_token, tokens, rtol=0, atol=0)
        torch.testing.assert_close(tail.slot_stamp, stamps, rtol=0, atol=0)
        torch.testing.assert_close(tail.clock, clock, rtol=0, atol=0)
        assert (tokens < 28).all()
        assert tail.sink not in tokens
        if sampled[0, 1] == 29:
            for actual, expected in zip((tokens, stamps, clock), initial):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert int(clock[0]) == int(initial[2][0]) + 2


def test_lru_repeated_input_and_real_request_count(tmp_path):
    a, _, _ = _tail(tmp_path)
    b, _, _ = _tail(tmp_path)
    sampled = torch.tensor([[30, 30, 29, 28, 27]] * 8)
    counts = torch.tensor([0, 5, 1, 2, 3, 4, 5, 5], dtype=torch.int32)
    a.update(sampled, counts, 6, torch.arange(6))
    b.update(sampled[:6].contiguous(), counts[:6], 6, torch.arange(6))
    torch.testing.assert_close(a.slot_token, b.slot_token, rtol=0, atol=0)
    torch.testing.assert_close(a.slot_stamp, b.slot_stamp, rtol=0, atol=0)
    torch.testing.assert_close(a.clock, b.clock, rtol=0, atol=0)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_greedy_k1_k2_total_order(tmp_path, tp):
    tails = [_tail(tmp_path, tp=tp, rank=r)[0] for r in range(tp)]
    T, B, M = 4, 8, 8
    gen = torch.Generator().manual_seed(72)
    all_vals, all_keys, pairs = [], [], []
    for r, tail in enumerate(tails):
        base_vals = torch.randn(M, generator=gen)
        base_keys = torch.full((M,), r * (B // tp), dtype=torch.int64)
        vals = torch.randn(M, T, generator=gen)
        # Equal base/tail, tail/tail and cross-rank values; infinities compare directly.
        base_vals[:3] = 1
        vals[:3] = 1
        vals[3, :2] = torch.tensor([torch.inf, -torch.inf])
        base_vals[4] = torch.inf
        if r == 0:
            tail.tail_valid.zero_()
        tail_mod._merge_local[(M,)](
            base_vals,
            base_keys,
            vals,
            tail.slot_token,
            tail.tail_valid,
            tail.local_pair,
            B,
            T,
            4,
        )
        pairs.append(tail.local_pair.clone())
        keys = (B + tail.slot_token).expand(M, -1).clone()
        keys[:, ~tail.tail_valid] = torch.iinfo(torch.int32).max
        vals[:, ~tail.tail_valid] = -torch.inf
        all_vals.extend([base_vals[:, None], vals])
        all_keys.extend([base_keys[:, None], keys])
    expected_key = tail_mod.merge_top1_reference(
        torch.cat(all_vals, -1), torch.cat(all_keys, -1)
    )
    tail_mod._merge_global[(M,)](
        torch.cat(pairs, -1),
        tails[0].runtime.token_id_map,
        tails[0].out_tokens,
        B,
        tp,
        tp,
    )
    expected = torch.where(
        expected_key < B,
        tails[0].runtime.token_id_map[expected_key.clamp(max=B - 1)],
        expected_key - B,
    )
    torch.testing.assert_close(tails[0].out_tokens, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_probabilistic_mapping_eviction_and_greedy_fallback(tmp_path, monkeypatch, tp):
    tails = [_tail(tmp_path, tp=tp, rank=r, probabilistic=True) for r in range(tp)]
    gen = torch.Generator().manual_seed(15)
    hidden = torch.randn(8, 4, generator=gen, dtype=torch.float16)
    mapping = torch.tensor([3, 1, 0, 2, 4, 5, 6, -1], dtype=torch.int32)
    temperature = torch.tensor([0.7, 1.2, 0.0, 0.9, 1.0, 1.0, 1.0, 1.0])
    seeds = torch.arange(8, dtype=torch.int64) * 37
    pos = torch.arange(8, dtype=torch.int64) * 19

    def gather(_local, dim):
        return torch.cat(
            [
                t._tail_logits(hidden).masked_fill(~t.tail_valid, -torch.inf)
                for t, _, _ in tails
            ],
            dim,
        )

    monkeypatch.setattr(tail_mod, "tensor_model_parallel_all_gather", gather)
    tail, weight, _ = tails[0]
    for iteration in range(2):
        for step in range(4):
            actual = tail.sample_probabilistic(
                hidden,
                mapping,
                temperature,
                seeds,
                pos,
                torch.tensor(step),
                tail.draft_logits,
                True,
            )
            union = torch.cat((tail.runtime.token_id_map, tail.slot_token))
            direct = hidden @ weight[union].T
            logits = tail.draft_logits[mapping[:7], step]
            temp = temperature[mapping[:7]].clone()
            temp[temp == 0] = 1
            expected = torch.zeros_like(logits)
            expected[:, union] = (direct[:7].float() / temp[:, None]).softmax(-1)
            torch.testing.assert_close(
                logits.softmax(-1), expected, atol=2e-7, rtol=2e-6
            )
            assert all(int(i) in union for i in actual[:7])
            # Greedy without local reduction shares concatenation and mapping.
            combined = tail._combined_logits(hidden)
            fallback = tail.sample_probabilistic(
                hidden, mapping, temperature, seeds, pos, torch.tensor(step), None, True
            )
            torch.testing.assert_close(
                fallback, tail.ext_token_map[combined.argmax(-1)], rtol=0, atol=0
            )
        assert torch.isneginf(tail.draft_logits[..., tail.sink]).all()
        finite = torch.isfinite(tail.draft_logits[:7])
        allowed = torch.zeros(tail.full_vocab_size, dtype=torch.bool)
        allowed[union] = True
        assert not finite[..., ~allowed].any()
        if iteration == 0:
            old = int(tail.slot_token[0])
            new = next(i for i in range(tail.full_vocab_size - 1) if i not in union)
            for t, _, _ in tails:
                t.update(
                    torch.tensor([[new]] + [[int(t.runtime.token_id_map[0])]] * 6),
                    torch.ones(7, dtype=torch.int32),
                    7,
                    mapping[:7],
                )
            assert torch.isneginf(tail.draft_logits[..., old]).all()
            assert new in tail.slot_token
        for t, _, _ in tails:
            local = (t.slot_token >= t.tp_rank * t.shard_width) & (
                t.slot_token < (t.tp_rank + 1) * t.shard_width
            )
            torch.testing.assert_close(t.tail_valid, local)
            torch.testing.assert_close(
                t.tail_weight[local], weight[t.slot_token[local]], rtol=0, atol=0
            )


def test_pending_proposal_q_survives_other_requests_tail_changes(tmp_path):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    tail, weight, _ = _tail(tmp_path, probabilistic=True)
    spec = object.__new__(eagle.EagleSpeculator)
    spec._static_draft_vocab = tail.runtime
    spec._draft_vocab_tail = tail
    spec.draft_logits = tail.draft_logits
    spec.use_fp64_gumbel = True
    spec.use_local_argmax_reduction = False
    spec.temperature = torch.ones(8)
    spec.seeds = torch.arange(8, dtype=torch.int64) + 93
    generator = torch.Generator().manual_seed(121)
    hidden = torch.randn(8, 4, generator=generator, dtype=torch.float16)
    positions = torch.arange(8, dtype=torch.int64) + 15
    mapping = torch.arange(8, dtype=torch.int32)
    for step in range(4):
        spec._sample_draft(
            hidden, mapping, positions, torch.tensor(step), spec.draft_logits
        )
    before = tail.draft_logits.clone()
    pending_q = before[7].numpy().tobytes()
    old = int(tail.slot_token[0])
    assert torch.isfinite(before[7, :, old]).all()
    union = torch.cat((tail.runtime.token_id_map, tail.slot_token))
    new = next(i for i in range(tail.tokenizer_vocab_size) if i not in union)

    # Only requests 6 and 2 are verified and drafted in the intervening batch.
    active = torch.tensor([6, 2], dtype=torch.int32)
    spec.update_draft_vocab_tail(
        torch.tensor([[new], [int(tail.runtime.token_id_map[0])]]),
        torch.ones(2, dtype=torch.int32),
        2,
        active,
    )
    assert old not in tail.slot_token and new in tail.slot_token
    assert torch.isneginf(tail.draft_logits[active.long()]).all()
    inactive = torch.tensor([0, 1, 3, 4, 5, 7])
    assert (
        tail.draft_logits[inactive].numpy().tobytes()
        == before[inactive].numpy().tobytes()
    )
    # Graph padding must not write request 0 or any other real request.
    padded_mapping = torch.tensor([6, 2, -1], dtype=torch.int32)
    for step in range(4):
        spec._sample_draft(
            hidden[[6, 2, 0]],
            padded_mapping,
            positions[[6, 2, 0]],
            torch.tensor(step),
            spec.draft_logits,
        )
        assert (
            tail.draft_logits[inactive].numpy().tobytes()
            == before[inactive].numpy().tobytes()
        )
    assert torch.isneginf(tail.draft_logits[active.long(), :, old]).all()
    assert torch.isfinite(tail.draft_logits[active.long(), :, new]).all()

    # Resume request 7 through the real runner verification call site.
    consumed = []

    def verify(logits, batch, q):
        consumed.append(q[batch.idx_mapping.long()].numpy().tobytes())
        return SimpleNamespace(num_sampled=torch.ones(1, dtype=torch.int32))

    runner = object.__new__(GPUModelRunner)
    runner.speculator = spec
    runner.rejection_sampler = Mock(side_effect=verify)
    runner.device = torch.device("cpu")
    runner.model = SimpleNamespace(compute_logits=lambda h: h @ weight.T)
    runner.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32))
    )
    batch = SimpleNamespace(
        logits_indices=torch.arange(5),
        num_draft_tokens=4,
        num_reqs=1,
        idx_mapping=torch.tensor([7], dtype=torch.int32),
        seq_lens=torch.tensor([25], dtype=torch.int32),
        cu_num_logits=torch.tensor([0, 5], dtype=torch.int32),
    )
    runner.sample(hidden[:5], batch, None)
    assert consumed == [pending_q]


def test_sampling_distribution_fixed_seed(tmp_path):
    N = 4096
    tail, weight, _ = _tail(tmp_path, probabilistic=True, rows=N, steps=1)
    hidden = (
        torch.tensor([[0.3, -0.2, 0.1, 0.4]], dtype=torch.float16)
        .expand(N, -1)
        .contiguous()
    )
    samples = tail.sample_probabilistic(
        hidden,
        torch.arange(N, dtype=torch.int32),
        torch.full((N,), 0.8),
        torch.arange(N, dtype=torch.int64) + 400,
        torch.full((N,), 23),
        torch.tensor(0),
        tail.draft_logits,
        True,
    )
    union = torch.cat((tail.runtime.token_id_map, tail.slot_token))
    expected = ((hidden[:1] @ weight[union].T).float() / 0.8).softmax(-1)[0]
    counts = torch.bincount(samples, minlength=tail.full_vocab_size)[union]
    # Fixed seed; broad enough for sampling noise, sensitive to mapping/weights.
    chi2 = ((counts - N * expected).square() / (N * expected)).sum()
    assert chi2 < 40


def test_initialization_preconditions(tmp_path):
    tail, _, _ = _tail(tmp_path)
    args = [
        tail.runtime,
        SimpleNamespace(weight=tail.source_weight, num_embeddings_per_partition=32),
        str(tmp_path / "ranking.pt"),
        4,
        8,
        4,
        None,
        1,
        0,
    ]
    for attr, value in [("soft_cap", 1.0), ("scale", 2.0)]:
        old = getattr(tail.runtime.logits_processor, attr)
        setattr(tail.runtime.logits_processor, attr, value)
        with pytest.raises(ValueError):
            tail_mod.GlobalDraftVocabTail(*args)
        setattr(tail.runtime.logits_processor, attr, old)
    args[1] = SimpleNamespace(
        weight=tail.source_weight.float(), num_embeddings_per_partition=32
    )
    with pytest.raises(ValueError, match="fp16"):
        tail_mod.GlobalDraftVocabTail(*args)
    args[1] = SimpleNamespace(
        weight=tail.source_weight, num_embeddings_per_partition=32
    )
    payload = torch.load(args[2], weights_only=True)
    payload["global_ranking"] = payload["global_ranking"][:8]
    torch.save(payload, args[2])
    with pytest.raises(ValueError, match="ranking length"):
        tail_mod.GlobalDraftVocabTail(*args)
    payload["global_ranking"] = torch.arange(32)
    payload["tokenizer_vocab_size"] = 32
    torch.save(payload, args[2])
    with pytest.raises(ValueError, match="SINK"):
        tail_mod.GlobalDraftVocabTail(*args)


@pytest.mark.parametrize("T", [-1, 2049, 1])
@pytest.mark.parametrize(
    "method,ranking,size",
    [
        ("mtp", None, 0),
        ("mtp", "ranking.pt", 0),
        ("mtp", None, 8),
        ("eagle", "ranking.pt", 8),
    ],
)
def test_configuration_errors(monkeypatch, T, method, ranking, size):
    spec = object.__new__(eagle.EagleSpeculator)
    spec.method = method
    monkeypatch.setenv("VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE", str(T))
    monkeypatch.setenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE", str(size))
    if ranking is None:
        monkeypatch.delenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING", ranking)
    with pytest.raises(ValueError):
        spec._maybe_initialize_static_draft_vocab()


@pytest.mark.parametrize(
    "suffix", ["GPU_LRU", "FUSED_PROPOSAL", "FULL_REFRESH_INTERVAL", "PREFILL_TOPK"]
)
def test_v1_controls_rejected_even_when_set_to_zero(monkeypatch, suffix):
    spec = object.__new__(eagle.EagleSpeculator)
    monkeypatch.setenv("VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_" + suffix, "0")
    with pytest.raises(ValueError, match="only by V1"):
        spec._maybe_initialize_static_draft_vocab()


def test_disabled_matches_baseline_and_does_not_import_tail(monkeypatch):
    # Execute the baseline sampling method from the fixed starting revision.
    import subprocess

    source = subprocess.check_output(
        [
            "git",
            "show",
            "9b882072f3:vllm/v1/worker/gpu/spec_decode/eagle/speculator.py",
        ],
        text=True,
    )
    baseline = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "_sample_draft"
    )
    namespace = dict(torch=torch, gumbel_sample=eagle.gumbel_sample)
    exec(
        compile(ast.Module(body=[baseline], type_ignores=[]), "<baseline>", "exec"),
        namespace,
    )
    spec = object.__new__(eagle.EagleSpeculator)
    spec.method = "mtp"
    monkeypatch.setenv("VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE", "0")
    monkeypatch.setenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE", "0")
    monkeypatch.delenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING", raising=False)
    spec._maybe_initialize_static_draft_vocab()
    assert spec._draft_vocab_tail is None
    spec._static_draft_vocab = None
    spec.model = Mock()
    hidden = torch.eye(2)
    spec.model.compute_logits.return_value = torch.tensor([[1.0, 2.0], [5.0, 3.0]])
    spec.model.get_top_tokens.return_value = torch.tensor([1, 0])
    for static in [False, True]:
        if static:
            processor = Mock(return_value=torch.tensor([[1.0, 3.0], [2.0, 1.0]]))
            processor.get_top_tokens.return_value = torch.tensor([1, 0])
            spec._static_draft_vocab = StaticDraftVocabRuntime(
                None, processor, torch.tensor([4, 2]), 8, 2, None
            )
        for local in [False, True]:
            spec.use_local_argmax_reduction = local
            args = (hidden, torch.arange(2), torch.arange(2), torch.tensor(0), None)
            expected = namespace["_sample_draft"](spec, *args)
            actual = spec._sample_draft(*args)
            assert expected.numpy().tobytes() == actual.numpy().tobytes()


@pytest.mark.parametrize("static", [False, True])
def test_disabled_probabilistic_matches_main_bytes(tmp_path, static):
    import subprocess

    source = subprocess.check_output(
        [
            "git",
            "show",
            "9b882072f3:vllm/v1/worker/gpu/spec_decode/eagle/speculator.py",
        ],
        text=True,
    )
    baseline = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "_sample_draft"
    )
    namespace = dict(torch=torch, gumbel_sample=eagle.gumbel_sample)
    exec(
        compile(ast.Module(body=[baseline], type_ignores=[]), "<baseline>", "exec"),
        namespace,
    )
    tail, weight, _ = _tail(tmp_path, probabilistic=True)
    spec = object.__new__(eagle.EagleSpeculator)
    spec._static_draft_vocab = tail.runtime if static else None
    spec._draft_vocab_tail = None
    spec.use_local_argmax_reduction = False
    spec.use_fp64_gumbel = True
    spec.temperature = torch.linspace(0, 1.4, 8)
    spec.seeds = torch.arange(8, dtype=torch.int64) + 47
    spec.model = SimpleNamespace(compute_logits=lambda h: h @ weight.T)
    actual_full = (
        torch.full((8, 4, 32), -torch.inf) if static else torch.zeros(8, 4, 32)
    )
    expected_full = actual_full.clone()
    actual_reduced = torch.zeros(8, 4, 8) if static else None
    expected_reduced = torch.zeros(8, 4, 8) if static else None
    generator = torch.Generator().manual_seed(81)
    mapping = torch.tensor([7, 0, 2, -1], dtype=torch.int32)
    positions = torch.tensor([9, 17, 18, 0])
    for step in range(4):
        hidden = torch.randn(4, 4, generator=generator, dtype=torch.float16)
        spec._reduced_draft_logits = expected_reduced
        expected = namespace["_sample_draft"](
            spec, hidden, mapping, positions, torch.tensor(step), expected_full
        )
        spec._reduced_draft_logits = actual_reduced
        actual = spec._sample_draft(
            hidden, mapping, positions, torch.tensor(step), actual_full
        )
        assert actual.numpy().tobytes() == expected.numpy().tobytes()
        assert actual_full.numpy().tobytes() == expected_full.numpy().tobytes()


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("fast", [False, True])
def test_get_top_tokens_refactor_matches_baseline(monkeypatch, tp, fast):
    import subprocess

    source = subprocess.check_output(
        ["git", "show", "9b882072f3:vllm/model_executor/layers/logits_processor.py"],
        text=True,
    )
    method = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "get_top_tokens"
    )
    namespace = dict(vars(lp))
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<baseline>", "exec"),
        namespace,
    )
    monkeypatch.setattr(lp, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(
        lp, "tensor_model_parallel_all_gather", lambda pair, dim: pair.repeat(1, tp)
    )
    namespace.update(
        get_tensor_model_parallel_world_size=lambda: tp,
        tensor_model_parallel_all_gather=lp.tensor_model_parallel_all_gather,
    )
    processor = lp.LogitsProcessor(4)
    processor._maybe_dump_top_token_margin = Mock()
    processor._maybe_custom_top1_argmax = Mock(return_value=None)
    hidden = torch.eye(2)
    logits = torch.tensor([[1.0, 1.0, 0.0, 9.0], [2.0, 3.0, 3.0, 9.0]])
    head = SimpleNamespace(
        maybe_get_sm70_lm_head_top1=Mock(
            return_value=(torch.tensor([1.0, 3.0]), torch.tensor([0, 1]))
            if fast
            else None
        ),
        quant_method=SimpleNamespace(apply=lambda *args, **kw: logits.clone()),
        shard_indices=SimpleNamespace(num_org_vocab_padding=1, org_vocab_start_index=0),
    )
    expected = namespace["get_top_tokens"](processor, head, hidden)
    actual = processor.get_top_tokens(head, hidden)
    assert actual.numpy().tobytes() == expected.numpy().tobytes()


def test_wiring_and_no_host_sync():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    source = inspect.getsource(GPUModelRunner.sample_tokens)
    assert (
        source.index("self.sample(")
        < source.index("self.postprocess_sampled(")
        < source.index("update_tail(")
        < source.index("self.speculator.propose(")
    )
    assert 'getattr(self.speculator, "update_draft_vocab_tail", None)' in source
    module = ast.parse(Path(inspect.getfile(GPUModelRunner)).read_text())
    callers = [
        n.name
        for n in ast.walk(module)
        if isinstance(n, ast.FunctionDef)
        and any(
            isinstance(c, ast.Constant) and c.value == "update_draft_vocab_tail"
            for c in ast.walk(n)
        )
    ]
    assert callers == ["sample_tokens"]
    tree = ast.parse(Path(tail_mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            assert not any(
                isinstance(c, ast.Attribute)
                and c.attr in {"item", "cpu", "tolist", "synchronize"}
                for c in ast.walk(node)
            )


@pytest.mark.parametrize(
    "probabilistic,local", [(False, True), (False, False), (True, False)]
)
def test_speculator_enabled_initialization_and_sampling(
    tmp_path, monkeypatch, probabilistic, local
):
    tail, _, _ = _tail(tmp_path, probabilistic=probabilistic)
    spec = object.__new__(eagle.EagleSpeculator)
    spec.method = "mtp"
    spec.model = Mock()
    spec.model.lm_head = SimpleNamespace(
        weight=tail.source_weight, num_embeddings_per_partition=tail.full_vocab_size
    )
    spec.device = torch.device("cpu")
    spec.max_num_reqs = 8
    spec.num_speculative_steps = 4
    spec.draft_logits = tail.draft_logits
    spec.use_local_argmax_reduction = local
    spec.use_fp64_gumbel = True
    spec.temperature = torch.ones(8)
    spec.seeds = torch.arange(8, dtype=torch.int64)
    spec.speculative_config = SimpleNamespace(
        draft_sample_method="probabilistic" if probabilistic else "greedy"
    )
    monkeypatch.setenv("VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE", "4")
    monkeypatch.setenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE", "8")
    monkeypatch.setenv(
        "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING", str(tmp_path / "ranking.pt")
    )
    monkeypatch.setattr(
        eagle, "initialize_static_draft_vocab", lambda *args: tail.runtime
    )
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    info = Mock()
    monkeypatch.setattr(eagle.logger, "info", info)
    spec._maybe_initialize_static_draft_vocab()
    active = spec._draft_vocab_tail
    assert active is not None
    if probabilistic:
        assert spec._reduced_draft_logits.shape == (8, 4, 12)
    messages = [call.args for call in info.call_args_list]
    assert messages[-1] == (
        "Enabled global MTP draft vocabulary tail: "
        "base=%d tail=%d tp=%d sink=%d draft_sample=%s",
        8,
        4,
        1,
        active.sink,
        spec.speculative_config.draft_sample_method,
    )
    hidden = torch.eye(4, dtype=torch.float16)
    mapping = torch.arange(4, dtype=torch.int32)
    pos = torch.arange(4, dtype=torch.int64)
    actual = spec._sample_draft(
        hidden, mapping, pos, torch.tensor(0), spec.draft_logits
    )
    if local:
        expected = active.sample_greedy(hidden).clone()
    else:
        expected = active.sample_probabilistic(
            hidden,
            mapping,
            spec.temperature,
            spec.seeds,
            pos,
            torch.tensor(0),
            spec.draft_logits,
            True,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    update = Mock()
    monkeypatch.setattr(active, "update", update)
    sampled, counts = (
        torch.ones((4, 1), dtype=torch.int64),
        torch.ones(4, dtype=torch.int32),
    )
    spec.update_draft_vocab_tail(sampled, counts, 4, mapping)
    update.assert_called_once_with(sampled, counts, 4, mapping)


@pytest.mark.parametrize("with_tail", [False, True])
def test_runner_actual_call_order(monkeypatch, with_tail):
    from vllm.v1.worker.gpu import model_runner as runner_mod

    calls = []
    runner = object.__new__(runner_mod.GPUModelRunner)
    ids, counts, rejected = (
        torch.tensor([[13]]),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    batch = SimpleNamespace(
        req_ids=["request"],
        idx_mapping=torch.tensor([0]),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1]),
    )
    runner.execute_model_state = SimpleNamespace(
        input_batch=batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=torch.zeros(1, 4),
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner._sm70_v2_mtp_profile_pending = None
    runner._sm70_v2_mtp_profile_start = Mock(return_value=None)
    runner._sm70_v2_mtp_profile_finish = Mock()
    runner._sm70_v2_mtp_profile_report = Mock()
    runner.is_last_pp_rank = True

    def sample(*args):
        calls.append("sample")
        return SimpleNamespace(sampled_token_ids=ids), counts, rejected

    runner.sample = sample
    runner.model = SimpleNamespace(compute_logits=Mock())
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=Mock(return_value={})
    )
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None, gpu=None),
        prefill_len=SimpleNamespace(np=None),
        num_computed_prefill_tokens=None,
        last_sampled_tokens=None,
        next_prefill_tokens=None,
        draft_tokens=torch.zeros(1, 4, dtype=torch.int64),
    )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=None), seeds=SimpleNamespace(gpu=None)
        )
    )
    runner.output_copy_stream = runner.main_stream = None
    runner.rejection_sampler = None
    runner.postprocess_sampled = lambda *args: calls.append("postprocess")

    def propose(*args, **kwargs):
        calls.append("propose")
        return torch.zeros(1, 4, dtype=torch.int64)

    runner.speculator = SimpleNamespace(supports_mm_inputs=False, propose=propose)
    if with_tail:

        def update(sampled, num_sampled, num_reqs, idx_mapping):
            assert sampled is ids and num_sampled is counts and num_reqs == 1
            assert idx_mapping is batch.idx_mapping
            calls.append("update")

        runner.speculator.update_draft_vocab_tail = update
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=Mock())
    runner.pp_handler = None
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.mixed_prefill_timer = SimpleNamespace(finish=Mock(return_value=None))
    runner.use_async_scheduling = True
    runner.eplb = SimpleNamespace(step=Mock())
    monkeypatch.setattr(runner_mod, "AsyncOutput", lambda **kwargs: SimpleNamespace())
    runner.sample_tokens(None)
    assert calls == (
        ["sample", "postprocess", "update", "propose"]
        if with_tail
        else ["sample", "postprocess", "propose"]
    )
