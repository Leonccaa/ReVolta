# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config import KernelConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.spec_decode import mtp_compact_target as compact
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


@pytest.fixture
def device():
    if not compact._is_sm70(torch.device("cuda")):
        pytest.skip("requires SM70")
    return "cuda"


def make_batch(c, device, vocab, use_fp64=False, *, k=5, max_k=5):
    slots = 12
    rows = (k + 1) * c
    idx = np.array([7, 2, 11, 0, 6, 3, 9, 5][:c], dtype=np.int32)
    cu = np.arange(c + 1, dtype=np.int32) * (k + 1)

    def tensor(data, **kwargs):
        return torch.tensor(data, device=device, **kwargs)

    def state(data, dtype):
        array = np.full(slots, data, dtype=dtype)
        return NS(np=array, gpu=tensor(array))

    states = object.__new__(SamplingStates)
    states.vocab_size = vocab
    states.temperature = state(0.0, np.float32)
    states.top_p = state(1.0, np.float32)
    states.top_k = state(vocab, np.int32)
    states.min_p = state(0.0, np.float32)
    states.seeds = state(0, np.int64)
    states.num_logprobs = np.full(slots, NO_LOGPROBS)
    for slot in idx:
        states.add_request(
            slot,
            SamplingParams(temperature=1, top_k=20, top_p=0.95, seed=17 + int(slot)),
        )
    for s in (states.temperature, states.top_p, states.top_k, states.seeds):
        s.gpu.copy_(torch.from_numpy(s.np))

    sampler = object.__new__(Sampler)
    sampler.sampling_states = states
    sampler.compute_nans = False
    sampler.use_fp64_gumbel = use_fp64
    sampler.penalties_state = NS(use_penalty=np.zeros(slots), apply_penalties=Mock())
    sampler.logit_bias_state = NS(
        use_logit_bias=np.zeros(slots), apply_logit_bias=Mock()
    )
    sampler.bad_words_state = NS(
        num_bad_words=state(0, np.int32), apply_bad_words=Mock()
    )
    sampler.logprob_token_ids_state = NS(max_num_token_ids=lambda _: 0)
    sampler.logprobs_mode = "raw_logprobs"
    spec = NS(
        method="mtp",
        num_speculative_tokens=max_k,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
    )
    rejection = RejectionSampler(sampler, spec, torch.device(device))

    expanded = tensor(np.repeat(idx, k + 1))
    local = tensor(np.tile(np.arange(k + 1, dtype=np.int32), c))
    indices = torch.arange(1, rows * 2, 2, device=device)
    inputs = torch.zeros(rows * 2, dtype=torch.int32, device=device)
    positions = torch.zeros(rows * 2, dtype=torch.int64, device=device)
    positions[indices] = 8192 + expanded.long() * 32 + local
    batch = NS(
        num_reqs=c,
        num_tokens=rows * 2,
        num_draft_tokens=c * k,
        is_prefilling_np=np.zeros(c, dtype=bool),
        is_dummy_batch=False,
        has_structured_output_reqs=False,
        cu_num_logits_np=cu,
        cu_num_logits=tensor(cu),
        idx_mapping_np=idx,
        idx_mapping=tensor(idx),
        expanded_idx_mapping=expanded,
        expanded_local_pos=local,
        logits_indices=indices,
        input_ids=inputs,
        positions=positions,
        seq_lens=torch.full((c,), 10000, dtype=torch.int32, device=device),
    )
    hidden = torch.rand(rows * 2, 4, device=device)
    target = torch.full((rows, vocab), -14.0, dtype=torch.float16, device=device)
    # FP16 logits, IDs scattered across the vocabulary, with top-p cutting
    # inside top-20. Some proposals have p=0, some p>0; the first request
    # deterministically reaches bonus from a single-token nucleus on rows 0..4.
    ids = torch.as_tensor(
        np.random.default_rng(19).choice(vocab, 64, replace=False), device=device
    )
    target[:, ids] = torch.linspace(5, -3, 64, device=device).half()
    target[:k, ids[0]] = 16
    inputs[indices] = torch.where(local == 2, ids[3], ids[0]).int()
    inputs[indices[: k + 1]] = ids[0].int()
    fallback = Mock(return_value=target)

    def probe(x, width):
        torch.testing.assert_close(x, hidden[indices], atol=0, rtol=0)
        values, top_ids = target.topk(width, dim=-1)
        return top_ids, values, fallback

    runner = NS(
        model=NS(
            compute_logits=Mock(return_value=target),
            get_mtp_target_topk_with_fallback=Mock(side_effect=probe),
        ),
        vllm_config=NS(kernel_config=KernelConfig()),
        speculative_config=spec,
        speculator=NS(draft_logits=None),
        sampler=sampler,
        rejection_sampler=rejection,
        device=torch.device(device),
        lora_config=None,
        req_states=NS(prefill_len=state(1024, np.int32)),
    )
    return runner, batch, hidden, target, fallback


@pytest.mark.parametrize("mode", [True, "device_top64"])
@pytest.mark.parametrize(
    "c,case,k,max_k",
    [
        (1, "hit", 5, 5),
        (2, "hit", 5, 5),
        (4, "mixed", 5, 5),
        (8, "hit", 5, 5),
        (1, "all", 5, 5),
        (1, "hit", 3, 5),
        (2, "hit", 3, 4),
        (2, "hit", 4, 5),
        (2, "mixed", 4, 5),
        (1, "all", 4, 4),
        (1, "hit", 3, 3),
    ],
)
@torch.inference_mode()
def test_mrv2_mtp_compact_vs_dense(
    caplog, monkeypatch, device, c, case, k, max_k, mode
):
    if mode == "device_top64" and case != "hit":
        pytest.skip("cutoff ties have separate bounded-policy coverage")
    caplog.set_level("DEBUG", logger=compact.__name__)
    vocab = 248320
    runner, batch, hidden, target, fallback = make_batch(
        c, device, vocab, use_fp64=c == 8, k=k, max_k=max_k
    )
    if case != "hit":
        # The tie straddles k=20 but not the last column of the wide probe.
        target[(c - 1) * (k + 1), :24] = 24
    dense, dense_counts, dense_rejected = GPUModelRunner.sample(
        runner, hidden, batch, None
    )
    runner.model.compute_logits.assert_called_once()
    runner.model.get_mtp_target_topk_with_fallback.assert_not_called()
    runner.vllm_config.kernel_config.sm70_sparse_target_sampling = mode
    if mode == "device_top64":
        monkeypatch.setattr(
            compact,
            "_compact_target_reference_rows_gpu",
            Mock(side_effect=AssertionError("unexpected host readback")),
        )
    actual, counts, rejected = GPUModelRunner.sample(runner, hidden, batch, None)
    runner.model.compute_logits.assert_called_once()  # No reprojection on fallback.
    runner.model.get_mtp_target_topk_with_fallback.assert_called_once()
    assert fallback.call_count == (case != "hit")
    route = {"hit": "compact", "mixed": "mixed", "all": "dense reason=reference_all"}
    assert f"SM70 MTP compact target route={route[case]}" in caplog.text
    assert torch.equal(counts, dense_counts)
    assert torch.equal(rejected, dense_rejected)
    assert actual.sampled_token_ids.shape == (c, max_k + 1)
    mask = torch.arange(max_k + 1, device=device) < counts[:, None]
    assert torch.equal(actual.sampled_token_ids[mask], dense.sampled_token_ids[mask])
    if case != "all":
        assert counts[0] == k + 1  # Bonus.
    if c > 1:
        assert torch.any(counts[1:] < k + 1)  # Residual.


@pytest.mark.parametrize("mode", [True, "device_top64"])
@torch.inference_mode()
def test_mtp_compact_uses_original_seed_and_rejection_position(device, mode):
    runner, batch, hidden, target, fallback = make_batch(1, device, 248320)
    row, slot = 4, int(batch.idx_mapping_np[0])
    target[row, target[row].argmax()] = 5.0
    batch.input_ids[batch.logits_indices[row + 1]] = target[row].argmin()
    seeds = runner.sampler.sampling_states.seeds
    residual = []
    for seed in (3, 51, 110, 453):
        seeds.np[slot] = seed
        seeds.gpu.copy_(torch.from_numpy(seeds.np))
        runner.vllm_config.kernel_config.sm70_sparse_target_sampling = False
        dense, dense_counts, dense_rejected = GPUModelRunner.sample(
            runner, hidden, batch, None
        )
        runner.vllm_config.kernel_config.sm70_sparse_target_sampling = mode
        actual, counts, rejected = GPUModelRunner.sample(runner, hidden, batch, None)
        assert counts.item() == row + 1 and torch.equal(counts, dense_counts)
        assert torch.equal(rejected, dense_rejected)
        assert torch.equal(
            actual.sampled_token_ids[:, : row + 1],
            dense.sampled_token_ids[:, : row + 1],
        )
        residual.append(actual.sampled_token_ids[0, row].item())
    assert len(set(residual)) > 1
    assert runner.model.compute_logits.call_count == 4
    fallback.assert_not_called()


@pytest.mark.parametrize("device_only", [False, True])
@pytest.mark.parametrize(
    "reason",
    [
        "not_greedy_draft",
        "grammar",
        "lora",
        "prefill_or_mixed_batch",
        "mixed_width",
        "actual_k2",
        "temperature",
        "top_k",
        "top_p",
        "penalties",
        "eos_logit_bias",
    ],
)
def test_unsupported_batch_does_not_project(monkeypatch, reason, device_only):
    monkeypatch.setattr(compact, "_is_sm70", lambda _: True)
    runner, batch, hidden, _, _ = make_batch(
        2, "cpu", 248320, k=2 if reason == "actual_k2" else 5
    )
    spec = runner.speculative_config
    sampler = runner.sampler
    states = sampler.sampling_states
    slot = batch.idx_mapping_np[1]  # One request can force a whole-batch fallback.
    if reason == "not_greedy_draft":
        spec.draft_sample_method = "probabilistic"
    elif reason == "grammar":
        batch.has_structured_output_reqs = True
    elif reason == "prefill_or_mixed_batch":
        batch.is_prefilling_np[1] = True
    elif reason == "mixed_width":
        batch.cu_num_logits_np[-1] -= 1
    elif reason in ("temperature", "top_k", "top_p"):
        getattr(states, reason).np[slot] = dict(temperature=2, top_k=40, top_p=0.8)[
            reason
        ]
    elif reason == "penalties":
        sampler.penalties_state.use_penalty[slot] = True
    elif reason == "eos_logit_bias":
        sampler.logit_bias_state.use_logit_bias[slot] = True
    assert (
        compact.try_mtp_compact_target_rejection(
            runner.model,
            spec,
            None,
            runner.rejection_sampler,
            hidden,
            batch,
            None,
            has_lora=reason == "lora",
            device_only=device_only,
        )
        is None
    )
    runner.model.get_mtp_target_topk_with_fallback.assert_not_called()


@pytest.mark.parametrize("ties", [20, 21, 65])
@torch.inference_mode()
def test_device_top64_cutoff_ties(monkeypatch, device, ties):
    runner, batch, hidden, target, fallback = make_batch(1, device, 248320, k=3)
    target[3].fill_(-14)
    target[3, :ties] = 24
    target[3, :19] = 24.125  # Keep top-p away from an FP32 rounding boundary.
    # The existing selector defines the visible shortlist, including which
    # 64 IDs survive a wider tie. Do not assume a full-vocabulary tie census.
    values, ids = target.topk(64, dim=-1)
    values, ids = compact.sort_topk_with_vocab_ties(
        values.float(), ids, vocab_size=target.shape[1], descending=True
    )

    def reference_support(values, ids):
        values = values.float().masked_fill(values < values[:, 19:20], -float("inf"))
        probabilities = values.softmax(-1)
        keep = probabilities.cumsum(-1) - probabilities < np.float32(0.95)
        filtered = torch.full_like(target, -float("inf"), dtype=torch.float32)
        filtered.scatter_(1, ids, values.masked_fill(~keep, -float("inf")))
        return filtered

    bounded = reference_support(values, ids)
    seeds = runner.sampler.sampling_states.seeds
    for seed in (3, 51, 110, 453):
        seeds.np[batch.idx_mapping_np[0]] = seed
        seeds.gpu.copy_(torch.from_numpy(seeds.np))
        runner.vllm_config.kernel_config.sm70_sparse_target_sampling = False
        if seed == 3:
            dense, dense_counts, dense_rejected = GPUModelRunner.sample(
                runner, hidden, batch, None
            )
            runner.vllm_config.kernel_config.sm70_sparse_target_sampling = True
            guarded, counts, rejected = GPUModelRunner.sample(
                runner, hidden, batch, None
            )
            assert fallback.call_count == (ties > 20)
            assert torch.equal(counts, dense_counts)
            assert torch.equal(rejected, dense_rejected)
            valid = torch.arange(6, device=device) < counts[:, None]
            assert torch.equal(
                guarded.sampled_token_ids[valid], dense.sampled_token_ids[valid]
            )
            fallback.reset_mock()
            runner.vllm_config.kernel_config.sm70_sparse_target_sampling = False
        # Independently filter in PyTorch, then use the dense rejection
        # verifier with the original request slot, seed and absolute positions.
        with monkeypatch.context() as patch:
            patch.setattr(
                runner.sampler, "apply_sampling_params", lambda *args: bounded
            )
            reference, ref_counts, ref_rejected = GPUModelRunner.sample(
                runner, hidden, batch, None
            )
        runner.vllm_config.kernel_config.sm70_sparse_target_sampling = "device_top64"
        with monkeypatch.context() as patch:
            patch.setattr(
                compact,
                "_compact_target_reference_rows_gpu",
                Mock(side_effect=AssertionError("unexpected host readback")),
            )
            actual, counts, rejected = GPUModelRunner.sample(
                runner, hidden, batch, None
            )
        fallback.assert_not_called()
        assert torch.equal(counts, ref_counts)
        assert torch.equal(rejected, ref_rejected)
        valid = torch.arange(6, device=device) < counts[:, None]
        assert torch.equal(
            actual.sampled_token_ids[valid], reference.sampled_token_ids[valid]
        )

    assert torch.isfinite(bounded[3]).sum().item() == {20: 19, 21: 20, 65: 61}[ties]
    if ties > 64:
        # With the same ordering, unbounded tie-inclusive top-k/top-p keeps
        # 62 tokens. This mode deliberately uses only the visible Top64.
        values, ids = target.float().sort(dim=-1, descending=True, stable=True)
        unbounded = reference_support(values, ids)
        assert torch.isfinite(unbounded[3]).sum().item() == 62


@pytest.mark.parametrize("mode", [True, "device_top64"])
@torch.inference_mode()
def test_eos_mask_keeps_mixed_batch_dense(device, mode):
    runner, batch, hidden, target, fallback = make_batch(2, device, 248320, k=3)
    slot = int(batch.idx_mapping_np[1])
    eos = int(target[4].argmax())
    bias = runner.sampler.logit_bias_state
    bias.use_logit_bias[slot] = True

    def mask_eos(logits, expanded, idx, pos):
        # Exercise the public sampler's existing EOS/logit-bias fallback,
        # with one masked request and one ordinary request in the same batch.
        logits[expanded == slot, eos] = -float("inf")

    bias.apply_logit_bias.side_effect = mask_eos
    dense, dense_counts, dense_rejected = GPUModelRunner.sample(
        runner, hidden, batch, None
    )
    runner.vllm_config.kernel_config.sm70_sparse_target_sampling = mode
    actual, counts, rejected = GPUModelRunner.sample(runner, hidden, batch, None)
    runner.model.get_mtp_target_topk_with_fallback.assert_not_called()
    fallback.assert_not_called()
    assert (
        runner.model.compute_logits.call_count == bias.apply_logit_bias.call_count == 2
    )
    assert torch.equal(counts, dense_counts)
    assert torch.equal(rejected, dense_rejected)
    valid = torch.arange(6, device=device) < counts[:, None]
    assert torch.equal(actual.sampled_token_ids[valid], dense.sampled_token_ids[valid])
    assert eos not in actual.sampled_token_ids[1, : counts[1]].tolist()
