# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config import KernelConfig
from vllm.model_executor.layers import logits_processor
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample import states as sampling_states
from vllm.v1.worker.gpu.sample.reasoning_eos import ReasoningEOSState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.spec_decode import mtp_compact_target as compact
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


@triton.jit
def _interpreter_log1p(x):
    return tl.log(1.0 + x)


@pytest.fixture
def device(monkeypatch):
    if os.getenv("TRITON_INTERPRET") == "1":
        monkeypatch.setattr(compact, "_is_sm70", lambda _: True)
        monkeypatch.setattr(tldevice, "log1p", _interpreter_log1p)
        from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

        # Interpreter 3.6 casts NumPy float comparison results back to the
        # operand dtype, which breaks bitwise ops on these declared predicates.
        for predicate, op in (("GE", np.greater_equal), ("LT", np.less)):

            def compare(builder, lhs, rhs, op=op):
                return TensorHandle(op(lhs.data, rhs.data), tl.int1)

            # Both are ordered float predicates (the `O` prefix).
            monkeypatch.setattr(InterpreterBuilder, f"create_fcmpO{predicate}", compare)
        # Triton's interpreter cannot cast the Python loop index in the
        # dense top-k/top-p kernel. Use the existing PyTorch reference on
        # these inputs; CUDA uses the real MRv2 dense filter without patches.
        monkeypatch.setattr(
            sampling_states,
            "apply_top_k_top_p",
            lambda logits, k, p, **_: apply_top_k_top_p_pytorch(logits, k, p),
        )
        return "cpu"
    if not compact._is_sm70(torch.device("cuda")):
        pytest.skip("requires SM70 or TRITON_INTERPRET=1")
    return "cuda"


def make_batch(c, device, vocab, use_fp64=False, *, k=5, max_k=5, drafts=None):
    slots = 12
    drafts = np.full(c, k, dtype=np.int32) if drafts is None else np.asarray(drafts)
    widths = drafts + 1
    idx = np.array([7, 2, 11, 0, 6, 3, 9, 5][:c], dtype=np.int32)
    cu = np.insert(widths.cumsum(dtype=np.int32), 0, 0)
    rows = int(cu[-1])

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

    expanded = tensor(np.repeat(idx, widths))
    local = tensor(np.concatenate([np.arange(w, dtype=np.int32) for w in widths]))
    indices = torch.arange(1, rows * 2, 2, device=device)
    inputs = torch.zeros(rows * 2, dtype=torch.int32, device=device)
    positions = torch.zeros(rows * 2, dtype=torch.int64, device=device)
    positions[indices] = 8192 + expanded.long() * 32 + local
    batch = NS(
        num_reqs=c,
        num_tokens=rows * 2,
        num_draft_tokens=int(drafts.sum()),
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
    target[: drafts[0], ids[0]] = 16
    inputs[indices] = torch.where(local == 2, ids[3], ids[0]).int()
    inputs[indices[: widths[0]]] = ids[0].int()
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


def test_config_off_and_no_compile_hash_change():
    off = KernelConfig()
    on = KernelConfig(sm70_sparse_target_sampling=True)
    assert off.sm70_sparse_target_sampling is False
    assert on.sm70_sparse_target_sampling is True
    assert off.compute_hash() == on.compute_hash()


@pytest.mark.parametrize(
    "k,reference_requests",
    [(5, [1, 5]), (4, [1, 5]), (3, [1, 5]), (5, []), (5, list(range(8)))],
)
@torch.inference_mode()
def test_partial_reference_gather_preserves_dense_result(
    monkeypatch, device, k, reference_requests
):
    vocab = 248320
    width = k + 1
    runner, batch, hidden, target, _ = make_batch(8, device, vocab, k=k, max_k=5)
    target[:, 31] = -20 + torch.arange(8 * width, device=device) / 32
    for req in reference_requests:
        # Force a strict top-20 tie spanning candidates 20/21 inside Top64.
        offset = (req // 2) * (vocab // 4)
        target[req * width, offset : offset + 24] = 24
    runner.vllm_config.parallel_config = NS(distributed_executor_backend="mp")
    processor = logits_processor.LogitsProcessor(vocab)
    processor.use_all_gather = True
    shards = target.chunk(4, dim=-1)
    cache = logits_processor.CachedLogitsFallback(processor, shards[0].contiguous())
    expected_rows = [
        req * width + step for req in reference_requests for step in range(width)
    ]
    gathered_counts = []

    def all_gather(local: torch.Tensor, dim: int = -1) -> torch.Tensor:
        gathered_counts.append(local.shape[0])
        parts = shards
        if local.shape[0] != target.shape[0]:
            parts = tuple(shard[expected_rows] for shard in shards)
        assert torch.equal(parts[0], local)
        return torch.cat(parts, dim=dim)

    monkeypatch.setattr(
        logits_processor, "tensor_model_parallel_all_gather", all_gather
    )

    def probe(_, top_k):
        values, ids = target.topk(top_k, dim=-1)
        return ids, values, cache

    runner.model.get_mtp_target_topk_with_fallback.side_effect = probe
    dense, dense_counts, dense_rejected = GPUModelRunner.sample(
        runner, hidden, batch, None
    )
    runner.vllm_config.kernel_config.sm70_sparse_target_sampling = True
    result, counts, rejected = GPUModelRunner.sample(runner, hidden, batch, None)
    assert torch.equal(counts, dense_counts)
    assert torch.equal(rejected, dense_rejected)
    assert torch.all(counts + rejected == width)
    assert result.sampled_token_ids.shape == (8, 6)
    assert result.sampled_token_ids.stride() == (6, 1)
    valid = torch.arange(6, device=device) < counts[:, None]
    assert torch.equal(result.sampled_token_ids[valid], dense.sampled_token_ids[valid])
    assert gathered_counts == ([len(expected_rows)] if expected_rows else [])
    runner.model.compute_logits.assert_called_once()


@pytest.mark.parametrize(
    "c,case,k,max_k",
    [
        (1, "hit", 5, 5),
        (2, "hit", 5, 5),
        (4, "mixed", 5, 5),
        (4, "mixed_device", 5, 5),
        (4, "mixed_device_top64", 5, 5),
        (8, "hit", 5, 5),
        (1, "all", 5, 5),
        (1, "hit", 2, 3),
        (1, "hit", 2, 4),
        (1, "hit", 2, 5),
        (1, "hit", 2, 6),
        (1, "all", 2, 6),
        (1, "hit", 3, 3),
        (2, "hit", 3, 4),
        (2, "hit", 3, 5),
        (4, "mixed", 3, 5),
        (1, "all", 3, 5),
        (2, "hit", 4, 4),
        (4, "mixed", 4, 4),
        (8, "mixed", 4, 5),
        (1, "all", 4, 5),
    ],
)
@torch.inference_mode()
def test_mrv2_mtp_compact_vs_dense(caplog, device, monkeypatch, c, case, k, max_k):
    caplog.set_level("DEBUG", logger=compact.__name__)
    typed_device = case == "mixed_device_top64"
    device_only = case == "mixed_device" or typed_device
    if device_only:
        case = "mixed"
    vocab = 248320
    runner, batch, hidden, target, fallback = make_batch(
        c, device, vocab, use_fp64=c == 8, k=k, max_k=max_k
    )
    if case != "hit":
        # The tie straddles k=20 but not the last column of the wide probe.
        target[(c - 1) * (k + 1), :24] = 24
    if case == "mixed":
        # Exercise original slot/position mapping in the dense subset, also
        # when a mask depends on the proposed tokens of a guarded request.
        reasoning = object.__new__(ReasoningEOSState)
        slot = int(batch.idx_mapping_np[1])
        reasoning.requests = {slot}
        slots = torch.zeros((12, 4), dtype=torch.int32, device=device)
        eos = int(target[k + 1].argmax())
        slots[slot] = torch.tensor([eos + 1, eos, vocab - 1, vocab - 2], device=device)
        reasoning.slots = NS(gpu=slots)
        runner.sampler.reasoning_eos = reasoning
    dense, dense_counts, dense_rejected = GPUModelRunner.sample(
        runner, hidden, batch, None
    )
    runner.model.compute_logits.assert_called_once()
    runner.model.get_mtp_target_topk_with_fallback.assert_not_called()
    runner.vllm_config.kernel_config.sm70_sparse_target_sampling = True
    if device_only:
        config = runner.vllm_config.kernel_config
        config.sm70_sparse_target_sampling = "device_top64"
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
    assert torch.all(counts + rejected == k + 1)
    assert (
        actual.sampled_token_ids.shape
        == dense.sampled_token_ids.shape
        == (c, max_k + 1)
    )
    assert (
        actual.sampled_token_ids.stride()
        == dense.sampled_token_ids.stride()
        == (max_k + 1, 1)
    )
    mask = torch.arange(max_k + 1, device=device) < counts[:, None]
    assert torch.equal(actual.sampled_token_ids[mask], dense.sampled_token_ids[mask])
    if case != "all":
        assert counts[0] == k + 1  # Bonus.
    if c > 1:
        assert torch.any(counts[1:] < k + 1)  # Residual.


@pytest.mark.parametrize(
    ("c", "k", "mode"), [(1, 2, "device_top64"), (8, 5, "device_top64")]
)
@torch.inference_mode()
def test_device_only_matches_dense_without_cutoff_ties(monkeypatch, device, c, k, mode):
    runner, batch, hidden, _, fallback = make_batch(
        c, device, 248320, use_fp64=c == 8, k=k
    )
    dense, dense_counts, dense_rejected = GPUModelRunner.sample(
        runner, hidden, batch, None
    )
    guard = Mock(side_effect=AssertionError("unexpected host readback"))
    monkeypatch.setattr(compact, "_compact_target_reference_rows_gpu", guard)
    config = runner.vllm_config.kernel_config
    config.sm70_sparse_target_sampling = mode
    actual, counts, rejected = GPUModelRunner.sample(runner, hidden, batch, None)
    guard.assert_not_called()
    fallback.assert_not_called()
    runner.model.compute_logits.assert_called_once()
    assert torch.equal(counts, dense_counts)
    assert torch.equal(rejected, dense_rejected)
    assert actual.sampled_token_ids.stride() == (6, 1)
    valid = torch.arange(6, device=device) < counts[:, None]
    assert torch.equal(actual.sampled_token_ids[valid], dense.sampled_token_ids[valid])


@torch.inference_mode()
def test_mtp_compact_uses_original_seed_and_rejection_position(device):
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
        runner.vllm_config.kernel_config.sm70_sparse_target_sampling = True
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


@pytest.mark.parametrize("reject_at", [0, 1, None])
@pytest.mark.parametrize("use_fp64", [False, True])
@torch.inference_mode()
def test_single_request_k2_uses_original_slot_and_seed(device, reject_at, use_fp64):
    runner, batch, hidden, target, fallback = make_batch(
        1, device, 248320, use_fp64, k=2, max_k=6
    )
    slot = int(batch.idx_mapping_np[0])
    row = 2 if reject_at is None else reject_at
    counts_expected = row + 1
    if reject_at is not None:
        top = target[row].argmax()
        target[row, top] = 5.0
        # This proposal is outside the compact support. Other rows before it
        # accept deterministically, so both residual positions are exercised.
        batch.input_ids[batch.logits_indices[reject_at + 1]] = target[row].argmin()
    seeds = runner.sampler.sampling_states.seeds
    sampled_at_branch = []
    for seed in (3, 51, 110, 453):
        seeds.np[slot] = seed
        seeds.gpu.copy_(torch.from_numpy(seeds.np))
        config = runner.vllm_config.kernel_config
        config.sm70_sparse_target_sampling = False
        dense, dense_counts, dense_rejected = GPUModelRunner.sample(
            runner, hidden, batch, None
        )
        config.sm70_sparse_target_sampling = True
        actual, counts, rejected = GPUModelRunner.sample(runner, hidden, batch, None)
        assert counts.item() == counts_expected
        assert rejected.item() == 3 - counts_expected
        assert torch.equal(counts, dense_counts)
        assert torch.equal(rejected, dense_rejected)
        assert torch.equal(
            actual.sampled_token_ids[:, :counts_expected],
            dense.sampled_token_ids[:, :counts_expected],
        )
        sampled_at_branch.append(actual.sampled_token_ids[0, row].item())
    # Validate that this probe actually distinguishes random streams.
    assert len(set(sampled_at_branch)) > 1
    assert runner.model.compute_logits.call_count == 4
    assert runner.model.get_mtp_target_topk_with_fallback.call_count == 4
    fallback.assert_not_called()


@pytest.mark.parametrize(("drafts", "max_k"), [([2, 5], 5), ([2, 6], 6), ([2, 2], 5)])
@torch.inference_mode()
def test_k2_mixed_or_multiple_requests_keep_dense(device, drafts, max_k):
    runner, batch, hidden, _, fallback = make_batch(
        2, device, 248320, max_k=max_k, drafts=drafts
    )
    dense, dense_counts, dense_rejected = GPUModelRunner.sample(
        runner, hidden, batch, None
    )
    runner.vllm_config.kernel_config.sm70_sparse_target_sampling = True
    actual, counts, rejected = GPUModelRunner.sample(runner, hidden, batch, None)
    assert torch.equal(counts, dense_counts)
    assert torch.equal(rejected, dense_rejected)
    assert torch.equal(counts + rejected, torch.tensor(drafts, device=device) + 1)
    valid = torch.arange(max_k + 1, device=device) < counts[:, None]
    assert torch.equal(actual.sampled_token_ids[valid], dense.sampled_token_ids[valid])
    assert runner.model.compute_logits.call_count == 2
    runner.model.get_mtp_target_topk_with_fallback.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize(
    "reason",
    [
        "not_sm70",
        "not_greedy_draft",
        "grammar",
        "lora",
        "prefill_or_mixed_batch",
        "mixed_width",
        "k1",
        "config_cap2",
        "uncertified_cap6_k5",
        "uncertified_k6",
        "above_configured",
        "unsupported_configured",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "penalties",
        "logit_bias",
        "logprobs",
        "bad_words",
        "compute_nans",
        "synthetic",
        "reasoning_eos_all",
        "unsupported_target",
    ],
)
def test_unsupported_batch_does_not_project(monkeypatch, reason):
    monkeypatch.setattr(compact, "_is_sm70", lambda _: reason != "not_sm70")
    k = {"k1": 1, "config_cap2": 2, "uncertified_k6": 6}.get(reason, 5)
    c = 1 if reason in ("config_cap2", "uncertified_cap6_k5", "uncertified_k6") else 2
    runner, batch, hidden, _, _ = make_batch(c, "cpu", 248320, k=k, max_k=max(k, 5))
    spec = runner.speculative_config
    sampler = runner.sampler
    states = sampler.sampling_states
    probe = runner.model.get_mtp_target_topk_with_fallback
    slot = batch.idx_mapping_np[-1]  # One request can force a whole-batch fallback.
    if reason == "not_greedy_draft":
        spec.draft_sample_method = "probabilistic"
    elif reason == "grammar":
        batch.has_structured_output_reqs = True
    elif reason == "prefill_or_mixed_batch":
        batch.is_prefilling_np[1] = True
    elif reason == "mixed_width":
        batch.cu_num_logits_np[-1] -= 1
    elif reason == "above_configured":
        runner.rejection_sampler.num_speculative_steps = 4
    elif reason == "unsupported_configured":
        runner.rejection_sampler.num_speculative_steps = 7
    elif reason in ("config_cap2", "uncertified_cap6_k5"):
        runner.rejection_sampler.num_speculative_steps = (
            2 if reason == "config_cap2" else 6
        )
    elif reason in ("temperature", "top_k", "top_p", "min_p"):
        getattr(states, reason).np[slot] = dict(
            temperature=0, top_k=40, top_p=0.8, min_p=0.1
        )[reason]
    elif reason == "penalties":
        sampler.penalties_state.use_penalty[slot] = True
    elif reason == "logit_bias":
        sampler.logit_bias_state.use_logit_bias[slot] = True
    elif reason == "logprobs":
        states.num_logprobs[slot] = 0
    elif reason == "bad_words":
        sampler.bad_words_state.num_bad_words.np[slot] = 1
    elif reason == "compute_nans":
        sampler.compute_nans = True
    elif reason == "synthetic":
        runner.rejection_sampler.rejection_sample_method = "synthetic"
    elif reason == "reasoning_eos_all":
        sampler.reasoning_eos = NS(active_requests=lambda _: [0, 1])
    elif reason == "unsupported_target":
        del runner.model.get_mtp_target_topk_with_fallback
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
        )
        is None
    )
    probe.assert_not_called()


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

    def filter_support(values):
        values = values.float().masked_fill(values < values[..., 19:20], -float("inf"))
        probabilities = values.softmax(-1)
        keep = probabilities.cumsum(-1) - probabilities < np.float32(0.95)
        return values.masked_fill(~keep, -float("inf"))

    support = filter_support(values)
    assert torch.isfinite(support[3]).sum().item() == {20: 19, 21: 20, 65: 61}[ties]
    if ties == 65:
        unbounded = filter_support(target[3].sort(descending=True).values)
        assert torch.isfinite(unbounded).sum().item() == 62

    bounded = torch.full_like(target, -float("inf"), dtype=torch.float32)
    bounded.scatter_(1, ids, support)
    seeds = runner.sampler.sampling_states.seeds
    for seed in (3, 51):
        seeds.np[batch.idx_mapping_np[0]] = seed
        seeds.gpu.copy_(torch.from_numpy(seeds.np))
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


@torch.inference_mode()
def test_eos_mask_keeps_mixed_batch_dense(device):
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
    runner.vllm_config.kernel_config.sm70_sparse_target_sampling = "device_top64"
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
