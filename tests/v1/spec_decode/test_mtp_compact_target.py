# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from functools import partial
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config import KernelConfig
from vllm.model_executor.layers import logits_processor
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample import states as sampling_states
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


def make_batch(c, device, vocab, use_fp64=False):
    k, slots = 5, 12
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
        num_speculative_tokens=k,
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
    target[:5, ids[0]] = 16
    inputs[indices] = torch.where(local == 2, ids[3], ids[0]).int()
    inputs[indices[:6]] = ids[0].int()
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


def test_public_qwen_target_reuses_original_local_logits(monkeypatch):
    vocab, local_vocab = 248320, 62080
    parts = [torch.full((2, local_vocab), -40.0, dtype=torch.float16) for _ in range(4)]
    for rank, part in enumerate(parts):
        part[:, :64] = torch.arange(64) / 32 + rank / 4
    processor = logits_processor.LogitsProcessor(vocab)
    processor.use_all_gather = True
    projection = Mock(return_value=parts[0])
    head = NS(
        maybe_get_sm70_dflash2_top20=Mock(return_value=None),
        quant_method=NS(apply=projection),
        shard_indices=NS(num_org_vocab_padding=0, org_vocab_start_index=0),
    )
    hidden = torch.zeros((2, 4), dtype=torch.float16)
    gathered = []

    def all_gather(local, dim=-1):
        gathered.append(local.shape[-1])
        assert dim == -1
        if local.shape[-1] == local_vocab:
            assert local is parts[0]
            return torch.cat(parts, dim=-1)
        top = [part.float().topk(64) for part in parts]
        if local.dtype == torch.int64:
            return torch.cat(
                [ids + rank * local_vocab for rank, (_, ids) in enumerate(top)], dim=-1
            )
        return torch.cat([values for values, _ in top], dim=-1)

    monkeypatch.setattr(
        logits_processor, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        logits_processor, "tensor_model_parallel_all_gather", all_gather
    )
    language_model = NS(logits_processor=processor, lm_head=head)
    language_model.get_mtp_target_topk_with_fallback = partial(
        Qwen4ExpForCausalLM.get_mtp_target_topk_with_fallback, language_model
    )
    wrapper = NS(language_model=language_model)
    ids, scores, reference = (
        Qwen4ExpForConditionalGeneration.get_mtp_target_topk_with_fallback(
            wrapper, hidden, 64
        )
    )
    dense = torch.cat(parts, dim=-1)
    assert gathered == [64, 64]
    assert torch.equal(scores, dense.float().topk(64).values)
    assert torch.equal(dense.float().gather(-1, ids), scores)
    assert reference is not None and torch.equal(reference(), dense)
    assert gathered == [64, 64, local_vocab]
    projection.assert_called_once_with(head, hidden, bias=None)


@pytest.mark.parametrize(
    "c,case", [(1, "hit"), (2, "hit"), (4, "mixed"), (8, "hit"), (1, "all")]
)
@torch.inference_mode()
def test_mrv2_mtp_compact_vs_dense(caplog, device, c, case):
    caplog.set_level("DEBUG", logger=compact.__name__)
    vocab = 248320
    runner, batch, hidden, target, fallback = make_batch(
        c, device, vocab, use_fp64=c == 8
    )
    if case != "hit":
        # The tie straddles k=20 but not the last column of the wide probe.
        target[(c - 1) * 6, :24] = 24
    dense, dense_counts, dense_rejected = GPUModelRunner.sample(
        runner, hidden, batch, None
    )
    runner.model.compute_logits.assert_called_once()
    runner.model.get_mtp_target_topk_with_fallback.assert_not_called()
    runner.vllm_config.kernel_config.sm70_sparse_target_sampling = True
    actual, counts, rejected = GPUModelRunner.sample(runner, hidden, batch, None)
    runner.model.compute_logits.assert_called_once()  # No reprojection on fallback.
    runner.model.get_mtp_target_topk_with_fallback.assert_called_once()
    assert fallback.call_count == (case != "hit")
    route = {"hit": "compact", "mixed": "mixed", "all": "dense reason=reference_all"}
    assert f"SM70 MTP compact target route={route[case]}" in caplog.text
    assert torch.equal(counts, dense_counts)
    assert torch.equal(rejected, dense_rejected)
    mask = torch.arange(6, device=device) < counts[:, None]
    assert torch.equal(actual.sampled_token_ids[mask], dense.sampled_token_ids[mask])
    if case != "all":
        assert counts[0] == 6  # Bonus.
    if c > 1:
        assert torch.any(counts[1:] < 6)  # Residual.


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


@pytest.mark.parametrize(
    "reason",
    [
        "not_sm70",
        "not_mtp",
        "not_greedy_draft",
        "grammar",
        "lora",
        "prefill_or_mixed_batch",
        "not_uniform_k5",
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
        "unsupported_target",
    ],
)
def test_unsupported_batch_does_not_project(monkeypatch, reason):
    monkeypatch.setattr(compact, "_is_sm70", lambda _: reason != "not_sm70")
    runner, batch, hidden, _, _ = make_batch(2, "cpu", compact._VOCAB_SIZE)
    spec = runner.speculative_config
    sampler = runner.sampler
    states = sampler.sampling_states
    slot = batch.idx_mapping_np[1]  # One request can force a whole-batch fallback.
    if reason == "not_mtp":
        spec.method = "eagle"
    elif reason == "not_greedy_draft":
        spec.draft_sample_method = "probabilistic"
    elif reason == "grammar":
        batch.has_structured_output_reqs = True
    elif reason == "prefill_or_mixed_batch":
        batch.is_prefilling_np[1] = True
    elif reason == "not_uniform_k5":
        batch.cu_num_logits_np[-1] -= 1
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
    elif reason == "unsupported_target":
        states.vocab_size = 32768
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
    runner.model.get_mtp_target_topk_with_fallback.assert_not_called()
