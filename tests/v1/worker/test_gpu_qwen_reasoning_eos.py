# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.test_w21_native_distribution import (
    _interpreter_compact_full_support,
    interpreter_math,
)
from vllm import envs
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from vllm.v1.sample.ops import topk_topp_sampler
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.input_batch import post_update
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils
from vllm.v1.worker.gpu.spec_decode.dflash2 import sparse_rejection
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _accept_draft
from vllm.v1.worker.gpu.states import RequestState

EOS = (4, 6)
END, TOOL, BODY = 7, 8, 10
VOCAB_SIZE = 32
CONFIG = (EOS, END, TOOL)
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(params=["cpu", "cuda"])
def device(request, monkeypatch):
    device = torch.device(request.param)
    if device.type == "cuda" and (
        os.environ.get("TRITON_INTERPRET") == "1" or not torch.cuda.is_available()
    ):
        pytest.skip("requires the CUDA runtime")
    if device.type == "cpu":
        if os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("requires the Triton interpreter")

        def cpu_uva(size, dtype):
            cpu = torch.zeros(size, dtype=dtype)
            return SimpleNamespace(cpu=cpu, np=cpu.numpy(), uva=cpu)

        monkeypatch.setattr(buffer_utils, "UvaBuffer", cpu_uva)
        monkeypatch.setattr(
            buffer_utils,
            "async_tensor_h2d",
            lambda data, dtype, device: torch.tensor(data, dtype=dtype, device=device),
        )
        monkeypatch.setattr(topk_topp_sampler, "HAS_TRITON", False)
    return device


def make_sampler(device, monkeypatch, temperature=1.0, enabled=True):
    monkeypatch.setattr(envs, "VLLM_QWEN_REASONING_EOS_MASK", enabled)
    reqs = RequestState(4, 128, 16, 8, VOCAB_SIZE, device)
    sampler = Sampler(4, VOCAB_SIZE, device, reqs, num_speculative_tokens=8)
    for slot in range(4):
        sampler.add_request(
            slot,
            2,
            SamplingParams(
                seed=157,
                temperature=temperature,
                top_k=2,
                top_p=0.5,
                _qwen_reasoning_eos=CONFIG if slot in (1, 3) else None,
            ),
            [END, 9],  # The first boundary is in history, before this open turn.
        )
    sampler.apply_staged_writes()
    return sampler, reqs


def make_batch(device, entries):
    slots, drafts = zip(*entries)
    widths = [len(row) + 1 for row in drafts]
    cu = np.cumsum([0, *widths], dtype=np.int32)

    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return SimpleNamespace(
        idx_mapping=ints(slots),
        idx_mapping_np=np.array(slots),
        expanded_idx_mapping=ints(np.repeat(slots, widths)),
        expanded_local_pos=ints([i for n in widths for i in range(n)]),
        cu_num_logits=ints(cu),
        cu_num_logits_np=cu,
        logits_indices=torch.arange(cu[-1], device=device),
        input_ids=ints([i for row in drafts for i in [9, *row]]),
        positions=ints([100 + i for n in widths for i in range(n)]),
        seq_lens=ints([100 + n for n in widths]),
        num_reqs=len(entries),
        num_tokens=int(cu[-1]),
        num_draft_tokens=int(cu[-1]) - len(entries),
        is_prefilling_np=np.zeros(len(entries), dtype=bool),
        has_structured_output_reqs=False,
    )


def processed(sampler, logits, batch):
    return sampler.apply_sampling_params(
        logits,
        batch.expanded_idx_mapping,
        batch.idx_mapping_np,
        batch.positions,
        batch.input_ids,
        batch.expanded_local_pos,
    )


@pytest.mark.parametrize("boundary", [END, TOOL])
@pytest.mark.parametrize("position", [0, 1, 3])
def test_packed_rows_follow_hypothetical_boundary(
    device, monkeypatch, boundary, position
):
    sampler, _ = make_sampler(device, monkeypatch)
    draft = [11, 12, 13, 9]  # Ordinary IDs and a further reasoning opener.
    draft[position] = boundary
    batch = make_batch(device, [(3, draft), (0, [END, EOS[0]]), (1, [])])
    logits = torch.arange(VOCAB_SIZE, device=device).float().repeat(batch.num_tokens, 1)
    logits[:, list(EOS)] = 100
    ref = logits.clone()
    ref[: position + 1, list(EOS)] = -torch.inf
    ref[-1, list(EOS)] = -torch.inf
    baseline, _ = make_sampler(device, monkeypatch, enabled=False)
    torch.testing.assert_close(
        processed(sampler, logits, batch),
        processed(baseline, ref, batch),
        atol=0,
        rtol=0,
    )
    assert sampler.reasoning_eos.slots.gpu[3, 0].item() == EOS[0] + 1


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_ordinary_sampler_keeps_both_eos_off_the_support(
    device, monkeypatch, enabled, temperature
):
    sampler, _ = make_sampler(device, monkeypatch, temperature, enabled)
    batch = make_batch(device, [(3, []), (2, [])])
    logits = torch.full((2, VOCAB_SIZE), -100.0, device=device)
    logits[:, BODY], logits[:, EOS[1]], logits[:, EOS[0]] = 20, 90, 100
    with interpreter_math(device):
        tokens = sampler(logits, batch).sampled_token_ids.flatten().tolist()
    assert tokens == [
        BODY if enabled else EOS[0],
        EOS[0],
    ]


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("mode", ["mtp", "dflash2", "synthetic"])
@pytest.mark.parametrize("eos", EOS)
def test_verifier_rejects_eos_and_accepts_it_after_a_confirmed_draft_boundary(
    device, monkeypatch, temperature, mode, eos
):
    sampler, _ = make_sampler(device, monkeypatch, temperature)
    batch = make_batch(device, [(3, [eos, END, 9]), (1, [TOOL, eos]), (0, [])])
    targets = torch.full((batch.num_tokens, VOCAB_SIZE), -100.0, device=device)
    targets[:, BODY], targets[:, eos] = 20, 100
    targets[4, TOOL] = 90
    q = None
    if mode == "dflash2":
        q = torch.full((4, 8, VOCAB_SIZE), -torch.inf, device=device)
        for slot, proposals in [(3, [eos, END, 9]), (1, [TOOL, eos])]:
            for i, proposal in enumerate(proposals):
                q[slot, i, proposal] = 0
    config = SimpleNamespace(
        num_speculative_tokens=8,
        rejection_sample_method="synthetic" if mode == "synthetic" else "standard",
        synthetic_acceptance_rates=[1.0] * 8,
    )
    with interpreter_math(device):
        result = RejectionSampler(sampler, config, device)(targets, batch, q)
    assert result.num_sampled.tolist() == [1, 3, 1]
    assert result.sampled_token_ids[0, 0].item() == BODY
    assert result.sampled_token_ids[1, :2].tolist() == [TOOL, eos]
    assert result.sampled_token_ids[2, 0].item() == eos
    assert sampler.reasoning_eos.slots.gpu[3, 0].item() == EOS[0] + 1


def test_confirmation_resume_and_slot_reuse(device, monkeypatch):
    sampler, reqs = make_sampler(device, monkeypatch)
    state = sampler.reasoning_eos
    batch = make_batch(device, [(3, []), (1, [])])
    sampled = torch.tensor(
        [[BODY, END], [TOOL, BODY]], dtype=torch.int64, device=device
    )
    counts = torch.tensor([1, 1], dtype=torch.int32, device=device)
    post_update(
        batch.idx_mapping,
        reqs.num_computed_tokens.gpu,
        reqs.last_sampled_tokens,
        None,
        sampled,
        counts,
        torch.zeros_like(counts),
        None,
        reqs.all_token_ids.gpu,
        reqs.total_len.gpu,
        state.slots.gpu,
    )
    # Async scheduling can sample again before the CPU receives the output.
    logits = torch.zeros(2, VOCAB_SIZE, device=device)
    state.mask(
        logits, batch.expanded_idx_mapping, batch.input_ids, batch.expanded_local_pos
    )
    assert torch.isneginf(logits[:, EOS[0]]).tolist() == [True, False]
    assert state.active_requests(batch.idx_mapping_np) == [0, 1]

    params = SamplingParams(_qwen_reasoning_eos=CONFIG)
    still_thinking = Request("thinking", [END, 9], params, None)
    closed = Request("closed", [END, 9], params, None)
    still_thinking.append_output_token_ids(BODY)
    closed.append_output_token_ids(TOOL)
    block_ids = SimpleNamespace(get_block_ids=lambda **kwargs: None)
    cached = Scheduler._make_cached_request_data(
        SimpleNamespace(use_pp=False, use_v2_model_runner=True),
        [still_thinking, closed],
        [],
        {},
        {},
        {"thinking": block_ids, "closed": block_ids},
    )
    reqs.req_id_to_index.update({"thinking": 3, "closed": 1})
    GPUModelRunner.update_requests(
        SimpleNamespace(sampler=sampler, req_states=reqs),
        SimpleNamespace(mamba_state_gathers=None, scheduled_cached_reqs=cached),
    )
    assert state.active_requests(batch.idx_mapping_np) == [0]

    sampler.add_request(3, 2, params, [END, 9, TOOL, 9])
    assert not state.active_requests(batch.idx_mapping_np)
    sampler.add_request(1, 2, params, [END, 9, BODY])
    sampler.apply_staged_writes()
    assert state.active_requests(batch.idx_mapping_np) == [1]
    logits.zero_()
    state.mask(
        logits, batch.expanded_idx_mapping, batch.input_ids, batch.expanded_local_pos
    )
    assert torch.isneginf(logits[:, EOS[0]]).tolist() == [False, True]


@triton.jit
def _interpreter_top20_support(
    ids,
    logits,
    stride,
    row,
    top_p,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LIMIT_K: tl.constexpr,
):
    # This fixture uses top_p=1. The older interpreter helper expects a
    # temperature argument even though the verifier now scales before dispatch.
    return _interpreter_compact_full_support(
        ids, logits, stride, row, 1.0, top_p, TOP_K, BLOCK_K, LIMIT_K
    )


@pytest.mark.parametrize(
    "mixed, retain_logits", [(False, True), (True, True), (True, False)]
)
def test_dflash2_compact_routes_guarded_request_to_dense(
    device, monkeypatch, mixed, retain_logits
):
    sampler, _ = make_sampler(device, monkeypatch)
    sampler.sampling_states.top_k.np[:] = 20
    sampler.sampling_states.top_p.np[:] = 1.0
    sampler.apply_staged_writes()
    config = SimpleNamespace(
        num_speculative_tokens=8, rejection_sample_method="standard"
    )
    rejection = RejectionSampler(sampler, config, device)
    batch = make_batch(device, [(3, [EOS[0]]), (0 if mixed else 1, [EOS[0]])])
    target = torch.arange(VOCAB_SIZE, device=device).float().repeat(4, 1)
    target[:, EOS[0]] = 100
    q = torch.full((4, 8, VOCAB_SIZE), -torch.inf, device=device)
    q[:, :, EOS[0]] = 0
    ids = torch.full((4, 8, 1), EOS[0], dtype=torch.int64, device=device)
    spec = object.__new__(DFlash2Speculator)
    spec.draft_logits = q
    spec.get_sparse_draft_logits = lambda: (ids, q[:, :, EOS[0] : EOS[0] + 1])
    calls = []

    def selector(hidden, k):
        calls.append("compact")
        values, ids = target.topk(k)

        def fallback():
            calls.append("dense")
            return target

        return ids, values, fallback if retain_logits else None

    model = SimpleNamespace(
        get_topk_tokens_and_logits=selector,
        get_topk_tokens_and_logits_with_fallback=selector,
    )
    monkeypatch.setattr(sparse_rejection, "sm70_dflash2_enabled", lambda *a: True)
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    elif torch.cuda.get_device_capability(device) != (7, 0):
        pytest.skip("requires a V100")
    hidden = SimpleNamespace(device=torch.device("cuda"))
    compact_target_row = rejection_sampler_utils._dflash2_compact_target_row
    with (
        interpreter_math(device),
        patch.object(
            rejection_sampler_utils,
            "_dflash2_compact_target_row",
            _interpreter_top20_support if device.type == "cpu" else compact_target_row,
        ),
    ):
        result = sparse_rejection.try_dflash2_sparse_target_rejection(
            model, spec, rejection, hidden, batch, None
        )
        if not mixed or not retain_logits:
            assert result is None and calls == (["compact"] if mixed else [])
            result = rejection(target, batch, q)
        else:
            assert calls == ["compact", "dense"]
    assert result.num_sampled.tolist() == ([1, 2] if mixed else [1, 1])
    assert result.sampled_token_ids[0, 0].item() not in EOS
    assert (result.sampled_token_ids[1, 0].item() == EOS[0]) == mixed


@triton.jit
def _test_zero_uniform(result):
    log_zero = tl.log(0.0)
    tl.store(result, _accept_draft(log_zero, log_zero + tl.log(0.5)))
    tl.store(result + 1, _accept_draft(tl.log(0.125), log_zero + tl.log(0.5)))


def test_zero_uniform_cannot_accept_an_excluded_proposal(device):
    result = torch.empty(2, dtype=torch.bool, device=device)
    _test_zero_uniform[(1,)](result)
    assert result.tolist() == [False, True]
