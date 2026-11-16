# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-RNG consumer oracle, with adversarial q injected at sampler call sites."""

import math
import os
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.test_w21_native_distribution import interpreter_math
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as dense_module
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as utils
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

pytestmark = pytest.mark.skip_global_cleanup
DEVICE = torch.device(os.environ.get("W21_TEST_DEVICE", "cpu"))


@triton.jit
def _fixed_draws(uniforms, noise):
    step = tl.program_id(0)
    pos = 101 + step
    tl.store(uniforms + step, tl_rand32(37, pos, includes_zero=False))
    keys = tl.arange(0, 4)
    gumbel_seed = tl.randint(37, pos)
    tl.store(noise + step * 4 + keys, tl_rand32(gumbel_seed, keys, includes_zero=False))


def oracle(proposals, p, q, uniforms, noise):
    result = []
    for step, token in enumerate(proposals):
        if uniforms[step] < min(1.0, p[step][token] / q[step][token]):
            result.append(token)
        else:
            residual = [max(0.0, a - b) for a, b in zip(p[step], q[step])]
            break
    else:
        step = len(proposals)
        residual = p[step]
    # Independent Python double-precision residual and token-keyed Gumbels.
    scores = [
        (math.log(prob) - math.log(-math.log1p(-noise[step][token])))
        if prob > 0
        else -math.inf
        for token, prob in enumerate(residual)
    ]
    result.append(max(range(4), key=scores.__getitem__))
    return result


def fixture(width):
    # Reordered noncontiguous slot 2 is the request, slot 0 is adversarial.
    q = torch.zeros((4, 7, 4), device=DEVICE)
    q[:] = torch.tensor([0.8, 0.1, 0.05, 0.05], device=DEVICE)
    q[2] = torch.tensor([0.01, 0.29, 0.35, 0.35], device=DEVICE)
    q[2, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=DEVICE)
    p = torch.tensor(
        [[0.05, 0.80, 0.10, 0.05]] + [[0.60, 0.01, 0.25, 0.14]] * width, device=DEVICE
    )
    proposals = [0] + [1] * (width - 1)
    sampled = torch.tensor([3] + proposals, dtype=torch.int64, device=DEVICE)
    batch = NS(
        num_draft_tokens=width,
        num_reqs=1,
        num_tokens=width + 1,
        is_prefilling_np=np.array([False]),
        has_structured_output_reqs=False,
        input_ids=sampled,
        positions=torch.arange(101, 102 + width, device=DEVICE),
        logits_indices=torch.arange(width + 1, device=DEVICE),
        idx_mapping=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        idx_mapping_np=np.array([2]),
        expanded_idx_mapping=torch.full(
            (width + 1,), 2, dtype=torch.int32, device=DEVICE
        ),
        expanded_local_pos=torch.arange(width + 1, dtype=torch.int32, device=DEVICE),
        cu_num_logits=torch.tensor([0, width + 1], dtype=torch.int32, device=DEVICE),
        seq_lens=torch.tensor([101], dtype=torch.int32, device=DEVICE),
    )
    uniforms = torch.empty(8, device=DEVICE)
    noise = torch.empty((8, 4), device=DEVICE)
    _fixed_draws[(8,)](uniforms, noise, num_warps=1)
    expected = oracle(
        proposals,
        p.double().tolist(),
        q[2].double().tolist(),
        uniforms.tolist(),
        noise.tolist(),
    )
    assert uniforms[0] > 0.1  # wrong-slot and point-mass thresholds below .063
    return batch, p, q, expected


def run_consumer(width, sparse, mutation=None, *, normal_entry=False):
    batch, p, q, expected = fixture(width)
    q_logits = q.log()
    temperature = torch.ones(4, device=DEVICE)
    seeds = torch.full((4,), 37, dtype=torch.int64, device=DEVICE)
    original_dense = utils.rejection_sample
    original_sparse = utils.dflash2_sparse_topk_rejection_sample

    def dense_call(target, draft, *args, **kwargs):
        draft = draft.clone()
        if mutation == "slot":
            draft[2] = draft[0]
        elif mutation == "point_mass":
            draft[2, 0] = -float("inf")
            draft[2, 0, 0] = 0.0
        return original_dense(target, draft, *args, **kwargs)

    def sparse_call(
        target_ids, target_scores, draft_ids, draft_scores, *args, **kwargs
    ):
        draft_ids, draft_scores = draft_ids.clone(), draft_scores.clone()
        if mutation == "slot":
            draft_ids[2], draft_scores[2] = draft_ids[0], draft_scores[0]
        elif mutation == "point_mass":
            draft_ids[2, 0] = torch.arange(4, device=DEVICE)
            draft_scores[2, 0] = -float("inf")
            draft_scores[2, 0, 0] = 0.0
        return original_sparse(
            target_ids, target_scores, draft_ids, draft_scores, *args, **kwargs
        )

    with interpreter_math(DEVICE):
        if sparse:
            # Nonidentity supports ensure ids and scores must travel together.
            ids = torch.tensor([2, 0, 3, 1], device=DEVICE).expand(4, 7, 4).clone()
            ids[0] = torch.tensor([3, 2, 1, 0], device=DEVICE)
            scores = q_logits.gather(-1, ids)
            target_ids = torch.arange(4, device=DEVICE).expand(width + 1, 4)
            tokens, counts = sparse_call(
                target_ids,
                p.log(),
                ids,
                scores,
                batch.input_ids,
                batch.cu_num_logits,
                batch.positions,
                batch.idx_mapping,
                temperature,
                temperature,
                seeds,
                7,
            )
        elif normal_entry:
            from unittest.mock import patch

            sampler = NS(
                compute_nans=False,
                use_fp64_gumbel=False,
                logprobs_mode="processed_logprobs",
                sampling_states=NS(
                    temperature=NS(gpu=temperature), seeds=NS(gpu=seeds)
                ),
                apply_sampling_params=lambda logits, *args: logits,
            )
            rejection = RejectionSampler(
                sampler,
                NS(num_speculative_tokens=7, rejection_sample_method="probabilistic"),
                DEVICE,
            )
            rejection._get_logprobs_tensors = Mock(return_value=None)
            runner = object.__new__(GPUModelRunner)
            runner.device = DEVICE
            runner.rejection_sampler = rejection
            runner.sampler = sampler
            runner.speculator = NS(draft_logits=q_logits)
            runner.model = NS(compute_logits=lambda hidden: p.log())
            runner.req_states = NS(
                prefill_len=NS(gpu=torch.zeros(4, dtype=torch.int32, device=DEVICE))
            )
            with patch.object(dense_module, "rejection_sample", dense_call):
                output, num_sampled, num_rejected = runner.sample(
                    torch.zeros((width + 1, 2), device=DEVICE), batch, None
                )
            tokens, counts = output.sampled_token_ids, output.num_sampled
            assert num_sampled.tolist() == counts.tolist()
            assert num_rejected.tolist() == [width + 1 - len(expected)]
        else:
            tokens, counts = dense_call(
                p.log(),
                q_logits,
                batch.input_ids,
                batch.cu_num_logits,
                batch.positions,
                batch.idx_mapping,
                batch.expanded_idx_mapping,
                batch.expanded_local_pos,
                temperature,
                seeds,
                7,
            )
    assert counts.tolist() == [len(expected)]
    assert tokens[0, : len(expected)].tolist() == expected


@pytest.mark.parametrize("width", [1, 3, 4, 7])
@pytest.mark.parametrize("sparse", [False, True])
def test_fixed_rng_consumer_matches_independent_reference(width, sparse):
    run_consumer(width, sparse)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mutation", ["slot", "point_mass"])
def test_callsite_negative_controls_fail_at_sampling_result(sparse, mutation):
    with pytest.raises(AssertionError):
        run_consumer(7, sparse, mutation)


@pytest.mark.parametrize("mutation", [None, "slot", "point_mass"])
def test_normal_model_runner_sample_combination(mutation):
    if mutation:
        with pytest.raises(AssertionError):
            run_consumer(3, False, mutation, normal_entry=True)
    else:
        run_consumer(3, False, normal_entry=True)
