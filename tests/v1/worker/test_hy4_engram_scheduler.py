# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.platforms import current_platform
from vllm.platforms.interface import PlatformEnum
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.skip_global_cleanup


def make_runner(monkeypatch):
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.input_batch = InputBatch(
        max_num_reqs=4,
        max_model_len=96,
        max_num_batched_tokens=9,
        device=runner.device,
        pin_memory=False,
        vocab_size=1000,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    runner.requests = {}
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.mamba_state_idx = {}
    runner._mamba_accepted_token_state_rows = {}
    runner._dynamic_draft_vocab_prefill_bootstrap = Mock()
    runner.late_interaction_runner = Mock()
    runner._may_reorder_batch = lambda output: None
    runner.speculative_config = None
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.use_async_scheduling = False
    runner.use_async_spec_decode = False
    runner.sync_spec_decode_accept_counts = False
    runner.uses_flash_engram = True
    runner.ngram_context_len = 2
    runner.ngram_eos_token_id = 999
    runner.enable_prompt_embeds = False
    runner._sm70_async_staged_input_prep_active = False
    runner.ngram_context = SimpleNamespace(
        np=np.zeros((4, 2), dtype=np.int32), gpu=torch.zeros((4, 2), dtype=torch.int32)
    )
    runner._copy_buffer_to_gpu = lambda buf, count: buf.gpu[:count].copy_(
        torch.from_numpy(buf.np[:count])
    )
    monkeypatch.setattr(
        gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    return runner


def make_request(req_id, prompt, max_tokens):
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens),
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


@pytest.mark.parametrize("budget", [1, 3, 5, 9])
def test_history_through_real_scheduler_and_runner(monkeypatch, budget):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CPU)
    # Initialize the shared block-hash seed using the scheduler test helper.
    create_requests(0)
    scheduler = create_scheduler(
        max_num_seqs=min(4, budget),
        max_num_batched_tokens=budget,
        max_model_len=96,
        long_prefill_token_threshold=5,
        enable_prefix_caching=True,
        num_blocks=32,
        block_size=16,
    )
    assert isinstance(scheduler, Scheduler)
    assert isinstance(scheduler.kv_cache_manager, KVCacheManager)
    runner = make_runner(monkeypatch)
    prompt = list(range(10, 50))
    warm = make_request("warm", prompt, 1)
    follower = make_request("follower", prompt[:32] + list(range(80, 89)), 6)
    other = make_request("other", list(range(100, 122)), 2)
    requests = {r.request_id: r for r in (warm, follower, other)}
    scheduler.add_request(warm)
    history = []
    resumed = False

    def step():
        nonlocal resumed
        output = scheduler.schedule()
        assert output.num_scheduled_tokens
        runner._update_states(output)
        kwargs: dict[str, torch.Tensor] = {}
        runner._maybe_add_flash_engram_kwargs(
            kwargs,
            num_reqs=runner.input_batch.num_reqs,
            num_reqs_padded=runner.input_batch.num_reqs,
            is_first_rank=True,
            is_encoder_decoder=False,
            use_dummy_context=False,
        )
        if "follower" in output.scheduled_cached_reqs.resumed_req_ids:
            resumed = True
        sampled = []
        for i, req_id in enumerate(runner.input_batch.req_ids):
            request = requests[req_id]
            cursor = request.num_computed_tokens - output.num_scheduled_tokens[req_id]
            previous = list(request.all_token_ids[:cursor])[-2:]
            expected = [999] * (2 - len(previous)) + previous
            assert kwargs["lookback_token_ids"][i].tolist() == expected
            history.append((req_id, cursor, expected))
            # Simulate the CPU sampler; runner and scheduler each retain its output.
            tokens = (
                [200 + len(request.output_token_ids)]
                if request.num_computed_tokens == request.num_tokens
                else []
            )
            sampled.append(tokens)
            if tokens:
                pos = runner.input_batch.num_tokens_no_spec[i]
                runner.input_batch.token_ids_cpu[i, pos] = tokens[0]
                runner.input_batch.is_token_ids[i, pos] = True
                runner.input_batch.num_tokens_no_spec[i] += 1
                runner.requests[req_id].output_token_ids.extend(tokens)
        ids = list(runner.input_batch.req_ids)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={r: i for i, r in enumerate(ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

    while warm.request_id in scheduler.requests:
        step()
    scheduler.add_request(follower)
    scheduler.add_request(other)
    preempted = False
    for _ in range(100):
        if not scheduler.requests:
            break
        step()
        if not preempted and len(follower.output_token_ids) >= 2:
            assert follower.status == RequestStatus.RUNNING
            scheduler._preempt_request(follower, 0.0)
            scheduler.running.remove(follower)
            scheduler.prev_step_scheduled_req_ids.discard(follower.request_id)
            preempted = True
    assert not scheduler.requests
    assert preempted and resumed and follower.num_preemptions == 1
    follower_cursors = [cursor for req, cursor, _ in history if req == "follower"]
    assert follower_cursors[0] == 32  # two cached full pages
    assert history[0] == ("warm", 0, [999, 999])
    if budget == 1:
        assert history[1] == ("warm", 1, [999, prompt[0]])
    assert len({cursor % 16 for req, cursor, _ in history if req == "warm"}) >= 4
    assert any(
        req == "follower" and cursor > len(follower.prompt_token_ids)
        for req, cursor, _ in history
    )
    assert any(req == "other" for req, _, _ in history)
