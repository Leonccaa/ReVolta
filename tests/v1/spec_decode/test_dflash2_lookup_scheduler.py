# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real scheduler, worker state, input and draft handoff with synthetic sampling.

CPU substitutes mapped storage and CUDA event allocation. The scheduling,
state updates, input kernels, native lookup and handler methods are real.
Set W21_TEST_DEVICE=cuda to exercise the same chain with UVA and CUDA events.
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.v1.spec_decode.test_dflash2_native_lookup import reference_lookup
from vllm import SamplingParams, envs
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.sampling_params import StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputGrammar, StructuredOutputManager
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash2.native_lookup import NativeDFlash2Lookup
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device(os.environ.get("W21_TEST_DEVICE", "cpu"))
pytestmark = pytest.mark.skip_global_cleanup
pytest_plugins = ("tests.v1.spec_decode.test_dflash2_native_lookup",)


class CpuMappedBuffer:
    def __init__(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.uva = self.cpu


class CpuStream:
    def __init__(self, *args, **kwargs):
        pass

    def wait_stream(self, other):
        pass


class CpuEvent:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args):
        pass

    def synchronize(self):
        pass


@pytest.fixture
def chain_factory(native_allocation_config, monkeypatch):
    if DEVICE.type == "cpu":
        monkeypatch.setattr(buffer_utils, "UvaBuffer", CpuMappedBuffer)
        monkeypatch.setattr(
            buffer_utils,
            "async_tensor_h2d",
            lambda data, dtype, device: torch.tensor(data, dtype=dtype, device=device),
        )
        monkeypatch.setattr(torch.cuda, "Stream", CpuStream)
        monkeypatch.setattr(torch.cuda, "Event", CpuEvent)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: CpuStream())
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    init_none_hash(sha256)
    monkeypatch.setattr(envs, "VLLM_USE_MEGA_AOT_ARTIFACT", False)

    def create(
        *,
        ladder=None,
        async_scheduling=False,
        sparse=False,
        blocks=128,
        budget=128,
        enabled=True,
        max_num_seqs=8,
        max_requests=1,
        collect_counts=False,
    ):
        original = native_allocation_config
        with monkeypatch.context() as capability:
            if DEVICE.type == "cpu":
                capability.setattr(current_platform, "is_cuda", lambda: True)
            spec = SpeculativeConfig(
                method="dflash",
                model=original.speculative_config.model,
                target_model_config=original.model_config,
                target_parallel_config=ParallelConfig(),
                num_speculative_tokens=7,
                draft_sample_method="probabilistic",
                dflash_concurrency_drafts=ladder,
                dflash2_lookup={
                    "enabled": enabled,
                    "policy": "longest",
                    "nmin": 5,
                    "nmax": 16,
                    "search_window": 65536,
                    "max_requests": max_requests,
                },
            )
        with monkeypatch.context() as capability:
            if DEVICE.type == "cpu":
                capability.setattr(current_platform, "is_cuda", lambda: False)
                capability.setattr(
                    current_platform, "check_and_update_config", lambda config: None
                )
            config = VllmConfig(
                model_config=original.model_config,
                speculative_config=spec,
                parallel_config=ParallelConfig(),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=budget,
                    max_model_len=512,
                    is_encoder_decoder=False,
                    enable_chunked_prefill=True,
                    async_scheduling=async_scheduling,
                ),
                cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
                compilation_config=CompilationConfig(cudagraph_mode="NONE"),
            )
        chain = DecodeChain(config, sparse=sparse, blocks=blocks)
        if collect_counts:
            chain.spec._native_lookup.stats = SimpleNamespace(
                counters=torch.zeros(18, dtype=torch.uint64, device=DEVICE)
            )
        return chain

    return create


class DecodeChain:
    def __init__(self, config, *, sparse, blocks):
        self.config = config
        capacity = config.scheduler_config.max_num_seqs
        config.cache_config.num_gpu_blocks = blocks
        kv = KVCacheConfig(
            num_blocks=blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
                    ),
                )
            ],
        )
        scheduler_cls = (
            AsyncScheduler if config.scheduler_config.async_scheduling else Scheduler
        )
        self.scheduler = scheduler_cls(
            vllm_config=config,
            kv_cache_config=kv,
            block_size=16,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(config),
        )
        assert self.scheduler.use_v2_model_runner
        self.runner = object.__new__(GPUModelRunner)
        runner = self.runner
        runner.device = DEVICE
        runner.max_num_reqs = capacity
        runner.num_speculative_steps = 7
        runner.use_dcp = runner.use_pp = False
        runner.prefix_anchor_lens_buffer = None
        runner.req_states = RequestState(capacity, 512, 128, 7, 128, DEVICE)
        runner.input_buffers = InputBuffers(capacity, 128, DEVICE)
        runner.pp_handler = runner.encoder_cache = runner.prompt_logprobs_worker = None
        runner.is_last_pp_rank = False
        runner.sampler = None
        runner.model_state = Mock()
        runner.block_tables = Mock()
        runner.lora_state = Mock()
        runner.draft_tokens_handler = DraftTokensHandler(DEVICE)
        self.spec = object.__new__(DFlash2Speculator)
        self.spec.speculative_config = config.speculative_config
        self.spec.num_speculative_steps = self.spec.draft_block = 7
        self.spec._running_reqs = 0
        self.spec._lookup_enabled = False
        self.spec._debug_real_proposal = True
        self.spec._native_lookup = (
            NativeDFlash2Lookup(
                config.speculative_config.resolved_dflash2_lookup, capacity, 512, DEVICE
            )
            if config.speculative_config.dflash2_lookup.enabled
            else None
        )
        self.spec.set_req_states(runner.req_states)
        self.spec.sample_idx_mapping = torch.full(
            (capacity * 7,), -1, dtype=torch.int32, device=DEVICE
        )
        self.spec.draft_tokens = torch.zeros(
            (capacity, 7), dtype=torch.int64, device=DEVICE
        )
        self.spec.draft_logits = torch.full(
            (capacity, 7, 128), -float("inf"), device=DEVICE
        )
        self.spec._cached_candidate_ids = torch.arange(16, device=DEVICE).repeat(
            capacity, 7, 1
        )
        self.spec._cached_candidate_scores = (
            torch.full((capacity, 7, 16), -float("inf"), device=DEVICE)
            if sparse
            else None
        )
        self.requests = {}
        self.previous = {}

    def add(self, name, *, prompt_length=40, max_tokens=128, grammar=False):
        prompt = [2 + index % 8 for index in range(prompt_length)]
        request = Request(
            request_id=name,
            prompt_token_ids=prompt,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                ignore_eos=True,
                structured_outputs=StructuredOutputsParams(regex=".*")
                if grammar
                else None,
            ),
            pooling_params=None,
            block_hasher=get_request_block_hasher(16, sha256),
        )
        if grammar:
            backend = Mock(spec=StructuredOutputGrammar)
            backend.accept_tokens.return_value = True
            backend.validate_tokens.side_effect = lambda tokens: tokens
            backend.is_terminated.return_value = False
            request.structured_output_request.grammar = backend
        self.requests[name] = request
        self.scheduler.add_request(request)
        return request

    def schedule(self):
        output = self.scheduler.schedule()
        runner = self.runner
        runner.finish_requests(output)
        runner.add_requests(output)
        runner.update_requests(output)
        if not output.num_scheduled_tokens:
            return output, None
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=output.total_num_scheduled_tokens,
            num_reqs=len(output.num_scheduled_tokens),
        )
        batch = runner.prepare_inputs(output, desc)
        assert batch.num_reqs == len(output.num_scheduled_tokens)
        # CPU placeholders reserve width, while the real input reads the previous
        # per-request proposal from device state in the current slot order.
        for row, name in enumerate(batch.req_ids):
            slot = runner.req_states.req_id_to_index[name]
            start, end = batch.query_start_loc_np[row : row + 2]
            computed = int(runner.req_states.num_computed_tokens.gpu[slot])
            assert torch.equal(
                batch.positions[start:end],
                torch.arange(computed, computed + end - start, device=DEVICE),
            )
            lo, hi = batch.cu_num_logits_np[row : row + 2]
            assert batch.expanded_idx_mapping[lo:hi].tolist() == [slot] * (hi - lo)
            assert batch.expanded_local_pos[lo:hi].tolist() == list(range(hi - lo))
            drafts = len(output.scheduled_spec_decode_tokens.get(name, ()))
            if drafts:
                expected = runner.req_states.draft_tokens[slot, :drafts]
                assert torch.equal(batch.input_ids[end - drafts : end], expected)
                proposal, logits, ids, scores = self.previous[name]
                assert torch.equal(expected, proposal[:drafts])
                assert torch.equal(
                    self.spec.draft_logits[slot, :drafts], logits[:drafts]
                )
                assert torch.equal(
                    self.spec._cached_candidate_ids[slot, :drafts], ids[:drafts]
                )
                if scores is not None:
                    assert torch.equal(
                        self.spec._cached_candidate_scores[slot, :drafts],
                        scores[:drafts],
                    )
                assert end - start >= drafts + 1
        return output, batch

    def execute(self, output, batch, *, accepted=0, no_sample=False, feedback=True):
        runner, spec = self.runner, self.spec
        states = runner.req_states
        tokens = torch.zeros((batch.num_reqs, 8), dtype=torch.int64, device=DEVICE)
        counts = torch.zeros(batch.num_reqs, dtype=torch.int32, device=DEVICE)
        actual_outputs = []
        expected_computed = []
        for row, name in enumerate(batch.req_ids):
            slot = states.req_id_to_index[name]
            width = len(output.scheduled_spec_decode_tokens.get(name, ()))
            take = min(width, accepted)
            sampled = states.draft_tokens[slot, :take].tolist()
            sampled.append(2 + (len(self.requests[name].all_token_ids) + take) % 8)
            incomplete = bool(batch.is_incomplete_prefilling_np[row])
            if incomplete or no_sample:
                sampled = []
            tokens[row, : len(sampled)] = torch.tensor(sampled, device=DEVICE)
            counts[row] = len(sampled)
            actual_outputs.append(sampled)
            computed = int(states.num_computed_tokens.gpu[slot])
            expected_computed.append(
                computed
                + int(batch.num_scheduled_tokens[row])
                - (width + 1 - len(sampled) if not incomplete else 0)
            )
        counts, rejected = get_num_sampled_and_rejected(
            counts,
            batch.seq_lens,
            batch.cu_num_logits,
            batch.idx_mapping,
            states.prefill_len.gpu,
        )
        runner.postprocess_sampled(
            batch.idx_mapping, tokens, counts, rejected, batch.query_start_loc
        )
        # This is the existing state/GDN callback's acceptance count, after the
        # real worker post-update kernel has rolled back rejected positions.
        call = runner.model_state.postprocess_state.call_args
        assert torch.equal(call.args[1], counts)
        assert (
            states.num_computed_tokens.gpu[batch.idx_mapping].tolist()
            == expected_computed
        )
        self.propose(batch, counts, rejected)
        model_output = ModelRunnerOutput(
            req_ids=batch.req_ids,
            req_id_to_index={name: i for i, name in enumerate(batch.req_ids)},
            sampled_token_ids=actual_outputs,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        engine_outputs = self.feedback(output, model_output) if feedback else None
        record = SimpleNamespace(
            output=output,
            batch=batch,
            counts=counts.clone(),
            rejected=rejected.clone(),
            engine_outputs=engine_outputs,
            proposals=spec.draft_tokens[: batch.num_reqs].clone(),
            model_output=model_output,
        )
        return record

    def feedback(self, output, model_output):
        engine_outputs = self.scheduler.update_from_output(output, model_output)
        if not self.config.scheduler_config.async_scheduling:
            self.scheduler.update_draft_token_ids(self.runner.take_draft_token_ids())
        return engine_outputs

    def propose(self, batch, counts, rejected):
        spec, states = self.spec, self.runner.req_states
        n = batch.num_reqs
        candidates = []
        for row, name in enumerate(batch.req_ids):
            slot = states.req_id_to_index[name]
            length = int(states.total_len.gpu[slot])
            history = states.all_token_ids.gpu[slot, :length].tolist()
            candidate, match, valid = reference_lookup(history, 5, 16, 65536)
            candidates.append((candidate, match, valid))
            first = candidate[0] if match else 2
            spec.draft_tokens[row] = torch.tensor(
                [first, 10, 11, 12, 13, 14, 15], device=DEVICE
            )
            spec.draft_logits[slot].fill_(-float("inf"))
            weights = (
                torch.arange(1, 17, device=DEVICE)[None, :]
                + slot
                + torch.arange(7, device=DEVICE)[:, None]
            ).float()
            spec.draft_logits[slot, :, :16] = (
                weights / weights.sum(-1, keepdim=True)
            ).log()
            spec._cached_candidate_ids[slot] = torch.arange(16, device=DEVICE)
            if spec._cached_candidate_scores is not None:
                spec._cached_candidate_scores[slot] = spec.draft_logits[slot, :, :16]
        original = (
            spec.draft_tokens.clone(),
            spec.draft_logits.clone(),
            spec._cached_candidate_ids.clone(),
        )
        scores = spec._cached_candidate_scores
        original_scores = scores.clone() if scores is not None else None
        spec.sample_idx_mapping.fill_(-1)
        spec.sample_idx_mapping[: n * 7] = batch.idx_mapping.repeat_interleave(7)
        spec._prepare_proposal_runtime(batch, counts, rejected)
        spec._apply_lookup(n)
        native = spec._native_lookup
        limit = self.config.speculative_config.dflash2_lookup.max_requests
        eligible = (
            1 <= n <= limit
            and spec.next_num_draft_tokens() == 7
            and not batch.has_structured_output_reqs
            and not batch.is_prefilling_np[:n].any()
            and all(
                int(counts[row]) > 0
                and 6 <= int(states.total_len.gpu[batch.idx_mapping[row]]) < 512
                for row in range(n)
            )
        )
        if native is not None:
            assert native.eligible[:n].tolist() == [int(eligible)] * n, (
                "SCHEDULER_BATCH_ADMISSION"
            )
            if not eligible:
                assert not native.use[:n].any()
                assert not native.match[:n].any()
                assert not native.valid[:n].any()
                for value, before in zip(
                    (spec.draft_tokens, spec.draft_logits, spec._cached_candidate_ids),
                    original,
                ):
                    assert torch.equal(value, before)
                if scores is not None:
                    assert torch.equal(scores, original_scores)
            else:
                for row in range(n):
                    candidate, match, valid = candidates[row]
                    assert int(native.match[row]) == match
                    assert int(native.valid[row]) == valid
                    assert native.use[row].tolist() == [
                        int(0 < col < valid) for col in range(7)
                    ]
                    if valid:
                        assert (
                            spec.draft_tokens[row, :valid].tolist() == candidate[:valid]
                        )
                    slot = int(batch.idx_mapping[row])
                    assert torch.equal(spec.draft_logits[slot, 0], original[1][slot, 0])
                    for col in range(1, int(native.valid[row])):
                        token = int(spec.draft_tokens[row, col])
                        assert spec.draft_logits[slot, col, token] == 0
                        assert torch.isfinite(spec.draft_logits[slot, col]).sum() == 1
                        if scores is not None:
                            assert scores[slot, col, 0] == 0
                            assert spec._cached_candidate_ids[slot, col, 0] == token
        for row, name in enumerate(batch.req_ids):
            slot = states.req_id_to_index[name]
            self.previous[name] = (
                spec.draft_tokens[row].clone(),
                spec.draft_logits[slot].clone(),
                spec._cached_candidate_ids[slot].clone(),
                scores[slot].clone() if scores is not None else None,
            )
        states.draft_tokens[batch.idx_mapping] = spec.draft_tokens[:n]
        self.runner.draft_tokens_handler.set_draft_tokens(
            batch,
            spec.draft_tokens[:n],
            num_draft_tokens=spec.next_num_draft_tokens(),
        )

    def step(self, **kwargs):
        output, batch = self.schedule()
        assert batch is not None
        return self.execute(output, batch, **kwargs)


@pytest.mark.parametrize("sparse", [False, True])
def test_scheduler_switches_one_two_four_eight_and_back(chain_factory, sparse):
    chain = chain_factory(ladder={4: 3, 5: 4}, sparse=sparse)
    chain.add("a")
    chain.step()  # prefill
    single = chain.step()
    assert chain.spec._native_lookup.use[0, 1:].any()
    chain.add("b")
    mixed = chain.step()
    assert mixed.batch.num_reqs == 2 and mixed.batch.is_prefilling_np.any()
    assert len(mixed.output.scheduled_spec_decode_tokens["a"]) == 7
    row = mixed.batch.req_ids.index("a")
    end = mixed.batch.query_start_loc_np[row + 1]
    assert torch.equal(mixed.batch.input_ids[end - 7 : end], single.proposals[0])
    chain.step()
    for name in ("c", "d"):
        chain.add(name)
    chain.step()
    four = chain.step(accepted=1)
    assert four.batch.num_reqs == 4
    assert chain.runner.draft_tokens_handler.num_draft_tokens == 3
    narrow = chain.step(accepted=1)
    assert set(map(len, narrow.output.scheduled_spec_decode_tokens.values())) == {3}
    for name in ("e", "f", "g", "h"):
        chain.add(name)
    chain.step()
    chain.step()
    eight = chain.step()
    assert eight.batch.num_reqs == 8
    assert set(map(len, eight.output.scheduled_spec_decode_tokens.values())) == {4}
    chain.scheduler.finish_requests(
        [name for name in chain.requests if name != "a"], RequestStatus.FINISHED_ABORTED
    )
    restored = chain.step()
    assert restored.batch.req_ids == ["a"]
    assert len(restored.output.scheduled_spec_decode_tokens["a"]) == 4
    assert chain.runner.draft_tokens_handler.num_draft_tokens == 7
    assert chain.spec._native_lookup.use[0, 1:].any()


@pytest.mark.parametrize("width", [1, 3, 4, 7])
@pytest.mark.parametrize("accepted", [0, 1, 7])
@pytest.mark.parametrize("enabled", [False])
def test_normal_handler_input_q_and_state_counts(
    chain_factory, width, accepted, enabled
):
    chain = chain_factory(ladder={2: width}, enabled=enabled)
    chain.add("a")
    chain.add("b")
    chain.step()
    chain.step()
    record = chain.step(accepted=accepted)
    assert set(map(len, record.output.scheduled_spec_decode_tokens.values())) == {width}
    assert record.counts.tolist() == [min(width, accepted) + 1] * 2
    assert record.rejected.tolist() == [width - min(width, accepted)] * 2
    for name in record.batch.req_ids:
        req = chain.requests[name]
        slot = chain.runner.req_states.req_id_to_index[name]
        assert (
            chain.runner.req_states.num_computed_tokens.gpu[slot]
            == req.num_computed_tokens
        )
        assert chain.runner.req_states.total_len.gpu[slot] == len(req.all_token_ids)


@pytest.mark.parametrize("inflight", [False, True])
def test_cancel_discards_old_proposal_and_reuses_slot(chain_factory, inflight):
    chain = chain_factory()
    old = chain.add("old")
    chain.step()
    chain.step()
    states = chain.runner.req_states
    slot = states.req_id_to_index["old"]
    stale = chain.runner.take_draft_token_ids()
    pending = chain.schedule() if inflight else None
    chain.scheduler.finish_requests("old", RequestStatus.FINISHED_ABORTED)
    if pending is not None:
        result = chain.execute(*pending)
        assert all(not group.outputs for group in result.engine_outputs.values())
    chain.scheduler.update_draft_token_ids(stale)
    assert "old" not in chain.scheduler.requests
    assert old.is_finished()
    new = chain.add("new", prompt_length=24)
    output, batch = chain.schedule()
    assert "old" in output.finished_req_ids
    assert states.req_id_to_index["new"] == slot
    assert states.total_len.gpu[slot] == 24
    assert not states.draft_tokens[slot].any()
    assert batch.input_ids[:24].tolist() == list(new.prompt_token_ids)
    chain.execute(output, batch)
    chain.step()
    assert chain.spec._native_lookup.eligible[0] == 1
    assert chain.spec._native_lookup.use[0, 1:].any()


def test_real_kv_preemption_resumes_from_committed_history(chain_factory):
    chain = chain_factory(blocks=9)
    chain.add("a", prompt_length=56)
    victim = chain.add("b", prompt_length=56)
    chain.step()
    committed = list(victim.all_token_ids)
    record = chain.step()
    assert "b" in record.output.preempted_req_ids
    assert victim.status == RequestStatus.PREEMPTED
    assert victim.num_preemptions == 1
    assert victim.spec_token_ids == []
    assert "b" not in chain.runner.req_states.req_id_to_index
    chain.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    output, batch = chain.schedule()
    assert batch.req_ids == ["b"]
    assert batch.is_prefilling_np.tolist() == [True]
    assert not output.scheduled_spec_decode_tokens
    assert output.scheduled_new_reqs[0].prefill_token_ids == committed
    chain.execute(output, batch)
    states = chain.runner.req_states
    slot = states.req_id_to_index["b"]
    assert states.all_token_ids.gpu[slot, : len(committed)].tolist() == committed
    chain.step()
    assert chain.spec._native_lookup.eligible[0] == 1


def test_chunked_prefill_and_grammar_are_real_scheduler_metadata(chain_factory):
    chain = chain_factory(budget=16)
    chain.add("chunked", prompt_length=40)
    first = chain.step()
    second = chain.step()
    assert first.batch.is_incomplete_prefilling_np.tolist() == [True]
    assert second.batch.is_incomplete_prefilling_np.tolist() == [True]
    assert first.counts.tolist() == second.counts.tolist() == [0]
    chain.step()
    chain.step()
    assert chain.spec._native_lookup.use[0, 1:].any()
    chain.scheduler.finish_requests("chunked", RequestStatus.FINISHED_ABORTED)
    grammar = chain.add("grammar", grammar=True)
    for _ in range(4):
        record = chain.step()
    assert record.output.has_structured_output_requests
    assert record.batch.has_structured_output_reqs
    assert chain.spec._native_lookup.eligible[0] == 0
    assert grammar.structured_output_request.grammar.accept_tokens.called
    assert chain.runner.draft_tokens_handler.draft_tokens_np is not None


@pytest.mark.parametrize("reason", ["no_sample", "no_space"])
def test_device_admission_uses_current_sample_and_space(chain_factory, reason):
    chain = chain_factory()
    chain.add("a")
    chain.step()
    chain.step()
    record = chain.step()
    batch = record.batch
    states = chain.runner.req_states
    slot = states.req_id_to_index["a"]
    committed = int(states.total_len.gpu[slot])
    if reason == "no_space":
        states.total_len.gpu[slot] = 512
    chain.propose(
        batch,
        torch.tensor([int(reason != "no_sample")], dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
    )
    states.total_len.gpu[slot] = committed
    assert not chain.spec._native_lookup.eligible.any()
    chain.step()
    assert chain.spec._native_lookup.eligible[0] == 1


@pytest.mark.parametrize("sparse", [False, True])
def test_async_delayed_feedback_and_cancel(chain_factory, sparse):
    chain = chain_factory(async_scheduling=True, sparse=sparse)
    assert isinstance(chain.scheduler, AsyncScheduler)
    request = chain.add("a")
    chain.step()
    chain.step()
    output, batch = chain.schedule()
    pending = chain.execute(output, batch, feedback=False)
    assert request.num_output_placeholders == 8
    next_output, next_batch = chain.schedule()
    newer = chain.execute(next_output, next_batch, feedback=False)
    assert request.num_output_placeholders == 16
    chain.feedback(pending.output, pending.model_output)
    assert request.num_output_placeholders == 8
    chain.feedback(newer.output, newer.model_output)
    assert request.num_output_placeholders == 0
    slot = chain.runner.req_states.req_id_to_index["a"]
    assert chain.runner.req_states.total_len.gpu[slot] == len(request.all_token_ids)
    assert (
        chain.runner.req_states.num_computed_tokens.gpu[slot]
        == request.num_computed_tokens
    )
    output, batch = chain.schedule()
    stale = chain.execute(output, batch, feedback=False)
    chain.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    chain.feedback(stale.output, stale.model_output)
    assert "a" not in chain.scheduler.requests
    chain.add("b")
    chain.step()
    chain.step()
    assert chain.spec._native_lookup.use[0, 1:].any()


def verify_rejection_round(chain, output, batch, accepted):
    from tests.v1.spec_decode.test_w21_native_distribution import interpreter_math
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        dflash2_sparse_topk_rejection_sample,
        rejection_sample,
    )

    widths = [
        len(output.scheduled_spec_decode_tokens.get(name, ())) for name in batch.req_ids
    ]
    takes = [min(width, accepted) for width in widths]
    target = torch.full(
        (int(batch.cu_num_logits_np[-1]), 128), -float("inf"), device=DEVICE
    )
    for row, name in enumerate(batch.req_ids):
        slot = chain.runner.req_states.req_id_to_index[name]
        start, end = batch.cu_num_logits_np[row : row + 2]
        take = takes[row]
        target[start : start + take] = chain.spec.draft_logits[slot, :take]
        target[start + take : end, 63] = 0
    draft_sampled = batch.input_ids[batch.logits_indices]
    pos = batch.positions[batch.logits_indices]
    temperature = torch.ones(8, device=DEVICE)
    seeds = torch.arange(8, dtype=torch.int64, device=DEVICE) + 37
    with interpreter_math(DEVICE):
        if chain.spec._cached_candidate_scores is not None:
            scores, ids = torch.topk(target, 16, dim=-1)
            sampled, counts = dflash2_sparse_topk_rejection_sample(
                ids,
                scores,
                chain.spec._cached_candidate_ids,
                chain.spec._cached_candidate_scores,
                draft_sampled,
                batch.cu_num_logits,
                pos,
                batch.idx_mapping,
                temperature,
                torch.ones_like(temperature),
                seeds,
                7,
            )
        else:
            sampled, counts = rejection_sample(
                target,
                chain.spec.draft_logits,
                draft_sampled,
                batch.cu_num_logits,
                pos,
                batch.idx_mapping,
                batch.expanded_idx_mapping,
                batch.expanded_local_pos,
                temperature,
                seeds,
                7,
            )
    assert counts.tolist() == [take + 1 for take in takes]
    for row, name in enumerate(batch.req_ids):
        take = takes[row]
        if take:
            proposal = chain.previous[name][0]
            assert torch.equal(sampled[row, :take], proposal[:take])
        assert sampled[row, take] == 63
    counts, rejected = get_num_sampled_and_rejected(
        counts,
        batch.seq_lens,
        batch.cu_num_logits,
        batch.idx_mapping,
        chain.runner.req_states.prefill_len.gpu,
    )
    assert rejected.tolist() == [width - take for width, take in zip(widths, takes)]
    chain.runner.postprocess_sampled(
        batch.idx_mapping, sampled, counts, rejected, batch.query_start_loc
    )
    assert torch.equal(
        chain.runner.model_state.postprocess_state.call_args.args[1], counts
    )
    chain.propose(batch, counts, rejected)
    model_output = ModelRunnerOutput(
        req_ids=batch.req_ids,
        req_id_to_index={name: row for row, name in enumerate(batch.req_ids)},
        sampled_token_ids=[
            sampled[row, : take + 1].tolist() for row, take in enumerate(takes)
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_outputs = chain.feedback(output, model_output)
    stats = engine_outputs[0].scheduler_stats.spec_decoding_stats
    assert stats.num_draft_tokens == sum(widths)
    assert stats.num_accepted_tokens == sum(takes)
    for row, name in enumerate(batch.req_ids):
        state = chain.runner.req_states
        slot = state.req_id_to_index[name]
        request = chain.requests[name]
        assert state.total_len.gpu[slot] == len(request.all_token_ids)
        assert state.num_computed_tokens.gpu[slot] == request.num_computed_tokens


@pytest.mark.parametrize("width", [1, 3, 4, 7])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("accepted", [0, 1, 7])
def test_scheduled_width_reaches_real_rejection_sampler(
    chain_factory, width, sparse, accepted
):
    chain = chain_factory(ladder={2: width}, sparse=sparse)
    chain.add("a")
    chain.add("b")
    chain.step()
    chain.step()
    output, batch = chain.schedule()
    assert set(map(len, output.scheduled_spec_decode_tokens.values())) == {width}
    verify_rejection_round(chain, output, batch, accepted)


@pytest.mark.parametrize("sparse", [False, True])
def test_old_fused_proposal_is_verified_in_new_mixed_batch(chain_factory, sparse):
    chain = chain_factory(ladder={4: 3, 5: 4}, sparse=sparse)
    chain.add("a")
    chain.step()
    chain.step()
    assert chain.spec._native_lookup.use[0, 1:].any()
    chain.add("b")
    output, batch = chain.schedule()
    assert batch.num_reqs == 2 and batch.is_prefilling_np.any()
    assert len(output.scheduled_spec_decode_tokens["a"]) == 7
    assert "b" not in output.scheduled_spec_decode_tokens
    verify_rejection_round(chain, output, batch, 7)
    assert not chain.spec._native_lookup.eligible.any()


@pytest.mark.skipif(DEVICE.type != "cuda", reason="Coordinator async capture/replay")
def test_async_native_replay_tracks_scheduler_composition(chain_factory, monkeypatch):
    chain = chain_factory(async_scheduling=True, sparse=True)
    chain.add("a")
    chain.step()
    native = chain.spec._native_lookup
    native.reset_for_capture()
    for _ in range(3):
        chain.spec._apply_lookup(8)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain.spec._apply_lookup(8)
    monkeypatch.setattr(native, "apply", lambda *args: graph.replay())
    chain.step()
    assert native.eligible.tolist() == [1] + [0] * 7
    chain.add("b")
    chain.step()
    chain.step()
    assert not native.eligible.any() and not native.use.any()
    chain.scheduler.finish_requests("b", RequestStatus.FINISHED_ABORTED)
    chain.step()
    assert native.eligible.tolist() == [1] + [0] * 7
    assert native.use[0, 1:].any()


def test_native_admission_has_no_host_readback():
    import ast
    import inspect

    source = ast.parse(inspect.getsource(NativeDFlash2Lookup))
    forbidden = {"item", "cpu", "numpy", "tolist", "synchronize"}
    calls = {
        node.func.attr
        for node in ast.walk(source)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & forbidden


@pytest.mark.parametrize("budget", [2, 4, 5, 8])
def test_scheduler_truncates_previous_fused_proposal_to_token_budget(
    chain_factory, budget
):
    chain = chain_factory(budget=budget, max_num_seqs=2, sparse=True)
    chain.add("a")
    for _ in range((40 + budget - 1) // budget):
        chain.step()
    chain.step()
    assert chain.runner.draft_tokens_handler.num_draft_tokens == 7
    assert chain.spec._native_lookup.use[0, 1:].any()
    output, batch = chain.schedule()
    assert len(output.scheduled_spec_decode_tokens["a"]) == budget - 1
    assert batch.num_draft_tokens == budget - 1
    verify_rejection_round(chain, output, batch, 7)


def test_waiting_request_does_not_disable_single_scheduled_decode(chain_factory):
    chain = chain_factory(budget=8)
    chain.add("a")
    for _ in range(5):
        chain.step()
    waiting = chain.add("waiting")
    record = chain.step()
    assert chain.scheduler.get_num_unfinished_requests() == 2
    assert waiting.status == RequestStatus.WAITING
    assert record.batch.req_ids == ["a"]
    assert chain.spec._native_lookup.eligible[0] == 1
    assert chain.spec._native_lookup.use[0, 1:].any()


@pytest.mark.parametrize("sparse", [False, True])
def test_multi_request_scheduler_switches_and_counts(chain_factory, sparse):
    chain = chain_factory(
        ladder={4: 3, 5: 4}, sparse=sparse, max_requests=3, collect_counts=True
    )
    native = chain.spec._native_lookup

    def step(expected, **kwargs):
        before = int(native.stats.counters[0])
        record = chain.step(**kwargs)
        assert (
            native.eligible[: record.batch.num_reqs].tolist()
            == [int(expected > 0)] * record.batch.num_reqs
        )
        assert int(native.stats.counters[0]) - before == expected, (
            "SCHEDULER_ROW_COUNTS"
        )
        return record

    chain.add("a")
    step(0)
    step(1)
    chain.add("b")
    step(0)  # mixed prefill invalidates the whole batch immediately
    two = step(2, accepted=1)
    assert two.batch.num_reqs == 2
    chain.add("c")
    step(0)
    three = step(3, accepted=1)
    assert three.batch.num_reqs == 3
    chain.add("d")
    step(0)
    four = step(0, accepted=1)
    assert four.batch.num_reqs == 4
    assert chain.runner.draft_tokens_handler.num_draft_tokens == 3
    step(0)  # consume the previous seven-token rows truncated by the handler
    chain.scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)
    restored = step(3)
    assert set(map(len, restored.output.scheduled_spec_decode_tokens.values())) == {3}
    assert chain.runner.draft_tokens_handler.num_draft_tokens == 7
    chain.scheduler.finish_requests("b", RequestStatus.FINISHED_ABORTED)
    step(2)
    chain.add("replacement", prompt_length=24)
    step(0)
    step(3)
    chain.scheduler.finish_requests(
        ["c", "replacement"], RequestStatus.FINISHED_ABORTED
    )
    step(1)


def test_multi_request_preemption_restores_qualification(chain_factory):
    chain = chain_factory(blocks=9, max_requests=3, collect_counts=True)
    chain.add("a", prompt_length=56)
    victim = chain.add("b", prompt_length=56)
    chain.step()
    record = chain.step()
    assert "b" in record.output.preempted_req_ids
    assert victim.status == RequestStatus.PREEMPTED
    chain.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    prefill = chain.step()
    assert prefill.batch.is_prefilling_np.tolist() == [True]
    before = int(chain.spec._native_lookup.stats.counters[0])
    chain.step()
    assert int(chain.spec._native_lookup.stats.counters[0]) == before + 1
