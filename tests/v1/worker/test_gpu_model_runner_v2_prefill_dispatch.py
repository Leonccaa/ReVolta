# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    _is_compatible,
    get_uniform_decode_token_count,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize("query_len", [1, 3, 4, 5, 7, 8, 16])
@pytest.mark.parametrize("phase", ["fresh", "tail", "mixed", "decode", "cached"])
def test_prefill_cannot_replay_decode_graph(query_len, phase):
    # Keep an unscheduled prefill at index 0 and schedule in reverse state order.
    # This also checks that request IDs, not batch positions, select state rows.
    prompt = 800 + query_len
    computed = [0, prompt, prompt]
    if phase == "fresh":
        computed[1:] = [0, 0]
    elif phase in ("tail", "mixed"):
        computed[1] = 800
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"unused": 0, "a": 1, "b": 2},
        num_computed_prefill_tokens=np.array(computed),
        prefill_len=SimpleNamespace(np=np.array([prompt] * 3)),
    )
    req_ids = ["b", "a"] if phase in ("mixed", "decode", "cached") else ["a"]
    schedule = SimpleNamespace(
        num_scheduled_tokens=dict.fromkeys(req_ids, query_len),
        total_num_scheduled_tokens=len(req_ids) * query_len,
    )
    uniform = runner._get_uniform_decode_token_count(schedule, False)
    expected = query_len if phase in ("decode", "cached") else None
    assert uniform == expected
    full = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=schedule.total_num_scheduled_tokens,
        num_reqs=len(req_ids),
        uniform_token_count=query_len,
    )
    assert _is_compatible(
        full, len(req_ids), schedule.total_num_scheduled_tokens, uniform
    ) == (expected is not None)


def test_dummy_and_nonuniform_batches_do_not_read_live_state():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    schedule = SimpleNamespace(
        num_scheduled_tokens={"dummy": 4}, total_num_scheduled_tokens=4
    )
    assert runner._get_uniform_decode_token_count(schedule, True) == 4
    schedule.num_scheduled_tokens = {"a": 4, "b": 3}
    schedule.total_num_scheduled_tokens = 7
    assert runner._get_uniform_decode_token_count(schedule, False) is None


@pytest.mark.parametrize("has_prefill", [False, True])
def test_draft_phase_aware_predicate(has_prefill):
    assert get_uniform_decode_token_count(2, 8, 4, has_prefill) == (
        None if has_prefill else 4
    )
    assert get_uniform_decode_token_count(2, 7, 4, has_prefill) is None
    assert get_uniform_decode_token_count(0, 0, 0, has_prefill) is None


@pytest.fixture
def runner():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.kv_connector = SimpleNamespace(
        handle_preemptions=lambda _: None, pre_forward=lambda _: None
    )
    runner.update_pp_decode_requests = lambda: None
    for name in ("finish_requests", "free_states", "add_requests", "update_requests"):
        setattr(runner, name, lambda _: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.is_encoder_decoder = False
    runner.cudagraph_manager = object()
    runner.dp_size, runner.dp_rank = 1, 0
    return runner


def test_execute_rejects_prefill_before_dp_graph_dispatch(monkeypatch, runner):
    runner.req_states = SimpleNamespace(
        req_id_to_index={"tail": 0},
        num_computed_prefill_tokens=np.array([800]),
        prefill_len=SimpleNamespace(np=np.array([804])),
    )
    schedule = SimpleNamespace(
        num_scheduled_tokens={"tail": 4},
        total_num_scheduled_tokens=4,
        new_block_ids_to_zero=[],
        skip_drafts=True,
    )

    class DispatchReached(Exception):
        pass

    def dispatch(manager, num_reqs, num_tokens, uniform_token_count, *args, **kwargs):
        assert (num_reqs, num_tokens) == (1, 4)
        assert uniform_token_count is None
        assert kwargs["allow_skip_drafts"]
        raise DispatchReached

    monkeypatch.setattr(mrv2, "dispatch_cg_and_sync_dp", dispatch)
    with pytest.raises(DispatchReached):
        runner.execute_model(schedule)


def test_piecewise_forward_publishes_live_qsa_prefill_phase(monkeypatch, runner):
    # Equal-shaped PIECEWISE batches can have different live phases.
    req_ids = [f"r{i}" for i in range(32)]
    computed = np.full(32, 800)
    prefill_len = np.full(32, 800)
    runner.req_states = SimpleNamespace(
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        num_computed_prefill_tokens=computed,
        prefill_len=SimpleNamespace(np=prefill_len),
        num_computed_tokens=SimpleNamespace(gpu=None),
    )
    schedule = SimpleNamespace(
        num_scheduled_tokens=dict.fromkeys(req_ids, 8),
        total_num_scheduled_tokens=256,
        new_block_ids_to_zero=[],
        finished_req_ids=set(),
        num_spec_tokens_to_schedule=0,
        skip_drafts=False,
    )
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=256, num_reqs=None
    )
    monkeypatch.setattr(mrv2, "dispatch_cg_and_sync_dp", lambda *a, **kw: (desc, None))
    runner.mixed_prefill_timer = SimpleNamespace(mark_cuda_graph=lambda: None)
    runner.prepare_inputs = lambda *_: SimpleNamespace(
        is_prefilling_np=computed < prefill_len,
        num_tokens_after_padding=256,
        num_draft_tokens=7 * int((computed == prefill_len).sum()),
        input_ids=torch.zeros(256, dtype=torch.int32),
        positions=torch.arange(256),
    )
    runner.prepare_attn = lambda _: ((), torch.empty(0))
    runner.model_state = SimpleNamespace(
        preprocess_state=lambda *_: None,
        prepare_attn=lambda *_: {},
        prepare_inputs=lambda *_: {},
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner.attn_groups = []
    runner._ple_offload_connector = None
    runner.speculative_config = SimpleNamespace(method="dflash")
    runner.lora_config = None
    runner._hc_sp_enabled = False
    runner.supports_mm_inputs = runner.use_aux_hidden_state_outputs = False
    runner.is_first_pp_rank = runner.is_last_pp_rank = True
    runner.vllm_config = SimpleNamespace(
        compilation_config=CompilationConfig(),
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )

    observed = []

    def forward(**inputs):
        context = get_forward_context()
        assert context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        observed.append(context.additional_kwargs["qsa_has_prefill"])
        return inputs["input_ids"]

    runner.model = forward
    computed[0] -= 8
    runner.execute_model(schedule)
    computed[0] += 8
    runner.execute_model(schedule)
    assert observed == [True, False]
