# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU interpreter: mixed GDN4 / PLE0 state-copy addressing."""

import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.mamba_align import (
    run_mamba_align_postprocess,
    run_mamba_align_precopy,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if request.param == "cpu" and not interpret:
        pytest.skip("CPU requires TRITON_INTERPRET=1")
    if request.param == "cuda" and (interpret or not current_platform.is_cuda()):
        pytest.skip("CUDA requires a GPU and compiled Triton execution")
    with torch.device(request.param):
        yield


@pytest.mark.parametrize("bias", [0, 1, 2, 3, 4])
def test_mixed_group_copy_uses_no_ple_spec_blocks(bias, device):
    gconv = torch.arange(8 * 7 * 4, dtype=torch.float32).reshape(8, 7, 4)
    temporal = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    ple = torch.arange(8 * 13 * 4, dtype=torch.float32).reshape(8, 13, 4)
    before = [x.clone() for x in (gconv, temporal, ple)]
    # Same padded physical stride, but PLE owns only blocks 1 and 2.
    gt = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]], dtype=torch.int32)
    pt = torch.tensor([[1, 2, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
    states = [gconv, temporal, ple]
    ctx = NS(
        is_initialized=True,
        num_states=3,
        block_size=16,
        block_table_ptrs=torch.tensor(
            [gt.data_ptr(), pt.data_ptr()], dtype=torch.int64
        ),
        block_table_stride_req=8,
        state_base_addrs=torch.tensor(
            [x.data_ptr() for x in states], dtype=torch.int64
        ),
        state_block_strides=torch.tensor(
            [x.stride(0) * 4 for x in states], dtype=torch.int64
        ),
        state_elem_sizes=torch.tensor([4, 4, 4], dtype=torch.int32),
        state_inner_sizes=torch.tensor([4, 16, 4], dtype=torch.int64),
        state_conv_widths=torch.tensor([7, 0, 13], dtype=torch.int32),
        state_group_indices=torch.tensor([0, 0, 1], dtype=torch.int32),
        num_accepted_tokens_out=torch.empty(1, dtype=torch.int32),
    )
    run_mamba_align_precopy(
        ctx,
        1,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([bias], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        gconv[2, : 7 - bias], before[0][1, bias:], rtol=0, atol=0
    )
    torch.testing.assert_close(temporal[2], before[1][1 + bias], rtol=0, atol=0)
    torch.testing.assert_close(ple[2, : 13 - bias], before[2][1, bias:], rtol=0, atol=0)
    # Boundary-state postprocess chooses accepted temporal block but the same
    # PLE running block, shifted within its 13 columns.
    for state, original in zip(states, before):
        state.copy_(original)
    accepted = torch.tensor([bias + 1], dtype=torch.int32)
    run_mamba_align_postprocess(
        ctx,
        1,
        accepted,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )
    torch.testing.assert_close(temporal[1], before[1][1 + bias], rtol=0, atol=0)
    torch.testing.assert_close(ple[1, : 13 - bias], before[2][1, bias:], rtol=0, atol=0)
    torch.testing.assert_close(ple[2:], before[2][2:], rtol=0, atol=0)


@pytest.mark.parametrize("accepted", [2, 1], ids=["at-boundary", "before-boundary"])
def test_r1_single_token_tail_materializes_replay_state(monkeypatch, device, accepted):
    from tests.kernels.mamba.test_mrv2_mamba_align import _make_context
    from tests.v1.core.test_long_context_admission import make_requests
    from tests.v1.core.test_mamba_sparse_retention import (
        STATE_BLOCK,
        _r1_sample,
        _r1_scheduler,
    )
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    b = STATE_BLOCK
    scheduler = _r1_scheduler(monkeypatch, 4, 4, batch_drafts=4)
    [short] = make_requests(1, b - 2, 6)
    [long] = make_requests(1, b - 5, 32, tag="long")
    reqs = [long, short]
    for request in reqs:
        scheduler.add_request(request)
    _r1_sample(scheduler, scheduler.schedule())
    assert short.num_computed_tokens == b - 2

    tables = torch.tensor([[3, 2], [4, 1]], dtype=torch.int32)
    table = torch.empty_like(tables)
    conv = torch.zeros(5, 7, 1)
    temporal = torch.zeros(5, 1)
    ctx = _make_context(conv, temporal, table, block_size=b)
    state = object.__new__(MambaHybridModelState)
    state._align_mode = state._spec_state_replay = True
    state._spec_replay_step = None
    state._mamba_ctx = ctx
    state._mamba_state_idx_gpu = torch.zeros(2, dtype=torch.int32)
    state._mamba_src_col_gpu = torch.empty_like(state._mamba_state_idx_gpu)
    state._mamba_token_bias_gpu = torch.empty_like(state._mamba_state_idx_gpu)
    state.num_accepted_tokens_gpu = torch.ones_like(state._mamba_state_idx_gpu)
    state._get_mamba_group_info = lambda _: ([0], NS(block_size=b))
    state._ensure_align_ctx = lambda *_: ctx

    def prepare(output, slots, computed):
        mapping = torch.tensor(slots, dtype=torch.int32)
        widths = [output.num_scheduled_tokens[reqs[i].request_id] for i in slots]
        qsl = torch.tensor([0, *widths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        table[: len(slots)].copy_(tables[slots])
        state.preprocess_state(
            NS(idx_mapping=mapping, num_reqs=len(slots), query_start_loc=qsl),
            (table,),
            None,
            computed,
        )
        return mapping

    def schedule():
        computed = torch.tensor(
            [r.num_computed_tokens for r in reqs], dtype=torch.int32
        )
        return scheduler.schedule(), computed

    def postprocess(mapping, count):
        computed = torch.tensor(
            [r.num_computed_tokens for r in reqs], dtype=torch.int32
        )
        state.postprocess_state(mapping, torch.full_like(mapping, count), computed)

    history = [1, 2, 3]
    conv[4, :3, 0] = torch.tensor(history)
    temporal[4, 0] = 10
    verification, computed = schedule()
    assert verification.num_scheduled_tokens[short.request_id] == 5
    assert (
        not verification.skip_drafts and verification.num_spec_tokens_to_schedule == 4
    )
    mapping = prepare(verification, [0, 1], computed)
    short_slot = 1
    assert int(state._mamba_state_idx_gpu[short_slot]) == 1
    # Substitute the verifier's convolution window and accepted GDN replay.
    conv[1, :, 0] = torch.arange(2, 9)
    temporal[1, 0] = 19 if accepted == 2 else 14
    _r1_sample(scheduler, verification, "some" if accepted == 2 else "none")
    postprocess(mapping, accepted)
    history += [4, 5][:accepted]
    reference = 10 + sum(history[3:])
    assert short.num_computed_tokens == b - 2 + accepted
    assert int(state.num_accepted_tokens_gpu[short_slot]) == accepted
    if accepted == 2:
        assert conv[4, :3, 0].tolist() == history[-3:]
        assert temporal[4, 0].item() == reference

    short.next_decode_eligible_step = scheduler.current_step + 2
    other, computed = schedule()
    assert list(other.num_scheduled_tokens) == [long.request_id]
    mapping = prepare(other, [0], computed)
    _r1_sample(scheduler, other, "none")
    postprocess(mapping, 1)
    assert int(state._mamba_state_idx_gpu[short_slot]) == 1
    assert int(state.num_accepted_tokens_gpu[short_slot]) == accepted

    for step, token in enumerate((11, 13)):
        tail, computed = schedule()
        assert tail.num_scheduled_tokens[short.request_id] == 1
        assert short.request_id not in tail.scheduled_spec_decode_tokens
        # MRv2 sorts by query width, so the short row now comes first.
        assert tail.num_scheduled_tokens[long.request_id] > 1
        mapping = prepare(tail, [1, 0], computed)
        column = int(state._mamba_state_idx_gpu[short_slot])
        assert column == (0 if accepted == 1 and step == 0 else 1)
        assert int(state.num_accepted_tokens_gpu[short_slot]) == 1
        block = table[0, column]
        window = conv[block, :3, 0].tolist()
        recurrent = temporal[block, 0].item()
        assert window == history[-3:]
        assert recurrent == reference
        # Advance and compare the state actually read by two ordinary q1 steps.
        conv[block, :3, 0] = torch.tensor(window[1:] + [token])
        temporal[block, 0] = recurrent + token
        history.append(token)
        reference += token
        _r1_sample(scheduler, tail, "none")
        postprocess(mapping, 1)
