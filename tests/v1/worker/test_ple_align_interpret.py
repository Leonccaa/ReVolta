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
