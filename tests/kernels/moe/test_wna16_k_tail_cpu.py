# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the WNA16 partial K tile with the Triton CPU interpreter."""

import os

import pytest
import torch

from vllm.triton_utils import tl

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        os.environ.get("TRITON_INTERPRET") != "1", reason="CPU interpreter required"
    ),
]


@pytest.mark.parametrize("bits", [4, 8])
def test_wna16_partial_k_tile(monkeypatch, bits):
    from triton.runtime.interpreter import InterpreterBuilder

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel_gptq_awq

    m, n, k, block_m, block_k = 2, 16, 96, 16, 64
    a = (torch.arange(m * k).reshape(m, k) % 7 - 3).half()
    q = (torch.arange(n * k).reshape(n, k) % (2**bits)).byte()
    packed = q[:, ::2] | (q[:, 1::2] << 4) if bits == 4 else q
    weight = packed.unsqueeze(0).contiguous()
    scale = torch.full((1, n, k // 32), 1 / 16, dtype=torch.float16)
    output = torch.empty(m, n, dtype=torch.float16)
    sorted_ids = torch.tensor([0, 1] + [m] * (block_m - m), dtype=torch.int32)
    expert_ids = torch.tensor([0], dtype=torch.int32)
    post_padded = torch.tensor([block_m], dtype=torch.int32)
    routing = torch.ones(m, dtype=torch.float32)
    real_load = InterpreterBuilder.create_masked_load
    weight_loads = []

    def check_weight_load(self, ptrs, mask, other, *args):
        if ptrs.get_element_ty() == tl.uint8:
            active = ptrs.data[mask.data]
            weight_loads.append(len(active))
            assert (active >= weight.data_ptr()).all()
            assert (active < weight.data_ptr() + weight.numel()).all()
        return real_load(self, ptrs, mask, other, *args)

    monkeypatch.setattr(InterpreterBuilder, "create_masked_load", check_weight_load)
    fused_moe_kernel_gptq_awq[(1,)](
        a,
        weight,
        output,
        scale,
        None,
        routing,
        sorted_ids,
        expert_ids,
        post_padded,
        n,
        k,
        block_m,
        m,
        a.stride(0),
        a.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        scale.stride(0),
        scale.stride(2),
        scale.stride(1),
        0,
        0,
        0,
        block_k_diviable=False,
        group_size=32,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=1,
        SPLIT_K=1,
        MUL_ROUTED_WEIGHT=False,
        top_k=1,
        compute_type=tl.float16,
        has_zp=False,
        use_int4_w4a16=bits == 4,
        use_int8_w8a16=bits == 8,
    )
    assert weight_loads == [n * block_k, n * (k - block_k)]
    reference_weight = (q.float() - 2 ** (bits - 1)) / 16
    torch.testing.assert_close(
        output.float(), a.float() @ reference_weight.T, rtol=0, atol=0
    )
