# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


@pytest.mark.parametrize("intermediate", [160, 320, 640])
@pytest.mark.parametrize("tokens", [5, 10, 20, 40])
def test_grouped_nvfp4_ops_match_across_tp_shards(intermediate, tokens):
    torch.manual_seed(intermediate * 100 + tokens)
    dev = "cuda"
    experts, hidden, top_k = 512, 2560, 10
    slots = tokens * top_k
    x = (
        torch.randn(tokens, hidden, device=dev, dtype=torch.float16) * 0.25
    ).contiguous()
    ids = torch.stack(
        [torch.randperm(experts, device=dev)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    weights = torch.rand(tokens, top_k, device=dev, dtype=torch.float32).softmax(-1)
    w13 = torch.randint(
        -(2**31),
        2**31 - 1,
        (experts, hidden, (2 * intermediate) // 8),
        device=dev,
        dtype=torch.int32,
    )
    s13 = (
        torch.rand(
            experts, hidden // 16, 2 * intermediate, device=dev, dtype=torch.float16
        )
        * 0.05
    )
    w2 = torch.randint(
        -(2**31),
        2**31 - 1,
        (experts, intermediate, hidden // 8),
        device=dev,
        dtype=torch.int32,
    )
    s2 = (
        torch.rand(experts, intermediate // 16, hidden, device=dev, dtype=torch.float16)
        * 0.05
    )

    rows = torch.empty(400, 8, device=dev, dtype=torch.int32)
    grp_experts = torch.empty(400, device=dev, dtype=torch.int32)
    grp_sizes = torch.empty(400, device=dev, dtype=torch.int32)
    grp_total = torch.empty(1, device=dev, dtype=torch.int32)

    mid = torch.empty(slots, intermediate, device=dev, dtype=torch.float16)
    split_k = 4 if intermediate == 160 or tokens <= 8 else 8
    ops.nvfp4_grouped_w13_sm70_v154_out(
        mid,
        x,
        w13,
        s13,
        ids.view(-1),
        rows,
        grp_experts,
        grp_sizes,
        grp_total,
        split_k,
        False,
    )
    routed = torch.empty(slots, hidden, device=dev, dtype=torch.float16)
    out = torch.empty(tokens, hidden, device=dev, dtype=torch.float16)
    ops.nvfp4_grouped_w2_sm70_v154_out(
        out, routed, mid, w2, s2, weights, rows, grp_experts, grp_sizes, grp_total
    )
    ref_fp32 = torch.zeros_like(out, dtype=torch.float32)
    routed_rows = routed.view(tokens, top_k, hidden)
    for slot in range(top_k):
        ref_fp32.add_(routed_rows[:, slot].float() * weights[:, slot, None])
    ref = ref_fp32.half()
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    if intermediate == 160:
        out_br = torch.empty_like(out)
        ops.nvfp4_grouped_w2_batch_reduce_sm70_v154_out(
            out_br,
            routed,
            mid,
            w2,
            s2,
            weights,
            rows,
            grp_experts,
            grp_sizes,
            grp_total,
        )
        torch.testing.assert_close(out_br, ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("tokens", [5, 40])
@pytest.mark.parametrize("interleaved", [False, True])
def test_grouped_nvfp4_raw_scale_matches_prepared(tokens, interleaved):
    torch.manual_seed(1544 + tokens)
    dev = "cuda"
    experts, hidden, inter, top_k = 512, 2560, 160, 10
    slots = tokens * top_k
    x = torch.randn(tokens, hidden, device=dev, dtype=torch.float16) * 0.25
    w13 = torch.randint(
        -(2**31),
        2**31 - 1,
        (experts, hidden, inter // 4),
        device=dev,
        dtype=torch.int32,
    )
    codes = torch.randint(
        0,
        89,
        (experts, hidden // 16, 2 * inter),
        device=dev,
        dtype=torch.uint8,
    )
    globals_ = torch.tensor([2**-10, 2**-9], device=dev).repeat(experts, 1)
    columns = (
        globals_.repeat(1, inter)
        if interleaved
        else globals_.repeat_interleave(inter, 1)
    )
    scales = (codes.view(torch.float8_e4m3fn).float() * columns[:, None, :]).half()
    ids = torch.arange(slots, device=dev, dtype=torch.int32) % 11
    rows = torch.empty(slots, 8, device=dev, dtype=torch.int32)
    groups = torch.empty(slots, device=dev, dtype=torch.int32)
    sizes = torch.empty_like(groups)
    total = torch.empty(1, device=dev, dtype=torch.int32)
    expected = torch.empty(slots, inter, device=dev, dtype=torch.float16)
    actual = torch.empty_like(expected)
    ops.nvfp4_grouped_w13_sm70_v154_out(
        expected, x, w13, scales, ids, rows, groups, sizes, total, 4, interleaved
    )
    ops.nvfp4_grouped_w13_raw_sm70_v154_out(
        actual, x, w13, codes, globals_, ids, rows, groups, sizes, total, 4, interleaved
    )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
