# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.models.hy_v4_flash.sm70.kernels import paged_attention, paged_attention_torch
from vllm.models.hy_v4_flash.sm70.splitk import (
    decode_workspace_specs,
    paged_attention_splitk,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cpu",
            marks=pytest.mark.skipif(
                not HAS_TRITON or os.environ.get("TRITON_INTERPRET") != "1",
                reason="Triton CPU interpreter required",
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda() or not torch.accelerator.is_available(),
                reason="CUDA required",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "heads,width,lengths",
    [(8, 2048, [1, 511, 512, 2047, 2048, 0]), (16, 512, [1, 511, 512, 0])],
)
def test_splitk_matches_fp32_reference(device, heads, width, lengths):
    rng = torch.Generator().manual_seed(23)
    rows = len(lengths)
    q = torch.randn(rows, heads, 576, generator=rng).half()
    storage = torch.randn(width // 64, 65, 576, generator=rng).half()
    counts = torch.tensor(lengths, dtype=torch.int32)
    pages = torch.randperm(width // 64, generator=rng)
    slots = (pages[:, None] * 64 + torch.arange(64)).flatten()
    # SWA uses a width-512 view of the shared index buffer.
    indices = torch.full((rows, 2048), -1, dtype=torch.int32)
    indices[:, :width] = slots.masked_fill(torch.arange(width) >= counts[:, None], -1)
    sinks = torch.linspace(-2, 5, heads)
    scale = 1 / 16
    expected = paged_attention_torch(
        q.float(), storage[:, :64], indices[:, :width], counts, sinks, scale, 512
    )
    args = (
        q.to(device),
        storage.to(device)[:, :64],
        indices.to(device)[:, :width],
        counts.to(device),
        sinks.to(device),
        scale,
    )
    workspace = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in decode_workspace_specs(rows, heads, width, 512)
    ]
    old = paged_attention(*args, 512).cpu().float()
    split = paged_attention_splitk(*args, *workspace).cpu().float()
    assert torch.equal(split[-1], torch.zeros_like(split[-1]))
    norms = expected[:-1].flatten(1).norm(dim=-1)
    old_error = (old[:-1] - expected[:-1]).flatten(1).norm(dim=-1) / norms
    split_error = (split[:-1] - expected[:-1]).flatten(1).norm(dim=-1) / norms
    print(
        f"{device=} {heads=} {lengths=} {old_error.tolist()=} {split_error.tolist()=}"
    )
    assert torch.all(split_error <= old_error)
