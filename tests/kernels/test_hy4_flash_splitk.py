# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.models.hy_v4_flash.sm70.kernels import (
    cache_fp8_ds_mla,
    pack_fp8_ds_mla_torch,
    paged_attention,
    paged_attention_torch,
)
from vllm.models.hy_v4_flash.sm70.splitk import (
    decode_workspace_specs,
    paged_attention_splitk,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.fixture(
    params=[
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
    ]
)
def device(request):
    return request.param


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


@pytest.mark.parametrize(
    "heads,width,lengths",
    [(8, 2048, [2047, 2048, 0]), (16, 512, [511, 512, 0])],
)
def test_fp8_main_kv_matches_reference(device, heads, width, lengths):
    rng = torch.Generator().manual_seed(23)
    rows = len(lengths)
    page = 64
    q = torch.randn(rows, heads, 576, generator=rng).half()
    kv_c = torch.randn(width, 4, 128, generator=rng)
    kv_c = (kv_c * torch.tensor([0.125, 1, 4, 2])[None, :, None]).flatten(1).half()
    kv_c[0, :128] = 0
    # A power-of-two scale makes these E4M3FN midpoints exact after division.
    # Cover the largest finite code, signed subnormals and ties on both sides.
    edge_values = (
        torch.tensor(
            [448, -448, 2**-10, -(3 * 2**-10), 15 * 2**-10, 1.0625, -1.1875],
            dtype=torch.float16,
        )
        / 128
    )
    kv_c[0, 128:256] = 0
    kv_c[0, 128 : 128 + edge_values.numel()] = edge_values
    k_pe = torch.randn(width, 64, generator=rng).half()
    storage = torch.zeros(
        width // page + 1, page + 1, 656, dtype=torch.uint8, device=device
    )
    cache = storage[1:, :page]
    if device == "cuda":
        cache_fp8_ds_mla(
            torch.cat((kv_c[-1:], kv_c)).to(device),
            torch.cat((k_pe[-1:], k_pe)).to(device),
            cache,
            torch.arange(-1, width, device=device),
        )
    else:
        # The shared E4M3 encoder rounds with PTX, which the interpreter
        # cannot run; the CUDA case checks the writer's bytes.
        cache.copy_(pack_fp8_ds_mla_torch(kv_c, k_pe).view_as(cache))
    assert storage[0].eq(0).all()

    packed = cache.cpu().reshape(width, 656)
    assert torch.equal(packed, pack_fp8_ds_mla_torch(kv_c, k_pe))
    scales = packed[:, 512:528].view(torch.float32)
    groups = kv_c.float().reshape(width, 4, 128)
    expected_scales = (groups.abs().amax(-1) / 448).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    torch.testing.assert_close(scales, expected_scales, rtol=2e-7, atol=0)
    assert torch.equal(packed[:, 528:], k_pe.view(torch.uint8))
    codes = packed[:, :512].view(torch.float8_e4m3fn).float().view_as(groups)
    scaled = groups / scales[..., None]
    nearest = scaled.to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(
        (codes - scaled).abs(), (nearest - scaled).abs(), rtol=0, atol=2e-4
    )
    kv_dequant = (codes * scales[..., None]).reshape_as(kv_c).half()
    kv_error = (kv_dequant.float() - kv_c).norm() / kv_c.float().norm()
    assert kv_error < 0.035

    counts = torch.tensor(lengths, dtype=torch.int32)
    pages = torch.randperm(width // page, generator=rng)
    slots = (pages[:, None] * page + torch.arange(page)).flatten()
    indices = (
        slots.expand(rows, -1)
        .masked_fill(torch.arange(width) >= counts[:, None], -1)
        .int()
    )
    sinks = torch.linspace(-2, 5, heads)
    scale = 1 / 16

    def reference(latent):
        kv = torch.cat((latent, k_pe), dim=-1).view(-1, page, 576)
        return paged_attention_torch(q.float(), kv, indices, counts, sinks, scale, 512)

    expected = reference(kv_dequant)
    fp16 = reference(kv_c)
    norm = fp16[:-1].flatten(1).norm(dim=-1)
    quant_error = (expected[:-1] - fp16[:-1]).flatten(1).norm(dim=-1) / norm
    print(f"{device=} {heads=} {kv_error.item()=:.5f} {quant_error.tolist()=}")

    packed_reference = paged_attention_torch(
        q.float(), cache.cpu(), indices, counts, sinks, scale, 512
    )
    torch.testing.assert_close(packed_reference, expected, rtol=0, atol=0)
    args = (
        q.to(device),
        cache,
        indices.to(device),
        counts.to(device),
        sinks.to(device),
        scale,
    )
    workspace = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in decode_workspace_specs(rows, heads, width, 512)
    ]
    prefill = paged_attention(*args, 512).cpu().float()
    decode = paged_attention_splitk(*args, *workspace).cpu().float()
    for output in (prefill, decode):
        assert torch.equal(output[-1], torch.zeros_like(output[-1]))
        error = (output[:-1] - expected[:-1]).flatten(1).norm(dim=-1) / norm
        assert torch.all(error < 6e-4), error
