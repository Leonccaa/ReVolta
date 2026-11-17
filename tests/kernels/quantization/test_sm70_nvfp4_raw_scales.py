# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-scale expansion and decode admission must cover E4M3 edge encodings."""

import pytest
import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    _raw_scales_match_prepared,
)


@pytest.fixture(autouse=True)
def require_sm70_raw_scale_ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    if not sm70_ops.has_nvfp4_qpn_raw_scale_dispatch():
        pytest.skip("requires raw-scale native operators")


@pytest.mark.parametrize(
    "stage,interleaved", [("w13", False), ("w13", True), ("w2", False)]
)
def test_raw_scale_strict_expansion_all_e4m3_codes(stage, interleaved):
    shape = (512, 160, 320) if stage == "w13" else (512, 10, 2560)
    # CPU conversion is an independent reference, including signed zeros,
    # subnormals and NaNs; no SM70 native FP8 arithmetic is assumed.
    codebook = torch.arange(256, dtype=torch.uint8)
    values = codebook.view(torch.float8_e4m3fn).float().cuda()
    codes = codebook.cuda().repeat(torch.Size(shape).numel() // 256).view(shape)
    globals_ = (
        torch.tensor(
            [0.0, 2**-24, 2**-10, 2**-4, 1.0, 2.0, -1.0, 256.0],
            device="cuda",
            dtype=torch.float32,
        )
        .repeat(128 if stage == "w13" else 64)
        .view(512, -1)
    )
    if stage == "w13":
        columns = torch.arange(shape[-1], device="cuda")
        slots = (columns % 2) if interleaved else (columns >= 160).long()
        global_by_column = globals_[:, slots].unsqueeze(1)
    else:
        global_by_column = globals_.unsqueeze(1)
    expected = (values[codes.long()] * global_by_column).half()
    out = torch.empty(shape, dtype=torch.float16, device="cuda")
    sm70_ops.nvfp4_expand_raw_scales_sm70_out(out, codes, globals_, interleaved, False)
    finite = ~expected.isnan()
    # Bitwise comparison also preserves signed zero; NaN payload is immaterial.
    assert torch.equal(
        out.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )
    assert torch.equal(out.isnan(), expected.isnan())

    # Replay must read current code/global buffers, not capture-time values.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        sm70_ops.nvfp4_expand_raw_scales_sm70_out(
            out, codes, globals_, interleaved, False
        )
    codes.fill_(64)  # E4M3 2.0
    globals_.fill_(2**-4)
    graph.replay()
    assert torch.equal(out, torch.full_like(out, 0.125))


@pytest.mark.parametrize(
    "stage,interleaved", [("w13", False), ("w13", True), ("w2", False)]
)
def test_raw_scale_fast_expansion_matches_hmma_reconstruction(stage, interleaved):
    # The fast path reconstructs raw * (16384 * global) as hi + lo FP16 terms:
    # one FP16 multiply for the low term, then one fused multiply-add.
    shape = (512, 160, 320) if stage == "w13" else (512, 10, 2560)
    gen = torch.Generator(device="cuda").manual_seed(17)
    # Positive normal E4M3 codes, which are the encodings the fast path admits.
    codes = torch.randint(8, 127, shape, generator=gen, device="cuda").to(torch.uint8)
    width = 2 if stage == "w13" else 1
    globals_ = torch.rand(512, width, generator=gen, device="cuda") * 2**-8 + 2**-12
    if stage == "w13":
        columns = torch.arange(shape[-1], device="cuda")
        slots = (columns % 2) if interleaved else (columns >= 160).long()
        global_by_column = globals_[:, slots].unsqueeze(1)
    else:
        global_by_column = globals_.unsqueeze(1)
    scaled = (global_by_column * 16384.0).double()
    hi = scaled.half()
    lo = (scaled - hi.double()).half()
    raw = ((codes.int() << 7) + 0x2000).short().view(torch.float16)
    correction = (raw.double() * lo.double()).half()
    expected = (raw.double() * hi.double() + correction.double()).half()
    out = torch.empty(shape, dtype=torch.float16, device="cuda")
    sm70_ops.nvfp4_expand_raw_scales_sm70_out(out, codes, globals_, interleaved, True)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("code", [0, 1, 7, 128, 129, 255])
def test_raw_scale_admission_rejects_unsupported_fast_encodings(code):
    shape = (512, 10, 2560)
    codes = torch.full(shape, code, dtype=torch.uint8, device="cuda")
    globals_ = torch.full((512, 1), 2**-10, dtype=torch.float32, device="cuda")
    prepared = torch.empty(shape, dtype=torch.float16, device="cuda")
    sm70_ops.nvfp4_expand_raw_scales_sm70_out(prepared, codes, globals_, False, False)
    assert not _raw_scales_match_prepared(
        torch.empty_like(prepared), codes, globals_, prepared, False
    )


def test_raw_scale_admission_checks_effective_hmma_scale():
    shape = (512, 10, 2560)
    codes = torch.full(shape, 64, dtype=torch.uint8, device="cuda")
    globals_ = torch.full((512, 1), 2**-10, dtype=torch.float32, device="cuda")
    prepared = torch.full(shape, 2**-9, dtype=torch.float16, device="cuda")
    out = torch.empty_like(prepared)
    assert _raw_scales_match_prepared(out, codes, globals_, prepared, False)
    assert torch.equal(out, prepared * 16384.0)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("equal_globals", [False, True])
def test_prepared_w13_matches_raw_consumers(interleaved, equal_globals):
    torch.manual_seed(442)
    codes = torch.randint(0, 16, (2560, 320), device="cuda", dtype=torch.uint8)
    raw = torch.randint(48, 80, (160, 320), device="cuda", dtype=torch.uint8)
    globals_ = torch.tensor([2**-10, 2**-10 if equal_globals else 2**-9], device="cuda")
    logical_global = globals_.repeat_interleave(160)
    effective = (raw.view(torch.float8_e4m3fn).float() * logical_global).half()
    weight, prepared, _ = sm70_ops.nvfp4_sm70_prepare(
        codes, effective, 16, interleave_gated_silu=interleaved
    )
    physical_global = globals_.repeat(160) if interleaved else logical_global
    raw_prepared = (
        (prepared.float() / physical_global)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .contiguous()
    )
    weight = weight.unsqueeze(0).repeat(512, 1, 1)
    prepared = prepared.unsqueeze(0).repeat(512, 1, 1)
    raw_prepared = raw_prepared.unsqueeze(0).repeat(512, 1, 1)
    globals_ = globals_.unsqueeze(0).repeat(512, 1)
    expanded = torch.empty_like(prepared)
    sm70_ops.nvfp4_expand_raw_scales_sm70_out(
        expanded, raw_prepared, globals_, interleaved, False
    )
    assert torch.equal(expanded.view(torch.int16), prepared.view(torch.int16))

    x = (torch.randn(8, 2560, device="cuda") * 0.1).half()
    ids = torch.arange(80, device="cuda", dtype=torch.int32) % 3
    projected, raw_projected = [
        torch.empty(10, 320, device="cuda", dtype=torch.float16) for _ in range(2)
    ]
    sm70_ops.nvfp4_moe_qpn_m1_sm70_out(
        projected, x[:1], weight, prepared, ids[:10], True, 4
    )
    sm70_ops.nvfp4_moe_qpn_raw_scale_sm70_out(
        raw_projected,
        x[:1],
        weight,
        raw_prepared,
        globals_,
        ids[:10],
        True,
        interleaved,
        4,
    )
    assert torch.equal(raw_projected.view(torch.int16), projected.view(torch.int16))

    fused, raw_fused = [
        torch.empty(80, 160, device="cuda", dtype=torch.float16) for _ in range(2)
    ]
    sm70_ops.nvfp4_moe_qpn_w13_swiglu_batch_sm70_out(
        fused, x, weight, prepared, ids, interleaved
    )
    sm70_ops.nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out(
        raw_fused, x, weight, raw_prepared, globals_, ids, interleaved
    )
    assert torch.equal(raw_fused.view(torch.int16), fused.view(torch.int16))

    if not sm70_ops.has_nvfp4_raw_grouped_decode_dispatch():
        pytest.skip("requires raw grouped-decode native operator")
    # W2 is unused by the assertion; compare the real grouped W13 intermediate
    # against the prepared-scale fused W13 result, with the same split-K=4.
    grouped = torch.empty_like(fused)
    sm70_ops.nvfp4_raw_grouped_decode_sm70_out(
        torch.empty_like(x),
        grouped,
        torch.empty(1608, device="cuda", dtype=torch.int32),
        x,
        weight,
        raw_prepared,
        globals_,
        torch.zeros(512, 160, 320, device="cuda", dtype=torch.int32),
        torch.full((512, 10, 2560), 64, device="cuda", dtype=torch.uint8),
        torch.full((512, 1), 2**-10, device="cuda"),
        ids.view(8, 10),
        torch.full((8, 10), 0.1, device="cuda"),
        4,
        interleaved,
    )
    assert torch.equal(grouped.view(torch.int16), fused.view(torch.int16))
