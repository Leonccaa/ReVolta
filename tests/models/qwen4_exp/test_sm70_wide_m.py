# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_SM70_QWEN38_WIDE_M*: opt-in admission widenings for 8-request MTP4
verify batches (M up to 40 tokens). Every switch is off by default and the
default admissions are unchanged; the CPU tests check both. The last test
runs the native exact gated RMSNorm above its former 192-row bound on SM70.
"""

import pytest
import torch

from vllm import envs

pytestmark = pytest.mark.skip_global_cleanup

ITEMS = (
    "VLLM_SM70_QWEN38_WIDE_M_GDN",
    "VLLM_SM70_QWEN38_WIDE_M_GATED_NORM",
    "VLLM_SM70_QWEN38_WIDE_M_FP8_MOE_BUFFERS",
    "VLLM_SM70_QWEN38_WIDE_M_PLE",
    "VLLM_SM70_QWEN38_WIDE_M_CAPTURE",
)


@pytest.fixture
def wide(monkeypatch):
    """Clear every wide-M switch; returns a setter for the umbrella/items."""
    for name in ("VLLM_SM70_QWEN38_WIDE_M", *ITEMS):
        monkeypatch.delenv(name, raising=False)

    def set_switch(name="VLLM_SM70_QWEN38_WIDE_M", value="1"):
        monkeypatch.setenv(name, value)

    return set_switch


def test_items_default_off_and_follow_umbrella(wide):
    assert not envs.VLLM_SM70_QWEN38_WIDE_M
    assert not any(getattr(envs, name) for name in ITEMS)
    wide()
    assert all(getattr(envs, name) for name in ITEMS)
    wide("VLLM_SM70_QWEN38_WIDE_M_PLE", "0")
    assert not envs.VLLM_SM70_QWEN38_WIDE_M_PLE
    assert envs.VLLM_SM70_QWEN38_WIDE_M_GDN


def test_single_item_without_umbrella(wide):
    wide("VLLM_SM70_QWEN38_WIDE_M_CAPTURE")
    assert envs.VLLM_SM70_QWEN38_WIDE_M_CAPTURE
    assert not envs.VLLM_SM70_QWEN38_WIDE_M_GDN


@pytest.mark.parametrize("tail_copy", [False, True])
@pytest.mark.parametrize("on", [False, True])
def test_gdn_token_bands(wide, monkeypatch, tail_copy, on):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    monkeypatch.setenv("VLLM_SM70_GDN_TAIL_COPY", "1" if tail_copy else "0")
    if on:
        wide("VLLM_SM70_QWEN38_WIDE_M_GDN")
    for t in range(0, 60):
        default_verify = 1 < t <= 16 or (tail_copy and t == 20)
        expected = default_verify or (on and 2 <= t <= 48)
        assert gdn._sm70_mixed_qkv_verify_tokens_ok(t) == expected, t
        default_tail = t in (5, 20)
        expected = default_tail or (on and 2 <= t <= 48)
        assert gdn._sm70_gdn_tail_copy_tokens_ok(t) == expected, t


@pytest.mark.parametrize("on", [False, True])
def test_gated_norm_rows(wide, on):
    from vllm.model_executor.layers.layernorm import _sm70_gated_norm_shape_supported

    if on:
        wide("VLLM_SM70_QWEN38_WIDE_M_GATED_NORM")
    weight = torch.ones(128, dtype=torch.float16)
    limit = 576 if on else 192
    for rows in (1, 192, 193, 480, 576, 577):
        x = torch.zeros(rows, 128, dtype=torch.float16)
        assert _sm70_gated_norm_shape_supported(x, x.clone(), weight) == (
            rows <= limit
        ), rows


@pytest.mark.parametrize("on", [False, True])
def test_fp8_moe_persistent_tokens(wide, on):
    from vllm.model_executor.layers.quantization import fp8_sm70_moe

    if on:
        wide("VLLM_SM70_QWEN38_WIDE_M_FP8_MOE_BUFFERS")
    assert fp8_sm70_moe._persistent_max_tokens() == (48 if on else 32)


@pytest.mark.parametrize("on", [False, True])
def test_ple_bounds(wide, on):
    from vllm.model_executor.kernels.ple.ngram import SM70_PLE_NGRAM
    from vllm.models.qwen4_exp.nvidia import model_state

    if on:
        wide("VLLM_SM70_QWEN38_WIDE_M_PLE")
    assert SM70_PLE_NGRAM.max_tokens() == (48 if on else 32)
    assert model_state._ple_input_prepare_max_padded_requests() == (8 if on else 4)


@pytest.mark.parametrize(
    "max_num_seqs,default,widened",
    [
        (4, [5, 10, 15, 20], [5, 10, 15, 20]),
        (6, [5, 10, 15, 20, 30], [5, 10, 15, 20, 25, 30]),
        (8, [5, 10, 15, 20, 30, 40], [5, 10, 15, 20, 25, 30, 35, 40]),
        (16, [5, 10, 15, 20, 30, 40, 60, 80], [5, 10, 15, 20, 25, 30, 35, 40, 60, 80]),
    ],
)
def test_mtp_capture_sizes(wide, max_num_seqs, default, widened):
    from vllm.config.vllm import (
        _sm70_mtp_cudagraph_capture_sizes,
        _sm70_speculative_cudagraph_capture_sizes,
    )

    assert _sm70_mtp_cudagraph_capture_sizes(max_num_seqs, 5) == default
    aux = {1, 2, 4, 8, 9, 18}
    assert _sm70_speculative_cudagraph_capture_sizes(max_num_seqs, 5) == sorted(
        aux | set(default)
    )
    wide("VLLM_SM70_QWEN38_WIDE_M_CAPTURE")
    assert _sm70_mtp_cudagraph_capture_sizes(max_num_seqs, 5) == widened
    assert _sm70_speculative_cudagraph_capture_sizes(max_num_seqs, 5) == sorted(
        aux | set(widened)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not hasattr(torch.ops._C, "sm70_rmsnorm_gated_exact_out"),
    reason="requires SM70 and the native exact gated RMSNorm",
)
@pytest.mark.parametrize("silu", [False, True])
def test_native_gated_norm_wide_rows_match_row_slices(silu):
    """Rows above the former 192 bound compute exactly what <=192-row calls
    compute (one warp per row), up to the new 576-row bound."""
    torch.manual_seed(7)
    rows = 576
    x = torch.randn(rows, 128, device="cuda").half()
    z = torch.randn(rows, 128, device="cuda").half()
    weight = (torch.rand(128, device="cuda") + 0.5).half()
    full = torch.empty_like(x)
    torch.ops._C.sm70_rmsnorm_gated_exact_out(full, x, z, weight, 1e-6, silu)
    for start in range(0, rows, 192):
        part = torch.empty_like(x[start : start + 192])
        torch.ops._C.sm70_rmsnorm_gated_exact_out(
            part,
            x[start : start + 192].contiguous(),
            z[start : start + 192].contiguous(),
            weight,
            1e-6,
            silu,
        )
        assert torch.equal(part, full[start : start + 192]), start
    for size in (193, 480):
        out = torch.empty_like(x[:size])
        torch.ops._C.sm70_rmsnorm_gated_exact_out(
            out, x[:size].contiguous(), z[:size].contiguous(), weight, 1e-6, silu
        )
        assert torch.equal(out, full[:size]), size
    big = torch.zeros(rows + 1, 128, device="cuda").half()
    with pytest.raises(RuntimeError):
        torch.ops._C.sm70_rmsnorm_gated_exact_out(
            torch.empty_like(big), big, big.clone(), weight, 1e-6, silu
        )
