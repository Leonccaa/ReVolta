# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of VLLM_SM70_QWEN38_GDN_NO_PACK admission and routing.

No GPU arithmetic: the native dual launch is replaced by an FP64 emulation, so
the switch, the 2..48 row range, the shape/layout/policy admission, the GDN
input route and the loader's pack skip are checked on a GPU-less host. Numerical
evidence: tests/kernels/test_sm70_gdn_input_rowmajor.py (V100).
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
from vllm.models.qwen4_exp.nvidia import sm70_gdn_input_rowmajor as rowmajor

pytestmark = pytest.mark.skip_global_cleanup

ENV = "VLLM_SM70_QWEN38_GDN_NO_PACK"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


@pytest.fixture
def fake_native(monkeypatch):
    """Admit CPU tensors and emulate fp16_skinny_gemm2_sm70_out in FP64."""
    calls = []

    def launch2(outs, x2d, w0, w1, workspace, config):
        m = x2d.shape[0]
        assert x2d.stride(-1) == 1 and x2d.data_ptr() % 16 == 0
        assert workspace.dtype == torch.float32 and workspace.numel() == 0
        y0 = (x2d.double() @ w0.double().t()).half()
        y1 = (x2d.double() @ w1.double().t()).half()
        s0, s1 = outs[0].shape[1], outs[2].shape[1]
        for out, src in zip(outs, (y0[:, :s0], y0[:, s0:], y1[:, :s1], y1[:, s1:])):
            assert out.is_contiguous() and out.shape == src.shape
            out.copy_(src)
        calls.append((m, config))

    monkeypatch.setattr(rowmajor, "_device_ok", lambda t: True)
    monkeypatch.setattr(rowmajor, "_is_sm70", lambda: True)
    monkeypatch.setattr(rowmajor, "native_available", lambda: True)
    monkeypatch.setattr(rowmajor, "_launch2", launch2)
    return calls


def _half(*shape, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).half()


@pytest.fixture(scope="module")
def weights():
    return _half(4096, 2560, seed=1, scale=0.02), _half(24, 2560, seed=2, scale=0.02)


def _reference(x, wq, wb):
    y0 = (x.double() @ wq.double().t()).half()
    y1 = (x.double() @ wb.double().t()).half()
    return y0[:, :2560], y0[:, 2560:], y1[:, :12], y1[:, 12:]


def _linear_split(x, wq, wb):
    qkvz, ba = F.linear(x, wq), F.linear(x, wb)
    return qkvz[:, :2560], qkvz[:, 2560:], ba[:, :12], ba[:, 12:]


def _assert_equal(actual, expected):
    assert len(actual) == len(expected) == 4
    for got, want in zip(actual, expected):
        assert got.is_contiguous() and got.shape == want.shape
        assert torch.equal(got.view(torch.int16), want.contiguous().view(torch.int16))


def test_env_default_off_and_opt_in(monkeypatch):
    assert envs.VLLM_SM70_QWEN38_GDN_NO_PACK is False
    monkeypatch.setenv(ENV, "1")
    assert envs.VLLM_SM70_QWEN38_GDN_NO_PACK is True
    monkeypatch.setenv(ENV, "0")
    assert envs.VLLM_SM70_QWEN38_GDN_NO_PACK is False


def test_row_range_and_plan():
    assert [m for m in range(0, 60) if rowmajor.rows_admitted(m)] == list(range(2, 49))
    # Same plan as the N=4120, K=2560 entry of the generic skinny table.
    assert rowmajor.SkinnyConfig(1, 1, 2, 1) == rowmajor.PLAN


@pytest.mark.parametrize("m", (2, 5, 16, 17, 33, 48))
def test_helper_launch(fake_native, weights, m):
    wq, wb = weights
    x = _half(m, 2560, seed=m)
    _assert_equal(rowmajor.gdn_input_rowmajor(x, wq, wb), _reference(x, wq, wb))
    assert fake_native == [(m, rowmajor.PLAN)]


def test_helper_strided_input_is_copied(fake_native, weights):
    wq, wb = weights
    big = _half(20, 2560 + 24, seed=3)
    x = big[:, 4 : 4 + 2560]  # 8-byte offset: not 16-byte aligned
    assert x.data_ptr() % 16 != 0
    _assert_equal(rowmajor.gdn_input_rowmajor(x, wq, wb), _reference(x, wq, wb))
    assert len(fake_native) == 1


def test_helper_rejections(fake_native, weights, monkeypatch):
    wq, wb = weights
    x = _half(20, 2560, seed=4)
    rejected = {
        "m1": (x[:1], wq, wb),
        "m49": (_half(49, 2560, seed=5), wq, wb),
        "3d": (x.view(4, 5, 2560), wq, wb),
        "x_dtype": (x.float(), wq, wb),
        "qkvz_shape": (x, wq[:4088].contiguous(), wb),
        "ba_shape": (x, wq, wb[:16].contiguous()),
        "swapped": (x, wb, wq),
        "qkvz_dtype": (x, wq.float(), wb),
        "qkvz_layout": (x, wq.t().contiguous().t(), wb),
        "ba_layout": (x, wq, wb.t().contiguous().t()),
    }
    for name, args in rejected.items():
        assert rowmajor.gdn_input_rowmajor(*args) is None, name
    assert not fake_native
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert rowmajor.gdn_input_rowmajor(x, wq, wb) is None
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", True)
    assert rowmajor.gdn_input_rowmajor(x, wq, wb) is None
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    monkeypatch.setattr(rowmajor, "_is_sm70", lambda: False)
    assert rowmajor.gdn_input_rowmajor(x, wq, wb) is None
    monkeypatch.setattr(rowmajor, "_is_sm70", lambda: True)
    monkeypatch.setattr(rowmajor, "native_available", lambda: False)
    assert rowmajor.gdn_input_rowmajor(x, wq, wb) is None
    assert not fake_native
    monkeypatch.setattr(rowmajor, "native_available", lambda: True)
    assert rowmajor.gdn_input_rowmajor(x, wq, wb) is not None


def test_route_unchanged_when_unset(fake_native, weights):
    wq, wb = weights
    for m in (2, 5, 16, 17, 40, 48):
        x = _half(m, 2560, seed=10 + m)
        _assert_equal(
            gemv._qwen38_sm70_fp16_gdn_input(x, wq, wb), _linear_split(x, wq, wb)
        )
    assert not fake_native


def test_route_serves_2_to_48_rows_when_set(fake_native, weights, monkeypatch):
    wq, wb = weights
    monkeypatch.setenv(ENV, "1")
    for m in (2, 5, 8, 9, 16, 17, 24, 25, 40, 48):
        x = _half(m, 2560, seed=20 + m)
        _assert_equal(
            gemv._qwen38_sm70_fp16_gdn_input(x, wq, wb), _reference(x, wq, wb)
        )
        assert fake_native[-1] == (m, rowmajor.PLAN)
    count = len(fake_native)
    # M=1 (decode GEMV / prefill fallback) and M>48 keep the existing routes.
    for m in (1, 49, 64):
        x = _half(m, 2560, seed=30 + m)
        _assert_equal(
            gemv._qwen38_sm70_fp16_gdn_input(x, wq, wb), _linear_split(x, wq, wb)
        )
    # Rejected inputs fall back to F.linear as well.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    x = _half(20, 2560, seed=40)
    _assert_equal(gemv._qwen38_sm70_fp16_gdn_input(x, wq, wb), _linear_split(x, wq, wb))
    assert len(fake_native) == count


def test_packed_route_keeps_priority(fake_native, weights, monkeypatch):
    """A layer that still holds packs keeps the packed 2..16-row kernel."""
    wq, wb = weights
    monkeypatch.setenv(ENV, "1")
    packed_calls = []

    def packed_op(qkv, z, b, a, x, pq, pb):
        packed_calls.append(x.shape[0])
        for out, ref in zip((qkv, z, b, a), _reference(x, wq, wb)):
            out.copy_(ref)

    monkeypatch.setattr(gemv, "_can_use_packed_gdn_input", lambda x, pq, pb: True)
    monkeypatch.setattr(
        torch.ops._C, "qwen38_gdn_input_batch_sm70_out", packed_op, raising=False
    )
    x = _half(8, 2560, seed=50)
    _assert_equal(
        gemv._qwen38_sm70_fp16_gdn_input(x, wq, wb, wq, wb), _reference(x, wq, wb)
    )
    assert packed_calls == [8] and not fake_native


class _CudaLikeWeight:
    """The loader's GDN branch only reads ``is_cuda`` and ``dtype``."""

    is_cuda = True
    dtype = torch.float16


def _gdn_layer():
    layer = nn.Module()
    layer.weight = _CudaLikeWeight()
    layer._sm70_qwen38_prepare_gdn_batch = True
    return layer


@pytest.mark.parametrize("no_pack", (False, True))
def test_loader_skips_pack_only_when_set(monkeypatch, no_pack):
    packed = []

    def pack(weight):
        packed.append(weight)
        return torch.zeros(1)

    # Only the subclass's GDN branch is under test here.
    monkeypatch.setattr(
        UnquantizedLinearMethod,
        "process_weights_after_loading",
        lambda self, layer: None,
    )
    monkeypatch.setattr(gemv, "_pack_gdn_input_weight", pack)
    monkeypatch.setattr(
        torch.ops._C, "qwen38_gdn_input_batch_sm70_out", object(), raising=False
    )
    if no_pack:
        monkeypatch.setenv(ENV, "1")
    layer = _gdn_layer()
    gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)
    if no_pack:
        assert not packed and not hasattr(layer, "_sm70_qwen38_gdn_packed")
    else:
        assert packed == [layer.weight]
        assert hasattr(layer, "_sm70_qwen38_gdn_packed")
        assert "_sm70_qwen38_gdn_packed" not in layer.state_dict()


def test_loader_ignores_unmarked_layers(monkeypatch):
    monkeypatch.setattr(
        UnquantizedLinearMethod,
        "process_weights_after_loading",
        lambda self, layer: None,
    )
    monkeypatch.setattr(gemv, "_pack_gdn_input_weight", lambda w: torch.zeros(1))
    monkeypatch.setattr(
        torch.ops._C, "qwen38_gdn_input_batch_sm70_out", object(), raising=False
    )
    for value in ("0", "1"):
        monkeypatch.setenv(ENV, value)
        layer = _gdn_layer()
        layer._sm70_qwen38_prepare_gdn_batch = False
        gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)
        assert not hasattr(layer, "_sm70_qwen38_gdn_packed")
