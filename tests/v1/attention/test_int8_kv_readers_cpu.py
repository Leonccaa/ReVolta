# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the production Triton reader on CPU with TRITON_INTERPRET=1."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

reader = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.qsa_int8")
from vllm.v1.attention.kv_cache_view import (  # noqa: E402
    KVCacheView,
    bind_planar_kv_cache,
)
from vllm.v1.attention.kv_codecs import INT8  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="requires CPU interpreter harness"
)


def cache(member=1, padding=0, device="cpu"):
    page = 424320 + padding
    raw = torch.zeros((3 * 2 * page,), dtype=torch.int8, device=device)
    carrier = bind_planar_kv_cache(
        raw,
        num_blocks=3,
        scheduler_block_size=816,
        kernel_block_size=16,
        num_heads=1,
        head_size=256,
        page_size_bytes=page,
        member_index=member,
        member_count=2,
    )
    return raw, carrier, KVCacheView.bind(carrier, INT8, 256)


def test_exact_half2_subtraction_all_byte_pairs():
    raw = np.arange(65536, dtype=np.uint32)
    values = (0x64006400 | (raw & 255) | ((raw & 65280) << 8)).astype(np.uint32)
    bias = (0x64006400 | ((raw & 128) << 1) | ((raw & 32768) << 9)).astype(np.uint32)
    actual = (values.view(np.float16) - bias.view(np.float16)).view(np.uint16)
    expected = raw.astype(np.uint16).view(np.int8).astype(np.float16).view(np.uint16)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("repeat", [1, 5])
@pytest.mark.parametrize("rows", [1, 5, 8, 17, 20, 40])
@pytest.mark.parametrize("member,padding", [(0, 0), (1, 816)])
def test_interpreted_reader_against_float64(rows, member, padding, repeat):
    assert os.environ.get("TRITON_INTERPRET") == "1"
    torch.manual_seed(412)
    _, _, view = cache(member, padding)
    view.key.random_(-128, 128)
    view.value.random_(-128, 128)
    view.key_scale.fill_(0.007)
    view.value_scale.fill_(0.013)
    q = torch.randn((rows, 6, 256), dtype=torch.float16) * 0.3
    tokens = torch.tensor([15, 16, 815, 816, 1631, 1632, 2447, -1, 2448])
    indices = tokens.to(torch.int32).repeat(rows, repeat)
    # Vary token/head scales; a small value contribution must survive late-scale.
    for t in tokens[:7].tolist():
        view.key_scale[t // 16, t % 16] = 2.0 ** (-(t % 9))
        view.value_scale[t // 16, t % 16] = 2.0 ** (-(t % 17))
    table = torch.arange(153, dtype=torch.int32)[None, :]
    req = torch.zeros(rows, dtype=torch.int32)
    req[-1] = -1 if rows > 1 else 0
    out = torch.empty_like(q)
    lse = torch.empty((rows, 6), dtype=torch.float32)
    reader.triton_attention(
        q,
        view.key,
        view.value,
        indices,
        table,
        req,
        out,
        view.key_scale,
        view.value_scale,
        lse,
    )
    k = torch.stack(
        [
            view.key[t // 16, t % 16, 0].double()
            * view.key_scale[t // 16, t % 16, 0].double()
            for t in tokens[:7]
        ]
    )
    v = torch.stack(
        [
            view.value[t // 16, t % 16, 0].double()
            * view.value_scale[t // 16, t % 16, 0].double()
            for t in tokens[:7]
        ]
    )
    k = k.repeat(repeat, 1)
    v = v.repeat(repeat, 1)
    score = q.double() @ k.T / 16
    ref = torch.softmax(score, -1) @ v
    ref_lse = torch.logsumexp(score, -1) / np.log(2)
    if rows > 1:
        ref[-1] = 0
        ref_lse[-1] = -float("inf")
    torch.testing.assert_close(out.float(), ref.float(), atol=0.004, rtol=0.015)
    torch.testing.assert_close(lse, ref_lse.float(), atol=0.002, rtol=0.002)


@pytest.mark.parametrize("kind", ["missing", "shape", "dtype"])
def test_reject_invalid_scale_planes(kind):
    _, _, v = cache()
    q = torch.empty((1, 6, 256), dtype=torch.float16)
    sk = {
        "missing": None,
        "shape": v.key_scale[:1],
        "dtype": v.key_scale.to(torch.float16),
    }[kind]
    with pytest.raises(ValueError, match="scale planes"):
        reader.validate_planes(q, v.key, v.value, sk, v.value_scale)


def test_fail_closed_native_abi(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "flash_attn_v100.flash_attn_interface",
        SimpleNamespace(flash_attn_v100_cuda=SimpleNamespace()),
    )
    with pytest.raises(RuntimeError, match="ABI v1"):
        reader.require_int8_abi()


@pytest.mark.parametrize("rows", [1, 5, 8, 17, 20, 40])
def test_interpreted_dense_draft(rows, monkeypatch):
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.ops import int8_dense

    monkeypatch.setattr(int8_dense, "require_int8_abi", lambda: None)
    torch.manual_seed(35)
    _, carrier, view = cache()
    view.key.random_(-128, 128)
    view.value.random_(-128, 128)
    view.key_scale.fill_(0.002)
    view.value_scale.fill_(0.007)
    query = torch.randn(rows, 6, 256, dtype=torch.float16) * 0.2
    impl = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        alibi_slopes=None,
        logits_soft_cap=0,
        sinks=None,
        sliding_window=(-1, -1),
        head_size=256,
        scale=0.0625,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        max_query_len=rows,
        max_seq_len=64,
        causal=True,
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
        seq_lens=torch.tensor([64], dtype=torch.int32),
        block_table=torch.arange(153, dtype=torch.int32)[None, :],
    )
    out = torch.empty_like(query)
    int8_dense.forward_int8_dense(impl, query, carrier, metadata, out)
    k = view.key[:4].reshape(64, 256).double() * 0.002
    v = view.value[:4].reshape(64, 256).double() * 0.007
    for row in range(rows):
        length = 64 - rows + row + 1
        scores = query[row].double() @ k[:length].T / 16
        expected = torch.softmax(scores, -1) @ v[:length]
        torch.testing.assert_close(
            out[row].float(), expected.float(), atol=0.004, rtol=0.015
        )


def test_dense_writer_uses_borrowed_planar_scales(monkeypatch):
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops import int8_dense

    monkeypatch.setattr(int8_dense, "require_int8_abi", lambda: None)
    received = []
    monkeypatch.setattr(
        ops, "reshape_and_cache_flash", lambda *a, **kw: received.append((a, kw))
    )
    _, carrier, view = cache()
    layer = SimpleNamespace(_k_scale=torch.tensor(1.0), _v_scale=torch.tensor(1.0))
    impl = SimpleNamespace(head_size=256)
    key = torch.zeros(1, 1, 256, dtype=torch.float16)
    int8_dense.write_int8_dense(
        impl, layer, key, key, carrier, torch.tensor([16], dtype=torch.int64)
    )
    args, kwargs = received[0]
    assert args[2].dtype == args[3].dtype == torch.int8
    assert kwargs["k_scale_cache"].data_ptr() == view.key_scale.data_ptr()
    assert kwargs["v_scale_cache"].stride() == view.value_scale.stride()


def test_grouped_int8_capability_does_not_inherit_legacy_abi():
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_grouped_page4_supported

    old = SimpleNamespace(
        grouped_sparse_page4_abi_version=lambda: 2,
        grouped_sparse_page4_fwd=lambda: None,
        grouped_sparse_page4_plan_fwd=lambda: None,
    )
    assert not _qsa_grouped_page4_supported(old, "int8")
    old.int8_page4_abi_version = lambda: 1
    old.grouped_sparse_int8_page4_fwd = lambda: None
    assert _qsa_grouped_page4_supported(old, "int8")


@pytest.mark.parametrize("visible", [False, True])
def test_nonfinite_scale_is_masked_or_propagated(visible):
    _, _, view = cache()
    view.key.zero_()
    view.value.fill_(1)
    view.key_scale.fill_(0.01)
    view.value_scale.fill_(0.02)
    view.key_scale[0, 0, 0] = float("nan")
    query = torch.zeros(1, 6, 256, dtype=torch.float16)
    ids = torch.full((1, 40), 0 if visible else -1, dtype=torch.int32)
    table = torch.arange(153, dtype=torch.int32)[None]
    req = torch.zeros(1, dtype=torch.int32)
    out = torch.empty_like(query)
    lse = torch.empty(1, 6)
    reader.triton_attention(
        query,
        view.key,
        view.value,
        ids,
        table,
        req,
        out,
        view.key_scale,
        view.value_scale,
        lse,
    )
    if visible:
        assert out.isnan().all() and lse.isnan().all()
    else:
        assert not out.count_nonzero() and lse.isneginf().all()


def test_late_scale_preserves_tiny_fp32_value_residual():
    _, _, view = cache()
    view.key.zero_()
    view.value.zero_()
    view.value[0, 0].fill_(1)
    view.value[0, 1].fill_(-1)
    view.key_scale.fill_(1e-6)
    view.value_scale.fill_(2.0**-19)
    view.value_scale[0, 0] += 2.0**-32
    query = torch.zeros(1, 6, 256, dtype=torch.float16)
    ids = torch.full((1, 40), -1, dtype=torch.int32)
    ids[0, :2] = torch.tensor([0, 1], dtype=torch.int32)
    out = torch.empty(query.shape, dtype=torch.float32)
    reader.triton_attention(
        query,
        view.key,
        view.value,
        ids,
        torch.arange(153, dtype=torch.int32)[None],
        torch.zeros(1, dtype=torch.int32),
        out,
        view.key_scale,
        view.value_scale,
        torch.empty(1, 6),
    )
    # Early Half KV or one Half P*sV both cancel this exact residual to zero.
    expected = torch.full_like(out, 2.0**-33)
    torch.testing.assert_close(out, expected, atol=0, rtol=1e-5)
