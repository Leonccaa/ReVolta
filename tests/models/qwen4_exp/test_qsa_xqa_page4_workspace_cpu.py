# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_int8 as reader
from vllm.v1.attention.kv_cache_view import KVCacheView, bind_planar_kv_cache
from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD, INT8


@pytest.fixture(autouse=True)
def workspace_cache(monkeypatch):
    saved = dict(ops._SM70_QSA_XQA_PAGE4_WORKSPACES)
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=123)
    )
    yield
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.update(saved)


def _query(rows=3):
    return torch.zeros(rows, 6, 256, dtype=torch.float16)


@pytest.mark.parametrize(
    "kv_dtype,expected",
    [("fp8_e4m3", torch.float32), ("auto", torch.float16), ("float16", torch.float16)],
)
def test_workspace_matches_native_output_contract_on_cpu(kv_dtype, expected):
    temporary, maximum, sums, active = ops._qsa_xqa_page4_workspace(
        _query(), 8, kv_dtype
    )
    assert temporary.dtype == expected
    assert temporary.shape == (3, 6, 8, 256)
    assert maximum.dtype == sums.dtype == torch.float32
    assert active.dtype == torch.int32 and active.tolist() == [8]


def test_workspace_cache_separates_incompatible_dtypes():
    query = _query()
    e4m3 = ops._qsa_xqa_page4_workspace(query, 8, "fp8_e4m3")[0]
    fp16 = ops._qsa_xqa_page4_workspace(query, 8, "auto")[0]
    assert e4m3.data_ptr() != fp16.data_ptr()
    cached_e4m3 = ops._qsa_xqa_page4_workspace(query, 8, "fp8_e4m3")[0]
    cached_fp16 = ops._qsa_xqa_page4_workspace(query, 8, "float16")[0]
    assert cached_e4m3.data_ptr() == e4m3.data_ptr()
    assert cached_fp16.data_ptr() == fp16.data_ptr()


def test_workspace_cache_separates_streams(monkeypatch):
    first = ops._qsa_xqa_page4_workspace(_query(), 8, "fp8_e4m3")[0]
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=124)
    )
    second = ops._qsa_xqa_page4_workspace(_query(), 8, "fp8_e4m3")[0]
    assert first.data_ptr() != second.data_ptr()


def test_workspace_grows_without_reusing_an_undersized_allocation():
    small = ops._qsa_xqa_page4_workspace(_query(), 8, "fp8_e4m3")[0]
    large = ops._qsa_xqa_page4_workspace(_query(7), 8, "fp8_e4m3")[0]
    assert large.shape == (7, 6, 8, 256)
    assert large.data_ptr() != small.data_ptr()
    again = ops._qsa_xqa_page4_workspace(_query(5), 8, "fp8_e4m3")[0]
    assert again.data_ptr() == large.data_ptr()


def test_workspace_partition_count_is_part_of_the_cache_key():
    first = ops._qsa_xqa_page4_workspace(_query(), 8, "fp8_e4m3")
    second = ops._qsa_xqa_page4_workspace(_query(), 16, "fp8_e4m3")
    assert second[0].shape == (3, 6, 16, 256)
    assert second[0].data_ptr() != first[0].data_ptr()
    assert second[3].tolist() == [16]


@pytest.mark.parametrize("block,route,rows", [(4, "decode", 3), (64, "grouped", 9)])
def test_planar_int8_native_mapping_and_workspace(block, route, rows, monkeypatch):
    codec = INT8
    raw = torch.zeros(2 * 2 * 128 * 520, dtype=torch.uint8)
    carrier = bind_planar_kv_cache(
        raw,
        num_blocks=2,
        scheduler_block_size=128,
        kernel_block_size=block,
        num_heads=1,
        head_size=256,
        page_size_bytes=128 * 520,
        member_index=1,
        member_count=2,
        dtype=codec.storage_dtype,
    )
    view = KVCacheView.bind(carrier, codec, 256)
    saved = raw.clone()
    q = _query(rows)
    ids = torch.zeros(rows, 2051, dtype=torch.int32)
    table = torch.zeros(1, view.key.shape[0], dtype=torch.int32)
    req = torch.zeros(rows, dtype=torch.int32)
    positions = torch.arange(rows)
    lengths = torch.tensor([128], dtype=torch.int32)
    calls = []

    def forward(*args):
        assert args[1].data_ptr() == view.key.data_ptr()
        assert args[2].stride() == view.value.stride()
        assert args[-4].data_ptr() == view.key_scale.data_ptr()
        assert args[-3].stride() == view.value_scale.stride()
        assert args[-2:] == (block // 4, 1 / 16)
        if route == "decode":
            assert args[6].dtype == args[7].dtype == args[8].dtype == torch.float32
            assert args[6].shape == (rows, 6, 9, 256)
        else:
            assert args[0].shape == (16, 6, 256)
            assert args[3].dtype == torch.float32
            args[7].fill_(2)
        args[3].fill_(3)
        calls.append("forward")

    def decode_plan(*args):
        assert args[-2:] == (block, block // 4)
        return torch.zeros(rows, 520, dtype=torch.int32), lengths.expand(rows)

    def grouped_plan(*args):
        assert args[-3:] == (block, block // 4, view.key.shape[0])
        calls.append("plan")

    native = SimpleNamespace(
        decode_int8_page4_fwd=forward,
        grouped_sparse_int8_page4_fwd=forward,
        grouped_sparse_page4_plan_fwd=grouped_plan,
    )
    monkeypatch.setattr(reader, "require_int8_abi", lambda: native)
    monkeypatch.setattr(ops, "_qsa_xqa_page4_block_table", decode_plan)
    monkeypatch.setattr(
        ops,
        "_qsa_grouped_page4_workspace",
        lambda q: (
            torch.zeros(2, 520, dtype=torch.int32),
            torch.zeros(2, 520, dtype=torch.uint32),
            torch.zeros(2, dtype=torch.int32),
            torch.empty(16, 6),
        ),
    )
    out = torch.empty_like(q, dtype=torch.float32 if route == "grouped" else q.dtype)
    lse = torch.empty(rows, 6)
    if route == "decode":
        reader.sparse_attention(
            q,
            view.key,
            view.value,
            ids,
            table,
            req,
            out,
            None,
            positions,
            lengths,
            view.key_scale,
            view.value_scale,
            codec=codec,
        )
    else:
        reader.native_grouped(
            q,
            view.key,
            view.value,
            ids,
            table,
            req,
            positions,
            lengths,
            out,
            view.key_scale,
            view.value_scale,
            lse,
            codec=codec,
        )
    assert calls == (["forward"] if route == "decode" else ["plan", "forward"])
    assert torch.all(out == 3)
    if route == "grouped":
        torch.testing.assert_close(lse, torch.full_like(lse, 2 * math.log2(math.e)))
    assert torch.equal(raw, saved)


def test_planar_e4m3_dcp_keeps_remainder_on_triton(monkeypatch):
    monkeypatch.setattr(reader, "require_int8_abi", lambda: None)
    monkeypatch.setattr(ops, "_SM70_QSA_DCP_GROUPED_PAGE4_MIN_ROWS", 64)
    calls = []

    def grouped(q, k, v, ids, table, req, pos, lengths, out, sk, sv, lse, *, codec):
        assert q.shape == (72, 12, 256) and codec is FP8_E4M3_PER_TOKEN_HEAD
        assert pos.tolist() == [127] * 72 and lengths.tolist() == [128]
        out.fill_(1)
        lse.fill_(2)
        calls.append(True)

    monkeypatch.setattr(reader, "native_grouped", grouped)
    q = torch.zeros(77, 12, 256, dtype=torch.float16)
    k = torch.zeros(2, 64, 1, 256, dtype=torch.uint8)
    sk = torch.ones(2, 64, 1)
    out, lse = torch.zeros(q.shape), torch.zeros(q.shape[:2])
    rows = ops.qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
        q,
        k,
        k,
        torch.zeros(77, 2051, dtype=torch.int32),
        torch.zeros(1, 2, dtype=torch.int32),
        torch.zeros(77, dtype=torch.int32),
        out,
        lse,
        FP8_E4M3_PER_TOKEN_HEAD.name,
        1.0,
        1.0,
        k_scale_cache=sk,
        v_scale_cache=sk,
    )
    assert rows == 72 and calls == [True]
    assert torch.all(out[:rows] == 1) and not out[rows:].count_nonzero()
    assert torch.all(lse[:rows] == 2) and not lse[rows:].count_nonzero()
