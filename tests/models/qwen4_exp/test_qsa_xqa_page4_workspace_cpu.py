# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
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
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(ops, "_SM70_QSA_GROUPED_PAGE4_SCRATCH", {})
    yield
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.update(saved)


def _query(rows=3):
    return torch.zeros(rows, 6, 256, dtype=torch.float16)


@pytest.mark.parametrize("partial_dtype", [torch.float16, torch.float32])
def test_workspace_matches_native_output_contract_on_cpu(partial_dtype):
    temporary, maximum, sums, active = ops._qsa_xqa_page4_workspace(
        _query(), 8, partial_dtype
    )
    assert temporary.dtype == partial_dtype
    assert temporary.shape == (3, 6, 8, 256)
    assert maximum.dtype == sums.dtype == torch.float32
    assert active.dtype == torch.int32 and active.tolist() == [8]


def test_workspace_cache_separates_incompatible_dtypes():
    query = _query()
    e4m3 = ops._qsa_xqa_page4_workspace(query, 8, torch.float32)[0]
    fp16 = ops._qsa_xqa_page4_workspace(query, 8, torch.float16)[0]
    assert e4m3.data_ptr() != fp16.data_ptr()
    cached_e4m3 = ops._qsa_xqa_page4_workspace(query, 8, torch.float32)[0]
    cached_fp16 = ops._qsa_xqa_page4_workspace(query, 8, torch.float16)[0]
    assert cached_e4m3.data_ptr() == e4m3.data_ptr()
    assert cached_fp16.data_ptr() == fp16.data_ptr()


def test_workspace_cache_separates_streams(monkeypatch):
    first = ops._qsa_xqa_page4_workspace(_query(), 8, torch.float32)[0]
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=124)
    )
    second = ops._qsa_xqa_page4_workspace(_query(), 8, torch.float32)[0]
    assert first.data_ptr() != second.data_ptr()


def test_workspace_reserves_capture_width_and_keeps_its_address():
    context = ForwardContext({}, {}, {}, max_cudagraph_capture_size=40)
    with override_forward_context(context):
        target = ops._qsa_xqa_page4_workspace(_query(5), 9, torch.float32)[0]
        draft = ops._qsa_xqa_page4_workspace(_query(40), 9, torch.float32)[0]
        eager = ops._qsa_xqa_page4_workspace(_query(48), 9, torch.float32)[0]
        replay = ops._qsa_xqa_page4_workspace(_query(5), 9, torch.float32)[0]
    assert draft.shape == (40, 6, 9, 256)
    assert eager.shape == (48, 6, 9, 256)
    assert target.data_ptr() == draft.data_ptr() == replay.data_ptr()
    assert eager.data_ptr() != target.data_ptr()


def test_workspace_partition_count_is_part_of_the_cache_key():
    first = ops._qsa_xqa_page4_workspace(_query(), 8, torch.float32)
    second = ops._qsa_xqa_page4_workspace(_query(), 16, torch.float32)
    assert second[0].shape == (3, 6, 16, 256)
    assert second[0].data_ptr() != first[0].data_ptr()
    assert second[3].tolist() == [16]


def test_grouped_scratch_reuses_buffers_and_separates_output_dtypes():
    query = _query(16)
    first = ops._qsa_grouped_page4_scratch(query, 16, torch.float32)
    second = ops._qsa_grouped_page4_scratch(query, 8, torch.float32)
    assert [t.shape[0] for t in second] == [8, 8]
    assert [t.data_ptr() for t in first] == [t.data_ptr() for t in second]
    different = ops._qsa_grouped_page4_scratch(query, 8, torch.float16)
    assert different[1].dtype == torch.float16
    assert different[1].data_ptr() != first[1].data_ptr()


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="CPU Triton interpreter required"
)
@pytest.mark.parametrize("owner_tokens", [4, 64, 784])
def test_xqa_page4_plan_matches_selected_tokens_during_mtp(owner_tokens):
    # Fresh selections and later steps reusing them, including both sides
    # of a four-token boundary and rows that fill the selector budget.
    prefixes = [(1, 1), (4, 4), (7, 7), (7, 8), (8, 12), (2051, 2051), (4103, 4107)]
    rows = len(prefixes)
    num_blocks = math.ceil(max(current for _, current in prefixes) / owner_tokens)
    table = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int32).expand(rows, -1)
    requests = torch.arange(rows, dtype=torch.int32)
    lengths = torch.tensor([current for _, current in prefixes], dtype=torch.int32)
    indices = torch.full((rows, 2051), -1, dtype=torch.int32)
    selections = []
    for row, (selected, _) in enumerate(prefixes):
        complete, tail = divmod(selected, 4)
        complete_pages = torch.arange(min(complete, 512), dtype=torch.int32)
        if complete > 512:
            # A normal top-k selection can omit pages and be out of order.
            complete_pages = (complete_pages * 2).flip(0)
        tokens = (complete_pages[:, None] * 4 + torch.arange(4)).flatten()
        tail_tokens = torch.arange(complete * 4, complete * 4 + tail)
        selection = torch.cat((tokens, tail_tokens))
        indices[row, : selection.numel()] = selection
        selections.append(selection.sort().values)

    pages, selected_lengths = ops._qsa_xqa_page4_block_table(
        indices,
        table,
        requests,
        lengths.to(torch.int64) - 1,
        lengths,
        num_blocks,
        owner_tokens,
    )
    for row, selection in enumerate(selections):
        selected_count = selected_lengths[row].item()
        assert selected_count == selection.numel()
        page_ids = pages[row, : math.ceil(selected_count / 4)]
        physical_tokens = (page_ids[:, None] * 4 + torch.arange(4)).flatten()
        expected = (
            table[row, selection // owner_tokens] * owner_tokens
            + selection % owner_tokens
        )
        torch.testing.assert_close(physical_tokens[:selected_count], expected)


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
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_v100.flash_attn_interface",
        SimpleNamespace(flash_attn_v100_cuda=native),
    )
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


def test_planar_int8_dcp_prefill_writes_all_rows_with_local_width(monkeypatch):
    rows = ops._SM70_QSA_DCP_GROUPED_PAGE4_MIN_ROWS + 1
    q = torch.zeros(rows, 12, 256, dtype=torch.float16)
    k = torch.zeros(2, 16, 1, 256, dtype=torch.int8)
    v = torch.empty_like(k)
    sk, sv = torch.ones(2, 16, 1), torch.full((2, 16, 1), 0.5)
    indices = torch.full((rows, 2051), -1, dtype=torch.int32)
    indices[:, :3] = torch.tensor([0, 15, 16], dtype=torch.int32)
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    requests = torch.zeros(rows, dtype=torch.int32)
    out, lse = torch.zeros(q.shape), torch.zeros(q.shape[:2])
    calls = []

    def rows_op(
        query, key, value, ks, vs, ids, blocks, req, pos, lengths, dst, dst_lse
    ):
        assert query is q and key is k and value is v
        assert ks is sk and vs is sv and blocks is table and req is requests
        assert ids.shape == (rows, 1026) and ids.stride() == (2051, 1)
        assert ids.data_ptr() == indices.data_ptr()
        assert torch.all(pos == 31) and lengths.tolist() == [32] * table.shape[0]
        assert pos.dtype == torch.int64 and lengths.dtype == torch.int32
        assert dst is out and dst_lse is lse
        dst.fill_(1)
        dst_lse.fill_(2)
        calls.append("rows")

    monkeypatch.setitem(sys.modules, "vllm._sm70_qsa_device_C", SimpleNamespace())
    monkeypatch.setattr(reader.current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setattr(
        torch.ops.vllm_sm70_qsa_device, "rows_int8_planar", rows_op, raising=False
    )

    def call():
        return ops.qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
            q,
            k,
            v,
            indices,
            table,
            requests,
            out,
            lse,
            INT8.name,
            1.0,
            1.0,
            sk,
            sv,
            local_width=1026,
        )

    context = ForwardContext(
        {},
        {},
        {},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        additional_kwargs={"qsa_has_prefill": True},
    )
    with override_forward_context(context):
        assert call() == rows and calls == ["rows"]
    assert torch.all(out == 1) and torch.all(lse == 2)

    def grouped(query, _k, _v, ids, *args, **kwargs):
        assert query.shape[0] == rows - 1 and ids.shape == (rows - 1, 2051)
        calls.append("grouped")

    monkeypatch.setattr(reader, "native_grouped", grouped)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with override_forward_context(context):
        assert call() == rows - 1 and calls == ["rows", "grouped"]

    # A PIECEWISE batch can have one decode row per request while this split
    # op runs outside stream capture.
    table = table.repeat(rows, 1)
    requests = torch.arange(rows, dtype=torch.int32)
    context.additional_kwargs["qsa_has_prefill"] = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with override_forward_context(context):
        assert call() == rows - 1 and calls == ["rows", "grouped", "grouped"]
