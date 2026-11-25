# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of when the multi-request QSA scorer is used (no GPU, no launch)."""

from types import SimpleNamespace as NS

import pytest
import torch

import vllm.envs as envs
from vllm.forward_context import set_forward_context
from vllm.models.qwen4_exp.nvidia import indexer_qsa
from vllm.models.qwen4_exp.nvidia.ops import qsa, qsa_shared_key
from vllm.models.qwen4_exp.nvidia.ops.qsa_request_segments import (
    request_segments_unsupported,
)

pytestmark = pytest.mark.skip_global_cleanup


def batch(requests=4, rows_per_request=4, pages=8, page_size=416):
    rows = requests * rows_per_request
    q = torch.zeros((rows, 4, 128), dtype=torch.float16)
    cache = torch.zeros((pages * requests, page_size, 1, 128), dtype=torch.float16)
    table = torch.arange(pages * requests, dtype=torch.int32).reshape(requests, pages)
    t2r = torch.arange(requests, dtype=torch.int32).repeat_interleave(rows_per_request)
    pos = torch.arange(rows, dtype=torch.int64) + 1000
    lens = torch.full((requests,), 2000, dtype=torch.int32)
    return [q, cache, table, t2r, pos, lens]


def test_supported_decode_batch():
    assert request_segments_unsupported(*batch()) is None
    assert request_segments_unsupported(*batch(requests=8, rows_per_request=5)) is None
    x = batch()
    x[4] = x[4].int()
    assert request_segments_unsupported(*x) is None


@pytest.mark.parametrize(
    "change, reason",
    [
        (
            lambda x: x.__setitem__(2, x[2][:1]) or x.__setitem__(5, x[5][:1]),
            "single_request",
        ),
        (lambda x: x.__setitem__(0, x[0].float()), "dtype"),
        (lambda x: x.__setitem__(1, x[1].float()), "dtype"),
        (
            lambda x: x.__setitem__(0, torch.zeros((16, 8, 128), dtype=torch.float16)),
            "query_shape",
        ),
        (
            lambda x: x.__setitem__(
                1, torch.zeros((32, 416, 1, 64), dtype=torch.float16)
            ),
            "cache_shape",
        ),
        (lambda x: x.__setitem__(3, x[3].long()), "metadata_dtype"),
        (lambda x: x.__setitem__(2, x[2].long()), "metadata_dtype"),
        (lambda x: x.__setitem__(4, x[4].float()), "position_dtype"),
        (lambda x: x.__setitem__(3, x[3][:-1]), "row_metadata_shape"),
        (lambda x: x.__setitem__(5, x[5][:-1]), "lengths_shape"),
    ],
)
def test_fallback_reasons(change, reason):
    x = batch()
    change(x)
    assert request_segments_unsupported(*x) == reason


@pytest.mark.parametrize("requests", [2, 3, 4, 8])
def test_one_row_per_request_keeps_paged(requests):
    # MTP drafting steps: one row per request, plus CUDA-graph padding rows
    # (position -1) up to the next graph size.
    x = batch(requests=requests, rows_per_request=1)
    assert request_segments_unsupported(*x) == "one_row_per_request"
    pad = requests - 1
    x[0] = torch.cat([x[0], torch.zeros((pad, 4, 128), dtype=torch.float16)])
    x[3] = torch.cat([x[3], torch.zeros(pad, dtype=torch.int32)])
    x[4] = torch.cat([x[4], torch.full((pad,), -1, dtype=torch.int64)])
    assert request_segments_unsupported(*x) == "one_row_per_request"
    assert request_segments_unsupported(*batch(requests, rows_per_request=2)) is None


def test_row_limit():
    assert (
        request_segments_unsupported(*batch(requests=64, rows_per_request=4)) is None
    )  # 256 rows
    assert (
        request_segments_unsupported(*batch(requests=65, rows_per_request=4)) == "rows"
    )


def test_flag_and_device_gate(monkeypatch):
    x = batch()
    monkeypatch.setattr(qsa, "_on_sm70_cuda", lambda q: True)
    monkeypatch.setattr(envs, "VLLM_SM70_QSA_REQUEST_SEGMENTS", False, raising=False)
    assert qsa._use_sm70_qsa_request_segments(*x) is False
    monkeypatch.setattr(envs, "VLLM_SM70_QSA_REQUEST_SEGMENTS", True, raising=False)
    assert qsa._use_sm70_qsa_request_segments(*x) is True
    single = batch(requests=1)
    assert qsa._use_sm70_qsa_request_segments(*single) is False
    monkeypatch.setattr(qsa, "_on_sm70_cuda", lambda q: False)
    assert qsa._use_sm70_qsa_request_segments(*x) is False


@pytest.mark.parametrize(
    "uniform,world,expected",
    [(True, 1, "batched"), (False, 1, "segment"), (True, 2, "segment")],
)
def test_indexer_decode_score_routing(monkeypatch, uniform, world, expected):
    # Exercise the actual indexer -> selection -> batched admission path. Only
    # projection, cache update, CUDA launches, and the DCP merge are substitutes.
    rows, requests = 48, 8
    starts = torch.tensor(
        list(range(0, 49, 6)) if uniform else [0, 4, 12, 18, 24, 30, 36, 42, 48],
        dtype=torch.int32,
    )
    counts = starts[1:] - starts[:-1]
    token_to_req = torch.arange(requests, dtype=torch.int32).repeat_interleave(counts)
    positions = torch.cat([torch.arange(n, dtype=torch.int64) + 1000 for n in counts])
    table = torch.arange(requests * 2, dtype=torch.int32).reshape(requests, 2)
    metadata = NS(
        num_actual_tokens=rows,
        slot_mapping=torch.arange(rows),
        block_table=table,
        query_start_loc=starts,
        query_start_loc_cpu=None,
        logical_positions=positions,
        token_to_req=token_to_req,
        seq_lens=torch.full((requests,), 2000, dtype=torch.int32),
        k_work_metadata=None,
        max_seq_len=None,
        seq_lens_cpu_upper_bound=None,
        index_dcp_world_size=world,
        index_dcp_rank=0,
    )
    indexer = NS(
        _metadata=lambda: (metadata, metadata),
        index_qk_proj=lambda hidden: (
            torch.zeros(rows, 5 * 128, dtype=torch.float16),
            None,
        ),
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        raw_key_cache=NS(kv_cache=torch.empty(0), rope_position_cache=None),
        compressed_key_cache=NS(
            kv_cache=torch.empty(requests * 2, 208, 1, 128, dtype=torch.float16)
        ),
        use_fused_pre_indexer=True,
        rotary_emb=NS(cos_sin_cache=torch.empty(0)),
        q_layernorm=NS(weight=torch.ones(128), variance_epsilon=1e-6),
        k_layernorm=NS(weight=torch.ones(128)),
        compress_ratio=4,
        token_topk=2048,
        skip_topk=False,
        shared_key_scoring=True,
    )
    calls = []

    def batched(scores, visible, q, cache, page_table, offsets, pos, lengths, *args):
        calls.append("batched")
        assert offsets is starts and page_table is table and pos is positions
        assert lengths is metadata.seq_lens
        visible.zero_()

    def segment(q, cache, page_table, req, pos, lengths, compress, **kwargs):
        calls.append("segment")
        assert torch.equal(req, token_to_req) and torch.equal(pos, positions)
        assert kwargs.get("index_dcp_world_size", 1) == world
        return torch.empty(rows, 416), torch.zeros(rows, dtype=torch.int32)

    monkeypatch.setattr(indexer_qsa, "qsa_pre_indexer", lambda *a, **kw: None)
    monkeypatch.setattr(qsa, "_on_sm70_cuda", lambda q: True)
    monkeypatch.setattr(envs, "VLLM_SM70_QSA_REQUEST_SEGMENTS", True, raising=False)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_indexer_cublas", lambda *a: False)
    monkeypatch.setattr(
        qsa_shared_key, "_load_operators", lambda: (True, batched), raising=False
    )
    monkeypatch.setattr(qsa, "qsa_mqa_request_segments_paged", segment)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_lexicographic_topk", lambda k: True)
    monkeypatch.setattr(qsa, "_sm70_qsa_lexicographic_topk_op", lambda: lambda *a: None)
    monkeypatch.setattr(
        qsa, "_merge_sharded_index_blocks", lambda blocks, *a, **kw: blocks
    )
    monkeypatch.setattr(qsa, "expand_qsa_block_indices_cuda", lambda *a: a[-1])
    config = NS(
        parallel_config=NS(data_parallel_size=1),
        compilation_config=NS(fast_moe_cold_start=False, static_forward_context={}),
    )
    with set_forward_context(None, config, is_uniform_decode=uniform):
        result = indexer_qsa.QSAIndexer.forward(
            indexer, torch.empty(rows, 1), positions
        )
    assert result.shape == (rows, 2051)
    assert calls == [expected]


def test_flag_default_off(monkeypatch):
    # Evaluate the registered reader directly: vllm.envs caches values on first access.
    reader = envs.environment_variables["VLLM_SM70_QSA_REQUEST_SEGMENTS"]
    monkeypatch.delenv("VLLM_SM70_QSA_REQUEST_SEGMENTS", raising=False)
    assert reader() is False
    monkeypatch.setenv("VLLM_SM70_QSA_REQUEST_SEGMENTS", "1")
    assert reader() is True
