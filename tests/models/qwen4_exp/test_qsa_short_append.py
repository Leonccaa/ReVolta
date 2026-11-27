# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import envs
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

pytestmark = pytest.mark.skip_global_cleanup


def _setup(monkeypatch, enabled):
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_CUBLAS", True)
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_CUBLAS_MIN_ROWS", 512)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_SHORT_APPEND", enabled)
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )


def _single(rows):
    q = torch.zeros(rows, 4, 128, dtype=torch.float16)
    cache = torch.zeros(2, 64, 1, 128, dtype=torch.float16)
    return q, cache, torch.zeros(1, 2, dtype=torch.int32)


@pytest.mark.parametrize(
    ("enabled", "rows", "columns", "expected"),
    [
        (False, 400, 4096, False),
        (True, 400, 2048, False),
        (True, 127, 4096, False),
        (True, 128, 4096, True),
        (True, 400, 4096, True),
    ],
)
def test_single_request_row_floor(monkeypatch, enabled, rows, columns, expected):
    _setup(monkeypatch, enabled)
    assert qsa_ops._use_sm70_qsa_indexer_cublas(*_single(rows), columns) is expected


@pytest.mark.parametrize("override", [None, "0"], ids=["default", "disabled"])
def test_tp4_mixed_warm_append_selects_scorer(monkeypatch, override):
    name = "VLLM_SM70_QSA_SHORT_APPEND"
    if override is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, override)
    # The QSA module reads the registered value at import time.
    _setup(monkeypatch, envs.environment_variables[name]())
    monkeypatch.setattr(
        qsa_ops.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(
        qsa_ops, "_qsa_visible_blocks", lambda req, pos, *a, **kw: torch.empty_like(pos)
    )
    monkeypatch.setattr(qsa_ops.ops, "qsa_lexicographic_topk_v154", lambda *a: None)
    monkeypatch.setattr(
        qsa_ops, "_sm70_qsa_lexicographic_topk_op", lambda: lambda *a: None
    )
    monkeypatch.setattr(
        qsa_ops, "_merge_sharded_index_blocks", lambda blocks, *a, **kw: blocks
    )
    monkeypatch.setattr(qsa_ops, "expand_qsa_block_indices_cuda", lambda *a: None)

    # Eight verification rows and a 400-row append to a 32K prefix. A 2048-token
    # page holds 128 local keys at compression 4 / TP4. Decode is longer than
    # the append, so cuBLAS must gather for the append's own host length.
    q = torch.empty(408, 4, 128, dtype=torch.float16)
    cache = torch.empty(35, 128, 1, 128, dtype=torch.float16)
    table = torch.arange(36, dtype=torch.int32).view(2, 18)
    table[1, -1] = -1
    lens = torch.tensor([36864, 33168], dtype=torch.int32)
    pos = torch.cat((torch.arange(36856, 36864), torch.arange(32768, 33168))).int()
    req = torch.repeat_interleave(
        torch.arange(2, dtype=torch.int32), torch.tensor([8, 400])
    )
    calls: list[tuple[str | int, ...]] = []

    def gather(k_cache, page_table, columns):
        torch.testing.assert_close(page_table, table[1:])
        calls.append(("gather", columns))
        return k_cache.new_empty((columns, 128)), torch.empty(
            columns, dtype=torch.uint8
        )

    def cublas(query, keys, key_valid, visible):
        calls.append(("cublas", query.shape[0], keys.shape[0]))
        return torch.empty(query.shape[0], keys.shape[0]), visible

    def paged(query, k_cache, page_table, *args, num_columns, **kwargs):
        calls.append(("paged", query.shape[0], page_table.shape[0], num_columns))
        return torch.empty(query.shape[0], num_columns), torch.empty(
            query.shape[0], dtype=torch.int32
        )

    monkeypatch.setattr(qsa_ops, "_qsa_gather_single_request_keys", gather)
    monkeypatch.setattr(qsa_ops, "_qsa_mqa_cublas", cublas)
    monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", paged)
    qsa_ops.qsa_select_paged_tokens(
        q,
        cache,
        table,
        req,
        pos,
        lens,
        2048,
        4,
        query_start_loc_cpu=torch.tensor([0, 8, 408]),
        max_seq_len=36864,
        seq_lens_cpu_upper_bound=lens,
        index_dcp_world_size=4,
        prefill_ranges=((8, 408),),
    )
    if override is None:
        # The row floor uses DCP2's 4146 route columns, not TP4's 2073 local keys.
        assert calls == [("paged", 8, 2, 2304), ("gather", 2073), ("cublas", 400, 2073)]
    else:
        assert calls == [("paged", 408, 2, 2304)]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("max_seq_len", [8192, 16384])
def test_multi_request_segment_floor(monkeypatch, enabled, max_seq_len):
    # 8 decode rows + a 400-row warm append.
    _setup(monkeypatch, enabled)
    q = torch.zeros(408, 4, 128, dtype=torch.float16)
    cache = torch.zeros(8, 64, 1, 128, dtype=torch.float16)
    table = torch.arange(9 * 2, dtype=torch.int32).view(-1, 2)
    start_loc = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 408])
    token_to_req = torch.repeat_interleave(
        torch.arange(9, dtype=torch.int32), torch.diff(start_loc)
    )
    pos = torch.arange(408, dtype=torch.int32)
    lens = torch.full((9,), 40000, dtype=torch.int32)
    out = torch.empty(408, 2051, dtype=torch.int32)
    calls = []
    monkeypatch.setattr(
        qsa_ops,
        "qsa_select_paged_tokens",
        lambda q_, *a, **kw: calls.append(q_.shape[0]),
    )
    handled = qsa_ops._qsa_select_by_request(
        q,
        cache,
        table,
        token_to_req,
        pos,
        lens,
        2048,
        4,
        out,
        start_loc,
        max_seq_len,
    )
    assert handled == (enabled and max_seq_len >= 16384)
    if handled:
        assert calls == [8, 400]
