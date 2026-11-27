# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

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


def test_tp4_warm_append_row_gate_uses_route_columns(monkeypatch):
    # 32K prefix + 400 rows, ratio 4, TP4: 2073 local columns, but the route
    # width is the DCP2 split (4146), which is what the row floor is judged on.
    _setup(monkeypatch, True)
    route = qsa_ops._qsa_cublas_route_columns(2073, 33168, 4, 4, 512, 1 << 20)
    assert route == 4146
    assert qsa_ops._use_sm70_qsa_indexer_cublas(*_single(400), route)
    assert not qsa_ops._use_sm70_qsa_indexer_cublas(*_single(400), 2073)


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
