# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of when the multi-request QSA scorer is used (no GPU, no launch)."""

import pytest
import torch

import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia.ops import qsa
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


def test_flag_default_off(monkeypatch):
    # Evaluate the registered reader directly: vllm.envs caches values on first access.
    reader = envs.environment_variables["VLLM_SM70_QSA_REQUEST_SEGMENTS"]
    monkeypatch.delenv("VLLM_SM70_QSA_REQUEST_SEGMENTS", raising=False)
    assert reader() is False
    monkeypatch.setenv("VLLM_SM70_QSA_REQUEST_SEGMENTS", "1")
    assert reader() is True
