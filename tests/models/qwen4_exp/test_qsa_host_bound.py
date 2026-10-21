# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU launch contracts; GPU score and top-k parity is a separate gate."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.nvidia.ops import qsa


@pytest.mark.parametrize(
    "bound,expected", [(None, None), ([], None), ([9, 8192, 0], 8192)]
)
def test_host_upper_bound(bound, expected):
    helper = getattr(qsa_cache, "_host_max_seq_len", None)
    assert helper is not None
    metadata = NS(
        seq_lens_cpu_upper_bound=None if bound is None else torch.tensor(bound)
    )
    assert helper(metadata) == expected


@pytest.mark.parametrize("split", [False, True])
def test_cublas_bound_never_reads_visible_tensor(monkeypatch, split):
    # Exercise the real entry and its recursive mixed-request split. Stop
    # immediately before the first CUDA key gather; no CUDA execution is faked.
    class NoHostRead:
        def max(self):
            raise AssertionError("device metadata read on host")

    class ReachedGather(Exception):
        pass

    monkeypatch.setattr(qsa, "_SM70_INDEXER_CUBLAS", True)
    monkeypatch.setattr(qsa.current_platform, "is_device_capability", lambda *_: True)
    monkeypatch.setattr(qsa, "_qsa_visible_blocks", lambda *_: NoHostRead())
    monkeypatch.setattr(qsa, "_qsa_indexer_cublas_work_supported", lambda *_: True)
    columns = []

    def gather(cache, table, width):
        columns.append(width)
        raise ReachedGather

    monkeypatch.setattr(qsa, "_qsa_gather_single_request_keys", gather)
    lengths = [5, 1024] if split else [1024]
    n = len(lengths)
    rows = sum(lengths)
    q = torch.zeros(rows, 4, 128, dtype=torch.float16)
    cache = torch.zeros(4, 64, 1, 128, dtype=torch.float16)
    table = torch.zeros(n, 200, dtype=torch.int32)
    # Short run never takes the cuBLAS path. Replace its paged scorer only
    # at the recursive call, leaving the long-request selection entry real.
    real_select = qsa.qsa_select_paged_tokens

    def select(*args, **kwargs):
        if args[0].shape[0] < 512:
            return args[8]
        return real_select(*args, **kwargs)

    monkeypatch.setattr(qsa, "qsa_select_paged_tokens", select)
    bounds = torch.tensor([131072, 8192] if split else [8192])
    with pytest.raises(ReachedGather):
        real_select(
            q,
            cache,
            table,
            torch.zeros(rows, dtype=torch.int32),
            torch.arange(rows),
            bounds,
            2048,
            4,
            query_start_loc_cpu=torch.tensor([0, 5, rows]) if split else None,
            max_seq_len=int(bounds.max()),
            seq_lens_cpu_upper_bound=bounds,
        )
    assert columns == [2048]
