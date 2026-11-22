# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch checks for the host-bounded QSA paged scorer."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize(
    "bound,capture",
    [
        (None, False),
        (1, False),
        (8191, False),
        (32771, False),
        (1048576, False),
        (32771, True),
    ],
)
@pytest.mark.parametrize("segmented", [False, True])
def test_width_and_collectives(monkeypatch, world, bound, capture, segmented):
    capacity, rows = 8192, 5
    calls = []
    q = torch.empty(rows, 4, 128)
    cache = torch.empty(1, capacity, 1, 128)
    table = torch.zeros(1, 1, dtype=torch.int32)
    request = torch.zeros(rows, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int32)
    lengths = torch.tensor([32771], dtype=torch.int32)
    monkeypatch.setattr(qsa.torch.cuda, "is_current_stream_capturing", lambda: capture)
    monkeypatch.setattr(qsa, "_LOGITS_WORKSPACE_BYTES", capacity * 4 * 2)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_indexer_cublas", lambda *a: False)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_request_segments", lambda *a: segmented)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_lexicographic_topk", lambda *a: False)
    monkeypatch.setattr(qsa.current_platform, "has_device_capability", lambda *a: False)

    def score(query, *args, **kwargs):
        width = kwargs.get("num_columns", capacity)
        calls.append(("score", query.shape[0], width))
        return torch.zeros(query.shape[0], width), torch.zeros(
            query.shape[0], dtype=torch.int32
        )

    def select(logits, visible, blocks, workspace, k, width):
        assert width == logits.shape[1]
        calls.append(("select", logits.shape[0], width))
        blocks.fill_(-1)

    def merge(blocks, logits, visible, world, rank, k, **kwargs):
        calls.append(("merge", blocks.shape[0], blocks.shape[1]))
        return blocks

    monkeypatch.setattr(qsa, "qsa_mqa_paged", score)
    monkeypatch.setattr(qsa, "qsa_mqa_request_segments_paged", score)
    monkeypatch.setattr(qsa, "_merge_sharded_index_blocks", merge)
    monkeypatch.setattr(qsa, "expand_qsa_block_indices_cuda", lambda *a: None)
    monkeypatch.setattr(
        qsa.torch, "ops", SimpleNamespace(_C=SimpleNamespace(persistent_topk=select))
    )

    traces = []
    for rank in range(world):
        calls.clear()
        qsa.qsa_select_paged_tokens(
            q,
            cache,
            table,
            request,
            positions,
            lengths,
            2048,
            4,
            max_seq_len=bound,
            index_dcp_world_size=world,
            index_dcp_rank=rank,
        )
        traces.append(calls.copy())
    assert all(trace == traces[0] for trace in traces)
    width = capacity
    if bound is not None and not capture and not segmented:
        live = (bound // 4 + world - 1) // world
        width = min(capacity, (max(512, live) + 3) // 4 * 4)
    assert [entry for entry in traces[0] if entry[0] == "score"] == [
        ("score", n, width) for n in (2, 2, 1)
    ]
    assert len([entry for entry in traces[0] if entry[0] == "merge"]) == (
        3 if world > 1 else 0
    )
