# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa, qsa_shared_key

requires_sm70 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70 CUDA",
)


def test_shared_key_dispatch_keeps_single_request_shape(monkeypatch):
    calls = []

    def record(name, positions, lengths):
        assert positions.tolist() == [15, 15, 15, 15, 19, 19, 19, -1]
        assert lengths.tolist() == [20]
        calls.append(name)

    indexer = SimpleNamespace(run_batch=lambda *args: record("batch", args[6], args[7]))
    monkeypatch.setattr(torch.ops, "vllm_sm70_qsa_indexer", indexer)
    monkeypatch.setattr(qsa_shared_key, "load_operator", lambda: True)
    monkeypatch.setattr(
        qsa_shared_key.current_platform, "is_device_capability", lambda cc: cc == 70
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    q = torch.zeros(8, 4, 128, dtype=torch.float16)
    cache = torch.zeros(1, 32, 1, 128, dtype=torch.float16)
    table = torch.zeros(1, 1, dtype=torch.int32)
    req = torch.zeros(8, dtype=torch.int32)
    pos = torch.tensor([35, 36, 37, 38, 39, 40, 41, -1])
    lens = torch.tensor([42], dtype=torch.int32)

    qsa.qsa_mqa_paged(
        q,
        cache,
        table,
        req,
        pos,
        lens,
        4,
        shared_key_scoring=True,
        index_dcp_world_size=2,
        index_dcp_rank=1,
    )
    assert (
        qsa_shared_key.maybe_shared_key_scores(
            q,
            cache,
            table.expand(2, 1),
            req,
            pos,
            lens.expand(2),
            4,
            32,
            math.sqrt(128),
        )
        is None
    )
    assert calls == ["batch"]


@pytest.mark.parametrize(
    "rows,enabled,overlay,expected",
    [
        (1, True, False, "legacy"),
        (5, True, False, "batch"),
        (8, True, False, "batch"),
        (16, True, False, "batch"),
        (24, True, False, "batch"),
        (40, True, False, "batch"),
        (64, True, False, "batch"),
        (65, True, False, "legacy"),
        (8, False, False, "legacy"),
        (8, True, True, "overlay"),
    ],
)
def test_paged_topk_changes_only_local_selection(
    monkeypatch, rows, enabled, overlay, expected
):
    events = []
    logits = torch.zeros(rows, 2048)
    visible = torch.full((rows,), 520, dtype=torch.int32)

    def local(*_):
        events.append("legacy")

    sidecar = SimpleNamespace()
    if overlay:
        sidecar.qsa_lexicographic_topk = lambda *_: events.append("overlay")

    def select(new_scores, new_visible, blocks, k, decode_batch):
        assert new_scores is logits and new_visible is visible
        assert blocks.shape == (rows, 512)
        assert k == 512 and decode_batch
        events.append("batch")

    ops = SimpleNamespace(
        qsa_lexicographic_topk=local, qsa_lexicographic_topk_v154=select
    )

    def merge(blocks, scores, lengths, world, rank, k, *, row_keys):
        assert scores is logits and lengths is visible
        assert (world, rank, k) == (2, 1, 512)
        assert all(key.shape == (rows,) for key in row_keys)
        events.append("merge")
        return blocks

    monkeypatch.setattr(torch, "ops", SimpleNamespace(_C=ops, _C_qsa_sm70=sidecar))
    monkeypatch.setattr(qsa.envs, "VLLM_SM70_QSA_COMPACT_TOPK", enabled)
    monkeypatch.setattr(qsa, "_SM70_QSA_MTP_TOPK", True)
    monkeypatch.setattr(qsa, "_SM70_QSA_TOPK_LIBRARY", "overlay" if overlay else None)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_lexicographic_topk", lambda k: k == 512)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_indexer_cublas", lambda *_: False)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_request_segments", lambda *_: False)
    monkeypatch.setattr(qsa, "qsa_mqa_paged", lambda *_, **__: (logits, visible))
    monkeypatch.setattr(qsa, "_merge_sharded_index_blocks", merge)
    monkeypatch.setattr(
        qsa, "expand_qsa_block_indices_cuda", lambda *_: events.append("expand")
    )
    qsa.qsa_select_paged_tokens(
        torch.zeros(rows, 4, 128),
        torch.empty(64, 32, 1, 128),
        torch.arange(64, dtype=torch.int32)[None],
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows),
        torch.tensor([4160], dtype=torch.int32),
        2048,
        4,
        index_dcp_world_size=2,
        index_dcp_rank=1,
    )
    assert events == [expected, "merge", "expand"]


@requires_sm70
@pytest.mark.parametrize("rows,rank", [(5, 0), (8, 0), (8, 1)])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_shared_key_matches_legacy_dcp_bits_in_graph(rows, rank, pos_dtype):
    assert qsa_shared_key.load_operator()
    assert hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_batch")
    torch.manual_seed(142)
    dev = "cuda"
    cache = torch.randn(512, 32, 1, 128, device=dev, dtype=torch.float16)
    table = torch.randperm(512, device=dev, dtype=torch.int32)[None]
    table[0, 37] = -1
    q = torch.randn(rows, 4, 128, device=dev, dtype=torch.float16)
    req = torch.zeros(rows, device=dev, dtype=torch.int32)
    lens = torch.tensor([128 * 1024], device=dev, dtype=torch.int32)
    pos = torch.arange(rows, device=dev, dtype=pos_dtype) + lens[0] - rows
    width = cache.shape[0] * cache.shape[1]
    old_logits = torch.empty(rows, width, device=dev)
    old_visible = torch.empty_like(req)

    def candidate():
        return qsa.qsa_mqa_paged(
            q,
            cache,
            table,
            req,
            pos,
            lens,
            4,
            shared_key_scoring=True,
            index_dcp_world_size=2,
            index_dcp_rank=rank,
        )

    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new_logits, new_visible = candidate()
    for length in (14, 8199, 128 * 1024):
        q.normal_()
        lens.fill_(length)
        pos.copy_(torch.arange(length - rows, length, device=dev))
        pos[-1] = -1  # MRv2 padding keeps owner 0 and sets logical position to -1.
        local_pos, local_lens = qsa._shared_key_local_inputs(pos, lens, 4, 2, rank)
        torch.ops.vllm_sm70_qsa_indexer.run(
            old_logits,
            old_visible,
            q,
            cache,
            table,
            req,
            local_pos,
            local_lens,
            math.sqrt(128),
            4,
        )
        graph.replay()
        assert torch.equal(new_visible, old_visible)
        mask = torch.arange(width, device=dev)[None] < old_visible[:, None]
        assert torch.equal(
            new_logits.view(torch.int32)[mask], old_logits.view(torch.int32)[mask]
        )


@requires_sm70
@pytest.mark.parametrize(
    "rows,width",
    [
        (5, 2047),
        (8, 2048),
        (10, 9216),
        (16, 9217),
        (24, 32767),
        (40, 32768),
        (64, 65600),
    ],
)
def test_compact_topk_matches_legacy_order_in_graph(rows, width):
    torch.manual_seed(512)
    logits = torch.empty(rows, width + 7, device="cuda")[:, :width]
    length = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    candidate = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    control = torch.empty_like(candidate)
    new = torch.ops._C.qsa_lexicographic_topk_v154
    old = torch.ops._C.qsa_lexicographic_topk

    logits.normal_()
    new(logits, length, candidate, 512, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new(logits, length, candidate, 512, True)
    for live_width in (512, 513, 2047, 2048, 3584, 3585, width + 5):
        length.copy_(live_width - torch.arange(rows, device="cuda"))
        logits.normal_()
        if live_width > 4096:
            # Mixed probe, but the pivot bucket exceeds the compact capacity.
            logits[0].uniform_(1.1, 1.2)
            logits[0, ::7] = 0.25
        logits[1::3].zero_()
        logits[1::3, ::2] = -0.0
        logits[::3, ::41] = -float("inf")
        logits[2::3, ::39] = float("nan")
        graph.replay()
        old(logits, length, control, 512, rows in (5, 10))
        assert torch.equal(candidate, control)
