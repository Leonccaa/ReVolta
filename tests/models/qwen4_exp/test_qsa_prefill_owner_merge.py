# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.distributed as distributed
from tests.models.qwen4_exp.test_qsa_index_dcp_shard import (
    DRAFT,
    TARGET,
    _config,
    _float_from_bits,
    _Packed,
    _reference_select,
    _side,
)
from tests.models.qwen4_exp.test_qsa_indexer_request_split import _inputs
from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.nvidia.ops import qsa
from vllm.utils import torch_utils
from vllm.v1.attention.backend import CommonAttentionMetadata

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize(("world", "rows"), [(2, 64), (2, 65), (4, 128), (4, 129)])
def test_owner_merge_matches_full_gather(monkeypatch, device, use_triton, world, rows):
    if device == "cuda":
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (7, 0)
            or not hasattr(torch.ops._C, "qsa_lexicographic_topk")
        ):
            pytest.skip("needs V100 and the native exact QSA selector")
        select = None
        local_select = qsa._sm70_qsa_lexicographic_topk_op()
    else:
        if use_triton and os.getenv("TRITON_INTERPRET") != "1":
            pytest.skip("CPU Triton execution needs TRITON_INTERPRET=1")
        select = local_select = _reference_select
    monkeypatch.setenv("VLLM_QSA_INDEX_SHARD_DEBUG", "0")

    # Keep one full-width padded TP4 case in the slower CPU interpreter.
    k = 7 if device == "cpu" and use_triton and rows != 129 else 512
    columns = world * (k + 17) + 3
    rng = torch.Generator().manual_seed(2026)
    scores = torch.randn(rows, columns, generator=rng)
    scores[::2].round_()
    special = _float_from_bits(
        0x80000000, 0, 0x7F800000, 0xFF800000, 0x7FC00123, 0x7F800001, 0xFF800001
    )
    scores.view(torch.int32)[::3, : special.numel()] = special.view(torch.int32)
    scores = scores.to(device)
    lengths = torch.tensor(
        [0, 1, k - 1, k, k + 1, world * k - 1, columns] * rows,
        dtype=torch.int32,
        device=device,
    )[:rows].contiguous()

    captured: list[torch.Tensor] = []

    def capture(packed):
        captured.append(packed)
        raise _Packed

    def merge(rank, gather, owner=False):
        logits = scores[:, rank::world].contiguous()
        visible = (lengths + world - 1 - rank) // world
        local = torch.empty((rows, k), dtype=torch.int32, device=device)
        local_select(logits, visible, local, k)
        return qsa._merge_sharded_index_blocks(
            local,
            logits,
            visible,
            world,
            rank,
            k,
            all_gather=gather,
            select=select,
            use_triton=use_triton,
            owner_merge=owner,
        )

    for rank in range(world):
        with pytest.raises(_Packed):
            merge(rank, capture)
    all_candidates = torch.cat(captured, dim=0)
    baseline = merge(0, lambda _: all_candidates)
    oracle = torch.empty_like(baseline)
    local_select(scores, lengths, oracle, k)
    assert torch.equal(baseline, oracle)

    owner_rows = (rows + world - 1) // world
    pad = owner_rows * world - rows
    padded = F.pad(all_candidates, (0, 0, 0, pad), value=qsa._INDEX_SHARD_PAD_ID)
    padded_result = F.pad(oracle, (0, 0, 0, pad), value=-1)
    device_group = object()
    monkeypatch.setattr(
        distributed, "get_tp_group", lambda: SimpleNamespace(device_group=device_group)
    )

    def check_owner(rank):
        lo, hi = rank * owner_rows, (rank + 1) * owner_rows
        calls = []

        def exchange(recv, send, *, group):
            calls.append("a2a")
            assert group is device_group
            assert send.shape == (world, owner_rows, 2 * k)
            assert send.dtype == torch.int32
            assert torch.equal(send.flatten(0, 1), padded[rank])
            recv.copy_(padded[:, lo:hi])

        def gather(result):
            calls.append("ag")
            assert result.shape == (owner_rows, k) and result.dtype == torch.int32
            assert torch.equal(result, padded_result[lo:hi])
            return padded_result

        monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange)
        assert torch.equal(merge(rank, gather, owner=True), baseline)
        assert calls == ["a2a", "ag"]

    for rank in range(world):
        check_owner(rank)


@pytest.mark.parametrize(
    ("method", "layer"),
    [("dflash", TARGET), ("mtp", TARGET), ("mtp", DRAFT), (None, TARGET)],
)
def test_real_prefill_ranges_from_sharded_metadata(monkeypatch, method, layer):
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(qsa_cache, "_current_tp_rank", lambda: 0)
    monkeypatch.setattr(
        qsa_cache, "build_qsa_metadata", qsa_cache._build_qsa_metadata_torch
    )
    config = _config(True, tp=4)
    config.speculative_config = SimpleNamespace(method=method) if method else None
    config.kernel_config.long_context_kv = SimpleNamespace(enabled=method == "mtp")
    config.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=192, max_num_seqs=5
    )
    starts = torch.tensor([0, 8, 72, 72, 137, 177], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=starts.diff(),
        num_reqs=5,
        num_actual_tokens=177,
        max_query_len=65,
        max_seq_len=65,
        block_table_tensor=torch.arange(5, dtype=torch.int32).view(5, 1),
        slot_mapping=torch.arange(177, dtype=torch.int64),
        is_prefilling=torch.tensor([False, True, False, True, False]),
    )
    builder = qsa_cache.QSAMetadataBuilder(
        _side(layer).get_kv_cache_spec(config),
        [layer],
        config,
        torch.device("cpu"),
        block_table_width=1,
    )
    assert builder.build(0, common).prefill_ranges == ((8, 137),)
    assert builder.build(0, replace(common, is_dummy_batch=True)).prefill_ranges == ()
    assert builder.build(0, replace(common, is_prefilling=None)).prefill_ranges == ()
    decode = replace(common, is_prefilling=torch.zeros(5, dtype=torch.bool))
    assert builder.build(0, decode).prefill_ranges == ()


@pytest.mark.parametrize("split_requests", [False, True])
@pytest.mark.parametrize(("world", "chunk_rows"), [(1, 129), (4, 127), (4, 129)])
def test_prefill_chunk_dispatch(monkeypatch, split_requests, world, chunk_rows):
    starts = [0, 5, 263, 274]
    q, cache, table, owners, positions, lengths = _inputs(starts)
    monkeypatch.setattr(qsa, "_SM70_INDEXER_CUBLAS", split_requests)
    monkeypatch.setattr(qsa, "_SM70_INDEXER_CUBLAS_MIN_ROWS", 128)
    monkeypatch.setattr(qsa, "_SM70_QSA_MTP_TOPK", False)
    monkeypatch.setattr(
        qsa.current_platform, "is_device_capability", lambda cc: cc == 70
    )
    monkeypatch.setattr(qsa, "_use_sm70_qsa_indexer_cublas", lambda *_: False)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_request_segments", lambda *_: False)
    monkeypatch.setattr(qsa, "_sm70_qsa_lexicographic_topk_op", lambda: lambda *_: None)
    monkeypatch.setattr(qsa, "expand_qsa_block_indices_cuda", lambda *_: None)
    monkeypatch.setattr(
        qsa, "_LOGITS_WORKSPACE_BYTES", chunk_rows * table.shape[1] * cache.shape[1] * 4
    )
    monkeypatch.setattr(qsa, "qsa_mqa_paged", lambda query, *_, **__: (query, None))
    calls = []

    def merge(blocks, *_, row_keys, owner_merge):
        calls.append((row_keys[1][0].item(), blocks.shape[0], owner_merge))
        return blocks

    monkeypatch.setattr(qsa, "_merge_sharded_index_blocks", merge)
    qsa.qsa_select_paged_tokens(
        q,
        cache,
        table,
        owners,
        positions,
        lengths,
        2048,
        4,
        query_start_loc_cpu=torch.tensor(starts),
        index_dcp_world_size=world,
        prefill_ranges=((5, 263),),
    )
    if world == 1:
        assert calls == []
    elif chunk_rows == 127:
        if split_requests:
            assert calls == [
                (0, 5, False),
                (5, 127, False),
                (132, 127, False),
                (259, 4, False),
                (263, 11, False),
            ]
        else:
            assert calls == [(0, 127, False), (127, 127, False), (254, 20, False)]
    elif split_requests:
        assert calls == [
            (0, 5, False),
            (5, 129, True),
            (134, 129, True),
            (263, 11, False),
        ]
    else:
        assert calls == [(0, 129, False), (129, 129, True), (258, 16, False)]
