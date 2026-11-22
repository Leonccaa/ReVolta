# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta
from pathlib import Path
from time import sleep
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import vllm._custom_ops as ops
import vllm.distributed as distributed
from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.forward_context import set_forward_context
from vllm.models.qwen4_exp.nvidia.ops import qsa

from .test_qsa_index_dcp_shard import _reference_select
from .test_qsa_tp4_push_allgather import _forward_config

pytestmark = pytest.mark.skip_global_cleanup
K = 512
WORLD = 4


@pytest.mark.parametrize(
    ("world", "capability", "decode", "capture", "mode", "expected"),
    [
        (4, 70, True, True, "native", "fused"),
        (4, 70, False, True, "native", "original"),
        (4, 70, True, False, "native", "original"),
        (2, 70, True, True, "native", "original"),
        (1, 70, True, True, "native", "local"),
        (4, 80, True, True, "native", "local"),
        (4, 70, True, True, "owner", "original"),
        (4, 70, True, True, "overlay", "original"),
        (4, 70, True, True, "debug", "original"),
        (4, 70, True, True, "reference", "original"),
    ],
)
def test_qsa_decode_fusion_at_selector_entry(
    monkeypatch, world, capability, decode, capture, mode, expected
):
    # Give both prefill and decode the same shapes. Only the stage, capture,
    # and the native op availability may admit a TP4 exact decode chunk.
    rows = 128 if mode == "owner" else 8
    scores = torch.zeros(rows, K)
    lengths = torch.full((rows,), K, dtype=torch.int32)
    calls = []

    def fused(logits, visible, blocks, rank, v154, decode_batch):
        assert logits is scores and visible is lengths and rank == 0
        calls.append("fused")

    def original(blocks, logits, visible, *args, owner_merge, **kwargs):
        assert owner_merge == (mode == "owner")
        if capability != 70:
            return blocks
        calls.append("original")
        return blocks

    wheel = SimpleNamespace(
        qsa_lexicographic_topk=_reference_select,
        persistent_topk=lambda *args: calls.append("local"),
    )
    monkeypatch.setattr(torch.ops, "_C", wheel)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)
    monkeypatch.setattr(
        qsa.current_platform, "is_device_capability", lambda value: capability == value
    )
    monkeypatch.setattr(
        qsa, "_SM70_QSA_TOPK_LIBRARY", "test" if mode == "overlay" else None
    )
    monkeypatch.setattr(qsa, "_use_sm70_qsa_indexer_cublas", lambda *args: False)
    monkeypatch.setattr(qsa, "_use_sm70_qsa_request_segments", lambda *args: False)
    monkeypatch.setattr(qsa, "_select_and_merge_qsa_shard_decode", fused)
    monkeypatch.setattr(qsa, "_merge_sharded_index_blocks", original)
    monkeypatch.setattr(
        qsa,
        "_sm70_qsa_lexicographic_topk_op",
        lambda: lambda *args: calls.append("local") if world == 1 else None,
    )
    monkeypatch.setattr(qsa, "qsa_mqa_paged", lambda *args, **kwargs: (scores, lengths))
    monkeypatch.setattr(qsa, "expand_qsa_block_indices_cuda", lambda *args: None)
    monkeypatch.setenv("VLLM_QSA_INDEX_SHARD_DEBUG", str(int(mode == "debug")))
    monkeypatch.setenv(
        "VLLM_QSA_INDEX_SHARD_FUSED_MERGE", str(int(mode != "reference"))
    )

    with set_forward_context(None, _forward_config(), is_uniform_decode=decode):
        qsa.qsa_select_paged_tokens(
            torch.empty(rows, 1, 128),
            torch.empty(8, 64, 1, 128),
            torch.zeros(1, 8, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
            torch.arange(rows),
            torch.tensor([rows]),
            K * 4,
            4,
            index_dcp_world_size=world,
            prefill_ranges=((0, rows),) if mode == "owner" else (),
        )
    assert calls == [expected]


@pytest.mark.parametrize(
    ("rows", "registered", "expected_route"),
    [(2, True, "push"), (2, False, "nccl"), (81, True, "nccl")],
)
def test_qsa_fusion_transport_routing(monkeypatch, rows, registered, expected_route):
    ca = object.__new__(CustomAllreduce)
    ca.disabled = False
    ca.world_size = WORLD
    ca.sm70_tp4_push_buffer_ptrs = [1] * WORLD if registered else None
    ca._ptr = 1
    ca.close = lambda: None
    ranks = torch.arange(WORLD, dtype=torch.int32).view(-1, 1, 1)
    scores = torch.arange(K, dtype=torch.float32).expand(rows, -1)
    lengths = torch.full((rows,), K, dtype=torch.int32)
    blocks = torch.arange(K, dtype=torch.int32).repeat(rows, 1)
    blocks[0].fill_(-1)
    blocks[1, 127:] = -1
    local_pack = qsa._pack_index_shard_candidates(blocks[:2], scores[:2], WORLD, 0)
    pads = local_pack[..., K:] == qsa._INDEX_SHARD_PAD_ID
    first_packs = torch.cat(
        [
            local_pack[..., :K].expand(WORLD, -1, -1),
            torch.where(pads, qsa._INDEX_SHARD_PAD_ID, local_pack[..., K:] + ranks),
        ],
        dim=-1,
    )
    packs = first_packs.new_full((WORLD, rows, 2 * K), qsa._INDEX_SHARD_PAD_ID)
    packs[..., :K].fill_(qsa._INDEX_SHARD_PAD_SCORE_BITS)
    packs[:, :2].copy_(first_packs)
    calls = []

    def pack(logits, lens, out, rank, v154, decode_batch):
        assert (
            logits is scores and lens is lengths and rank == 0 and v154 and decode_batch
        )
        out.copy_(packs[0])

    def gather(inp, dim):
        calls.append("nccl")
        assert dim == 0 and torch.equal(inp[0], packs[0])
        return packs

    def merge(inp, out, fa=0):
        calls.append("push" if fa else "consume")
        assert fa in (0, ca._ptr)
        assert torch.equal(inp, packs[0] if fa else packs)
        sorted_scores, groups, counts = qsa._sort_index_shard_candidates(first_packs)
        positions = torch.empty_like(out[:2])
        _reference_select(sorted_scores, counts, positions, K)
        out.fill_(-1)
        out[:2].copy_(qsa._map_index_shard_positions(positions, groups))

    monkeypatch.setattr(
        distributed,
        "get_tp_group",
        lambda: SimpleNamespace(
            all_gather=gather, device_communicator=SimpleNamespace(ca_comm=ca)
        ),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(ops, "qsa_lexicographic_topk_pack", pack)
    monkeypatch.setattr(ops, "sm70_qsa_shard_merge", merge)
    out = torch.empty_like(blocks)
    qsa._select_and_merge_qsa_shard_decode(scores, lengths, out, 0, True, True)
    assert calls == (["push"] if expected_route == "push" else ["nccl", "consume"])
    assert (out[0] == -1).all()
    assert torch.equal(out[1, :508], torch.arange(508, dtype=torch.int32))
    assert (out[1, 508:] == -1).all()


def _cuda_scores(rows: int, columns: int, rank: int, kind: str):
    # Use a nonzero storage offset and a row stride beyond the logical width.
    scores = torch.empty((rows, columns + 7), device="cuda")[:, 1 : columns + 1]
    if kind == "normal":
        scores.normal_()
    else:
        scores.random_(8)
    if kind == "special":
        scores.view(torch.int32)[:, :8] = torch.tensor(
            [
                -(1 << 31),
                0,
                0x7F800000,
                -0x800000,
                0x7FC00001,
                0x7F800001,
                -0x7FFFFF,
                0x7F7F7F7F,
            ],
            dtype=torch.int32,
            device="cuda",
        )
    # Globally 0, 508, 512, 513 and more than 512 visible groups.
    valid = torch.tensor([columns * WORLD, 0, 508, 512, 513, 2049], device="cuda")
    lengths = (valid[torch.arange(rows, device="cuda") % len(valid)] + 3 - rank) // 4
    return scores, lengths.clamp_max(columns).to(torch.int32)


def _cuda_local_reference(scores, lengths, rank, v154, decode_batch):
    blocks = torch.empty((scores.shape[0], K), device="cuda", dtype=torch.int32)
    select = (
        ops.qsa_lexicographic_topk_v154 if v154 else torch.ops._C.qsa_lexicographic_topk
    )
    select(scores, lengths, blocks, K, decode_batch)
    return blocks, qsa._pack_index_shard_candidates(blocks, scores, WORLD, rank)


def _require_sm70_d3(num_devices):
    if any(torch.cuda.get_device_capability(i) != (7, 0) for i in range(num_devices)):
        pytest.skip("needs SM70 GPUs")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    ("rows", "columns", "v154", "kind"),
    [
        (1, 514, False, "special"),
        (5, 4097, False, "ties"),
        (8, 777, False, "normal"),
        (16, 6001, False, "special"),
        (24, 6001, True, "special"),
        (40, 32771, True, "normal"),
        (64, 65537, True, "ties"),
        (81, 6001, False, "special"),
    ],
)
def test_qsa_native_pack_and_merge_matches_d1(rows, columns, v154, kind):
    _require_sm70_d3(1)
    torch.manual_seed(rows)
    inputs = [_cuda_scores(rows, columns, rank, kind) for rank in range(WORLD)]
    refs = [
        _cuda_local_reference(scores, lengths, rank, v154, rows in (5, 10) or v154)
        for rank, (scores, lengths) in enumerate(inputs)
    ]
    gathered = torch.stack([pack for _, pack in refs])
    out = torch.empty_like(refs[0][0])
    with sm70_decode_graph_compilation(True):
        reference = qsa._merge_sharded_index_blocks(
            refs[0][0],
            *inputs[0],
            WORLD,
            0,
            K,
            all_gather=lambda _: gathered,
            use_triton=True,
        )
    packs = torch.empty_like(gathered)
    ops.sm70_qsa_shard_merge(gathered, out)
    ops.qsa_lexicographic_topk_pack(
        *inputs[0], packs[0], 0, v154, rows in (5, 10) or v154
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for rank, (scores, lengths) in enumerate(inputs):
            ops.qsa_lexicographic_topk_pack(
                scores, lengths, packs[rank], rank, v154, rows in (5, 10) or v154
            )
        ops.sm70_qsa_shard_merge(packs, out)
    graph.replay()
    assert torch.equal(packs, gathered)
    assert torch.equal(out, reference)


def _tp4_fusion_worker(rank: int, rendezvous: str) -> None:
    torch.accelerator.set_device_index(rank)
    torch.manual_seed(42 + rank)
    timeout = timedelta(seconds=60)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=WORLD, timeout=timeout
    )
    nccl = dist.new_group(backend="nccl", timeout=timeout)
    group = dist.group.WORLD
    assert group is not None
    ca = CustomAllreduce(group, rank)
    try:
        dist.all_reduce(torch.zeros(1, device="cuda"), group=nccl)
        torch.accelerator.synchronize()
        fixtures = {}
        for rows in (8, 16, 24, 40, 64, 80, 81):
            scores, lengths = _cuda_scores(rows, 6001, rank, "special")
            blocks, packed = _cuda_local_reference(scores, lengths, rank, True, True)
            gathered = torch.empty(
                (WORLD, *packed.shape), dtype=packed.dtype, device="cuda"
            )
            nccl_out = torch.empty_like(blocks)
            push_out = torch.empty_like(blocks)
            dist.all_gather_into_tensor(gathered.view(-1), packed.view(-1), group=nccl)
            ops.sm70_qsa_shard_merge(gathered, nccl_out)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ops.qsa_lexicographic_topk_pack(
                    scores, lengths, packed, rank, True, True
                )
                used_push = ca.try_sm70_qsa_shard_merge(packed, push_out)
                assert used_push == (
                    rows <= 80 and ca.sm70_tp4_push_buffer_ptrs is not None
                )
                # Mix the two consumers. M24 advances once, so later replays
                # also open the other mailbox page for the same tile.
                copied = (
                    ca.try_sm70_tp4_push_allgather(packed)
                    if used_push and rows != 24
                    else None
                )
                dist.all_gather_into_tensor(
                    gathered.view(-1), packed.view(-1), group=nccl
                )
                ops.sm70_qsa_shard_merge(gathered, nccl_out)
            fixtures[rows] = (
                scores,
                lengths,
                packed,
                gathered,
                nccl_out,
                push_out if used_push else None,
                graph,
                copied,
            )
        for step, rows in enumerate((80, 8, 40, 81, 16, 24, 64, 8, 80)):
            scores, lengths, _, gathered, nccl_out, push_out, graph, copied = fixtures[
                rows
            ]
            scores[:, 8:].random_(16)
            if rank == step % WORLD:
                sleep(0.02)
            graph.replay()
            blocks, expected_pack = _cuda_local_reference(
                scores, lengths, rank, True, True
            )
            reference_packs = torch.empty_like(gathered)
            dist.all_gather_into_tensor(
                reference_packs.view(-1), expected_pack.view(-1), group=nccl
            )
            with sm70_decode_graph_compilation(True):
                expected = qsa._merge_sharded_index_blocks(
                    blocks,
                    scores,
                    lengths,
                    WORLD,
                    rank,
                    K,
                    all_gather=lambda _, value=reference_packs: value,
                    use_triton=True,
                )
            assert torch.equal(gathered, reference_packs), (rank, step, rows)
            assert torch.equal(nccl_out, expected), (rank, step, rows)
            if push_out is not None:
                assert torch.equal(push_out, expected), (rank, step, rows)
            if copied is not None:
                assert torch.equal(copied, gathered), (rank, step, rows)
    finally:
        ca.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < WORLD, reason="needs 4 GPUs")
def test_qsa_native_push_fusion_replays_both_consumers(tmp_path: Path) -> None:
    _require_sm70_d3(WORLD)
    torch.multiprocessing.spawn(
        _tp4_fusion_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=WORLD
    )
