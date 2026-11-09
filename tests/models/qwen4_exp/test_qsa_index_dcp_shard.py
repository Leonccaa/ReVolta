# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP sharding of the QSA compressed indexer: spec, slots and page tables."""

import contextlib
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.common.qsa_cache import (
    PAD_SLOT_ID,
    QSACompressedKeyCache,
    QSAMetadataBuilder,
    compressed_qsa_slot_mapping,
    qsa_index_dcp_sharded,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

pytestmark = pytest.mark.skip_global_cleanup

TARGET = "model.layers.3.self_attn.indexer.compressed_key_cache"
DRAFT = "mtp.layers.0.self_attn.indexer.compressed_key_cache"


def _config(shard: bool, block_size: int = 1664, dcp: int = 2):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        kernel_config=SimpleNamespace(qsa_dcp_index_shard=shard),
    )


def _side(prefix: str) -> QSACompressedKeyCache:
    side = object.__new__(QSACompressedKeyCache)
    torch.nn.Module.__init__(side)
    side.prefix = prefix
    side.head_size = 128
    side.dtype = torch.float16
    side.compress_ratio = 4
    return side


def test_flag_off_keeps_replicated_indexer() -> None:
    spec = _side(TARGET).get_kv_cache_spec(_config(False))
    assert spec.block_size == 1664 and not spec.dcp_sharded
    assert spec.storage_block_size == 416
    assert not qsa_index_dcp_sharded(_config(False), TARGET, 4)


def test_sharded_target_indexer_geometry() -> None:
    config = _config(True)
    spec = _side(TARGET).get_kv_cache_spec(config)
    assert spec.dcp_sharded and spec.block_size == 832
    assert spec.global_block_size(2) == 1664
    assert spec.storage_block_size == 208
    replicated = _side(TARGET).get_kv_cache_spec(_config(False))
    assert spec.page_size_bytes * 2 == replicated.page_size_bytes


def test_draft_indexer_stays_replicated() -> None:
    spec = _side(DRAFT).get_kv_cache_spec(_config(True))
    assert spec.block_size == 1664 and not spec.dcp_sharded


def test_sharding_needs_dcp_and_whole_groups_per_rank() -> None:
    assert not qsa_index_dcp_sharded(_config(True, dcp=1), TARGET, 4)
    with pytest.raises(ValueError, match="divisible"):
        qsa_index_dcp_sharded(_config(True, block_size=1668), TARGET, 4)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_slot_mapping_writes_each_group_once(world: int) -> None:
    ratio, groups_per_block = 4, 416
    local_per_block = groups_per_block // world
    generator = torch.Generator().manual_seed(world)
    blocks = 3
    table = torch.randperm(50, generator=generator)[:blocks].to(torch.int32)[None]
    positions = torch.arange(blocks * groups_per_block * ratio, dtype=torch.int64)
    requests = torch.zeros_like(positions, dtype=torch.int32)
    owners: dict[tuple[int, int], int] = {}
    writes_per_group: dict[int, int] = {}
    for rank in range(world):
        slots = compressed_qsa_slot_mapping(
            table,
            requests,
            positions,
            local_per_block,
            ratio,
            index_dcp_world_size=world,
            index_dcp_rank=rank,
        )
        for pos, slot in zip(positions.tolist(), slots.tolist()):
            if slot == PAD_SLOT_ID:
                continue
            assert (pos + 1) % ratio == 0
            group = pos // ratio
            assert group % world == rank
            block, offset = divmod(slot, local_per_block)
            column = table[0].tolist().index(block) * local_per_block + offset
            assert column * world + rank == group
            assert (rank, slot) not in owners
            owners[(rank, slot)] = group
            writes_per_group[group] = writes_per_group.get(group, 0) + 1
    assert sorted(writes_per_group) == list(range(blocks * groups_per_block))
    assert set(writes_per_group.values()) == {1}


def test_slot_mapping_world_one_matches_replicated_layout() -> None:
    table = torch.tensor([[7, 2]], dtype=torch.int32)
    positions = torch.arange(-2, 2 * 416 * 4, dtype=torch.int64)
    requests = torch.zeros_like(positions, dtype=torch.int32)
    sharded = compressed_qsa_slot_mapping(
        table, requests, positions, 416, 4, index_dcp_world_size=1, index_dcp_rank=0
    )
    expected = torch.full_like(positions, PAD_SLOT_ID)
    for i, pos in enumerate(positions.tolist()):
        if pos >= 0 and (pos + 1) % 4 == 0:
            group = pos // 4
            expected[i] = table[0, group // 416] * 416 + group % 416
    assert torch.equal(sharded, expected)


def test_slot_mapping_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        compressed_qsa_slot_mapping(
            torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int64),
            208,
            4,
            index_dcp_world_size=2,
            index_dcp_rank=2,
        )


@pytest.mark.parametrize("sharded", [False, True])
def test_canonical_block_table_does_not_shrink_sharded_page(sharded: bool) -> None:
    # Both layouts cover 832 local tokens of the shared, DCP-sharded table, so
    # both must collapse the same number of 16-token kernel entries per page.
    builder = object.__new__(QSAMetadataBuilder)
    builder.kv_cache_spec = MLAAttentionSpec(
        block_size=832 if sharded else 1664,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        dcp_sharded=sharded,
    )
    builder.kernel_block_size = 16
    builder.dcp_world_size = 2
    builder.reads_sharded_group = True
    builder.block_table_buffer = torch.empty((2, 4), dtype=torch.int32)
    expansion = 832 // 16
    pages = torch.tensor([[3, 9, 4], [6, 1, 8]], dtype=torch.int32)
    table = (pages[:, :, None] * expansion + torch.arange(expansion)).reshape(2, -1)
    canonical = builder._canonical_block_table(table.to(torch.int32))
    assert torch.equal(canonical, pages)


def test_metadata_carries_index_ownership() -> None:
    fields = qsa_cache.QSAForwardMetadata.__dataclass_fields__
    assert fields["index_dcp_world_size"].default == 1
    assert fields["index_dcp_rank"].default == 0


@pytest.mark.parametrize("world", [1, 2])
def test_triton_metadata_matches_torch_with_sharded_indexer(
    world: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from vllm.triton_utils import HAS_TRITON
    from vllm.utils import torch_utils
    from vllm.v1.attention.backend import CommonAttentionMetadata

    if not HAS_TRITON:
        pytest.skip("Triton unavailable")
    if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("Triton metadata needs CUDA or the Triton interpreter")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(qsa_cache, "_metadata_launch_pdl", lambda: False)
    # Two requests: a long prefill chunk and a 3-token verify step.
    q = torch.tensor([0, 37, 40], dtype=torch.int32)
    seq = torch.tensor([1700, 905], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_actual_tokens=40,
        num_reqs=2,
        max_query_len=37,
        max_seq_len=1700,
        query_start_loc=q.to(device),
        query_start_loc_cpu=q,
        seq_lens=seq.to(device),
        slot_mapping=torch.arange(40, dtype=torch.int64, device=device),
        block_table_tensor=torch.tensor(
            [[4, 9], [7, 2]], dtype=torch.int32, device=device
        ),
    )
    for rank in range(world):
        outs = []
        for build in (
            qsa_cache._build_qsa_metadata_torch,
            qsa_cache.build_qsa_metadata_triton,
        ):
            outs.append(
                build(
                    common,
                    torch.empty(41, dtype=torch.int32, device=device),
                    torch.empty(40, dtype=torch.int64, device=device),
                    torch.empty(40, dtype=torch.int64, device=device),
                    storage_block_size=416 // world,
                    compress_ratio=4,
                    ignore_common_slot_mask=True,
                    index_dcp_world_size=world,
                    index_dcp_rank=rank,
                )
            )
        for torch_out, triton_out in zip(*outs):
            assert torch.equal(torch_out.cpu(), triton_out.cpu())
        assert (outs[0][2] >= 0).any() or world > 1


def _ordered_key(bits: int) -> int:
    """Total order of csrc/qsa_lexicographic_topk.cuh on raw FP32 bits."""
    bits &= 0xFFFFFFFF
    if bits == 0x80000000:  # -0.0 folds into +0.0
        bits = 0
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)


def _reference_select(scores, lengths, out, k) -> None:
    """Python model of the SM70 lexicographic top-k, on raw score bits.

    See csrc/qsa_lexicographic_topk.cuh. Bits are read through an int32 view
    (not Python floats) so NaN payloads, including signaling NaNs, keep
    their exact order.
    """
    bits = scores.contiguous().view(torch.int32)
    for row in range(scores.shape[0]):
        n = max(0, min(int(lengths[row]), scores.shape[1]))
        values = bits[row].tolist()
        if n <= k:
            chosen = list(range(n))
        else:
            order = sorted(range(n), key=lambda i: (-_ordered_key(values[i]), i))
            chosen = sorted(order[:k])
        out[row] = torch.tensor(chosen + [-1] * (k - len(chosen)), dtype=torch.int32)


class _Packed(Exception):
    """Raised by the capturing all-gather once a rank has packed its rows."""


def _float_from_bits(*words: int) -> torch.Tensor:
    return torch.tensor(
        [w - (1 << 32) if w & 0x80000000 else w for w in words], dtype=torch.int32
    ).view(torch.float32)


def _merge_all_ranks(
    scores, visible, world, k, select=None, garbage_col0=False, use_triton=None
):
    """Shard ``scores`` by g % world, merge on every rank, return results."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _merge_sharded_index_blocks

    rows = scores.shape[0]
    local = []
    for rank in range(world):
        local_scores = scores[:, rank::world].contiguous()
        local_visible = (visible + world - 1 - rank).div(world, rounding_mode="floor")
        if local_scores.shape[1] == 0:  # real pages always keep a column
            local_scores = torch.empty(rows, 1, device=scores.device)
        if garbage_col0:  # unwritten scorer output where this rank sees nothing
            empty = local_visible == 0
            local_scores[empty, 0] = _float_from_bits(0x7FC0DEAD)
        blocks = torch.empty(rows, k, dtype=torch.int32, device=scores.device)
        _reference_select(local_scores, local_visible, blocks, k)
        local.append((blocks, local_scores, local_visible))
    packed = []
    for rank, (blocks, local_scores, local_visible) in enumerate(local):

        def capture(t):
            packed.append(t)
            raise _Packed

        with pytest.raises(_Packed):
            _merge_sharded_index_blocks(
                blocks,
                local_scores,
                local_visible,
                world,
                rank,
                k,
                all_gather=capture,
                use_triton=use_triton,
            )
    everything = torch.cat(packed, dim=0)
    merged = [
        _merge_sharded_index_blocks(
            blocks,
            local_scores,
            local_visible,
            world,
            rank,
            k,
            all_gather=lambda t: everything,
            select=select or _reference_select,
            use_triton=use_triton,
        )
        for rank, (blocks, local_scores, local_visible) in enumerate(local)
    ]
    return merged, packed


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("kind", ["normal", "ties", "specials", "nans"])
@pytest.mark.parametrize("use_triton", [False, True])
def test_shard_merge_matches_unsharded_selection(
    world: int, kind: str, use_triton: bool
) -> None:
    if use_triton and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU Triton execution needs TRITON_INTERPRET=1")
    torch.manual_seed(world * 7 + len(kind))
    k, columns, rows = 8, 61, 6
    if kind == "normal":
        scores = torch.randn(rows, columns)
    elif kind == "ties":
        scores = torch.randint(0, 3, (rows, columns)).float()
    else:
        scores = torch.randn(rows, columns)
        scores[:, ::5] = -0.0
        scores[:, 1::7] = 0.0
        scores[:, 2::11] = float("inf")
        scores[:, 3::13] = -float("inf")
    if kind == "nans":
        # quiet +/- NaN with different payloads, and signaling NaNs.
        for col, word in zip(
            (4, 9, 17, 22, 30, 41),
            (0x7FC00000, 0x7FC00001, 0xFFC00000, 0xFFC00123, 0x7F800001, 0xFF800001),
        ):
            scores[:, col] = _float_from_bits(word)
    visible = torch.tensor([0, 3, 8, 9, 40, 61], dtype=torch.int32)
    expected = torch.empty(rows, k, dtype=torch.int32)
    _reference_select(scores, visible, expected, k)
    merged, _ = _merge_all_ranks(scores, visible, world, k, use_triton=use_triton)
    for rank, result in enumerate(merged):
        assert torch.equal(result, expected), (rank, result, expected)


def test_reference_keeps_signaling_nan_order() -> None:
    # Review F4 counterexample: Python floats quiet the sNaN and flip the pick.
    scores = _float_from_bits(0x7F800001, 0x7FC00000).reshape(1, 2)
    out = torch.empty(1, 1, dtype=torch.int32)
    _reference_select(scores, torch.tensor([2]), out, 1)
    assert out.tolist() == [[1]]


@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("use_triton", [False, True])
def test_shard_merge_one_rank_empty(columns: int, use_triton: bool) -> None:
    if use_triton and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU Triton execution needs TRITON_INTERPRET=1")
    # V = 1: rank 0 owns the only group, rank 1 sees nothing (V_r = 0).
    k, world = 4, 2
    scores = torch.randn(3, columns)
    visible = torch.tensor([1, 0, columns], dtype=torch.int32)
    expected = torch.empty(3, k, dtype=torch.int32)
    _reference_select(scores, visible, expected, k)
    merged, packed = _merge_all_ranks(
        scores, visible, world, k, garbage_col0=True, use_triton=use_triton
    )
    for result in merged:
        assert torch.equal(result, expected)
    # Lanes without a candidate carry the fixed pad, never the unwritten score.
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        _INDEX_SHARD_PAD_ID,
        _INDEX_SHARD_PAD_SCORE_BITS,
    )

    for rank_packed in packed:
        assert rank_packed.dtype == torch.int32 and rank_packed.is_contiguous()
        score_bits, groups = rank_packed[0, :, :k], rank_packed[0, :, k:]
        pad = groups == _INDEX_SHARD_PAD_ID
        assert (score_bits[pad] == _INDEX_SHARD_PAD_SCORE_BITS).all()
    assert (packed[1][0, 1, k:] == _INDEX_SHARD_PAD_ID).all()


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("k", [7, 512])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_fused_merge_stages_match_torch(world: int, k: int, device: str) -> None:
    """Compare packing, stable sort, lengths and mapping as raw integer words."""
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    if device == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU Triton execution needs TRITON_INTERPRET=1")
    rows, columns = 16, 4 * k + 3  # non-divisible tail for W=2/3
    torch.manual_seed(83 + world + k)
    # Build the raw-bit fixture on CPU to avoid float conversion of sNaNs.
    scores = torch.randint(0, 4, (rows, columns)).float()
    words = _float_from_bits(
        0,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x7FC00001,
        0xFFC00000,
        0xFFC00123,
        0x7F800001,
        0xFF800001,
        0x00000001,
        0x80000001,
    ).view(torch.int32)
    scores.view(torch.int32)[:, : len(words)] = words
    scores = scores.to(device)
    pattern = [0, 1, k - 1, k, k + 1, world * k - 1, world * k, columns]
    visible = torch.tensor(pattern * 2, dtype=torch.int32, device=device)
    reference_stages = []

    def record(s, v, out, kk):
        reference_stages.append((s.view(torch.int32).clone(), v.clone()))
        _reference_select(s, v, out, kk)

    reference, packed = _merge_all_ranks(
        scores,
        visible,
        world,
        k,
        select=record,
        garbage_col0=True,
        use_triton=False,
    )
    stage = iter(reference_stages)

    def compare(s, v, out, kk):
        expected_scores, expected_lengths = next(stage)
        assert torch.equal(s.view(torch.int32), expected_scores)
        assert torch.equal(v, expected_lengths)
        _reference_select(s, v, out, kk)

    actual, fused_packed = _merge_all_ranks(
        scores,
        visible,
        world,
        k,
        select=compare,
        garbage_col0=True,
        use_triton=True,
    )
    for expected_pack, actual_pack, expected, result in zip(
        packed, fused_packed, reference, actual
    ):
        assert torch.equal(actual_pack, expected_pack)
        assert torch.equal(result, expected)

    for rank in range(world):
        logits = scores[:, rank::world]
        local_visible = (visible + world - 1 - rank).div(world, rounding_mode="floor")
        storage = torch.empty((rows, 2 * k + 3), dtype=torch.int32, device=device)
        blocks = storage[:, 1 : 2 * k + 1 : 2]
        _reference_select(logits, local_visible, blocks, k)
        # Both row/column strides and the storage offset reach the pack kernel.
        strided_pack = qsa_ops._pack_index_shard_candidates(blocks, logits, world, rank)
        assert torch.equal(strided_pack, packed[rank][0])

    gathered = torch.cat(packed, dim=0)
    # Give every pad a distinguishable raw score to expose unstable equal-id
    # writes, even though the production pack uses a single deterministic word.
    tags = torch.arange(world * rows * k, dtype=torch.int32, device=device).reshape(
        world, rows, k
    )
    gathered[:, :, :k] = torch.where(
        gathered[:, :, k:] == qsa_ops._INDEX_SHARD_PAD_ID, tags, gathered[:, :, :k]
    )
    # Exercise all strides and storage offsets, including all-gather wrappers
    # returning a view. Real collectives return the contiguous specialization.
    storage = torch.empty((world, rows, 4 * k + 3), dtype=torch.int32, device=device)
    strided = storage[:, :, 1 : 4 * k + 1 : 2]
    strided.copy_(gathered)
    sorted_scores, sorted_groups, lengths = qsa_ops._sort_index_shard_candidates(
        strided
    )
    candidates = gathered[:, :, k:].permute(1, 0, 2).reshape(rows, -1)
    order = torch.argsort(candidates, dim=1, stable=True)
    assert torch.equal(sorted_groups, torch.gather(candidates, 1, order))
    score_bits = gathered[:, :, :k].permute(1, 0, 2).reshape(rows, -1)
    assert torch.equal(
        sorted_scores.view(torch.int32), torch.gather(score_bits, 1, order)
    )
    assert torch.equal(lengths, reference_stages[0][1])
    positions_storage = torch.empty(rows, 2 * k + 3, dtype=torch.int32, device=device)
    positions = positions_storage[:, 1 : 2 * k + 1 : 2]
    _reference_select(sorted_scores, lengths, positions, k)
    mapped = qsa_ops._map_index_shard_positions(positions, sorted_groups)
    expected = torch.gather(sorted_groups, 1, positions.clamp_min(0).long())
    expected[positions < 0] = -1
    assert torch.equal(mapped, expected)


def test_fused_merge_dispatch_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    calls = []

    def capture_pack(*args):
        calls.append("fused")
        raise _Packed

    monkeypatch.setattr(qsa_ops, "_pack_index_shard_candidates", capture_pack)
    blocks = torch.zeros(1, 512, dtype=torch.int32)
    logits = torch.zeros(1, 1)
    for sm70, k in [(False, 512), (True, 8), (True, 512)]:
        monkeypatch.setattr(
            qsa_ops.current_platform,
            "is_device_capability",
            lambda c, sm70=sm70: sm70 and c == 70,
        )

        def capture_gather(t):
            calls.append("torch")
            raise _Packed

        with pytest.raises(_Packed):
            qsa_ops._merge_sharded_index_blocks(
                blocks[:, :k], logits, None, 2, 0, k, all_gather=capture_gather
            )
    assert calls == ["torch", "torch", "fused"]


def test_fused_merge_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    calls = []

    def capture_pack(*args):
        calls.append("fused")
        raise _Packed

    def capture_gather(t):
        calls.append("torch")
        raise _Packed

    monkeypatch.setattr(qsa_ops, "_pack_index_shard_candidates", capture_pack)
    monkeypatch.setattr(
        qsa_ops.current_platform, "is_device_capability", lambda c: c == 70
    )
    blocks = torch.zeros(1, 512, dtype=torch.int32)
    logits = torch.zeros(1, 1)
    for flag in ("1", "0"):
        monkeypatch.setenv("VLLM_QSA_INDEX_SHARD_FUSED_MERGE", flag)
        with pytest.raises(_Packed):
            qsa_ops._merge_sharded_index_blocks(
                blocks, logits, None, 2, 0, 512, all_gather=capture_gather
            )
    assert calls == ["fused", "torch"]


@pytest.mark.parametrize(
    "row",
    [[-1, 3, -1, -1], [2, 2, -1, -1], [5, 1, 7, -1], [0, -1, 4, -1]],
)
def test_fused_merge_debug_rejects_unordered_local_columns(
    monkeypatch: pytest.MonkeyPatch, row: list[int]
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    monkeypatch.setenv("VLLM_QSA_INDEX_SHARD_DEBUG", "1")
    monkeypatch.setattr(
        qsa_ops,
        "_pack_index_shard_candidates",
        lambda *args: (_ for _ in ()).throw(_Packed()),
    )
    good = torch.tensor([[0, 1, 4, -1]], dtype=torch.int32)
    logits = torch.zeros(1, 8)
    with pytest.raises(_Packed):
        qsa_ops._merge_sharded_index_blocks(
            good, logits, None, 2, 0, 4, use_triton=True
        )
    bad = torch.tensor([row], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="ascending local columns"):
        qsa_ops._merge_sharded_index_blocks(bad, logits, None, 2, 0, 4, use_triton=True)


def test_fused_pack_checks_inputs_on_host() -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    blocks = torch.zeros(3, 4, dtype=torch.int32)
    with pytest.raises(TypeError):
        qsa_ops._pack_index_shard_candidates(
            blocks, torch.zeros(3, 8, dtype=torch.float64), 2, 0
        )
    with pytest.raises(TypeError):
        qsa_ops._pack_index_shard_candidates(blocks.long(), torch.zeros(3, 8), 2, 0)
    with pytest.raises(ValueError):
        qsa_ops._pack_index_shard_candidates(blocks, torch.zeros(2, 8), 2, 0)


def test_fused_map_writes_row_major_for_transposed_positions() -> None:
    if os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available():
        pytest.skip("CPU Triton execution needs TRITON_INTERPRET=1")
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, k = 3, 7
    groups = torch.arange(rows * 2 * k, dtype=torch.int32, device=device).reshape(
        rows, 2 * k
    )
    base = torch.tensor(
        [[0, 2, 4, 6, 8, -1, -1], [1, 3, 5, 7, -1, -1, -1], [13, 0, 1, 2, 3, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    positions = base.t().contiguous().t()  # dense, column-major
    assert not positions.is_contiguous()
    merged = qsa_ops._map_index_shard_positions(positions, groups)
    expected = torch.where(
        base >= 0,
        torch.gather(groups, 1, base.clamp_min(0).long()),
        torch.full_like(base, -1),
    )
    assert merged.is_contiguous()
    assert torch.equal(merged, expected)


def test_shard_merge_debug_plan_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    monkeypatch.setenv("VLLM_QSA_INDEX_SHARD_DEBUG", "1")
    k, rows = 4, 3
    blocks = torch.tensor([[0, 1, -1, -1]] * rows, dtype=torch.int32)
    logits = torch.randn(rows, 2)
    req = torch.tensor([0, 0, 1], dtype=torch.int32)
    pos = torch.tensor([7, 8, 3], dtype=torch.int64)
    calls = []

    def gather_with(other_rows_keys):
        def gather(t):
            calls.append(t.shape)
            if t.dtype == torch.int64:  # the plan descriptor
                other = t.clone()
                if other_rows_keys is not None:
                    other[0, 5] += 1  # peer saw a different position layout
                return torch.cat((t, other), dim=0)
            return torch.cat((t, t), dim=0)

        return gather

    merged = qsa_ops._merge_sharded_index_blocks(
        blocks,
        logits,
        None,
        2,
        0,
        k,
        row_keys=(req, pos),
        all_gather=gather_with(None),
        select=_reference_select,
    )
    assert merged.shape == (rows, k)
    assert calls[0] == (1, 6) and len(calls) == 2
    with pytest.raises(RuntimeError, match="disagree"):
        qsa_ops._merge_sharded_index_blocks(
            blocks,
            logits,
            None,
            2,
            0,
            k,
            row_keys=(req, pos),
            all_gather=gather_with(True),
            select=_reference_select,
        )


def test_runtime_validator_rejects_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    platform = SimpleNamespace(
        is_cuda=lambda: True, is_device_capability=lambda c: c == 70
    )
    monkeypatch.setattr(qsa_ops, "current_platform", platform)
    monkeypatch.setattr(
        qsa_ops.torch.ops,
        "_C",
        SimpleNamespace(qsa_lexicographic_topk=object()),
        raising=False,
    )
    monkeypatch.setattr(
        qsa_ops.torch.ops, "_C_qsa_sm70", SimpleNamespace(), raising=False
    )
    qsa_ops.validate_qsa_index_dcp_shard_runtime(512)
    with pytest.raises(ValueError, match="512"):
        qsa_ops.validate_qsa_index_dcp_shard_runtime(256)
    platform.is_device_capability = lambda c: c == 80  # SM80
    with pytest.raises(ValueError, match="SM70"):
        qsa_ops.validate_qsa_index_dcp_shard_runtime(512)
    platform.is_cuda = lambda: False  # ROCm / CPU
    with pytest.raises(ValueError, match="SM70"):
        qsa_ops.validate_qsa_index_dcp_shard_runtime(512)
    platform.is_cuda, platform.is_device_capability = (
        (lambda: True),
        (lambda c: c == 70),
    )
    monkeypatch.setattr(qsa_ops.torch.ops, "_C", SimpleNamespace(), raising=False)
    with pytest.raises(ValueError, match="op"):
        qsa_ops.validate_qsa_index_dcp_shard_runtime(512)


def test_shard_merge_without_group_order_is_caught() -> None:
    # Negative control: skipping the global-group ordering breaks ties.
    scores = torch.ones(1, 40)
    expected = torch.empty(1, 8, dtype=torch.int32)
    _reference_select(scores, torch.tensor([40]), expected, 8)
    candidates = torch.cat([torch.arange(1, 40, 2)[:8], torch.arange(0, 40, 2)[:8]])
    positions = torch.empty(1, 8, dtype=torch.int32)
    _reference_select(torch.ones(1, 16), torch.tensor([16]), positions, 8)
    unordered = candidates[positions[0].long()].to(torch.int32)
    assert not torch.equal(unordered, expected[0])


@pytest.mark.parametrize("world", [2])
def test_paged_scorer_local_columns_match_global(world: int) -> None:
    import os

    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
    from vllm.triton_utils import triton

    if os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available():
        pytest.skip("needs CUDA or the Triton interpreter")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(5)
    page, pages, ratio = 16, 6, 4  # 16 global groups per page -> 8 local
    keys = torch.randn(pages * page, 128).half()
    table = torch.tensor([[3, 0, 5, 1]], dtype=torch.int32)
    positions = torch.tensor([7, 40, 150, 255], dtype=torch.int64)
    rows = positions.numel()
    q = torch.randn(rows, 4, 128).half()
    token_to_req = torch.zeros(rows, dtype=torch.int32)
    seq_lens = torch.tensor([256], dtype=torch.int32)

    def score(cache, page_size, world_, rank):
        columns = table.shape[1] * page_size
        logits = torch.full((rows, columns), float("nan"), device=device)
        visible = torch.empty(rows, dtype=torch.int32, device=device)
        qsa_ops._qsa_mqa_paged_kernel[(rows, triton.cdiv(columns, 64))](
            q.to(device),
            cache.to(device),
            table.to(device),
            token_to_req.to(device),
            positions.to(device),
            seq_lens.to(device),
            visible,
            logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            table.stride(0),
            table.stride(1),
            logits.stride(0),
            rows,
            columns,
            cache.shape[0],
            1,
            float(128**0.5),
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=table.shape[1],
            NUM_HEADS=4,
            HEAD_DIM=128,
            BLOCK_N=64,
            BLOCK_D=128,
            TILES_PER_PROG=1,
            STAGES=1,
            MAX_N=16,
            COMPRESS_RATIO=ratio,
            INDEX_DCP_WORLD=world_,
            INDEX_DCP_RANK=rank,
        )
        return logits.cpu(), visible.cpu()

    # Global (replicated) cache: logical group g lives at page table[g // page].
    glob = torch.zeros(pages, page, 1, 128).half()
    groups = table.shape[1] * page
    for g in range(groups):
        glob[table[0, g // page], g % page, 0] = keys[g]
    g_logits, g_visible = score(glob, page, 1, 0)
    local_page = page // world
    for rank in range(world):
        loc = torch.zeros(pages, local_page, 1, 128).half()
        for g in range(rank, groups, world):
            c = g // world
            loc[table[0, c // local_page], c % local_page, 0] = keys[g]
        l_logits, l_visible = score(loc, local_page, world, rank)
        expect_visible = (g_visible + world - 1 - rank).div(
            world, rounding_mode="floor"
        )
        assert torch.equal(l_visible, expect_visible.to(l_visible.dtype))
        for row in range(rows):
            n = int(l_visible[row])
            cols = torch.arange(n) * world + rank
            local_row, global_row = l_logits[row, :n].cpu(), g_logits[row, cols].cpu()
            if device.type == "cuda":
                # Each output column has a fixed K-reduction order on the GPU,
                # independent of its tile position: require identical bits.
                assert torch.equal(local_row, global_row), (rank, row)
            else:
                # The Triton interpreter runs tl.dot through NumPy BLAS, whose
                # result can depend on a row's position in the tile (~1 ulp).
                torch.testing.assert_close(local_row, global_row, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the SM70 top-k op")
@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("kind", ["normal", "ties", "specials"])
@pytest.mark.parametrize(
    ("rows", "batch"),
    [(1, False), (5, False), (5, True), (10, False), (10, True), (16, False)],
)
def test_shard_merge_with_native_topk_matches_unsharded(
    world: int, kind: str, rows: int, batch: bool
) -> None:
    # Local selection and the unsharded reference use the same op variant
    # (MTP-batch for 5/10 rows when VLLM_SM70_QSA_MTP_TOPK is on); the merge
    # always reruns the standard exact op, as in qsa_select_paged_tokens.
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    if not qsa_ops._use_sm70_qsa_lexicographic_topk(512):
        pytest.skip("exact SM70 top-k not selected on this device")
    if batch:

        def op(s, v, out, kk):
            torch.ops._C.qsa_lexicographic_topk(s, v, out, kk, True)
    else:
        op = qsa_ops._sm70_qsa_lexicographic_topk_op()
    torch.manual_seed(11 + rows)
    k, columns = 512, 6001
    if kind == "normal":
        scores = torch.randn(rows, columns, device="cuda")
    else:
        scores = torch.randint(0, 4, (rows, columns), device="cuda").float()
    if kind == "specials":
        scores.view(torch.int32)[:, :8] = (
            _float_from_bits(
                0x80000000,
                0,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0xFFC00123,
                0x7F800001,
                0xFF800001,
            )
            .view(torch.int32)
            .to("cuda")
        )
    # R=1 must exercise selection, while the larger batches also cover pads.
    pattern = [6001, 0, 512, 2049, 300, 1, 511, 513, 4097, 5999]
    visible = torch.tensor(
        [pattern[i % len(pattern)] for i in range(rows)],
        dtype=torch.int32,
        device="cuda",
    )
    expected = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    op(scores, visible, expected, k)
    local = []
    for rank in range(world):
        local_scores = scores[:, rank::world].contiguous()
        local_visible = (visible + world - 1 - rank).div(world, rounding_mode="floor")
        blocks = torch.empty(rows, k, dtype=torch.int32, device="cuda")
        op(local_scores, local_visible, blocks, k)
        local.append((blocks, local_scores, local_visible))
    packed = []
    for rank, (blocks, local_scores, local_visible) in enumerate(local):
        captured: dict[str, torch.Tensor] = {}

        def grab(t, captured=captured):
            captured["t"] = t
            raise StopIteration

        with contextlib.suppress(StopIteration):
            qsa_ops._merge_sharded_index_blocks(
                blocks, local_scores, local_visible, world, rank, k, all_gather=grab
            )
        packed.append(captured["t"])
    everything = torch.cat(packed, dim=0)
    for rank, (blocks, local_scores, local_visible) in enumerate(local):
        for use_triton in (False, True, None):
            merged = qsa_ops._merge_sharded_index_blocks(
                blocks,
                local_scores,
                local_visible,
                world,
                rank,
                k,
                all_gather=lambda t, e=everything: e,
                use_triton=use_triton,
            )
            assert torch.equal(merged, expected), (rank, use_triton)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graph and SM70")
@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("rows", [1, 5, 10, 16])
def test_fused_merge_graph_replays_changed_candidates(world: int, rows: int) -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    if not qsa_ops._use_sm70_qsa_lexicographic_topk(512):
        pytest.skip("exact SM70 top-k not selected on this device")
    k, columns = 512, 2051
    op = qsa_ops._sm70_qsa_lexicographic_topk_op()
    scores = torch.randn(rows, columns, device="cuda")
    visible = torch.full((rows,), columns, dtype=torch.int32, device="cuda")
    # Retain storage with non-contiguous row and column strides in both inputs.
    local_scores = [scores[:, rank::world] for rank in range(world)]
    local_blocks = [
        torch.empty(rows, 2 * k + 3, dtype=torch.int32, device="cuda")[
            :, 1 : 2 * k + 1 : 2
        ]
        for _ in range(world)
    ]

    def step():
        packed = []
        for rank in range(world):
            logits = local_scores[rank].contiguous()  # native op needs unit col stride
            lengths = (visible + world - 1 - rank).div(world, rounding_mode="floor")
            blocks = torch.empty(rows, k, dtype=torch.int32, device="cuda")
            op(logits, lengths, blocks, k)
            local_blocks[rank].copy_(blocks)
            packed.append(
                qsa_ops._pack_index_shard_candidates(
                    local_blocks[rank], local_scores[rank], world, rank
                ).unsqueeze(0)
            )
        gathered = torch.cat(packed, dim=0)
        return qsa_ops._merge_sharded_index_blocks(
            local_blocks[0],
            local_scores[0],
            None,
            world,
            0,
            k,
            all_gather=lambda t: gathered,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()  # compile and warm up outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        merged = step()
    expected = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    for length in (0, 1, k - 1, k, k + 1, columns):
        scores.random_(0, 4)
        visible.copy_(
            torch.tensor(
                [min(columns, length + row) for row in range(rows)],
                dtype=torch.int32,
                device="cuda",
            )
        )
        graph.replay()
        op(scores, visible, expected, k)
        assert torch.equal(merged, expected), length


def test_sharding_rejects_mtp_drafting() -> None:
    config = _config(True)
    config.speculative_config = SimpleNamespace(method="mtp")
    with pytest.raises(ValueError, match="MTP"):
        qsa_index_dcp_sharded(config, TARGET, 4)
    config.speculative_config = SimpleNamespace(method="dflash")
    assert qsa_index_dcp_sharded(config, TARGET, 4)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_local_visible_prefix_and_group_round_trip(world: int) -> None:
    # Owned groups g < V map to the contiguous local prefix [0, V_r).
    for visible in range(0, 200):
        for rank in range(world):
            owned = [g for g in range(visible) if g % world == rank]
            local = [g // world for g in owned]
            assert local == list(range(len(local)))
            assert len(owned) == (visible + world - 1 - rank) // world
            assert all(c * world + rank == g for c, g in zip(local, owned))


def test_visible_kernel_reports_local_prefix() -> None:
    import os

    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    if os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available():
        pytest.skip("needs CUDA or the Triton interpreter")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = torch.arange(-3, 70, dtype=torch.int64, device=device)
    rows = positions.numel()
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([66], dtype=torch.int32, device=device)
    glob = qsa_ops._qsa_visible_blocks(token_to_req, positions, seq_lens, 4).cpu()
    for world in (2, 3):
        for rank in range(world):
            local = qsa_ops._qsa_visible_blocks(
                token_to_req,
                positions,
                seq_lens,
                4,
                index_dcp_world_size=world,
                index_dcp_rank=rank,
            ).cpu()
            expect = (glob.clamp_min(0) + world - 1 - rank).div(
                world, rounding_mode="floor"
            )
            assert torch.equal(local, expect.to(local.dtype)), (world, rank)


def test_canonical_block_table_early_return_keeps_physical_ids() -> None:
    builder = object.__new__(QSAMetadataBuilder)
    builder.kv_cache_spec = MLAAttentionSpec(
        block_size=832,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        dcp_sharded=True,
    )
    builder.kernel_block_size = 16
    builder.dcp_world_size = 2
    builder.reads_sharded_group = True
    builder.block_table_buffer = torch.empty((2, 8), dtype=torch.int32)
    table = torch.tensor([[3, 9, 4], [6, 1, 8]], dtype=torch.int32)
    assert builder._canonical_block_table(table) is table


def test_uniform_spec_with_sharded_indexer_keeps_main_representative() -> None:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

    main = FullAttentionSpec(
        block_size=832,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        dcp_sharded=True,
    )
    index = MLAAttentionSpec(
        block_size=832,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        dcp_sharded=True,
    )
    uniform = UniformTypeKVCacheSpecs.from_specs(
        {"layers.3.attn": main, "layers.3.attn.indexer.k": index}, 2
    )
    assert uniform is not None
    assert uniform.dcp_sharded and uniform.block_size == 832
    assert uniform.global_block_size(2) == 1664
    assert uniform.page_size_bytes == main.page_size_bytes + index.page_size_bytes


def test_builder_injects_this_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qsa_cache, "_current_dcp_rank", lambda: 1)
    spec = MLAAttentionSpec(
        block_size=832,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        compress_ratio=4,
        dcp_sharded=True,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=2),
    )
    from vllm.v1.attention.backend import AttentionMetadataBuilder

    monkeypatch.setattr(
        AttentionMetadataBuilder,
        "__init__",
        lambda self, spec, *a, **k: setattr(self, "kv_cache_spec", spec),
    )
    builder = QSAMetadataBuilder(spec, [TARGET], config, torch.device("cpu"), 4)
    assert builder.index_dcp_sharded
    assert (builder.index_dcp_world_size, builder.index_dcp_rank) == (2, 1)


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_shared_key_local_inputs_give_local_prefix(world: int, dtype) -> None:
    # The native shared-key kernel sees min(pos + 1, len) // c (truncated)
    # visible groups and scores column j iff j < (pos + 1) // c and
    # j < len // c. With the converted inputs this must be V_r and select
    # exactly the local columns of the globally valid groups.
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _shared_key_local_inputs

    torch.manual_seed(world)
    c = 4
    for length in (0, 1, 3, 4, 5, 63, 64, 1000, 4097):
        positions = torch.tensor(
            [-6, -5, -4, -3, -2, -1, 0, 2, 3, 4, 7, length - 2, length - 1, length + 5],
            dtype=dtype,
        )
        lengths = torch.tensor([length], dtype=torch.int32)
        for rank in range(world):
            pos_l, len_l = _shared_key_local_inputs(positions, lengths, c, world, rank)
            assert pos_l.dtype == dtype and len_l.dtype == torch.int32
            for row, pos in enumerate(positions.tolist()):
                v = max(0, min(pos + 1, length) // c)
                v_r = (v + world - 1 - rank) // world
                lim = min(int(pos_l[row]) + 1, int(len_l[0]))
                kernel_v = -((-lim) // c) if lim < 0 else lim // c  # C truncation
                assert kernel_v == v_r, (length, pos, rank)
                local_valid = [
                    j
                    for j in range(v_r + 2)
                    if j < (int(pos_l[row]) + 1) // c and j < int(len_l[0]) // c
                ]
                global_valid = [g for g in range(v) if g % world == rank]
                assert [j * world + rank for j in local_valid] == global_valid


def test_noncontiguous_positions_keep_the_unsharded_shared_key_gate(
    monkeypatch,
) -> None:
    # Sharding must not widen the shared-key domain: inputs the unsharded gate
    # rejects (non-contiguous positions) reach the gate unconverted.
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    seen = {}

    def fake(q, cache, table, requests, positions, lengths, *rest):
        seen["contiguous"] = positions.is_contiguous()
        raise StopIteration

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.ops.qsa_shared_key.maybe_shared_key_scores", fake
    )
    monkeypatch.setattr(qsa_ops.torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(qsa_ops, "HAS_TRITON", True)
    q = torch.zeros(2, 4, 128, dtype=torch.float16)
    cache = torch.zeros(2, 32, 1, 128, dtype=torch.float16)
    table = torch.zeros(1, 2, dtype=torch.int32)
    positions = torch.arange(10, dtype=torch.int64)[::5]
    assert not positions.is_contiguous()
    with pytest.raises(StopIteration):
        qsa_ops.qsa_mqa_paged(
            q,
            cache,
            table,
            torch.zeros(2, dtype=torch.int32),
            positions,
            torch.tensor([64], dtype=torch.int32),
            4,
            shared_key_scoring=True,
            index_dcp_world_size=2,
            index_dcp_rank=0,
        )
    assert seen["contiguous"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA and the SM70 shared-key op"
)
@pytest.mark.parametrize("world", [2])
@pytest.mark.parametrize("rows", [2, 5, 8])
def test_shared_key_scores_sharded_match_global(world: int, rows: int) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _shared_key_local_inputs
    from vllm.models.qwen4_exp.nvidia.ops.qsa_shared_key import shared_key_reason

    torch.manual_seed(rows)
    dev = "cuda"
    c, page, columns_global = 4, 64, 6000
    length = columns_global * c + 3
    q = torch.randn(rows, 4, 128, device=dev, dtype=torch.float16)
    keys = torch.randn(columns_global, 128, device=dev, dtype=torch.float16)
    pages_g = (columns_global + page - 1) // page
    cache_g = torch.zeros(pages_g * page, 1, 128, device=dev, dtype=torch.float16)
    cache_g[:columns_global, 0] = keys
    cache_g = cache_g.view(pages_g, page, 1, 128)
    table_g = torch.arange(pages_g, device=dev, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=dev, dtype=torch.int32)
    positions = torch.arange(length - rows, length, device=dev, dtype=torch.int64)
    lengths = torch.tensor([length], device=dev, dtype=torch.int32)
    if shared_key_reason(q, cache_g, table_g, requests, positions, lengths) is not None:
        pytest.skip("native shared-key operator unavailable")
    run = torch.ops.vllm_sm70_qsa_indexer.run
    scores_g = torch.empty(rows, columns_global, device=dev)
    visible_g = torch.empty(rows, device=dev, dtype=torch.int32)
    run(
        scores_g,
        visible_g,
        q,
        cache_g,
        table_g,
        requests,
        positions,
        lengths,
        128**0.5,
        c,
    )
    lpage = page // world
    for rank in range(world):
        local_keys = keys[rank::world]
        cols = local_keys.shape[0]
        lp = (cols + lpage - 1) // lpage
        cache_l = torch.zeros(lp * lpage, 1, 128, device=dev, dtype=torch.float16)
        cache_l[:cols, 0] = local_keys
        cache_l = cache_l.view(lp, lpage, 1, 128)
        table_l = torch.arange(lp, device=dev, dtype=torch.int32)[None]
        pos_l, len_l = _shared_key_local_inputs(positions, lengths, c, world, rank)
        scores_l = torch.empty(rows, cols, device=dev)
        visible_l = torch.empty(rows, device=dev, dtype=torch.int32)
        run(
            scores_l,
            visible_l,
            q,
            cache_l,
            table_l,
            requests,
            pos_l,
            len_l,
            128**0.5,
            c,
        )
        expect_v = (visible_g.clamp_min(0) + world - 1 - rank) // world
        assert torch.equal(visible_l, expect_v)
        for row in range(rows):
            n = int(expect_v[row])
            assert torch.equal(
                scores_l[row, :n].view(torch.int32),
                scores_g[row, rank::world][:n].view(torch.int32),
            ), (rank, row)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA and the SM70 shared-key op"
)
@pytest.mark.parametrize("rows", [2, 5, 8])
def test_paged_wrapper_sharded_shared_key_matches_global(
    rows: int, monkeypatch
) -> None:
    # Through qsa_mqa_paged itself: the sharded call must take the native
    # shared-key path (counted) and return the global columns' exact bits.
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
    from vllm.models.qwen4_exp.nvidia.ops import qsa_shared_key as sk

    calls = []
    real = sk.maybe_shared_key_scores

    def counting(*a, **k):
        out = real(*a, **k)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(sk, "maybe_shared_key_scores", counting)
    torch.manual_seed(3 + rows)
    dev, c, world, page, cols = "cuda", 4, 2, 64, 5001
    length = cols * c + 2
    q = torch.randn(rows, 4, 128, device=dev, dtype=torch.float16)
    keys = torch.randn(cols, 128, device=dev, dtype=torch.float16)

    def paged(k, page_size):
        n = (k.shape[0] + page_size - 1) // page_size
        buf = torch.zeros(n * page_size, 1, 128, device=dev, dtype=torch.float16)
        buf[: k.shape[0], 0] = k
        return buf.view(n, page_size, 1, 128), torch.arange(
            n, device=dev, dtype=torch.int32
        )[None]

    req = torch.zeros(rows, device=dev, dtype=torch.int32)
    pos = torch.arange(length - rows, length, device=dev, dtype=torch.int64)
    lens = torch.tensor([length], device=dev, dtype=torch.int32)
    cache_g, table_g = paged(keys, page)
    scores_g, vis_g = qsa_ops.qsa_mqa_paged(
        q, cache_g, table_g, req, pos, lens, c, shared_key_scoring=True
    )
    if calls != [True]:
        pytest.skip("native shared-key operator unavailable")
    for rank in range(world):
        cache_l, table_l = paged(keys[rank::world], page // world)
        scores_l, vis_l = qsa_ops.qsa_mqa_paged(
            q,
            cache_l,
            table_l,
            req,
            pos,
            lens,
            c,
            shared_key_scoring=True,
            index_dcp_world_size=world,
            index_dcp_rank=rank,
        )
        assert calls[-1] is True  # the sharded call used the native scorer
        expect = (vis_g.clamp_min(0) + world - 1 - rank) // world
        assert torch.equal(vis_l, expect)
        for row in range(rows):
            n = int(expect[row])
            assert torch.equal(
                scores_l[row, :n].view(torch.int32),
                scores_g[row, rank::world][:n].view(torch.int32),
            )
