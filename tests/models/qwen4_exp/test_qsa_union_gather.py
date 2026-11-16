# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""host KV selection-union gather: chunk union gather keeps the sparse attention inputs
bit-identical.

CPU parts run the planner, the transfer kernel and the Triton sparse kernel
in the Triton interpreter (``TRITON_INTERPRET=1``). The grouped page4 planner
is CUDA-only; its CPU check uses a Python port of the planner, and the real
planner and forward kernel are compared in the CUDA tests at the end.
"""

import os
from dataclasses import dataclass

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa_union_gather import (
    MICROBLOCK_TOKENS,
    QSAArena,
    allocate_qsa_arena,
    plan_qsa_union_gather,
    qsa_union_gather_attention,
    qsa_union_gather_copy,
    qsa_union_gather_copy_reference,
)
from vllm.triton_utils import triton

pytestmark = pytest.mark.skip_global_cleanup

interpreter = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
    reason="Triton kernels need CUDA or TRITON_INTERPRET=1",
)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DIM = 32
_INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
GROUP = 16  # query heads per KV head; tl.dot needs at least 16 rows
BLOCK_N = 16


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Case:
    key: torch.Tensor  # [blocks, page, 1, DIM] view of an interleaved cache
    value: torch.Tensor
    block_table: torch.Tensor
    indices: torch.Tensor
    token_to_req: torch.Tensor
    query: torch.Tensor
    e4m3: bool


def _cache(blocks, page, e4m3, generator, members=1):
    """Model-ABI cache ``[blocks, members, 2, page, 1, DIM]``; one member."""
    cache = torch.randn((blocks, members, 2, page, 1, DIM), generator=generator)
    if e4m3:
        cache = (cache * 4).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        cache = cache.half()
    key, value = cache.to(_device())[:, members - 1].unbind(1)
    return key, value


def _selection_row(capacity, width, unit, generator):
    """Score-ordered aligned runs, a duplicate, invalid and stray entries."""
    runs = max(1, width // (2 * unit))
    starts = torch.randperm(capacity // unit, generator=generator)[:runs] * unit
    tokens = (starts[:, None] + torch.arange(unit)).flatten().tolist()
    if tokens:
        tokens.append(tokens[0])  # duplicate column
    tokens.append(capacity + 1)  # past the block table
    stray = torch.randint(0, capacity, (2,), generator=generator).tolist()
    tokens += stray
    tokens = tokens[:width]
    return tokens + [-1] * (width - len(tokens))


def make_case(
    seed=0,
    *,
    rows=20,
    width=24,
    page=8,
    blocks=24,
    requests=3,
    table_width=4,
    unit=4,
    e4m3=False,
    members=1,
) -> Case:
    generator = torch.Generator().manual_seed(seed)
    key, value = _cache(blocks, page, e4m3, generator, members)
    table = torch.randperm(blocks, generator=generator)[: requests * table_width]
    table = table.view(requests, table_width).clone()
    if requests > 1:
        table[1, 0] = table[0, 0]  # a shared prefix page
    table[-1, -1] = -1  # an unallocated page
    if table_width > 2:
        table[0, 2] = blocks + 5  # beyond the cache: invalid in every kernel
    token_to_req = torch.randint(0, requests, (rows,), generator=generator)
    token_to_req[0] = -1
    if rows > 2:
        token_to_req[2] = requests  # out of range request
    capacity = table_width * page
    indices = torch.tensor(
        [_selection_row(capacity, width, unit, generator) for _ in range(rows)]
    )
    if rows > 5:
        indices[5] = -1  # a row without any selection
    query = (torch.randn((rows, GROUP, DIM), generator=generator) * 0.5).half()
    device = _device()
    return Case(
        key,
        value,
        table.to(torch.int32).to(device),
        indices.to(torch.int32).to(device),
        token_to_req.to(torch.int32).to(device),
        query.to(device),
        e4m3,
    )


def _arena(case, *, hot=0, staging=64):
    return allocate_qsa_arena(
        hot_microblocks=hot,
        staging_microblocks=staging,
        kv_heads=case.key.shape[2],
        head_dim=DIM,
        dtype=case.key.dtype,
        device=case.key.device,
    )


def _resident(case, arena: QSAArena, fraction, seed):
    """Copy a random subset of source microblocks into the hot region."""
    blocks, page = case.key.shape[:2]
    microblocks = blocks * page // MICROBLOCK_TOKENS
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(microblocks, generator=generator)
    chosen = chosen[: min(arena.staging_base, int(microblocks * fraction))]
    resident = torch.full((microblocks,), -1, dtype=torch.int32)
    per_page = page // MICROBLOCK_TOKENS
    for slot, microblock in enumerate(chosen.tolist()):
        resident[microblock] = slot
        block, first = divmod(microblock, per_page)
        tokens = slice(first * 4, first * 4 + 4)
        arena.key[slot] = case.key[block, tokens]
        arena.value[slot] = case.value[block, tokens]
    return resident.to(case.key.device)


def _source_entry(case, row, column, page):
    """(block, offset) a sparse kernel reads for one entry, or None."""
    token = int(case.indices[row, column])
    request = int(case.token_to_req[row])
    requests, width = case.block_table.shape
    if token < 0 or not 0 <= request < requests or token // page >= width:
        return None
    block = int(case.block_table[request, token // page])
    if not 0 <= block < case.key.shape[0]:
        return None
    return block, token % page


def _check_transport(case, arena, plan):
    """Every entry resolves to identical bytes through the arena (B/C gate)."""
    page = case.key.shape[1]
    table = plan.microblock_table
    rows, width = case.indices.shape
    for row in range(rows):
        for column in range(width):
            source = _source_entry(case, row, column, page)
            token = int(case.indices[row, column])
            request = int(case.token_to_req[row])
            arena_microblock = -1
            if (
                token >= 0
                and 0 <= request < table.shape[0]
                and token // 4 < table.shape[1]
            ):
                arena_microblock = int(table[request, token // 4])
            if source is None:
                assert arena_microblock < 0, (row, column)
                continue
            assert 0 <= arena_microblock < arena.num_microblocks, (row, column)
            block, offset = source
            for arena_view, source_view in (
                (arena.key, case.key),
                (arena.value, case.value),
            ):
                assert torch.equal(
                    arena_view[arena_microblock, token % 4].view(torch.uint8),
                    source_view[block, offset].view(torch.uint8),
                ), (row, column)


def _source_microblocks(case):
    page = case.key.shape[1]
    rows, width = case.indices.shape
    found: dict[int, set[int]] = {}
    for row in range(rows):
        for column in range(width):
            source = _source_entry(case, row, column, page)
            if source is not None:
                block, offset = source
                found.setdefault(block * page // 4 + offset // 4, set()).add(offset % 4)
    return found


# --------------------------------------------------------------------------
# Planner and transfer


@interpreter
@pytest.mark.parametrize("unit", [1, 2, 4])
@pytest.mark.parametrize("resident_fraction", [0.0, 0.3])
def test_transport_and_counts(unit, resident_fraction):
    case = make_case(unit, unit=unit, e4m3=True)
    arena = _arena(case, hot=16, staging=64)
    resident = (
        _resident(case, arena, resident_fraction, seed=unit)
        if resident_fraction
        else None
    )
    plan = plan_qsa_union_gather(
        case.indices,
        case.token_to_req,
        case.block_table,
        arena,
        page_size=case.key.shape[1],
        num_source_blocks=case.key.shape[0],
        unit_tokens=unit,
        resident_microblocks=resident,
    )
    qsa_union_gather_copy(case.key, case.value, arena, plan)
    _check_transport(case, arena, plan)

    referenced = _source_microblocks(case)
    hot = (
        {m for m in referenced if int(resident[m]) >= 0}
        if resident is not None
        else set()
    )
    misses = sorted(set(referenced) - hot)
    expected_units = sorted(
        {(m * 4 + offset) // unit for m in misses for offset in referenced[m]}
    )
    assert int(plan.union_microblocks) == len(referenced)
    assert int(plan.miss_microblocks) == len(misses)
    assert not bool(plan.overflow)
    count = int(plan.transfer_count)
    assert plan.transfer_source_units[:count].tolist() == expected_units
    assert (plan.transfer_source_units[count:] == -1).all()
    rows, width = case.indices.shape
    selected = sum(
        _source_entry(case, r, c, case.key.shape[1]) is not None
        for r in range(rows)
        for c in range(width)
    )
    assert int(plan.selected_tokens) == selected
    # Misses occupy distinct staging slots; hits keep their hot slots.
    staged = [int(plan.arena_of_source[m]) for m in misses]
    assert len(set(staged)) == len(staged)
    assert all(
        arena.staging_base <= s < arena.staging_base + arena.staging_microblocks
        for s in staged
    )
    for m in hot:
        assert resident is not None
        assert int(plan.arena_of_source[m]) == int(resident[m])


@interpreter
def test_shared_source_iff_shared_arena_microblock():
    case = make_case(7, rows=24, unit=4)
    arena = _arena(case)
    plan = plan_qsa_union_gather(
        case.indices,
        case.token_to_req,
        case.block_table,
        arena,
        page_size=case.key.shape[1],
        num_source_blocks=case.key.shape[0],
    )
    page = case.key.shape[1]
    pairs = {}
    rows, width = case.indices.shape
    for row in range(rows):
        for column in range(width):
            source = _source_entry(case, row, column, page)
            if source is None:
                continue
            request = int(case.token_to_req[row])
            token = int(case.indices[row, column])
            source_microblock = source[0] * page // 4 + source[1] // 4
            arena_microblock = int(plan.microblock_table[request, token // 4])
            pairs[(request, token // 4)] = (source_microblock, arena_microblock)
    values = list(pairs.values())
    # The shared prefix page is selected by two requests in this case.
    assert len({s for s, _ in values}) < len(values)
    for source_a, arena_a in values:
        for source_b, arena_b in values:
            assert (source_a == source_b) == (arena_a == arena_b)


@interpreter
def test_copy_kernel_matches_reference_and_respects_count():
    case = make_case(3, unit=2, e4m3=True)
    arena = _arena(case, staging=48)
    plan = plan_qsa_union_gather(
        case.indices,
        case.token_to_req,
        case.block_table,
        arena,
        page_size=case.key.shape[1],
        num_source_blocks=case.key.shape[0],
        unit_tokens=2,
    )
    sentinel = 0x5A
    arena.key.fill_(sentinel)
    arena.value.fill_(sentinel)
    reference = _arena(case, staging=48)
    reference.key.fill_(sentinel)
    reference.value.fill_(sentinel)
    qsa_union_gather_copy(case.key, case.value, arena, plan)
    qsa_union_gather_copy_reference(case.key, case.value, reference, plan)
    assert torch.equal(arena.key, reference.key)
    assert torch.equal(arena.value, reference.value)
    # Only the planned units changed; the list tail beyond the count did not.
    changed = (arena.key != sentinel).flatten(2).any(-1) | (
        arena.value != sentinel
    ).flatten(2).any(-1)
    count = int(plan.transfer_count)
    targets = plan.transfer_arena_units[:count]
    planned = torch.zeros_like(changed)
    planned.view(-1, 2, 2)[targets // 2, targets % 2] = True  # two tokens/unit
    assert not (changed & ~planned.view_as(changed)).any()


@interpreter
def test_overflow_flag_and_empty_selection():
    case = make_case(4)
    small = _arena(case, staging=2)
    plan = plan_qsa_union_gather(
        case.indices,
        case.token_to_req,
        case.block_table,
        small,
        page_size=case.key.shape[1],
        num_source_blocks=case.key.shape[0],
    )
    assert bool(plan.overflow)
    assert int(plan.miss_microblocks) > 2
    # Only the two microblocks that fit are transferred (one unit each).
    assert int(plan.transfer_count) == 2
    empty = plan_qsa_union_gather(
        torch.full_like(case.indices, -1),
        case.token_to_req,
        case.block_table,
        small,
        page_size=case.key.shape[1],
        num_source_blocks=case.key.shape[0],
    )
    assert int(empty.union_microblocks) == 0
    assert int(empty.transfer_count) == 0
    assert int(empty.selected_tokens) == 0
    assert not bool(empty.overflow)
    assert (empty.microblock_table == -1).all()


# --------------------------------------------------------------------------
# Sparse attention bit identity (Triton route)


def _triton_attention(
    q,
    key,
    value,
    indices,
    table,
    token_to_req,
    outputs,
    *,
    e4m3,
    splits,
    k_scale=0.5,
    v_scale=0.25,
):
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    out = outputs[0]
    lse = outputs[1] if len(outputs) > 1 else None
    rows, heads, dim = q.shape
    group = heads // key.shape[2]
    num_tiles = triton.cdiv(indices.shape[1], BLOCK_N)
    if splits == 1:
        partial_output, partial_lse = out, out
    else:
        partial_output = torch.empty(
            (splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (splits, rows, heads), dtype=torch.float32, device=q.device
        )
    if not e4m3:
        k_scale = v_scale = 1.0
    qsa_ops._qsa_sparse_paged_gqa_splitk_kernel[(rows, key.shape[2], splits)](
        q,
        key,
        value,
        indices,
        table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        lse,
        None,
        q.stride(0),
        q.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        indices.stride(0),
        table.stride(0),
        out.stride(0),
        out.stride(1),
        0,
        0,
        rows,
        key.shape[0],
        table.shape[0],
        k_scale,
        v_scale,
        TOPK=indices.shape[1],
        PAGE_SIZE=key.shape[1],
        PAGE_TABLE_WIDTH=table.shape[1],
        GROUP_SIZE=group,
        HEAD_DIM=dim,
        NUM_QUERY_HEADS=heads,
        NUM_SPLITS=splits,
        NUM_TILES=num_tiles,
        BLOCK_M=triton.next_power_of_2(group),
        BLOCK_N=BLOCK_N,
        KV_E4M3=e4m3,
    )
    if splits > 1:
        # Identical partials imply an identical merge. The Triton interpreter
        # cannot run the merge kernel (numpy rejects its scalar-and-vector
        # mask), so on CPU the partials are the compared outputs.
        outputs[2].copy_(partial_output.transpose(0, 1))
        outputs[3].copy_(partial_lse.transpose(0, 1))
        if _INTERPRETED:
            assert lse is not None
            out.zero_()
            lse.zero_()
            return
        qsa_ops._qsa_merge_splitk_kernel[(rows, heads)](
            partial_output,
            partial_lse,
            out,
            lse,
            None,
            out.stride(0),
            out.stride(1),
            0,
            0,
            rows,
            v_scale,
            HEAD_DIM=dim,
            NUM_QUERY_HEADS=heads,
            NUM_SPLITS=splits,
            BLOCK_SPLITS=triton.next_power_of_2(splits),
            KV_E4M3=e4m3,
        )


def _outputs(case, with_lse, splits=1):
    rows = case.query.shape[0]
    dtype = torch.float32 if with_lse else case.query.dtype
    out = torch.full(case.query.shape, 7.0, dtype=dtype, device=case.query.device)
    if not with_lse:
        assert splits == 1
        return [out]
    lse = torch.full((rows, GROUP), 7.0, dtype=torch.float32, device=out.device)
    if splits == 1:
        return [out, lse]
    partial_out = torch.full((rows, splits, GROUP, DIM), 7.0, device=out.device)
    partial_lse = torch.full((rows, splits, GROUP), 7.0, device=out.device)
    return [out, lse, partial_out, partial_lse]


def _assert_bitwise(expected, actual):
    for e, a in zip(expected, actual, strict=True):
        assert e.dtype == a.dtype and e.shape == a.shape
        assert torch.equal(e.view(torch.uint8), a.view(torch.uint8))


@interpreter
@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize("splits,with_lse", [(1, False), (2, True)])
@pytest.mark.parametrize("unit", [2, 4])
def test_triton_attention_bitwise(e4m3, splits, with_lse, unit):
    case = make_case(11 + unit, unit=unit, e4m3=e4m3, members=2)
    expected = _outputs(case, with_lse, splits)
    _triton_attention(
        case.query,
        case.key,
        case.value,
        case.indices,
        case.block_table,
        case.token_to_req,
        expected,
        e4m3=e4m3,
        splits=splits,
    )
    arena = _arena(case, hot=8, staging=64)
    resident = _resident(case, arena, 0.25, seed=unit)
    actual = _outputs(case, with_lse, splits)

    def attention(key, value, table, token_to_req, outputs):
        _triton_attention(
            case.query,
            key,
            value,
            case.indices,
            table,
            token_to_req,
            outputs,
            e4m3=e4m3,
            splits=splits,
        )

    result = qsa_union_gather_attention(
        attention,
        actual,
        case.indices,
        case.token_to_req,
        case.block_table,
        case.key,
        case.value,
        arena,
        unit_tokens=unit,
        resident_microblocks=resident,
        union_bound=64,
    )
    assert result.windows == 1 and bool(result.complete)
    _assert_bitwise(expected, actual)


@interpreter
@pytest.mark.parametrize("rows", [1, 3])
def test_triton_attention_single_and_tiny_batches(rows):
    case = make_case(21, rows=rows, width=16, e4m3=True)
    expected = _outputs(case, True, 2)
    _triton_attention(
        case.query,
        case.key,
        case.value,
        case.indices,
        case.block_table,
        case.token_to_req,
        expected,
        e4m3=True,
        splits=2,
    )
    arena = _arena(case, staging=40)
    actual = _outputs(case, True, 2)

    def attention(key, value, table, token_to_req, outputs):
        _triton_attention(
            case.query,
            key,
            value,
            case.indices,
            table,
            token_to_req,
            outputs,
            e4m3=True,
            splits=2,
        )

    qsa_union_gather_attention(
        attention,
        actual,
        case.indices,
        case.token_to_req,
        case.block_table,
        case.key,
        case.value,
        arena,
    )
    _assert_bitwise(expected, actual)


# --------------------------------------------------------------------------
# Row windows


@interpreter
@pytest.mark.parametrize("unit", [2, 4])
def test_windows_bitwise_and_bounded(unit):
    case = make_case(
        31,
        rows=40,
        width=24,
        page=8,
        blocks=40,
        requests=2,
        table_width=16,
        unit=unit,
        e4m3=True,
    )
    expected = _outputs(case, True, 2)
    _triton_attention(
        case.query,
        case.key,
        case.value,
        case.indices,
        case.block_table,
        case.token_to_req,
        expected,
        e4m3=True,
        splits=2,
    )
    full = plan_qsa_union_gather(
        case.indices,
        case.token_to_req,
        case.block_table,
        _arena(case, staging=256),
        page_size=8,
        num_source_blocks=40,
        unit_tokens=unit,
    )
    total = int(full.miss_microblocks)
    staging = max(total // 2, 1)
    group_max = 0
    for start in range(0, 40, 8):
        mask = torch.zeros(40, dtype=torch.bool, device=case.indices.device)
        mask[start : start + 8] = True
        group_plan = plan_qsa_union_gather(
            case.indices,
            case.token_to_req,
            case.block_table,
            _arena(case, staging=256),
            page_size=8,
            num_source_blocks=40,
            unit_tokens=unit,
            row_mask=mask,
        )
        group_max = max(group_max, int(group_plan.miss_microblocks))
    staging = max(staging, group_max)
    assert staging < total, "the case must not fit one window"
    arena = _arena(case, staging=staging)
    calls = []

    def attention(key, value, table, token_to_req, outputs):
        # Every window keeps the original shapes; masked rows are discarded.
        assert token_to_req.shape == case.token_to_req.shape
        calls.append(int((token_to_req >= 0).sum()))
        _triton_attention(
            case.query,
            key,
            value,
            case.indices,
            table,
            token_to_req,
            outputs,
            e4m3=True,
            splits=2,
        )

    actual = _outputs(case, True, 2)
    result = qsa_union_gather_attention(
        attention,
        actual,
        case.indices,
        case.token_to_req,
        case.block_table,
        case.key,
        case.value,
        arena,
        unit_tokens=unit,
    )
    assert result.windows is not None and result.windows > 1
    assert len(calls) == result.windows
    assert bool(result.complete)
    assert int(result.transfer_units) >= int(full.transfer_count)
    assert int(result.selected_tokens) == int(full.selected_tokens)
    _assert_bitwise(expected, actual)

    # Device-paced windows give the same bits without a host read.
    graph_actual = _outputs(case, True, 2)
    calls.clear()
    graph = qsa_union_gather_attention(
        attention,
        graph_actual,
        case.indices,
        case.token_to_req,
        case.block_table,
        case.key,
        case.value,
        arena,
        unit_tokens=unit,
        max_windows=result.windows + 1,
    )
    assert graph.windows is None and bool(graph.complete)
    _assert_bitwise(expected, graph_actual)

    # Too few device windows must be reported, not hidden.
    short = qsa_union_gather_attention(
        attention,
        _outputs(case, True, 2),
        case.indices,
        case.token_to_req,
        case.block_table,
        case.key,
        case.value,
        arena,
        unit_tokens=unit,
        max_windows=1,
    )
    assert not bool(short.complete)


@interpreter
def test_window_smaller_than_one_group_raises():
    case = make_case(41, rows=16, unit=4)
    arena = _arena(case, staging=1)

    def attention(*args):
        raise AssertionError("must not run")

    with pytest.raises(RuntimeError, match="eight-row group"):
        qsa_union_gather_attention(
            attention,
            _outputs(case, False),
            case.indices,
            case.token_to_req,
            case.block_table,
            case.key,
            case.value,
            arena,
        )


# --------------------------------------------------------------------------
# Grouped page4 plan (Python port of grouped_sparse_page4_plan_kernel)


def _active_m_tiles(mask):
    tiles = 0
    for query in range(8):
        if mask & (0xF << (query * 4)):
            first, last = query * 6, query * 6 + 5
            tiles |= 1 << (first // 16)
            tiles |= 1 << (last // 16)
    return tiles


def _grouped_plan(
    indices, table, token_to_req, positions, lengths, page, stride, blocks
):
    """Planner semantics: union by physical microblock, logical ordering."""
    indices, table = indices.tolist(), table.tolist()
    token_to_req, positions, lengths = (
        token_to_req.tolist(),
        positions.tolist(),
        lengths.tolist(),
    )
    width = len(indices[0])

    def physical(token, request):
        if token < 0 or token // page >= len(table[0]):
            return -1
        block = table[request][token // page]
        if not 0 <= block < blocks:
            return -1
        return block * stride + (token % page) // 4

    plans = []
    for group in range(len(indices) // 8):
        entries: dict[int, tuple[int, int]] = {}

        def insert(microblock, mask, query, token, entries=entries):
            if microblock < 0 or mask == 0:
                return
            owner = (query << 29) | (token >> 2)
            old_mask, old_owner = entries.get(microblock, (0, 2**32))
            entries[microblock] = (old_mask | mask, min(old_owner, owner))

        for query in range(8):
            row = group * 8 + query
            request = token_to_req[row]
            if not 0 <= request < len(table):
                continue
            length = lengths[request]
            visible = max(0, min(positions[row] + 1, length))
            complete = min(visible // 4, length // 4, width // 4)
            selected = indices[row]
            for page4 in range(complete):
                first = selected[page4 * 4]
                if first < 0:
                    continue
                full = (
                    selected[page4 * 4 + 1 : page4 * 4 + 4]
                    == [first + 1, first + 2, first + 3]
                    and first % 4 == 0
                    and first + 3 < length
                )
                if full:
                    insert(physical(first, request), 0xF << (query * 4), query, first)
                    continue
                for token in selected[page4 * 4 : page4 * 4 + 4]:
                    if 0 <= token < length:
                        insert(
                            physical(token, request),
                            1 << (query * 4 + (token & 3)),
                            query,
                            token,
                        )
            tail = visible & 3
            tail_token = selected[complete * 4] if complete * 4 < width else -1
            if tail and tail_token == visible // 4 * 4 and tail_token < length:
                insert(
                    physical(tail_token, request),
                    ((1 << tail) - 1) << (query * 4),
                    query,
                    tail_token,
                )
        ordered = sorted(
            entries.items(),
            key=lambda item: (_active_m_tiles(item[1][0]), item[1][1]),
        )
        plans.append([(microblock, mask) for microblock, (mask, _) in ordered])
    return plans


@interpreter
@pytest.mark.parametrize("dcp_local", [False, True])
def test_grouped_plan_port_is_unchanged(dcp_local):
    case = make_case(51, rows=24, width=28, page=8, blocks=24, unit=4, members=2)
    page, blocks = case.key.shape[:2]
    requests, table_width = case.block_table.shape
    capacity = table_width * page
    if dcp_local:
        positions = torch.full((24,), capacity - 1)
        lengths = torch.full((requests,), capacity)
    else:
        positions = torch.randint(
            0, capacity, (24,), generator=torch.Generator().manual_seed(5)
        )
        lengths = torch.full((requests,), capacity - 3)
    arena = _arena(case, hot=8, staging=96)
    resident = _resident(case, arena, 0.2, seed=9)
    plan = plan_qsa_union_gather(
        case.indices,
        case.token_to_req,
        case.block_table,
        arena,
        page_size=page,
        num_source_blocks=blocks,
        resident_microblocks=resident,
    )
    source_stride = case.key.stride(0) // (4 * DIM)
    arena_stride = arena.key.stride(0) // (4 * DIM)
    expected = _grouped_plan(
        case.indices,
        case.block_table,
        case.token_to_req,
        positions,
        lengths,
        page,
        source_stride,
        blocks,
    )
    actual = _grouped_plan(
        case.indices,
        plan.microblock_table,
        case.token_to_req,
        positions,
        lengths,
        4,
        arena_stride,
        arena.num_microblocks,
    )
    arena_to_source = {
        int(a): s for s, a in enumerate(plan.arena_of_source.tolist()) if a >= 0
    }
    per_page = page // 4
    assert any(expected), "the case must produce grouped entries"
    for expected_group, actual_group in zip(expected, actual, strict=True):
        assert [m for _, m in expected_group] == [m for _, m in actual_group]
        mapped = []
        for microblock, _ in actual_group:
            source = arena_to_source[microblock // arena_stride]
            block, first = divmod(source, per_page)
            mapped.append(block * source_stride + first)
        assert mapped == [m for m, _ in expected_group]


# --------------------------------------------------------------------------
# CUDA: the real kernels (run on V100)


def _sm70_flash():
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability(70):
        pytest.skip("SM70 grouped page4 route")
    return pytest.importorskip("flash_attn_v100.flash_attn_interface")


def _production_case(seed, rows, e4m3, packed, share=True):
    """DCP-local production geometry: 12 gathered heads, head dim 256."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    generator = torch.Generator().manual_seed(seed)
    page, pages, dim = 16, 96, 256
    members = 2 if packed else 1
    cache = torch.randn((2 * pages + 3, members, 2, page, 1, dim), generator=generator)
    k_scale, v_scale = (0.25, 0.5) if e4m3 else (1.0, 1.0)
    if e4m3:
        cache[:, :, 0] /= k_scale
        cache[:, :, 1] /= v_scale
        cache = cache.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        cache = cache.half()
    key, value = cache.cuda()[:, members - 1].unbind(1)
    table = (
        torch.randperm(2 * pages + 3, generator=generator)[: 2 * pages]
        .view(2, pages)
        .to(torch.int32)
    )
    if share:
        table[1, :4] = table[0, :4]  # shared prefix pages
    split = min(rows, 40)  # a multiple of eight: groups hold one request
    positions = torch.cat(
        [torch.arange(2400, 2400 + split), torch.arange(100, 100 + rows - split)]
    )
    token_to_req = torch.tensor([0] * split + [1] * (rows - split), dtype=torch.int32)
    width = 2051
    selection = []
    for position in positions.tolist():
        visible = (position + 1) // 4
        groups = torch.randperm(visible, generator=generator)[:512]
        tokens = (groups[:, None] * 4 + torch.arange(4)).flatten().tolist()
        tokens += list(range(visible * 4, position + 1))
        selection.append(tokens + [-1] * (width - len(tokens)))
    indices = torch.tensor(selection, dtype=torch.int32).cuda()
    indices[3] = -1
    local = torch.empty_like(indices)
    qsa_localize_dcp_indices(
        indices,
        local,
        dcp_world_size=2,
        dcp_rank=1,
        interleave_size=1,
        local_block_size=page,
    )
    q = (torch.randn((rows, 12, dim), generator=generator) * 0.3).half().cuda()
    return dict(
        q=q,
        key=key,
        value=value,
        local=local,
        table=table.cuda(),
        token_to_req=token_to_req.cuda(),
        k_scale=k_scale,
        v_scale=v_scale,
    )


def _dcp_attention(case, key, value, table, token_to_req, outputs, kv_dtype):
    """The kernel sequence of ``_forward_qsa_dcp`` before the cross-rank merge."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        qsa_dcp_sparse_paged_attention_sm70_grouped_page4,
        qsa_sparse_paged_attention,
    )
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import (
        qsa_dcp_local_selection_width,
    )

    out, lse = outputs
    q, local = case["q"], case["local"]
    width = qsa_dcp_local_selection_width(2048, 4, 2, 1, local.shape[1])
    grouped = qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
        q,
        key,
        value,
        local,
        table,
        token_to_req,
        out,
        lse,
        kv_dtype,
        case["k_scale"],
        case["v_scale"],
    )
    if grouped < q.shape[0]:
        qsa_sparse_paged_attention(
            q[grouped:],
            key,
            value,
            local[grouped:, :width],
            table,
            token_to_req[grouped:],
            out[grouped:],
            kv_cache_dtype=kv_dtype,
            k_scale=case["k_scale"],
            v_scale=case["v_scale"],
            lse=lse[grouped:],
        )
    return grouped


@cuda
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize("staging", [None, "windows"])
def test_cuda_dcp_grouped_route_bitwise(e4m3, packed, staging, monkeypatch):
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    flash = _sm70_flash()
    kv_dtype = "fp8_e4m3" if e4m3 else "auto"
    if not qsa_ops._qsa_grouped_page4_supported(flash.flash_attn_v100_cuda, kv_dtype):
        pytest.skip("Flash-V100 build has no grouped page4 route for this cache")
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_DCP_GROUPED_PAGE4_MIN_ROWS", 64)
    case = _production_case(816, 77, e4m3, packed, share=staging is None)
    rows = case["q"].shape[0]

    def fresh():
        return [
            torch.full(case["q"].shape, torch.nan, dtype=torch.float32, device="cuda"),
            torch.full((rows, 12), torch.nan, dtype=torch.float32, device="cuda"),
        ]

    expected = fresh()
    grouped = _dcp_attention(
        case,
        case["key"],
        case["value"],
        case["table"],
        case["token_to_req"],
        expected,
        kv_dtype,
    )
    assert grouped == 72
    page, blocks = case["key"].shape[1], case["key"].shape[0]
    probe = allocate_qsa_arena(
        hot_microblocks=0,
        staging_microblocks=blocks * page,
        kv_heads=1,
        head_dim=256,
        dtype=case["key"].dtype,
        device="cuda",
    )
    full = plan_qsa_union_gather(
        case["local"],
        case["token_to_req"],
        case["table"],
        probe,
        page_size=page,
        num_source_blocks=blocks,
        unit_tokens=2,
    )
    capacity = int(full.miss_microblocks)
    if staging == "windows":
        # The smallest staging that still fits every eight-row group.
        groups = torch.arange(rows, device="cuda") // 8
        capacity = max(
            int(
                plan_qsa_union_gather(
                    case["local"],
                    case["token_to_req"],
                    case["table"],
                    probe,
                    page_size=page,
                    num_source_blocks=blocks,
                    unit_tokens=2,
                    row_mask=groups == group,
                ).miss_microblocks
            )
            for group in range((rows + 7) // 8)
        )
        assert capacity < int(full.miss_microblocks)
    arena = allocate_qsa_arena(
        hot_microblocks=0,
        staging_microblocks=capacity,
        kv_heads=1,
        head_dim=256,
        dtype=case["key"].dtype,
        device="cuda",
    )
    actual = fresh()
    result = qsa_union_gather_attention(
        lambda k, v, t, r, o: _dcp_attention(case, k, v, t, r, o, kv_dtype),
        actual,
        case["local"],
        case["token_to_req"],
        case["table"],
        case["key"],
        case["value"],
        arena,
        unit_tokens=2,
    )
    assert bool(result.complete)
    if staging == "windows":
        assert result.windows is not None and result.windows > 1
    _assert_bitwise(expected, actual)


@cuda
@pytest.mark.parametrize("e4m3", [False, True])
def test_cuda_public_attention_bitwise(e4m3):
    """``qsa_sparse_paged_attention`` with real positions (DCP1 routes)."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    case = _production_case(917, 40, e4m3, packed=False)
    # DCP1 sees global tokens; reuse the localized buffer as a DCP1 selection.
    kv_dtype = "fp8_e4m3" if e4m3 else "auto"
    q = case["q"][:, :6].contiguous()
    rows = q.shape[0]
    positions = torch.full((rows,), 1599, dtype=torch.int64, device="cuda")
    lengths = torch.full((2,), 1600, dtype=torch.int32, device="cuda")

    def run(key, value, table, token_to_req, outputs):
        qsa_sparse_paged_attention(
            q,
            key,
            value,
            case["local"],
            table,
            token_to_req,
            outputs[0],
            query_positions=positions,
            sequence_lengths=lengths,
            kv_cache_dtype=kv_dtype,
            k_scale=case["k_scale"],
            v_scale=case["v_scale"],
        )

    expected = [torch.zeros_like(q)]
    run(case["key"], case["value"], case["table"], case["token_to_req"], expected)
    arena = allocate_qsa_arena(
        hot_microblocks=0,
        staging_microblocks=4096,
        kv_heads=1,
        head_dim=256,
        dtype=case["key"].dtype,
        device="cuda",
    )
    actual = [torch.zeros_like(q)]
    qsa_union_gather_attention(
        run,
        actual,
        case["local"],
        case["token_to_req"],
        case["table"],
        case["key"],
        case["value"],
        arena,
        union_bound=4096,
    )
    _assert_bitwise(expected, actual)


@cuda
def test_cuda_gather_from_pinned_host_pool():
    """Plain vector loads read a UVA-addressable pinned host source."""
    case = make_case(61, unit=2, e4m3=True)
    host = torch.empty(
        (case.key.shape[0], 2, *case.key.shape[1:]),
        dtype=torch.uint8,
        pin_memory=True,
    )
    host[:, 0].copy_(case.key.cpu())
    host[:, 1].copy_(case.value.cpu())
    host_key, host_value = host.unbind(1)
    arena = _arena(case, staging=64)
    reference = _arena(case, staging=64)
    plan = plan_qsa_union_gather(
        case.indices,
        case.token_to_req,
        case.block_table,
        arena,
        page_size=case.key.shape[1],
        num_source_blocks=case.key.shape[0],
        unit_tokens=2,
    )
    qsa_union_gather_copy(host_key, host_value, arena, plan)
    qsa_union_gather_copy(case.key, case.value, reference, plan)
    torch.accelerator.synchronize()
    assert torch.equal(arena.key, reference.key)
    assert torch.equal(arena.value, reference.value)
