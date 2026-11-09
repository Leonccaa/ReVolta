# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk-level union gather of QSA main K/V into a GPU arena (host KV selection-union
gather prototype).

A QSA prefill chunk has hundreds to thousands of query rows. Each row selects
up to 2,051 history tokens, and neighbouring rows select mostly the same ones.
When the main K/V history lives in host memory, gathering per row would move
the same token many times. This module plans one transfer for the whole chunk:

1. Resolve every selected (row, column) entry to its source token through the
   block table, exactly as the sparse attention kernels resolve it.
2. Take the union of the referenced source microblocks (4 local tokens, the
   unit the grouped page4 planner deduplicates and the forward kernel loads).
   Microblocks already resident in the arena (hot pages) are hits; every other
   referenced microblock gets one staging slot in the arena.
3. Copy only the selected units (``unit_tokens`` consecutive tokens) of the
   missing microblocks from the source into their staging slots, once.
4. Build a per-request microblock table (a block table with page size 4) that
   points every referenced logical microblock at its arena slot.

The attention kernels then run unchanged on ``(arena.key, arena.value,
microblock_table)`` with the original selection: indices keep their logical
values, order, duplicates and -1 slots, and the table keeps the original
logical capacity (``table_width * page_size`` tokens). Two (request, logical
microblock) pairs share an arena microblock exactly when they share a source
microblock, so the grouped page4 plan (deduplicated by physical microblock,
ordered by logical owner) keeps its entries, masks and order.

All planning uses fixed shapes and no host synchronisation. When the missing
microblocks of a chunk exceed the staging capacity, rows are processed in
windows of whole eight-row groups (the grouped page4 planning unit). Rows
outside a window are masked by setting their request to -1, so every window
call keeps the original shapes, routes and launch parameters, and each row is
computed by exactly one window with its full selection.

Nothing in the model calls this module yet; the host-history integration integration
wires it into
``forward_qsa``. INT8 caches (per-token scale caches) are not supported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

MICROBLOCK_TOKENS = 4
# The grouped page4 planner deduplicates microblocks per group of eight rows.
ROW_GROUP = 8


@dataclass(frozen=True)
class QSAArena:
    """GPU K/V microblocks that the sparse attention kernels read.

    ``key`` and ``value`` are ``[num_microblocks, 4, kv_heads, head_dim]``
    views with contiguous tokens. Microblocks ``[0, staging_base)`` belong to
    the hot page pool (owned by the host KV pool, host KV pool); microblocks
    ``[staging_base, staging_base + staging_microblocks)`` are the staging area
    this module fills.
    """

    key: torch.Tensor
    value: torch.Tensor
    staging_base: int
    staging_microblocks: int

    @property
    def num_microblocks(self) -> int:
        return self.key.shape[0]


def allocate_qsa_arena(
    *,
    hot_microblocks: int,
    staging_microblocks: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> QSAArena:
    """Allocate a zero-filled arena with K/V interleaved per microblock.

    The layout matches the model cache ABI ``[blocks, 2, page, heads, dim]``
    with a page of four tokens.
    """
    if hot_microblocks < 0 or staging_microblocks < 0:
        raise ValueError("QSA arena regions must be non-negative")
    if hot_microblocks + staging_microblocks == 0:
        raise ValueError("QSA arena must hold at least one microblock")
    storage = torch.zeros(
        (
            hot_microblocks + staging_microblocks,
            2,
            MICROBLOCK_TOKENS,
            kv_heads,
            head_dim,
        ),
        dtype=dtype,
        device=device,
    )
    key, value = storage.unbind(1)
    return QSAArena(key, value, hot_microblocks, staging_microblocks)


@dataclass(frozen=True)
class QSAUnionGatherPlan:
    """Device-side plan for one union gather; every field has a fixed shape."""

    # [num_requests, table_width * page_size // 4] int32: arena microblock of
    # every referenced logical microblock, -1 elsewhere.
    microblock_table: torch.Tensor
    # [num_source_microblocks] int64: arena microblock of each referenced
    # source microblock, -1 for unreferenced ones and staging overflow.
    arena_of_source: torch.Tensor
    # [staging_microblocks * units_per_microblock] int64, ascending source
    # units; entries at and after ``transfer_count`` are -1.
    transfer_source_units: torch.Tensor
    transfer_arena_units: torch.Tensor
    transfer_count: torch.Tensor  # 0-d int64
    union_microblocks: torch.Tensor  # 0-d int64, hits and misses
    miss_microblocks: torch.Tensor  # 0-d int64, staging slots needed
    selected_tokens: torch.Tensor  # 0-d int64, valid entries (per-row reads)
    overflow: torch.Tensor  # 0-d bool, misses exceed the staging capacity
    unit_tokens: int


@dataclass(frozen=True)
class QSAUnionGatherResult:
    """Totals over every window of one union-gather attention call."""

    windows: int | None  # None when the window count stays on the device
    transfer_units: torch.Tensor  # 0-d int64
    selected_tokens: torch.Tensor  # 0-d int64
    complete: torch.Tensor  # 0-d bool: every row group was computed


def _check_views(name: str, key: torch.Tensor, value: torch.Tensor) -> None:
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError(f"QSA {name} K/V must be matching 4-D paged views")
    if key.dtype != value.dtype or key.device != value.device:
        raise ValueError(f"QSA {name} K/V must share dtype and device")
    heads, dim = key.shape[2:]
    for view in (key, value):
        if (
            view.stride(3) != 1
            or view.stride(2) != dim
            or view.stride(1) != heads * dim
        ):
            raise ValueError(f"QSA {name} K/V tokens must be contiguous in a page")


def _check_arena(arena: QSAArena) -> None:
    _check_views("arena", arena.key, arena.value)
    if arena.key.shape[1] != MICROBLOCK_TOKENS:
        raise ValueError("QSA arena pages must hold four tokens")
    if arena.key.stride(0) != arena.value.stride(0):
        raise ValueError("QSA arena K and V must share the microblock stride")
    if (
        arena.staging_base < 0
        or arena.staging_microblocks < 0
        or arena.staging_base + arena.staging_microblocks > arena.num_microblocks
    ):
        raise ValueError("QSA arena staging region is out of range")


def _check_metadata(
    indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    num_source_blocks: int,
    unit_tokens: int,
) -> None:
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("QSA union gather needs int32 [rows, width] indices")
    if token_to_req.shape != indices.shape[:1] or token_to_req.dtype != torch.int32:
        raise ValueError("QSA union gather needs one int32 request per row")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA union gather needs an int32 block table")
    if not all(block_table.shape):
        raise ValueError("QSA union gather block table must be nonempty")
    if page_size <= 0 or page_size % MICROBLOCK_TOKENS:
        raise ValueError("QSA union gather page size must be a multiple of four")
    if num_source_blocks <= 0:
        raise ValueError("QSA union gather needs a positive source block count")
    if unit_tokens not in (1, 2, 4):
        raise ValueError("QSA union gather transfer unit must be 1, 2 or 4 tokens")
    device = indices.device
    if token_to_req.device != device or block_table.device != device:
        raise ValueError("QSA union gather metadata must share one device")


def _resolve(
    indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    num_source_blocks: int,
    row_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve entries as the sparse kernels do.

    Returns the validity mask, the source token of every valid entry (0 for
    invalid ones), and the flat (request, logical microblock) table slot.
    """
    tokens = indices.to(torch.int64)
    requests = token_to_req.to(torch.int64)[:, None]
    num_requests, table_width = block_table.shape
    valid = (requests >= 0) & (requests < num_requests) & (tokens >= 0)
    if row_mask is not None:
        valid &= row_mask[:, None]
    safe_tokens = tokens.clamp_min(0)
    logical_pages = safe_tokens // page_size
    valid &= logical_pages < table_width
    safe_requests = requests.clamp(0, num_requests - 1)
    pages = block_table[safe_requests, logical_pages.clamp_max(table_width - 1)].to(
        torch.int64
    )
    valid &= (pages >= 0) & (pages < num_source_blocks)
    source_tokens = torch.where(valid, pages * page_size + safe_tokens % page_size, 0)
    table_microblocks = table_width * page_size // MICROBLOCK_TOKENS
    logical_slots = safe_requests * table_microblocks + safe_tokens // MICROBLOCK_TOKENS
    return valid, source_tokens, logical_slots


def _resident_lookup(
    resident_microblocks: torch.Tensor | None,
    num_microblocks: int,
    device: torch.device,
) -> torch.Tensor:
    if resident_microblocks is None:
        return torch.full((num_microblocks,), -1, dtype=torch.int64, device=device)
    if resident_microblocks.shape != (num_microblocks,):
        raise ValueError("QSA resident lookup needs one entry per source microblock")
    if resident_microblocks.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA resident lookup must hold integer arena microblocks")
    if resident_microblocks.device != device:
        raise ValueError("QSA resident lookup must be on the metadata device")
    return resident_microblocks.to(torch.int64)


def plan_qsa_union_gather(
    indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    arena: QSAArena,
    *,
    page_size: int,
    num_source_blocks: int,
    unit_tokens: int = MICROBLOCK_TOKENS,
    resident_microblocks: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
) -> QSAUnionGatherPlan:
    """Plan the union gather of one attention call without host sync.

    ``indices`` are the token IDs the kernels receive (DCP-local after
    localization), ``block_table`` maps logical pages of ``page_size`` tokens
    to source blocks ``[0, num_source_blocks)``. ``resident_microblocks``
    maps a source microblock (``block * page_size // 4 + offset // 4``) to the
    arena microblock that already holds all four of its tokens, or -1.
    ``unit_tokens`` is the transfer granularity inside a microblock: the
    unselected units of a staged microblock keep stale finite bytes, which the
    kernels never attend (Triton reads selected tokens only; grouped page4
    zeroes K/V rows that no query of the group attends).
    """
    _check_metadata(
        indices, token_to_req, block_table, page_size, num_source_blocks, unit_tokens
    )
    _check_arena(arena)
    if arena.key.device != indices.device:
        raise ValueError("QSA arena must be on the metadata device")
    device = indices.device
    units_per_microblock = MICROBLOCK_TOKENS // unit_tokens
    num_microblocks = num_source_blocks * page_size // MICROBLOCK_TOKENS
    num_units = num_microblocks * units_per_microblock
    resident = _resident_lookup(resident_microblocks, num_microblocks, device)
    capacity = arena.staging_microblocks

    valid, source_tokens, logical_slots = _resolve(
        indices, token_to_req, block_table, page_size, num_source_blocks, row_mask
    )
    # Mark referenced units in a dense source-space map; the extra slot takes
    # every invalid entry, so no boolean indexing (and no sync) is needed.
    source_units = torch.where(valid, source_tokens // unit_tokens, num_units)
    unit_marks = torch.zeros(num_units + 1, dtype=torch.int32, device=device)
    unit_marks.index_fill_(0, source_units.flatten(), 1)
    unit_marks = unit_marks[:num_units].view(num_microblocks, units_per_microblock)
    referenced = unit_marks.amax(dim=1) > 0
    hit = referenced & (resident >= 0)
    miss = referenced & (resident < 0)
    miss_rank = torch.cumsum(miss, 0, dtype=torch.int64) - 1
    staged = miss & (miss_rank < capacity)
    arena_of_source = torch.where(
        hit,
        resident,
        torch.where(staged, arena.staging_base + miss_rank, -1),
    )

    # Transfer list: the selected units of every staged microblock, in
    # ascending source order (one pass over the source pages).
    needed = ((unit_marks > 0) & staged[:, None]).flatten()
    unit_rank = torch.cumsum(needed, 0, dtype=torch.int64) - 1
    list_size = capacity * units_per_microblock
    list_slots = torch.where(needed, unit_rank, list_size)
    all_units = torch.arange(num_units, dtype=torch.int64, device=device)
    transfer_source = torch.full((list_size + 1,), -1, dtype=torch.int64, device=device)
    transfer_source.index_put_((list_slots,), all_units)
    transfer_source = transfer_source[:list_size]
    safe_source = transfer_source.clamp_min(0)
    arena_microblocks = arena_of_source[safe_source // units_per_microblock]
    transfer_arena = torch.where(
        transfer_source >= 0,
        arena_microblocks * units_per_microblock + safe_source % units_per_microblock,
        -1,
    )

    num_requests, table_width = block_table.shape
    table_size = num_requests * (table_width * page_size // MICROBLOCK_TOKENS)
    table = torch.full((table_size + 1,), -1, dtype=torch.int32, device=device)
    targets = torch.where(valid, logical_slots, table_size)
    # Every valid write to one slot carries the same value (same source page).
    values = arena_of_source[source_tokens // MICROBLOCK_TOKENS].to(torch.int32)
    table.index_put_((targets.flatten(),), values.flatten())
    microblock_table = table[:table_size].view(num_requests, -1)

    miss_count = miss.sum()
    return QSAUnionGatherPlan(
        microblock_table=microblock_table,
        arena_of_source=arena_of_source,
        transfer_source_units=transfer_source,
        transfer_arena_units=transfer_arena,
        transfer_count=needed.sum(),
        union_microblocks=referenced.sum(),
        miss_microblocks=miss_count,
        selected_tokens=valid.sum(),
        overflow=miss_count > capacity,
        unit_tokens=unit_tokens,
    )


if HAS_TRITON:

    @triton.jit
    def _qsa_union_gather_kernel(
        source_key_ptr,
        source_value_ptr,
        arena_key_ptr,
        arena_value_ptr,
        source_units_ptr,
        arena_units_ptr,
        count_ptr,
        stride_source_key_block,
        stride_source_value_block,
        stride_arena_key_block,
        stride_arena_value_block,
        UNITS_PER_PAGE: tl.constexpr,
        UNITS_PER_MICROBLOCK: tl.constexpr,
        UNIT_NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        # One program per list slot; slots past the device-side count exit, so
        # the launch shape is fixed and capture-safe. Plain vector loads also
        # serve a UVA-mapped host source.
        slot = tl.program_id(0)
        count = tl.load(count_ptr)
        if slot < count:
            source_unit = tl.load(source_units_ptr + slot).to(tl.int64)
            arena_unit = tl.load(arena_units_ptr + slot).to(tl.int64)
            page = source_unit // UNITS_PER_PAGE
            in_page = (source_unit % UNITS_PER_PAGE) * UNIT_NUMEL
            microblock = arena_unit // UNITS_PER_MICROBLOCK
            in_microblock = (arena_unit % UNITS_PER_MICROBLOCK) * UNIT_NUMEL
            source_key = source_key_ptr + page * stride_source_key_block + in_page
            source_value = source_value_ptr + page * stride_source_value_block + in_page
            arena_key = (
                arena_key_ptr + microblock * stride_arena_key_block + in_microblock
            )
            arena_value = (
                arena_value_ptr + microblock * stride_arena_value_block + in_microblock
            )
            for start in tl.static_range(0, UNIT_NUMEL, BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                mask = offsets < UNIT_NUMEL
                tl.store(
                    arena_key + offsets,
                    tl.load(source_key + offsets, mask=mask),
                    mask=mask,
                )
                tl.store(
                    arena_value + offsets,
                    tl.load(source_value + offsets, mask=mask),
                    mask=mask,
                )


def qsa_union_gather_copy(
    source_key: torch.Tensor,
    source_value: torch.Tensor,
    arena: QSAArena,
    plan: QSAUnionGatherPlan,
) -> None:
    """Copy the planned units from the source pages into the staging slots.

    ``source_key``/``source_value`` are ``[blocks, page_size, heads, dim]``
    views of the authoritative copy: the GPU cache in the reference setup, or
    UVA-addressable host pool pages in the host KV setup.
    """
    _check_views("source", source_key, source_value)
    _check_arena(arena)
    if source_key.shape[2:] != arena.key.shape[2:]:
        raise ValueError("QSA source and arena head geometry differ")
    if source_key.dtype != arena.key.dtype:
        raise ValueError("QSA source and arena dtypes differ")
    if source_key.shape[1] % plan.unit_tokens:
        raise ValueError("QSA source page size must hold whole transfer units")
    list_size = plan.transfer_source_units.shape[0]
    if list_size == 0:
        return
    unit_numel = plan.unit_tokens * source_key.shape[2] * source_key.shape[3]
    _qsa_union_gather_kernel[(list_size,)](
        source_key,
        source_value,
        arena.key,
        arena.value,
        plan.transfer_source_units,
        plan.transfer_arena_units,
        plan.transfer_count,
        source_key.stride(0),
        source_value.stride(0),
        arena.key.stride(0),
        arena.value.stride(0),
        UNITS_PER_PAGE=source_key.shape[1] // plan.unit_tokens,
        UNITS_PER_MICROBLOCK=MICROBLOCK_TOKENS // plan.unit_tokens,
        UNIT_NUMEL=unit_numel,
        BLOCK=min(triton.next_power_of_2(unit_numel), 1024),
    )


def qsa_union_gather_copy_reference(
    source_key: torch.Tensor,
    source_value: torch.Tensor,
    arena: QSAArena,
    plan: QSAUnionGatherPlan,
) -> None:
    """Torch reference of ``qsa_union_gather_copy`` (synchronizes the host)."""
    _check_views("source", source_key, source_value)
    count = int(plan.transfer_count)
    if count == 0:
        return
    unit = plan.unit_tokens
    units_per_page = source_key.shape[1] // unit
    units_per_microblock = MICROBLOCK_TOKENS // unit
    offsets = torch.arange(unit, device=source_key.device)
    source = plan.transfer_source_units[:count]
    pages = (source // units_per_page)[:, None]
    tokens = (source % units_per_page * unit)[:, None] + offsets
    target = plan.transfer_arena_units[:count]
    microblocks = (target // units_per_microblock)[:, None]
    arena_tokens = (target % units_per_microblock * unit)[:, None] + offsets
    arena.key[microblocks, arena_tokens] = source_key[pages, tokens]
    arena.value[microblocks, arena_tokens] = source_value[pages, tokens]


# --------------------------------------------------------------------------
# Row windows for chunks whose missing microblocks exceed the staging area.


@dataclass(frozen=True)
class _WindowPairs:
    """Sorted distinct (missing source microblock, row group) pairs."""

    groups: torch.Tensor
    live: torch.Tensor
    continues: torch.Tensor  # same microblock as the previous pair
    previous_groups: torch.Tensor
    num_groups: int


def _window_pairs(
    indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    num_source_blocks: int,
    resident: torch.Tensor,
) -> _WindowPairs:
    valid, source_tokens, _ = _resolve(
        indices, token_to_req, block_table, page_size, num_source_blocks, None
    )
    microblocks = source_tokens // MICROBLOCK_TOKENS
    # Hits never use staging, so only misses constrain a window.
    valid &= resident[microblocks] < 0
    rows = indices.shape[0]
    num_groups = math.ceil(rows / ROW_GROUP)
    num_microblocks = resident.shape[0]
    groups = torch.arange(rows, dtype=torch.int64, device=indices.device) // ROW_GROUP
    sentinel = num_microblocks * num_groups
    keys = torch.where(valid, microblocks * num_groups + groups[:, None], sentinel)
    keys = keys.flatten().sort().values
    pair_microblocks = keys // num_groups
    pair_groups = keys % num_groups
    continues = torch.zeros_like(keys, dtype=torch.bool)
    continues[1:] = pair_microblocks[1:] == pair_microblocks[:-1]
    previous_groups = torch.full_like(pair_groups, -1)
    previous_groups[1:] = pair_groups[:-1]
    return _WindowPairs(
        pair_groups, keys < sentinel, continues, previous_groups, num_groups
    )


def _next_window_end(
    pairs: _WindowPairs, start: torch.Tensor, capacity: int
) -> torch.Tensor:
    """Largest row-group end whose window from ``start`` fits the staging.

    A missing microblock enters the window at the first row group, at or after
    ``start``, that references it. The greedy end is the first group at which
    the running count of entering microblocks would exceed ``capacity``.
    Returns ``start`` itself when that group alone does not fit.
    """
    in_range = pairs.live & (pairs.groups >= start)
    enters = in_range & ~(pairs.continues & (pairs.previous_groups >= start))
    histogram = torch.zeros(
        pairs.num_groups + 1, dtype=torch.int64, device=pairs.groups.device
    )
    histogram.index_add_(
        0,
        torch.where(enters, pairs.groups, pairs.num_groups),
        enters.to(torch.int64),
    )
    entered = histogram[: pairs.num_groups].cumsum(0)
    # Groups before ``start`` count zero, so this is the fitting prefix.
    return (entered <= capacity).sum()


AttentionFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Sequence[torch.Tensor]],
    None,
]


def qsa_union_gather_attention(
    attention: AttentionFn,
    outputs: Sequence[torch.Tensor],
    indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    source_key: torch.Tensor,
    source_value: torch.Tensor,
    arena: QSAArena,
    *,
    unit_tokens: int = MICROBLOCK_TOKENS,
    resident_microblocks: torch.Tensor | None = None,
    union_bound: int | None = None,
    max_windows: int | None = None,
    copy: Callable[
        [torch.Tensor, torch.Tensor, QSAArena, QSAUnionGatherPlan], None
    ] = qsa_union_gather_copy,
) -> QSAUnionGatherResult:
    """Run ``attention`` on arena copies of the selected source K/V.

    ``attention(key, value, microblock_table, token_to_req, outputs)`` must
    run the unchanged sparse kernels with the original ``indices`` and write
    every row of ``outputs`` (rows with request -1 are discarded).

    Window policy:
    - ``union_bound`` (a host-known upper bound on missing microblocks, e.g.
      from the requests' history lengths) at most the staging capacity: one
      window, no host sync.
    - ``max_windows`` set: device-side windows, no host sync; ``complete``
      reports whether they covered every row group. The caller must check it
      (or size ``max_windows`` so that it always holds).
    - otherwise: plan the whole chunk, read its overflow flag (one sync) and,
      if needed, run host-paced windows (one sync per window).
    """
    page_size = source_key.shape[1]
    num_source_blocks = source_key.shape[0]
    capacity = arena.staging_microblocks
    device = indices.device
    num_microblocks = num_source_blocks * page_size // MICROBLOCK_TOKENS

    def make_plan(row_mask: torch.Tensor | None = None) -> QSAUnionGatherPlan:
        return plan_qsa_union_gather(
            indices,
            token_to_req,
            block_table,
            arena,
            page_size=page_size,
            num_source_blocks=num_source_blocks,
            unit_tokens=unit_tokens,
            resident_microblocks=resident_microblocks,
            row_mask=row_mask,
        )

    def run_single(plan: QSAUnionGatherPlan) -> QSAUnionGatherResult:
        copy(source_key, source_value, arena, plan)
        attention(arena.key, arena.value, plan.microblock_table, token_to_req, outputs)
        return QSAUnionGatherResult(
            1,
            plan.transfer_count,
            plan.selected_tokens,
            ~plan.overflow,
        )

    if union_bound is not None and union_bound <= capacity:
        plan = make_plan()
        return run_single(plan)

    if max_windows is None:
        plan = make_plan()
        if not bool(plan.overflow):
            return run_single(plan)

    resident = _resident_lookup(resident_microblocks, num_microblocks, device)
    pairs = _window_pairs(
        indices, token_to_req, block_table, page_size, num_source_blocks, resident
    )
    groups = torch.arange(indices.shape[0], device=device) // ROW_GROUP
    temporaries = [torch.empty_like(output) for output in outputs]
    transfer_units = torch.zeros((), dtype=torch.int64, device=device)
    selected_tokens = torch.zeros((), dtype=torch.int64, device=device)

    def run_window(row_mask: torch.Tensor) -> None:
        nonlocal transfer_units, selected_tokens
        plan = make_plan(row_mask)
        copy(source_key, source_value, arena, plan)
        masked_requests = torch.where(row_mask, token_to_req, -1)
        attention(
            arena.key, arena.value, plan.microblock_table, masked_requests, temporaries
        )
        for output, temporary in zip(outputs, temporaries, strict=True):
            mask = row_mask.view(-1, *([1] * (output.ndim - 1)))
            output.copy_(torch.where(mask, temporary, output))
        transfer_units = transfer_units + plan.transfer_count
        selected_tokens = selected_tokens + plan.selected_tokens

    if max_windows is not None:
        start = torch.zeros((), dtype=torch.int64, device=device)
        for _ in range(max_windows):
            end = torch.maximum(_next_window_end(pairs, start, capacity), start)
            run_window((groups >= start) & (groups < end))
            start = end
        complete = start >= pairs.num_groups
        return QSAUnionGatherResult(None, transfer_units, selected_tokens, complete)

    windows = 0
    start_group = 0
    while start_group < pairs.num_groups:
        start = torch.tensor(start_group, dtype=torch.int64, device=device)
        end_group = int(_next_window_end(pairs, start, capacity))
        if end_group <= start_group:
            raise RuntimeError(
                "QSA union gather: one eight-row group needs more staging "
                f"microblocks than the {capacity} available"
            )
        run_window((groups >= start_group) & (groups < end_group))
        windows += 1
        start_group = end_group
    return QSAUnionGatherResult(
        windows,
        transfer_units,
        selected_tokens,
        torch.ones((), dtype=torch.bool, device=device),
    )
