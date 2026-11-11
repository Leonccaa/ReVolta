# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical device bytes of the block-addressed host history store.

The hot-pool budget covers fixed tensors, including shared scratch and write
staging. Per-history-ID tensors are charged to the history segment instead.
Counts exclude allocator rounding, transient tensors and attention workspaces
outside the store.
This module needs neither Torch nor a CUDA context.
"""

# 512 selected groups plus one open group per row; the newest 32 tokens of a
# request touch at most nine four-token groups.
QSA_ROW_GROUPS = 513
TAIL_GROUPS = 9
READ_ROWS = 256
HISTORY_METADATA = frozenset(
    {"ready", "generation", "group_to_slot", "wanted", "page_of"}
)
# shared hot region shared region: slot payload aliases the index pages, and these
# per-slot
# tensors scale with history IDs; both are billed per ID, not as fixed bytes.
SHARED_PER_SLOT = frozenset(
    {
        "hot",
        "slot_owner",
        "age",
        "wanted_groups",
        "miss_groups",
        "miss_slots",
        "victim_slots",
        "hot_block_table",
    }
)


def history_metadata_bytes(local_tokens: int, num_layers: int) -> dict[str, int]:
    """Device bytes per history block ID, including the null block."""
    return {
        "ready": local_tokens,
        "generation": 4,
        "page_of": 4,
        "group_to_slot": num_layers * local_tokens,
        "wanted": local_tokens,
    }


def history_fixed_memory_bytes(
    *,
    slots: int,
    num_layers: int,
    slot_bytes: int,
    max_requests: int,
    selection_width: int,
    max_write_rows: int,
    row_groups: int = QSA_ROW_GROUPS,
    tail_groups: int = TAIL_GROUPS,
    read_rows: int | None = READ_ROWS,
    staged_writethrough: bool = False,
) -> dict[str, int]:
    """Exact bytes by tensor name; ``slot_bytes`` includes every layer.

    Also accepts zero slots / less than one pass for capacity inversion;
    HostHistoryShape.validate rejects those before allocation.
    """
    tiles = min((slots + 511) // 512, 256)
    rows = (
        max((slots - max_requests * tail_groups) // row_groups, 0)
        if read_rows is None
        else read_rows
    )
    parts = {
        "hot": slots * slot_bytes,
        "slot_owner": 4 * num_layers * slots,
        "age": 8 * num_layers * slots,
        "clock": 8 * num_layers,
        "eviction_hand": 4 * num_layers,
        "wanted_groups": 4 * slots,
        "wanted_count": 4,
        "miss_groups": 4 * slots,
        "miss_slots": 4 * slots,
        "miss_count": 4,
        "victim_slots": 4 * slots,
        "victim_counts": 12 * tiles,
        "indices": 4 * rows * selection_width,
        "hot_block_table": 4 * max_requests * slots,
        "staging": max_write_rows * (slot_bytes // (4 * num_layers)),
        "staging_slots": 8 * max_write_rows,
    }

    if staged_writethrough:
        parts["write_stage"] = parts["staging"]
        parts["write_targets"] = 16 * max_write_rows
    return parts


def union_gather_memory_bytes(
    *,
    table_rows: int,
    table_tokens: int,
    max_tokens: int,
    max_requests: int,
    output_heads: int,
    head_dim: int,
    dcp_world_size: int = 1,
    output_itemsize: int = 2,
) -> dict[str, int]:
    """Peak device bytes the selection-union gather union path adds per worker (one
    layer at a
    time, shared by the layers; everything is reused across windows).

    ``table_tokens`` is the block table's logical capacity in rank-local
    tokens (``width * kernel_block_tokens``), ``table_rows`` its rows.
    Multi-window steps hold a temporary copy of the attention outputs (DCP:
    FP32 partial output and LSE of all gathered heads; otherwise the gated
    output rows); windows merge by row slice, so no ``where`` result is
    allocated. Grouped page4 calls with whole-batch shapes grow its cached
    plan workspace (pages/masks, 8 bytes x 4160 per eight-row group) and the
    DCP wrapper's FP16 slice output; both are billed here because the remap
    passes never reached them. The ``passes`` arm allocates none of this.
    """
    if min(table_rows, table_tokens, max_tokens, max_requests, output_heads) < 1:
        raise ValueError("Invalid union-gather byte geometry")
    dcp = dcp_world_size > 1
    bill = {
        "microblock_table": 4 * table_rows * (table_tokens // 4),
        "union_scan": 4 * (max_tokens + 1),
        "union_counters": 8 * 4,
        "masked_requests": 4 * max_tokens,
        "request_classes": 4 * max_requests,
        "window_output": max_tokens
        * output_heads
        * head_dim
        * (4 if dcp else output_itemsize),
        "window_lse": 4 * max_tokens * output_heads if dcp else 0,
        "grouped_plan_workspace": (max_tokens // 8) * 4160 * 8,
        "grouped_slice_output": (
            max_tokens * (output_heads // dcp_world_size) * head_dim * 2 if dcp else 0
        ),
    }
    return bill


def shared_history_memory_bytes(
    *,
    num_blocks: int,
    page_strides: tuple[int, ...],
    slot_bytes_per_layer: int,
    local_tokens: int,
    max_requests: int,
    selection_width: int,
    max_write_rows: int,
    read_rows: int = READ_ROWS,
    shadow_bytes_per_group: int = 1,
    counter_bytes: int = 64,
) -> tuple[dict[str, int], dict[str, int]]:
    """Frozen v2 shared-arena bill: per-ID and fixed device bytes.

    This is a sizing interface, not a shared hot region allocator. Index/hot payload
    alias
    one allocation, billed only as P_pad. num_blocks includes null. Shadow
    and counter widths are explicit reservations for later phases; changing
    their representation requires rebilling. Transient/allocator costs are
    not silently included in logical storage bytes.
    """
    if (
        num_blocks < 1
        or not page_strides
        or min(page_strides) < 1
        or slot_bytes_per_layer < 1
        or local_tokens < 1
        or local_tokens % 4
        or max_requests < 1
        or read_rows < 1
        or shadow_bytes_per_group < 0
        or counter_bytes < 0
        or any(p % slot_bytes_per_layer for p in page_strides)
        or len(set(page_strides)) != 1
    ):
        raise ValueError("Invalid shared host-history byte geometry")
    layers = len(page_strides)
    k = page_strides[0] // slot_bytes_per_layer
    per_block = {
        "P_pad": sum(page_strides),
        **history_metadata_bytes(local_tokens, layers),
        "page_of": 4,
        "shadow": (local_tokens // 4) * shadow_bytes_per_group,
        "slot_owner": 4 * layers * k,
        "age": 8 * layers * k,
        "wanted_groups": 4 * k,
        "miss_groups": 4 * k,
        "miss_slots": 4 * k,
        "victim_slots": 4 * k,
        "hot_block_table": 4 * max_requests * k,
    }
    fixed = history_fixed_memory_bytes(
        slots=num_blocks * k,
        num_layers=layers,
        slot_bytes=layers * slot_bytes_per_layer,
        max_requests=max_requests,
        selection_width=selection_width,
        max_write_rows=max_write_rows,
        read_rows=read_rows,
    )
    fixed = {
        name: size
        for name, size in fixed.items()
        if name not in per_block and name != "hot"
    }
    fixed["v2_counters"] = counter_bytes
    return per_block, fixed
