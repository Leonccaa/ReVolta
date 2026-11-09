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
HISTORY_METADATA = frozenset({"ready", "generation", "group_to_slot", "wanted"})


def history_metadata_bytes(local_tokens: int, num_layers: int) -> dict[str, int]:
    """Device bytes per history block ID, including the null block."""
    return {
        "ready": local_tokens,
        "generation": 4,
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
) -> dict[str, int]:
    """Exact bytes by tensor name; ``slot_bytes`` includes every layer.

    Also accepts zero slots / less than one pass for capacity inversion;
    HostHistoryShape.validate rejects those before allocation.
    """
    rows = max((slots - max_requests * tail_groups) // row_groups, 0)
    tiles = min((slots + 511) // 512, 256)
    return {
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
