# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only bounds shared by admission and host-history pass planning."""

from collections.abc import Sequence

from .budget import QSA_ROW_GROUPS, TAIL_GROUPS


def max_local_tokens(length: int, dcp: int = 1, interleave: int = 1) -> int:
    """Maximum rank-local length under interleaved context parallelism."""
    if length < 0 or dcp < 1 or interleave < 1:
        raise ValueError("Invalid context-parallel length or geometry")
    span = dcp * interleave
    return length // span * interleave + min(length % span, interleave)


def history_groups(length: int, dcp: int = 1, interleave: int = 1) -> int:
    return (max_local_tokens(length, dcp, interleave) + 3) // 4


def pass_working_set(rows_by_req: Sequence[int], groups: Sequence[int]) -> int:
    """Include every request's tail, even requests with no rows in this pass."""
    return sum(
        min(g, QSA_ROW_GROUPS * rows + TAIL_GROUPS)
        for rows, g in zip(rows_by_req, groups)
    )


def plan_passes(
    rows_by_req: Sequence[int],
    G: Sequence[int],
    avail: int,
    R_cap: int,
    *,
    row_groups: int = QSA_ROW_GROUPS,
    tail_groups: int = TAIL_GROUPS,
) -> list[tuple[int, int]]:
    """Greedily take the longest contiguous interval in request/query order.

    Rows of each request must be contiguous. All branches obey R_cap. Raise
    if even one row cannot fit; callers must establish the admission floor.
    This function never reads a device tensor and is identical on every rank.
    """
    if (
        len(rows_by_req) != len(G)
        or any(type(n) is not int or n < 0 for n in (*rows_by_req, *G))
        or type(avail) is not int
        or avail < 0
        or type(R_cap) is not int
        or R_cap < 1
        or row_groups < 1
        or tail_groups < 0
    ):
        raise ValueError("Invalid host-history pass geometry")
    tails = [min(g, tail_groups) for g in G]
    base = sum(tails)
    passes: list[tuple[int, int]] = []
    start = end = 0
    used = base
    counts = [0] * len(G)
    for req, rows in enumerate(rows_by_req):
        for _ in range(rows):
            old = min(G[req], row_groups * counts[req] + tail_groups)
            new = min(G[req], row_groups * (counts[req] + 1) + tail_groups)
            if end - start == R_cap or used + new - old > avail:
                if end == start:
                    raise ValueError("Host-history hot floor cannot hold one row")
                passes.append((start, end))
                start = end
                counts = [0] * len(G)
                used = base
                new = min(G[req], row_groups + tail_groups)
                old = tails[req]
                if used + new - old > avail:
                    raise ValueError("Host-history hot floor cannot hold one row")
            used += new - old
            counts[req] += 1
            end += 1
    if end > start:
        passes.append((start, end))
    return passes


def validate_graph_capacity(
    R_cap: int,
    capture_sizes: Sequence[int],
    max_requests: int,
    decode_rows: int,
    *,
    mixed_full: bool,
) -> None:
    """Check resolved capture sizes, including padded decode batches."""
    if mixed_full:
        raise ValueError("Host history requires mixed_mode != FULL")
    need = max(max_requests * decode_rows, max(capture_sizes, default=0), 1)
    if R_cap < need:
        raise ValueError(f"Host history R_cap={R_cap} must be >= decode rows {need}")
