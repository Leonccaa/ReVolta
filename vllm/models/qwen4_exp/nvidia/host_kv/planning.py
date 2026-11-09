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


UNION_ROW_GROUP = 8
"""Mixed-step windows hold whole groups of eight query rows: the grouped
page4 planner's unit, so a window never splits one of its plans."""


def union_bound(
    rows_by_req: Sequence[int],
    G: Sequence[int],
    *,
    row_groups: int = QSA_ROW_GROUPS,
    tail_groups: int = TAIL_GROUPS,
) -> int:
    """Upper bound on one call's union, every request's tail included once.

    Unlike ``plan_passes`` this has no R_cap: the union path's row count does
    not use the indices scratch."""
    if len(rows_by_req) != len(G) or any(n < 0 for n in (*rows_by_req, *G)):
        raise ValueError("Invalid union-bound geometry")
    return sum(
        min(g, row_groups * rows + tail_groups) for rows, g in zip(rows_by_req, G)
    )


def union_fast_path(
    rows_by_req: Sequence[int],
    G: Sequence[int],
    U: int,
    *,
    row_groups: int = QSA_ROW_GROUPS,
    tail_groups: int = TAIL_GROUPS,
) -> bool:
    """L0: the whole mixed step fits one window by a CPU bound (no sync).

    ``G`` must use the maximum rank-local length (``history_groups``) and
    ``U`` the step's common hot capacity, so every rank decides alike."""
    return (
        union_bound(rows_by_req, G, row_groups=row_groups, tail_groups=tail_groups) <= U
    )


def plan_union_windows_upper(
    rows_by_req: Sequence[int],
    G: Sequence[int],
    U: int,
    *,
    row_groups: int = QSA_ROW_GROUPS,
    tail_groups: int = TAIL_GROUPS,
) -> list[tuple[int, int]] | None:
    """Greedy windows of whole eight-row groups by the CPU bound.

    Rows of a request are contiguous, in request order. Windows are planned on
    group boundaries directly (rounding row-wise ends would either overflow or
    stall). Returns None if one eight-row group alone exceeds the bound, which
    the executor resolves by its per-row fallback (L3)."""
    owner = [req for req, rows in enumerate(rows_by_req) for _ in range(rows)]
    total = len(owner)
    windows: list[tuple[int, int]] = []

    def cost(start: int, end: int) -> int:
        counts = [0] * len(G)
        for req in owner[start:end]:
            counts[req] += 1
        return union_bound(counts, G, row_groups=row_groups, tail_groups=tail_groups)

    start = 0
    while start < total:
        end = start
        while end < total:
            nxt = min(end + UNION_ROW_GROUP, total)
            if cost(start, nxt) > U:
                break
            end = nxt
        if end == start:
            return None
        windows.append((start, end))
        start = end
    return windows


def union_window_groups(
    tail: int, first_entries: Sequence[int], U: int, max_groups: int | None = None
) -> int:
    """Eight-row groups the exact greedy window takes from its start.

    ``first_entries[g]`` counts gids whose first reference at or after the
    window start lies in its g-th row group, tails excluded; ``tail`` counts
    the tail gids once. The union of the first k groups is therefore
    ``tail + sum(first_entries[:k])``. Returns 0 when one group does not fit.
    """
    used, taken = tail, 0
    limit = len(first_entries) if max_groups is None else max_groups
    while taken < min(limit, len(first_entries)):
        if used + first_entries[taken] > U:
            break
        used += first_entries[taken]
        taken += 1
    return taken


def plan_step_passes(
    batch_req_ids: Sequence[str],
    num_rows: int,
    requests: dict[str, tuple[int, int, int]],
    hot_slots: int,
    R_cap: int,
    *,
    row_groups: int = QSA_ROW_GROUPS,
    tail_groups: int = TAIL_GROUPS,
) -> list[tuple[int, int]]:
    """A mixed step's pass intervals from scheduler metadata (shared hot region).

    ``batch_req_ids`` is the worker's row order; every rank has the same
    order, U_step and per-request (rows, G_q), so every rank computes the same
    intervals with no device readback. Rows past the scheduled tokens
    (padding) belong to no request (G = 0) and only consume R_cap.
    """
    rows = [requests[req_id][0] for req_id in batch_req_ids]
    groups = [requests[req_id][1] for req_id in batch_req_ids]
    pad = num_rows - sum(rows)
    if pad < 0:
        raise ValueError("Host-history batch has fewer rows than scheduled")
    if pad:
        rows.append(pad)
        groups.append(0)
    return plan_passes(
        rows,
        groups,
        hot_slots,
        R_cap,
        row_groups=row_groups,
        tail_groups=tail_groups,
    )


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
