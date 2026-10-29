# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic synthetic inputs and a deferred GPU comparison matrix."""

from dataclasses import dataclass
from math import ceil

import numpy as np

from .prototype import Inputs, trunc_div


@dataclass(frozen=True)
class Case:
    name: str
    widths: tuple[int, ...]
    contexts: tuple[int, ...]
    columns: int
    page_size: int = 204
    strided: bool = False
    invalid_pages: bool = False
    interleaved: bool = False
    padding: bool = False
    ties: bool = False


def make_inputs(case: Case, seed: int = 1029) -> Inputs:
    if len(case.widths) != len(case.contexts) or not case.widths:
        raise ValueError("one context is required per request")
    if min(case.widths) < 1 or min(case.contexts) < 0:
        raise ValueError("invalid case shape")
    rng = np.random.default_rng(seed)
    rows, requests = sum(case.widths), len(case.widths)
    table_width = max(1, ceil(case.columns / case.page_size))
    needed = [
        max(1, min(table_width, ceil(length / (4 * case.page_size))))
        for length in case.contexts
    ]
    pages = sum(needed)
    if case.strided:
        q = rng.standard_normal((rows + 1, 8, 256)).astype(np.float16)[1:, ::2, ::2]
        cache = rng.standard_normal((pages, case.page_size * 2, 1, 256)).astype(
            np.float16
        )[:, ::2, :, ::2]
        table = np.full((requests * 2, table_width * 2), -1, np.int32)[::2, 1::2]
    else:
        q = rng.standard_normal((rows, 4, 128)).astype(np.float16)
        cache = rng.standard_normal((pages, case.page_size, 1, 128)).astype(np.float16)
        table = np.full((requests, table_width), -1, np.int32)
    page_ids = rng.permutation(pages).astype(np.int32)
    offset = 0
    for request, n in enumerate(needed):
        table[request, :n] = page_ids[offset : offset + n]
        offset += n
    if case.invalid_pages:
        table[:, ::7] = -1
        table[:, 3::11] = pages
    row_requests = np.repeat(np.arange(requests, dtype=np.int32), case.widths)
    positions = np.concatenate(
        [
            np.arange(length - width, length, dtype=np.int64)
            for width, length in zip(case.widths, case.contexts)
        ]
    )
    if case.interleaved:
        order = np.argsort(
            np.concatenate([np.arange(n) for n in case.widths]), kind="stable"
        )
        row_requests, positions, q = row_requests[order], positions[order], q[order]
    if case.padding:
        row_requests[::7] = -1
        row_requests[3::11] = requests
        positions[::5] = -6
    if case.ties:
        q.fill(0)
    return Inputs(
        q,
        cache,
        table,
        row_requests,
        positions,
        np.asarray(case.contexts, np.int32),
        case.columns,
    )


def quick_cases() -> list[Case]:
    return [
        Case("single8", (8,), (3073,), 1025),
        Case("c4_rows16", (4,) * 4, (4097,) * 4, 1100),
        Case("c4_rows20", (5,) * 4, (4097,) * 4, 1100),
        Case("c8_rows40", (5,) * 8, (3073,) * 8, 1025),
        Case("c7_rows35", (5,) * 7, (3073,) * 7, 1025),
        Case("c8_rows64_stress", (8,) * 8, (2053,) * 8, 517),
        Case("ragged", (8, 5, 2), (1600, 2400, 3200), 1025),
        Case("split_long_run", (17, 3), (2600, 3100), 1025),
        Case("interleaved", (5, 4, 3), (2500, 3000, 3500), 1025, interleaved=True),
        Case("strided", (4, 5), (2500, 3200), 1031, strided=True),
        Case("invalid_pages", (5, 3), (2800, 3100), 1025, invalid_pages=True),
        Case("padding_negative", (8, 5, 2), (2500, 3000, 3500), 1025, padding=True),
        Case("all_ties", (5, 5), (3100, 3100), 1025, ties=True),
        Case("width_truncated", (5, 4), (2500, 3100), 63),
        Case("zero_visible", (4, 4), (0, 0), 65),
        Case("zero_columns", (3,), (10,), 0),
        Case("page16_tail", (5, 5), (2053, 2057), 515, page_size=16),
    ]


def long_cases() -> list[Case]:
    # Full-width allocations are synthetic. 131456 is a rounded grid envelope,
    # not a claim that every deployment has this exact cache capacity.
    return [
        Case(f"{name}_{context}", (width,) * n, (context,) * n, 131456)
        for name, n, width in [
            ("c4_rows16", 4, 4),
            ("c4_rows20", 4, 5),
            ("c8_rows40", 8, 5),
        ]
        for context in (32768, 131072, 262144)
    ]


def geometry_cases() -> list[dict]:
    """No cache allocation is needed to prepare the full GPU shape matrix."""
    cases = [
        Case(f"{name}_{length}", (width,) * n, (length,) * n, 131456)
        for name, n, width in [
            ("single8", 1, 8),
            ("c4_rows16", 4, 4),
            ("c4_rows20", 4, 5),
            ("c8_rows40", 8, 5),
        ]
        for length in (32768, 131072, 262144)
    ]
    out = []
    for case in cases:
        requests = [r for r, n in enumerate(case.widths) for _ in range(n)]
        visible = [
            trunc_div(position + 1, 4)
            for n, length in zip(case.widths, case.contexts)
            for position in range(length - n, length)
        ]
        out.append(
            {
                "case": case.name,
                "rows_per_request": list(case.widths),
                "contexts": list(case.contexts),
                "columns": case.columns,
                "requests": requests,
                "visible": visible,
                "page_size": case.page_size,
                "runtime_paged_tiles_per_program": 1 if sum(case.widths) <= 32 else 8,
                "existing_single_request_shared_key_shape": len(case.widths) == 1,
                "gpu_validation": "not_run",
            }
        )
    return out
