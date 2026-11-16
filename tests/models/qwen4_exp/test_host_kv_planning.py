# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen v2 pass bounds, finite progress and DCP/interleave admission."""

from dataclasses import replace
from itertools import product

import pytest

from tests.v1.core.test_long_context_kv_capacity import geometry
from vllm.models.qwen4_exp.nvidia.host_kv.planning import (
    history_groups,
    max_local_tokens,
    pass_working_set,
    plan_passes,
    validate_graph_capacity,
)

pytestmark = pytest.mark.cpu_test


def _check(rows, groups, avail, cap):
    plan = plan_passes(rows, groups, avail, cap)
    request_rows = [req for req, n in enumerate(rows) for _ in range(n)]
    cursor = 0
    for start, end in plan:
        assert start == cursor and 1 <= end - start <= cap
        counts = [request_rows[start:end].count(req) for req in range(len(rows))]
        assert pass_working_set(counts, groups) <= avail
        # Greediness: the next row must fail either R_cap or the byte-free
        # group bound, including tails of requests absent from this pass.
        if end < len(request_rows) and end - start < cap:
            counts[request_rows[end]] += 1
            assert pass_working_set(counts, groups) > avail
        cursor = end
    assert cursor == sum(rows)
    assert len(plan) <= sum(rows)
    return plan


def test_two_long_prefills():
    plan = _check([1024, 1024], [65536, 65536], 8242, 256)
    assert len(plan) == 128
    assert all(end - start == 16 for start, end in plan)


def test_short_history_still_obeys_row_cap():
    assert _check([512], [64], 78, 256) == [(0, 256), (256, 512)]


@pytest.mark.parametrize("dcp,interleave", product((1, 2), (1, 4, 16)))
def test_128_token_requests_make_progress(dcp, interleave):
    groups = [history_groups(128, dcp, interleave)] * 8
    plan = _check([128] * 8, groups, sum(groups), 256)
    assert plan and all(end > start for start, end in plan)


def test_nonparticipating_requests_pay_tail():
    assert plan_passes([2, 0], [10000, 10000], 1043, 256) == [(0, 1), (1, 2)]
    with pytest.raises(ValueError, match="one row"):
        plan_passes([1, 0], [10000, 10000], 530, 256)


def test_exhaustive_progress_and_maximality():
    for rows in product((0, 1, 3, 17), repeat=2):
        for groups in product((0, 1, 16, 1024, 65536), repeat=2):
            # One-row feasibility is the only progress assumption.
            avail = max(
                [sum(min(g, 9) for g in groups)]
                + [
                    pass_working_set([int(i == q) for i in range(2)], groups)
                    for q in range(2)
                    if rows[q]
                ]
            )
            for cap in (1, 7, 256):
                _check(rows, groups, avail, cap)


@pytest.mark.parametrize(
    "rows,groups,avail,cap",
    [
        ([1], [], 100, 1),
        ([-1], [2], 100, 1),
        ([1], [-2], 100, 1),
        ([1], [2], -1, 1),
        ([1], [2], 100, 0),
    ],
)
def test_bad_plans_fail_closed(rows, groups, avail, cap):
    with pytest.raises(ValueError):
        plan_passes(rows, groups, avail, cap)


def test_interleave_matches_maximum_over_ranks_and_ledger():
    geo = geometry()
    for length, dcp, il in product(range(257), (1, 2, 4), (1, 2, 4, 16)):
        lengths = [
            length // (dcp * il) * il + min(max(length % (dcp * il) - rank * il, 0), il)
            for rank in range(dcp)
        ]
        assert max_local_tokens(length, dcp, il) == max(lengths) == lengths[0]
        groups = (max(lengths) + 3) // 4
        assert history_groups(length, dcp, il) == groups
        g = replace(geo, dcp_world_size=dcp, dcp_interleave=il)
        assert g.hot_reserve_groups(length) == min(groups, 8 * 513 + 9)
    # ceil(17 / 2) would give 9 tokens / 3 groups, but rank 0 holds 16 / 4.
    assert history_groups(17, 2, 16) == 4


@pytest.mark.parametrize(
    "capture,requests,decode,full",
    [
        ([264], 8, 8, False),
        ([64], 33, 8, False),
        ([64], 8, 8, True),
    ],
)
def test_graph_contract_rejects_unsafe_shapes(capture, requests, decode, full):
    with pytest.raises(ValueError):
        validate_graph_capacity(256, capture, requests, decode, mixed_full=full)


def test_graph_contract_accepts_boundary_and_empty_capture_set():
    validate_graph_capacity(256, [8, 64, 256], 32, 8, mixed_full=False)
    validate_graph_capacity(256, [], 8, 8, mixed_full=False)


def test_for_each_pass_consumes_planner_intervals_and_fixed_decode():
    from types import SimpleNamespace

    import torch

    from vllm.models.qwen4_exp.nvidia.host_kv.history import (
        HostHistoryShape,
        for_each_pass,
    )

    shape = HostHistoryShape(
        num_blocks=2,
        kernel_blocks_per_block=1,
        kernel_block_tokens=16,
        num_layers=1,
        kv_heads=1,
        head_dim=16,
        dtype=torch.float16,
        hot_slots=8242,
        max_requests=2,
        selection_width=1,
        max_write_rows=1,
    )
    seen = []
    layer = SimpleNamespace(
        store=SimpleNamespace(shape=shape), gather=lambda selection, *args: selection
    )
    selection = torch.arange(2048, dtype=torch.int32).view(-1, 1)
    args = (
        layer,
        selection,
        selection[:, 0],
        torch.empty(2, 1),
        torch.empty(2),
        lambda start, end, ids: seen.append((start, end, ids[:, 0].tolist())),
    )
    for_each_pass(*args)
    assert len(seen) == 128
    assert all(end - start == 16 for start, end, _ in seen)
    assert [i for _, _, values in seen for i in values] == list(range(2048))
    with pytest.raises(ValueError, match="R_cap"):
        for_each_pass(*args, single_pass=True)
    # Fixed decode can use the history-clamped admission bound instead of
    # the deliberately conservative prefill union bound.
    seen.clear()
    args = (
        layer,
        selection[:32],
        selection[:32, 0],
        torch.empty(2, 1),
        torch.empty(2),
        args[-1],
    )
    for_each_pass(*args, single_pass=True)
    assert [(start, end) for start, end, _ in seen] == [(0, 32)]


def test_short_static_pool_has_a_proven_positive_pass_bound():
    import torch

    from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryShape

    shape = HostHistoryShape(
        num_blocks=2,
        kernel_blocks_per_block=1,
        kernel_block_tokens=16,
        num_layers=1,
        kv_heads=1,
        head_dim=16,
        dtype=torch.float16,
        hot_slots=26,
        max_requests=1,
        selection_width=1,
        max_write_rows=1,
        dcp_world_size=2,
        max_context_tokens=128,
    )
    shape.validate()
    assert shape.rows_per_pass == 256
    assert replace(shape, hot_slots=15).rows_per_pass < 1
