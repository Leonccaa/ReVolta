# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)

pytestmark = pytest.mark.cpu_test


def test_aggregate_sums_counts():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 2, 7: 2}


def test_aggregate_disjoint_jobs():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={43: 1, 8: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 1, 7: 1, 43: 1, 8: 1}


def test_aggregate_multiple_workers():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1, 8: 1})
    meta3 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 8: 1})
    result = meta1.aggregate(meta2).aggregate(meta3)
    assert result.completed_jobs == {42: 3, 43: 2, 7: 2, 8: 2}


@pytest.mark.parametrize("bad", ["duplicate", "short", "negative"])
def test_invalid_value_counter_aggregate_cannot_complete_window(bad):
    from tests.v1.core.test_value_arbiter import make
    from vllm.v1.core.value_arbiter import NUM_HOT_COUNTERS

    arbiter, _ = make()
    arbiter.begin_step(1)
    values = (1,) * NUM_HOT_COUNTERS
    first = OffloadingWorkerMetadata(value_counters={1: (1, values)})
    other_values = values[:-1] if bad == "short" else values
    if bad == "negative":
        other_values = (-1,) + values[1:]
    other = OffloadingWorkerMetadata(value_counters={1: (2, other_values)})
    merged = (
        first.aggregate(first).aggregate(other)
        if bad == "duplicate"
        else first.aggregate(other)
    )
    # Invalidity survives further aggregation in either operand position.
    extra = OffloadingWorkerMetadata(value_counters={1: (4, values)})
    for result in (merged.aggregate(extra), extra.aggregate(merged)):
        ranks, counters = result.value_counters[1]
        arbiter.add_rank_counters(1, ranks, counters, 7)
    assert 1 not in arbiter.windows[0].reported
    assert not arbiter._partial
