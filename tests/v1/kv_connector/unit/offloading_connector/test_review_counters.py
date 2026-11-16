# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent counter review contracts, with corrected offer accounting."""

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_cpu_draft_offload import (
    DRAFT,
    _standard_run,
)
from tests.v1.kv_connector.unit.offloading_connector.test_offload_counters import (
    ALL,
    _admit,
    _lookup,
    _scheduler,
)
from vllm.v1.kv_offload.base import OffloadKey, get_offload_group_idx
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_documented_draft_reconciliation_with_admission_threshold(grouped, threshold):
    pools = [
        CPUOffloadingManager(128, store_threshold=threshold)
        for _ in range(3 if grouped else 1)
    ]
    manager = (
        GroupedCPUOffloadingManager(dict(enumerate(pools))) if grouped else pools[0]
    )
    scheduler = _scheduler("window", manager)
    accepted: list[OffloadKey] = []
    prepare_store = manager.prepare_store

    def record_offers(keys, ctx):
        result = prepare_store(keys, ctx)
        if result is not None:
            accepted.extend(k for k in keys if get_offload_group_idx(k) == DRAFT)
        return result

    manager.prepare_store = record_offers
    run = _standard_run(scheduler, complete=True)
    state = scheduler._req_status[run.req.request_id]
    deferred = sum(len(blocks) for blocks in state.draft_deferred.values())
    assert len(accepted) + scheduler.num_skipped_draft_blocks + deferred == 12
    run.finish()
    assert not scheduler._jobs
    assert not scheduler._req_status
    assert len(accepted) == 5
    assert len(accepted) + scheduler.num_skipped_draft_blocks == 12
    completed = scheduler._offload_counters.stored_blocks["draft"]
    if threshold > 1:
        assert not any(pool.resident_keys() for pool in pools)
        assert completed == 0
        assert completed + scheduler.num_skipped_draft_blocks == 7
    else:
        assert completed == len(accepted)


@pytest.mark.parametrize("counters", [None, False])
def test_task_requires_default_off(counters):
    assert _scheduler(counters=counters)._offload_counters is None


@pytest.mark.parametrize("local", range(193))
@pytest.mark.parametrize("eagle", [False, True])
def test_partial_offset_shrink_identity(local, eagle):
    s, req, hit, shrink = _lookup(
        {**ALL, 1: [1, 3, 5]}, num_computed=local, eagle=eagle
    )
    if 192 - local >= 32:
        assert sum(shrink.values()) == 192 - local - hit
    else:
        assert not shrink
    _admit(s, req, 0)  # GPU partial-prefix resolution may cancel external load.
    assert dict(s._offload_counters.lookup_shrink_tokens) == shrink
    _admit(s, req, 0)
    assert dict(s._offload_counters.lookup_shrink_tokens) == shrink
    assert not s._offload_counters.hit_draft_window_needed
