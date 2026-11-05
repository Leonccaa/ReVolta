# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C7 preserves semantic kinds without enabling unmodelled deletion modes."""

from dataclasses import replace

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_cpu_draft_offload import (
    _kv_config,
    _request,
    _vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("eviction", [False, True], ids=["stats-only", "eviction"])
@pytest.mark.parametrize("grouped", [False, True])
def test_nonalign_mamba_is_state_unknown(eviction, grouped):
    config = _vllm_config("all")
    config.kv_transfer_config.kv_connector_extra_config.update(
        {
            "cpu.residency_stats": True,
            "cpu.enable_dependency_eviction": eviction,
            "cpu.offload_counters": True,
        }
    )
    kv = _kv_config()
    kv.kv_cache_groups.pop()  # No drafter: equal 32-token pages in all mode.
    kv.kv_cache_groups[1].kv_cache_spec = replace(
        kv.kv_cache_groups[1].kv_cache_spec, mamba_cache_mode="all"
    )
    spec = MockOffloadingSpec(config, kv)
    pools = [CPUOffloadingManager(128) for _ in range(2 if grouped else 1)]
    spec.manager = (
        GroupedCPUOffloadingManager(dict(enumerate(pools))) if grouped else pools[0]
    )
    scheduler = OffloadingConnectorScheduler(spec)
    req = _request("r", 200)
    scheduler.on_new_request(req)
    status = scheduler._req_status["r"]
    status.update_offload_keys()
    key = status.group_states[1].offload_keys[0]
    out = scheduler._prepare_store([key], status)
    scheduler.manager.complete_store(out.keys_to_store, status.req_context)
    index = scheduler._dependency_index
    assert not index.supported
    assert not index.evict(set())
    assert key in index.records
    snapshot = scheduler.cpu_residency_snapshot()
    assert snapshot["state", "unknown"] == 1
    assert snapshot["target", "unknown"] == 0
    counters = scheduler._offload_counters
    assert counters.kind_by_group == {i: g.kind for i, g in index.groups.items()}
    counters.count_stored(out.keys_to_store)
    assert counters.stored_blocks == {"state": 1}
