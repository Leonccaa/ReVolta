# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""value arbitration index-side signal: only a receive that every rank completed without
a
failure reports its NEW_LOAD hashes, and they are the same prefix hashes the
GPU index cache gives up on eviction."""

from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
    SPAN,
    publish_sources,
    setup_receive,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("local_blocks", [0, 2])
@pytest.mark.parametrize("failed", [False, True])
def test_completed_receive_reports_new_load_hashes(local_blocks, failed):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    scheduler, connector, request, _ = setup_receive(local_blocks=local_blocks)
    publish_sources(scheduler, request, range(local_blocks, 4))
    wrapper = object.__new__(OffloadingConnector)
    wrapper.connector_scheduler, wrapper.connector_worker = connector, None
    scheduler.connector = wrapper
    restored: list[list[bytes]] = []
    scheduler.value_arbiter = SimpleNamespace(
        on_index_restored=restored.append,
        begin_step=lambda step, seconds: 0,
        end_step=lambda step, **kwargs: None,
        add_rank_counters=lambda *args: None,
    )
    connector.value_restore_hook = scheduler._value_restore
    scheduler.current_step = 1
    scheduler.add_request(request)
    output = scheduler.schedule()
    job_id, job = next(iter(output.kv_connector_metadata.load_jobs.items()))
    txn = job.transfer_spec[0].txn
    table = scheduler.host_history_pool.host_pages
    receipt = KVConnectorOutput(
        finished_recving={request.request_id},
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            receive_results={job_id: (table.all_ranks, int(failed))},
            completed_jobs={job_id: 4},
            completed_steps={2: table.all_ranks},
        ),
    )
    scheduler.update_host_kv_output(receipt)
    scheduler._update_from_kv_xfer_finished(receipt)
    if failed:
        assert restored == []
        return
    assert len(restored) == 1 and len(restored[0]) == 4 - local_blocks
    assert request.num_computed_tokens == 4 * SPAN
    assert scheduler._try_promote_blocked_waiting_request(request)
    pool = scheduler.host_history_pool
    assert restored[0] == [table.get(e.handle).key for e in txn.new_loads]
    # Once the request is gone, evicting those cached IDs reports exactly the
    # keys a later restore of them reports: the index shadow can match.
    evicted: list[bytes] = []
    pool.value_observer = evicted.append
    scheduler.kv_cache_manager.free(request)
    hids = {e.hid for e in txn.new_loads}
    assert all(pool.blocks[hid].ref_cnt == 0 for hid in hids)
    pool.evict_blocks(hids)
    assert sorted(evicted) == sorted(restored[0])
