# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP lookbehind through scheduler production, eviction and external receive."""

import pytest

from tests.v1.core.long_context_kv_utils import SPAN, host_step_receipts
from tests.v1.core.test_long_context_admission import build, make_requests
from tests.v1.core.test_long_context_host_main_kv import host_settings
from tests.v1.kv_connector.unit.offloading_connector.test_unified_lifecycle import (
    advance,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.kv_offload.host.receive import HostReceiveSpec, ReceiveSource
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("prompt_len", [8 * SPAN, 8 * SPAN + 1])
def test_mtp_external_hit_and_receive_cover_lookbehind_depth(prompt_len):
    scheduler = build(
        host_settings(host_pool_tokens=128 * SPAN),
        max_num_seqs=2,
        speculative_tokens=4,
        speculative_method="mtp",
        with_connector=True,
    )
    spec = scheduler.vllm_config.speculative_config
    assert spec.method == "mtp" and spec.use_eagle_kv_cache()
    connector = scheduler.connector.connector_scheduler
    unified = connector.unified
    assert unified.attached and connector._eagle_state_shift == SPAN
    (producer,) = make_requests(1, prompt_len, 1, tag="producer")
    scheduler.add_request(producer)
    # Use the existing connector fixture's real store/seal receipts, with a
    # sampled token on the final prefill so the producer finishes normally.
    while scheduler.has_unfinished_requests():
        out = scheduler.schedule()
        ids = list(out.num_scheduled_tokens)
        meta = host_step_receipts(scheduler, out) or OffloadingWorkerMetadata()
        mask = scheduler.host_history_pool.host_pages.all_ranks
        meta.job_results = {
            (jid, job.generation): (mask, 0)
            for jid, job in connector._jobs.items()
            if job.generation is not None
        }
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                sampled_token_ids=[
                    [1] if producer.num_computed_tokens >= prompt_len else []
                    for _ in ids
                ],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                kv_connector_output=KVConnectorOutput(kv_connector_worker_meta=meta),
            ),
        )
    assert producer.status == RequestStatus.FINISHED_LENGTH_CAPPED
    advance(scheduler)  # Retire the producer's last cleanup step.
    assert unified.units.units

    manager = scheduler.kv_cache_manager
    # Evict every cached history/state group, including the host group's
    # history IDs. The independent canonical host pages remain available.
    for group_id, single in enumerate(manager.coordinator.single_type_managers):
        single.block_pool.evict_blocks(
            {
                block.block_id
                for block in single.block_pool.blocks
                if block.block_hash is not None
                and get_group_id(block.block_hash) == group_id
            }
        )
    (consumer,) = make_requests(1, prompt_len, 1, tag="consumer")
    scheduler.add_request(consumer)
    _, local = manager.get_computed_blocks(consumer)
    assert local == 0
    expected = (prompt_len - 1 - SPAN) // SPAN * SPAN
    assert connector.get_num_new_matched_tokens(consumer, local) == (expected, True)

    out = scheduler.schedule()
    assert consumer.num_computed_tokens == expected
    jobs = out.kv_connector_metadata.load_jobs.values()
    receives = [job.transfer_spec[0] for job in jobs]
    assert len(receives) == 1 and isinstance(receives[0], HostReceiveSpec)
    txn = receives[0].txn
    assert txn.req_id == consumer.request_id
    assert [entry.logical_index for entry in txn.entries] == list(
        range(expected // SPAN)
    )
    assert all(entry.source == ReceiveSource.NEW_LOAD for entry in txn.entries)
    assert len(txn.new_loads) * SPAN == expected
