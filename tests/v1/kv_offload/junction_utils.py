# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real scheduler driver with synthetic worker receipts; no model execution."""

from tests.v1.core.long_context_kv_utils import SPAN, host_step_receipts
from tests.v1.core.test_long_context_admission import build, make_requests
from tests.v1.core.test_long_context_host_main_kv import host_settings
from tests.v1.kv_connector.unit.offloading_connector.test_unified_lifecycle import (
    advance,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_group_id, get_request_block_hasher
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request

__all__ = [
    "SPAN",
    "build",
    "make_requests",
    "host_settings",
    "advance",
    "new_request",
    "evict_gpu_prefix_only",
    "decode_step",
]


def new_request(name, ids, max_tokens=16):
    (template,) = make_requests(1, len(ids), max_tokens, tag=name)
    return Request(
        request_id=name,
        prompt_token_ids=list(ids),
        sampling_params=template.sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def evict_gpu_prefix_only(scheduler):
    """Controlled ordinary cache-cold condition; preserve all host MAIN pages."""
    host_gid = scheduler.host_history_pool.host_group_id
    for group_id, single in enumerate(
        scheduler.kv_cache_manager.coordinator.single_type_managers
    ):
        if group_id != host_gid:
            single.block_pool.evict_blocks(
                {
                    b.block_id
                    for b in single.block_pool.blocks
                    if b.block_hash is not None
                    and get_group_id(b.block_hash) == group_id
                }
            )


def decode_step(scheduler, emit=True):
    out = scheduler.schedule()
    ids = list(out.num_scheduled_tokens)
    connector = scheduler.connector.connector_scheduler
    meta = host_step_receipts(scheduler, out) or OffloadingWorkerMetadata()
    mask = scheduler.host_history_pool.host_pages.all_ranks
    meta.job_results = {
        (jid, job.generation): (mask, 0)
        for jid, job in connector._jobs.items()
        if job.generation is not None
    }
    load_jobs = (
        {} if out.kv_connector_metadata is None else out.kv_connector_metadata.load_jobs
    )
    meta.receive_results = {jid: (mask, 0) for jid in load_jobs}
    meta.completed_jobs = {jid: connector.config.num_workers for jid in load_jobs}
    sampled = [
        [17]
        if emit
        and scheduler.requests[rid].num_computed_tokens
        >= scheduler.requests[rid].num_tokens
        else []
        for rid in ids
    ]
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=KVConnectorOutput(kv_connector_worker_meta=meta),
        ),
    )
    return out
