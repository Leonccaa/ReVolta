# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""transfer regression. Failed DMA submission stays fenced until a drain succeeds.

Only CUDA stream operations and device storage are replaced. Transfer
submission, both worker layers, CPU pins and scheduler recovery run normally.
"""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector import (
    test_load_failure_recompute as recovery,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingHandler
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID, create_model_runner_output
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _handler(monkeypatch, failure):
    state = SimpleNamespace(ready=False, attempts=0, submitted=0)

    def synchronize():
        state.attempts += 1
        if not state.ready:
            raise RuntimeError("injected drain failure")

    stream = SimpleNamespace(wait_stream=lambda _: None, synchronize=synchronize)
    start = SimpleNamespace(record=lambda _: None)

    def record_end(_):
        raise RuntimeError("injected end-event failure")

    end = SimpleNamespace(record=record_end)

    def copy(*args, **kwargs):
        state.submitted += 1
        if failure == "copy":
            raise RuntimeError("injected partial-copy failure")

    handler = gpu_worker.SingleDirectionOffloadingHandler.__new__(
        gpu_worker.SingleDirectionOffloadingHandler
    )
    handler.gpu_to_cpu = False
    handler.src_tensors = [torch.empty((256, 32), dtype=torch.int8) for _ in range(2)]
    handler.dst_tensors = [torch.empty((100, 32), dtype=torch.int8) for _ in range(2)]
    handler.src_block_size_factor = handler.dst_block_size_factor = 1
    handler.kv_cache_groups_data_refs = [
        [CanonicalKVCacheRef(0, 32)],
        [CanonicalKVCacheRef(1, 32)],
    ]
    handler._stream_pool = [stream]
    handler._event_pool = [end, start]
    handler._transfers = deque()
    handler._transfer_events = {}
    handler._failed_transfers = {}
    handler._mmap_region = None
    handler._group_mmap_regions = {}
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(gpu_worker.ops, "swap_blocks_batch", copy)
    return handler, state


@pytest.mark.parametrize("failure", ["copy", "end_event"])
@pytest.mark.parametrize("is_load", [False, True])
def test_failed_drain_retains_fence_and_reports_once(monkeypatch, failure, is_load):
    handler, state = _handler(monkeypatch, failure)
    worker = OffloadingConnectorWorker(
        SimpleNamespace(
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(qsa_cross_group_hit_check=True)
            )
        )
    )
    cpu_spec = CPULoadStoreSpec([0])
    gpu_spec = GPULoadStoreSpec([0], [1, 0], [0, 0])
    if is_load:
        worker.worker.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, handler)
        spec = (cpu_spec, gpu_spec)
    else:
        handler.gpu_to_cpu = True
        worker.worker.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, handler)
        spec = (gpu_spec, cpu_spec)
    for wave in range(64):
        state.ready = False
        if is_load:
            worker._load_jobs[wave] = "request"
        worker._submit_transfer(wave, spec)
        for _ in range(256):
            assert worker.get_finished(set()) == (set(), set())
            assert worker.build_connector_worker_meta() is None
            assert wave in handler._failed_transfers
            assert not handler._stream_pool and not handler._event_pool
        state.ready = True
        assert worker.get_finished(set()) == (set(), {"request"} if is_load else set())
        meta = worker.build_connector_worker_meta()
        assert meta.failed_jobs == meta.completed_jobs == {wave: 1}
        assert not handler._failed_transfers
        assert len(handler._stream_pool) == 1 and len(handler._event_pool) == 2
        assert worker.get_finished(set()) == (set(), set())
        assert worker.build_connector_worker_meta() is None
    assert state.submitted == 64
    handler.shutdown()


def test_shutdown_retains_failed_stream_resources(monkeypatch):
    handler, state = _handler(monkeypatch, "copy")
    spec = (CPULoadStoreSpec([0]), GPULoadStoreSpec([0], [1, 0], [0, 0]))
    with pytest.raises(RuntimeError, match="drain failure"):
        handler.transfer_async(42, spec)
    with pytest.raises(RuntimeError, match="drain failure"):
        handler.shutdown()
    assert 42 in handler._failed_transfers
    assert handler.src_tensors and handler.dst_tensors
    state.ready = True
    handler.shutdown()
    assert not handler._failed_transfers
    assert not handler.src_tensors and not handler.dst_tensors


@pytest.mark.parametrize("failure", ["copy", "end_event"])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_drain_retry_releases_cpu_pins_and_recomputes(
    request_runner, monkeypatch, failure, cancel, async_scheduling
):
    runner = recovery._runner(request_runner, monkeypatch, "contract", async_scheduling)
    scheduler = runner.scheduler
    connector = runner.connector_scheduler
    cpu = CPUOffloadingManager(256)
    connector.manager = cpu
    runner.new_request(token_ids=[7] * 13)
    request = scheduler.requests[str(runner.req_id)]
    status = connector._req_status[request.request_id]
    status.update_offload_keys()
    keys = [key for group in status.group_states for key in group.offload_keys]
    cpu.prepare_store(keys, status.req_context)
    cpu.complete_store(keys, status.req_context)

    handler, state = _handler(monkeypatch, failure)
    worker = OffloadingConnectorWorker(runner.offloading_spec)
    worker.worker.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, handler)
    stores = MockOffloadingHandler()
    worker.worker.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, stores)

    def step():
        output = scheduler.schedule()
        metadata = output.kv_connector_metadata
        worker.handle_preemptions(metadata)
        worker.start_kv_transfers(metadata)
        worker.prepare_store_kv(metadata)
        stores.complete_jobs(set(stores.waiting_jobs))
        sending, receiving = worker.get_finished(output.finished_req_ids)
        meta = worker.build_connector_worker_meta()
        scheduler.update_from_output(
            output,
            create_model_runner_output(
                reqs=scheduler.running,
                token_id=EOS_TOKEN_ID,
                finished_sending=sending,
                finished_recving=receiving,
                kv_connector_worker_meta=meta,
            ),
        )
        return output, receiving, meta

    first, receiving, meta = step()
    assert not first.num_scheduled_tokens and not receiving and meta is None
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert any(cpu._policy.get(key).ref_cnt for key in keys)
    blocks = scheduler.kv_cache_manager.get_blocks(request.request_id)
    held = [block for group in blocks.blocks for block in group if not block.is_null]
    assert held and all(
        block.ref_cnt > 0 and block.block_hash is None for block in held
    )
    if cancel:
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    for _ in range(512):
        output, receiving, meta = step()
        assert not output.num_scheduled_tokens and not receiving and meta is None
        assert all(block.ref_cnt > 0 for block in held)
        assert any(cpu._policy.get(key).ref_cnt for key in keys)
    state.ready = True
    received = 0
    recomputed = 0
    for _ in range(128):
        output, receiving, meta = step()
        received += request.request_id in receiving
        if request.request_id in output.num_scheduled_tokens:
            assert output.scheduled_new_reqs[0].num_computed_tokens == 0
            recomputed += 1
        if not scheduler.requests and not connector._jobs:
            break
    assert received == 1
    assert recomputed == (0 if cancel else 1)
    assert not scheduler.requests and not connector._jobs
    assert not worker._pending_failed_submissions and not worker._load_jobs
    assert not handler._failed_transfers
    assert all(cpu._policy.get(key).ref_cnt == 0 for key in keys)
    for pool in scheduler.kv_cache_manager.coordinator.block_pools:
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert request.status == (
        RequestStatus.FINISHED_ABORTED if cancel else RequestStatus.FINISHED_STOPPED
    )
