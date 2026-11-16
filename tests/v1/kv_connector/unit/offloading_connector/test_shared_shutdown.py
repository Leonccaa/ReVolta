# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B. Both DMA directions must retire before shared backing is released.

Handlers, submissions, both worker layers and mmap storage are real. CUDA
streams/events, GPU tensor views and the actual copy operation use CPU doubles.
"""

import mmap
import uuid
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(
    "order", ["load_first", "store_first", "set_first", "set_last"]
)
@pytest.mark.parametrize("failure", ["copy", "end_event", "normal_event"])
def test_shared_shutdown_drains_all_before_cleanup(
    monkeypatch, grouped, order, failure
):
    regions = [
        SharedOffloadRegion(
            instance_id=str(uuid.uuid4()),
            total_size_bytes=4 * mmap.PAGESIZE,
            num_blocks=4,
            rank=0,
            num_workers=1,
            cpu_page_size=mmap.PAGESIZE,
        )
        for _ in range(2 if grouped else 1)
    ]
    cpu = [regions[i if grouped else 0].create_next_view(16) for i in range(2)]
    storage = [torch.empty((4, 16), dtype=torch.int8) for _ in range(2)]
    gpu = [
        SimpleNamespace(
            dtype=t.dtype,
            ndim=t.ndim,
            is_cuda=True,
            device=SimpleNamespace(type="cuda"),
            shape=t.shape,
            data_ptr=t.data_ptr,
            stride=t.stride,
        )
        for t in storage
    ]
    refs = [[CanonicalKVCacheRef(i, 16)] for i in range(2)]
    store = gpu_worker.SingleDirectionOffloadingHandler(
        gpu,
        cpu,
        1,
        refs,
        True,
        mmap_region=None if grouped else regions[0],
        group_mmap_regions=dict(enumerate(regions)) if grouped else None,
    )
    load = gpu_worker.SingleDirectionOffloadingHandler(gpu, cpu, 1, refs, False)
    assert store.src_tensors is load.dst_tensors is gpu
    assert store.dst_tensors is load.src_tensors is cpu
    worker = OffloadingConnectorWorker(
        SimpleNamespace(
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(qsa_cross_group_hit_check=True)
            )
        )
    )
    worker.worker.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, store)
    worker.worker.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, load)
    if order in ("load_first", "store_first"):
        handlers = [load, store] if order == "load_first" else [store, load]
        # Enumerate both permitted set orders deterministically as well.
        monkeypatch.setattr(worker.worker, "handlers", handlers)
        pending = handlers[1]
    else:
        assert isinstance(worker.worker.handlers, set)
        handlers = list(worker.worker.handlers)
        pending = handlers[0 if order == "set_first" else 1]
    active = [SimpleNamespace(copies=0, failure=None)]
    states: list[SimpleNamespace] = []
    cleanup_calls = []
    original_cleanup = SharedOffloadRegion.cleanup

    def cleanup(region):
        cleanup_calls.append(region)
        assert all(state.retired for state in states)
        assert all(not h._failed_transfers for h in handlers)
        assert not cpu and not gpu
        original_cleanup(region)

    def copy(*args, **kwargs):
        active[0].copies += 1
        if active[0].failure == "copy":
            raise RuntimeError("injected partial-copy failure")

    monkeypatch.setattr(SharedOffloadRegion, "cleanup", cleanup)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(gpu_worker.ops, "swap_blocks_batch", copy)

    def prepare(handler):
        state = SimpleNamespace(
            failure=failure if handler is pending else None,
            ready=handler is not pending,
            retired=False,
            copies=0,
        )

        def synchronize():
            # Even a ready sibling must still have backing when its fence
            # is checked; release may only follow the last successful drain.
            assert cpu and gpu
            if not state.ready:
                raise RuntimeError("injected drain failure")
            state.retired = True

        def record_end(_):
            if state.failure == "end_event":
                raise RuntimeError("injected end-event failure")

        handler._stream_pool.append(
            SimpleNamespace(wait_stream=lambda _: None, synchronize=synchronize)
        )
        start = SimpleNamespace(record=lambda _: None)
        end = SimpleNamespace(record=record_end, synchronize=synchronize)
        handler._event_pool.extend((end, start))
        states.append(state)
        return state

    try:
        for handler in handlers:
            active[0] = prepare(handler)
            job_id = 42 if handler is pending else 41
            cpu_spec = CPULoadStoreSpec([0, 0])
            gpu_spec = GPULoadStoreSpec([0, 0], [1, 1], [0, 0])
            spec = (gpu_spec, cpu_spec) if handler is store else (cpu_spec, gpu_spec)
            if handler is load:
                worker._load_jobs[job_id] = "load"
            worker._submit_transfer(job_id, spec)
        assert all(state.copies == 1 for state in states)
        metadata = worker._connector_worker_meta
        load_jobs = dict(worker._load_jobs)
        pending_jobs = set(worker._pending_failed_submissions)
        for _ in range(3):
            with pytest.raises(RuntimeError, match="injected drain failure"):
                worker.shutdown()
            assert cpu and gpu
            assert not cleanup_calls
            assert all(r.fd is not None and r.mmap_obj is not None for r in regions)
            assert worker._load_jobs == load_jobs
            assert worker._pending_failed_submissions == pending_jobs
            assert worker._connector_worker_meta is metadata
            if failure == "normal_event":
                assert pending._transfers[0].job_id == 42
                assert 42 in pending._transfer_events
            else:
                assert 42 in pending._failed_transfers
        for state in states:
            state.ready = True
        worker.shutdown()
        assert all(state.retired for state in states)
        assert not cpu and not gpu
        assert len(cleanup_calls) == len(regions)
        assert all(cleanup_calls.count(r) == 1 for r in regions)
        assert all(r.fd is None and r.mmap_obj is None for r in regions)
        assert not worker._load_jobs and not worker._pending_failed_submissions
        for handler in handlers:
            assert not handler._transfers and not handler._transfer_events
            assert not handler._failed_transfers
            assert not handler._stream_pool and not handler._event_pool
        worker.shutdown()
        assert len(cleanup_calls) == len(regions)
    finally:
        # Also release the real mmap storage when this test detects a defect.
        cpu.clear()
        gpu.clear()
        for region in regions:
            original_cleanup(region)
