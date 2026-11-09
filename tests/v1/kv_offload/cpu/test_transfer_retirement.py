# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU fault injection for real offload submission and retirement paths."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, OffloadingWorker


@pytest.mark.parametrize("failure", ["copy", "end_event", "wait_stream"])
@pytest.mark.parametrize("gpu_to_cpu", [False, True])
def test_failed_submission_retains_resources_until_retry(
    monkeypatch, failure, gpu_to_cpu
):
    state = SimpleNamespace(ready=False, attempts=0)

    def synchronize():
        state.attempts += 1
        if not state.ready:
            raise RuntimeError("injected drain failure")

    def wait_stream(_):
        if failure == "wait_stream":
            raise RuntimeError("injected fence failure")

    def record_end(_):
        if failure == "end_event":
            raise RuntimeError("injected event failure")

    def copy(*args, **kwargs):
        if failure == "copy":
            raise RuntimeError("injected copy failure")

    stream = SimpleNamespace(wait_stream=wait_stream, synchronize=synchronize)
    start = SimpleNamespace(record=lambda _: None)
    end = SimpleNamespace(record=record_end)
    handler = gpu_worker.SingleDirectionOffloadingHandler.__new__(
        gpu_worker.SingleDirectionOffloadingHandler
    )
    handler.gpu_to_cpu = gpu_to_cpu
    handler.src_tensors = [torch.empty((8, 32), dtype=torch.int8)]
    handler.dst_tensors = [torch.empty((8, 32), dtype=torch.int8)]
    handler.src_block_size_factor = handler.dst_block_size_factor = 1
    handler.kv_cache_groups_data_refs = [[CanonicalKVCacheRef(0, 32)]]
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
    cpu = CPULoadStoreSpec([0])
    gpu = GPULoadStoreSpec([0], [1], [0])
    with pytest.raises(RuntimeError, match="drain failure"):
        handler.transfer_async(42, (gpu, cpu) if gpu_to_cpu else (cpu, gpu))
    assert 42 in handler._failed_transfers
    assert not handler._stream_pool and not handler._event_pool
    with pytest.raises(RuntimeError, match="drain failure"):
        handler.wait({42})
    with pytest.raises(RuntimeError, match="drain failure"):
        handler.shutdown()
    assert handler.src_tensors and handler.dst_tensors
    state.ready = True
    handler.wait({42})
    assert not handler._failed_transfers
    assert handler._stream_pool == [stream]
    assert handler._event_pool == [end, start]
    handler.wait({42})
    handler.shutdown()
    assert not handler.src_tensors and not handler.dst_tensors


def test_worker_drains_every_direction_before_releasing_any_backing():
    calls = []
    ready = False

    class Handler(OffloadingHandler):
        def transfer_async(self, job_id, spec):
            return True

        def get_finished(self):
            return []

        def wait(self, job_ids):
            pass

        def drain(self):
            calls.append("drain")
            if not ready:
                raise RuntimeError("not retired")

        def shutdown(self):
            calls.append("release")

    worker = OffloadingWorker()
    worker.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, Handler())
    worker.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, Handler())
    with pytest.raises(RuntimeError, match="not retired"):
        worker.shutdown()
    assert calls == ["drain"]
    calls.clear()
    ready = True
    worker.shutdown()
    assert calls == ["drain", "drain", "release", "release"]


def test_success_receipt_survives_drain_until_delivery():
    """Another direction can fail its drain; this success remains reportable."""
    calls: list[str] = []
    stream = SimpleNamespace()
    start = SimpleNamespace(elapsed_time=lambda other: 1.0)
    end = SimpleNamespace(
        synchronize=lambda: calls.append("retired"), query=lambda: False
    )
    handler = gpu_worker.SingleDirectionOffloadingHandler.__new__(
        gpu_worker.SingleDirectionOffloadingHandler
    )
    handler.gpu_to_cpu = False
    handler.transfer_type = ("CPU", "GPU")
    handler._transfers = deque([gpu_worker.Transfer(7, stream, start, end, 32)])
    handler._transfer_events = {7: end}
    handler._failed_transfers = {}
    handler._stream_pool = []
    handler._event_pool = []
    handler.drain()
    handler.drain()
    assert calls == ["retired"]
    receipts = handler.get_finished()
    assert len(receipts) == 1
    assert receipts[0].job_id == 7 and receipts[0].success
    assert handler.get_finished() == []
    assert handler._stream_pool == [stream]


def test_connector_tracking_survives_shutdown_failure():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    retired = False

    def shutdown():
        if not retired:
            raise RuntimeError("still draining")

    connector = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    connector.worker = SimpleNamespace(shutdown=shutdown)
    connector._load_jobs = {1: "r"}
    connector._unsubmitted_store_jobs = [
        (2, (CPULoadStoreSpec([0]), GPULoadStoreSpec([0], [1], [0])))
    ]
    meta = connector._connector_worker_meta = OffloadingWorkerMetadata()
    connector._failed_submissions = {1}
    connector._pending_failed_submissions = {2}
    with pytest.raises(RuntimeError, match="still draining"):
        connector.shutdown()
    assert connector._load_jobs == {1: "r"}
    assert len(connector._unsubmitted_store_jobs) == 1
    assert connector._connector_worker_meta is meta
    retired = True
    connector.shutdown()
    assert not connector._load_jobs and not connector._unsubmitted_store_jobs
    assert connector._connector_worker_meta is not meta
