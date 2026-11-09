# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real logical payload copies with CPU substitutes for CUDA streams/fences."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingHandler
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
)
from vllm.v1.attention.kv_cache_view import pack_page_payload
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.host import worker as receive_worker
from vllm.v1.kv_offload.host.page_table import PageHandle
from vllm.v1.kv_offload.host.receive import (
    HostReceiveSpec,
    ReceiveEntry,
    ReceiveSource,
    ReceiveTxn,
)
from vllm.v1.kv_offload.host.region import HostRegionLayout
from vllm.v1.kv_offload.host.worker import HostReceiveHandler

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def receive(monkeypatch):
    state = SimpleNamespace(retired=False, fail_drain=False, fail_end=False)

    def synchronize():
        if state.fail_drain:
            raise RuntimeError("index drain pending")
        state.retired = True

    stream = SimpleNamespace(wait_stream=lambda _: None, synchronize=synchronize)

    event_count = 0

    def event(**kwargs):
        nonlocal event_count
        event_count += 1
        event_index = event_count

        def record(_):
            if state.fail_end and event_index == 2:
                state.fail_drain = True
                raise RuntimeError("index end-event failed")

        return SimpleNamespace(
            record=record, query=lambda: state.retired, elapsed_time=lambda _: 1
        )

    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch, "Event", event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    import vllm.distributed

    monkeypatch.setattr(
        vllm.distributed, "get_world_group", lambda: SimpleNamespace(rank_in_group=1)
    )
    shape = HostHistoryShape(
        num_blocks=8,
        kernel_blocks_per_block=2,
        kernel_block_tokens=4,
        num_layers=1,
        kv_heads=1,
        head_dim=4,
        dtype=torch.uint8,
        hot_slots=16,
        max_requests=1,
        selection_width=4,
        max_write_rows=4,
        row_groups=1,
        tail_groups=0,
    )
    layout = HostRegionLayout(uuid4().hex, 10, 1, shape.page_bytes, (96,))
    owner = layout.open(None)
    region_path = owner.mmap_path
    store = HostHistoryStore(shape, ["main"], device="cpu", region_layout=layout)
    raw = torch.full((8 * 128 + 32,), 165, dtype=torch.uint8)
    index = raw[16:-16].view(torch.float16).as_strided((16, 3, 8), (32, 8, 1))
    ordinary = MockOffloadingHandler()
    handler = HostReceiveHandler(ordinary, store, [index])
    worker = OffloadingConnectorWorker(
        SimpleNamespace(
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(qsa_cross_group_hit_check=True)
            )
        )
    )
    worker._host_history, worker._host_group_id = store, 0
    worker.worker.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, handler)
    entries = (
        ReceiveEntry(3, ReceiveSource.REUSE_LOCAL, 1, 3, 11, PageHandle(2, 4)),
        ReceiveEntry(4, ReceiveSource.NEW_LOAD, 1, 1, 12, PageHandle(7, 6)),
        ReceiveEntry(5, ReceiveSource.NEW_LOAD, 1, 5, 13, PageHandle(4, 3)),
    )
    txn = ReceiveTxn(19, "receiver", 23, entries)
    store.bind_pages([(e.hid, e.handle.page_id) for e in entries])
    store.begin_blocks([e.hid for e in txn.new_loads])
    store.mark_ready([3])
    for i, entry in enumerate(entries):
        payload = store.index_payloads[0][entry.handle.page_id]
        payload.copy_((torch.arange(96) + i * 43).to(torch.uint8))
        store.host[entry.handle.page_id].fill_(37 + i)
    meta = OffloadingConnectorMetadata(
        {
            19: TransferJob(
                "receiver",
                (
                    HostReceiveSpec(CPULoadStoreSpec([8, 9]), txn),
                    GPULoadStoreSpec([4, 6], [0, 1, 1], [0, 5, 5]),
                ),
            )
        },
        {},
    )
    yield SimpleNamespace(
        state=state,
        store=store,
        raw=raw,
        index=index,
        ordinary=ordinary,
        handler=handler,
        worker=worker,
        txn=txn,
        metadata=meta,
    )
    state.fail_drain = False
    worker.shutdown()
    store.close()
    owner.cleanup()
    assert not Path(region_path).exists()


def test_mixed_restore_B_C_bitwise_and_padding_sentinels(receive):
    r = receive
    before = r.raw.clone()
    borrowed_ready = r.store.ready[24:32].clone()
    borrowed_generation = r.store.generation[3].clone()
    r.worker.start_kv_transfers(r.metadata)
    assert r.handler._copies[19].staging.numel() == 96
    assert r.worker.get_finished(set()) == (set(), set())
    r.ordinary.complete_jobs({19})
    assert r.worker.get_finished(set()) == (set(), set())
    r.state.retired = True
    assert r.worker.get_finished(set()) == (set(), {"receiver"})
    meta = r.worker.build_connector_worker_meta()
    assert meta.receive_results == {19: (2, 0)}
    assert meta.completed_jobs == {19: 1} and not meta.failed_jobs
    for entry in r.txn.new_loads:
        assert torch.equal(
            pack_page_payload(r.index, entry.hid, subdivisions=2),
            r.store.index_payloads[0][entry.handle.page_id],
        )
        assert torch.equal(
            r.store.host_page(entry.hid), r.store.host[entry.handle.page_id]
        )
        start = 16 + entry.hid * 128
        assert torch.equal(
            r.raw[start + 48 : start + 64], before[start + 48 : start + 64]
        )
        assert torch.equal(
            r.raw[start + 112 : start + 128], before[start + 112 : start + 128]
        )
    for hid in (0, 2, 3, 4, 6, 7):
        start = 16 + hid * 128
        assert torch.equal(r.raw[start : start + 128], before[start : start + 128])
    assert torch.equal(r.store.ready[24:32], borrowed_ready)
    assert torch.equal(r.store.generation[3], borrowed_generation)
    assert r.store.ready[8:16].all() and r.store.ready[40:48].all()
    assert not r.handler._copies
    assert r.worker.get_finished(set()) == (set(), set())


@pytest.mark.parametrize("failure", ["index-copy", "index-end", "state-submit"])
def test_partial_failure_drains_both_parts_before_receipt(
    receive, monkeypatch, failure
):
    r = receive
    if failure == "index-copy":
        scatter = receive_worker.scatter_page_payload

        def fail(*args, **kwargs):
            scatter(*args, **kwargs)
            r.state.fail_drain = True
            raise RuntimeError("partial logical scatter")

        monkeypatch.setattr(receive_worker, "scatter_page_payload", fail)
    elif failure == "index-end":
        r.state.fail_end = True
    else:

        def fail(*args, **kwargs):
            r.state.fail_drain = True
            raise RuntimeError("partial STATE submission")

        monkeypatch.setattr(r.ordinary, "transfer_async", fail)
    borrowed = r.raw[16 + 3 * 128 : 16 + 4 * 128].clone()
    r.worker.start_kv_transfers(r.metadata)
    for _ in range(3):
        assert r.worker.get_finished(set()) == (set(), set())
        assert r.worker.build_connector_worker_meta() is None
        assert 19 in r.handler._copies
        assert not r.store.ready[8:16].any()
    r.state.fail_drain = False
    r.ordinary.complete_jobs({19})
    assert r.worker.get_finished(set()) == (set(), {"receiver"})
    meta = r.worker.build_connector_worker_meta()
    assert meta.failed_jobs == {19: 1}
    assert meta.receive_results == {19: (2, 2)}
    assert not r.store.ready[8:16].any()
    assert torch.equal(r.raw[16 + 3 * 128 : 16 + 4 * 128], borrowed)
    assert not r.handler._copies
    assert r.worker.get_finished(set()) == (set(), set())


def test_ready_failure_voids_receive_after_dma_retirement(receive, monkeypatch):
    r = receive
    r.worker.start_kv_transfers(r.metadata)
    r.ordinary.complete_jobs({19})
    r.state.retired = True
    monkeypatch.setattr(r.store, "mark_ready", MockFailure())
    assert r.worker.get_finished(set()) == (set(), {"receiver"})
    meta = r.worker.build_connector_worker_meta()
    assert meta.failed_jobs == {19: 1} and meta.receive_results == {19: (2, 2)}


class MockFailure:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("ready publication failed")


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize("control_drain", [False, True])
def test_index_and_state_window_payloads_restore_bitwise(
    receive, monkeypatch, factor, control_drain
):
    import ctypes

    from vllm.v1.kv_offload.base import CanonicalKVCacheRef
    from vllm.v1.kv_offload.cpu import gpu_worker

    r = receive
    cpu = [
        torch.arange(10 * 32 * factor, dtype=torch.int64)
        .to(torch.int8)
        .view(10, 32 * factor),
        torch.arange(10 * 48 * factor, dtype=torch.int64)
        .to(torch.int8)
        .view(10, 48 * factor),
    ]
    gpu = [
        torch.full((10, 32), 31, dtype=torch.int8),
        torch.full((10, 48), 32, dtype=torch.int8),
    ]
    handler = gpu_worker.SingleDirectionOffloadingHandler(
        gpu_tensors=gpu,
        cpu_tensors=cpu,
        block_size_factor=factor,
        kv_cache_groups_data_refs=[
            [],
            [CanonicalKVCacheRef(0, 32)],
            [CanonicalKVCacheRef(1, 48)],
        ],
        gpu_to_cpu=False,
    )
    handler._host_tensor_indices = set()

    def copy_batch(sources, targets, sizes, **kwargs):
        for source, target, size in zip(
            sources.tolist(), targets.tolist(), sizes.tolist()
        ):
            ctypes.memmove(target, source, size)

    monkeypatch.setattr(gpu_worker.ops, "swap_blocks_batch", copy_batch)
    monkeypatch.setattr(
        torch,
        "Event",
        lambda **kwargs: SimpleNamespace(
            record=lambda _: None,
            query=lambda: r.state.retired,
            synchronize=lambda: setattr(r.state, "retired", True),
            elapsed_time=lambda _: 1,
        ),
    )
    r.handler.ordinary = handler
    r.worker.start_kv_transfers(r.metadata)
    assert r.worker.get_finished(set()) == (set(), set())
    if control_drain:
        sending, receiving, meta = r.worker.drain()
        assert (sending, receiving) == (set(), {"receiver"})
    else:
        r.state.retired = True
        assert r.worker.get_finished(set()) == (set(), {"receiver"})
        meta = r.worker.build_connector_worker_meta()
    assert torch.equal(gpu[0][4], cpu[0][8, (factor - 1) * 32 : factor * 32])
    assert torch.equal(gpu[1][6], cpu[1][9, (factor - 1) * 48 : factor * 48])
    assert (gpu[0][:4] == 31).all() and (gpu[0][5:] == 31).all()
    assert (gpu[1][:6] == 32).all() and (gpu[1][7:] == 32).all()
    for entry in r.txn.new_loads:
        assert torch.equal(
            pack_page_payload(r.index, entry.hid, subdivisions=2),
            r.store.index_payloads[0][entry.handle.page_id],
        )
    assert not meta.failed_jobs
    assert r.worker.get_finished(set()) == (set(), set())


def test_reuse_only_receive_allocates_no_index_staging(receive, monkeypatch):
    from unittest.mock import Mock

    r = receive
    r.txn.entries = (r.txn.entries[0],)
    before = r.raw.clone()
    monkeypatch.setattr(
        receive_worker,
        "scatter_page_payload",
        Mock(side_effect=AssertionError("borrowed index H2D")),
    )
    r.worker.start_kv_transfers(r.metadata)
    assert r.handler._copies[19].staging is None
    r.ordinary.complete_jobs({19})
    r.state.retired = True
    assert r.worker.get_finished(set()) == (set(), {"receiver"})
    assert torch.equal(r.raw, before)
    assert not receive_worker.scatter_page_payload.called
    assert not r.worker.build_connector_worker_meta().failed_jobs


def test_receive_drain_retry_keeps_completed_receipts(receive):
    r = receive
    r.worker.start_kv_transfers(r.metadata)
    r.handler.drain()
    r.handler.drain()
    assert len(r.handler._drained_results) == 1
    assert r.worker.get_finished(set()) == (set(), {"receiver"})
    meta = r.worker.build_connector_worker_meta()
    assert meta.receive_results == {19: (2, 0)}
    assert r.worker.get_finished(set()) == (set(), set())
