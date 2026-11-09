# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-page SSD prefetch on the source-aware connector lookup (host SSD persistence x
receive transactions).

A fresh process writes G0 host pages to SSD. A new scheduler with an empty
host page table then looks the same prefix up: the lookup prefetches the
pages, waits while reads are in flight, and uses only verified READY pages
as NEW_LOAD sources of a ReceiveTxn. Timeouts, failed verification,
cancellation and exhausted quotas end the wait and fall back to a shorter
hit or a recompute.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.v1.core.test_long_context_host_main_kv import SPAN
from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
    lookup_allocate,
    setup_receive,
)
from tests.v1.kv_offload.test_host_page_ssd import LAYOUT, ROW, options, payload
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host import ssd as ssd_module
from vllm.v1.kv_offload.host.page_table import HostPageTable, PageState
from vllm.v1.kv_offload.host.persist import HostPageSsd
from vllm.v1.kv_offload.host.receive import ReceiveTxn
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

# Runs in a fresh interpreter: publish one page per key, final flush, exit.
_WRITER = r"""
import sys
from pathlib import Path
from tests.v1.kv_offload import test_host_page_ssd as t
from vllm.v1.kv_offload.host.page_table import Result

root = Path(sys.argv[1])
keys = [bytes.fromhex(h) for h in sys.argv[2:]]
env = t.Env(root, pages=len(keys) + 2, write_delay_s=300)
for i, key in enumerate(keys):
    handle = env.table.reserve()
    start = handle.page_id * t.ROW
    env.backing[start : start + t.ROW] = t.payload(i)
    page = env.table.get(handle)
    page.compute_result = page.seal_result = Result.OK
    env.table.publish(handle, key)
report = env.ssd.final_flush(budget_s=10.0, io_grace_s=5.0)
assert (report.candidates, report.completed) == (len(keys), len(keys)), report
env.ssd.close(io_grace_s=5)
print("WROTE", report.completed)
"""


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def write_in_fresh_process(root: Path, keys: list[bytes]) -> None:
    options(root)  # checkpoint fixtures and namespace seed, before the child
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[5]), env.get("PYTHONPATH", "")]
    )
    done = subprocess.run(
        [sys.executable, "-c", _WRITER, str(root), *(k.hex() for k in keys)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"WROTE {len(keys)}" in done.stdout


def g0_keys(scheduler, request, indices) -> list[bytes]:
    hits = scheduler.kv_cache_manager.host_kv_hits
    chain_hash = chain_hash_from_request(request.block_hashes, 16)
    return [
        make_offload_key(chain_hash(hits.rule, i), hits.rule.group_id) for i in indices
    ]


def attach_ssd(scheduler, connector, root, clock=None, **overrides):
    """The reader: a new controller over the scheduler's empty page table."""
    table = scheduler.host_history_pool.host_pages
    spec, opts = options(root, **overrides)
    backing = bytearray(len(table.pages) * ROW)
    view = memoryview(backing).cast("B", shape=[len(table.pages), ROW])
    ssd = HostPageSsd(table, view, spec, LAYOUT, opts, clock=clock or time.monotonic)
    assert ssd.enabled
    connector.host_page_ssd = scheduler.host_kv_ssd = ssd
    return ssd, backing


def settle(ssd, cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        ssd.step()
        if cond():
            return True
        time.sleep(0.005)
    return cond()


@pytest.fixture
def gated_reads(monkeypatch):
    release = threading.Event()
    read = ssd_module.load_verified_block

    def slow(*args, **kwargs):
        release.wait(30)
        return read(*args, **kwargs)

    monkeypatch.setattr(ssd_module, "load_verified_block", slow)
    yield release
    release.set()


def cold_reader(tmp_path, extra_keys=(), clock=None, **overrides):
    root = tmp_path / "ssd"
    scheduler, connector, request, _ = setup_receive(local_blocks=0)
    table = scheduler.host_history_pool.host_pages
    keys = g0_keys(scheduler, request, range(4))
    write_in_fresh_process(root, keys + list(extra_keys))
    assert all(table.lookup(k) is None for k in keys)  # nothing in RAM
    ssd, backing = attach_ssd(scheduler, connector, root, clock, **overrides)
    return scheduler, connector, request, keys, ssd, backing


def complete_receive(scheduler, connector, request, job_id):
    table = scheduler.host_history_pool.host_pages
    table.complete_step(2, table.all_ranks)
    output = KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(
            completed_jobs={job_id: 1},
            receive_results={job_id: (table.all_ranks, 0)},
        )
    )
    connector.update_connector_output(output)
    return output


def test_cold_restart_hits_host_pages_read_back_from_ssd(tmp_path):
    scheduler, connector, request, keys, ssd, backing = cold_reader(tmp_path)
    manager, table = scheduler.kv_cache_manager, scheduler.host_history_pool.host_pages
    rid = request.request_id
    try:
        _, tokens = manager.get_computed_blocks(request)
        assert tokens == 0
        # Reads in flight: the lookup defers instead of reporting a hit.
        assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        assert ssd.waiting(rid) and ssd.stats()["prefetch_issued"] == 4
        assert all(table.get(table.lookup(k)).state == PageState.LOADING for k in keys)
        assert settle(ssd, lambda: not ssd.waiting(rid))
        # Reads finish in any order; index the held pages by chain position.
        held = [table.lookup(k) for k in keys]
        assert len(ssd.holds(rid)) == 4 and set(ssd.holds(rid)) == set(held)
        assert all(table.is_ready(h) and table.is_persisted(table.get(h)) for h in held)
        for i, handle in enumerate(held):
            row = handle.page_id * ROW
            assert bytes(backing[row : row + ROW]) == payload(i)
        job_id, txn = lookup_allocate(scheduler, connector, request)
        # Verified pages are the NEW_LOAD sources; the round's holds are gone
        # and the lookup leases now protect them.
        assert [e.handle for e in txn.new_loads] == held
        assert [e.logical_index for e in txn.entries] == [0, 1, 2, 3]
        assert rid not in ssd._rounds
        assert all(len(table.get(h).leases) == 1 for h in held)
        # value arbitration value arbitration still sees the SSD-sourced NEW_LOAD
        # receive.
        restored: list[ReceiveTxn] = []
        connector.value_restore_hook = restored.append
        output = complete_receive(scheduler, connector, request, job_id)
        assert output.finished_recving == {rid}
        assert not connector.take_failed_loads()
        assert restored == [txn] and len(restored[0].new_loads) == 4
        manager.cache_blocks(request, 4 * SPAN)
        group = manager.coordinator.single_type_managers[
            scheduler.host_history_pool.host_group_id
        ]
        blocks = group.req_to_blocks[rid]
        assert all(b.block_hash is not None for b in blocks[:4])
        assert ssd.stats()["prefetch_verified"] == 4
    finally:
        ssd.close(io_grace_s=5)


def test_prefetch_timeout_falls_back_to_recompute(tmp_path, gated_reads):
    clock = Clock()
    scheduler, connector, request, keys, ssd, _ = cold_reader(
        tmp_path, clock=clock, prefetch_timeout_s=5
    )
    table = scheduler.host_history_pool.host_pages
    rid = request.request_id
    try:
        assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        clock.now += 4.9
        assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        clock.now += 0.1
        # T_pf passed: waiting ends, no second round, the lookup is a miss.
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        assert ssd.stats()["prefetch_timeouts"] == 1 and rid not in ssd._rounds
        assert ssd.stats()["prefetch_issued"] == 4
        # The reads still own their rows until they physically finish.
        loading = [table.get(table.lookup(k)) for k in keys]
        assert all(p.state == PageState.LOADING and p.io for p in loading)
        gated_reads.set()
        assert settle(ssd, lambda: not ssd._jobs)
        for k in keys:
            page = table.get(table.lookup(k))
            assert page.state == PageState.READY and not page.leases
            assert table.reclaimable(page)
    finally:
        gated_reads.set()
        ssd.close(io_grace_s=5)


def test_failed_verification_is_never_used(tmp_path):
    scheduler, connector, request, keys, ssd, _ = cold_reader(tmp_path)
    table = scheduler.host_history_pool.host_pages
    rid = request.request_id
    path = Path(ssd._tier.file_mapper.get_file_name(ssd._tier_key(keys[0])))
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF  # same length, bad CRC
    path.write_bytes(bytes(data))
    try:
        assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        assert settle(ssd, lambda: not ssd.waiting(rid))
        stats = ssd.stats()
        assert stats["prefetch_failed"] == 1 and stats["prefetch_verified"] == 3
        assert table.lookup(keys[0]) is None
        # Block 0 is missing, so nothing is a prefix hit: recompute.
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        assert rid not in ssd._rounds
        for k in keys[1:]:
            page = table.get(table.lookup(k))
            assert table.is_ready(page.handle) and not page.leases
    finally:
        ssd.close(io_grace_s=5)


def test_request_cancelled_during_prefetch_drops_its_wait(tmp_path, gated_reads):
    scheduler, connector, request, keys, ssd, _ = cold_reader(tmp_path)
    table = scheduler.host_history_pool.host_pages
    rid = request.request_id
    try:
        assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        request.status = RequestStatus.FINISHED_ABORTED
        connector.request_finished(request)
        assert rid not in ssd._rounds and not ssd.waiting(rid)
        handles = [table.lookup(k) for k in keys]
        # Withdrawn waiting only: each read keeps its row until it finishes.
        assert all(
            table.get(h).io and not table.reclaimable(table.get(h)) for h in handles
        )
        gated_reads.set()
        assert settle(ssd, lambda: not ssd._jobs)
        for handle in handles:
            page = table.get(handle)
            assert page.state == PageState.READY and not page.leases and not page.io
            assert table.reclaimable(page)
    finally:
        gated_reads.set()
        ssd.close(io_grace_s=5)


def test_global_prefetch_quota_exhausted_is_a_miss(tmp_path, gated_reads):
    extra = [make_offload_key(sha256(("blocker", i)), 0) for i in range(2)]
    scheduler, connector, request, keys, ssd, _ = cold_reader(
        tmp_path, extra_keys=extra, prefetch_max_inflight=2
    )
    rid = request.request_id
    try:
        # Another request holds every global read slot.
        assert ssd.prefetch("blocker", extra) == extra
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        assert ssd.stats()["prefetch_global_quota"] == 1
        assert rid not in ssd._rounds
        # One round per lookup episode: no retry until admission or finish.
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        assert ssd.stats()["prefetch_global_quota"] == 1
    finally:
        gated_reads.set()
        ssd.close(io_grace_s=5)


def test_per_request_quota_limits_the_hit_to_the_prefetched_pages(tmp_path):
    scheduler, connector, request, keys, ssd, _ = cold_reader(
        tmp_path, prefetch_max_per_request=2
    )
    table = scheduler.host_history_pool.host_pages
    rid = request.request_id
    try:
        assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        assert ssd.stats()["prefetch_request_quota"] == 1
        assert settle(ssd, lambda: not ssd.waiting(rid))
        assert [table.lookup(k) is not None for k in keys] == [True, True, False, False]
        # G0 now reaches 2 blocks, but the state checkpoint is at block 4:
        # no common boundary, so the request recomputes.
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        assert rid not in ssd._rounds
        assert all(not table.get(table.lookup(k)).leases for k in keys[:2])
    finally:
        ssd.close(io_grace_s=5)


def test_full_length_key_written_by_another_process_is_read_back(tmp_path):
    """98 x 32-byte hashes + group: the file name stays short and the page is
    read back byte for byte by a new process with the same ABI and root."""
    root = tmp_path / "ssd"
    long_key = make_offload_key(b"".join(sha256(i) for i in range(98)), 0)
    assert len(long_key) == 98 * 32 + 4
    write_in_fresh_process(root, [long_key])
    names = [p.name for p in root.rglob("*.bin")]
    assert len(names) == 1 and len(names[0]) < 255
    table = HostPageTable(4, num_workers=1)
    spec, opts = options(root)
    backing = bytearray(4 * ROW)
    view = memoryview(backing).cast("B", shape=[4, ROW])
    ssd = HostPageSsd(table, view, spec, LAYOUT, opts)
    try:
        assert ssd.prefetch("r", [long_key]) == [long_key]
        assert settle(ssd, lambda: not ssd.waiting("r"))
        (handle,) = ssd.holds("r")
        assert table.lookup(long_key) == handle and table.is_ready(handle)
        row = handle.page_id * ROW
        assert bytes(backing[row : row + ROW]) == payload(0)
        ssd.cancel("r")
        assert not table.get(handle).leases
    finally:
        ssd.close(io_grace_s=5)
