# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host region name lives only until every process has mapped it.

Real processes follow the engine's startup order: workers map their rank
first, then the engine core maps the whole region and removes the name. Data
stays shared after the unlink, and SIGKILL of every process leaves nothing in
/dev/shm because the kernel frees the backing with the last mapping.
"""

import multiprocessing
import os
import signal
from pathlib import Path
from uuid import uuid4

import pytest

from vllm.v1.kv_offload.host.region import HostRegionLayout

pytestmark = pytest.mark.cpu_test
WORKERS = 2
PAGES = 8
MAIN = 1 << 16


def _row(region, page: int, rank: int) -> int:
    return page * region._row_stride + rank * region.worker_bytes


def _worker(layout, rank, ready, release, report):
    region = layout.open(rank)
    mapping = region.mmap_obj
    mapping[_row(region, 1, rank)] = rank + 1
    ready.send(True)
    release.recv()
    # The core wrote into this rank's row after the name was removed.
    report.send(mapping[_row(region, 2, rank)])
    signal.pause()


def _core(layout, ready):
    region = layout.open(None)
    region.unlink()  # As EngineCore does once the last mapping exists.
    mapping = region.mmap_obj
    seen = [mapping[_row(region, 1, rank)] for rank in range(WORKERS)]
    for rank in range(WORKERS):
        mapping[_row(region, 2, rank)] = 100 + rank
    ready.send((seen, Path(region.mmap_path).exists()))
    signal.pause()


def _mapped_by(pid: int | None, path: str) -> bool:
    if pid is None:
        return False
    try:
        with open(f"/proc/{pid}/maps") as maps:
            return any(path in line for line in maps)
    except OSError:
        return False


def test_sigkill_of_every_process_leaves_no_named_region():
    layout = HostRegionLayout(uuid4().hex, PAGES, WORKERS, MAIN, ())
    path = f"/dev/shm/vllm_offload_host_v2_{layout.instance_id}.mmap"
    context = multiprocessing.get_context("fork")
    procs = []
    try:
        workers = []
        for rank in range(WORKERS):
            ready_r, ready_w = context.Pipe(duplex=False)
            release_r, release_w = context.Pipe(duplex=False)
            report_r, report_w = context.Pipe(duplex=False)
            proc = context.Process(
                target=_worker, args=(layout, rank, ready_w, release_r, report_w)
            )
            proc.start()
            procs.append(proc)
            workers.append((ready_r, release_w, report_r))
        for ready_r, _, _ in workers:
            assert ready_r.poll(30) and ready_r.recv()
        assert Path(path).exists()  # Named only while startup attaches.
        core_r, core_w = context.Pipe(duplex=False)
        core = context.Process(target=_core, args=(layout, core_w))
        core.start()
        procs.append(core)
        assert core_r.poll(30)
        seen, exists_in_core = core_r.recv()
        assert seen == [rank + 1 for rank in range(WORKERS)]
        assert not exists_in_core and not Path(path).exists()
        for rank, (_, release_w, report_r) in enumerate(workers):
            release_w.send(True)
            assert report_r.poll(30) and report_r.recv() == 100 + rank
        assert all(_mapped_by(p.pid, path) for p in procs)
        for proc in procs:
            proc.kill()  # SIGKILL to this exact pid
        for proc in procs:
            proc.join(30)
            assert proc.exitcode == -signal.SIGKILL
        assert not Path(path).exists()
        assert not any(
            name.endswith(f"{layout.instance_id}.mmap")
            for name in os.listdir("/dev/shm")
        )
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()  # SIGKILL to this exact pid
                proc.join(5)
        # Only this test's random name; present only if an assertion failed early.
        Path(path).unlink(missing_ok=True)
