# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Busy shutdown: real source, SSD files and independent writer/reader processes."""

import json
import multiprocessing
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_host_page_ssd import LAYOUT, key, options
from tests.v1.kv_offload.test_unified_ssd import attach
from tests.v1.kv_offload.test_unified_units import publish
from vllm.v1.kv_offload.host.page_table import HostPageTable, PageKind
from vllm.v1.kv_offload.host.persist import (
    HostPageSsd,
    plan_final_flush,
    run_final_flush,
)

pytestmark = pytest.mark.cpu_test
ROW = 1 << 20
COUNT = 50
GRACE = 2.0


def _environment(root):
    table = HostPageTable(COUNT + 2, num_workers=1)
    backing = bytearray(len(table.pages) * ROW)
    view = memoryview(backing).cast("B", shape=[len(table.pages), ROW])
    spec, opts = options(
        root,
        max_size_bytes=128 << 20,
        write_delay_s=300,
        write_bytes_per_second=64 << 20,
        shutdown_io_grace_s=GRACE,
    )
    ssd = HostPageSsd(table, view, spec, LAYOUT, opts)
    env = SimpleNamespace(table=table, backing=backing, ssd=ssd)
    env.manager = attach(env)
    return env


def _write(root, source, result, case):
    env = _environment(root)
    with source.open("rb") as stream:
        for n in range(COUNT):
            handle = publish(env.table, key(n, 1), PageKind.STATE)
            start = handle.page_id * ROW
            env.backing[start : start + ROW] = stream.read(ROW)
    now = env.ssd.clock()
    if case == "budget":
        # The submission budget is shorter than the 2 s IO grace.
        report = env.ssd.final_flush(budget_s=0.1)
    else:
        # A busy stop with an explicit --shutdown-timeout 30: request drain left
        # GRACE + 0.1 s of the shared deadline (zero: none), not the 25 s budget.
        left = GRACE + 0.1 if case == "deadline" else 0.0
        plan = plan_final_flush(25, 30, now + left, now)
        report = run_final_flush(env.ssd, plan, env.ssd.clock())
    persisted = [
        n
        for n in range(COUNT)
        if env.table.is_persisted(env.table.get(env.table.lookup(key(n, 1))))
    ]
    result.write_text(json.dumps({"report": asdict(report), "persisted": persisted}))
    env.ssd.close(io_grace_s=GRACE)


def _read(root, source, result):
    env = _environment(root)
    persisted = set(json.loads(result.read_text())["persisted"])
    tier = env.ssd._component_tier
    with source.open("rb") as stream:
        for n in range(COUNT):
            tier_key = env.ssd._tier_key(key(n, 1))
            # Written pages read back byte for byte; the rest are misses.
            assert bool(tier.lookup(tier_key)) == (n in persisted)
            if n not in persisted:
                continue
            path = tier.file_mapper.get_file_name(tier_key)
            tier._read_page(tier_key, path, 0)
            stream.seek(n * ROW)
            assert env.backing[:ROW] == stream.read(ROW)
    env.ssd.close(io_grace_s=GRACE)


@pytest.mark.parametrize("case", ["budget", "deadline", "zero"])
def test_short_final_flush_writes_and_restart_reads_real_source(tmp_path, case):
    source = tmp_path / "source.bin"
    with source.open("wb") as stream:
        for n in range(COUNT):
            stream.write(bytes([n + 1]) * ROW)
    result = tmp_path / "result.json"
    root = tmp_path / "ssd"
    context = multiprocessing.get_context("fork")
    writer = context.Process(target=_write, args=(root, source, result, case))
    writer.start()
    writer.join(30)
    assert writer.exitcode == 0
    writer.close()
    saved = json.loads(result.read_text())
    print(json.dumps(saved, sort_keys=True))
    report = saved["report"]
    assert report["failed"] == 0
    assert report["stuck"] == report["other_io_stuck"] == 0
    assert not report["quarantined"]
    if case == "zero":
        assert report["skipped"] == "shutdown window exhausted"
        assert report["completed"] == 0 and not saved["persisted"]
    else:
        assert report["completed"] > 0
        assert saved["persisted"]
    reader = context.Process(target=_read, args=(root, source, result))
    reader.start()
    reader.join(30)
    assert reader.exitcode == 0
    reader.close()
