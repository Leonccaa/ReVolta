# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential lifecycle oracle frozen at da99, including allocator order."""

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host.page_table import HostPageTable, PageKind, PageState

spec = importlib.util.spec_from_file_location(
    "da99_page_table", Path(__file__).parent / "fixtures" / "da99_page_table.py"
)
assert spec is not None and spec.loader is not None
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


def key(i):
    return make_offload_key(i.to_bytes(32, "little"), 0)


def norm(value):
    if hasattr(value, "__dataclass_fields__"):
        return {name: norm(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {
            tuple(k) if isinstance(k, tuple) else k: norm(v) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [norm(v) for v in value]
    return value


def snapshot(table):
    pages = []
    for p in table.pages:
        row = {
            name: norm(getattr(p, name)) for name in old.HostPage.__dataclass_fields__
        }
        # Reclaiming already-retired identity records is the intended metadata change.
        row["retirements"] = [
            norm(r) for r in p.retirements if r.retire_after > table.watermark
        ]
        pages.append(row)
    return (
        pages,
        norm(table.index),
        list(table._lru),
        table._free[:],
        table.available(),
        table.watermark,
        table.ssd_epoch,
    )


def apply(table, op, pid, arg):
    p = table.pages[pid]
    h = p.handle
    if op == "reserve":
        return table.reserve(loading_key=key(arg) if arg % 2 else None)
    if op == "complete":
        return table.complete_step(arg[0], arg[1])
    if op == "reset":
        return table.reset()
    if op == "collect":
        return table.collect()
    if op == "touch":
        if isinstance(table, HostPageTable):
            return table.touch(h, arg)
        p.last_use = max(p.last_use, arg)
    if op == "bind":
        return table.bind(h, arg[0], arg[1])
    if op == "unbind":
        return table.unbind(h, arg[0], arg[1], arg[2])
    if op == "pin":
        return table.pin(h, arg)
    if op == "release":
        return table.release(h, arg)
    if op == "io":
        return table.start_io(h, arg)
    if op == "finish":
        return table.finish_io(h, arg)
    if op == "load":
        return table.complete_load(h, arg)
    if op == "persist":
        return table.mark_persisted(h)
    if op == "seal":
        return table.open_seal(h, *arg[:2], key(arg[2]), arg[3])
    if op == "receipt":
        return table.seal_result(p.ticket, *arg)
    if op == "role":
        if isinstance(table, HostPageTable):
            return table.set_bind_role(h, *arg, "cached")
        p.binds[arg] = "cached"


@pytest.mark.parametrize("seed", range(20))
def test_randomized_da99_differential(seed):
    rng = random.Random(seed)
    tables = [old.HostPageTable(13, 3), HostPageTable(13, 3)]
    for event in range(3000):
        ref = tables[0]
        live = [p for p in ref.pages[1:] if p.state != old.PageState.FREE]
        p = rng.choice(live) if live else ref.pages[1]
        step = ref.watermark + rng.randrange(1, 6)
        ops = [
            ("reserve", rng.randrange(12)),
            ("complete", (step, rng.randrange(1, 8))),
            ("collect", None),
        ]
        if event % 71 == 0:
            ops.append(("reset", None))
        if live:
            ops += [
                ("touch", step),
                ("bind", (rng.randrange(4), rng.randrange(3))),
                ("pin", rng.randrange(4)),
                ("io", rng.randrange(4)),
                ("persist", None),
            ]
            if p.binds:
                b = rng.choice(list(p.binds))
                ops += [("unbind", (*b, step)), ("role", b)]
                if (
                    p.state == old.PageState.PRIVATE
                    and p.valid
                    and p.ticket is None
                    and p.compute_result != old.Result.FAILED
                    and (p.writer is None or p.writer[:2] == b)
                ):
                    ops.append(("seal", (*b, rng.randrange(12), step)))
            if p.leases:
                ops.append(("release", rng.choice(sorted(p.leases))))
            if p.io:
                ops.append(("finish", rng.choice(sorted(p.io))))
            if p.state == old.PageState.LOADING:
                ops.append(("load", rng.choice([True, False])))
            if p.ticket:
                ranks = rng.randrange(1, 8)
                ops.append(
                    (
                        "receipt",
                        (ranks, ranks & rng.randrange(8), ranks & rng.randrange(8)),
                    )
                )
        op, arg = rng.choice(ops)
        results = [norm(apply(t, op, p.handle.page_id, arg)) for t in tables]
        assert results[0] == results[1], (seed, event, op, arg)
        assert snapshot(tables[0]) == snapshot(tables[1]), (seed, event, op, arg)
        new = tables[1]
        assert Counter(new.state_counts) == Counter(p.state for p in new.pages[1:])
        assert new.persisted_pages == sum(
            new.is_persisted(p) for p in new.pages[1:] if p.state != PageState.FREE
        )
        assert len(new._pending) <= 4 * len(new.pages) + 64
        assert len(new._retiring) <= 4 * len(new.pages) + 64


def test_metadata_bound_and_partial_retirement_with_permanent_owner():
    table = HostPageTable(3, 2)
    h = table.reserve(loading_key=key(1))
    table.start_io(h, 1)
    assert table.complete_load(h, True)
    for step in range(1, 10001):
        table.bind(h, 7, step)
        table.unbind(h, 7, step, step)
        table.touch(h, step + 1)
        table.complete_step(step, 3)
        assert not table.get(h).retirements
        # A newer access must not retain already-retired binding identities.
        assert len(table._pending) <= 76
        assert len(table._retiring) <= 76
    table.complete_step(10001, 3)
    assert not table.get(h).retirements
    table.finish_io(h, 1)
    assert table.available() == 2
    for _ in range(1000):
        table.touch(h, 10002)
    assert table.heap_compactions > 0


def test_no_hot_path_table_iteration_and_live_loading_gap():
    table = HostPageTable(100)
    h = table.reserve(loading_key=key(1))

    class IndexedOnly(list):
        def __iter__(self):
            raise AssertionError("hot path table scan")

        def __getitem__(self, index):
            assert not isinstance(index, slice)
            return super().__getitem__(index)

    table.pages = IndexedOnly(table.pages)
    table.collect()
    assert table.available() == 98
    assert table.get(h).kind == PageKind.MAIN and table.get(h).group == 0
    assert not table.reclaimable(table.get(h))


def test_same_epoch_touch_invalidates_due_candidate_and_lru_head_protection():
    table = HostPageTable(3, 2)
    a = table.reserve(loading_key=key(1))
    table.complete_load(a, True)
    b = table.reserve(loading_key=key(2))
    table.complete_load(b, True)
    table.touch(a, 1)
    table.touch(a, 3)
    table.complete_step(1, 3)
    assert table.reserve().page_id == b.page_id
    table.complete_step(3, 3)
    table.complete_step(2, 1)
    assert not table.reclaimable(table.get(a))
    table.complete_step(2, 2)
    assert table.reclaimable(table.get(a))
