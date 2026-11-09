# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU value ordering over all live references, separate from owner safety."""

from collections import deque
from collections.abc import Callable
from heapq import heapify, heappop, heappush

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.host.page_table import HostPage, HostPageTable
from vllm.v1.kv_offload.unified.units import UnitId, UnitIndex, ValueClass


class UnitValuePolicy:
    def __init__(self, table: HostPageTable, units: UnitIndex):
        self.table, self.units = table, units
        self.heap: list[tuple[int, int, int, int, int]] = []
        self.compactions = 0
        self.work_max = 0
        self.on_ready: Callable[[OffloadKey], None] = lambda key: None
        self.on_reset: Callable[[], None] = lambda: None
        table.value_policy = self

    def _valid(self, entry: tuple[int, int, int, int, int]) -> bool:
        _, _, pid, epoch, revision = entry
        p = self.table.pages[pid]
        return (
            p.handle.epoch == epoch
            and p.revision == revision
            and p.key is not None
            and self.table.reclaimable(p)
        )

    def refresh(self, page: HostPage) -> None:
        if self.table.reclaimable(page) and page.key is not None:
            heappush(
                self.heap,
                (
                    int(self.units.value(page.key, page.kind)),
                    page.lru_clock,
                    page.handle.page_id,
                    page.handle.epoch,
                    page.revision,
                ),
            )
        if len(self.heap) > 4 * len(self.table.pages) + 64:
            self.heap[:] = [e for e in self.heap if self._valid(e)]
            heapify(self.heap)
            self.compactions += 1

    def published(self, page: HostPage) -> None:
        assert page.key is not None
        self.units.cpu_ready(page.key, page.handle, page.kind)
        if self.table.is_persisted(page):
            self.units.persisted(page.key, page.handle)
        self.on_ready(page.key)

    def lost(self, page: HostPage) -> None:
        if page.key is not None:
            self.units.cpu_lost(page.key, page.handle)

    def persisted(self, page: HostPage) -> None:
        assert page.key is not None
        self.units.persisted(page.key, page.handle)
        self.on_ready(page.key)

    def ssd_lost(self, key, handle) -> None:
        self.units.ssd_lost(key, handle)

    def reset(self) -> None:
        self.on_reset()
        self.units.reset()
        self.heap.clear()

    def order(self, page: HostPage) -> tuple[int, int]:
        # A successful computation with a failed seal remains keyless PRIVATE.
        # Retired local bindings have no canonical unit value to preserve.
        value = self.units.value(page.key, page.kind) if page.key is not None else 0
        return int(value), page.lru_clock

    def select(self, pending: int, writeback: bool) -> HostPage | None:
        store = getattr(self.table.persistence, "unit_store", None)
        if store is not None:
            return self._select_units(store, pending, writeback)
        seen: set[UnitId] = set()
        deferred: deque[tuple[int, int, int, int, int]] = deque()
        work = 0
        try:
            while self.heap or deferred:
                if not self.heap:
                    work += 1
                    entry = deferred.popleft()
                    if not self._valid(entry):
                        continue
                    page = self.table.pages[entry[2]]
                    assert page.key is not None
                    # An opportunistic write protected one page of this
                    # unit, but the admitted caller still needs a frame.
                    # Revoke best-effort value, retaining every IO owner,
                    # and re-enter the heap so newly zero-ref MAIN wins.
                    deferred_refs = self.units.components[page.key].refs
                    u = min(
                        (self.units.units[uid] for uid in deferred_refs),
                        key=lambda u: (
                            not self.units.interior(u),
                            u.parent is None and bool(u.variants),
                            u.last_use,
                        ),
                    )
                    self.units.events["unit_evict_after_writeback"] += 1
                    self.units.revoke(u.uid, u.gen, "value")
                    continue
                entry = heappop(self.heap)
                work += 1
                if not self._valid(entry):
                    continue
                value, _, pid, _, _ = entry
                page = self.table.pages[pid]
                assert page.key is not None
                if value <= ValueClass.SSD_REDUNDANT:
                    return page
                c = self.units.components[page.key]
                refs = [self.units.units[uid] for uid in c.refs if uid not in seen]
                if not refs:
                    deferred.append(entry)
                    continue
                # Visit each unit at most once in this allocation attempt.
                # Shared pages are reclassified after every revocation.
                u = min(
                    refs,
                    key=lambda u: (
                        not self.units.interior(u),
                        u.parent is None and bool(u.variants),
                        u.last_use,
                    ),
                )
                seen.add(u.uid)
                if not self.units.interior(u):
                    if self.table.is_persisted(page):
                        # unified host budgeting's page-level writeback transition can
                        # spill a
                        # leaf across CPU and SSD before all its files exist.
                        # Recycling this verified copy preserves R_union;
                        # unit-wide write transactions arrive in unit SSD transactions.
                        self.units.events["leaf_page_spilled"] += 1
                        return page
                    hooks = self.table.persistence if writeback else None
                    if hooks is not None and hooks.try_writeback(
                        page, self.table.net_headroom(pending)
                    ):
                        continue
                    self.units.events["unit_evict_unwritten"] += 1
                self.units.revoke(u.uid, u.gen, "value")
                # Revocation refreshes all affected pages through _refresh.
                # A remaining shared leaf may still require another pass.
            return None
        finally:
            for entry in deferred:
                if self._valid(entry):
                    heappush(self.heap, entry)
            self.work_max = max(self.work_max, work)

    def _select_units(self, store, pending: int, writeback: bool) -> HostPage | None:
        seen: set[UnitId] = set()
        deferred = []
        work = 0
        try:
            while True:
                while self.heap:
                    entry = heappop(self.heap)
                    work += 1
                    if not self._valid(entry):
                        continue
                    page = self.table.pages[entry[2]]
                    assert page.key is not None
                    if (
                        self.units.value(page.key, page.kind)
                        <= ValueClass.SSD_REDUNDANT
                    ):
                        return page
                    refs = [
                        self.units.units[uid]
                        for uid in self.units.components[page.key].refs
                        if uid not in seen
                    ]
                    if not refs:
                        deferred.append(entry)
                        continue
                    unit = min(
                        refs,
                        key=lambda u: (
                            not self.units.interior(u),
                            u.parent is None and bool(u.variants),
                            u.last_use,
                        ),
                    )
                    seen.add(unit.uid)
                    if self.units.interior(unit):
                        self.units.revoke(unit.uid, unit.gen, "value")
                        continue
                    if unit.parent is None and unit.variants:
                        deferred.append(entry)
                        continue
                    if (
                        writeback
                        and unit.parent is not None
                        and store.start_write(unit, self.table.net_headroom(pending))
                    ):
                        deferred.append(entry)
                        continue
                    if unit.uid in store.credit.guaranteed:
                        deferred.append(entry)
                        continue
                    if (
                        unit.parent is None
                        and writeback
                        and self.table.persistence is not None
                        and self.table.persistence.try_writeback(
                            page, self.table.net_headroom(pending)
                        )
                    ):
                        deferred.append(entry)
                        continue
                    self.units.events["unit_evict_unwritten"] += 1
                    self.units.revoke(unit.uid, unit.gen, "value")
                ordinary = {}
                for entry in deferred:
                    if not self._valid(entry):
                        continue
                    page = self.table.pages[entry[2]]
                    assert page.key is not None
                    for uid in self.units.components[page.key].refs:
                        unit = self.units.units[uid]
                        if (
                            uid not in store.credit.guaranteed
                            and not unit.variants & store.credit.guaranteed
                        ):
                            ordinary[uid] = unit
                if ordinary:
                    unit = min(ordinary.values(), key=lambda u: u.last_use)
                    self.units.revoke(unit.uid, unit.gen, "value")
                    writeback = False
                    continue
                # Every ordinary candidate was considered. Revoke the oldest
                # remaining guarantee, retaining all physical IO owners.
                candidates = [
                    self.units.units[uid] for uid in seen & store.credit.guaranteed
                ]
                if not candidates:
                    return None
                unit = min(candidates, key=lambda u: u.last_use)
                store.credit.pressure(unit.uid)
                self.units.revoke(unit.uid, unit.gen, "value")
                writeback = False
        finally:
            for entry in deferred:
                if self._valid(entry):
                    heappush(self.heap, entry)
            self.work_max = max(self.work_max, work)
