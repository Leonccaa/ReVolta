# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Variant guarantees charged on the union of missing persistence objects."""

from collections import Counter
from collections.abc import Callable

from vllm.v1.kv_offload.unified.units import Unit, UnitId, UnitIndex

Obligation = tuple[str, object]


class PersistenceCredit:
    def __init__(
        self,
        units: UnitIndex,
        obligations: Callable[[Unit], dict[Obligation, int]],
        *,
        window_s: float,
        io_bytes: int,
        initial_rate: float,
        floor_rate: float,
        clock: Callable[[], float],
        available: Callable[[], bool],
        capacity_bytes: int,
        eligible: Callable[[Unit], bool],
    ):
        self.units, self.obligations = units, obligations
        self.window_s = max(0.0, window_s)
        self.io_bytes = io_bytes
        self.floor_rate = max(1.0, floor_rate)
        self.rate = max(self.floor_rate, initial_rate)
        self.clock, self.available = clock, available
        self.capacity_bytes = capacity_bytes
        self.eligible = eligible
        self.guaranteed: set[UnitId] = set()
        self.promotions_enabled = True
        self.excluded: set[UnitId] = set()
        self.events: Counter[str] = Counter()
        self.last_completion = clock()
        self.measurement_start: float | None = None
        self._measurement_bytes = 0
        self.bytes = 0
        self.h = self.q_eff = self.d_dyn = 0
        self.reconcile(promote=False)

    def _union(
        self,
        selected: set[UnitId],
        bills: dict[UnitId, dict[Obligation, int]] | None = None,
    ) -> int:
        objects: dict[Obligation, int] = {}
        for uid in selected:
            unit = self.units.units.get(uid)
            if unit is not None:
                objects.update(self.obligations(unit) if bills is None else bills[uid])
        return sum(objects.values())

    def _variants(self) -> list[Unit]:
        return sorted(
            (u for u in self.units.units.values() if u.parent is not None),
            key=lambda u: u.last_use,
        )

    def reconcile(self, *, promote: bool = True, reason: str = "shrink") -> None:
        before = set(self.guaranteed)
        self.h = (
            min(self.capacity_bytes, int(self.window_s * self.rate * 0.8))
            if self.available()
            else 0
        )
        self.q_eff = min(self.io_bytes, self.h)
        self.d_dyn = self.h - self.q_eff
        self.guaranteed.intersection_update(self.units.units)
        self.excluded.intersection_update(self.units.units)
        variants = self._variants()
        bills = {u.uid: self.obligations(u) for u in variants}
        for uid in tuple(self.guaranteed):
            if not self.eligible(self.units.units[uid]):
                self.guaranteed.remove(uid)
                self.events[f"downgraded_{reason}"] += 1
        for uid in tuple(self.guaranteed):
            if not bills[uid]:
                self.guaranteed.remove(uid)
                self.events["release_committed"] += 1
        for unit in variants:
            if self._union(self.guaranteed, bills) <= self.d_dyn:
                break
            if unit.uid in self.guaranteed and bills[unit.uid]:
                self.guaranteed.remove(unit.uid)
                self.events[f"downgraded_{reason}"] += 1
        if promote and self.promotions_enabled:
            for unit in variants:
                if (
                    unit.uid in self.guaranteed
                    or unit.uid in self.excluded
                    or not self.eligible(unit)
                ):
                    continue
                if not bills[unit.uid]:
                    continue
                if self._union(self.guaranteed | {unit.uid}, bills) <= self.d_dyn:
                    self.guaranteed.add(unit.uid)
                    self.events["promote"] += 1
        self.bytes = self._union(self.guaranteed, bills)
        assert self.bytes <= self.d_dyn
        for uid in before ^ self.guaranteed:
            self.units.evidence_changed(uid)

    def event(self, event: str, unit: Unit) -> None:
        if event == "register":
            self.reconcile(promote=False)
            if (
                self.eligible(unit)
                and self._union(self.guaranteed | {unit.uid}) <= self.d_dyn
            ):
                self.guaranteed.add(unit.uid)
                self.events["acquire"] += 1
            else:
                self.events["acquire_best_effort"] += 1
            self.bytes = self._union(self.guaranteed)
        elif event == "revoke":
            if unit.uid in self.guaranteed:
                self.events["release_revoked"] += 1
            self.guaranteed.discard(unit.uid)
            self.excluded.discard(unit.uid)
            self.reconcile()

    def supersede(self, uids: set[UnitId], target: UnitId | None = None) -> None:
        released = 0
        for uid in uids:
            self.excluded.add(uid)
            if uid in self.guaranteed:
                self.guaranteed.remove(uid)
                released += 1
        self.events["guarantee_superseded"] += released
        self.reconcile(promote=False)
        unit = self.units.units.get(target) if target is not None else None
        if (
            released
            and unit is not None
            and unit.uid not in self.guaranteed
            and unit.uid not in uids
            and self.eligible(unit)
            and self._union(self.guaranteed | {unit.uid}) <= self.d_dyn
        ):
            self.guaranteed.add(unit.uid)
            self.excluded.discard(unit.uid)
            self.events["transfer"] += 1
        else:
            self.events["release_superseded"] += released
        self.reconcile()

    def pressure(self, uid: UnitId) -> None:
        self.guaranteed.discard(uid)
        self.excluded.add(uid)
        self.events["guarantee_revoked_pressure"] += 1
        self.reconcile(promote=False)
        if uid in self.units.units:
            self.units.evidence_changed(uid)

    def started(self) -> None:
        if self.measurement_start is None:
            self.measurement_start = self.clock()
            self._measurement_bytes = 0

    def committed(self, nbytes: int, *, busy: bool = False) -> None:
        now = self.clock()
        if self.measurement_start is not None:
            # Receipts can arrive together after seconds of queued IO, CRC
            # and component/manifest fsync. Keep the entire busy interval's
            # bytes and wall time, even across scheduler polls, instead of
            # giving each receipt's bytes a fresh timing origin or weight.
            self._measurement_bytes += nbytes
            measured = self._measurement_bytes / max(
                now - self.measurement_start, 0.001
            )
            self.rate = max(self.floor_rate, measured)
        if not busy:
            self.measurement_start = None
        self.last_completion = now
        self.events["committed_bytes"] += nbytes
        self.reconcile()

    def tick(self) -> None:
        if (
            self.bytes
            and self.clock() - self.last_completion > max(1.0, self.window_s)
            and self.rate != self.floor_rate
        ):
            self.rate = self.floor_rate
            self.events["rate_stalled"] += 1
            self.reconcile()

    def snapshot(self) -> dict[str, int | float]:
        return {
            "guaranteed_bytes": self.bytes,
            "guaranteed_variants": len(self.guaranteed),
            "best_effort_variants": sum(
                u.parent is not None
                and u.uid not in self.guaranteed
                and bool(self.obligations(u))
                for u in self.units.units.values()
            ),
            "shutdown_budget_bytes": self.h,
            "write_inflight_limit_bytes": self.q_eff,
            "dirty_limit_bytes": self.d_dyn,
            "write_rate_bytes_s": self.rate,
            "shutdown_completion_s": self.bytes / self.rate,
            **{f"credit_{k}": v for k, v in self.events.items()},
        }
