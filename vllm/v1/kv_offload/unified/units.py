# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned reuse graph. GPU residency is deliberately unknown.

Content references never pin physical frames. CPU handles and SSD evidence
are separate from the graph, and all events carry their incarnation.
SSD cold recovery additionally requires a committed variant manifest.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import IntEnum

from vllm.v1.kv_offload.base import OffloadKey, get_offload_block_hash
from vllm.v1.kv_offload.host.page_table import PageHandle, PageKind


class ValueClass(IntEnum):
    ZERO_MAIN = 0
    ZERO_COMPONENT = 1
    SSD_REDUNDANT = 2
    INTERIOR = 3
    LEAF = 4


@dataclass(frozen=True)
class UnitId:
    boundary: bytes
    reset: int
    ahead: bytes | None = None


@dataclass
class Component:
    kind: PageKind
    cpu: PageHandle | None = None
    # Content-incarnation evidence, retained after CPU eviction. unit SSD transactions
    # replaces
    # it with verified file identities and reports loss through ssd_lost().
    persisted: PageHandle | None = None
    refs: set[UnitId] = field(default_factory=set)
    demoted: bool = False


@dataclass
class Unit:
    uid: UnitId
    gen: int
    keys: tuple[OffloadKey, ...]
    complete: bool = False
    parent: UnitId | None = None
    variants: set[UnitId] = field(default_factory=set)
    last_use: int = 0
    missing: dict[str, int] = field(default_factory=dict)
    boundary_position: int = 0
    ahead_position: int = 0
    draft_range: tuple[int, int] = (0, 0)
    retention: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VariantOffer:
    boundary: UnitId
    gen: int
    ahead: bytes
    main: tuple[OffloadKey, ...]
    boundary_position: int = 0
    ahead_position: int = 0
    draft_range: tuple[int, int] = (0, 0)
    retention: dict = field(default_factory=dict)


class UnitIndex:
    def __init__(self, refresh: Callable[[PageHandle], None] = lambda h: None):
        self.units: dict[UnitId, Unit] = {}
        self.components: dict[OffloadKey, Component] = {}
        self.refresh = refresh
        self.reset_epoch = 0
        self._generation = 0
        self._clock = 0
        self.events: Counter[str] = Counter()
        self.propagation_work_max = 0
        # None means unified host budgeting's explicit files-only approximation, never a
        # claim
        # that a manifest has been committed.
        self.manifest_ok: Callable[[Unit], bool] | None = None
        self.on_event: Callable[[str, Unit], None] = lambda event, unit: None
        self.on_reset: Callable[[], None] = lambda: None
        self.protected: set[UnitId] = set()
        self.registration_blocked: set[UnitId] = set()

    def _changed(self, keys: Iterable[OffloadKey]) -> None:
        unique = set(keys)
        self.propagation_work_max = max(self.propagation_work_max, len(unique))
        for key in unique:
            component = self.components.get(key)
            if component is not None and component.cpu is not None:
                self.refresh(component.cpu)
            self._gc_component(key)

    def _gc_component(self, key: OffloadKey) -> None:
        c = self.components.get(key)
        if c is not None and not (c.cpu or c.persisted or c.refs):
            del self.components[key]

    def _presence_changed(
        self, key: OffloadKey, cpu: bool, ssd: bool
    ) -> set[OffloadKey]:
        c = self.components[key]
        before = {"cpu": cpu, "ssd": ssd, "union": cpu or ssd}
        after = {
            "cpu": c.cpu is not None,
            "ssd": c.persisted is not None,
            "union": c.cpu is not None or c.persisted is not None,
        }
        affected = {key}
        for uid in c.refs:
            u = self.units[uid]
            was_complete = u.missing["ssd"] == 0
            for tier in before:
                u.missing[tier] += int(before[tier]) - int(after[tier])
            if was_complete != (u.missing["ssd"] == 0):
                affected.update(u.keys)
        return affected

    def _initialize_presence(self, u: Unit) -> None:
        u.missing = {
            tier: sum(not self._resident(k, tier) for k in u.keys)
            for tier in ("cpu", "ssd", "union")
        }

    def cpu_ready(self, key: OffloadKey, handle: PageHandle, kind: PageKind) -> None:
        c = self.components.setdefault(key, Component(kind))
        if c.kind != kind:
            raise ValueError("Component kind changed for the same key")
        cpu, ssd = c.cpu is not None, c.persisted is not None
        c.cpu = handle
        self._changed(self._presence_changed(key, cpu, ssd))

    def persisted(self, key: OffloadKey, handle: PageHandle) -> None:
        c = self.components.get(key)
        if c is None or c.cpu != handle:
            return
        cpu, ssd = c.cpu is not None, c.persisted is not None
        c.persisted = handle
        self._changed(self._presence_changed(key, cpu, ssd))

    def cpu_lost(self, key: OffloadKey, handle: PageHandle) -> None:
        c = self.components.get(key)
        if c is None or c.cpu != handle:
            return
        cpu, ssd = c.cpu is not None, c.persisted is not None
        c.cpu = None
        affected = self._presence_changed(key, cpu, ssd)
        self._loss((key,))
        self._changed(affected)

    def ssd_lost(self, key: OffloadKey, incarnation: PageHandle) -> None:
        c = self.components.get(key)
        if c is None or c.persisted != incarnation:
            return
        cpu, ssd = c.cpu is not None, c.persisted is not None
        c.persisted = None
        affected = self._presence_changed(key, cpu, ssd)
        self._loss((key,))
        self._changed(affected)

    def _loss(self, keys: Iterable[OffloadKey]) -> None:
        keys = tuple(keys)
        refs = {uid for key in keys for uid in self.components[key].refs}
        for uid in refs:
            u = self.units.get(uid)
            if u is not None and u.complete and not self.recoverable(uid, "union"):
                self.revoke(uid, u.gen, "last_copy")
        self._changed(keys)

    def affected_keys(self, keys: Iterable[OffloadKey]) -> set[OffloadKey]:
        out = set(keys)
        for key in tuple(out):
            c = self.components.get(key)
            if c is not None:
                for uid in c.refs:
                    out.update(self.units[uid].keys)
        return out

    def begin(self, boundary: bytes, keys: Iterable[OffloadKey]) -> Unit:
        uid = UnitId(boundary, self.reset_epoch)
        if uid in self.units:
            return self.units[uid]
        keys = tuple(dict.fromkeys(keys))
        if not keys or any(k not in self.components for k in keys):
            raise ValueError("Building units need a complete component inventory")
        states = [k for k in keys if self.components[k].kind == PageKind.STATE]
        if not states or any(get_offload_block_hash(k) != boundary for k in states):
            raise ValueError("Boundary identity must match every STATE component")
        self._generation += 1
        u = Unit(uid, self._generation, keys)
        self._initialize_presence(u)
        self.units[uid] = u
        for key in keys:
            self.components[key].refs.add(uid)
        self.events["build_admitted"] += 1
        self._changed(keys)
        return u

    def finish(self, uid: UnitId, gen: int) -> bool:
        u = self.units.get(uid)
        if u is None or u.gen != gen or not self.recoverable(uid, "union"):
            return False
        if u.complete:
            return True
        u.complete = True
        self.touch(uid)
        self.events["build_complete"] += 1
        self._changed(u.keys)
        return True

    def register(self, offer: VariantOffer) -> Unit | None:
        if offer.boundary in self.registration_blocked:
            return None
        b = self.units.get(offer.boundary)
        reason = (
            "variant_b_absent_terminal"
            if b is None
            else "variant_b_regen"
            if b.gen != offer.gen
            else "variant_b_building_wait"
            if not b.complete
            else "variant_main_missing"
            if not offer.main
            or any(
                k not in self.components or not self._resident(k, "union")
                for k in offer.main
            )
            else None
        )
        if reason is not None:
            self.events[reason] += 1
            if reason in ("variant_b_absent_terminal", "variant_b_building_wait"):
                self.events["variant_b_missing"] += 1
            return None
        assert b is not None
        if (
            not offer.main
            or get_offload_block_hash(offer.main[-1]) != offer.ahead
            or any(self.components[k].kind != PageKind.MAIN for k in offer.main)
            or (
                offer.ahead != b.uid.boundary
                and (
                    len(offer.main) < 2
                    or get_offload_block_hash(offer.main[-2]) != b.uid.boundary
                )
            )
        ):
            self.events["variant_identity_invalid"] += 1
            return None
        uid = UnitId(b.uid.boundary, b.uid.reset, offer.ahead)
        if uid in self.units:
            u = self.units[uid]
            if offer.retention:
                # Reusing an existing checkpoint does not publish a newer
                # request's point. Its prior resend exemption lasts until a
                # genuinely newer boundary has registered successfully.
                if not u.retention:
                    u.retention.update(offer.retention)
                for flag in ("junction", "branch"):
                    if offer.retention.get(flag):
                        u.retention[flag] = True
                self.on_event("observe", u)
            return u
        self._generation += 1
        # Materialize transitive component references. A shared STATE/DRAFT
        # page must consider every variant's cold-recovery predicate too.
        keys = tuple(dict.fromkeys((*b.keys, *offer.main)))
        u = Unit(uid, self._generation, keys, complete=True, parent=b.uid)
        self._initialize_presence(u)
        self.units[uid] = u
        b.variants.add(uid)
        for key in keys:
            self.components[key].refs.add(uid)
        self.touch(uid)
        self.events["variant_registered"] += 1
        u.boundary_position = offer.boundary_position
        u.ahead_position = offer.ahead_position
        u.draft_range = offer.draft_range
        u.retention = dict(offer.retention)
        self._changed(keys)
        self.on_event("register", u)
        return u

    def _resident(self, key: OffloadKey, tier: str) -> bool:
        c = self.components[key]
        if tier == "cpu":
            return c.cpu is not None
        if tier == "ssd":
            return c.persisted is not None
        if tier == "union":
            return c.cpu is not None or c.persisted is not None
        raise ValueError("GPU residency is unknown; use cpu, ssd or union")

    def recoverable(self, uid: UnitId, tier: str) -> bool:
        u = self.units.get(uid)
        if tier not in ("cpu", "ssd", "union"):
            raise ValueError("GPU residency is unknown; use cpu, ssd or union")
        return u is not None and u.missing[tier] == 0

    def persistence_obligations(
        self, uid: UnitId
    ) -> tuple[frozenset[OffloadKey], bool]:
        """Read-only input for unit SSD transactions credit. unified host budgeting
        issues no guarantees.

        Files and the future manifest predicate are deliberately independent;
        an empty file set can still owe one manifest.
        """
        u = self.units[uid]
        return (
            frozenset()
            if u.missing["ssd"] == 0
            else frozenset(k for k in u.keys if not self._resident(k, "ssd")),
            self.manifest_ok is not None and not self.manifest_ok(u),
        )

    def files_complete(self, uid: UnitId) -> bool:
        return self.recoverable(uid, "ssd")

    def cold_recoverable(self, uid: UnitId) -> bool:
        u = self.units.get(uid)
        return (
            u is not None
            and self.files_complete(uid)
            and (self.manifest_ok is None or self.manifest_ok(u))
        )

    def evidence_changed(self, uid: UnitId) -> None:
        """unit SSD transactions manifest-only changes must invalidate CPU value
        revisions too."""
        u = self.units.get(uid)
        if u is not None:
            self._changed(u.keys)

    def touch(self, uid: UnitId) -> None:
        self._clock += 1
        self.units[uid].last_use = self._clock

    def demote(self, keys: Iterable[OffloadKey], demoted: bool) -> None:
        changed = {
            key
            for key in keys
            if key in self.components and self.components[key].demoted != demoted
        }
        if changed:
            affected = {uid for key in changed for uid in self.components[key].refs}
            before = {uid: self.interior(self.units[uid]) for uid in affected}
            for key in changed:
                self.components[key].demoted = demoted
            refresh = set(changed)
            for uid in affected:
                unit = self.units[uid]
                # Seven separate STATE-group updates must not refresh the
                # complete MAIN chain seven times. Only the first revival
                # or the final demotion changes the unit's CPU value class.
                if before[uid] != self.interior(unit):
                    refresh.update(unit.keys)
            self._changed(refresh)

    def value(self, key: OffloadKey, kind: PageKind) -> ValueClass:
        c = self.components.get(key)
        if c is None or not c.refs:
            return (
                ValueClass.ZERO_MAIN
                if kind == PageKind.MAIN
                else ValueClass.ZERO_COMPONENT
            )
        if all(self.cold_recoverable(uid) for uid in c.refs):
            return ValueClass.SSD_REDUNDANT
        # Every live reference must be interior. One live leaf wins over
        # any number of superseded boundaries sharing the physical page.
        if all(self.interior(self.units[uid]) for uid in c.refs):
            return ValueClass.INTERIOR
        return ValueClass.LEAF

    def interior(self, u: Unit) -> bool:
        b = self.units[u.parent] if u.parent is not None else u
        states = [
            self.components[k]
            for k in b.keys
            if self.components[k].kind == PageKind.STATE
        ]
        return bool(states) and all(c.demoted for c in states)

    def revoke(self, uid: UnitId, gen: int, reason: str) -> bool:
        u = self.units.get(uid)
        if u is None or u.gen != gen:
            return False
        if uid in self.protected:
            return False
        # Remove eligibility first; then all logical references; only then
        # refresh pages and expose zero-reference reclaim candidates.
        u.complete = False
        for child in tuple(u.variants):
            self.revoke(child, self.units[child].gen, reason)
        del self.units[uid]
        if u.parent is not None and u.parent in self.units:
            self.units[u.parent].variants.discard(uid)
        for key in u.keys:
            self.components[key].refs.discard(uid)
        self.events[f"revoke_{reason}"] += 1
        self._changed(u.keys)
        self.on_event("revoke", u)
        return True

    def reset(self) -> None:
        self.protected.clear()
        self.registration_blocked.clear()
        for uid in tuple(self.units):
            if uid in self.units:
                self.revoke(uid, self.units[uid].gen, "reset")
        self.components.clear()
        self.reset_epoch += 1
        self.on_reset()

    def cpu_status(self, key: OffloadKey) -> str:
        c = self.components.get(key)
        if c is None or not c.refs:
            return "unknown"
        # B describes only STATE/DRAFT. Until a complete MAIN inventory has
        # registered a variant, its CPU serviceability is unknown too.
        variants = [self.units[uid] for uid in c.refs if self.units[uid].parent]
        if not variants:
            return "unknown"
        if any(u.complete and self.recoverable(u.uid, "cpu") for u in variants):
            return "potentially_servable"
        return "missing_dependency"

    def snapshot(self) -> dict[str, int]:
        counts: Counter[str] = Counter(
            {
                f"{level}_{tag}": 0
                for level in ("boundary", "variant")
                for tag in (
                    "complete",
                    "building",
                    "interior",
                    "leaf",
                    "cpu",
                    "ssd",
                    "union",
                )
            }
        )
        for u in self.units.values():
            name = "variant" if u.parent is not None else "boundary"
            counts[f"{name}_{'complete' if u.complete else 'building'}"] += 1
            counts[f"{name}_{'interior' if self.interior(u) else 'leaf'}"] += 1
            for tier in ("cpu", "ssd", "union"):
                counts[f"{name}_{tier}"] += int(
                    u.complete and self.recoverable(u.uid, tier)
                )
        counts["best_effort_variants"] = sum(
            u.parent is not None for u in self.units.values()
        )
        counts["guaranteed_bytes"] = 0
        counts["gpu_residency_unknown"] = 1
        counts["components"] = len(self.components)
        counts["shared_components"] = sum(
            len(c.refs) > 1 for c in self.components.values()
        )
        counts["component_refs"] = sum(len(c.refs) for c in self.components.values())
        counts["propagation_work_max"] = self.propagation_work_max
        return dict(counts)
