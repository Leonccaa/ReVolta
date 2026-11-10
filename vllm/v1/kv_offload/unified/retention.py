# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content-prefix checkpoint retention, owned by the scheduler thread."""

from dataclasses import dataclass, field
from typing import Any

from vllm.v1.kv_offload.base import OffloadKey, get_offload_block_hash
from vllm.v1.kv_offload.host.page_table import PageKind
from vllm.v1.kv_offload.unified.units import Unit, UnitId


@dataclass(eq=False)
class Prefix:
    children: dict[OffloadKey, "Prefix"] = field(default_factory=dict)
    boundary: UnitId | None = None
    variants: set[UnitId] = field(default_factory=set)


class LatestCheckpoints:
    def __init__(self, manager: Any, keep_ssd: bool):
        self.manager = manager
        self.units, self.table = manager.units, manager.table
        self.keep_ssd = keep_ssd
        self.root = Prefix()
        self.nodes: dict[UnitId, Prefix] = {}
        self.variants: dict[UnitId, Unit] = {}
        self.branches: set[UnitId] = set()
        self.latest: set[UnitId] = set()
        self.resends: set[UnitId] = set()
        self.pending: dict[UnitId, UnitId] = {}
        self.cursor = 0
        self.dirty = False
        self.epoch = self.units.reset_epoch
        for name in (
            "superseded_revoked",
            "superseded_freed_bytes",
            "superseded_ssd_deleted_bytes",
        ):
            manager.events[name] = 0

    def event(self, event: str, unit: Unit) -> None:
        if self.epoch != self.units.reset_epoch:
            self.reset()
        if event in ("register", "observe"):
            self.observe(unit)
        if event == "revoke" and unit.parent is None:
            store = self.store
            if store is None or not any(
                v.boundary == unit.uid.boundary and v.reset == unit.uid.reset
                for v in store.manifests
            ):
                self.branches.discard(unit.uid)
        self.dirty = True

    def observe(self, unit: Unit) -> None:
        if unit.parent is None or not unit.complete:
            return
        if unit.uid in self.variants:
            self.variants[unit.uid] = unit
            if unit.retention.get("branch") or unit.retention.get("junction"):
                self.branches.add(unit.parent)
            self.dirty = True
            return
        main = [k for k in unit.keys if self.table.kind_for_key(k) == PageKind.MAIN]
        if not main or not unit.boundary_position or not unit.ahead_position:
            return
        span, remainder = divmod(unit.ahead_position, len(main))
        if remainder or not span or unit.boundary_position % span:
            return
        depth = unit.boundary_position // span
        if not 0 < depth <= len(main):
            return
        if get_offload_block_hash(main[depth - 1]) != unit.uid.boundary:
            return
        node = self.root
        for i, key in enumerate(main, 1):
            if key not in node.children:
                node.children[key] = Prefix()
            node = node.children[key]
            if i == depth:
                node.boundary = unit.parent
                self.nodes[unit.parent] = node
        node.variants.add(unit.uid)
        self.variants[unit.uid] = unit
        if unit.retention.get("branch") or unit.retention.get("junction"):
            self.branches.add(unit.parent)
        self.dirty = True

    def reset(self) -> None:
        self.root = Prefix()
        self.nodes.clear()
        self.variants.clear()
        self.branches.clear()
        self.latest.clear()
        self.resends.clear()
        self.pending.clear()
        self.cursor = 0
        self.epoch = self.units.reset_epoch
        self.dirty = False

    def refresh(self) -> None:
        if self.epoch != self.units.reset_epoch:
            self.reset()
        if not self.dirty:
            return
        store = self.store
        # Keep disk obligations after normal RAM value eviction. A deleted
        # graph incarnation is never substituted for a committed manifest.
        for uid in tuple(self.variants):
            if uid in self.units.units:
                self.variants[uid] = self.units.units[uid]
            elif store is None or uid not in store.manifests:
                del self.variants[uid]
        live_b = {u.parent for u in self.variants.values()}
        self.branches.intersection_update(live_b)
        for uid in tuple(self.nodes):
            if uid not in live_b:
                self.nodes[uid].boundary = None
                del self.nodes[uid]
        # Prune unused suffixes. Shared MAIN prefixes occur once in this tree.
        order = [self.root]
        for node in order:
            order.extend(node.children.values())
        tips: dict[Prefix, Unit | None] = {}
        counts: dict[Prefix, int] = {}
        self.pending.clear()
        self.latest.clear()
        self.resends.clear()
        live_variants = set(self.variants)
        for node in reversed(order):
            node.variants.intersection_update(live_variants)
            witnesses = [self.variants[uid] for uid in node.variants]
            descendants = 0
            for key, child in tuple(node.children.items()):
                descendants += counts[child]
                tip = tips[child]
                if tip is not None:
                    witnesses.append(tip)
                if tip is None and not counts[child]:
                    del node.children[key]
            tip = max(
                witnesses,
                key=lambda u: (
                    u.boundary_position,
                    u.retention.get("request_length", 0),
                ),
                default=None,
            )
            boundary = node.boundary
            counts[node] = descendants + int(boundary is not None)
            tips[node] = tip
        # Branch evidence uses only the live, pruned paths. A revoked suffix
        # must not latch a branch when a new generation reuses its prefix.
        nearest: dict[Prefix, UnitId | None] = {self.root: None}
        chain_tips: dict[Prefix, Unit | None] = {self.root: None}
        predecessors: dict[UnitId, UnitId | None] = {}
        live_order = [self.root]
        for node in live_order:
            if node.boundary is not None:
                predecessors[node.boundary] = nearest[node]
            ancestor = node.boundary or nearest[node]
            if len(node.children) > 1 and ancestor is not None:
                self.branches.add(ancestor)
            witnesses = [self.variants[v] for v in node.variants]
            prior = chain_tips[node]
            if prior is not None:
                witnesses.append(prior)
            tip = max(
                witnesses,
                key=lambda u: (
                    u.boundary_position,
                    u.retention.get("request_length", 0),
                ),
                default=None,
            )
            for child in node.children.values():
                nearest[child] = ancestor
                chain_tips[child] = tip
            live_order.extend(node.children.values())
            if not node.children and tip is not None:
                assert tip.parent is not None
                self.latest.add(tip.parent)
                # Every leaf chain contributes its own request's resend. A
                # longer sibling must not replace this chain's exemption.
                boundary = tip.parent
                while boundary is not None:
                    if tip.retention.get("resend") == boundary.boundary.hex():
                        self.resends.add(boundary)
                        break
                    boundary = predecessors[boundary]
                # Branch/junction annotations do not upgrade legacy request
                # metadata. Preserve its immediate predecessor until a new
                # complete request checkpoint supplies the exact boundary.
                if not (
                    isinstance(tip.retention.get("request_length"), int)
                    and tip.retention["request_length"] > 0
                    and "resend" in tip.retention
                ):
                    predecessor = predecessors[tip.parent]
                    if predecessor is not None:
                        self.resends.add(predecessor)
        retained = self.latest | self.resends | self.branches
        for node in live_order:
            boundary, tip = node.boundary, tips[node]
            if boundary is None or tip is None:
                continue
            if boundary not in retained:
                self.pending[boundary] = tip.uid
        for uid in self.branches:
            b = self.units.units.get(uid)
            if b is not None:
                for v in b.variants:
                    variant = self.units.units[v]
                    if not variant.retention.get("branch"):
                        variant.retention["branch"] = True
                        self.units.on_event("retention", variant)
        self.dirty = False

    @property
    def store(self) -> Any:
        ssd = self.manager.ssd
        return None if ssd is None else ssd.unit_store

    def blocked_key(self, key: OffloadKey) -> bool:
        c = self.units.components.get(key)
        if (
            self.table.kind_for_key(key) == PageKind.STATE
            and UnitId(get_offload_block_hash(key), self.units.reset_epoch)
            in self.units.registration_blocked
        ):
            return True
        return bool(
            c is not None
            and c.kind != PageKind.MAIN
            and c.refs & self.units.registration_blocked
        )

    def quiescent(self, uid: UnitId) -> bool:
        b = self.units.units.get(uid)
        store = self.store
        if store is not None:
            variants = (
                b.variants
                if b is not None
                else (v for v, unit in self.variants.items() if unit.parent == uid)
            )
            if any(v in store.writes or v in store.prefetches for v in variants):
                return False
        if any(build.uid == uid for build in self.manager.builds.values()):
            return False
        keys = (
            b.keys
            if b is not None
            else next((u.keys for u in self.variants.values() if u.parent == uid), ())
        )
        return all(
            self.table.reclaimable(self.table.get(c.cpu))
            for k in keys
            if (c := self.units.components.get(k)) is not None
            and c.kind != PageKind.MAIN
            and c.cpu is not None
        )

    def free_components(self, keys: tuple[OffloadKey, ...]) -> None:
        for key in keys:
            c = self.units.components.get(key)
            if c is None or c.kind == PageKind.MAIN or c.cpu is None:
                continue
            if c.refs and not (
                self.keep_ssd
                and all(self.units.cold_recoverable(uid) for uid in c.refs)
            ):
                continue
            handle = c.cpu
            ssd = self.manager.ssd
            if ssd is not None:
                ssd._queue.drop([handle], "superseded")
                ssd._deferred.pop(handle, None)
            if self.table.discard_cached(handle):
                self.manager.events["superseded_freed_bytes"] += getattr(
                    self.manager.spec, "unified_frame_bytes", 0
                )

    def can_commit(self, uid: UnitId, witness: Unit) -> bool:
        self.refresh()
        return (
            self.pending.get(uid) == witness.uid
            and self.units.units.get(witness.uid) is witness
            and self.units.recoverable(witness.uid, "union")
        )

    def commit(self, uid: UnitId, gen: int, keys: tuple[OffloadKey, ...]) -> None:
        if self.units.revoke(uid, gen, "superseded"):
            self.manager.events["superseded_revoked"] += 1
        self.free_components(keys)
        self.dirty = True

    def step(self) -> None:
        self.refresh()
        candidates = list(self.pending.items())
        start = self.cursor % max(1, len(candidates))
        self.cursor = start + 8
        for uid, witness in (candidates[start:] + candidates[:start])[:8]:
            unit = self.units.units.get(witness)
            if (
                unit is None
                and self.store is not None
                and witness in self.store.manifests
            ):
                unit = self.store._restore(
                    self.store.manifests[witness], self.store.manifest_paths[witness]
                )
            if unit is None or not self.units.recoverable(witness, "union"):
                continue
            if not self.quiescent(uid):
                continue
            b = self.units.units.get(uid)
            if self.keep_ssd:
                if b is not None and self.units.cold_recoverable(uid):
                    self.free_components(b.keys)
                    if all(self.units.components[k].cpu is None for k in b.keys):
                        self.pending.pop(uid, None)
            elif self.store is not None:
                self.store.supersede(uid, unit)
            elif b is not None:
                self.commit(uid, b.gen, b.keys)

    def snapshot(self) -> dict[str, int]:
        self.refresh()
        return {
            "branch_points": len(self.branches),
            "chain_latest_units": len(self.latest),
            "resend_points": len(self.resends),
            "superseded_pending": len(self.pending),
            "superseded_delete_transactions": len(self.units.registration_blocked),
        }
