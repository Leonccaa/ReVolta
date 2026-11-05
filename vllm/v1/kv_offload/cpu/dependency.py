# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative CPU residency analysis without probing or touching a backend.

GPU prefix residency is deliberately unknown. A missing CPU ancestor therefore
does NOT make its descendants dead: lookup can start after a GPU prefix hit.
We retain a superset of every possible hit for every such starting position.
Secondary tiers and unmodelled group geometries disable dependency removal.
"""

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import partial

from vllm.v1.kv_offload.base import OffloadKey, get_offload_group_idx
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

KINDS = ("target", "draft", "state")
STATUSES = (
    "potentially_servable",
    "chain_broken",
    "missing_dependency",
    "no_state",
    "outside_window",
    "unknown",
)


@dataclass(frozen=True)
class DependencyGroup:
    group_idx: int
    block_size: int
    kind: str
    window: int | None


@dataclass(frozen=True)
class BlockDependency:
    # The primary full-attention key at this boundary and its predecessor.
    anchor: OffloadKey | None
    parent: OffloadKey | None
    targets: tuple[OffloadKey, ...] = ()
    states: tuple[OffloadKey, ...] = ()
    # Full windows are retained, even if a GPU hit could supply their heads.
    windows: tuple[OffloadKey, ...] = ()
    # Only these draft suffix blocks are mandatory for EVERY possible lookup.
    draft_tails: tuple[OffloadKey, ...] = ()
    junction: bool = False


class CPUBlockDependencyIndex:
    """One constant-size record per resident block; no historical tombstones.

    Parent/dependency keys may name missing slots, but never own another
    record. All cache removal paths synchronously drop the corresponding row.
    Request hashes are used only while annotating accepted store offers.
    """

    def __init__(
        self,
        groups: Sequence[DependencyGroup],
        managers: Sequence[CPUOffloadingManager],
        supported: bool,
        eagle_lookahead: bool = False,
    ):
        self.groups = {g.group_idx: g for g in groups}
        self.full = tuple(g for g in groups if g.kind == "target" and g.window is None)
        self.states = tuple(g for g in groups if g.kind == "state")
        self.drafts = tuple(g for g in groups if g.kind == "draft")
        self.supported = supported and bool(self.full and self.states)
        self.managers = tuple(managers)
        self.eagle_lookahead = eagle_lookahead
        self.records: dict[OffloadKey, BlockDependency | None] = {}
        self.owners: dict[OffloadKey, CPUOffloadingManager] = {}
        for manager in managers:
            manager.cache_observer = partial(self._observe, manager)
            for key in manager.resident_keys():
                self._observe(manager, key, True)

    def _observe(self, manager: CPUOffloadingManager, key: OffloadKey, added: bool):
        if added:
            self.records[key] = None
            self.owners[key] = manager
        else:
            self.records.pop(key, None)
            self.owners.pop(key, None)

    def annotate(
        self,
        offered: Collection[OffloadKey],
        group_keys: Sequence[Sequence[OffloadKey]],
        junction: int,
    ) -> None:
        if not self.supported:
            return
        offered_set = set(offered) & self.records.keys()
        primary = self.full[0]

        def at(g: DependencyGroup, end: int) -> OffloadKey | None:
            idx = end // g.block_size - 1
            keys = group_keys[g.group_idx]
            return keys[idx] if 0 <= idx < len(keys) else None

        for group_idx in {get_offload_group_idx(k) for k in offered_set}:
            g = self.groups[group_idx]
            for i, key in enumerate(group_keys[group_idx]):
                if key not in offered_set:
                    continue
                end = (i + 1) * g.block_size
                if g.kind == "draft":
                    self.records[key] = BlockDependency(None, None)
                    continue
                targets = tuple(at(f, end) for f in self.full)
                states = tuple(at(s, end) for s in self.states)
                if any(k is None for k in (*targets, *states)):
                    continue  # Incomplete hashes: unknown, never infer absence.
                windows: list[OffloadKey] = []
                tails: list[OffloadKey] = []
                for draft in self.drafts:
                    stop = end // draft.block_size
                    keys = group_keys[draft.group_idx]
                    assert draft.window is not None
                    if stop > len(keys):
                        break
                    windows.extend(keys[max(0, stop - draft.window) : stop])
                    # A lookup returning this state must load at least one
                    # full state page beyond its GPU prefix. Keep the rest of
                    # the window conservatively, but its last block is required.
                    tails.extend(keys[max(0, stop - 1) : stop])
                else:
                    old = self.records[key]
                    self.records[key] = BlockDependency(
                        anchor=at(primary, end),
                        parent=at(primary, end - primary.block_size),
                        targets=tuple(k for k in targets if k is not None),
                        states=tuple(k for k in states if k is not None),
                        windows=tuple(windows),
                        draft_tails=tuple(tails),
                        junction=(bool(junction) and end == junction)
                        or bool(old and old.junction),
                    )

    def classify(self, protected: Collection[OffloadKey] = ()) -> dict[OffloadKey, str]:
        if not self.supported:
            return dict.fromkeys(self.records, "unknown")
        resident = self.records.keys()
        # Missing annotations might hide a boundary or an ancestor relation.
        # Such a boundary can depend on any earlier target/draft row.
        if any(record is None for record in self.records.values()):
            return dict.fromkeys(resident, "unknown")
        result = dict.fromkeys(resident, "potentially_servable")
        needed_targets: set[OffloadKey] = set()
        needed_draft: set[OffloadKey] = set()

        def prefix(anchor: OffloadKey | None) -> None:
            while anchor is not None and anchor in resident:
                if anchor in needed_targets:
                    break
                record = self.records[anchor]
                assert record is not None
                needed_targets.update(record.targets)
                anchor = record.parent

        keep = set(protected) & resident
        for key, owner in self.owners.items():
            block = owner._policy.get(key)
            record = self.records[key]
            assert record is not None and block is not None
            if block.ref_cnt != 0 or record.junction:
                keep.add(key)
        pending = list(keep)
        while pending:
            key = pending.pop()
            record = self.records[key]
            assert record is not None
            extra = set(record.targets + record.states + record.windows) & resident
            pending.extend(extra - keep)
            keep.update(extra)
            prefix(record.anchor)

        viable_anchors: set[OffloadKey] = set()
        for key, record in self.records.items():
            assert record is not None
            if self.groups[get_offload_group_idx(key)].kind != "state":
                continue
            if not all(k in resident for k in record.targets):
                result[key] = "chain_broken"
            elif not all(k in resident for k in record.states + record.draft_tails):
                result[key] = "missing_dependency"
            else:
                prefix(record.anchor)
                needed_draft.update(record.windows)
                if record.anchor is not None:
                    viable_anchors.add(record.anchor)

        if self.eagle_lookahead:
            # _lookup first probes a full-attention page beyond the restored
            # state, then shifts back one page. That probe page is required
            # even though it is not returned in the final load specification.
            for record in self.records.values():
                assert record is not None
                if record.parent in viable_anchors:
                    needed_targets.update(record.targets)

        for key in resident:
            kind = self.groups[get_offload_group_idx(key)].kind
            if key in keep:
                result[key] = "unknown"  # Protected, including a possible future store.
            elif kind == "target" and key not in needed_targets:
                result[key] = "no_state"
            elif kind == "draft" and key not in needed_draft:
                result[key] = "outside_window"
        return result

    def evict(self, protected: Collection[OffloadKey]) -> list[OffloadKey]:
        removed: list[OffloadKey] = []
        while True:
            dead = [
                key
                for key, status in self.classify(protected).items()
                if status not in ("potentially_servable", "unknown")
            ]
            batch = []
            for key in dead:
                owner = self.owners.get(key)
                if owner is not None and isinstance(owner._policy, LRUCachePolicy):
                    batch.extend(owner.remove_dependency_blocks([key]))
            if not batch:
                return removed
            removed.extend(batch)

    def totals(self, protected: Collection[OffloadKey]) -> dict[tuple[str, str], int]:
        counts = Counter(
            (self.groups[get_offload_group_idx(key)].kind, status)
            for key, status in self.classify(protected).items()
        )
        return {
            (kind, status): counts[kind, status]
            for kind in KINDS
            for status in STATUSES
        }
