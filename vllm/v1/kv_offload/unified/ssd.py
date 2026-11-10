# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned SSD transactions and the single unified eviction authority.

The page controller remains the only physical IO queue. Transaction leases and
file pins cover the whole inventory, including pages in later IO batches.
"""

import functools
import hashlib
import json
import os
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadKey,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.host.page_table import PageHandle, PageKind, PageState
from vllm.v1.kv_offload.tiering.fs.revolta import read_page_identity
from vllm.v1.kv_offload.unified.credit import Obligation, PersistenceCredit
from vllm.v1.kv_offload.unified.units import Component, Unit, UnitId, VariantOffer

logger = init_logger(__name__)
SCHEMA = 1


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def atomic_json(path: str, value: Any, chunk_bytes: int | None = None) -> int:
    payload = encode(value)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb", buffering=0) as stream:
        chunk_bytes = len(payload) if chunk_bytes is None else chunk_bytes
        for offset in range(0, len(payload), max(1, chunk_bytes)):
            stream.write(payload[offset : offset + chunk_bytes])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(os.path.dirname(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return len(payload)


@dataclass
class WriteTxn:
    uid: UnitId
    gen: int
    keys: tuple[OffloadKey, ...]
    missing: set[OffloadKey]
    handles: dict[OffloadKey, PageHandle]
    paths: dict[OffloadKey, str]
    manifest_path: str
    reservation: dict[str, int]
    pins: ExitStack
    owner: int
    started: float
    active: dict[OffloadKey, int] = field(default_factory=dict)
    failed: bool = False
    remaining: set[OffloadKey] = field(default_factory=set)
    manifest_job: int | None = None
    manifest: dict | None = None


@dataclass
class PrefetchTxn:
    uid: UnitId
    gen: int
    keys: tuple[OffloadKey, ...]
    handles: dict[OffloadKey, PageHandle]
    pending: list[OffloadKey]
    pins: ExitStack
    owner: int
    started: float
    waiters: set[str] = field(default_factory=set)
    active: dict[OffloadKey, int] = field(default_factory=dict)
    failed: bool = False
    complete: bool = False
    sources: frozenset[OffloadKey] = frozenset()
    waiter_deadlines: dict[str, float] = field(default_factory=dict)


class UnitSsdStore:
    def __init__(self, manager: Any, ssd: Any):
        self.manager, self.ssd = manager, ssd
        self.table, self.units = manager.table, manager.units
        self.cache = ssd._tier._cache
        self.cache.external_eviction = True
        self.manifests: dict[UnitId, dict] = {}
        self.manifest_paths: dict[UnitId, str] = {}
        self.manifest_refs: dict[OffloadKey, set[UnitId]] = {}
        self.files: dict[OffloadKey, dict] = {}
        self.writes: dict[UnitId, WriteTxn] = {}
        self.max_write_transactions = max(2, min(ssd._max_pending, 2 * ssd._window))
        self.manifest_jobs: dict[int, WriteTxn] = {}
        self.prefetches: dict[UnitId, PrefetchTxn] = {}
        self.requests: dict[str, set[UnitId]] = {}
        self.single_claims: dict[int, str] = {}
        self.retired_manifests: set[str] = set()
        self.due: dict[UnitId, float] = {}
        self.filtered_variants: set[UnitId] = set()
        self._due_heap: list[tuple[float, int, UnitId]] = []
        self._due_sequence = 0
        self._manifest_sizes: dict[UnitId, tuple[int, int]] = {}
        self._manifest_names: dict[UnitId, tuple[int, str]] = {}
        self._obligation_cache: dict[UnitId, tuple[int, dict[Obligation, int]]] = {}
        self.events: Counter[str] = Counter()
        self.dirty_since: dict[PageHandle, float] = {}
        self.uncommitted_bytes: dict[str, int] = {}
        self._over_budget_since: float | None = None
        self._over_budget_s = 0.0
        self.read_budget_bytes = ssd.page_bytes * ssd.options["prefetch_max_inflight"]
        self.request_read_bytes = (
            ssd.page_bytes * ssd.options["prefetch_max_per_request"]
        )
        self._owner = -(10**12)
        self._advancing = False
        self._rebuilding = True
        self.flushing = False
        identities = [
            ssd._tier._identity_digest.hex(),
            ssd._component_tier._identity_digest.hex(),
        ]
        self.namespace = hashlib.sha256(encode(identities)).hexdigest()
        self.directory = os.path.join(self.cache.root, "unified-" + self.namespace[:32])
        self.control_path = os.path.join(self.directory, "epoch.json")
        os.makedirs(self.directory, exist_ok=True)
        self.units.manifest_ok = self.manifest_ok
        self.units.on_event = self.unit_event
        self.units.on_reset = self.reset
        opts = ssd.options
        window = min(
            opts["shutdown_flush_timeout_s"],
            max(0, getattr(ssd._spec.vllm_config, "shutdown_timeout", 30) - 5),
        )
        self.credit = PersistenceCredit(
            self.units,
            self.obligations,
            window_s=max(0, window - opts["shutdown_io_grace_s"]),
            io_bytes=ssd.page_bytes * opts["max_pending_blocks"],
            initial_rate=opts["write_bytes_per_second"],
            floor_rate=min(16 << 20, opts["write_bytes_per_second"]),
            clock=ssd.clock,
            available=lambda: ssd.enabled
            and self.cache.max_size_bytes >= ssd.page_bytes,
            capacity_bytes=self.cache.max_size_bytes,
            eligible=lambda unit: len(unit.keys) + 1 <= self.cache.max_files
            and len(unit.keys) * ssd.page_bytes + self.manifest_bytes(unit)
            <= self.cache.max_size_bytes,
        )
        self.rebuild()
        self._rebuilding = False
        self.credit.reconcile()
        logger.info(
            "Unified SSD credit D_dyn=%d W=%.0f B/s completion=%.3fs",
            self.credit.d_dyn,
            self.credit.rate,
            self.credit.bytes / self.credit.rate,
        )

    def owner(self) -> int:
        self._owner -= 1
        return self._owner

    def manifest_ok(self, unit: Unit) -> bool:
        if unit.parent is None:
            return any(self.manifest_ok(self.units.units[uid]) for uid in unit.variants)
        record = self.manifests.get(unit.uid)
        return record is not None and record["gen"] == unit.gen

    def path(self, key: OffloadKey) -> str:
        record = self.files.get(key)
        if record is not None:
            tier = self.ssd._tier_for_key(key)
            return tier.file_mapper.get_file_name(bytes.fromhex(record["tier_key"]))
        tier = self.ssd._tier_for_key(key)
        return tier.file_mapper.get_file_name(self.ssd._tier_key(key))

    def tier_key(self, key: OffloadKey) -> OffloadKey | None:
        record = self.files.get(key)
        return None if record is None else OffloadKey(bytes.fromhex(record["tier_key"]))

    def manifest_path(self, unit: Unit) -> str:
        assert unit.uid.ahead is not None
        cached = self._manifest_names.get(unit.uid)
        if cached is not None and cached[0] == unit.gen:
            return cached[1]
        name = hashlib.sha256(
            encode(
                [
                    unit.uid.boundary.hex(),
                    unit.uid.ahead.hex(),
                    unit.uid.reset,
                    unit.gen,
                ]
            )
        ).hexdigest()
        path = os.path.join(self.directory, name + ".unit.json")
        self._manifest_names[unit.uid] = unit.gen, path
        return path

    def descriptor(self, key: OffloadKey, *, placeholder: bool = False) -> dict:
        tier = self.ssd._tier_for_key(key)
        tier_key = self.ssd._tier_key(key)
        identity = None if placeholder else read_page_identity(self.path(key))
        if identity is None:
            if not placeholder:
                raise ValueError("Missing component identity")
            identity = (0, tier._identity_digest, bytes(tier_key))
        if identity[1:] != (tier._identity_digest, bytes(tier_key)):
            raise ValueError("Component namespace or key mismatch")
        return {
            "key": bytes(key).hex(),
            "kind": self.table.kind_for_key(key).value,
            "group": get_offload_group_idx(key),
            "format": "revolta-fs-page-v1",
            "length": self.ssd.page_bytes,
            "crc32c": f"{identity[0]:08x}",
            "namespace": identity[1].hex(),
            "tier_key": identity[2].hex(),
        }

    def document(self, unit: Unit, *, placeholder: bool = False) -> dict:
        assert unit.uid.ahead is not None
        main = [k for k in unit.keys if self.table.kind_for_key(k) == PageKind.MAIN]
        states: dict[str, list[str]] = {}
        drafts = []
        for key in unit.keys:
            kind = self.table.kind_for_key(key)
            if kind == PageKind.STATE:
                states.setdefault(str(get_offload_group_idx(key)), []).append(key.hex())
            elif kind == PageKind.DRAFT:
                drafts.append(key.hex())
        return {
            "schema": SCHEMA,
            "namespace": self.namespace,
            "reset_epoch": unit.uid.reset,
            "gen": unit.gen,
            "h_B": unit.uid.boundary.hex(),
            "h_A": unit.uid.ahead.hex(),
            "B": unit.boundary_position,
            "A": unit.ahead_position,
            "main": [k.hex() for k in main],
            "state": states,
            "draft": drafts,
            "draft_range": list(unit.draft_range),
            "components": [
                self.descriptor(k, placeholder=placeholder) for k in unit.keys
            ],
        }

    def manifest_bytes(self, unit: Unit) -> int:
        cached = self._manifest_sizes.get(unit.uid)
        if cached is None or cached[0] != unit.gen:
            cached = unit.gen, len(encode(self.document(unit, placeholder=True)))
            self._manifest_sizes[unit.uid] = cached
        return cached[1]

    def schedule(self, unit: Unit, when: float) -> None:
        self.filtered_variants.discard(unit.uid)
        self.due[unit.uid] = when
        self._due_sequence += 1
        heappush(self._due_heap, (when, self._due_sequence, unit.uid))

    def install_manifest(self, unit: Unit, record: dict, path: str) -> None:
        self._obligation_cache.pop(unit.uid, None)
        previous = self.manifest_paths.get(unit.uid)
        if previous is not None and previous != path:
            self.retired_manifests.add(previous)
        self.manifests[unit.uid], self.manifest_paths[unit.uid] = record, path
        for key in unit.keys:
            self.manifest_refs.setdefault(key, set()).add(unit.uid)
        self.units.evidence_changed(unit.uid)

    def remove_manifest(self, uid: UnitId) -> dict:
        self._obligation_cache.pop(uid, None)
        record = self.manifests.pop(uid)
        self.retired_manifests.add(self.manifest_paths.pop(uid))
        for component in record["components"]:
            key = OffloadKey(bytes.fromhex(component["key"]))
            refs = self.manifest_refs[key]
            refs.discard(uid)
            if not refs:
                del self.manifest_refs[key]
        self.units.evidence_changed(uid)
        return record

    def obligations(self, unit: Unit) -> dict[Obligation, int]:
        if unit.parent is None:
            return {}
        cached = self._obligation_cache.get(unit.uid)
        if cached is not None and cached[0] == unit.gen:
            return cached[1]
        # Keep the entire transaction obligation charged until manifest commit.
        txn = self.writes.get(unit.uid)
        missing = (
            txn.missing
            if txn is not None and txn.gen == unit.gen
            else self.units.persistence_obligations(unit.uid)[0]
        )
        result: dict[Obligation, int] = {
            ("component", k): self.ssd.page_bytes for k in missing
        }
        if not self.manifest_ok(unit):
            result["manifest", (unit.uid, unit.gen)] = self.manifest_bytes(unit)
        self._obligation_cache[unit.uid] = unit.gen, result
        return result

    def invalidate_key(self, key: OffloadKey) -> None:
        component = self.units.components.get(key)
        if component is None:
            return
        for uid in component.refs:
            txn = self.writes.get(uid)
            if txn is None or txn.gen != self.units.units[uid].gen:
                self._obligation_cache.pop(uid, None)

    def unit_event(self, event: str, unit: Unit) -> None:
        self._obligation_cache.pop(unit.uid, None)
        if self._rebuilding:
            return
        if event == "register":
            self.schedule(unit, self.ssd.clock() + self.ssd.options["write_delay_s"])
        elif event == "revoke":
            self.filtered_variants.discard(unit.uid)
            self.due.pop(unit.uid, None)
            self._manifest_sizes.pop(unit.uid, None)
            self._manifest_names.pop(unit.uid, None)
            txn = self.writes.get(unit.uid)
            if txn is not None:
                txn.failed = True
            prefetch = self.prefetches.get(unit.uid)
            if prefetch is not None:
                prefetch.failed = True
            # A CPU value revocation need not discard an independently committed
            # disk variant. Its graph can be rebuilt by a later cold lookup.
        self.credit.event(event, unit)

    def rebuild(self) -> None:
        started = time.monotonic()
        if os.path.exists(self.control_path):
            control = json.loads(Path(self.control_path).read_text())
            self.units.reset_epoch = control["reset_epoch"]
            self.ssd._component_salt = bytes.fromhex(control["component_salt"])
            self.table.ssd_epoch = control["ssd_epoch"]
            if self.table.ssd_epoch:
                self.ssd._salts[self.table.ssd_epoch] = bytes.fromhex(
                    control["main_salt"]
                )
        records = []
        for path in Path(self.directory).glob("*.unit.json"):
            try:
                record = json.loads(path.read_text())
                if (
                    record["schema"] != SCHEMA
                    or record["namespace"] != self.namespace
                    or record["reset_epoch"] != self.units.reset_epoch
                ):
                    raise ValueError("Manifest namespace or epoch mismatch")
                if type(record["gen"]) is not int or record["gen"] < 1:
                    raise ValueError("Invalid manifest generation")
                records.append((record["gen"], str(path), record))
            except (ValueError, KeyError, TypeError, OSError):
                self.cache.discard(str(path))
                self.events["bad_manifests"] += 1
        for _, manifest_path, record in sorted(
            records, key=lambda item: item[0], reverse=True
        ):
            try:
                uid = UnitId(
                    bytes.fromhex(record["h_B"]),
                    record["reset_epoch"],
                    bytes.fromhex(record["h_A"]),
                )
                if uid in self.manifests:
                    self.cache.discard(manifest_path)
                else:
                    self._restore(record, manifest_path)
            except (ValueError, KeyError, TypeError, OSError):
                self.cache.discard(manifest_path)
                self.events["bad_manifests"] += 1
        self.events["startup_rebuild_us"] = int((time.monotonic() - started) * 1e6)

    def _restore(self, record: dict, path: str) -> Unit:
        if (
            not isinstance(record.get("state"), dict)
            or not isinstance(record.get("components"), list)
            or type(record.get("B")) is not int
            or type(record.get("A")) is not int
            or not 0 <= record["B"] <= record["A"]
            or not isinstance(record.get("draft_range"), list)
            or len(record["draft_range"]) != 2
            or any(type(n) is not int for n in record["draft_range"])
            or not 0
            <= record["draft_range"][0]
            <= record["draft_range"][1]
            <= record["B"]
        ):
            raise ValueError("Invalid manifest schema")
        descriptors = {
            OffloadKey(bytes.fromhex(c["key"])): c for c in record["components"]
        }
        main = tuple(OffloadKey(bytes.fromhex(k)) for k in record["main"])
        state = tuple(
            OffloadKey(bytes.fromhex(k))
            for group in record["state"].values()
            for k in group
        )
        drafts = tuple(OffloadKey(bytes.fromhex(k)) for k in record["draft"])
        expected_states = {
            g
            for g, kind in self.manager.kind_by_group.items()
            if kind == PageKind.STATE
        }
        if {get_offload_group_idx(k) for k in state} != expected_states:
            raise ValueError("Manifest STATE inventory is incomplete")
        if not state or not main or set(descriptors) != set((*state, *drafts, *main)):
            raise ValueError("Manifest inventory mismatch")
        h_b, h_a = bytes.fromhex(record["h_B"]), bytes.fromhex(record["h_A"])
        if (
            len(descriptors) != len(record["components"])
            or any(
                self.table.kind_for_key(k) != PageKind.STATE
                or get_offload_block_hash(k) != h_b
                for k in state
            )
            or any(self.table.kind_for_key(k) != PageKind.DRAFT for k in drafts)
            or any(self.table.kind_for_key(k) != PageKind.MAIN for k in main)
            or get_offload_block_hash(main[-1]) != h_a
            or h_a != h_b
            and (len(main) < 2 or get_offload_block_hash(main[-2]) != h_b)
        ):
            raise ValueError("Invalid manifest branch inventory")
        paths = {}
        for key, desc in descriptors.items():
            if (
                desc["kind"] != self.table.kind_for_key(key).value
                or desc["group"] != get_offload_group_idx(key)
                or desc["length"] != self.ssd.page_bytes
                or desc["format"] != "revolta-fs-page-v1"
            ):
                raise ValueError("Manifest component format mismatch")
            tier = self.ssd._tier_for_key(key)
            if desc["namespace"] != tier._identity_digest.hex():
                raise ValueError("Manifest component namespace mismatch")
            component_path = tier.file_mapper.get_file_name(
                bytes.fromhex(desc["tier_key"])
            )
            if os.stat(component_path).st_size != desc["length"]:
                raise ValueError("Manifest component missing or truncated")
            paths[key] = component_path
        for key, desc in descriptors.items():
            self.files[key] = desc
            component = self.units.components.setdefault(
                key, Component(self.table.kind_for_key(key))
            )
            before = component.cpu is not None, component.persisted is not None
            if component.persisted is None:
                component.persisted = PageHandle(0, -len(self.files))
                self.invalidate_key(key)
                self.units._changed(self.units._presence_changed(key, *before))
            self.ssd._file_evidence[paths[key]] = key, component.persisted
        b = self.units.begin(bytes.fromhex(record["h_B"]), (*state, *drafts))
        if not self.units.finish(b.uid, b.gen):
            raise ValueError("Incomplete manifest boundary")
        u = self.units.register(
            VariantOffer(
                b.uid,
                b.gen,
                bytes.fromhex(record["h_A"]),
                main,
                record["B"],
                record["A"],
                tuple(record["draft_range"]),
            )
        )
        if u is None:
            raise ValueError("Manifest branch identity mismatch")
        # Restored generations are authoritative for this committed inventory.
        u.gen = record["gen"]
        self.units._generation = max(self.units._generation, u.gen)
        self.install_manifest(u, record, path)
        if not self._rebuilding:
            self.credit.reconcile()
        return u

    def reserve(self, paths: dict[str, int], protected: set[str] | None = None) -> bool:
        protected = protected or set()
        if self.cache.reserve_paths(paths):
            return True
        # Orphans first. Pending claims and IO pins are checked under cache lock.
        referenced = {
            self.path(OffloadKey(bytes.fromhex(c["key"])))
            for record in self.manifests.values()
            for c in record["components"]
        }
        for path in tuple(self.cache._pages):
            if (
                path not in referenced
                and path not in self.manifest_paths.values()
                and path not in protected
            ):
                # Components without manifests get one delayed-write grace window.
                if (
                    path.startswith(self.ssd._component_tier.file_mapper.base_path)
                    and time.time() - os.stat(path).st_mtime
                    < self.ssd.options["write_delay_s"]
                ):
                    continue
                if self.discard_file(path) and self.cache.reserve_paths(paths):
                    return True
        for uid in sorted(
            self.manifests,
            key=lambda uid: self.units.units[uid].last_use
            if uid in self.units.units
            else -1,
        ):
            if uid in self.writes or uid in self.prefetches:
                continue
            self.drop(uid)
            if self.cache.reserve_paths(paths):
                return True
        self.events["quota_rejected"] += 1
        return False

    def discard_file(self, path: str) -> bool:
        if not self.cache.discard(path):
            return False
        self.uncommitted_bytes.pop(path, None)
        # Deletion and CPU/union evidence withdrawal are one scheduler event.
        self.ssd._poll_losses()
        return True

    def file_lost(self, key: OffloadKey) -> None:
        self.files.pop(key, None)
        affected = set(self.manifest_refs.get(key, ()))
        self.credit.events["redirty"] += len(affected & self.credit.guaranteed)
        for uid in affected:
            self.remove_manifest(uid)
            if uid in self.units.units:
                self.schedule(self.units.units[uid], self.ssd.clock())
        self.credit.reconcile(promote=False, reason="redirty")

    def drop(self, uid: UnitId) -> bool:
        if uid in self.writes or uid in self.prefetches:
            return False
        if uid not in self.manifests:
            return False
        path = self.manifest_paths[uid]
        record = self.remove_manifest(uid)
        self.cache.discard(path)
        remaining = self.manifest_refs.keys()
        candidates = {
            OffloadKey(bytes.fromhex(c["key"])) for c in record["components"]
        } - remaining
        removable = {
            k
            for k in candidates
            if self.path(k) not in self.cache._pins
            and self.path(k) not in self.cache._claims
        }
        affected = {uid} | {
            ref
            for k in removable
            if k in self.units.components
            for ref in self.units.components[k].refs
        }
        for ref in affected:
            self._obligation_cache.pop(ref, None)
        # Withdraw all logical evidence before exposing physical deletion.
        for key in removable:
            component = self.units.components.get(key)
            if component is not None and component.persisted is not None:
                before = component.cpu is not None, True
                incarnation = component.persisted
                component.persisted = None
                self.units._changed(self.units._presence_changed(key, *before))
                page = self.ssd._page(incarnation)
                if page is not None:
                    self.table.unpersist(incarnation)
        for ref in affected:
            unit = self.units.units.get(ref)
            if unit is None:
                continue
            if not self.units.recoverable(ref, "union"):
                self.units.revoke(ref, unit.gen, "ssd_value_drop")
            else:
                self.units.evidence_changed(ref)
                self.events["value_drop_union_retained"] += 1
        self.credit.events["redirty"] += len(affected & self.credit.guaranteed)
        self.credit.reconcile(promote=False, reason="redirty")
        for key in removable:
            self.cache.discard(self.path(key))
            self.files.pop(key, None)
        self.events["value_drop"] += 1
        self.events["files_shared_retained"] += (
            len(candidates)
            - len(removable)
            + len(record["components"])
            - len(candidates)
        )
        return True

    def start_write(self, unit: Unit, headroom: int | None = None) -> bool:
        if unit.parent is None or not unit.complete:
            return False
        if self.manifest_ok(unit):
            return True
        if unit.uid in self.writes:
            return self.writes[unit.uid].gen == unit.gen
        if len(self.writes) >= self.max_write_transactions:
            self.events["write_transaction_queue_full"] += 1
            return False
        handles, paths, missing = {}, {}, set()
        for key in unit.keys:
            paths[key] = self.path(key)
            if self.cache.lookup(paths[key]):
                continue
            handle = self.table.lookup(key)
            if handle is None or not self.table.is_ready(handle):
                self.events["write_source_missing"] += 1
                return False
            handles[key] = handle
            missing.add(key)
        cost = sum(
            self.table.reclaimable(self.table.get(h)) for h in set(handles.values())
        )
        if missing and self.credit.q_eff < self.ssd.page_bytes:
            self.events["write_no_io_budget"] += 1
            return False
        if cost > (self.table.net_headroom() if headroom is None else headroom):
            self.events["write_no_headroom"] += 1
            return False
        owner, pins = self.owner(), ExitStack()
        for handle in set(handles.values()):
            self.table.pin(handle, owner)
        for path in paths.values():
            pins.enter_context(self.cache.read(path))
        manifest_path = self.manifest_path(unit)
        reservation = {paths[k]: self.ssd.page_bytes for k in missing}
        reservation[manifest_path] = self.manifest_bytes(unit)
        if not self.reserve(reservation, set(paths.values())):
            pins.close()
            for handle in set(handles.values()):
                self.table.release(handle, owner)
            return False
        txn = WriteTxn(
            unit.uid,
            unit.gen,
            unit.keys,
            missing,
            handles,
            paths,
            manifest_path,
            reservation,
            pins,
            owner,
            self.ssd.clock(),
        )
        self.writes[unit.uid] = txn
        txn.remaining = set(missing)
        self._obligation_cache.pop(unit.uid, None)
        self.credit.reconcile(promote=False)
        self.credit.started()
        self.events["write_transactions_started"] += 1
        self.advance_writes()
        return True

    def prepare_job(self, job: Any) -> bool:
        if job.loading:
            return True
        page = self.table.get(job.handle)
        path = self.path(page.key)
        if path in self.cache._claims:
            # The transaction keeps this reservation and source pin until
            # every page receipt and its manifest have retired.
            return True
        if not self.reserve({path: self.ssd.page_bytes}, {path}):
            return False
        self.single_claims[job.job_id] = path
        return True

    def job_done(self, job: Any, success: bool) -> None:
        claim = self.single_claims.pop(job.job_id, None)
        if claim is not None:
            self.cache.release_paths([claim])
        page = self.ssd._page(job.handle)
        key = None if page is None else page.key
        success = success and page is not None and page.valid
        if success and key is not None and page.valid:
            try:
                self.files[key] = self.descriptor(key)
            except ValueError:
                success = False
        if success and key is not None and not job.loading:
            self.uncommitted_bytes[self.path(key)] = self.ssd.page_bytes
            if self.flushing:
                guaranteed = any(
                    txn.uid in self._flush_guaranteed and key in txn.missing
                    for txn in self.writes.values()
                )
                if not guaranteed:
                    self._flush_best_effort_bytes += self.ssd.page_bytes
        for txn in tuple(self.writes.values()):
            matches = [k for k, j in txn.active.items() if j == job.job_id]
            for k in matches:
                del txn.active[k]
                txn.failed |= not success
                if success:
                    txn.remaining.discard(k)
        for prefetch in tuple(self.prefetches.values()):
            matches = [k for k, j in prefetch.active.items() if j == job.job_id]
            for k in matches:
                del prefetch.active[k]
                prefetch.failed |= not success
                if success:
                    self.events["prefetch_verified_bytes"] += self.ssd.page_bytes
        if not job.loading and key is not None:
            component = self.units.components.get(key)
            # An active unit write keeps its complete obligation until commit.
            # Orphan page IO changes no variant credit at all.
            if component is not None and any(
                uid in self.credit.guaranteed
                and (
                    uid not in self.writes
                    or self.writes[uid].gen != self.units.units[uid].gen
                )
                for uid in component.refs
            ):
                self.credit.reconcile(promote=False)

    def advance_writes(self) -> None:
        if self._advancing:
            return
        self._advancing = True
        try:
            transactions = list(self.writes.values())
            guaranteed_pending = False
            if self.flushing:
                guaranteed_pending = any(
                    uid in self.units.units
                    and not self.manifest_ok(self.units.units[uid])
                    for uid in self._flush_guaranteed
                )
                transactions = sorted(
                    transactions,
                    key=lambda txn: (
                        txn.uid not in self._flush_guaranteed,
                        -self.units.units[txn.uid].last_use
                        if txn.uid in self.units.units
                        else 0,
                    ),
                )
            for txn in transactions:
                unit = self.units.units.get(txn.uid)
                txn.failed |= unit is None or unit.gen != txn.gen
                txn.failed |= self.credit.q_eff < self.ssd.page_bytes
                if (
                    guaranteed_pending
                    and txn.uid not in self._flush_guaranteed
                    and not txn.failed
                ):
                    continue
                if not txn.failed:
                    limit = min(
                        self.ssd._max_pending, self.credit.q_eff // self.ssd.page_bytes
                    )
                    if txn.remaining and self.ssd._store_jobs() >= limit:
                        continue
                    for key in tuple(txn.remaining):
                        if key in txn.active:
                            continue
                        if self.cache.lookup(txn.paths[key]):
                            txn.remaining.remove(key)
                            continue
                        handle = txn.handles[key]
                        existing = self.ssd._page_jobs.get(handle.page_id)
                        if existing is not None:
                            txn.active[key] = existing
                            continue
                        limit = min(
                            self.ssd._max_pending,
                            self.credit.q_eff // self.ssd.page_bytes,
                        )
                        if self.ssd._store_jobs() >= limit:
                            break
                        job = self.ssd._new_job(self.table.get(handle), False)
                        txn.active[key] = job.job_id
                if txn.active or txn.manifest_job is not None:
                    continue
                if txn.failed:
                    self._finish_write(txn, False)
                elif not txn.remaining:
                    assert unit is not None
                    try:
                        self._submit_manifest(txn, unit)
                    except (OSError, ValueError):
                        self._finish_write(txn, False)
            self.ssd._pump()
        finally:
            self._advancing = False

    def _submit_manifest(self, txn: WriteTxn, unit: Unit) -> None:
        limit = min(self.ssd._window, self.credit.q_eff // self.ssd.page_bytes)
        if (
            self.ssd._inflight_stores >= limit
            or self.ssd._store_jobs() >= self.ssd._max_pending
        ):
            return
        txn.manifest = self.document(unit)
        self.ssd._next_job -= 1
        job_id = self.ssd._next_job
        txn.manifest_job = job_id
        self.manifest_jobs[job_id] = txn
        self.ssd._inflight_stores += 1
        task = functools.partial(
            self._write_manifest,
            txn.manifest_path,
            txn.manifest,
            txn.reservation[txn.manifest_path],
        )
        self.ssd._tier._pool.enqueue_store(job_id, 1, [task], n_blocks=1)

    def _write_manifest(self, path: str, record: dict, size: int) -> None:
        self.ssd._limiter.acquire(size)
        with self.cache.write(path, size):
            atomic_json(path, record, self.ssd.page_bytes)

    def manifest_done(self, job_id: int, success: bool) -> None:
        txn = self.manifest_jobs.pop(job_id, None)
        if txn is None:
            return
        self.ssd._inflight_stores -= 1
        txn.manifest_job = None
        unit = self.units.units.get(txn.uid)
        if success:
            size = txn.reservation[txn.manifest_path]
            self.events["manifest_written_bytes"] += size
            if self.ssd._queue.volume is not None:
                self.ssd._queue.volume.record(size)
        success = (
            success and not txn.failed and unit is not None and unit.gen == txn.gen
        )
        if success:
            assert unit is not None and txn.manifest is not None
            self.install_manifest(unit, txn.manifest, txn.manifest_path)
        self._finish_write(txn, success)

    def _finish_write(self, txn: WriteTxn, success: bool) -> None:
        del self.writes[txn.uid]
        self._obligation_cache.pop(txn.uid, None)
        txn.pins.close()
        self.cache.release_paths(txn.reservation)
        for handle in set(txn.handles.values()):
            self.table.release(handle, txn.owner)
        if success:
            nbytes = (
                sum(self.uncommitted_bytes.pop(p, 0) for p in txn.paths.values())
                + txn.reservation[txn.manifest_path]
            )
            if self.flushing:
                if txn.uid in self._flush_guaranteed:
                    objects = {
                        b"k" + bytes(k): self.ssd.page_bytes for k in txn.missing
                    }
                    objects[b"m" + txn.manifest_path.encode()] = txn.reservation[
                        txn.manifest_path
                    ]
                    self._flush_completed_objects.update(objects)
                    self._flush_guaranteed_bytes = sum(
                        self._flush_completed_objects.values()
                    )
                else:
                    self._flush_best_effort_bytes += txn.reservation[txn.manifest_path]
            self.events["committed_bytes"] += nbytes
            self.events["committed_manifests"] += 1
            self.credit.committed(nbytes, busy=bool(self.writes))
            if self.flushing:
                self._flush_guaranteed.discard(txn.uid)
            self.due.pop(txn.uid, None)
        else:
            self.discard_file(txn.manifest_path)
            referenced = {
                OffloadKey(bytes.fromhex(c["key"]))
                for r in self.manifests.values()
                for c in r["components"]
            }
            for key in txn.missing - referenced:
                if self.discard_file(txn.paths[key]):
                    self.uncommitted_bytes.pop(txn.paths[key], None)
            self.events["write_aborted"] += 1
            self.credit.reconcile()

    def prefetch(self, req_id: str, keys: Any) -> list[OffloadKey] | None:
        candidates: dict[UnitId, None] = {}
        for key in keys:
            for uid in tuple(self.manifest_refs.get(key, ())):
                if uid not in self.units.units:
                    self._restore(self.manifests[uid], self.manifest_paths[uid])
                candidates[uid] = None
        if not candidates:
            return None
        if all(self.table.kind_for_key(k) == PageKind.MAIN for k in keys):
            requested = set(keys)
            candidates = {
                uid: None
                for uid in candidates
                if OffloadKey(bytes.fromhex(self.manifests[uid]["main"][-1]))
                in requested
            }
            if not candidates:
                return None
        for uid in sorted(
            candidates,
            key=lambda uid: (self.manifests[uid]["A"], self.units.units[uid].last_use),
            reverse=True,
        ):
            if self._prefetch_unit(req_id, self.units.units[uid]):
                return list(self.units.units[uid].keys)
        return []

    def _prefetch_unit(self, req_id: str, unit: Unit) -> bool:
        if not self.manifest_ok(unit):
            self.events["prefetch_manifest_generation_mismatch"] += 1
            return False
        txn = self.prefetches.get(unit.uid)
        if txn is not None:
            if txn.failed:
                return False
            txn.waiters.add(req_id)
            txn.waiter_deadlines.setdefault(
                req_id, self.ssd.clock() + self.ssd.options["prefetch_timeout_s"]
            )
            self.requests.setdefault(req_id, set()).add(unit.uid)
            return True
        record = self.manifests[unit.uid]
        pins, owner = ExitStack(), self.owner()
        paths = [self.manifest_paths[unit.uid]] + [self.path(k) for k in unit.keys]
        # Scheduler owns every admission/deletion. Once pinned, IO can run
        # concurrently without allowing a source or manifest to disappear.
        for path in paths:
            pins.enter_context(self.cache.read(path))
        expected = {
            OffloadKey(bytes.fromhex(c["key"])): c for c in record["components"]
        }
        for key in unit.keys:
            identity = read_page_identity(self.path(key))
            desc = expected[key]
            if identity is None or [
                f"{identity[0]:08x}",
                identity[1].hex(),
                identity[2].hex(),
            ] != [desc["crc32c"], desc["namespace"], desc["tier_key"]]:
                pins.close()
                self.events["prefetch_identity_failed"] += 1
                self.drop(unit.uid)
                return False
        handles, missing = {}, []
        for key in unit.keys:
            handle = self.table.lookup(key)
            if handle is None:
                missing.append(key)
            elif not self.table.is_ready(handle):
                pins.close()
                return False
            else:
                handles[key] = handle
        cost = len(missing) + sum(
            self.table.reclaimable(self.table.get(h)) for h in handles.values()
        )
        if self.table.net_headroom() < cost:
            pins.close()
            self.events["prefetch_no_headroom"] += 1
            return False
        for handle in handles.values():
            self.table.pin(handle, owner)
        ledger = self.table.reservation_ledger
        fresh = ledger.reserve_frames(len(missing), "unit_prefetch", writeback=False)
        if fresh is None:
            for handle in handles.values():
                self.table.release(handle, owner)
            pins.close()
            return False
        for key, handle in zip(missing, fresh):
            self.table.install_loading(handle, key)
            self.table.pin(handle, owner)
            handles[key] = handle
        ledger.release_temporary(fresh)
        txn = PrefetchTxn(
            unit.uid,
            unit.gen,
            unit.keys,
            handles,
            missing,
            pins,
            owner,
            self.ssd.clock(),
            {req_id},
        )
        self.prefetches[unit.uid] = txn
        txn.sources = frozenset(missing)
        txn.waiter_deadlines[req_id] = (
            self.ssd.clock() + self.ssd.options["prefetch_timeout_s"]
        )
        self.requests.setdefault(req_id, set()).add(unit.uid)
        self.events["prefetch_transactions_started"] += 1
        self.advance_prefetches()
        return True

    def advance_prefetches(self) -> None:
        inflight_by_request: Counter[str] = Counter()
        for current in self.prefetches.values():
            for req_id in current.waiters:
                inflight_by_request[req_id] += len(current.active) * self.ssd.page_bytes
        share = max(
            self.ssd.page_bytes,
            min(
                self.request_read_bytes,
                self.read_budget_bytes
                // max(1, sum(bool(t.pending) for t in self.prefetches.values())),
            ),
        )
        for txn in tuple(self.prefetches.values()):
            unit = self.units.units.get(txn.uid)
            txn.failed |= unit is None or unit.gen != txn.gen
            for req_id in tuple(txn.waiters):
                if self.ssd.clock() >= txn.waiter_deadlines[req_id]:
                    txn.waiters.remove(req_id)
                    self.requests.get(req_id, set()).discard(txn.uid)
                    self.ssd._expired_rounds.add(req_id)
                    self.events["prefetch_waiter_timeout"] += 1
            if not txn.waiters:
                txn.failed = True
            if not txn.failed:
                while (
                    txn.pending
                    and (len(txn.active) + 1) * self.ssd.page_bytes <= share
                    and (self.ssd._inflight_loads + 1) * self.ssd.page_bytes
                    <= self.read_budget_bytes
                    and all(
                        inflight_by_request[req_id] + self.ssd.page_bytes
                        <= self.request_read_bytes
                        for req_id in txn.waiters
                    )
                ):
                    key = txn.pending.pop(0)
                    job = self.ssd._new_job(self.table.get(txn.handles[key]), True)
                    txn.active[key] = job.job_id
                    for req_id in txn.waiters:
                        inflight_by_request[req_id] += self.ssd.page_bytes
                    # The next admission starts with another pending unit.
                    self.prefetches.pop(txn.uid)
                    self.prefetches[txn.uid] = txn
            if txn.active:
                continue
            if txn.failed:
                self._finish_prefetch(txn)
            elif not txn.pending:
                if not txn.complete:
                    self.events["prefetch_latency_us"] += int(
                        (self.ssd.clock() - txn.started) * 1e6
                    )
                txn.complete = True

    def _finish_prefetch(self, txn: PrefetchTxn) -> None:
        del self.prefetches[txn.uid]
        for key, handle in txn.handles.items():
            if not self.table.is_ready(handle):
                self.table.invalidate(handle)
            self.table.release(handle, txn.owner)
        txn.pins.close()
        for req_id in txn.waiters:
            requests = self.requests.get(req_id)
            if requests is not None:
                requests.discard(txn.uid)
                if not requests:
                    del self.requests[req_id]
        self.events["prefetch_retired"] += 1

    def cancel(self, req_id: str) -> None:
        for uid in self.requests.pop(req_id, ()):
            txn = self.prefetches.get(uid)
            if txn is not None:
                txn.waiters.discard(req_id)
                self.events["prefetch_waiter_cancelled"] += 1
        self.advance_prefetches()

    def waiting(self, req_id: str) -> bool:
        return any(
            not self.prefetches[uid].complete
            for uid in self.requests.get(req_id, ())
            if uid in self.prefetches
        )

    def step(self) -> None:
        self.advance_prefetches()
        self.advance_writes()
        stalled = self.credit.events["rate_stalled"]
        self.credit.tick()
        if self.credit.events["rate_stalled"] != stalled:
            logger.warning(
                "Unified SSD has no completed writes; W reduced to %.0f B/s",
                self.credit.rate,
            )
        now = self.ssd.clock()
        over_budget = (
            self.ssd._inflight_stores * self.ssd.page_bytes > self.credit.q_eff
        )
        if over_budget and self._over_budget_since is None:
            self._over_budget_since = now
            self.events["over_budget_events"] += 1
            logger.warning(
                "Unified SSD writes exceed the reduced IO budget; new writes paused"
            )
        elif not over_budget and self._over_budget_since is not None:
            self._over_budget_s += now - self._over_budget_since
            self._over_budget_since = None
        for path in tuple(self.retired_manifests):
            if self.cache.discard(path):
                self.retired_manifests.remove(path)
        now = self.ssd.clock()
        if (
            self.credit.bytes > self.credit.d_dyn * 0.5
            and len(self.writes) < self.max_write_transactions
        ):
            for uid in sorted(
                self.credit.guaranteed, key=lambda uid: self.units.units[uid].last_use
            ):
                unit = self.units.units[uid]
                if self.obligations(unit):
                    self.start_write(unit)
                if len(self.writes) >= self.max_write_transactions:
                    break
        work = 0
        while (
            self._due_heap
            and self._due_heap[0][0] <= now
            and work < self.ssd._max_pending
            and len(self.writes) < self.max_write_transactions
        ):
            due, _, uid = heappop(self._due_heap)
            if self.due.get(uid) != due:
                continue
            self.due.pop(uid)
            unit = self.units.units.get(uid)
            if unit is not None:
                if not self.start_write(unit):
                    self.schedule(unit, now + 1.0)
                work += 1
        if len(self._due_heap) > 4 * len(self.due) + 64:
            self._due_heap.clear()
            for uid, due in tuple(self.due.items()):
                self.schedule(self.units.units[uid], due)

    def reset(self) -> None:
        for txn in self.writes.values():
            txn.failed = True
        for prefetch in self.prefetches.values():
            prefetch.failed = True
        for path in self.manifest_paths.values():
            self.retired_manifests.add(path)
            self.discard_file(path)
        self.manifests.clear()
        self.manifest_paths.clear()
        self.manifest_refs.clear()
        self.files.clear()
        self.due.clear()
        self.filtered_variants.clear()
        self._due_heap.clear()
        self._manifest_sizes.clear()
        self._manifest_names.clear()
        self._obligation_cache.clear()
        epoch = self.table.ssd_epoch
        if epoch:
            self.ssd._salts.setdefault(epoch, os.urandom(16))
        atomic_json(
            self.control_path,
            {
                "reset_epoch": self.units.reset_epoch,
                "component_salt": self.ssd._component_salt.hex(),
                "ssd_epoch": epoch,
                "main_salt": self.ssd._salts.get(epoch, b"").hex(),
            },
        )
        self.credit.reconcile(promote=False)

    def snapshot(self) -> dict[str, int | float]:
        recoverable = self.manifest_refs.keys()
        dirty_pages = [
            p
            for p in self.table.pages[1:]
            if p.state not in (PageState.FREE, PageState.LOADING)
            and not self.table.is_persisted(p)
        ]
        now = self.ssd.clock()
        self.dirty_since = {
            p.handle: self.dirty_since.get(p.handle, now) for p in dirty_pages
        }
        dirty_by_kind: Counter[str] = Counter()
        categories = (
            "inflight_io",
            "ready_queued_due",
            "ready_queued_not_due",
            "ready_unreferenced",
            "ready_filtered",
            "ready_unqueued",
            "private",
            "shadow",
            "other_unwritable",
        )
        classified: dict[str, int | float] = {
            f"dirty_{category}_{metric}": 0
            for category in categories
            for metric in ("bytes", "oldest_age_s")
        }
        # Snapshot-only joins: deduplicate physical handles and ignore marked
        # queue entries. No classification scan runs in the scheduler step.
        queued: dict[PageHandle, float] = {}
        queue = self.ssd._queue
        for batch in queue._batches:
            for handle in batch.keys:
                if handle not in queue._drops:
                    queued[handle] = min(queued.get(handle, batch.due), batch.due)
        for handle in self.ssd._deferred:
            queued[handle] = now
        writable_oldest = 0.0
        writable_bytes = 0
        for page in dirty_pages:
            dirty_by_kind[f"dirty_{page.kind.value}_bytes"] += self.ssd.page_bytes
            age = max(0.0, now - self.dirty_since[page.handle])
            component = self.units.components.get(page.key)
            variants = (
                set()
                if component is None
                else {uid for uid in component.refs if uid.ahead is not None}
            )
            due = [self.due[uid] for uid in variants if uid in self.due]
            due.extend(
                txn.started
                for uid in variants
                if (txn := self.writes.get(uid)) is not None and not txn.failed
            )
            # With no variant, the ordinary single-page writer consumes the
            # queue. Variant pages require an eligible transaction instead.
            if not variants and page.handle in queued:
                due.append(queued[page.handle])
            eligible = bool(due) and (bool(variants) or page.handle not in queue._drops)
            inflight = bool(
                page.io
                or page.ticket is not None
                or page.handle.page_id in self.ssd._page_jobs
            )
            writable = not inflight and self.ssd._writable(page)
            if inflight:
                category = "inflight_io"
            elif not writable:
                category = (
                    page.state.value
                    if page.state in (PageState.PRIVATE, PageState.SHADOW)
                    else "other_unwritable"
                )
            elif not eligible and (
                page.handle in queue._drops
                or variants
                and variants <= self.filtered_variants
            ):
                category = "ready_filtered"
            elif not variants:
                category = "ready_unreferenced"
            elif eligible:
                category = (
                    "ready_queued_due" if min(due) <= now else "ready_queued_not_due"
                )
            else:
                category = "ready_unqueued"
            classified[f"dirty_{category}_bytes"] += self.ssd.page_bytes
            oldest_key = f"dirty_{category}_oldest_age_s"
            classified[oldest_key] = max(classified[oldest_key], age)
            if eligible and writable:
                writable_bytes += self.ssd.page_bytes
                writable_oldest = max(writable_oldest, age)
        referenced = {self.path(k) for k in recoverable} | set(
            self.manifest_paths.values()
        )
        pages = tuple(self.cache._pages.items())
        current_roots = tuple(
            path + os.sep
            for path in (
                f"{self.ssd._tier.file_mapper.base_path}_r"
                f"{self.ssd._tier.file_mapper.rank}",
                f"{self.ssd._component_tier.file_mapper.base_path}_r"
                f"{self.ssd._component_tier.file_mapper.rank}",
                self.directory,
            )
        )
        orphans: Counter[str] = Counter()
        for path, size in pages:
            if path.endswith(".tmp"):
                orphans["ssd_temporary_bytes"] += size
            elif path not in referenced:
                orphans["ssd_orphan_bytes"] += size
                category = (
                    "current_namespace_no_manifest"
                    if path.startswith(current_roots)
                    else "old_namespace_referenced"
                    if self.cache._pins.get(path) or path in self.cache._writing
                    else "old_namespace_zero_ref"
                )
                orphans[f"ssd_orphan_{category}_bytes"] += size
        return {
            **self.credit.snapshot(),
            **self.events,
            "manifests": len(self.manifests),
            "recoverable_bytes": len(recoverable) * self.ssd.page_bytes,
            "total_dirty_bytes": len(dirty_pages) * self.ssd.page_bytes,
            "total_unpersisted_physical_bytes": len(dirty_pages) * self.ssd.page_bytes,
            "oldest_unpersisted_physical_age_s": now
            - min(self.dirty_since.values(), default=now),
            "oldest_dirty_age_s": now - min(self.dirty_since.values(), default=now),
            "writable_dirty_bytes": writable_bytes,
            "oldest_writable_dirty_age_s": writable_oldest,
            **classified,
            **dirty_by_kind,
            "ssd_bytes": self.cache._bytes,
            "ssd_reserved_bytes": self.cache._reserved,
            **{
                key: orphans[key]
                for key in (
                    "ssd_orphan_bytes",
                    "ssd_temporary_bytes",
                    "ssd_orphan_old_namespace_zero_ref_bytes",
                    "ssd_orphan_old_namespace_referenced_bytes",
                    "ssd_orphan_current_namespace_no_manifest_bytes",
                )
            },
            "ssd_inflight_write_bytes": self.ssd._inflight_stores * self.ssd.page_bytes,
            "ssd_inflight_read_bytes": self.ssd._inflight_loads * self.ssd.page_bytes,
            "prefetch_budget_bytes": self.read_budget_bytes,
            "write_transactions": len(self.writes),
            "max_write_transactions": self.max_write_transactions,
            "prefetch_transactions": len(self.prefetches),
            "file_transaction_refs": sum(self.cache._pins.values()),
            "file_manifest_refs": sum(
                len(refs) for refs in self.manifest_refs.values()
            ),
            "file_write_refs": sum(len(txn.paths) for txn in self.writes.values()),
            "file_prefetch_refs": sum(
                len(txn.keys) + 1 for txn in self.prefetches.values()
            ),
            "file_io_refs": len(self.ssd._jobs) + len(self.manifest_jobs),
            "over_budget_inflight_bytes": max(
                0, self.ssd._inflight_stores * self.ssd.page_bytes - self.credit.q_eff
            ),
            "over_budget_duration_s": self._over_budget_s
            + (
                now - self._over_budget_since
                if self._over_budget_since is not None
                else 0.0
            ),
        }

    def final_flush(
        self, budget_s: float | None, deadline: float | None, io_grace_s: float | None
    ):
        from vllm.v1.kv_offload.host.persist import FlushReport

        budget = (
            self.ssd.options["shutdown_flush_timeout_s"]
            if budget_s is None
            else budget_s
        )
        grace = (
            self.ssd.options["shutdown_io_grace_s"]
            if io_grace_s is None
            else io_grace_s
        )
        started = self.ssd.clock()
        end = started + max(0.0, budget)
        if deadline is not None:
            end = min(end, deadline)
        report = FlushReport(planned_budget_s=max(0.0, end - started))
        if self.ssd._closed:
            report.skipped = "closed"
            return report
        self.credit.window_s = min(
            self.credit.window_s, max(0.0, end - started - grace)
        )
        self.credit.reconcile(promote=False)
        guaranteed = set(self.credit.guaranteed)
        self._flush_guaranteed = guaranteed
        self._flush_completed_objects: dict[bytes, int] = {}
        self._flush_guaranteed_bytes = self._flush_best_effort_bytes = 0
        self.credit.promotions_enabled = False
        self.flushing = self.ssd._flushing = True
        self.ssd._report, self.ssd._phase = report, "budget"
        self.ssd._poll()
        self.ssd._discard_queue()
        self.ssd._deferred.clear()
        for req_id in tuple(set(self.requests) | set(self.ssd._rounds)):
            self.ssd.cancel(req_id)
        # Started IO retires before releasing best-effort reservations.
        # Large best-effort reservations must leave room for guarantees.
        for txn in self.writes.values():
            if txn.uid not in guaranteed:
                txn.failed = True
        variants = sorted(
            (u for u in self.units.units.values() if u.parent is not None),
            key=lambda u: (u.uid not in guaranteed, -u.last_use),
        )
        # Return unsubmitted best-effort pages to their transactions. Their
        # transaction leases remain held, and started IO drains normally.
        for job_id in tuple(self.ssd._submit):
            job = self.ssd._jobs[job_id]
            page = self.table.get(job.handle)
            component = self.units.components.get(page.key)
            if component is not None and component.refs & guaranteed:
                continue
            for txn in self.writes.values():
                for key in tuple(txn.active):
                    if txn.active[key] == job_id:
                        del txn.active[key]
            self.ssd._submit.remove(job_id)
            self.ssd._finish(job, False)
        for job in self.ssd._jobs.values():
            if not job.loading:
                job.flush = True
                report.candidates += 1
        # IO already submitted retires normally. New work uses this one queue.
        queue = list(variants)
        orphan_handles = [
            self.table.pages[pid].handle
            for pid in reversed(self.table._lru)
            if self.ssd._writable(self.table.pages[pid])
            and not any(
                uid.ahead is not None
                for uid in self.units.components.get(
                    self.table.pages[pid].key, Component(PageKind.MAIN)
                ).refs
            )
        ]
        if end <= started or not self.ssd.enabled:
            report.skipped = "no budget" if self.ssd.enabled else "disabled"
        else:
            self.ssd._limiter.unthrottle()
            while self.ssd.clock() < end:
                self.ssd._poll()
                self.advance_prefetches()
                if queue:
                    unit = queue[0]
                    if (
                        unit.uid not in self.units.units
                        or self.manifest_ok(unit)
                        or unit.uid not in self.writes
                        and not self.start_write(unit)
                        and (unit.uid not in guaranteed or not self.writes)
                    ):
                        queue.pop(0)
                elif not self.writes and orphan_handles:
                    handle = orphan_handles.pop(0)
                    page = self.ssd._page(handle)
                    if page is not None and self.ssd._writable(page):
                        self.ssd._new_job(page, False)
                self.advance_writes()
                self.ssd._pump()
                if (
                    not queue
                    and not orphan_handles
                    and not self.writes
                    and not self.ssd._jobs
                ):
                    break
                time.sleep(0.002)
            else:
                report.deadline_hit = True
        self.ssd._phase = "cancel"
        for txn in self.writes.values():
            txn.failed = True
        for job_id in tuple(self.ssd._submit):
            job = self.ssd._jobs[job_id]
            report.dropped_unsubmitted += int(job.flush)
            self.ssd._finish(job, False)
        self.ssd._submit.clear()
        report.in_flight_at_deadline = sum(j.flush for j in self.ssd._jobs.values())
        self.ssd._phase = "grace"
        grace_end = end + max(0.0, grace)
        if deadline is not None:
            grace_end = min(grace_end, deadline)
        while (self.ssd._jobs or self.manifest_jobs) and self.ssd.clock() < grace_end:
            self.ssd._poll()
            time.sleep(0.002)
        self.ssd._poll()
        self.advance_writes()
        self.advance_prefetches()
        report.stuck = sum(j.flush for j in self.ssd._jobs.values())
        report.other_io_stuck = (
            len(self.ssd._jobs) - report.stuck + len(self.manifest_jobs)
        )
        self.ssd.quarantined |= bool(self.ssd._jobs or self.manifest_jobs)
        report.quarantined = self.ssd.quarantined
        report.guaranteed_completed_bytes = self._flush_guaranteed_bytes
        report.best_effort_completed_bytes = self._flush_best_effort_bytes
        outstanding: dict[Obligation, int] = {}
        for unit in self.units.units.values():
            if unit.parent is not None:
                outstanding.update(self.obligations(unit))
        for page in self.table.pages[1:]:
            if page.state not in (
                PageState.FREE,
                PageState.LOADING,
            ) and not self.table.is_persisted(page):
                identity = (
                    ("component", page.key)
                    if page.key is not None
                    else ("unpublished", page.handle)
                )
                outstanding[identity] = self.ssd.page_bytes
        report.unfinished_bytes = sum(outstanding.values())
        report.flush_elapsed_s = self.ssd.clock() - started
        self.ssd._report, self.ssd._phase = None, "done"
        return report

    def begin_close(self) -> None:
        self.credit.promotions_enabled = False
        for txn in self.writes.values():
            txn.failed = True
        for req_id in tuple(self.requests):
            self.cancel(req_id)

    def retire(self) -> None:
        self.advance_prefetches()
        self.advance_writes()
