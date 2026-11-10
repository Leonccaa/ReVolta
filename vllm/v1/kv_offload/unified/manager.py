# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context component adapter over the MAIN allocator and pinned region."""

from collections import Counter, deque
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from vllm.v1.kv_offload.base import (
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.host.page_table import (
    HostPageTable,
    PageHandle,
    PageKind,
    PageState,
    Result,
)
from vllm.v1.kv_offload.unified.ledger import HostReservationLedger
from vllm.v1.kv_offload.unified.units import Component, UnitId, UnitIndex, VariantOffer
from vllm.v1.kv_offload.unified.value import UnitValuePolicy


@dataclass
class Build:
    req_id: str
    uid: UnitId
    gen: int
    handles: dict[OffloadKey, PageHandle]
    new: set[OffloadKey]
    cost: int
    pending: set[OffloadKey]
    dispatched: set[OffloadKey] = field(default_factory=set)
    failed: bool = False
    failure_reason: str | None = None
    newly_protected: tuple[PageHandle, ...] = ()
    already_owned: tuple[PageHandle, ...] = ()


class ComponentTransferSpec(CPULoadStoreSpec):
    def __init__(self, handles: tuple[PageHandle, ...], gen: int):
        super().__init__([h.page_id for h in handles])
        self.handles = handles
        self.gen = gen


class UnifiedStoreManager(OffloadingManager):
    def __init__(self, spec: Any):
        self.spec = spec
        self.table: HostPageTable
        self.units: UnitIndex
        self.ledger: HostReservationLedger
        self.value: UnitValuePolicy
        self.attached = False
        self.build_frames_per_request = 0
        self.rejected_step: set[OffloadKey] = set()
        self._receipts: dict[int, tuple[int, int]] = {}
        self.ssd: Any = None
        self.kind_by_group: dict[int, PageKind] = {}
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

        self.main_group = spec.kv_cache_config.host_kv_group_id
        for i, group in enumerate(spec.kv_cache_config.kv_cache_groups):
            if i == self.main_group or not group.kv_cache_spec.prefix_cacheable:
                continue
            s = group.kv_cache_spec
            if isinstance(s, UniformTypeKVCacheSpecs):
                s = next(iter(s.kv_cache_specs.values()))
            self.kind_by_group[i] = (
                PageKind.STATE if isinstance(s, MambaSpec) else PageKind.DRAFT
            )
        self.builds: dict[tuple[UnitId, int], Build] = {}
        self.dependents: dict[OffloadKey, set[tuple[UnitId, int]]] = {}
        self.pending: dict[OffloadKey, tuple[PageHandle, Build | None]] = {}
        self._dispatched: set[OffloadKey] = set()
        self.jobs: dict[
            int, tuple[int, str, tuple[OffloadKey, ...], tuple[PageHandle, ...], bool]
        ] = {}
        self._prepared: dict[
            int, tuple[tuple[OffloadKey, ...], tuple[PageHandle, ...], bool]
        ] = {}
        self._lookup_holds: dict[str, tuple[PageHandle, ...]] = {}
        self._loads: dict[str, deque[tuple[int, tuple[PageHandle, ...]]]] = {}
        self._serial = 0
        self.events: Counter[str] = Counter()
        self.max_pending_variants = 8
        self.waiting_on: dict[OffloadKey, set[str]] = {}
        self._wait_keys: dict[str, set[OffloadKey]] = {}
        self.waiting: dict[str, dict[tuple[UnitId, bytes], VariantOffer]] = {}
        self._latest_variant_offers: dict[str, VariantOffer] = {}
        self._missing_main: dict[str, dict[tuple[UnitId, bytes], set[OffloadKey]]] = {}
        self._waiting_boundaries: dict[UnitId, set[str]] = {}
        self._draining_variants: set[str] = set()
        self._draining_main: dict[OffloadKey, PageHandle] = {}
        self.num_stored_blocks = 0
        self.num_evicted_blocks = 0
        self.evicted_blocks_by_reason: Counter[str] = Counter()
        self.state_latest_only = any(
            t.get("state_latest_only", False)
            for t in spec.extra_config.get("secondary_tiers", [])
        )
        self.bytes: Counter[str] = Counter()
        self.payload_bytes_by_group: dict[int, int] = {}
        world = (
            spec.vllm_config.parallel_config.world_size
            if hasattr(spec, "vllm_config")
            else 1
        )
        for group_id in self.kind_by_group:
            group = spec.kv_cache_config.kv_cache_groups[group_id]
            sizes = group.kv_cache_spec
            members = (
                sizes.kv_cache_specs.values()
                if isinstance(sizes, UniformTypeKVCacheSpecs)
                else [sizes] * len(getattr(group, "layer_names", [None]))
            )
            self.payload_bytes_by_group[group_id] = (
                sum(
                    replace(x, page_size_padded=None).page_size_bytes
                    if isinstance(x, MambaSpec)
                    else x.page_size_bytes
                    for x in members
                )
                * world
            )

    def attach(self, pool: Any) -> None:
        if self.attached:
            if self.table is not pool.host_pages:
                raise ValueError(
                    "Unified manager already attached to another allocator"
                )
            return
        self.table = pool.host_pages
        self.table.kind_for_key = lambda key: self.kind_by_group.get(
            get_offload_group_idx(key), PageKind.MAIN
        )
        self.units = UnitIndex(self.table.refresh_value)
        self.value = UnitValuePolicy(self.table, self.units)
        self.value.on_ready = self._on_ready
        self.value.on_reset = self._on_table_reset
        self.value.on_seal_retired = self._on_seal_retired
        reclaim = pool._reclaim_cached_page if pool.elastic else lambda: False
        self.ledger = HostReservationLedger(self.table, reclaim)
        self.ledger.build_frames_per_request = self.build_frames_per_request
        self.attached = True
        for handle in tuple(self.table.index.values()):
            page = self.table.get(handle)
            if page.state == PageState.READY:
                self.value.published(page)

    def attach_ssd(self, ssd: Any) -> None:
        self.ssd = ssd
        ssd.enable_components(
            {
                "group_kinds": {str(g): k.value for g, k in self.kind_by_group.items()},
                "transport_bytes": self.spec.cpu_page_size_per_worker,
            }
        )
        if ssd.enabled and not ssd._component_tier.disabled:
            from vllm.v1.kv_offload.unified.ssd import UnitSsdStore

            ssd.unit_store = UnitSsdStore(self, ssd)

    def admit_build(self, req_id: str, frames: int) -> bool:
        return self.ledger.admit(req_id, frames)

    def kind(self, key: OffloadKey) -> PageKind:
        group = get_offload_group_idx(key)
        if group not in self.kind_by_group:
            raise ValueError("Component key is outside the unified support matrix")
        return self.kind_by_group[group]

    def component_available(self, key: OffloadKey) -> bool:
        """READY or already dispatched; never a promise to read an old source."""
        pending = self.pending.get(key)
        return self.is_ready_resident(key) or (
            pending is not None
            and key in self._dispatched
            and self.table.get(pending[0]).valid
            and (pending[1] is None or not pending[1].failed)
        )

    def _fail_build(self, build: Build, reason: str) -> None:
        build.failed = True
        if build.failure_reason is None:
            build.failure_reason = reason
            self.units.events[f"build_failed_{reason}"] += 1
        self.units.revoke(build.uid, build.gen, "build_failed")

    def begin_boundary(
        self, req_id: str, boundary: bytes, keys: Iterable[OffloadKey]
    ) -> Build | None:
        keys = tuple(dict.fromkeys(keys))
        states = {
            get_offload_group_idx(k) for k in keys if self.kind(k) == PageKind.STATE
        }
        expected_states = {
            g for g, kind in self.kind_by_group.items() if kind == PageKind.STATE
        }
        if (
            not keys
            or states != expected_states
            or any(
                get_offload_block_hash(k) != boundary
                for k in keys
                if self.kind(k) == PageKind.STATE
            )
        ):
            self.events["build_inventory_rejected"] += 1
            return None
        uid = UnitId(boundary, self.units.reset_epoch)
        existing = self.units.units.get(uid)
        if existing is not None:
            return self.builds.get((uid, existing.gen))
        for k in keys:
            pending = self.pending.get(k)
            if pending is not None:
                handle, owner = pending
                if not self.table.get(handle).valid or (
                    owner is not None and owner.failed
                ):
                    self.events["build_stale_inflight_rejected"] += 1
                    return None
        handles = {
            k: self.pending[k][0] if k in self.pending else self.table.lookup(k)
            for k in keys
        }
        if any(
            h is not None and k not in self.pending and not self.table.is_ready(h)
            for k, h in handles.items()
        ):
            self.events["build_loading_rejected"] += 1
            return None
        new = [k for k, h in handles.items() if h is None]
        # Three inventories; unique physical pages, independent of key count.
        protected = {
            h
            for h in handles.values()
            if h is not None and self.table.reclaimable(self.table.get(h))
        }
        already_owned = {h for h in handles.values() if h is not None} - protected
        cost = len(new) + len(protected)
        if not self.ledger.consume(req_id, cost):
            return None
        marker = id(handles)
        for h in (*protected, *already_owned):
            self.table.pin(h, marker)
        fresh = self.ledger.reserve_frames(len(new), "build", reserved=True)
        if fresh is None:
            for h in (*protected, *already_owned):
                self.table.release(h, marker)
            self.ledger.retire(req_id, cost)
            return None
        for k, h in zip(new, fresh):
            handles[k] = h
            self.table.describe(h, self.kind(k), get_offload_group_idx(k))
        concrete = {k: h for k, h in handles.items() if h is not None}
        for k in keys:
            self.units.components.setdefault(k, Component(self.kind(k)))
        unit = self.units.begin(boundary, keys)
        shared = {k for k in keys if k in self.pending}
        build = Build(
            req_id,
            uid,
            unit.gen,
            concrete,
            set(new),
            cost,
            set(new) | shared,
            dispatched=shared.copy(),
            newly_protected=tuple(protected),
            already_owned=tuple(already_owned),
        )
        self.builds[uid, unit.gen] = build
        for k, h in concrete.items():
            self.table.pin(h, id(build))
            if k in build.new:
                self.pending[k] = (h, build)
            if k in build.pending:
                self.dependents.setdefault(k, set()).add((uid, unit.gen))
        self.ledger.charge(req_id, [*protected, *fresh])
        # Owners exist before the temporary reservations leave.
        self.ledger.release_temporary(fresh)
        for h in (*protected, *already_owned):
            self.table.release(h, marker)
        self.events["build_new_frames"] += len(new)
        self.events["build_new_protection"] += len(protected)
        self.events["build_already_owned"] += len(already_owned)
        self._finish_build(build)
        return build

    def _finish_build(self, build: Build) -> None:
        if build.pending:
            return
        if build.failed:
            self._fail_build(build, build.failure_reason or "finish_union_incomplete")
        elif not self.units.finish(build.uid, build.gen):
            self._fail_build(build, "finish_union_incomplete")
        for h in build.handles.values():
            self.table.release(h, id(build))
        self.builds.pop((build.uid, build.gen), None)
        for req_id in tuple(self._waiting_boundaries.get(build.uid, ())):
            self.retry_variants(req_id)

    def _component_retired(
        self, key: OffloadKey, ok: bool, reason: str = "finish_union_incomplete"
    ) -> None:
        for identity in self.dependents.pop(key, ()):
            build = self.builds.get(identity)
            if build is not None:
                if not ok:
                    self._fail_build(build, reason)
                build.pending.discard(key)
                self._finish_build(build)

    def end_step(self) -> None:
        # A source hand-off lasts one step. Shared in-flight components depend
        # on their original job and never cause a second source read.
        for build in tuple(self.builds.values()):
            absent = build.pending - build.dispatched
            if absent:
                self._fail_build(build, "source_not_dispatched")
                for key in absent:
                    self.pending.pop(key, None)
                    self.table.invalidate(build.handles[key])
                    self._component_retired(key, False, "source_not_dispatched")

    def _unindex_waiter(self, req_id: str) -> None:
        for key in self._wait_keys.pop(req_id, ()):
            waiters = self.waiting_on[key]
            waiters.discard(req_id)
            if not waiters:
                del self.waiting_on[key]

    def _index_waiter(self, req_id: str) -> None:
        keys = set().union(*self._missing_main.get(req_id, {}).values())
        prior = self._wait_keys.get(req_id, set())
        for key in prior - keys:
            waiters = self.waiting_on[key]
            waiters.discard(req_id)
            if not waiters:
                del self.waiting_on[key]
        for key in keys - prior:
            self.waiting_on.setdefault(key, set()).add(req_id)
        if keys:
            self._wait_keys[req_id] = keys
        else:
            self._wait_keys.pop(req_id, None)

    def _on_ready(self, key: OffloadKey) -> None:
        for req_id in tuple(self.waiting_on.get(key, ())):
            ready = False
            for missing in self._missing_main[req_id].values():
                if key in missing:
                    missing.remove(key)
                    ready |= not missing
            if ready:
                self.retry_variants(req_id)
            else:
                self._index_waiter(req_id)

    def clear_waiter(self, req_id: str) -> None:
        prior_keys = self._wait_keys.get(req_id, set()).copy()
        self._unindex_waiter(req_id)
        for key in prior_keys:
            if key not in self.waiting_on:
                self._draining_main.pop(key, None)
        self._missing_main.pop(req_id, None)
        boundaries = {uid for uid, _ in self.waiting.pop(req_id, {})}
        for uid in boundaries:
            waiters = self._waiting_boundaries[uid]
            waiters.discard(req_id)
            if not waiters:
                del self._waiting_boundaries[uid]
        self._draining_variants.discard(req_id)

    def _on_seal_retired(self, key: OffloadKey) -> None:
        self._draining_main.pop(key, None)
        for req_id in tuple(self.waiting_on.get(key, ())):
            if req_id in self._draining_variants:
                self.retry_variants(req_id)

    def _can_drain(self, offer: VariantOffer) -> bool:
        b = self.units.units.get(offer.boundary)
        if b is None or b.gen != offer.gen:
            return False
        if not b.complete and (b.uid, b.gen) not in self.builds:
            return False
        return all(
            k in self._draining_main
            and self.table.get(self._draining_main[k]).ticket is not None
            for k in self._missing_keys(offer)
        )

    def _missing_keys(self, offer: VariantOffer) -> set[OffloadKey]:
        return {
            k
            for k in offer.main
            if k not in self.units.components or not self.units._resident(k, "union")
        }

    def offer_variant(self, req_id: str, offer: VariantOffer) -> None:
        prior = self._latest_variant_offers.get(req_id)
        if prior is None or (offer.boundary_position, offer.ahead_position) >= (
            prior.boundary_position,
            prior.ahead_position,
        ):
            self._latest_variant_offers[req_id] = offer
        identity = offer.boundary, offer.ahead
        if self.waiting.get(req_id, {}).get(identity) == offer:
            return
        if self.units.register(offer) is None:
            offers = self.waiting.setdefault(req_id, {})
            if identity not in offers and len(offers) >= self.max_pending_variants:
                old = next(iter(offers))
                self._remove_offer(req_id, old)
                self.events["variant_wait_limit"] += 1
            offers[identity] = offer
            self._missing_main.setdefault(req_id, {})[identity] = self._missing_keys(
                offer
            )
            self._waiting_boundaries.setdefault(offer.boundary, set()).add(req_id)
            self._index_waiter(req_id)

    def _remove_offer(self, req_id: str, identity: tuple[UnitId, bytes]) -> None:
        del self.waiting[req_id][identity]
        del self._missing_main[req_id][identity]
        uid = identity[0]
        if not any(key[0] == uid for key in self.waiting[req_id]):
            waiters = self._waiting_boundaries[uid]
            waiters.discard(req_id)
            if not waiters:
                del self._waiting_boundaries[uid]

    def retry_variants(self, req_id: str) -> None:
        offers = self.waiting.get(req_id, {})
        for key, offer in tuple(offers.items()):
            b = self.units.units.get(offer.boundary)
            if b is None:
                self.units.events["variant_b_absent_terminal"] += 1
                self.units.events["variant_b_missing"] += 1
            if req_id in self._draining_variants and not self._can_drain(offer):
                self.events["variant_finish_incomplete"] += 1
                self._remove_offer(req_id, key)
                continue
            if (
                b is None
                or b.gen != offer.gen
                or (
                    not self._missing_main[req_id][key]
                    and self.units.register(offer) is not None
                )
            ):
                self._remove_offer(req_id, key)
            elif not self._missing_main[req_id][key]:
                # Registration rechecks the complete inventory atomically.
                # A formerly present page may have disappeared while waiting.
                self._missing_main[req_id][key] = self._missing_keys(offer)
        if not offers:
            self.clear_waiter(req_id)
        else:
            self._index_waiter(req_id)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        # Normal end: accepted stores are allowed to publish after completion.
        # Only unused credit is returned now; accepted DMA moves to DRAINING.
        req_id = req_context.req_id
        # Retain only offers whose remaining inputs have already been accepted.
        # Seal tickets name exact physical incarnations, without retaining GPU IDs.
        if self.waiting.get(req_id):
            missing = set().union(*self._missing_main[req_id].values())
            for page in self.table.pages[1:]:
                if (
                    page.valid
                    and page.ticket is not None
                    and page.ticket.key in missing
                ):
                    self._draining_main[page.ticket.key] = page.handle
            self._draining_variants.add(req_id)
            self.retry_variants(req_id)
        self._latest_variant_offers.pop(req_context.req_id, None)
        self.release_lookup(req_context.req_id)
        self.end_step()
        self.ledger.finish(req_context.req_id)
        if self.ssd is not None:
            self.ssd.cancel(req_context.req_id)

    def cancel_request(self, req_id: str, reason: str = "cancel") -> None:
        self.clear_waiter(req_id)
        self._latest_variant_offers.pop(req_id, None)
        self.release_lookup(req_id)
        for b in tuple(self.builds.values()):
            if b.req_id == req_id:
                self._fail_build(b, reason)
        self.end_step()
        self.ledger.finish(req_id)
        if self.ssd is not None:
            self.ssd.cancel(req_id)

    def is_ready_resident(self, key: OffloadKey) -> bool:
        handle = self.table.lookup(key)
        return handle is not None and self.table.is_ready(handle)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if (
            self.ssd is not None
            and self.ssd.unit_store is not None
            and req_context.req_id in self.ssd.unit_store.requests
        ):
            self.ssd.poll()
            if self.ssd.unit_store.waiting(req_context.req_id):
                return None
        if self.is_ready_resident(key):
            return True
        if key in self.pending:
            return None
        if self.ssd is not None:
            self.ssd.poll()
            if self.is_ready_resident(key):
                return True
            self.ssd.prefetch(req_context.req_id, [key], extend=True)
            if self.ssd.waiting(req_context.req_id):
                return None
        return False

    def _next(self) -> int:
        self._serial += 1
        return self._serial

    def hold_lookup(self, req_id: str, keys: Iterable[OffloadKey]) -> bool:
        ordered = tuple(dict.fromkeys(keys))
        handles = tuple(self.table.lookup(k) for k in ordered)
        if any(h is None or not self.table.is_ready(h) for h in handles):
            return False
        concrete = tuple(dict.fromkeys(h for h in handles if h is not None))
        prior = self._lookup_holds.get(req_id)
        if prior == concrete:
            return True
        cost = sum(self.table.reclaimable(self.table.get(h)) for h in concrete)
        if self.table.net_headroom() < cost:
            self.events["receive_reserve_rejected"] += 1
            return False
        for h in concrete:
            self.table.pin(h, id(concrete))
        self.release_lookup(req_id)
        self._lookup_holds[req_id] = concrete
        return True

    def release_lookup(self, req_id: str) -> None:
        handles = self._lookup_holds.pop(req_id, ())
        for h in handles:
            self.table.release(h, id(handles))

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> ComponentTransferSpec:
        ordered = tuple(sorted(keys, key=get_offload_group_idx))
        handles = tuple(self.table.lookup(k) for k in ordered)
        if any(h is None or not self.table.is_ready(h) for h in handles):
            raise ValueError("Unified load requires all components READY")
        concrete = tuple(h for h in handles if h is not None)
        gen = self._next()
        for h in concrete:
            self.table.pin(h, gen)
        self._loads.setdefault(req_context.req_id, deque()).append((gen, concrete))
        self.release_lookup(req_context.req_id)
        self._prepared[gen] = (ordered, concrete, False)
        return ComponentTransferSpec(concrete, gen)

    def _release_load(
        self, req_id: str, gen: int, handles: tuple[PageHandle, ...]
    ) -> None:
        for h in handles:
            self.table.release(h, gen)
        loads = self._loads.get(req_id)
        if loads is not None:
            loads.remove((gen, handles))
            if not loads:
                del self._loads[req_id]
        self._prepared.pop(gen, None)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        loads = self._loads.get(req_context.req_id)
        if loads:
            gen, handles = loads[0]
            self._release_load(req_context.req_id, gen, handles)

    def receive_components(self, spec: ComponentTransferSpec):
        from vllm.v1.kv_offload.host.receive import ReceiveComponent

        keys, handles, _ = self._prepared[spec.gen]
        return tuple(
            ReceiveComponent(k, h, spec.gen, spec.gen, get_offload_group_idx(k))
            for k, h in zip(keys, handles)
        )

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        ordered = tuple(sorted(dict.fromkeys(keys), key=get_offload_group_idx))
        ordered = tuple(
            k for k in ordered if k not in self.rejected_step or k in self.pending
        )
        missing = tuple(k for k in ordered if not self.is_ready_resident(k))
        if any(k not in self.pending for k in missing):
            # Window/all-policy extras may be cached opportunistically, but
            # never spend build reservations belonging to another boundary.
            extra = [k for k in missing if k not in self.pending]
            fresh = self.ledger.reserve_frames(len(extra), "optional_store")
            if fresh is None:
                self.events["optional_store_rejected"] += len(extra)
                missing = tuple(k for k in missing if k not in extra)
                fresh = []
            for k, h in zip(extra, fresh):
                self.table.describe(h, self.kind(k), get_offload_group_idx(k))
                self.table.start_io(h, id(h))
                self.pending[k] = (h, None)
            self.ledger.release_temporary(fresh)
        # Optional window extras have no Build, but still have exactly one
        # physical producer. Deduplicate them just like shared build keys.
        ready = tuple(k for k in missing if k not in self._dispatched)
        if not ready:
            return PrepareStoreOutput([], ComponentTransferSpec((), 0), [])
        gen = self._next()
        handles = tuple(self.pending[k][0] for k in ready)
        for k, h in zip(ready, handles):
            self._dispatched.add(k)
            build = self.pending[k][1]
            if build is not None:
                build.dispatched.add(k)
            self.table.start_io(h, gen)
            if build is None:
                self.table.finish_io(h, id(h))
        self._prepared[gen] = (ready, handles, True)
        return PrepareStoreOutput(list(ready), ComponentTransferSpec(handles, gen), [])

    def bind_job(
        self, job_id: int, req_id: str, spec: ComponentTransferSpec, store: bool
    ) -> None:
        keys, handles, is_store = self._prepared.pop(spec.gen)
        if store != is_store or handles != spec.handles:
            raise ValueError("Mismatched unified transfer generation")
        self.jobs[job_id] = (spec.gen, req_id, keys, handles, store)

    def receipt(self, job_id: int, gen: int, completed: int, failed: int) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job[0] != gen:
            return False
        if (completed | failed) & ~self.table.all_ranks or failed & ~completed:
            raise ValueError("Invalid unified rank receipt")
        old = self._receipts.get(job_id, (0, 0))
        masks = (old[0] | completed, old[1] | failed)
        self._receipts[job_id] = masks
        if masks[0] == self.table.all_ranks:
            self.complete_job(job_id, not masks[1])
            return True
        return False

    def complete_job(self, job_id: int, success: bool = True) -> None:
        job = self.jobs.pop(job_id, None)
        if job is None:
            return
        gen, req_id, keys, handles, store = job
        self._receipts.pop(job_id, None)
        if not store:
            self._release_load(req_id, gen, handles)
            return
        for k, h in zip(keys, handles):
            current = self.pending.get(k)
            build = current[1] if current is not None and current[0] == h else None
            dependent_builds = [
                self.builds[i] for i in self.dependents.get(k, ()) if i in self.builds
            ]
            ok = (
                success
                and self.table.get(h).valid
                and (build is None or any(not b.failed for b in dependent_builds))
            )
            if ok:
                self.table.producer_result(h, compute=Result.OK, seal=Result.OK)
                self.table.publish(h, k)
                self.num_stored_blocks += 1
            else:
                self.table.invalidate(h)
            if current is not None and current[0] == h:
                del self.pending[k]
                self._dispatched.discard(k)
            self.table.finish_io(h, gen)
            self._component_retired(
                k, ok, "dma_failed_mask" if not success else "finish_union_incomplete"
            )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        # Compatibility entry point resolves the original job's physical
        # handles. Production passes job_id through complete_job directly.
        for job_id, (_, req, job_keys, _, store) in tuple(self.jobs.items()):
            if store and req == req_context.req_id and set(job_keys) == set(keys):
                self.complete_job(job_id, success)
                return

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self.units.demote(keys, False)
        touched: dict[UnitId, None] = {}
        for key in keys:
            c = self.units.components.get(key)
            if c is not None:
                for uid in c.refs:
                    # Preserve the order of last occurrences from the old
                    # per-component loop, updating each unit just once.
                    touched.pop(uid, None)
                    touched[uid] = None
            h = self.table.lookup(key)
            if h is not None:
                lease = self._next()
                self.table.pin(h, lease)
                self.table.release(h, lease)
        for uid in touched:
            self.units.touch(uid)

    def demote(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self.units.demote(keys, True)

    def drop_pending_secondary_writes(
        self, keys: Collection[OffloadKey], req_id: str | None = None
    ) -> None:
        if self.ssd is not None:
            handles = [self.table.lookup(k) for k in keys]
            self.ssd._queue.drop([h for h in handles if h is not None], "superseded")
            store = self.ssd.unit_store
            if store is not None:
                variants = {
                    uid
                    for key in keys
                    if key in self.units.components
                    for uid in self.units.components[key].refs
                    if uid.ahead is not None
                }
                offer = (
                    self._latest_variant_offers.get(req_id)
                    if req_id is not None
                    else None
                )
                target = (
                    None
                    if offer is None
                    else UnitId(
                        offer.boundary.boundary, offer.boundary.reset, offer.ahead
                    )
                )
                store.credit.supersede(variants, target)
                for uid in variants:
                    store.due.pop(uid, None)
                    store.filtered_variants.add(uid)

    def _on_table_reset(self) -> None:
        self._draining_variants.clear()
        self._draining_main.clear()
        self._latest_variant_offers.clear()
        for req_id in tuple(self._lookup_holds):
            self.release_lookup(req_id)
        self.waiting.clear()
        self.waiting_on.clear()
        self._wait_keys.clear()
        self._missing_main.clear()
        self._waiting_boundaries.clear()
        for build in self.builds.values():
            self._fail_build(build, "reset")

    def reset_cache(self) -> None:
        self._draining_variants.clear()
        self._draining_main.clear()
        self._latest_variant_offers.clear()
        for req_id in tuple(self._lookup_holds):
            self.release_lookup(req_id)
        for b in tuple(self.builds.values()):
            self._fail_build(b, "reset")
        self.waiting.clear()
        self.waiting_on.clear()
        self._wait_keys.clear()
        self._missing_main.clear()
        self._waiting_boundaries.clear()
        self.end_step()
        # A connector-only reset withdraws component evidence without touching
        # MAIN's running PRIVATE/SHADOW bindings. Every submitted IO/lease
        # retains its physical incarnation until its original receipt.
        components = [
            p.handle
            for p in self.table.pages[1:]
            if p.kind != PageKind.MAIN and p.state != PageState.FREE
        ]
        for handle in components:
            self.table.invalidate(handle)
            self.table.unpersist(handle)
        if self.ssd is not None:
            self.ssd.reset_components(components)
        self.value.reset()
        for handle in tuple(self.table.index.values()):
            page = self.table.get(handle)
            if page.state == PageState.READY:
                self.value.published(page)
        for page in self.table.pages[1:]:
            if page.state != PageState.FREE:
                self.table.refresh_value(page.handle)
        self.table.collect()

    def record_dma(
        self, job_id: int, gen: int, rank: int, nbytes: int, kind: str
    ) -> None:
        job = self.jobs.get(job_id)
        if job is None or job[0] != gen:
            return
        if nbytes < 0 or kind not in ("main", "state", "draft", "unknown"):
            raise ValueError("Invalid unified DMA byte receipt")
        self.bytes[f"dma_{'d2h' if job[4] else 'h2d'}_{kind}"] += nbytes

    def residency_totals(self) -> dict[tuple[str, str], int]:
        totals: Counter[tuple[str, str]] = Counter()
        for key, c in self.units.components.items():
            if c.cpu is not None:
                totals[
                    "target" if c.kind == PageKind.MAIN else c.kind.value,
                    self.units.cpu_status(key),
                ] += 1
        from vllm.v1.kv_offload.cpu.dependency import KINDS, STATUSES

        return {
            (kind, status): totals[kind, status]
            for kind in KINDS
            for status in STATUSES
        }

    def snapshot(self) -> dict[str, int]:
        occupancy: Counter[str] = Counter(
            {
                f"{bill}_{kind.value}_bytes": 0
                for bill in ("allocated", "payload")
                for kind in PageKind
            }
        )
        frame_bytes = getattr(self.spec, "unified_frame_bytes", 0)
        for page in self.table.pages[1:]:
            if page.state != PageState.FREE:
                occupancy[f"allocated_{page.kind.value}_bytes"] += frame_bytes
                occupancy[f"payload_{page.kind.value}_bytes"] += (
                    frame_bytes
                    if page.kind == PageKind.MAIN
                    else self.payload_bytes_by_group.get(page.group, 0)
                )
        return {
            "zero_page_bytes": frame_bytes,
            "region_bytes": len(self.table.pages) * frame_bytes,
            **occupancy,
            **self.units.snapshot(),
            **self.ledger.snapshot(),
            "value_heap": len(self.value.heap),
            "value_heap_stale": sum(not self.value._valid(e) for e in self.value.heap),
            "value_heap_compactions": self.value.compactions,
            "foreground_lookup_leases": sum(
                len(h) for h in self._lookup_holds.values()
            ),
            "physical_charges": len(self.ledger.charges),
            "variant_registration_draining_requests": len(self._draining_variants),
            "variant_registration_draining_offers": sum(
                len(self.waiting.get(req_id, ())) for req_id in self._draining_variants
            ),
            "variant_registration_draining_main": len(self._draining_main),
            "value_work_max": self.value.work_max,
            **(self.bytes + self.table.dma_bytes),
            **(
                self.ssd.unit_store.snapshot()
                if self.ssd is not None and self.ssd.unit_store is not None
                else {}
            ),
        }

    def shutdown(self) -> None:
        # MAIN's region owner performs cleanup after all worker access retires.
        pass
