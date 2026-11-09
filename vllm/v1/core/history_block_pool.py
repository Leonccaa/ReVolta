# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""History IDs, physical host-page bindings and the elastic GPU shared region.

page ownership added history-ID generations and host-page bindings. shared hot region
adds the shared
region: the free history IDs whose index pages carry no prefix (DONATED) lend
their pages to the hot four-token slots of every layer. Each block ID has one
role, owned here on the scheduler thread:

- NULL (ID 0): never donated.
- INUSE: referenced, including ring scratch; its slots are reserved (-2).
- CACHED: unreferenced with a valid hash; slots reserved, prefix retained.
- DONATED: unreferenced without a hash; its k slots belong to the hot region.
- DRAINING: a failed load target or a block with in-flight transfers; reserved
  until those accesses retire, then donated.

The hot floor F (slots) keeps ceil(F / k) IDs DONATED. Allocation takes cached
IDs above the index-cache target first, then DONATED IDs above the floor, then
the remaining cached IDs; it never goes below the floor. The worker applies
the per-step role changes (donate, then claim) on the forward stream before
zeroing and beginning new IDs, so every rank sees the same hot capacity.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
)
from vllm.v1.core.long_context_kv import HOST
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host.page_table import (
    HostPageTable,
    PageHandle,
    PageState,
    Result,
    SealTicket,
)
from vllm.v1.request import Request


class _DonatedQueue(FreeKVCacheBlockQueue):
    """The donated free list with an O(1) membership flag per history ID.

    ``finish_roles`` runs every host-KV step and only needs the membership of
    the IDs touched this step; walking the whole list (tens of thousands of
    IDs) dominated the scheduler step.
    """

    def __init__(self, num_blocks: int) -> None:
        self.member = bytearray(num_blocks)
        super().__init__([])

    def popleft(self) -> KVCacheBlock:
        block = super().popleft()
        self.member[block.block_id] = 0
        return block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        blocks = super().popleft_n(n)
        for block in blocks:
            self.member[block.block_id] = 0
        return blocks

    def remove(self, block: KVCacheBlock) -> None:
        super().remove(block)
        self.member[block.block_id] = 0

    def append(self, block: KVCacheBlock) -> None:
        super().append(block)
        self.member[block.block_id] = 1

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        super().prepend_n(blocks)
        for block in blocks:
            self.member[block.block_id] = 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        super().append_n(blocks)
        for block in blocks:
            self.member[block.block_id] = 1


@dataclass
class SharedRegionStep:
    """Per-step role changes of the shared region, identical on every rank."""

    donate: list[int] = field(default_factory=list)
    """IDs whose slots join the hot region (retire their groups, owner -1)."""
    claim: list[int] = field(default_factory=list)
    """IDs whose slots leave the hot region (unmap, owner -2), before zeroing."""
    hot_slots: int = 0
    """U_step: hot slots per layer available during this step."""
    floor_slots: int = 0
    """F: guaranteed hot slots after this step's admissions."""


class HistoryBlockPool(BlockPool):
    slots_per_block = 0
    """k: hot slots per history ID page; 0 keeps the static page ownership hot pool."""
    value_observer: Callable[[bytes], None] | None = None
    """value arbitration: told every unreferenced cached ID that gives up its prefix
    (value, allocation, floor or host reclaim), as the offload key of its
    bound host page (the key a later NEW_LOAD restores), else its GPU hash."""

    def configure_host(self, num_pages: int, num_workers: int, group_id: int) -> None:
        self.host_pages = HostPageTable(num_pages, num_workers)
        self.host_group_id = group_id
        self.hgen = [0] * self.num_gpu_blocks
        self.bindings: dict[int, PageHandle] = {}
        self.last_use: dict[tuple[int, int], int] = {}
        self.binding_updates: dict[int, int] = {}
        self.new_hashes: dict[str, list[tuple[int, int, bytes]]] = {}
        self.pending_seals: dict[str, dict[tuple[int, int, PageHandle], bytes]] = {}
        self.last_scheduled: dict[str, int] = {}
        self.step = 0
        self._jobs: dict[int, tuple[list[PageHandle], int]] = {}
        self._job_hids: dict[int, list[int]] = {}

    # ----- shared hot region shared region
    # ---------------------------------------------------

    def configure_elastic(
        self,
        slots_per_block: int,
        *,
        min_floor_slots: int = 0,
        index_cache_target: int | None = None,
    ) -> None:
        """Lend unhashed free ID pages to the hot region.

        ``index_cache_target`` is T_idx: cached IDs above it are evicted
        before the hot region shrinks; below it, the region above the floor
        is used first. None starts at half of the region above the minimum
        floor until value arbitration (value arbitration) adjusts it.
        """
        if slots_per_block < 1 or min_floor_slots < 0:
            raise ValueError("Invalid shared-region geometry")
        self.slots_per_block = slots_per_block
        self.min_floor_slots = min_floor_slots
        self.floor_slots = min_floor_slots
        usable = self.num_gpu_blocks - 1
        if self.floor_blocks() > usable:
            raise ValueError(
                f"Hot floor of {min_floor_slots} slots needs {self.floor_blocks()} "
                f"history IDs, more than the {usable} available"
            )
        self.index_cache_target = (
            (usable - self.floor_blocks()) // 2
            if index_cache_target is None
            else index_cache_target
        )
        self.donated_queue = _DonatedQueue(self.num_gpu_blocks)
        for block in self.free_block_queue.get_all_free_blocks():
            if block.block_hash is None:
                self.free_block_queue.remove(block)
                self.donated_queue.append(block)
        # Worker view after the last emitted step: 1 = slots in the hot region.
        # The worker starts with every non-null ID donated (owner -1).
        self.device_free = bytearray(self.num_gpu_blocks)
        for block in self.donated_queue.get_all_free_blocks():
            self.device_free[block.block_id] = 1
        self.num_device_free = self.donated_queue.num_free_blocks
        self.held: dict[int, int] = {}
        """Free unhashed IDs not yet donated: hid -> retire_after step (or -1
        while transfers are in flight)."""
        self.draining: dict[int, int] = {}
        """Referenced IDs that must drain when freed: hid -> retire_after."""
        self._io_hids: dict[int, int] = {}
        self._dirty: set[int] = set()
        self.keep_hids: frozenset[int] = frozenset()
        self.stats = {
            "cached_evicted_for_value": 0,
            "cached_evicted_for_floor": 0,
            "cached_evicted_for_host": 0,
            "cached_evicted_for_allocation": 0,
        }

    @property
    def elastic(self) -> bool:
        return self.slots_per_block > 0

    def floor_blocks(self, floor_slots: int | None = None) -> int:
        slots = self.floor_slots if floor_slots is None else floor_slots
        return -(-slots // self.slots_per_block)

    def num_donated(self) -> int:
        return self.donated_queue.num_free_blocks

    def num_cached_free(self) -> int:
        return self.free_block_queue.num_free_blocks

    # ----- value arbitration value arbitration
    # ----------------------------------------------

    def arbiter_snapshot(self):
        from vllm.v1.core.value_arbiter import PoolSnapshot

        usable = self.num_gpu_blocks - 1
        cached, donated = self.num_cached_free(), self.num_donated()
        # In use: referenced (ring, parked, idle sessions) plus held/draining.
        return PoolSnapshot(
            usable, usable - cached - donated, self.floor_blocks(), cached
        )

    def set_index_cache_target(self, target: int) -> None:
        self.index_cache_target = target

    def donate_cached_for_value(self, num_blocks: int) -> int:
        """Give the LRU cached IDs to the hot region (a value adjustment)."""
        given = 0
        while given < num_blocks and self.free_block_queue.num_free_blocks:
            block = self.free_block_queue.popleft()
            self._maybe_evict_cached_block(block)
            self.stats["cached_evicted_for_value"] += 1
            self._to_donated(block)
            given += 1
        return given

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        block_hash = block.block_hash
        evicted = super()._maybe_evict_cached_block(block)
        observer = self.value_observer
        if evicted and observer is not None and block.ref_cnt == 0:
            assert block_hash is not None
            key: bytes | None = None
            handle = getattr(self, "bindings", {}).get(block.block_id)
            if handle is not None:
                try:
                    key = self.host_pages.get(handle).key
                except ValueError:
                    key = None
            observer(get_block_hash(block_hash) if key is None else key)
        return evicted

    def raw_free_blocks(self) -> int:
        """Unreferenced IDs that allocation may take (cached plus donated)."""
        if not self.elastic:
            return super().get_num_free_blocks()
        return self.free_block_queue.num_free_blocks + self.num_donated()

    def hot_slots(self) -> int:
        """U: hot slots per layer with the current roles."""
        return self.num_donated() * self.slots_per_block

    def set_hot_floor(self, floor_slots: int) -> None:
        """Raise or lower F; donate cached IDs until the floor is DONATED."""
        self.floor_slots = max(self.min_floor_slots, floor_slots)
        self._ensure_floor()

    def _ensure_floor(self) -> None:
        need = self.floor_blocks()
        while self.num_donated() < need and self.free_block_queue.num_free_blocks:
            block = self.free_block_queue.popleft()
            self._maybe_evict_cached_block(block)
            self.stats["cached_evicted_for_floor"] += 1
            self._to_donated(block)

    def _to_donated(self, block: KVCacheBlock) -> None:
        """A free, unhashed ID: donate it, or hold it while accesses remain."""
        hid = block.block_id
        self.unbind(hid)
        self._dirty.add(hid)
        retire_after = self.draining.pop(hid, None)
        if self._io_hids.get(hid):
            self.held[hid] = -1 if retire_after is None else retire_after
        elif retire_after is not None and (self.host_pages.watermark < retire_after):
            self.held[hid] = retire_after
        else:
            self.donated_queue.append(block)

    def _release_held(self) -> None:
        watermark = self.host_pages.watermark
        for hid, retire_after in list(self.held.items()):
            if self._io_hids.get(hid) or watermark < retire_after:
                continue
            del self.held[hid]
            self.donated_queue.append(self.blocks[hid])
            self._dirty.add(hid)

    def mark_draining(self, hids: Iterable[int], retire_after: int) -> None:
        """NEW_LOAD failure targets: reserve until accesses up to the step
        ``retire_after`` retire, then donate (receive transactions merge point)."""
        for hid in hids:
            self.draining[hid] = max(self.draining.get(hid, 0), retire_after)

    def host_reclaimable_after(self) -> int:
        """|Reclaimable_after| for HOST admission, excluding ``keep_hids``.

        FREE and owner-free retired pages, plus pages whose every binding is
        a retired, unreferenced cached ID outside this candidate's hits.
        """
        table = self.host_pages
        count = table.available()
        pages: dict[int, list[int]] = {}
        block = self.free_block_queue.fake_free_list_head.next_free_block
        tail = self.free_block_queue.fake_free_list_tail
        while block is not None and block is not tail:
            handle = self.bindings.get(block.block_id)
            if handle is not None:
                pages.setdefault(handle.page_id, []).append(block.block_id)
            block = block.next_free_block
        for hids in pages.values():
            if self._cached_page_reclaimable(hids):
                count += 1
        return count

    def host_kept_free(self) -> int:
        """This candidate's unreferenced hit IDs that hold a page: counted in
        its block demand, but they need no fresh page."""
        return sum(
            self.blocks[hid].ref_cnt == 0 and hid in self.bindings
            for hid in self.keep_hids
        )

    def _cached_page_reclaimable(self, hids: list[int]) -> bool:
        if any(hid in self.keep_hids for hid in hids):
            return False
        page = self.host_pages.get(self.bindings[hids[0]])
        if page.leases or page.io or page.ticket or len(page.binds) != len(hids):
            return False
        if any((hid, self.hgen[hid]) not in page.binds for hid in hids):
            return False
        watermark = self.host_pages.watermark
        return (
            watermark >= page.last_use
            and all(watermark >= r.retire_after for r in page.retirements)
            and all(
                watermark >= self.last_use.get((hid, self.hgen[hid]), 0) for hid in hids
            )
        )

    def _reclaim_cached_page(self) -> bool:
        """Free one host page held only by retired cached bindings.

        The GPU IDs lose their hashes and become DONATED first, which is the
        index-cache-first order; then their bindings retire the page.
        """
        pages: dict[int, list[KVCacheBlock]] = {}
        block = self.free_block_queue.fake_free_list_head.next_free_block
        tail = self.free_block_queue.fake_free_list_tail
        while block is not None and block is not tail:
            handle = self.bindings.get(block.block_id)
            if handle is not None:
                pages.setdefault(handle.page_id, []).append(block)
            block = block.next_free_block
        for blocks in pages.values():
            if self._cached_page_reclaimable([b.block_id for b in blocks]):
                for victim in blocks:
                    self.free_block_queue.remove(victim)
                    self._maybe_evict_cached_block(victim)
                    self.stats["cached_evicted_for_host"] += 1
                    self._to_donated(victim)
                return True
        return False

    def _reserve_page(self, pending: int = 1) -> PageHandle | None:
        """Reserve one page; when none is reclaimable, retire one page held
        only by retired cached bindings and retry. The unbound READY page is
        then recycled by the table's LRU scan, so it passes the host SSD persistence
        pre-eviction write-back/evict hooks like any other READY page."""
        handle = self.host_pages.reserve(pending=pending)
        while handle is None and self.elastic and self._reclaim_cached_page():
            handle = self.host_pages.reserve(pending=pending)
        return handle

    def _pop_elastic(self, num_blocks: int) -> list[KVCacheBlock]:
        cached = self.free_block_queue.num_free_blocks
        donated = self.num_donated()
        spare = donated - self.floor_blocks()
        if num_blocks > cached + max(spare, 0):
            raise ValueError(f"Cannot get {num_blocks} free blocks above the hot floor")
        first = min(num_blocks, max(cached - self.index_cache_target, 0))
        then = min(num_blocks - first, max(spare, 0))
        last = num_blocks - first - then
        blocks = self.free_block_queue.popleft_n(first)
        blocks += self.donated_queue.popleft_n(then)
        blocks += self.free_block_queue.popleft_n(last)
        self.stats["cached_evicted_for_allocation"] += sum(
            b.block_hash is not None for b in blocks
        )
        self._dirty.update(b.block_id for b in blocks)
        return blocks

    def finish_roles(self) -> SharedRegionStep:
        """Net role changes against the worker's view, after the final
        executed set and job tracking of this step are known."""
        if not self.elastic:
            return SharedRegionStep()
        self._release_held()
        self._ensure_floor()
        step = SharedRegionStep(floor_slots=self.floor_slots)
        # Every membership change marks the ID dirty, so only dirty IDs can
        # differ from the worker view.
        donated = self.donated_queue.member
        for hid in sorted(self._dirty):
            free = bool(donated[hid])
            if free != bool(self.device_free[hid]):
                (step.donate if free else step.claim).append(hid)
                self.device_free[hid] = int(free)
                self.num_device_free += 1 if free else -1
        self._dirty.clear()
        step.hot_slots = self.hot_slots()
        assert self.num_device_free == self.donated_queue.num_free_blocks, (
            "shared-region roles diverged"
        )
        return step

    def filter_zero_ids(self, block_ids: list[int] | None) -> list[int] | None:
        """Never zero a page the hot region owns after this step's roles."""
        if not self.elastic or not block_ids:
            return block_ids
        kept = [hid for hid in block_ids if not self.device_free[hid]]
        return kept or None

    def _can_retire_binding(self, hid: int) -> bool:
        handle = self.bindings.get(hid)
        if handle is None:
            return False
        page = self.host_pages.get(handle)
        return (
            len(page.binds) == 1
            and not (page.leases or page.io or page.ticket)
            and self.host_pages.watermark
            >= max(page.last_use, self.last_use.get((hid, self.hgen[hid]), 0))
        )

    def get_num_free_blocks(self) -> int:
        if self.elastic:
            # GPU domain only; HOST is answered by host_reclaimable_after().
            return max(self.raw_free_blocks() - self.floor_blocks(), 0)
        if not hasattr(self, "host_pages"):
            return super().get_num_free_blocks()
        # page ownership conservatively charges each candidate, including ring scratch.
        # The normal free-queue order is retained. shared hot region replaces this bound
        # with
        # its source-aware four-domain admission contract.
        available = self.host_pages.available()
        count = 0
        block = self.free_block_queue.fake_free_list_head.next_free_block
        tail = self.free_block_queue.fake_free_list_tail
        while block is not None and block is not tail:
            available += int(self._can_retire_binding(block.block_id))
            if available <= 0:
                break
            count += 1
            available -= 1
            block = block.next_free_block
        return count

    def unbind(self, hid: int) -> None:
        handle = self.bindings.pop(hid, None)
        if handle is not None:
            hgen = self.hgen[hid]
            for request_id, pending in list(self.pending_seals.items()):
                pending.pop((hid, hgen, handle), None)
                if not pending:
                    del self.pending_seals[request_id]
            self.host_pages.unbind(handle, hid, hgen, self.last_use.pop((hid, hgen), 0))
            self.binding_updates[hid] = -1

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        if self.elastic:
            blocks = self._pop_elastic(num_blocks)
            for block in blocks:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
            if self.admission_observer is not None:
                self.admission_observer.on_allocated(blocks)
        else:
            blocks = super().get_new_blocks(num_blocks)
        scope = self.admission_observer._scope if self.admission_observer else ()
        for index, block in enumerate(blocks):
            hid = block.block_id
            self.unbind(hid)
            self.hgen[hid] += 1
            if scope is not None and HOST in scope:
                # host SSD persistence: a pre-eviction SSD write may only take pages
                # beyond the
                # rest of this admitted batch.
                handle = self._reserve_page(pending=len(blocks) - index)
                assert handle is not None, "Host admission did not reserve enough pages"
                self.host_pages.bind(handle, hid, self.hgen[hid], self.step)
                self.bindings[hid] = handle
                self.binding_updates[hid] = handle.page_id
        return blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        super().cache_full_blocks(
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
            block_mask,
        )
        if kv_cache_group_id == self.host_group_id:
            entries = self.new_hashes.setdefault(request.request_id, [])
            for index in range(num_cached_blocks, num_full_blocks):
                block = blocks[index]
                if (
                    not block.is_null
                    and block.block_hash is not None
                    and self.host_pages.get(self.bindings[block.block_id]).state
                    == PageState.PRIVATE
                ):
                    entries.append(
                        (
                            block.block_id,
                            self.hgen[block.block_id],
                            request.block_hashes[
                                (index + 1) * (block_size // self.hash_block_size) - 1
                            ],
                        )
                    )
            # A receive may cache these blocks before admission. Preserve
            # their physical identities across steps until an actual producer
            # executes. Include pending local hits shared by another request.
            pending = self.pending_seals.setdefault(request.request_id, {})
            for index, block in enumerate(blocks[:num_full_blocks]):
                hid = block.block_id
                handle = self.bindings.get(hid)
                if handle is not None and block.block_hash is not None:
                    page = self.host_pages.get(handle)
                    if (
                        page.valid
                        and page.state == PageState.PRIVATE
                        and not page.ticket
                    ):
                        # Host keys use the terminal request-chain hash, while
                        # the GPU group hash concatenates all base-block hashes.
                        pending[hid, self.hgen[hid], handle] = request.block_hashes[
                            (index + 1) * (block_size // self.hash_block_size) - 1
                        ]

    def start_step(self, step: int) -> None:
        self.step = step
        self.new_hashes.clear()

    def cancel_scheduled(self, request_id: str) -> None:
        self.pending_seals.pop(request_id, None)
        for hid, hgen, _ in self.new_hashes.pop(request_id, []):
            if self.hgen[hid] == hgen:
                self.evict_blocks({hid})
                handle = self.bindings.get(hid)
                if handle is not None:
                    self.host_pages.get(handle).compute_result = Result.FAILED

    def finish_schedule(self, executed: Iterable[str]) -> list[SealTicket]:
        executed = set(executed)
        tickets = []
        for request_id in executed:
            self.last_scheduled[request_id] = self.step
            for (hid, hgen, handle), block_hash in self.pending_seals.pop(
                request_id, {}
            ).items():
                if self.hgen[hid] != hgen or self.bindings.get(hid) != handle:
                    continue
                page = self.host_pages.get(handle)
                if not page.valid or page.state != PageState.PRIVATE or page.ticket:
                    continue
                block = self.blocks[hid]
                if block.block_hash is None or not get_block_hash(
                    block.block_hash
                ).endswith(block_hash):
                    continue
                tickets.append(
                    self.host_pages.open_seal(
                        handle,
                        hid,
                        hgen,
                        make_offload_key(block_hash, self.host_group_id),
                        self.step,
                    )
                )
        return tickets

    def release_request(self, request_id: str, blocks: Iterable[KVCacheBlock]) -> None:
        self.pending_seals.pop(request_id, None)
        last_use = self.last_scheduled.pop(request_id, 0)
        for block in blocks:
            hid = block.block_id
            if hid in self.bindings:
                key = (hid, self.hgen[hid])
                self.last_use[key] = max(self.last_use.get(key, 0), last_use)
                page = self.host_pages.get(self.bindings[hid])
                page.last_use = max(page.last_use, last_use)

    def touch(self, blocks) -> None:
        if not self.elastic:
            return super().touch(blocks)
        observer = self.admission_observer
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is not None:
                    self.free_block_queue.remove(block)
                elif block.block_id in self.held:
                    raise ValueError("A held history ID cannot be referenced")
                else:
                    self.donated_queue.remove(block)
                self._dirty.add(block.block_id)
                if observer is not None:
                    observer.on_acquired(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def evict_blocks(self, block_ids: set[int]) -> None:
        if not self.elastic:
            return super().evict_blocks(block_ids)
        for block_id in block_ids:
            assert block_id < len(self.blocks), f"Invalid block_id {block_id}"
            block = self.blocks[block_id]
            free_cached = (
                block.ref_cnt == 0
                and not block.is_null
                and block.block_hash is not None
            )
            if self._maybe_evict_cached_block(block) and free_cached:
                self.free_block_queue.remove(block)
                self._to_donated(block)

    def _free_elastic(self, blocks: list[KVCacheBlock], reuse_first: bool) -> None:
        for block in blocks:
            block.ref_cnt -= 1
        freed = [b for b in blocks if b.ref_cnt == 0 and not b.is_null]
        if self.admission_observer is not None:
            self.admission_observer.on_released(freed)
        cached = []
        for block in freed:
            if block.block_hash is not None and (
                reuse_first or not self.enable_caching
            ):
                # A prefix that can never hit again is not worth an ID page.
                self._maybe_evict_cached_block(block)
            if block.block_hash is None:
                self._to_donated(block)
            else:
                cached.append(block)
        self.free_block_queue.append_n(cached)

    def free_blocks(
        self, ordered_blocks: Iterable[KVCacheBlock], reuse_first: bool = False
    ) -> None:
        blocks = list(ordered_blocks)
        if self.elastic:
            self._free_elastic(blocks, reuse_first)
        else:
            super().free_blocks(blocks, reuse_first)
        for block in blocks:
            hid = block.block_id
            if hid in self.bindings and block.ref_cnt == 0:
                if block.block_hash is None:
                    self.unbind(hid)
                else:
                    self.host_pages.get(self.bindings[hid]).binds[
                        hid, self.hgen[hid]
                    ] = "cached"

    def num_physically_free_blocks(self) -> int:
        """Unreferenced IDs, independent of host and hot-floor caps."""
        if self.elastic:
            return self.raw_free_blocks() + len(self.held)
        return super().get_num_free_blocks()

    def reset_prefix_cache(self) -> bool:
        # Check physical GPU references independently of the host capacity cap.
        if self.num_physically_free_blocks() != self.num_gpu_blocks - 1:
            return False
        if self.elastic and self.held:
            return False
        self.host_pages.reset()
        self.binding_updates.update({hid: -1 for hid in self.bindings})
        self.bindings.clear()
        self.last_use.clear()
        self.new_hashes.clear()
        self.pending_seals.clear()
        if not self.elastic:
            return super().reset_prefix_cache()
        # One queue for the base check; afterwards no ID has a hash, so every
        # one of them is donated.
        self.free_block_queue.append_n(self.donated_queue.popleft_n(self.num_donated()))
        if not super().reset_prefix_cache():
            return False
        for block in self.free_block_queue.popleft_n(
            self.free_block_queue.num_free_blocks
        ):
            self.donated_queue.append(block)
            self._dirty.add(block.block_id)
        return True

    def track_job(self, job_id: int, hids: list[int]) -> None:
        if self.elastic:
            # A transfer reads/writes whole index pages: keep those IDs out of
            # the hot region until every rank finished the job.
            self._job_hids[job_id] = list(hids)
            for hid in hids:
                self._io_hids[hid] = self._io_hids.get(hid, 0) + 1
        handles = list({self.bindings[hid] for hid in hids if hid in self.bindings})
        for handle in handles:
            self.host_pages.start_io(handle, job_id)
        self._jobs[job_id] = (handles, 0)

    def complete_jobs(self, completed: dict[int, int]) -> None:
        # Existing connector job counts retain their established contract.
        # Seal and step completion use rank masks independently.
        for job_id, count in completed.items():
            if job_id not in self._jobs:
                continue
            handles, seen = self._jobs[job_id]
            seen += count
            if seen < self.host_pages.num_workers:
                self._jobs[job_id] = (handles, seen)
                continue
            for handle in handles:
                self.host_pages.finish_io(handle, job_id)
            del self._jobs[job_id]
            for hid in self._job_hids.pop(job_id, ()):
                left = self._io_hids[hid] - 1
                if left:
                    self._io_hids[hid] = left
                else:
                    del self._io_hids[hid]
