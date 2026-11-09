# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-thread ownership of physical host pages.

Hashes name published immutable content. Every owner and completion instead
names a physical incarnation, so invalidating a hash never retires DMA.
"""

from collections import Counter, OrderedDict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.v1.kv_offload.unified.ledger import HostReservationLedger

from vllm.v1.kv_offload.base import OffloadKey, get_offload_group_idx


class PageState(str, Enum):
    FREE = "free"
    PRIVATE = "private"
    READY = "ready"
    SHADOW = "shadow"
    LOADING = "loading"


class PageKind(str, Enum):
    MAIN = "main"
    STATE = "state"
    DRAFT = "draft"


class Result(str, Enum):
    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True)
class PageHandle:
    page_id: int
    epoch: int


@dataclass(frozen=True)
class Retirement:
    hid: int
    hgen: int
    handle: PageHandle
    retire_after: int


@dataclass(frozen=True)
class SealTicket:
    ticket: int
    hid: int
    hgen: int
    handle: PageHandle
    key: OffloadKey
    step: int


@dataclass
class HostKVStep:
    step: int
    bindings: list[tuple[int, int]]
    seals: list[SealTicket]
    # shared hot region shared region, applied in order on the forward stream: donate,
    # claim, then the zeroing/begin of new IDs. Empty for the static pool.
    donate: list[int] = field(default_factory=list)
    claim: list[int] = field(default_factory=list)
    hot_slots: int = 0
    """U_step: hot slots per layer during this step (0: static pool)."""
    floor_slots: int = 0
    requests: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    """CPU batch metadata: req_id -> (query rows, G_q, class), class 0 for
    decode/verify rows and 1 for prefill. Planner input on every rank."""
    arbiter_epoch: int | None = None
    """value arbitration value-arbitration window; None when arbitration is off."""


@dataclass
class HostPage:
    handle: PageHandle
    state: PageState = PageState.FREE
    key: OffloadKey | None = None
    binds: dict[tuple[int, int], str] = field(default_factory=dict)
    leases: set[int] = field(default_factory=set)
    io: set[int] = field(default_factory=set)
    ticket: SealTicket | None = None
    writer: tuple[int, int, int] | None = None
    compute_result: Result = Result.PENDING
    seal_result: Result = Result.PENDING
    retirements: list[Retirement] = field(default_factory=list)
    last_use: int = 0
    persisted_epoch: int | None = None
    valid: bool = True
    completed_ranks: int = 0
    failed_ranks: int = 0
    kind: PageKind = PageKind.MAIN
    group: int = 0
    revision: int = 0
    lru_clock: int = 0


class PageValuePolicy(Protocol):
    def refresh(self, page: HostPage) -> None: ...
    def published(self, page: HostPage) -> None: ...
    def lost(self, page: HostPage) -> None: ...
    def persisted(self, page: HostPage) -> None: ...
    def reset(self) -> None: ...
    def ssd_lost(self, key: OffloadKey, handle: PageHandle) -> None: ...
    def order(self, page: HostPage) -> tuple[int, int]: ...
    def select(self, pending: int, writeback: bool) -> HostPage | None: ...


class PersistenceHooks(Protocol):
    """host SSD persistence SSD controller callbacks; every call runs on the scheduler
    thread.

    SSD IO owners and prefetch holds use negative ids. Connector jobs and
    request leases keep their existing non-negative ids.
    """

    def on_published(self, handle: PageHandle) -> None: ...

    def try_writeback(self, page: "HostPage", headroom_before: int) -> bool: ...

    def on_evict(self, page: "HostPage") -> None: ...

    def on_reset(self) -> None: ...


def is_ssd_owner(owner_id: int) -> bool:
    return owner_id < 0


class HostPageTable:
    """One allocator, four owner classes, and a gap-free rank watermark.

    Page zero is the permanent ZERO_PAGE and is excluded from allocation.
    Bind roles are descriptive in page ownership. Cached bindings continue to protect
    their page until the GPU allocator explicitly unbinds them.
    """

    def __init__(self, num_pages: int, num_workers: int = 1, *, zero_page: bool = True):
        if num_pages < 1 or num_workers < 1:
            raise ValueError("Host pages and workers must be positive")
        self.pages = [HostPage(PageHandle(i, 0)) for i in range(num_pages)]
        self._first = int(zero_page)
        if zero_page:
            self.pages[0].state = PageState.READY
        self.num_workers = num_workers
        self.all_ranks = (1 << num_workers) - 1
        self.index: dict[OffloadKey, PageHandle] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._free = list(range(num_pages - 1, self._first - 1, -1))
        self._next_ticket = 0
        self._steps: dict[int, int] = {}
        self.watermark = 0
        self.ssd_epoch = 0
        # Two views: all allocatable pages, and only pages collect may free.
        # READY stays in the LRU until reserve explicitly selects it.
        self._candidates: set[int] = set()
        self._auto: set[int] = set()
        # (threshold, page_id, epoch, revision). The second heap prunes
        # individual retirements even while a newer access still protects a page.
        self._pending: list[tuple[int, int, int, int]] = []
        self._retiring: list[tuple[int, int, int, int]] = []
        self._pending_live: set[int] = set()
        self._retiring_live: set[int] = set()
        self.heap_compactions = 0
        self.collect_work_max = 0
        self.state_counts = dict.fromkeys(PageState, 0)
        self.state_counts[PageState.FREE] = len(self._free)
        self.persisted_pages = 0
        # host SSD persistence: SSD persistence hooks (None keeps the page ownership
        # behaviour exactly).
        self.persistence: PersistenceHooks | None = None
        # The sum of committed R_host
        # reservations (parked
        # readbacks, protected successors). Background IO may only consume
        # pages beyond it . page ownership has no such reservation.
        self.committed_reserve: Callable[[], int] = lambda: 0
        self.reservation_ledger: HostReservationLedger | None = None
        self.kind_for_key: Callable[[OffloadKey], PageKind] = lambda key: PageKind.MAIN
        self.value_policy: PageValuePolicy | None = None
        self._lru_clock = 0
        self.dma_bytes: Counter[str] = Counter()

    def refresh_value(self, handle: PageHandle) -> None:
        """Logical references and SSD evidence share incremental reclamation's revision
        boundary."""
        with suppress(ValueError):
            self._refresh(self.get(handle))

    def get(self, handle: PageHandle) -> HostPage:
        if not 0 <= handle.page_id < len(self.pages):
            raise ValueError("Invalid host page")
        page = self.pages[handle.page_id]
        if page.handle != handle or page.state == PageState.FREE:
            raise ValueError("Stale host page epoch")
        return page

    def diagnostics(self) -> dict[str, int]:
        pages = self.pages[self._first :]
        return {
            "binds": sum(len(p.binds) for p in pages),
            "leases": sum(len(p.leases) for p in pages),
            "io": sum(len(p.io) for p in pages),
            "tickets": sum(p.ticket is not None for p in pages),
            "unretired_pages": sum(not self._retired(p) for p in pages),
            "watermark": self.watermark,
        }

    def lookup(self, key: OffloadKey) -> PageHandle | None:
        return self.index.get(key)

    def is_ready(self, handle: PageHandle) -> bool:
        page = self.get(handle)
        return page.valid and page.state == PageState.READY

    def _retired(self, page: HostPage) -> bool:
        return self.watermark >= page.last_use and all(
            self.watermark >= r.retire_after for r in page.retirements
        )

    def reclaimable(self, page: HostPage) -> bool:
        return (
            page.handle.page_id >= self._first
            and not (page.state == PageState.LOADING and page.valid)
            and not (page.binds or page.leases or page.io or page.ticket)
            and self._retired(page)
        )

    def num_free(self) -> int:
        """Pages on the free list: O(1) lower bound of ``available()``."""
        return len(self._free)

    def available(self) -> int:
        return len(self._free) + len(self._candidates)

    def _set_state(self, page: HostPage, state: PageState) -> None:
        self.state_counts[page.state] -= 1
        self.state_counts[state] += 1
        page.state = state

    def _current(self, entry: tuple[int, int, int, int]) -> bool:
        _, pid, epoch, revision = entry
        page = self.pages[pid]
        return page.handle.epoch == epoch and page.revision == revision

    def _refresh(self, page: HostPage) -> None:
        """Invalidate same-incarnation pending work after every eligibility event.

        All retirement thresholds originate from last_use, which is monotone
        within an incarnation. Keep the explicit max for the retirement contract.
        No table scan; lazy heap garbage is bounded by the physical pool size.
        """
        pid = page.handle.page_id
        page.revision += 1
        self._candidates.discard(pid)
        self._auto.discard(pid)
        self._pending_live.discard(pid)
        self._retiring_live.discard(pid)
        if pid < self._first or page.state == PageState.FREE:
            return
        if page.retirements:
            page.retirements[:] = [
                r for r in page.retirements if r.retire_after > self.watermark
            ]
        threshold = max(
            page.last_use,
            max((r.retire_after for r in page.retirements), default=0),
        )
        identity = (pid, page.handle.epoch, page.revision)
        if threshold > self.watermark:
            heappush(self._pending, (threshold, *identity))
            self._pending_live.add(pid)
        if page.retirements:
            heappush(
                self._retiring,
                (min(r.retire_after for r in page.retirements), *identity),
            )
            self._retiring_live.add(pid)
        reclaimable = self.reclaimable(page)
        if reclaimable:
            self._candidates.add(pid)
            if page.state != PageState.READY:
                self._auto.add(pid)
        limit = 4 * len(self.pages) + 64
        for heap in (self._pending, self._retiring):
            if len(heap) > limit:
                heap[:] = [entry for entry in heap if self._current(entry)]
                heapify(heap)
                self.heap_compactions += 1
        # The revision above invalidates old heap entries for every event.
        # Payment retirement and heap insertion only become possible once
        # the physical owners have actually retired.
        if reclaimable and self.reservation_ledger is not None:
            self.reservation_ledger.owner_changed(page.handle)
        if reclaimable and self.value_policy is not None:
            self.value_policy.refresh(page)

    def describe(self, handle: PageHandle, kind: PageKind, group: int) -> None:
        page = self.get(handle)
        page.kind, page.group = kind, group
        self._refresh(page)

    def touch(self, handle: PageHandle, step: int) -> None:
        """Record an access without changing the canonical LRU order."""
        page = self.get(handle)
        page.last_use = max(page.last_use, step)
        self._refresh(page)

    def set_bind_role(self, handle: PageHandle, hid: int, hgen: int, role: str) -> None:
        page = self.get(handle)
        if (hid, hgen) not in page.binds:
            raise ValueError("Unknown host binding")
        page.binds[hid, hgen] = role
        self._refresh(page)

    def producer_result(
        self, handle: PageHandle, *, compute: Result, seal: Result
    ) -> None:
        page = self.get(handle)
        page.compute_result, page.seal_result = compute, seal
        self._refresh(page)

    def invalidate(self, handle: PageHandle) -> None:
        page = self.get(handle)
        if page.key is not None and self.index.get(page.key) == handle:
            del self.index[page.key]
        page.valid = False
        if self.value_policy is not None:
            self.value_policy.lost(page)
        self._refresh(page)

    def _recycle(self, page: HostPage) -> None:
        assert self.reclaimable(page)
        handle = page.handle
        if self.value_policy is not None:
            self.value_policy.lost(page)
        if page.key is not None and self.index.get(page.key) == handle:
            del self.index[page.key]
        self._lru.pop(handle.page_id, None)
        self._candidates.discard(handle.page_id)
        self._auto.discard(handle.page_id)
        self._pending_live.discard(handle.page_id)
        self._retiring_live.discard(handle.page_id)
        self.state_counts[page.state] -= 1
        self.state_counts[PageState.FREE] += 1
        self.persisted_pages -= int(self.is_persisted(page))
        self.pages[handle.page_id] = HostPage(
            PageHandle(handle.page_id, handle.epoch + 1)
        )
        self._free.append(handle.page_id)

    def collect(self) -> None:
        work = 0
        for heap in (self._retiring, self._pending):
            while heap and heap[0][0] <= self.watermark:
                entry = heappop(heap)
                work += 1
                if self._current(entry):
                    self._refresh(self.pages[entry[1]])
        # The old full scan appended free IDs in ascending physical order.
        # Preserve that order, including simultaneous rank-prefix retirement.
        for pid in sorted(self._auto):
            page = self.pages[pid]
            work += 1
            if self.reclaimable(page) and page.state != PageState.READY:
                self._recycle(page)
        self.collect_work_max = max(self.collect_work_max, work)

    def is_persisted(self, page: HostPage) -> bool:
        return page.persisted_epoch == page.handle.epoch

    def net_headroom(self, pending: int = 0) -> int:
        """Reclaimable pages left after R_host and this step's allocations."""
        return self.available() - self.committed_reserve() - pending

    def reserve(
        self,
        *,
        loading_key: OffloadKey | None = None,
        pending: int = 1,
        writeback: bool = True,
    ) -> PageHandle | None:
        """Allocate one page. ``pending`` counts this and the caller's later
        reservations of the same batch, so a pre-eviction write never takes a
        page that the already-admitted batch still needs."""
        if loading_key is not None and loading_key in self.index:
            return None
        self.collect()
        if not self._free and self.value_policy is not None:
            victim = self.value_policy.select(pending, writeback)
            if victim is not None:
                if self.persistence is not None:
                    self.persistence.on_evict(victim)
                self._recycle(victim)
            if not self._free:
                return None
        if not self._free:
            hooks = self.persistence if writeback else None
            headroom: int | None = None
            for page_id in list(self._lru):
                page = self.pages[page_id]
                if not self.reclaimable(page):
                    continue
                if hooks is not None:
                    if headroom is None:
                        headroom = self.net_headroom(pending)
                    if hooks.try_writeback(page, headroom):
                        # The write added an IO owner: no longer reclaimable.
                        headroom -= 1
                        continue
                    hooks.on_evict(page)
                elif self.persistence is not None:
                    self.persistence.on_evict(page)
                self._recycle(page)
                break
        if not self._free:
            return None
        page = self.pages[self._free.pop()]
        self._set_state(
            page, PageState.PRIVATE if loading_key is None else PageState.LOADING
        )
        page.key = loading_key
        if loading_key is not None:
            page.group = get_offload_group_idx(loading_key)
            page.kind = self.kind_for_key(loading_key)
            self.index[loading_key] = page.handle
            self._lru[page.handle.page_id] = None
        self._refresh(page)
        return page.handle

    def bind(self, handle: PageHandle, hid: int, hgen: int, step: int = 0) -> None:
        page = self.get(handle)
        page.binds[hid, hgen] = "inuse"
        ledger = self.reservation_ledger
        if ledger is not None and handle in ledger.main_pending:
            ledger.release_temporary(ledger.main_pending.pop(handle))
        if page.state == PageState.PRIVATE and page.writer is None:
            page.writer = (hid, hgen, step)
        self._refresh(page)

    def unbind(self, handle: PageHandle, hid: int, hgen: int, last_use: int) -> None:
        page = self.get(handle)
        if (hid, hgen) not in page.binds:
            return
        del page.binds[hid, hgen]
        page.last_use = max(page.last_use, last_use)
        page.retirements.append(Retirement(hid, hgen, handle, page.last_use))
        self._refresh(page)
        self.collect()

    def pin(self, handle: PageHandle, lease_id: int) -> None:
        page = self.get(handle)
        page.leases.add(lease_id)
        self._lru_clock += 1
        page.lru_clock = self._lru_clock
        self._refresh(page)
        if handle.page_id in self._lru:
            self._lru.move_to_end(handle.page_id)

    def release(self, handle: PageHandle, lease_id: int) -> None:
        page = self.get(handle)
        page.leases.remove(lease_id)
        self._refresh(page)
        self.collect()

    def start_io(self, handle: PageHandle, job_id: int) -> None:
        page = self.get(handle)
        page.io.add(job_id)
        self._refresh(page)

    def finish_io(self, handle: PageHandle, job_id: int) -> None:
        page = self.get(handle)
        page.io.remove(job_id)
        self._refresh(page)
        self.collect()

    def publish(self, handle: PageHandle, key: OffloadKey) -> bool:
        page = self.get(handle)
        if not page.valid:
            return False
        if page.state in (PageState.READY, PageState.SHADOW):
            return page.state == PageState.READY
        if page.compute_result != Result.OK or page.seal_result != Result.OK:
            raise ValueError("Host publication requires successful compute and seal")
        canonical = self.index.get(key)
        page.key = key
        page.group = get_offload_group_idx(key)
        self._set_state(
            page, PageState.READY if canonical in (None, handle) else PageState.SHADOW
        )
        self._refresh(page)
        if page.state == PageState.READY:
            self.index[key] = handle
            self._lru[handle.page_id] = None
            self._lru_clock += 1
            page.lru_clock = self._lru_clock
            if self.value_policy is not None:
                self.value_policy.published(page)
                self._refresh(page)
            if self.persistence is not None:
                self.persistence.on_published(handle)
        return page.state == PageState.READY

    def complete_load(self, handle: PageHandle, ok: bool) -> bool:
        """Finish an SSD read into a LOADING page; the caller retires its IO.

        A verified read becomes READY and is already persisted. A failed or
        invalidated (reset) read leaves lookup and is recycled after its IO.
        """
        try:
            page = self.get(handle)
        except ValueError:
            return False
        if page.state != PageState.LOADING:
            return False
        if ok and page.valid:
            self._set_state(page, PageState.READY)
            page.compute_result = page.seal_result = Result.OK
            page.persisted_epoch = handle.epoch
            self.persisted_pages += 1
            self._refresh(page)
            if self.value_policy is not None:
                self.value_policy.published(page)
            return True
        if page.key is not None and self.index.get(page.key) == handle:
            del self.index[page.key]
        self._lru.pop(handle.page_id, None)
        page.valid = False
        self._refresh(page)
        return False

    def mark_persisted(self, handle: PageHandle) -> bool:
        """Record a completed SSD store of exactly this incarnation."""
        try:
            page = self.get(handle)
        except ValueError:
            return False
        if not page.valid or page.state != PageState.READY:
            return False
        if not self.is_persisted(page):
            page.persisted_epoch = handle.epoch
            self.persisted_pages += 1
            self._refresh(page)
            if self.value_policy is not None:
                self.value_policy.persisted(page)
        return True

    def unpersist(self, handle: PageHandle) -> None:
        page = self.get(handle)
        if self.is_persisted(page):
            self.persisted_pages -= 1
            page.persisted_epoch = None
            self._refresh(page)

    def open_seal(
        self, handle: PageHandle, hid: int, hgen: int, key: OffloadKey, step: int
    ) -> SealTicket:
        page = self.get(handle)
        if page.ticket is not None or page.state != PageState.PRIVATE:
            raise ValueError("Only an unticketed private page can be sealed")
        if (hid, hgen) not in page.binds or not page.valid:
            raise ValueError("Seal writer has no valid binding")
        if page.writer is not None and page.writer[:2] != (hid, hgen):
            raise ValueError("Seal belongs to another producer")
        if page.compute_result == Result.FAILED:
            raise ValueError("A failed producer cannot retry seal")
        self._next_ticket += 1
        ticket = SealTicket(self._next_ticket, hid, hgen, handle, key, step)
        page.ticket = ticket
        # Completion belongs to this ticket, never a previous failed attempt.
        page.completed_ranks = page.failed_ranks = 0
        page.compute_result = page.seal_result = Result.PENDING
        page.writer = (hid, hgen, step)
        page.last_use = max(page.last_use, step)
        self._refresh(page)
        return ticket

    def seal_result(
        self,
        ticket: SealTicket,
        completed: int,
        failed: int,
        compute_failed: int = 0,
        dma_bytes: dict[int, int] | None = None,
    ) -> bool:
        try:
            page = self.get(ticket.handle)
        except ValueError:
            return False
        if page.ticket != ticket:
            return False
        if (completed | failed | compute_failed) & ~self.all_ranks:
            raise ValueError("Unknown worker rank in seal receipt")
        if (failed | compute_failed) & ~completed:
            raise ValueError("Failure must have a retired rank receipt")
        if dma_bytes is not None:
            if any(
                not 0 <= rank < self.num_workers or nbytes < 0
                for rank, nbytes in dma_bytes.items()
            ):
                raise ValueError("Invalid seal DMA bytes")
            fresh = completed & ~page.completed_ranks & ~failed
            self.dma_bytes["dma_d2h_main"] += sum(
                nbytes for rank, nbytes in dma_bytes.items() if fresh & (1 << rank)
            )
        page.completed_ranks |= completed
        page.failed_ranks |= failed
        if compute_failed:
            page.compute_result = Result.FAILED
        if page.completed_ranks != self.all_ranks:
            return False
        page.ticket = None
        if page.compute_result != Result.FAILED:
            page.compute_result = Result.OK
        page.seal_result = Result.FAILED if page.failed_ranks else Result.OK
        self._refresh(page)
        if page.compute_result == Result.OK and page.seal_result == Result.OK:
            self.publish(ticket.handle, ticket.key)
        self.collect()
        return True

    def complete_step(self, step: int, ranks: int) -> None:
        if step <= self.watermark:
            return
        if ranks & ~self.all_ranks:
            raise ValueError("Unknown rank in step receipt")
        self._steps[step] = self._steps.get(step, 0) | ranks
        while self._steps.get(self.watermark + 1) == self.all_ranks:
            self.watermark += 1
            del self._steps[self.watermark]
        self.collect()

    def reset(self) -> None:
        self.index.clear()
        self._lru.clear()
        self.ssd_epoch += 1
        if self.persistence is not None:
            self.persistence.on_reset()
        if self.value_policy is not None:
            self.value_policy.reset()
        for page in self.pages[self._first :]:
            # unbind may collect another private page from this snapshot.
            if self.pages[page.handle.page_id] is not page:
                continue
            if page.state != PageState.FREE:
                page.valid = False
                page.key = None
                self._set_state(page, PageState.PRIVATE)
                self._refresh(page)
                for hid, hgen in list(page.binds):
                    self.unbind(page.handle, hid, hgen, page.last_use)
        self.collect()
