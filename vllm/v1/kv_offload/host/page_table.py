# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-thread ownership of physical host pages.

Hashes name published immutable content. Every owner and completion instead
names a physical incarnation, so invalidating a hash never retires DMA.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum

from vllm.v1.kv_offload.base import OffloadKey


class PageState(str, Enum):
    FREE = "free"
    PRIVATE = "private"
    READY = "ready"
    SHADOW = "shadow"
    LOADING = "loading"


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

    def get(self, handle: PageHandle) -> HostPage:
        if not 0 <= handle.page_id < len(self.pages):
            raise ValueError("Invalid host page")
        page = self.pages[handle.page_id]
        if page.handle != handle or page.state == PageState.FREE:
            raise ValueError("Stale host page epoch")
        return page

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
            and not (page.binds or page.leases or page.io or page.ticket)
            and self._retired(page)
        )

    def available(self) -> int:
        return len(self._free) + sum(
            self.reclaimable(p) and p.state != PageState.FREE
            for p in self.pages[self._first :]
        )

    def _recycle(self, page: HostPage) -> None:
        assert self.reclaimable(page)
        handle = page.handle
        if page.key is not None and self.index.get(page.key) == handle:
            del self.index[page.key]
        self._lru.pop(handle.page_id, None)
        self.pages[handle.page_id] = HostPage(
            PageHandle(handle.page_id, handle.epoch + 1)
        )
        self._free.append(handle.page_id)

    def collect(self) -> None:
        for page in self.pages[self._first :]:
            if page.state in (
                PageState.PRIVATE,
                PageState.SHADOW,
                PageState.LOADING,
            ) and self.reclaimable(page):
                self._recycle(page)

    def reserve(self, *, loading_key: OffloadKey | None = None) -> PageHandle | None:
        if loading_key is not None and loading_key in self.index:
            return None
        self.collect()
        if not self._free:
            for page_id in list(self._lru):
                page = self.pages[page_id]
                if self.reclaimable(page):
                    self._recycle(page)
                    break
        if not self._free:
            return None
        page = self.pages[self._free.pop()]
        page.state = PageState.PRIVATE if loading_key is None else PageState.LOADING
        page.key = loading_key
        if loading_key is not None:
            self.index[loading_key] = page.handle
            self._lru[page.handle.page_id] = None
        return page.handle

    def bind(self, handle: PageHandle, hid: int, hgen: int, step: int = 0) -> None:
        page = self.get(handle)
        page.binds[hid, hgen] = "inuse"
        if page.state == PageState.PRIVATE and page.writer is None:
            page.writer = (hid, hgen, step)

    def unbind(self, handle: PageHandle, hid: int, hgen: int, last_use: int) -> None:
        page = self.get(handle)
        if (hid, hgen) not in page.binds:
            return
        del page.binds[hid, hgen]
        page.last_use = max(page.last_use, last_use)
        page.retirements.append(Retirement(hid, hgen, handle, page.last_use))
        self.collect()

    def pin(self, handle: PageHandle, lease_id: int) -> None:
        self.get(handle).leases.add(lease_id)
        if handle.page_id in self._lru:
            self._lru.move_to_end(handle.page_id)

    def release(self, handle: PageHandle, lease_id: int) -> None:
        self.get(handle).leases.remove(lease_id)
        self.collect()

    def start_io(self, handle: PageHandle, job_id: int) -> None:
        self.get(handle).io.add(job_id)

    def finish_io(self, handle: PageHandle, job_id: int) -> None:
        self.get(handle).io.remove(job_id)
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
        page.state = (
            PageState.READY if canonical in (None, handle) else PageState.SHADOW
        )
        if page.state == PageState.READY:
            self.index[key] = handle
            self._lru[handle.page_id] = None
        return page.state == PageState.READY

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
        self._next_ticket += 1
        ticket = SealTicket(self._next_ticket, hid, hgen, handle, key, step)
        page.ticket = ticket
        page.writer = (hid, hgen, step)
        page.last_use = max(page.last_use, step)
        return ticket

    def seal_result(
        self, ticket: SealTicket, completed: int, failed: int, compute_failed: int = 0
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
        for page in self.pages[self._first :]:
            if page.state != PageState.FREE:
                page.valid = False
                page.key = None
                page.state = PageState.PRIVATE
                for hid, hgen in list(page.binds):
                    self.unbind(page.handle, hid, hgen, page.last_use)
        self.collect()
