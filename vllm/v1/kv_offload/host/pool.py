# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix metadata over caller-owned host KV pages, with no payload allocation.

The controller thread owns this API. The I/O threads only access leased bytes.
host KV pool/host-history integration must delegate slot ownership to this adapter (or
implement the same lease
contract in their allocator); two allocators must never own the same buffer.
Only full, immutable, host-visible pages may be published under prefix hashes.
Active requests, DMA and SSD jobs all retain leases. Unreferenced pages remain
as RAM prefixes until LRU eviction. Mutations require a new reservation/COW.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey, get_offload_group_idx
from vllm.v1.kv_offload.host.page_table import HostPageTable, PageHandle, Result


@dataclass(eq=False)
class HostKVPage:
    """A single-use lease. Its bytes must not be retained after release."""

    key: OffloadKey
    handle: PageHandle
    _pool: "HostKVPool"
    _writing: bool
    _closed: bool = False

    @property
    def block_id(self) -> int:
        return self.handle.page_id

    @property
    def view(self) -> memoryview:
        self._pool._check_lease(self)
        start = self.block_id * self._pool.page_bytes
        view = self._pool.view.cast("B")[start : start + self._pool.page_bytes]
        return view if self._writing else view.toreadonly()

    def release(self) -> None:
        self._pool.release(self)


@dataclass(eq=False)
class HostPageLease:
    """Scheduler lease over an existing shared-region physical page."""

    key: OffloadKey
    handle: PageHandle
    table: HostPageTable
    _closed: bool = False

    def release(self) -> None:
        if not self._closed:
            self.table.release(self.handle, id(self))
            self._closed = True


@dataclass
class HostKVPrefix:
    """This group's contiguous prefix, pinned until release; no global hit.

    complete-prefix contract intersects this absolute length with index/state/draft
    availability
    across every rank, then commits its own complete-prefix transaction.
    """

    group_idx: int
    available_tokens: int
    pages: tuple[HostKVPage | HostPageLease, ...]
    start_block: int = 0

    @property
    def entries(self) -> tuple[tuple[int, HostKVPage | HostPageLease], ...]:
        return tuple(enumerate(self.pages, self.start_block))

    def release(self) -> None:
        for page in self.pages:
            page.release()


class HostKVPool:
    """Borrow a fixed C-contiguous CPU buffer shaped [slots, ...].

    The buffer already contains the authoritative KV storage. This class only
    delegates physical handles, leases and LRU to HostPageTable. It
    neither copies into another CPU pool nor allocates a shared offload region.
    All rows contain the same main-KV group, including every owned layer.
    """

    def __init__(
        self,
        view: memoryview,
        *,
        group_idx: int,
        block_tokens: int,
        table: HostPageTable | None = None,
    ):
        if (
            view.readonly
            or not view.c_contiguous
            or view.ndim < 2
            or view.shape is None
            or view.shape[0] < 1
        ):
            raise ValueError("Host KV requires a writable contiguous CPU row buffer")
        if type(group_idx) is not int or not 0 <= group_idx < 2**32:
            raise ValueError("Invalid host KV group index")
        if type(block_tokens) is not int or block_tokens <= 0:
            raise ValueError("block_tokens must be a positive global token span")
        self.view = view
        self.group_idx = group_idx
        self.block_tokens = block_tokens
        self.num_blocks = view.shape[0]
        self.page_bytes = view.nbytes // self.num_blocks
        if not self.page_bytes:
            raise ValueError("Host KV pages must not be empty")
        if table is not None and len(table.pages) != self.num_blocks:
            raise ValueError("Host view and physical page table differ in size")
        self.table = (
            table
            if table is not None
            else HostPageTable(self.num_blocks, zero_page=False)
        )
        self._leases: set[HostKVPage] = set()

    def _check_key(self, key: OffloadKey) -> None:
        if len(key) <= 4 or get_offload_group_idx(key) != self.group_idx:
            raise ValueError("Key does not identify this host KV group")

    def _check_lease(self, page: HostKVPage) -> None:
        if page._pool is not self or page not in self._leases:
            raise ValueError("Foreign or released host KV lease")
        self.table.get(page.handle)

    def _lease(self, key: OffloadKey, handle: PageHandle, writing: bool) -> HostKVPage:
        page = HostKVPage(key, handle, self, writing)
        self._leases.add(page)
        self.table.pin(handle, id(page))
        return page

    def lookup(self, key: OffloadKey) -> PageHandle | None:
        self._check_key(key)
        return self.table.lookup(key)

    def is_ready(self, handle: PageHandle | None) -> bool:
        """Side-effect-free readiness for the complete-prefix contract residency
        probe."""
        if handle is None:
            return False
        try:
            return self.table.is_ready(handle)
        except ValueError:
            return False

    def reserve(self, key: OffloadKey) -> HostKVPage | None:
        """Reserve an unpublished row; None means resident, in flight or full."""
        if self.lookup(key) is not None:
            return None
        handle = self.table.reserve(loading_key=key)
        if handle is None:
            return None
        return self._lease(key, handle, True)

    def publish(self, page: HostKVPage) -> None:
        """After the producer's host-visible fence, make an immutable read lease.

        The producer must have filled the whole row, including deterministic
        padding. Publishing partial/speculative/uncommitted bytes is forbidden.
        """
        self._check_lease(page)
        if not page._writing:
            raise ValueError("Host KV page is already published")
        self.table.producer_result(page.handle, compute=Result.OK, seal=Result.OK)
        self.table.publish(page.handle, page.key)
        page._writing = False

    def pin(self, handle: PageHandle | None) -> HostKVPage | None:
        if handle is None or not self.is_ready(handle):
            return None
        key = self.table.get(handle).key
        assert key is not None
        return self._lease(key, handle, False)

    def release(self, page: HostKVPage) -> None:
        """Abort unpublished bytes, or leave a published row cached in place."""
        if page._pool is self and page._closed:
            return
        self._check_lease(page)
        if page._writing:
            self.table.invalidate(page.handle)
        self.table.release(page.handle, id(page))
        page._closed = True
        self._leases.remove(page)

    def acquire_prefix(self, keys: Sequence[OffloadKey]) -> HostKVPrefix:
        """Keys are in the scheduler's prefix-chain order, starting at token 0."""
        # Validate before pinning so a mixed-group caller cannot leak references.
        for key in keys:
            self._check_key(key)
        pages = []
        try:
            for key in keys:
                page = self.pin(self.lookup(key))
                if page is None:
                    break
                pages.append(page)
        except Exception:
            for page in pages:
                self.release(page)
            raise
        return HostKVPrefix(
            self.group_idx, len(pages) * self.block_tokens, tuple(pages)
        )
