# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Atomic physical frame reservations for Host KV."""

from collections import Counter
from collections.abc import Callable

from vllm.v1.kv_offload.host.page_table import HostPageTable, PageHandle


class HostReservationLedger:
    def __init__(self, table: HostPageTable, reclaim_cached: Callable[[], bool]):
        self.table = table
        self.reclaim_cached = reclaim_cached
        self.events: Counter[str] = Counter()
        self.main_pending: dict[PageHandle, list[PageHandle]] = {}
        table.reservation_ledger = self

    def reserve_frames(
        self,
        n: int,
        purpose: str,
        *,
        reserved: bool = False,
        pending: int | None = None,
        loading_key=None,
        writeback: bool = True,
    ) -> list[PageHandle] | None:
        """Reserve an atomic batch, immediately protecting every fresh frame.

        Caller transfers the temporary owner before returning to the event
        loop. Background purposes may consume only net surplus. The MAIN
        allocator has already passed foreground admission when reserved=True.
        """
        if n < 0:
            raise ValueError("Negative frame count")
        pending = n if pending is None else pending
        if pending < n:
            raise ValueError("Pending demand cannot be smaller than this batch")
        if not reserved:
            while self.table.net_headroom() < n and self.reclaim_cached():
                pass
            if self.table.net_headroom() < n:
                self.events[f"reserve_rejected_{purpose}"] += 1
                return None
        if loading_key is not None and n != 1:
            raise ValueError("One loading identity requires one frame")
        handles: list[PageHandle] = []
        for i in range(n):
            handle = self.table.reserve(
                pending=pending - i, loading_key=loading_key, writeback=writeback
            )
            while handle is None and self.reclaim_cached():
                handle = self.table.reserve(
                    pending=pending - i, loading_key=loading_key, writeback=writeback
                )
            if handle is None:
                for held in handles:
                    self.table.finish_io(held, id(handles))
                self.events[f"reserve_failed_{purpose}"] += 1
                return None
            self.table.start_io(handle, id(handles))
            handles.append(handle)
        return handles

    def release_temporary(self, handles: list[PageHandle]) -> None:
        for handle in handles:
            self.table.finish_io(handle, id(handles))

    def snapshot(self) -> dict[str, int]:
        return {
            "host_committed": self.table.committed_reserve(),
            "host_net_surplus": self.table.net_headroom(),
        }
