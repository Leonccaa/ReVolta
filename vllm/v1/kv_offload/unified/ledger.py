# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unique-frame reservations and request build-credit retirement.

Unused credit is a reservation. Consumed credit is represented by physical
owners, never a second reservation. A finished request's consumed credit is
DRAINING until its last DMA/owner retires.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain

from vllm.v1.kv_offload.host.page_table import HostPageTable, PageHandle


@dataclass
class BuildCredit:
    limit: int
    unused: int
    owned: int = 0
    draining: bool = False
    generation: int = 0


class HostReservationLedger:
    def __init__(self, table: HostPageTable, reclaim_cached: Callable[[], bool]):
        self.table = table
        self.reclaim_cached = reclaim_cached
        self.requests: dict[str, BuildCredit] = {}
        self.draining: dict[tuple[str, int], BuildCredit] = {}
        self._generation = 0
        self.events: Counter[str] = Counter()
        self.build_frames_per_request = 0
        self.charges: dict[PageHandle, tuple[str, int]] = {}
        self.main_pending: dict[PageHandle, list[PageHandle]] = {}
        self._external = table.committed_reserve
        table.committed_reserve = self.committed
        table.reservation_ledger = self

    def committed(self) -> int:
        return self._external() + sum(c.unused for c in self.requests.values())

    def admit(self, req_id: str, frames: int) -> bool:
        prior = self.requests.get(req_id)
        if prior is not None and not prior.draining:
            return True
        if frames < 0:
            raise ValueError("Negative build credit")
        # Retiring cached bindings changes no data and makes the same frames
        # available to every kind. It is safe before deciding admission.
        while self.table.net_headroom() < frames and self.reclaim_cached():
            pass
        if self.table.net_headroom() < frames:
            self.events["build_credit_rejected"] += 1
            return False
        if prior is not None:
            self.draining[req_id, prior.generation] = prior
        self._generation += 1
        self.requests[req_id] = BuildCredit(frames, frames, generation=self._generation)
        return True

    def consume(self, req_id: str, cost: int) -> bool:
        credit = self.requests.get(req_id)
        if credit is None or credit.draining or cost > credit.unused:
            self.events["build_budget_rejected"] += 1
            return False
        if cost < 0:
            raise ValueError("Negative build cost")
        credit.unused -= cost
        credit.owned += cost
        return True

    def retire(self, req_id: str, cost: int, generation: int | None = None) -> None:
        credit = self.requests.get(req_id)
        current = credit is not None and (
            generation is None or credit.generation == generation
        )
        if not current:
            assert generation is not None
            credit = self.draining[req_id, generation]
        assert credit is not None
        if not 0 <= cost <= credit.owned:
            raise ValueError("Build retirement exceeds its owned cost")
        credit.owned -= cost
        if credit.draining:
            if credit.owned == 0:
                if current:
                    del self.requests[req_id]
                else:
                    del self.draining[req_id, credit.generation]
        else:
            credit.unused += cost

    def charge(self, req_id: str, handles: list[PageHandle]) -> None:
        for h in handles:
            if h in self.charges:
                raise ValueError("A physical frame cannot be charged twice")
            self.charges[h] = (req_id, self.requests[req_id].generation)

    def owner_changed(self, handle: PageHandle) -> None:
        # A subsequent build/receive can share a charged frame at zero
        # incremental cost. Keep its original payment until the last owner
        # retires; otherwise returning unused credit would double reserve it.
        admission = self.charges.get(handle)
        if admission is not None and self.table.reclaimable(self.table.get(handle)):
            del self.charges[handle]
            req_id, generation = admission
            self.retire(req_id, 1, generation)

    def finish(self, req_id: str) -> None:
        credit = self.requests.get(req_id)
        if credit is None:
            return
        credit.unused = 0
        credit.draining = True
        if not credit.owned:
            del self.requests[req_id]

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
        loop. Background purposes may consume only net surplus. Foreground
        credit must be consumed before calling with reserved=True.
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
            "build_reserved": sum(c.unused for c in self.requests.values()),
            "build_owned": sum(
                c.owned for c in self.requests.values() if not c.draining
            ),
            "build_draining": sum(
                c.owned
                for c in chain(self.requests.values(), self.draining.values())
                if c.draining
            ),
            "host_committed": self.committed(),
            "host_net_surplus": self.table.net_headroom(),
        }
