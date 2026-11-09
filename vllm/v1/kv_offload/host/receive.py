# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source identities and rank retirement for one external prefix receive."""

from dataclasses import dataclass
from enum import Enum

from vllm.v1.kv_offload.base import LoadStoreSpec
from vllm.v1.kv_offload.host.page_table import PageHandle


class ReceiveSource(str, Enum):
    NEW_LOAD = "new_load"
    REUSE_LOCAL = "reuse_local"


@dataclass(frozen=True)
class ReceiveEntry:
    logical_index: int
    source: ReceiveSource
    segment: int
    hid: int
    hgen: int
    handle: PageHandle
    # Paired with the request's block table. KVCacheManager.free returns
    # exactly this reference, including when another request shares the ID.
    acquired_ref: bool = True


@dataclass
class ReceiveTxn:
    attempt_id: int
    req_id: str
    step: int
    entries: tuple[ReceiveEntry, ...]
    completed_ranks: int = 0
    failed_ranks: int = 0
    cancelled: bool = False

    def complete(self, completed: int, failed: int, all_ranks: int) -> bool:
        if (completed | failed) & ~all_ranks or failed & ~completed:
            raise ValueError("Invalid receive rank receipt")
        self.completed_ranks |= completed
        self.failed_ranks |= failed
        return self.completed_ranks == all_ranks

    @property
    def new_loads(self) -> tuple[ReceiveEntry, ...]:
        return tuple(e for e in self.entries if e.source == ReceiveSource.NEW_LOAD)


@dataclass
class HostReceiveSpec(LoadStoreSpec):
    """Join ordinary STATE/WINDOW transfers with logical index H2D.

    The ordinary source retains its backend ABI. G0 uses only the physical
    handles in the transaction, independently of canonical transfer tensors.
    """

    source: LoadStoreSpec
    txn: ReceiveTxn

    @staticmethod
    def medium() -> str:
        return "CPU"
