# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request leases joining host reader readiness to both complete-prefix contract lookup
paths.

host-history integration explicitly attaches this controller after assigning the host
pool its group
and byte ABI. All methods run on the scheduler thread. Probes do not mutate the
pool. Before returning a hit, each path pins the entire host prefix. Admission
commits those leases, and free releases them after the consumers have stopped.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_hit_contract import ChainHashFn, GroupHitRule, HitNeed
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.host.pool import HostKVPage, HostKVPrefix

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_offload.host.ssd import HostKVPrefixCache


class HostKVHitManager:
    """Pin the common host prefix through lookup, admission and consumption.

    ``gpu_binding_ready`` must check the live GPU block's identity/generation
    against the host key. Without that attestation GPU hits fail closed. The
    connector has no destination block at lookup time; host-history integration must
    bind its newly
    admitted blocks before making the restored prefix runnable (complete-prefix contract
    destination binding).
    """

    def __init__(
        self,
        cache: "HostKVPrefixCache",
        gpu_binding_ready: Callable[[int, OffloadKey], bool] | None,
    ):
        self.cache = cache
        self.rule = GroupHitRule(
            cache.pool.group_idx, HitNeed.PREFIX, cache.pool.block_tokens
        )
        self._gpu_binding_ready = gpu_binding_ready
        self._pending: dict[str, HostKVPrefix] = {}
        self._active: dict[str, HostKVPrefix] = {}

    @classmethod
    def attach(
        cls,
        cache: "HostKVPrefixCache",
        gpu_manager: "KVCacheManager",
        connector: "OffloadingConnectorScheduler",
        *,
        gpu_binding_ready: Callable[[int, OffloadKey], bool] | None = None,
    ) -> "HostKVHitManager | None":
        """Register the same reader-ready probe on both enabled contracts.

        Missing/incompatible contracts leave both paths untouched and return
        None. A probe-only group may be absent from either path's own storage.
        This does not split G0 or alter connector store/load membership.
        """
        contracts = (gpu_manager.hit_contract, connector.hit_contract)
        if gpu_manager.host_kv_hits is not None or connector.host_kv_hits is not None:
            return None
        for contract in contracts:
            if contract is None:
                return None
            if cache.pool.block_tokens % contract.hash_block_size:
                return None
            rule = contract.rules.get(cache.pool.group_idx)
            if rule is not None and (
                rule.need is not HitNeed.PREFIX
                or rule.block_size != cache.pool.block_tokens
            ):
                return None
        hits = cls(cache, gpu_binding_ready)
        for contract in contracts:
            # The validation above covers both contracts before either changes.
            if contract is not None:
                contract.register_probe(
                    cache.pool.group_idx,
                    hits.probe,
                    block_size=cache.pool.block_tokens,
                )
        gpu_manager.host_kv_hits = connector.host_kv_hits = hits
        return hits

    def probe(
        self, group_id: int, index: int, block_hash: BlockHash, gpu_block_id: int | None
    ) -> bool:
        """A file or an unfinished write/read is insufficient for a hit."""
        if group_id != self.rule.group_id:
            return False
        key = make_offload_key(block_hash, group_id)
        if not self.cache.reader_ready(key):
            return False
        if gpu_block_id is None:
            return True
        try:
            return self._gpu_binding_ready is not None and self._gpu_binding_ready(
                gpu_block_id, key
            )
        except Exception:
            # A stale/failed binding lookup is a cache miss.
            return False

    def begin_lookup(self, request_id: str) -> None:
        self.release_lookup(request_id)
        self.cache.poll()

    def pin_hit(
        self, request_id: str, chain_hash: ChainHashFn, end_tokens: int
    ) -> bool:
        """Acquire the entire host prefix before reporting any path's hit.

        Acquire the new leases before releasing an earlier local lookup's
        leases. An external extension therefore cannot expose its head to LRU.
        A failed extension preserves the earlier local hit and releases every
        newly acquired lease. Partial final blocks require the whole host row.
        """
        keys = []
        for index in self.rule.required_blocks(end_tokens):
            block_hash = chain_hash(self.rule, index)
            if block_hash is None:
                return False
            keys.append(make_offload_key(block_hash, self.rule.group_id))
        prefix = self.cache.acquire_prefix(keys)
        if len(prefix.pages) != len(keys):
            prefix.release()
            return False
        self.release_lookup(request_id)
        self._pending[request_id] = prefix
        return True

    def commit_lookup(self, request_id: str) -> None:
        prefix = self._pending.pop(request_id, None)
        if prefix is not None:
            previous = self._active.pop(request_id, None)
            self._active[request_id] = prefix
            if previous is not None:
                previous.release()

    def pinned_pages(self, request_id: str) -> tuple[HostKVPage, ...]:
        """Borrow the pinned chain-order rows for host-history integration's binding
        transaction.

        The controller owns these leases. Callers must not release them or
        retain their views past request free or lookup cancellation.
        """
        prefix = self._pending.get(request_id) or self._active.get(request_id)
        return () if prefix is None else prefix.pages

    def release_lookup(self, request_id: str) -> None:
        prefix = self._pending.pop(request_id, None)
        if prefix is not None:
            prefix.release()

    def release_uncommitted(self) -> None:
        """End-of-step cleanup also covers deferral before allocate_slots."""
        for prefix in self._pending.values():
            prefix.release()
        self._pending.clear()

    def release_request(self, request_id: str) -> None:
        """Call only after active consumers/fences no longer need the prefix."""
        self.release_lookup(request_id)
        prefix = self._active.pop(request_id, None)
        if prefix is not None:
            prefix.release()
