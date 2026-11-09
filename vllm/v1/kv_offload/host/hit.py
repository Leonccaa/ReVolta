# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-aware host hits and leases on the scheduler thread."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_hit_contract import (
    ChainHashFn,
    GroupHitRule,
    HitNeed,
    KVHitContract,
    chain_hash_from_request,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
)
from vllm.v1.kv_offload.base import OffloadKey, get_offload_block_hash, make_offload_key
from vllm.v1.kv_offload.host.page_table import PageHandle, PageState, Result
from vllm.v1.kv_offload.host.pool import HostKVPage, HostKVPrefix, HostPageLease
from vllm.v1.kv_offload.host.receive import ReceiveEntry, ReceiveSource, ReceiveTxn

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.v1.core.history_block_pool import HistoryBlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_offload.host.ssd import HostKVPrefixCache
    from vllm.v1.request import Request


class HostKVHitManager:
    """Local bindings attest G0 reuse. Only external NEW_LOAD pages get leases."""

    def __init__(
        self,
        cache: "HostKVPrefixCache | None",
        gpu_binding_ready: Callable[[int, OffloadKey], bool] | None,
        *,
        history_pool: "HistoryBlockPool | None" = None,
        block_tokens: int = 0,
        segment: int = 0,
    ):
        self.cache = cache
        self.history_pool = history_pool
        self.segment = segment
        if cache is not None:
            group_id, block_tokens = cache.pool.group_idx, cache.pool.block_tokens
        else:
            assert history_pool is not None
            group_id = history_pool.host_group_id
        self.rule = GroupHitRule(group_id, HitNeed.PREFIX, block_tokens)
        self._gpu_binding_ready = gpu_binding_ready
        self.local_only = False
        self._pending: dict[str, HostKVPrefix] = {}
        self._active: dict[str, HostKVPrefix] = {}
        self._reuse: dict[str, tuple[ReceiveEntry, ...]] = {}
        self._active_reuse: dict[str, tuple[ReceiveEntry, ...]] = {}

    @classmethod
    def for_history(cls, manager: "KVCacheManager") -> "HostKVHitManager":
        from vllm.v1.core.history_block_pool import HistoryBlockPool

        pool = manager.block_pool
        assert isinstance(pool, HistoryBlockPool)
        group_id = pool.host_group_id
        block_tokens = manager.coordinator.single_type_managers[group_id].block_size
        segments = manager.kv_cache_config.block_segments
        hits = cls(
            None,
            None,
            history_pool=pool,
            block_tokens=block_tokens,
            segment=segments.primary if segments is not None else 0,
        )
        assert manager.hit_contract is not None
        manager.hit_contract.register_local_validator(group_id, hits.local_validator)
        hits.local_only = True
        return hits

    def attach_connector(self, connector: "OffloadingConnectorScheduler") -> None:
        if connector.hit_contract is None:
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
                scheduler,
            )
            from vllm.v1.core.kv_cache_hit_contract import connector_hit_rules

            connector.hit_contract = KVHitContract(
                connector_hit_rules(
                    scheduler._connector_hit_needs(
                        connector.config, connector.kv_cache_config
                    )
                ),
                "offloading-connector",
                connector._hash_block_size,
            )
        connector.hit_contract.register_probe(
            self.rule.group_id, self.probe, block_size=self.rule.block_size
        )
        connector.host_kv_hits = self

    @classmethod
    def attach(
        cls,
        cache: "HostKVPrefixCache",
        gpu_manager: "KVCacheManager",
        connector: "OffloadingConnectorScheduler",
        *,
        gpu_binding_ready: Callable[[int, OffloadKey], bool] | None = None,
    ) -> "HostKVHitManager | None":
        """Attach a caller-owned host cache after validating both contracts.

        Probe-only groups still depend on their host store. A concrete local
        G0 group uses its binding validator and acquires no canonical lease.
        """
        contracts = (gpu_manager.hit_contract, connector.hit_contract)
        if gpu_manager.host_kv_hits is not None or connector.host_kv_hits is not None:
            return None
        for contract in contracts:
            if contract is None or cache.pool.block_tokens % contract.hash_block_size:
                return None
            rule = contract.rules.get(cache.pool.group_idx)
            if rule is not None and (
                rule.need is not HitNeed.PREFIX
                or rule.block_size != cache.pool.block_tokens
            ):
                return None
        hits = cls(cache, gpu_binding_ready)
        local, external = contracts
        assert local is not None and external is not None
        if cache.pool.group_idx in local.rules:
            local.register_local_validator(cache.pool.group_idx, hits.local_validator)
            hits.local_only = True
        else:
            local.register_probe(
                cache.pool.group_idx, hits.probe, block_size=cache.pool.block_tokens
            )
        external.register_probe(
            cache.pool.group_idx, hits.probe, block_size=cache.pool.block_tokens
        )
        gpu_manager.host_kv_hits = connector.host_kv_hits = hits
        return hits

    def local_binding(self, hid: int, key: OffloadKey) -> PageHandle | None:
        pool = self.history_pool
        if pool is None:
            return None
        try:
            handle = pool.bindings[hid]
            page = pool.host_pages.get(handle)
            block = pool.blocks[hid]
            if (
                not block.is_null
                and block.block_hash is not None
                and get_group_id(block.block_hash) == self.rule.group_id
                and get_block_hash(block.block_hash).endswith(
                    get_offload_block_hash(key)
                )
                and (hid, pool.hgen[hid]) in page.binds
                and page.valid
                and page.state in (PageState.PRIVATE, PageState.SHADOW, PageState.READY)
                and page.compute_result == Result.OK
                and page.key in (None, key)
            ):
                return handle
        except (IndexError, KeyError, ValueError):
            pass
        return None

    def local_validator(
        self, group_id: int, index: int, block_hash: BlockHash, gpu_block_id: int | None
    ) -> bool:
        if group_id != self.rule.group_id or gpu_block_id is None:
            return False
        key = make_offload_key(block_hash, group_id)
        if self.history_pool is not None:
            return self.local_binding(gpu_block_id, key) is not None
        try:
            return self._gpu_binding_ready is not None and self._gpu_binding_ready(
                gpu_block_id, key
            )
        except Exception:
            return False

    def probe(
        self, group_id: int, index: int, block_hash: BlockHash, gpu_block_id: int | None
    ) -> bool:
        """External probes accept only canonical READY pages."""
        if group_id != self.rule.group_id:
            return False
        key = make_offload_key(block_hash, group_id)
        if self.cache is not None:
            return self.cache.reader_ready(key)
        assert self.history_pool is not None
        table = self.history_pool.host_pages
        handle = table.lookup(key)
        return handle is not None and table.is_ready(handle)

    def begin_lookup(self, request_id: str) -> None:
        self.release_lookup(request_id)
        if self.cache is not None:
            self.cache.poll()

    def plan_reuse(self, request: "Request", start_tokens: int, hash_size: int) -> int:
        """Find the continuous G0-only extension beyond the complete local L."""
        self._reuse.pop(request.request_id, None)
        pool = self.history_pool
        if pool is None or not pool.enable_caching:
            return start_tokens
        chain_hash = chain_hash_from_request(request.block_hashes, hash_size)
        gpu_hashes = BlockHashListWithBlockSize(
            request.block_hashes, hash_size, self.rule.block_size
        )
        entries = []
        end = start_tokens
        for index in range(
            start_tokens // self.rule.block_size,
            (request.num_tokens - 1) // self.rule.block_size,
        ):
            block_hash = chain_hash(self.rule, index)
            if block_hash is None:
                break
            cached = pool.get_cached_block(gpu_hashes[index], [self.rule.group_id])
            if cached is None:
                break
            hid = cached[0].block_id
            handle = self.local_binding(
                hid, make_offload_key(block_hash, self.rule.group_id)
            )
            if handle is None:
                break
            entries.append(
                ReceiveEntry(
                    index,
                    ReceiveSource.REUSE_LOCAL,
                    self.segment,
                    hid,
                    pool.hgen[hid],
                    handle,
                )
            )
            end = (index + 1) * self.rule.block_size
        self._reuse[request.request_id] = tuple(entries)
        return end

    def pin_hit(
        self,
        request_id: str,
        chain_hash: ChainHashFn,
        end_tokens: int | None = None,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> HostKVPrefix | None:
        """Pin the external slice [start, end), with absolute logical indices.

        Acquire the replacement before releasing a prior lookup. Failed
        extensions release their partial leases and preserve the prior slice.
        The controller owns the returned leases through request release.
        """
        if end is not None:
            if end_tokens is not None and end_tokens != end:
                raise ValueError("Conflicting host prefix endpoints")
            end_tokens = end
        if end_tokens is None or not 0 <= start <= end_tokens:
            raise ValueError("Invalid host prefix slice")
        indices = self.rule.required_blocks(end_tokens, start)
        keys = []
        for index in indices:
            block_hash = chain_hash(self.rule, index)
            if block_hash is None:
                return None
            keys.append(make_offload_key(block_hash, self.rule.group_id))
        if self.cache is not None:
            prefix = self.cache.acquire_prefix(keys)
        else:
            assert self.history_pool is not None
            table = self.history_pool.host_pages
            pages = []
            try:
                for key in keys:
                    handle = table.lookup(key)
                    if handle is None or not table.is_ready(handle):
                        break
                    page = HostPageLease(key, handle, table)
                    table.pin(handle, id(page))
                    pages.append(page)
            except Exception:
                for page in pages:
                    page.release()
                return None
            prefix = HostKVPrefix(self.rule.group_id, 0, tuple(pages))
        if len(prefix.pages) != len(keys):
            prefix.release()
            return None
        prefix.start_block = start // self.rule.block_size
        prefix.available_tokens = (
            prefix.start_block + len(keys)
        ) * self.rule.block_size
        reuse = self._reuse.get(request_id, ())
        self.release_lookup(request_id)
        self._reuse[request_id] = reuse
        self._pending[request_id] = prefix
        return prefix

    def trim_reuse(self, request_id: str, end_tokens: int) -> None:
        self._reuse[request_id] = tuple(
            entry
            for entry in self._reuse.get(request_id, ())
            if (entry.logical_index + 1) * self.rule.block_size <= end_tokens
        )

    def extend_local_blocks(
        self,
        request: "Request",
        groups: tuple[Sequence[KVCacheBlock], ...],
        end_tokens: int,
    ) -> tuple[Sequence[KVCacheBlock], ...] | None:
        pool = self.history_pool
        if pool is None:
            return groups
        chain_hash = chain_hash_from_request(request.block_hashes, pool.hash_block_size)
        result = list(groups)
        blocks = list(groups[self.rule.group_id])
        for entry in self._reuse.get(request.request_id, ()):
            if entry.logical_index * self.rule.block_size >= end_tokens:
                break
            block_hash = chain_hash(self.rule, entry.logical_index)
            if (
                block_hash is None
                or pool.hgen[entry.hid] != entry.hgen
                or self.local_binding(
                    entry.hid, make_offload_key(block_hash, self.rule.group_id)
                )
                != entry.handle
                or entry.logical_index != len(blocks)
            ):
                return None
            blocks.append(pool.blocks[entry.hid])
        result[self.rule.group_id] = blocks
        return tuple(result)

    def commit_lookup(self, request_id: str) -> None:
        prefix = self._pending.pop(request_id, None)
        if prefix is not None:
            previous = self._active.pop(request_id, None)
            self._active[request_id] = prefix
            if previous is not None:
                previous.release()
        reuse = self._reuse.pop(request_id, ())
        if reuse:
            self._active_reuse[request_id] = reuse

    def pinned_pages(
        self, request_id: str
    ) -> tuple[tuple[int, HostKVPage | HostPageLease], ...]:
        """Borrow indexed leases. Callers leave release to this controller."""
        prefix = self._pending.get(request_id) or self._active.get(request_id)
        return () if prefix is None else prefix.entries

    def build_receive(
        self,
        request: "Request",
        groups: tuple[Sequence[KVCacheBlock], ...],
        start: int,
        end: int,
        attempt_id: int,
    ) -> ReceiveTxn:
        pool = self.history_pool
        assert pool is not None
        if start % self.rule.block_size or end % self.rule.block_size:
            # A bound READY row is immutable. Continuing inside that row
            # would require a private copy before the next producer write.
            raise ValueError("Host receives require complete G0 block boundaries")
        borrowed = {
            e.logical_index: e for e in self._active_reuse.get(request.request_id, ())
        }
        pinned = dict(self.pinned_pages(request.request_id))
        chain_hash = chain_hash_from_request(request.block_hashes, pool.hash_block_size)
        entries = []
        for index in self.rule.required_blocks(end, start):
            block = groups[self.rule.group_id][index]
            key_hash = chain_hash(self.rule, index)
            if block.is_null or key_hash is None:
                raise ValueError("Receive has no concrete G0 destination")
            key = make_offload_key(key_hash, self.rule.group_id)
            entry = borrowed.get(index)
            if entry is not None:
                if (
                    block.block_id != entry.hid
                    or pool.hgen[entry.hid] != entry.hgen
                    or self.local_binding(entry.hid, key) != entry.handle
                ):
                    raise ValueError("Receive borrowed binding changed")
            else:
                page = pinned.get(index)
                if (
                    page is None
                    or page.key != key
                    or block.block_hash is not None
                    or not pool.host_pages.is_ready(page.handle)
                ):
                    raise ValueError("Receive lacks a READY leased G0 source")
                entry = ReceiveEntry(
                    index,
                    ReceiveSource.NEW_LOAD,
                    self.segment,
                    block.block_id,
                    pool.hgen[block.block_id],
                    page.handle,
                )
            entries.append(entry)
        txn = ReceiveTxn(attempt_id, request.request_id, pool.step, tuple(entries))
        # Validate the entire plan before changing any binding. Fresh temporary
        # PRIVATE reservations are returned to the same physical allocator.
        for entry in txn.new_loads:
            pool.unbind(entry.hid)
            pool.host_pages.bind(entry.handle, entry.hid, entry.hgen, pool.step)
            pool.bindings[entry.hid] = entry.handle
            pool.binding_updates[entry.hid] = entry.handle.page_id
        return txn

    def start_receive(self, txn: ReceiveTxn) -> None:
        pool = self.history_pool
        assert pool is not None
        for entry in txn.entries:
            identity = (entry.hid, entry.hgen)
            pool.last_use[identity] = max(pool.last_use.get(identity, 0), txn.step)
        for handle in {e.handle for e in txn.new_loads}:
            pool.host_pages.touch(handle, txn.step)
            pool.host_pages.start_io(handle, txn.attempt_id)

    def retire_receive(self, txn: ReceiveTxn) -> None:
        assert self.history_pool is not None
        for handle in {e.handle for e in txn.new_loads}:
            self.history_pool.host_pages.finish_io(handle, txn.attempt_id)

    def release_lookup(self, request_id: str) -> None:
        prefix = self._pending.pop(request_id, None)
        if prefix is not None:
            prefix.release()
        self._reuse.pop(request_id, None)

    def release_uncommitted(self) -> None:
        for request_id in set(self._pending) | set(self._reuse):
            self.release_lookup(request_id)

    def release_request(self, request_id: str) -> None:
        """Called after receive retirement and all request consumers stopped."""
        self.release_lookup(request_id)
        self._active_reuse.pop(request_id, None)
        prefix = self._active.pop(request_id, None)
        if prefix is not None:
            prefix.release()
