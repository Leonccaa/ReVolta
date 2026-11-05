# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter, OrderedDict
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Literal

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}

# Why a stored CPU-tier block left the cache (observability only).
# "capacity" and "superseded" are policy evictions under space pressure;
# together with "dependency" they equal ``num_evicted_blocks`` and removal events.
# "superseded" marks blocks the scheduler had demoted as superseded states.
# "reset" counts blocks dropped by ``reset_cache`` (no removal event).
EVICTION_REASONS: tuple[str, ...] = ("capacity", "superseded", "reset", "dependency")


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: str = CPULoadStoreSpec.medium()
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size

        # Cumulative counters for observability (blocks).
        self.num_stored_blocks: int = 0
        self.num_evicted_blocks: int = 0
        # Read-only bookkeeping: evicted blocks by reason, and the cached keys
        # currently demoted (only for policies that implement demote). Never
        # consulted by any admission or eviction decision.
        self.evicted_blocks_by_reason: Counter[str] = Counter()
        self._tracks_demotion: bool = (
            type(self._policy).demote is not CachePolicy.demote
        )
        self._demoted: set[OffloadKey] = set()
        # Optional resident-only dependency bookkeeping, installed by scheduler.
        self.cache_observer: Callable[[OffloadKey, bool], None] | None = None

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return False
        if not block.is_ready:
            return None  # write in-flight; caller should retry
        return True

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys)
        if self._demoted:
            self._demoted.difference_update(keys)

    def demote(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.demote(keys)
        if self._tracks_demotion:
            self._demoted.update(k for k in keys if self._policy.get(k) is not None)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self.counts is not None:
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None
            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)
            self.num_evicted_blocks += len(evicted)
            self._record_evictions(to_evict)
            if self.cache_observer is not None:
                for key in to_evict:
                    self.cache_observer(key, False)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
            if self.cache_observer is not None:
                self.cache_observer(key, True)
        self.num_stored_blocks += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._policy.remove(key)
                    self._free_block(block)
                    self._demoted.discard(key)
                    if self.cache_observer is not None:
                        self.cache_observer(key, False)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    def reset_cache(self) -> None:
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        num_cached = self._num_allocated_blocks - len(self._free_list)
        if num_cached:
            self.evicted_blocks_by_reason["reset"] += num_cached
        self._demoted.clear()
        if self.cache_observer is not None:
            # The observer owns no nonresident entries; notify before clear.
            for key in self.resident_keys():
                self.cache_observer(key, False)
        self._policy.clear()

        self._free_list.clear()
        self._num_allocated_blocks = 0

    def is_ready_resident(self, key: OffloadKey) -> bool:
        """Resident and readable; no recency, admission-counter or ref change."""
        block = self._policy.get(key)
        return block is not None and block.is_ready

    def is_demoted(self, key: OffloadKey) -> bool:
        """Demoted (superseded) and not touched since; read-only."""
        return key in self._demoted

    def resident_keys(self) -> list[OffloadKey]:
        """Snapshot without touching recency or lookup/admission counters."""
        if isinstance(self._policy, LRUCachePolicy):
            return list(self._policy.blocks)
        assert isinstance(self._policy, ARCCachePolicy)
        return list(self._policy.t1) + list(self._policy.t2)

    def remove_dependency_blocks(
        self, keys: Collection[OffloadKey]
    ) -> list[OffloadKey]:
        """Remove proven dead blocks, retaining all pinned or unfinished slots."""
        removed = []
        for key in keys:
            block = self._policy.get(key)
            if block is None or block.ref_cnt != 0:
                continue
            self._policy.remove(key)
            self._free_block(block)
            self._demoted.discard(key)
            if self.cache_observer is not None:
                self.cache_observer(key, False)
            removed.append(key)
        if removed:
            self.num_evicted_blocks += len(removed)
            self.evicted_blocks_by_reason["dependency"] += len(removed)
            if self.events is not None:
                self.events.append(
                    OffloadingEvent(keys=removed, medium=self.medium, removed=True)
                )
        return removed

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def _record_evictions(self, keys: list[OffloadKey]) -> None:
        superseded = 0
        if self._demoted:
            for key in keys:
                if key in self._demoted:
                    self._demoted.discard(key)
                    superseded += 1
        if superseded:
            self.evicted_blocks_by_reason["superseded"] += superseded
        if len(keys) > superseded:
            self.evicted_blocks_by_reason["capacity"] += len(keys) - superseded


class GroupedCPUOffloadingManager(OffloadingManager):
    """Independent group pools whose block IDs address disjoint CPU tensors.

    A key only stores one KV group. Giving each group its own pool avoids
    charging small groups for the largest shared GPU tensor layout. IDs may
    repeat across groups; the GPU transfer spec carries the original group ID.
    """

    def __init__(self, managers: Mapping[int, OffloadingManager]):
        self.managers = managers

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        contexts = [m.on_new_request(req_context) for m in self.managers.values()]
        return RequestOffloadingContext(
            policy=(
                OffloadPolicy.REQUEST_LEVEL
                if any(c.policy == OffloadPolicy.REQUEST_LEVEL for c in contexts)
                else OffloadPolicy.BLOCK_LEVEL
            )
        )

    def on_request_finished(self, req_context: ReqContext) -> None:
        for manager in self.managers.values():
            manager.on_request_finished(req_context)

    @property
    def state_latest_only(self) -> bool:
        """Any group's tiering manager has a state_latest_only tier."""
        return any(
            getattr(m, "state_latest_only", False) is True
            for m in self.managers.values()
        )

    def drop_pending_secondary_writes(self, keys: Collection[OffloadKey]) -> int:
        dropped = 0
        for group_idx, group_keys in self._partition(keys).items():
            drop = getattr(
                self.managers[group_idx], "drop_pending_secondary_writes", None
            )
            if drop is not None:
                dropped += drop(group_keys)
        return dropped

    def delayed_write_stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for group_idx, manager in self.managers.items():
            stats = getattr(manager, "delayed_write_stats", None)
            for tier, values in (stats() if stats is not None else {}).items():
                out[f"{tier}[group {group_idx}]"] = values
        return out

    def shutdown(self) -> None:
        for manager in self.managers.values():
            manager.shutdown()

    def _partition(self, keys: Collection[OffloadKey]) -> dict[int, list[OffloadKey]]:
        groups: dict[int, list[OffloadKey]] = {}
        for key in keys:
            group_idx = get_offload_group_idx(key)
            assert group_idx in self.managers
            groups.setdefault(group_idx, []).append(key)
        return groups

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        return self.managers[get_offload_group_idx(key)].lookup(key, req_context)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        for group_idx, group_keys in self._partition(keys).items():
            self.managers[group_idx].touch(group_keys, req_context)

    def demote(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        for group_idx, group_keys in self._partition(keys).items():
            self.managers[group_idx].demote(group_keys, req_context)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> CPULoadStoreSpec:
        block_ids: dict[OffloadKey, int] = {}
        for group_idx, group_keys in self._partition(keys).items():
            spec = self.managers[group_idx].prepare_load(group_keys, req_context)
            assert isinstance(spec, CPULoadStoreSpec)
            block_ids.update(zip(group_keys, map(int, spec.block_ids)))
        return CPULoadStoreSpec([block_ids[key] for key in keys])

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        for group_idx, group_keys in self._partition(keys).items():
            self.managers[group_idx].complete_load(group_keys, req_context)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        block_ids: dict[OffloadKey, int] = {}
        evicted_keys: list[OffloadKey] = []
        for group_idx, group_keys in self._partition(keys).items():
            output = self.managers[group_idx].prepare_store(group_keys, req_context)
            if output is None:
                # The scheduler advances every group's store cursor on success.
                # Release only this call's new reservations so all groups retry.
                # Already performed evictions are not undone; pre-existing
                # in-flight stores must not be cancelled.
                self.complete_store(list(block_ids), req_context, success=False)
                return None
            assert isinstance(output.store_spec, CPULoadStoreSpec)
            block_ids.update(
                zip(output.keys_to_store, map(int, output.store_spec.block_ids))
            )
            evicted_keys.extend(output.evicted_keys)
        stored = [key for key in keys if key in block_ids]
        return PrepareStoreOutput(
            keys_to_store=stored,
            store_spec=CPULoadStoreSpec([block_ids[key] for key in stored]),
            evicted_keys=evicted_keys,
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        for group_idx, group_keys in self._partition(keys).items():
            self.managers[group_idx].complete_store(group_keys, req_context, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        for manager in self.managers.values():
            yield from manager.take_events()

    def reset_cache(self) -> None:
        for manager in self.managers.values():
            manager.reset_cache()
