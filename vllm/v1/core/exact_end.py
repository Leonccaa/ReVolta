# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local, consuming checkpoints adapted from vLLM #66881 frontier.py/keys.py.

Source: c19296d49934329b0fbb6853759690bc6d6c11d9. State gathers follow
#67045 (d2a3815636ccc9eeedf31574a75e1e2d8afeca37); no partial-hit CoW.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from vllm.utils.hashing import sha256
from vllm.v1.core.history_block_pool import HistoryBlockPool
from vllm.v1.core.kv_cache_hit_contract import chain_hash_from_request
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    KVCacheBlock,
    MambaStateGather,
    generate_block_hash_extra_keys,
    hash_block_tokens,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.kv_offload.host.hit import HostKVHitManager


@dataclass
class ExactEnd:
    length: int
    key: BlockHash
    suffix: tuple[int, ...]
    # Per group: first logical block and blocks with their expected cache keys.
    groups: list[tuple[int, list[tuple[KVCacheBlock, BlockHashWithGroupId]]]]


class ExactEndIndex:
    def __init__(self, coordinator: "KVCacheCoordinator", hash_block_size: int):
        self.coordinator = coordinator
        self.hash_block_size = hash_block_size
        self.state_groups = [
            i
            for i, m in enumerate(coordinator.single_type_managers)
            if isinstance(m, MambaManager)
        ]
        self.index: dict[BlockHash | None, list[ExactEnd]] = {}
        self.pending_gathers: list[MambaStateGather] = []
        self.hits: dict[str, ExactEnd] = {}

    def _anchor(self, request: Request, boundary: int) -> BlockHash | None:
        return (
            request.block_hashes[boundary // self.hash_block_size - 1]
            if boundary
            else None
        )

    def _key(self, request: Request, length: int) -> BlockHash:
        start = length // self.hash_block_size * self.hash_block_size
        extra, _ = generate_block_hash_extra_keys(request, start, length, 0)
        return hash_block_tokens(
            sha256,
            self._anchor(request, start),
            request.all_token_ids[start:length],
            (*(extra or ()), ("revolta-exact-end", length)),
        )

    def publish(self, request: Request) -> None:
        self.hits.pop(request.request_id, None)
        managers = self.coordinator.single_type_managers
        if not self.state_groups or not self.coordinator.enable_caching:
            return
        states = [m for m in managers if isinstance(m, MambaManager)]
        span = states[0].block_size
        c = request.num_computed_tokens
        if (
            not request.is_finished()
            or request.status == RequestStatus.FINISHED_ABORTED
            or request.resumable
            or request.num_in_flight_tokens
            or c != request.num_tokens - 1
            or c % span == 0
            or any(
                m.num_speculative_blocks or m.mamba_cache_mode != "align"
                for m in states
            )
        ):
            return
        key = self._key(request, c + 1)
        groups = []
        dst = []
        for g, manager in enumerate(managers):
            blocks = manager.req_to_blocks[request.request_id]
            if isinstance(manager, MambaManager):
                first = c // manager.block_size
                selected = blocks[
                    (request.state_end - 1) // manager.block_size : (
                        request.state_end - 1
                    )
                    // manager.block_size
                    + 1
                ]
                dst.append(selected[0].block_id)
            elif isinstance(manager, CircularBufferManager):
                first, selected = 0, blocks[:1]
            elif isinstance(manager, SlidingWindowManager):
                first = manager.get_num_skipped_tokens(c) // manager.block_size
                selected = blocks[
                    first : (c + manager.block_size - 1) // manager.block_size
                ]
            else:
                first = c // manager.block_size
                selected = blocks[first : first + 1]
            if not selected or any(b.is_null for b in selected):
                return
            exact_key = make_block_hash_with_group_id(key, g)
            if not isinstance(manager, SlidingWindowManager) and any(
                b.block_hash is not None for b in selected
            ):
                return
            groups.append((first, [(b, b.block_hash or exact_key) for b in selected]))
        for g, (_, held) in enumerate(groups):
            pool = self.coordinator.group_block_pool(g)
            for block, expected in held:
                if block.block_hash is None:
                    pool.cache_under_key(block, expected)
        entry = ExactEnd(
            c + 1, key, tuple(request.all_token_ids[max(0, c - 7) : c + 1]), groups
        )
        self.index.setdefault(self._anchor(request, c // span * span), []).append(entry)
        if request.last_accepted > 1:
            self.pending_gathers.append(
                MambaStateGather(request.last_accepted - 1, dst)
            )

    def extend(
        self,
        request: Request,
        blocks: tuple[Sequence[KVCacheBlock], ...],
        hit: int,
        host_hits: "HostKVHitManager | None",
    ) -> tuple[tuple[Sequence[KVCacheBlock], ...], int]:
        self.hits.pop(request.request_id, None)
        if not self.state_groups:
            return blocks, hit
        managers = self.coordinator.single_type_managers
        span = managers[self.state_groups[0]].block_size
        for boundary in range((request.num_tokens - 1) // span * span, hit - 1, -span):
            entries = self.index.get(self._anchor(request, boundary), [])
            for entry in sorted(entries, key=lambda e: e.length, reverse=True):
                n = entry.length
                if (
                    not hit < n - 1 < request.num_tokens
                    or tuple(request.all_token_ids[max(0, n - 8) : n]) != entry.suffix
                    or self._key(request, n) != entry.key
                ):
                    continue
                result = []
                for g, (first, held) in enumerate(entry.groups):
                    pool = self.coordinator.group_block_pool(g)
                    if any(
                        b.block_hash != expected
                        or pool.cached_block_hash_to_block.get_one_block(expected)
                        is None
                        for b, expected in held
                    ):
                        break
                    if host_hits is not None and g == host_hits.rule.group_id:
                        assert isinstance(pool, HistoryBlockPool)
                        pool.tail_ready(held[-1][0].block_id)
                        if (
                            host_hits.local_binding(
                                held[-1][0].block_id, make_offload_key(entry.key, g)
                            )
                            is None
                        ):
                            break
                    manager = managers[g]
                    if isinstance(manager, (MambaManager, SlidingWindowManager)):
                        prefix = [pool.null_block] * first
                    elif isinstance(manager, CircularBufferManager):
                        prefix = []
                    else:
                        prefix = list(blocks[g])
                        hashes = BlockHashListWithBlockSize(
                            request.block_hashes,
                            self.hash_block_size,
                            manager.block_size,
                        )
                        chain_hash = chain_hash_from_request(
                            request.block_hashes, self.hash_block_size
                        )
                        for i in range(len(prefix), first):
                            cached = pool.get_cached_block(hashes[i], [g])
                            if cached is None:
                                break
                            if (
                                host_hits is not None
                                and g == host_hits.rule.group_id
                                and not host_hits.local_validator(
                                    g,
                                    i,
                                    cast(BlockHash, chain_hash(host_hits.rule, i)),
                                    cached[0].block_id,
                                )
                            ):
                                break
                            prefix.append(cached[0])
                        if len(prefix) != first:
                            break
                    result.append(prefix + [b for b, _ in held])
                if len(result) == len(managers):
                    self.hits[request.request_id] = entry
                    return tuple(result), n - 1
        return blocks, hit

    def consume(self, request: Request) -> None:
        entry = self.hits.pop(request.request_id, None)
        if entry is None:
            return
        for g, (_, held) in enumerate(entry.groups):
            exact_key = make_block_hash_with_group_id(entry.key, g)
            pool = self.coordinator.group_block_pool(g)
            for block, expected in held:
                if expected == exact_key:
                    pool._maybe_evict_cached_block(block)
        span = self.coordinator.single_type_managers[self.state_groups[0]].block_size
        self.index[self._anchor(request, (entry.length - 1) // span * span)].remove(
            entry
        )
