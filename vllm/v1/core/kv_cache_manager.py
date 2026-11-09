# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_hit_contract import (
    KVHitContract,
    chain_hash_from_request,
    local_hit_rules,
    settle_local_hit,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.long_context_kv import SEGMENT_NAMES, LongContextAdmissionLedger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.kv_offload.host.hit import HostKVHitManager

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    # Long-context mode: joint admission over the GPU state segment, GPU
    # history segment, host history pool and hot-page pool. Without a plan
    # (the default) allocation is exactly the unified-pool path.
    admission_ledger: LongContextAdmissionLedger | None = None
    # Explicitly attached by the host allocator after both complete-prefix contract
    # paths exist.
    host_kv_hits: "HostKVHitManager | None" = None

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        prefix_cache_retention_interval: int | None = None,
        cross_group_hit_check: bool = False,
    ) -> None:
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.coordinator.configure_prefix_cache_retention(
            prefix_cache_retention_interval
        )
        # Opt-in cross-group hit contract (kernel_config.qsa_cross_group_hit_check).
        # None keeps the coordinator's answer unchanged.
        self.hash_block_size = hash_block_size
        self.hit_contract: KVHitContract | None = None
        if cross_group_hit_check and enable_caching:
            self.hit_contract = KVHitContract(
                local_hit_rules(self.coordinator.single_type_managers),
                path="gpu-prefix-cache",
                hash_block_size=hash_block_size,
            )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config
        self.kv_cache_event_metadata = tuple(
            (
                get_kv_cache_spec_kind(group.kv_cache_spec).value,
                get_kv_cache_spec_sliding_window(group.kv_cache_spec),
            )
            for group in kv_cache_config.kv_cache_groups
        )

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        self._pending_reset_events: list[KVCacheEvent] = []
        self.admission_ledger = None
        plan = kv_cache_config.long_context_plan
        if plan is not None:
            if len(plan.geometry.group_domains) != self.num_kv_cache_groups:
                raise ValueError(
                    "Long-context plan describes "
                    f"{len(plan.geometry.group_domains)} KV cache groups, but "
                    f"the scheduler has {self.num_kv_cache_groups}."
                )
            # With separate block spaces the state and GPU history pools
            # enforce themselves; the ledger observes the history pool (the
            # main-K/V group's host pages live there) for the host domain.
            self.admission_ledger = LongContextAdmissionLedger(
                plan,
                kv_cache_config.num_blocks,
                physical_segments=kv_cache_config.block_segments is not None,
            )
            self.block_pool.admission_observer = self.admission_ledger
            self.coordinator.admission_ledger = self.admission_ledger
            logger.info("%s", plan.describe())

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0). With separate block
            spaces, the fullest one.
        """
        pools = self._segment_pools()
        if pools is None:
            return self.block_pool.get_usage()
        return max(pool.get_usage() for pool in pools)

    def _segment_pools(self) -> "tuple[BlockPool, ...] | None":
        """The pools of separate block spaces, or None for one shared pool."""
        # getattr: tests may build these objects without __init__.
        pools = getattr(getattr(self, "coordinator", None), "block_pools", ())
        return pools if len(pools) > 1 else None

    def _segment_pools_have_room(
        self,
        per_group: Sequence[int],
        reserved_blocks: int,
        reserved_per_group: Sequence[int] | None = None,
        *,
        reason: str = "allocation",
    ) -> bool:
        """Separate block spaces: whether each pool has room for its groups.

        Future demand stays in group units so state/history IDs and host
        pages are reserved in their own domains.
        """
        coordinator = self.coordinator
        demand: dict[int, int] = {}
        for group_id, blocks in enumerate(per_group):
            if reserved_per_group is not None:
                blocks += reserved_per_group[group_id]
            if blocks:
                pool = coordinator.group_block_pool(group_id)
                demand[id(pool)] = demand.get(id(pool), 0) + blocks
        for segment, pool in enumerate(coordinator.block_pools):
            # Preserve the scalar API for callers without a group vector.
            reserve = (
                reserved_blocks
                if reserved_per_group is None and pool is self.block_pool
                else 0
            )
            if demand.get(id(pool), 0) > pool.get_num_free_blocks() - reserve:
                logger.debug(
                    "Long-context admission blocked: reason=%s domain=%s "
                    "demand=%d free=%d reserved=%d",
                    reason,
                    SEGMENT_NAMES[segment],
                    demand.get(id(pool), 0),
                    pool.get_num_free_blocks(),
                    reserve,
                )
                return False
        return True

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        request.shared_prefix_boundary = 0
        host_hits = self.host_kv_hits
        if host_hits is not None:
            host_hits.begin_lookup(request.request_id)
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        # getattr: tests may build this object without __init__.
        contract = getattr(self, "hit_contract", None)
        probe_cap = None
        if contract is not None:
            chain_hash = chain_hash_from_request(
                request.block_hashes, self.hash_block_size
            )
            if contract.has_probes:
                # Stop before the first block a probed store no longer holds,
                # so the fixed point lands on the common complete prefix.
                probe_cap = contract.probe_cap(chain_hash, 0, max_cache_hit_length)
        if probe_cap is None:
            computed_blocks, num_new_computed_tokens = (
                self.coordinator.find_longest_cache_hit(
                    request.block_hashes, max_cache_hit_length
                )
            )
        else:
            computed_blocks, num_new_computed_tokens = (
                self.coordinator.find_longest_cache_hit(
                    request.block_hashes,
                    max_cache_hit_length,
                    max_final_hit_length=probe_cap,
                )
            )
        if contract is not None:
            computed_blocks, num_new_computed_tokens = settle_local_hit(
                contract,
                request.request_id,
                computed_blocks,
                num_new_computed_tokens,
                chain_hash,
            )
            if (
                host_hits is not None
                and num_new_computed_tokens
                and not host_hits.pin_hit(
                    request.request_id, chain_hash, num_new_computed_tokens
                )
            ):
                contract.reject("host prefix could not be pinned", request.request_id)
                computed_blocks = tuple([] for _ in computed_blocks)
                num_new_computed_tokens = 0

        # Keep the existing two-value lookup API used by 1Cat connectors. The
        # synchronous coordinator exposes the junction from this lookup only.
        request.shared_prefix_boundary = self.coordinator.shared_prefix_boundary

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        reserved_blocks_per_group: Sequence[int] | None = None,
        num_tokens_to_reserve: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
            full_sequence_must_fit: Only allocate blocks if the KV cache has enough
                free blocks to hold the full sequence, accounting for prefix cache hits
                and sliding window. Used as an admission gate to prevent over-admitting
                requests when chunked prefill would otherwise only check the first chunk
            reserved_blocks: Free blocks to leave for other in-flight prefills.
            reserved_blocks_per_group: The same future demand in group units
                for split pools and domain admission.
            num_tokens_to_reserve: Minimum slot coverage for an async load's
                executable successor. This is an admission check only. Existing
                blocks, prefix hits and this allocation count toward it.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        ledger = self.admission_ledger
        # Separate block spaces (long-context mode): each pool is checked
        # against its own groups' demand.
        split = self._segment_pools() is not None
        per_group_needed = ledger is not None or split
        if full_sequence_must_fit or num_tokens_to_reserve:
            reason = "async successor" if num_tokens_to_reserve else "full ISL"
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)
            reserve_num_tokens = min(
                max(full_num_tokens, num_tokens_to_reserve), self.max_model_len
            )

            if not per_group_needed:
                num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                    request_id=request.request_id,
                    num_tokens=reserve_num_tokens,
                    new_computed_blocks=new_computed_block_list,
                    num_encoder_tokens=num_encoder_tokens,
                    total_computed_tokens=total_computed_tokens,
                    num_tokens_main_model=full_num_tokens,
                    apply_admission_cap=True,
                )
            else:
                per_group = self.coordinator.get_num_blocks_to_allocate_per_group(
                    request_id=request.request_id,
                    num_tokens=reserve_num_tokens,
                    new_computed_blocks=new_computed_block_list,
                    num_encoder_tokens=num_encoder_tokens,
                    total_computed_tokens=total_computed_tokens,
                    num_tokens_main_model=full_num_tokens,
                    apply_admission_cap=True,
                )
                num_blocks_to_allocate = sum(per_group)
            if split:
                if not self._segment_pools_have_room(
                    per_group, reserved_blocks, reserved_blocks_per_group, reason=reason
                ):
                    if self.host_kv_hits is not None:
                        self.host_kv_hits.release_lookup(request.request_id)
                    return None
            elif num_blocks_to_allocate > (
                self.block_pool.get_num_free_blocks() - reserved_blocks
            ):
                if self.host_kv_hits is not None:
                    self.host_kv_hits.release_lookup(request.request_id)
                return None
            if ledger is not None and not ledger.admit(
                request, per_group, reserved_blocks_per_group, reason=reason
            ):
                if self.host_kv_hits is not None:
                    self.host_kv_hits.release_lookup(request.request_id)
                return None

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # Free on the processed-token basis: in-flight steps' attention windows
        # still read blocks below the optimistic boundary, and rejected spec
        # tokens can roll it back.
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        if not per_group_needed:
            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=num_local_computed_tokens
                + num_external_computed_tokens,
                num_tokens_main_model=num_tokens_main_model,
            )
        else:
            per_group = self.coordinator.get_num_blocks_to_allocate_per_group(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=num_local_computed_tokens
                + num_external_computed_tokens,
                num_tokens_main_model=num_tokens_main_model,
            )
            num_blocks_to_allocate = sum(per_group)

        if split:
            # Separate block spaces: the state and history pools each need
            # room for their own groups.
            if not self._segment_pools_have_room(
                per_group, reserved_blocks, reserved_blocks_per_group
            ):
                if self.host_kv_hits is not None:
                    self.host_kv_hits.release_lookup(request.request_id)
                return None
        elif num_blocks_to_allocate > (
            self.block_pool.get_num_free_blocks() - reserved_blocks
        ):
            # Cannot allocate new blocks
            if self.host_kv_hits is not None:
                self.host_kv_hits.release_lookup(request.request_id)
            return None
        # Long-context mode: every domain must have room too; otherwise the
        # scheduler waits or preempts exactly as for a full unified pool.
        if ledger is not None and not ledger.admit(
            request, per_group, reserved_blocks_per_group
        ):
            if self.host_kv_hits is not None:
                self.host_kv_hits.release_lookup(request.request_id)
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )
        if ledger is not None:
            ledger.commit(request)
        if self.host_kv_hits is not None:
            self.host_kv_hits.commit_lookup(request.request_id)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)
        if self.admission_ledger is not None:
            self.admission_ledger.release(request.request_id)
        if self.host_kv_hits is not None:
            self.host_kv_hits.release_request(request.request_id)

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for prefix-anchored SWA
                gap eviction.
        """
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        pools = self._segment_pools()
        if pools is None:
            self.block_pool.evict_blocks(block_ids)
            return
        # Separate block spaces: a reported ID cannot name its space, so evict
        # it from every pool that has it. Losing an extra cached prefix only
        # costs a recompute; the hit lookup needs every group anyway.
        for pool in pools:
            pool.evict_blocks({i for i in block_ids if i < pool.num_gpu_blocks})

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        pools = self._segment_pools()
        if pools is not None:
            # Separate block spaces: reset all of them or none.
            if any(
                pool.get_num_free_blocks() != pool.num_gpu_blocks - 1 for pool in pools
            ):
                logger.warning(
                    "Failed to reset prefix cache because some blocks are not freed yet"
                )
                return False
            # Keep the pre-reset batch before one global clear. Per-pool
            # concatenation after reset would put another clear after new
            # stores from an earlier pool and erase the consumer's new state.
            self._pending_reset_events.extend(
                event for pool in pools for event in pool.take_events()
            )
            reset_events = []
            for pool in pools:
                pool.reset_prefix_cache()
                reset_events.extend(pool.take_events())
            if reset_events:
                self._pending_reset_events.append(reset_events[0])
        elif not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        pools = self._segment_pools()
        if pools is None:
            events = self.block_pool.take_events()
        else:
            events = self._pending_reset_events
            self._pending_reset_events = []
            events.extend(event for pool in pools for event in pool.take_events())
        for event in events:
            if not isinstance(event, BlockStored):
                continue
            if event.group_idx is None:
                continue
            if event.group_idx < 0 or event.group_idx >= len(
                self.kv_cache_event_metadata
            ):
                logger.warning(
                    "Group index `%s` not in KV cache metadata", event.group_idx
                )
                continue
            # Annotate here so BlockPool can keep emitting structural cache
            # events without owning semantic KV cache spec metadata.
            kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
            event.kv_cache_spec_kind = kind
            event.kv_cache_spec_sliding_window = sliding_window
        return events

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        split = self._segment_pools() is not None
        for mgr in self.coordinator.single_type_managers:
            new_ids = mgr.take_new_block_ids()
            # Separate block spaces: the workers zero only the primary
            # (history) segment's tensors, where every attention group that
            # records new IDs lives; an ID of another space would name a
            # different block there.
            if not split or mgr.block_pool is self.block_pool:
                ids.extend(new_ids)
        return ids

    def take_boundary_state_offloads(
        self,
    ) -> dict[str, list[tuple[int, int, int]]]:
        """Drain exact Mamba ``align`` state blocks for KV connectors.

        Returns ``{request_id: [(group_id, block_id, boundary_tokens), ...]}``.
        These scheduler-local handoffs avoid reconstructing mutable Mamba block
        tables from a connector's append-only block-ID mirror.
        """
        offloads: dict[str, list[tuple[int, int, int]]] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                request_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_boundary_state_offloads():
                offloads.setdefault(request_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads

    def new_step_starts(self) -> None:
        """Called when a new step is started."""
        self.coordinator.new_step_starts()
