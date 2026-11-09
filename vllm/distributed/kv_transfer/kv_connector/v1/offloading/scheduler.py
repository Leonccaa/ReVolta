# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import TYPE_CHECKING, Any, NamedTuple

import regex as re

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.counters import (
    LookupShrink,
    OffloadCounters,
    build_offload_counters,
    offload_group_kind,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_hit_contract import (
    GroupHitRule,
    HitNeed,
    KVHitContract,
    KVHitContractError,
    chain_hash_from_request,
    connector_hit_rules,
    cross_group_hit_check_enabled,
    load_failure_recompute_enabled,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.host.receive import HostReceiveSpec, ReceiveTxn
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)

# Unified pools retain their existing integer identity. Split pools must carry
# the segment through every store fence, including destinations and zeroing.
BlockIdentity = int | tuple[int, int]

if TYPE_CHECKING:
    from vllm.v1.kv_offload.cpu.dependency import CPUBlockDependencyIndex
    from vllm.v1.kv_offload.host.hit import HostKVHitManager


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[BlockIdentity] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[BlockIdentity] | None = None
    # A worker reported this load as failed (load_failure_recompute_enabled).
    failed: bool = False
    receive_txn: ReceiveTxn | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    # None below means full attention
    sliding_window_size_in_blocks: int | None
    # Number of this group's offloaded blocks per full-attention alignment
    # segment. Used to skip storing SWA blocks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_block_count: int | None = None
    # Mamba ``align`` tables mutate existing logical positions. Such groups
    # must be stored from exact core-provided boundary block IDs instead of the
    # connector's append-only positional mirror.
    requires_exact_boundary_source: bool = False
    # Sliding window (tokens) of a speculative draft model's sliding-window
    # group spanning more than one offloaded block; None for every other group.
    # Such groups follow SchedulerOffloadConfig.cpu_draft_offload.
    draft_window_tokens: int | None = None


def get_sliding_window_size_in_blocks(
    kv_cache_spec: KVCacheSpec, offloaded_block_size: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, offloaded_block_size)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


# ``layers.<index>`` in an attention layer name, e.g. "model.layers.48.self_attn".
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")


def is_draft_sliding_window_group(spec: "OffloadingSpec", group_idx: int) -> bool:
    """Whether a KV cache group holds a speculative draft model's
    sliding-window attention (e.g. the DFlash drafter).

    The scheduler has no model, so the group is recognised from its spec and
    layer names: speculative decoding is configured, the spec is a plain
    sliding window (managed by SlidingWindowManager, which frees a head prefix
    of the request's blocks), and the group is either flagged as a draft group
    or all its layers are numbered past the target model's layers (drafters
    such as DFlash and EAGLE number their layers after the target's).
    Recurrent (Mamba) and full-attention groups are never draft groups here.
    """
    vllm_config = spec.vllm_config
    if getattr(vllm_config, "speculative_config", None) is None:
        return False
    group = spec.kv_cache_config.kv_cache_groups[group_idx]
    if type(group.kv_cache_spec) not in (SlidingWindowSpec, SlidingWindowMLASpec):
        return False
    if getattr(group, "is_eagle_group", False):
        return True
    model_config = getattr(vllm_config, "model_config", None)
    try:
        num_target_layers = model_config.get_total_num_hidden_layers()  # type: ignore[union-attr]
    except Exception:
        return False
    if not isinstance(num_target_layers, int) or not group.layer_names:
        return False
    for layer_name in group.layer_names:
        indices = _LAYER_INDEX_RE.findall(layer_name)
        if len(indices) != 1 or int(indices[0]) < num_target_layers:
            return False
    return True


def resolve_mamba_align_size(spec: "OffloadingSpec") -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be rounded
    down to a multiple of the offloaded block size. Asserts that all such
    groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, gpu_block_size in enumerate(spec.gpu_block_size):
        kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            offload_block_size = gpu_block_size * spec.block_size_factor
            assert mamba_align_size is None or mamba_align_size == offload_block_size
            mamba_align_size = offload_block_size
    return mamba_align_size


def _connector_hit_needs(
    config: "SchedulerOffloadConfig", kv_cache_config: KVCacheConfig
) -> dict[int, tuple[HitNeed, int, int]]:
    """Per offloaded group: what a prefix hit needs, in offloaded blocks.

    A sliding-window group needs the blocks its GPU manager keeps for the
    next token, i.e. those covering the last ``sliding_window - 1`` tokens
    (``SlidingWindowManager.get_num_skipped_tokens``). The lookup itself asks
    for ``cdiv(sliding_window, block)`` blocks, which is never fewer.
    """
    needs: dict[int, tuple[HitNeed, int, int]] = {}
    for group in config.kv_group_configs:
        spec = kv_cache_config.kv_cache_groups[group.group_idx].kv_cache_spec
        block = group.offloaded_block_size
        if group.sliding_window_size_in_blocks is None:
            needs[group.group_idx] = (HitNeed.PREFIX, block, 0)
        elif isinstance(spec, MambaSpec):
            needs[group.group_idx] = (HitNeed.STATE, block, 0)
        elif isinstance(spec, SlidingWindowSpec):
            needs[group.group_idx] = (
                HitNeed.WINDOW,
                block,
                spec.sliding_window - 1,
            )
        else:
            needs[group.group_idx] = (HitNeed.UNCHECKED, block, 0)
    return needs


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int
    offload_prompt_only: bool
    # Worker/state arrays retain original group indices, including scratch.
    num_kv_cache_groups: int
    # See OffloadingSpec.demote_superseded_states / superseded_state_rule.
    demote_superseded_states: bool = True
    superseded_state_rule: str = "request"
    # See OffloadingSpec.cpu_draft_offload.
    cpu_draft_offload: str = "all"
    cpu_enable_dependency_eviction: bool = False
    cpu_residency_stats: bool = False
    cpu_offload_counters: bool = False

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the offloaded_block_size of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_offloaded_block_sizes: set[int] = set()
        for idx, gpu_block_size in enumerate(spec.gpu_block_size):
            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            if not kv_spec.prefix_cacheable:
                continue
            sw = get_sliding_window_size_in_blocks(
                kv_spec, gpu_block_size * spec.block_size_factor
            )
            if sw is None:
                full_attn_offloaded_block_sizes.add(
                    gpu_block_size * spec.block_size_factor
                )

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_offloaded_block_sizes) == 1:
            alignment_tokens = full_attn_offloaded_block_sizes.pop()

        def _alignment_block_count(
            offloaded_block_size: int,
            sliding_window_size_in_blocks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_blocks is None:
                return None
            if alignment_tokens <= offloaded_block_size:
                return None
            per_segment = alignment_tokens // offloaded_block_size
            if sliding_window_size_in_blocks >= per_segment:
                return None
            return per_segment

        def _requires_exact_boundary_source(group_idx: int) -> bool:
            kv_spec = spec.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec
            return (
                isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align"
            )

        def _draft_window_tokens(
            group_idx: int, sliding_window_size_in_blocks: int | None
        ) -> int | None:
            if (
                sliding_window_size_in_blocks is None
                or sliding_window_size_in_blocks <= 1
                or not is_draft_sliding_window_group(spec, group_idx)
            ):
                return None
            kv_spec = spec.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec
            assert isinstance(kv_spec, SlidingWindowSpec)
            return kv_spec.sliding_window

        # Sparse retention (see CacheConfig.prefix_cache_retention_interval)
        # keeps Mamba ``align`` states on the GPU block grid: the replay and
        # junction boundaries of a request. Boundary stores can only take
        # states at offloaded-block boundaries, so with a block-size factor
        # above one most prompts would offload their attention blocks but no
        # Mamba state, and the prefix could never be restored. Retaining
        # offload-aligned checkpoints needs the core mask to learn the
        # offload alignment; until then reject the combination up front.
        retention_interval = (
            spec.vllm_config.cache_config.prefix_cache_retention_interval
        )
        if retention_interval is not None and spec.block_size_factor > 1:
            for idx, gpu_block_size in enumerate(spec.gpu_block_size):
                if not _requires_exact_boundary_source(idx):
                    continue
                raise ValueError(
                    "Sparse prefix-cache retention "
                    f"(prefix_cache_retention_interval={retention_interval}) "
                    "keeps Mamba align states at GPU block boundaries "
                    f"({gpu_block_size} tokens), but kv_connector_extra_config "
                    f"block_size={gpu_block_size * spec.block_size_factor} can "
                    "only offload states at offloaded-block boundaries. Omit "
                    "'block_size' (factor 1) or use "
                    "--prefix-cache-retention-interval none."
                )

        return cls(
            num_workers=spec.vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                    sliding_window_size_in_blocks=(
                        sw := get_sliding_window_size_in_blocks(
                            spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                            gpu_block_size * spec.block_size_factor,
                        )
                    ),
                    alignment_block_count=_alignment_block_count(
                        gpu_block_size * spec.block_size_factor, sw
                    ),
                    requires_exact_boundary_source=(
                        _requires_exact_boundary_source(idx)
                    ),
                    draft_window_tokens=_draft_window_tokens(idx, sw),
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
                if spec.kv_cache_config.kv_cache_groups[
                    idx
                ].kv_cache_spec.prefix_cacheable
                and idx != spec.kv_cache_config.host_kv_group_id
            ),
            block_size_factor=spec.block_size_factor,
            offload_prompt_only=spec.offload_prompt_only,
            num_kv_cache_groups=len(spec.kv_cache_config.kv_cache_groups),
            demote_superseded_states=getattr(spec, "demote_superseded_states", True),
            superseded_state_rule=getattr(spec, "superseded_state_rule", "request"),
            cpu_draft_offload=getattr(spec, "cpu_draft_offload", "all"),
            cpu_enable_dependency_eviction=getattr(
                spec, "cpu_enable_dependency_eviction", False
            ),
            cpu_residency_stats=getattr(spec, "cpu_residency_stats", False),
            cpu_offload_counters=getattr(spec, "cpu_offload_counters", False),
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_blocks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # Longest full-attention prefix (absolute tokens) the external tier holds
    # for this request, recorded by the last lookup. When it exceeds the hit
    # served to the scheduler, a sparse group (Mamba / sliding window) lacks
    # its checkpoint there: an externally discovered shared-prefix junction.
    external_full_attention_hit_tokens: int = 0
    # Lookup shrink tokens by cause from the last lookup, counted (observability
    # only) when the request is admitted; see LookupShrink.
    lookup_shrink: dict[str, int] | None = None
    # Mamba ``align`` boundary-state hand-offs ``(group_idx, block_id,
    # boundary_tokens)`` the host tier could not accept yet. The core offers
    # each hand-off once; the source block stays owned by the request, so the
    # store is retried every step until it is accepted, the request finishes
    # or it is preempted.
    pending_boundary_offloads: list[tuple[int, int, int]] = field(default_factory=list)
    # group_idx -> offload block indices of boundary states this request has
    # issued stores for (superseded_state_rule "extend").
    stored_state_blocks: dict[int, set[int]] = field(default_factory=dict)
    # cpu.draft_offload != "all" (draft groups only, keyed by group_idx):
    # next offloaded block index the draft store frontier examines,
    draft_frontier: dict[int, int] = field(default_factory=dict)
    # ("window") blocks awaiting a needed window / accepted store: one unsettled
    # accounting obligation each, including live blocks requeued by a CPU reset,
    draft_deferred: dict[int, set[int]] = field(default_factory=dict)
    # ("window") blocks offered successfully (or loaded) in this cache
    # generation, retired behind the processed-token window. A CPU reset
    # returns the still-held ones to draft_deferred.
    draft_reofferable: dict[int, set[int]] = field(default_factory=dict)
    # ("window") and the token boundaries whose recurrent state this request
    # handed to the host tier: the draft window before each is offloaded.
    draft_window_boundaries: set[int] = field(default_factory=set)
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in range(self.config.num_kv_cache_groups)
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config in self.config.kv_group_configs:
            group_state = self.group_states[group_config.group_idx]
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config in self.config.kv_group_configs:
            group_state = self.group_states[group_config.group_idx]
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            group_state.next_stored_block_idx = num_blocks

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config in self.config.kv_group_configs:
            group_state = self.group_states[group_config.group_idx]
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


def _create_req_context(req: Request) -> ReqContext:
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=req.kv_transfer_params,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    host_kv_hits: "HostKVHitManager | None" = None
    # value arbitration value arbitration: called with each successfully completed
    # receive
    # transaction before it retires (None: arbitration off).
    value_restore_hook: Callable[[ReceiveTxn], None] | None = None

    # Mamba boundary-state hand-offs the host tier could not accept in the
    # step they were offered (see _build_boundary_state_store_jobs).
    num_rejected_boundary_offers: int = 0
    # Draft-model sliding-window blocks not offloaded because of
    # cpu.draft_offload ("window" / "none"); see _select_draft_blocks.
    num_skipped_draft_blocks: int = 0

    def __init__(self, spec: OffloadingSpec):
        self.kv_cache_config = spec.kv_cache_config
        self.config = SchedulerOffloadConfig.from_spec(spec)
        if (
            spec.kv_cache_config.host_kv_layers
            and self.config.cpu_draft_offload == "none"
            and any(
                g.draft_window_tokens is not None for g in self.config.kv_group_configs
            )
        ):
            raise ValueError(
                "Host KV prefix recovery requires cpu.draft_offload='all' or "
                "'window' for the draft WINDOW group"
            )
        self._group_config_by_idx = {
            group.group_idx: group for group in self.config.kv_group_configs
        }
        self.manager: OffloadingManager = spec.get_manager()

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self._group_config_by_idx[i].sliding_window_size_in_blocks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(spec)
        # With EAGLE/MTP (same condition as Scheduler.use_eagle), restored Mamba
        # states must sit one page before the full-attention reach; see _lookup.
        speculative_config = spec.vllm_config.speculative_config
        self._eagle_state_shift: int | None = (
            self._mamba_align_size
            if speculative_config is not None
            and speculative_config.use_eagle_kv_cache()
            else None
        )

        # Opt-in cross-group hit contract (kernel_config.qsa_cross_group_hit_check).
        self._hash_block_size: int = spec.hash_block_size
        self.hit_contract: KVHitContract | None = None
        if cross_group_hit_check_enabled(spec.vllm_config):
            self.hit_contract = KVHitContract(
                connector_hit_rules(
                    _connector_hit_needs(self.config, spec.kv_cache_config)
                ),
                path="offloading-connector",
                hash_block_size=self._hash_block_size,
            )

        # Broken loads void the external hit and recompute instead of raising
        # (load_failure_recompute_enabled). Requests whose load was abandoned
        # since the scheduler last asked, and requests whose next lookup must
        # not offer an external hit again.
        self._recompute_failed_loads: bool = load_failure_recompute_enabled(
            spec.vllm_config
        )
        self._failed_loads: set[ReqId] = set()
        self._skip_external_once: set[ReqId] = set()
        self.num_abandoned_loads: int = 0

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        # Recurrent states demoted by _demote_superseded_states (all requests).
        self.num_demoted_states: int = 0
        self._counters_logged_at: float = time.monotonic()
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}
        # Load jobs still in flight when reset_cache discarded the job table
        # (load_failure_recompute_enabled only). Their completion is otherwise
        # skipped as stale, but a load that fails after the reset must still
        # void the request's external hit before finished_recving promotes it.
        self._stale_load_jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_segments = spec.kv_cache_config.block_segments
        self._block_id_to_pending_jobs: dict[BlockIdentity, set[int]] = {}
        self.num_rejected_boundary_offers = 0
        self.num_skipped_draft_blocks = 0
        self._reported_skipped_draft_blocks = 0
        self._dependency_index: CPUBlockDependencyIndex | None = None
        if (
            self.config.cpu_enable_dependency_eviction
            or self.config.cpu_residency_stats
        ):
            self._init_dependency_index(spec.kv_cache_config)
        # Read-only counters (stored blocks by kind, lookup shrink, draft
        # window use); never consulted by any decision.
        self._offload_counters: OffloadCounters | None = None
        if self.config.cpu_offload_counters:
            self._offload_counters = build_offload_counters(
                self.config.kv_group_configs, spec.kv_cache_config
            )
            logger.info(
                "Offload counter kinds by KV cache group: %s (draft window: %s tokens)",
                self._offload_counters.kind_by_group,
                self._offload_counters.draft_window_tokens,
            )

        draft_groups = [
            group.group_idx
            for group in self.config.kv_group_configs
            if group.draft_window_tokens is not None
        ]
        if self.config.cpu_draft_offload != "all":
            if draft_groups:
                logger.info(
                    "cpu.draft_offload=%s for draft sliding-window KV cache "
                    "group(s) %s",
                    self.config.cpu_draft_offload,
                    draft_groups,
                )
            else:
                logger.warning(
                    "cpu.draft_offload=%s has no effect: no KV cache group was "
                    "recognised as a speculative draft sliding-window group",
                    self.config.cpu_draft_offload,
                )

    def _init_dependency_index(self, kv_cache_config: KVCacheConfig) -> None:
        from vllm.v1.kv_offload.cpu.dependency import (
            CPUBlockDependencyIndex,
            DependencyGroup,
        )
        from vllm.v1.kv_offload.cpu.manager import (
            CPUOffloadingManager,
            GroupedCPUOffloadingManager,
        )
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
        from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

        pools = []
        supported = True
        stack = [self.manager]
        while stack:
            manager = stack.pop()
            if isinstance(manager, GroupedCPUOffloadingManager):
                stack.extend(manager.managers.values())
            elif isinstance(manager, TieringOffloadingManager):
                supported &= not bool(manager.secondary_tiers)
                stack.append(manager.primary_tier)
            elif isinstance(manager, CPUOffloadingManager):
                pools.append(manager)
                supported &= isinstance(manager._policy, LRUCachePolicy)
            else:
                supported = False
        groups = []
        sizes = set()
        for g in self.config.kv_group_configs:
            kind = offload_group_kind(
                g, kv_cache_config.kv_cache_groups[g.group_idx].kv_cache_spec
            )
            groups.append(
                DependencyGroup(
                    g.group_idx,
                    g.offloaded_block_size,
                    kind,
                    g.sliding_window_size_in_blocks,
                )
            )
            if kind != "draft":
                sizes.add(g.offloaded_block_size)
                # Statistics recognise all recurrent modes, but deletion is
                # modelled only for align states on this checkpoint grid.
                supported &= (
                    g.requires_exact_boundary_source
                    and g.sliding_window_size_in_blocks == 1
                    if kind == "state"
                    else g.sliding_window_size_in_blocks is None
                )
        supported &= len(sizes) == 1
        if sizes:
            size = next(iter(sizes))
            supported &= all(
                size % g.block_size == 0 for g in groups if g.kind == "draft"
            )
        self._dependency_index = CPUBlockDependencyIndex(
            groups,
            pools,
            supported,
            eagle_lookahead=self._eagle_state_shift is not None,
        )
        if not self._dependency_index.supported:
            logger.warning(
                "CPU dependency removal disabled for this tier/group geometry; "
                "residency is classified as unknown"
            )
        self._residency_sampled_at: float | None = None
        self._residency_totals: dict[tuple[str, str], int] = {}
        self._residency_pending = False

    def _dependency_protected_keys(self) -> set[OffloadKey]:
        # Include waiting/preempted requests and finished requests with jobs.
        # A running request can still hand off a later checkpoint or window.
        keys: set[OffloadKey] = set()
        for status in self._req_status.values():
            status.update_offload_keys()
            if any(
                len(status.group_states[g.group_idx].offload_keys)
                < status.req.num_tokens // g.offloaded_block_size
                for g in self.config.kv_group_configs
            ):
                index = self._dependency_index
                assert index is not None
                return set(index.records)  # An active prefix has unknown hashes.
            for group in status.group_states:
                keys.update(group.offload_keys)
        return keys

    def _prepare_store(
        self, keys: list[OffloadKey], status: RequestOffloadState
    ) -> PrepareStoreOutput | None:
        index = getattr(self, "_dependency_index", None)
        before = (
            sum(m.num_evicted_blocks for m in index.managers)
            if index is not None and self.config.cpu_enable_dependency_eviction
            else 0
        )
        output = self.manager.prepare_store(keys, status.req_context)
        if index is not None:
            # Even a rejected grouped reservation can already have evicted
            # earlier groups. The observer has cleaned those records.
            index.annotate(
                keys,
                [g.offload_keys for g in status.group_states],
                status.req.shared_prefix_boundary,
            )
            if (
                self.config.cpu_enable_dependency_eviction
                and sum(m.num_evicted_blocks for m in index.managers) > before
            ):
                removed = index.evict(self._dependency_protected_keys())
                if output is not None:
                    # The manager's capacity event may own the original list.
                    output.evicted_keys = output.evicted_keys + removed
        return output

    def cpu_residency_snapshot(
        self, consume: bool = True
    ) -> dict[tuple[str, str], int] | None:
        """New C7 snapshot at most once per minute; no work when disabled."""
        index = getattr(self, "_dependency_index", None)
        if index is None:
            return None
        now = time.monotonic()
        if self._residency_sampled_at is None or now - self._residency_sampled_at >= 60:
            self._residency_sampled_at = now
            self._residency_totals = index.totals(self._dependency_protected_keys())
            self._residency_pending = True
        if consume:
            if not self._residency_pending:
                return None
            self._residency_pending = False
        return self._residency_totals.copy()

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _block_identity(self, group_idx: int, block_id: int) -> BlockIdentity:
        segments = getattr(self, "_block_segments", None)
        if segments is None:
            return block_id
        return segments.group_segments[group_idx], block_id

    def _transfer_block_identities(
        self, block_ids: list[int], group_sizes: list[int]
    ) -> Sequence[BlockIdentity]:
        if getattr(self, "_block_segments", None) is None:
            return block_ids
        offset = 0
        identities: list[BlockIdentity] = []
        for group_idx, size in enumerate(group_sizes):
            identities.extend(
                self._block_identity(group_idx, bid)
                for bid in block_ids[offset : offset + size]
            )
            offset += size
        return identities

    def _remove_pending_job(
        self, job_id: int, block_ids: Iterable[BlockIdentity]
    ) -> None:
        # Registration is idempotent because each block maps to a set of job IDs.
        # MTP lookahead may report the same physical block ID more than once, so
        # completion must use the same set semantics instead of removing twice.
        for bid in set(block_ids):
            pending = self._block_id_to_pending_jobs.get(bid)
            if pending is None:
                continue
            pending.discard(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _maximal_prefix_lookup(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> int | None:
        """Return the number of consecutive offloaded blocks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            result = self.manager.lookup(key, req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a miss is detected)
                result = True
            if not result:
                break
            hit_count += 1
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            result = self.manager.lookup(keys[idx], req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a hit is detected)
                result = False
            if not result:
                consecutive_hits = 0
            else:
                consecutive_hits += 1
                if consecutive_hits == sliding_window_size:
                    return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config in self.config.kv_group_configs:
            group_state = req_status.group_states[group_config.group_idx]
            if group_config.sliding_window_size_in_blocks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_blocks
                    - group_config.sliding_window_size_in_blocks,
                )
                self.manager.touch(
                    group_state.offload_keys[blocks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(
        self,
        req_status: RequestOffloadState,
        shrink: LookupShrink | None = None,
        hit_cap: int | None = None,
    ) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.

        ``shrink`` (optional, observability only) records how far the
        sliding-window groups lower the hit below the full-attention reach.
        ``hit_cap`` (cross-group hit contract only) bounds the hit in absolute
        tokens, e.g. where a probed store stops holding the prefix.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        req_status.external_full_attention_hit_tokens = 0
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )
        if hit_cap is not None and hit_cap < max_hit_size_tokens:
            # PREFIX probes cap the final usable prefix. Mamba's EAGLE
            # lookbehind still needs the original extra-page search horizon.
            max_hit_size_tokens = min(
                max_hit_size_tokens, hit_cap + (self._eagle_state_shift or 0)
            )
            if self._mamba_align_size is not None:
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )
        num_hit_tokens: int = 0
        defer_lookup = False
        eagle_state_shift = self._eagle_state_shift
        lookup_groups = self._lookup_groups
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self._group_config_by_idx[group_idx]
                if (
                    shrink is not None
                    and group_config.sliding_window_size_in_blocks is not None
                ):
                    shrink.enter(group_idx, max_hit_size_tokens)
                if (
                    eagle_state_shift is not None
                    and group_config.sliding_window_size_in_blocks is not None
                ):
                    # With EAGLE/MTP a GPU prefix hit ends one Mamba page before
                    # the full-attention reach (MambaManager.find_longest_cache_hit),
                    # and sparse retention keeps states only on such pages. Look
                    # the sparse groups up one page before the reach as well (the
                    # full-attention groups come first): a state on the last
                    # matched page would be restored where the GPU cannot use it,
                    # so the GPU would keep the attention blocks but no usable
                    # state, and every later request with this prefix would
                    # reload the whole prefix from this tier.
                    max_hit_size_tokens = max(
                        round_down(max_hit_size_tokens, eagle_state_shift)
                        - eagle_state_shift,
                        0,
                    )
                    eagle_state_shift = None
                    if hit_cap is not None:
                        max_hit_size_tokens = min(max_hit_size_tokens, hit_cap)
                    if shrink is not None:
                        shrink.lower(max_hit_size_tokens, "page_align")
                group_state: RequestGroupState = req_status.group_states[group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys)
                    >= req_status.req.num_tokens // offloaded_block_size
                )

                # Constrain to block-aligned boundary for this group
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * offloaded_block_size
                )
                if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                num_blocks = min(
                    cdiv(max_hit_size_tokens, offloaded_block_size), len(offload_keys)
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_blocks: int | None
                if sliding_window_size_in_blocks is None:
                    num_hit_blocks = self._maximal_prefix_lookup(
                        offload_keys, req_status.req_context
                    )
                else:
                    num_hit_blocks = self._sliding_window_lookup(
                        offload_keys,
                        sliding_window_size_in_blocks,
                        req_status.req_context,
                    )
                if num_hit_blocks == 0:
                    return 0

                if num_hit_blocks is None:
                    defer_lookup = True
                else:
                    group_hit_tokens = offloaded_block_size * (
                        start_block_idx + num_hit_blocks
                    )
                    if sliding_window_size_in_blocks is None:
                        req_status.external_full_attention_hit_tokens = max(
                            req_status.external_full_attention_hit_tokens,
                            group_hit_tokens,
                        )
                    max_hit_size_tokens = min(max_hit_size_tokens, group_hit_tokens)
                    if shrink is not None and sliding_window_size_in_blocks is not None:
                        shrink.lower(max_hit_size_tokens)
                    if self._mamba_align_size is not None:
                        # A group with smaller blocks (e.g. a drafter's sliding
                        # window) can end its hit inside a Mamba page. The
                        # recurrent lookup would then take the state at the
                        # end of that page as the state at the hit. Keep the
                        # hit on the page grid; a sliding-window group cut
                        # back this way must be looked up again, as its
                        # window has to end at the new hit.
                        aligned = round_down(
                            max_hit_size_tokens, self._mamba_align_size
                        )
                        if (
                            aligned < max_hit_size_tokens
                            and sliding_window_size_in_blocks is not None
                            and not lookup_groups
                        ):
                            lookup_groups = self._sliding_window_groups
                        max_hit_size_tokens = aligned
                        if (
                            shrink is not None
                            and sliding_window_size_in_blocks is not None
                        ):
                            shrink.lower(aligned, "page_align")

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_blocks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        if self._blocks_being_loaded:
            for group_config in self.config.kv_group_configs:
                group_state = req_status.group_states[group_config.group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                if sliding_window_size_in_blocks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_blocks:]
                if any(key in self._blocks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its"
                        " blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if not getattr(self, "_recompute_failed_loads", False):
            return self._get_num_new_matched_tokens(request, num_computed_tokens)
        try:
            return self._get_num_new_matched_tokens(request, num_computed_tokens)
        except (AssertionError, IndexError, KeyError, ValueError, RuntimeError):
            # Nothing has been admitted or transferred yet. A lookup failure
            # is a miss, preserving any independently verified local prefix.
            logger.debug("KV lookup failed for %s", request.request_id, exc_info=True)
            return 0, False

    def _get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()
            # Restarting at token 0: offer every block again; resident keys
            # are filtered by prepare_store (vllm-project/vllm#63010).
            group_state.next_stored_block_idx = 0
        # The block table is rebuilt as well: draft blocks deferred from an
        # earlier run are no longer held (cpu.draft_offload "window").
        self._drop_draft_deferred(req_status)
        req_status.draft_frontier.clear()

        # A preempted request can return before its stores have completed.
        # Its preemption metadata will flush them before the next forward.
        if req_status.transfer_jobs:
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        counters = getattr(self, "_offload_counters", None)
        if counters is not None:
            req_status.lookup_shrink = None
        # getattr: tests may build this object without __init__.
        skip_once = getattr(self, "_skip_external_once", None)
        if skip_once and request.request_id in skip_once:
            # The previous external load of this request was abandoned
            # (take_failed_loads): recompute from the local hit this time.
            num_hit_tokens = 0
        elif request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            shrink = counters.new_lookup() if counters is not None else None
            # getattr: tests may build this object without __init__.
            contract = getattr(self, "hit_contract", None)
            if contract is None:
                num_hit_tokens = self._lookup(req_status, shrink)
            else:
                num_hit_tokens = self._contract_lookup(
                    contract, req_status, shrink, num_computed_tokens
                )
            if shrink is not None and num_hit_tokens is not None:
                # Served at the local hit when there is no external hit.
                shrink.lower(num_computed_tokens + num_hit_tokens)
                req_status.lookup_shrink = shrink.tokens
        served_tokens = num_computed_tokens + (num_hit_tokens or 0)
        req_status.update_num_hit_blocks(served_tokens)

        # Marconi-style junction discovered in the external tier: the full
        # attention prefix reaches further than the sparse groups can serve.
        # Record it on the request (like the GPU prefix cache does) so the
        # scheduler ends a chunk there, the retention mask keeps the state and
        # the boundary hand-off offloads it; the next sibling then hits.
        junction = req_status.external_full_attention_hit_tokens
        if self._eagle_state_shift is not None:
            # A usable state sits one page before the reach (see _lookup).
            junction = max(
                round_down(junction, self._eagle_state_shift) - self._eagle_state_shift,
                0,
            )
        if (
            num_hit_tokens is not None
            and self._sliding_window_groups
            and junction > served_tokens
        ):
            request.shared_prefix_boundary = max(
                request.shared_prefix_boundary, junction
            )

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if not getattr(self, "_recompute_failed_loads", False):
            return self._update_state_after_alloc(request, blocks, num_external_tokens)
        try:
            return self._update_state_after_alloc(request, blocks, num_external_tokens)
        except (
            AssertionError,
            IndexError,
            KeyError,
            ValueError,
            RuntimeError,
        ) as error:
            # No worker I/O is submitted until metadata is built. Known tier
            # managers make prepare_load transactional, so a planning failure
            # can safely void the complete request before it becomes runnable.
            self._abandon_load(request.request_id, f"load preparation failed: {error}")

    def _update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        counters = getattr(self, "_offload_counters", None)
        if counters is not None:
            self._count_admitted_lookup(counters, request, num_external_tokens)

        if num_external_tokens == 0:
            # Admission succeeded: this request will actually recompute. A
            # lookup followed by capacity/budget deferral must keep the marker.
            skip_once = getattr(self, "_skip_external_once", None)
            if skip_once:
                skip_once.discard(request.request_id)
            return

        req_status = self._req_status[request.request_id]
        # Check before prepare_load pins or a job is inserted.
        assert not req_status.transfer_jobs

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = [0] * self.config.num_kv_cache_groups
        block_indices: list[int] = [0] * self.config.num_kv_cache_groups
        # group_idx -> (first loaded GPU block, end GPU block); contract only
        contract = getattr(self, "hit_contract", None)
        load_plan: dict[int, tuple[int, int]] | None = (
            {} if contract is not None else None
        )
        for group_config in self.config.kv_group_configs:
            group_state = req_status.group_states[group_config.group_idx]
            group_blocks = blocks.blocks[group_config.group_idx]
            gpu_block_size = group_config.gpu_block_size
            offloaded_block_size = group_config.offloaded_block_size
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)

            assert len(group_blocks) >= num_gpu_blocks
            # The load boundary is a slice index, not a count of computed blocks.
            # Sparse groups (Mamba, SWA) can hold non-null unhashed blocks below
            # the local hit. Scan only beyond it, still skipping null padding.
            first_fresh_gpu_block_idx = cdiv(
                num_locally_computed_tokens, gpu_block_size
            )
            load_start_gpu_block_idx = num_gpu_blocks
            for i in range(first_fresh_gpu_block_idx, num_gpu_blocks):
                block = group_blocks[i]
                if not block.is_null and block.block_hash is None:
                    load_start_gpu_block_idx = i
                    break

            assert num_locally_computed_tokens % gpu_block_size == 0
            num_pending_gpu_blocks = num_gpu_blocks - load_start_gpu_block_idx

            if group_config.sliding_window_size_in_blocks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_blocks
                    * self.config.block_size_factor
                )

            num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
            assert len(offload_keys) >= num_blocks
            if num_pending_gpu_blocks:
                start_block_idx = (
                    load_start_gpu_block_idx // self.config.block_size_factor
                )
                keys_to_load.extend(offload_keys[start_block_idx:num_blocks])

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[load_start_gpu_block_idx:num_gpu_blocks]
            )
            group_sizes[group_config.group_idx] = num_pending_gpu_blocks
            block_indices[group_config.group_idx] = load_start_gpu_block_idx
            if load_plan is not None:
                load_plan[group_config.group_idx] = (
                    load_start_gpu_block_idx,
                    num_gpu_blocks,
                )

            # Skip prefix-hit blocks for block-level policy; for
            # request-level, next_stored_block_idx stays at 0 so all
            # blocks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_block_idx = num_blocks

            if (
                self.config.cpu_draft_offload == "window"
                and group_config.draft_window_tokens is not None
            ):
                # A reset can also evict the prefix just loaded from the tier.
                # Remember its GPU window even though the store frontier skips
                # prefix-hit blocks. Include locally cached non-null blocks.
                window = group_config.sliding_window_size_in_blocks
                assert window is not None
                req_status.draft_reofferable.setdefault(
                    group_config.group_idx, set()
                ).update(
                    b
                    for b in range(
                        max(0, num_blocks - window),
                        num_cached_tokens // offloaded_block_size,
                    )
                    if not group_blocks[
                        (b + 1) * self.config.block_size_factor - 1
                    ].is_null
                )

        if contract is not None:
            assert load_plan is not None
            reason = self._check_load_plan(
                contract,
                req_status,
                blocks,
                load_plan,
                keys_to_load,
                num_locally_computed_tokens,
                num_cached_tokens,
            )
            if reason is not None:
                contract.stats.load_plan_errors += 1
                if getattr(self, "_recompute_failed_loads", False):
                    self._abandon_load(request.request_id, reason)
                    return
                raise KVHitContractError(
                    f"request {request.request_id}: load plan for "
                    f"{num_external_tokens} external tokens after "
                    f"{num_locally_computed_tokens} local tokens would restore "
                    f"an incomplete prefix: {reason}"
                )

        # Fence dst blocks against finished-request pending stores.
        dst_identities = self._transfer_block_identities(dst_block_ids, group_sizes)
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(dst_identities)
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in dst_identities
                for jid in self._block_id_to_pending_jobs.get(bid, ())
            )

        if getattr(self, "_recompute_failed_loads", False) and contract is None:
            # Without the contract nothing re-checked the keys since lookup;
            # prepare_load asserts on a key that is gone.
            missing = next(
                (
                    key
                    for key in keys_to_load
                    if not self._key_ready(key, req_status.req_context)
                ),
                None,
            )
            if missing is not None:
                self._abandon_load(
                    request.request_id, f"offload key {missing!r} is not readable"
                )
                return

        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )
        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)

        load_job_id = self._generate_job_id()
        txn = None
        host_hits = self.host_kv_hits
        if host_hits is not None and host_hits.history_pool is not None:
            try:
                txn = host_hits.build_receive(
                    request,
                    blocks.blocks,
                    num_locally_computed_tokens,
                    num_cached_tokens,
                    load_job_id,
                )
                host_hits.start_receive(txn)
            except Exception:
                self.manager.complete_load(keys_to_load, req_status.req_context)
                raise
            src_spec = HostReceiveSpec(src_spec, txn)
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            transfer_spec=(src_spec, dst_spec),
        )
        # a load can only be issued when no other jobs are pending.
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
            receive_txn=txn,
        )

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(keys_to_load)

    # --- abandoned loads (load_failure_recompute_enabled) -----------------------

    def _abandon_load(self, req_id: ReqId, reason: str) -> None:
        """Void a committed external hit: no transfer is issued, the scheduler
        releases the request's blocks and recomputes, and the next lookup of
        this request offers no external hit."""
        self.num_abandoned_loads += 1
        status = self._req_status.get(req_id)
        if status is not None and not status.req.is_finished():
            self._failed_loads.add(req_id)
            self._skip_external_once.add(req_id)
            status.draft_reofferable.clear()
        else:
            self._failed_loads.discard(req_id)
            skip_once = getattr(self, "_skip_external_once", None)
            if skip_once:
                skip_once.discard(req_id)
        logger.warning_once(
            "Offloading connector abandoned an external KV load and recomputes "
            "the request instead (further cases are logged at debug level)."
        )
        logger.debug("Abandoned external KV load of %s: %s", req_id, reason)

    def take_failed_loads(self) -> set[ReqId]:
        failed = self._failed_loads
        self._failed_loads = set()
        return failed

    # --- cross-group hit contract (kernel_config.qsa_cross_group_hit_check) --

    def _key_ready(self, key: OffloadKey, req_context: ReqContext) -> bool:
        """Whether ``key`` is resident and readable in the tier loads read
        from, without touching recency, admission counters or promotions.
        Managers this method does not know answer through their own lookup."""
        from vllm.v1.kv_offload.cpu.manager import (
            CPUOffloadingManager,
            GroupedCPUOffloadingManager,
        )
        from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

        manager: OffloadingManager | None = self.manager
        while True:
            if isinstance(manager, GroupedCPUOffloadingManager):
                manager = manager.managers.get(get_offload_group_idx(key))
            elif isinstance(manager, TieringOffloadingManager):
                manager = manager.primary_tier
            elif isinstance(manager, CPUOffloadingManager):
                return manager.is_ready_resident(key)
            elif manager is None:
                return False
            else:
                return manager.lookup(key, req_context) is True

    def _contract_lookup(
        self,
        contract: KVHitContract,
        req_status: RequestOffloadState,
        shrink: LookupShrink | None,
        num_computed_tokens: int,
    ) -> int | None:
        """``_lookup`` under the cross-group hit contract: bounded by the
        probed stores, and an incomplete answer is reported as a miss."""
        req = req_status.req
        host_hits = self.host_kv_hits
        if host_hits is not None and host_hits.cache is not None:
            host_hits.cache.poll()
        group_starts = None
        reuse_end = num_computed_tokens
        if host_hits is not None:
            reuse_end = host_hits.plan_reuse(
                req, num_computed_tokens, self._hash_block_size
            )
            group_starts = {host_hits.rule.group_id: reuse_end}
        chain_hash = chain_hash_from_request(req.block_hashes, self._hash_block_size)
        hit_cap = None
        if contract.has_probes:
            hit_cap = contract.probe_cap(
                chain_hash,
                num_computed_tokens,
                req.num_tokens,
                group_starts=group_starts,
            )
        source_cap = hit_cap
        if (
            host_hits is not None
            and host_hits.history_pool is not None
            and hit_cap is not None
            and self._eagle_state_shift is not None
        ):
            # G0's virtual full-attention reach replaces its old backend
            # lookup. Preserve the existing one-page state lookbehind.
            hit_cap = max(0, hit_cap - self._eagle_state_shift)
        num_hit_tokens = self._lookup(req_status, shrink, hit_cap)
        if host_hits is not None and host_hits.history_pool is not None:
            # Preserve junction discovery after G0 leaves backend membership.
            # Its source-aware probe cap is the full-attention reach.
            req_status.external_full_attention_hit_tokens = max(
                req_status.external_full_attention_hit_tokens, source_cap or 0
            )
        if not num_hit_tokens:
            return num_hit_tokens
        if host_hits is not None and host_hits.history_pool is not None:
            # Borrow only full blocks within the shared endpoint. build_receive
            # rejects partial G0 endpoints before rebinding immutable sources.
            reuse_end = min(
                reuse_end,
                (num_computed_tokens + num_hit_tokens)
                // host_hits.rule.block_size
                * host_hits.rule.block_size,
            )
            host_hits.trim_reuse(req.request_id, reuse_end)
            group_starts = {host_hits.rule.group_id: reuse_end}

        def _block_ok(rule: GroupHitRule, index: int) -> bool | None:
            if rule.group_id not in self._group_config_by_idx:
                return None
            keys = req_status.group_states[rule.group_id].offload_keys
            return index < len(keys) and self._key_ready(
                keys[index], req_status.req_context
            )

        reason = contract.check(
            num_computed_tokens + num_hit_tokens,
            num_computed_tokens,
            _block_ok,
            chain_hash,
            group_starts=group_starts,
        )
        if reason is not None:
            contract.reject(reason, req.request_id)
            return 0
        if host_hits is not None and not host_hits.pin_hit(
            req.request_id,
            chain_hash,
            start=reuse_end if host_hits.local_only else 0,
            end=num_computed_tokens + num_hit_tokens,
        ):
            contract.reject("host prefix could not be pinned", req.request_id)
            return 0
        return num_hit_tokens

    def _check_load_plan(
        self,
        contract: KVHitContract,
        req_status: RequestOffloadState,
        blocks: KVCacheBlocks,
        load_plan: dict[int, tuple[int, int]],
        keys_to_load: list[OffloadKey],
        num_local_tokens: int,
        num_cached_tokens: int,
    ) -> str | None:
        """Check that the loads plus the local hit restore every group at
        exactly ``num_cached_tokens`` and nothing beyond it."""
        factor = self.config.block_size_factor
        loading = set(keys_to_load)
        for group_idx, (load_start, load_end) in load_plan.items():
            group_blocks = blocks.blocks[group_idx]
            if any(b.is_null for b in group_blocks[load_start:load_end]):
                return f"group {group_idx}: a load targets the null block"

        def _block_ok(rule: GroupHitRule, index: int) -> bool | None:
            plan = load_plan.get(rule.group_id)
            if plan is None:
                return None
            keys = req_status.group_states[rule.group_id].offload_keys
            if index < len(keys) and keys[index] in loading:
                # Restored by this load: the key must be readable.
                return self._key_ready(keys[index], req_status.req_context)
            if index * factor >= plan[0]:
                return False  # fresh GPU block that nothing restores
            # Below the first loaded block: held by the local GPU hit.
            gpu_idx = (index + 1) * factor - 1
            group_blocks = blocks.blocks[rule.group_id]
            if gpu_idx >= len(group_blocks):
                return False
            block = group_blocks[gpu_idx]
            return not block.is_null and block.block_hash is not None

        return contract.check(
            num_cached_tokens,
            num_local_tokens,
            _block_ok,
            chain_hash_from_request(req_status.req.block_hashes, self._hash_block_size),
            # G0's concrete source identities are checked by build_receive.
            # STATE/WINDOW keep num_local_tokens as their independent start.
            group_starts=(
                {self.host_kv_hits.rule.group_id: num_cached_tokens}
                if self.host_kv_hits is not None
                and self.host_kv_hits.history_pool is not None
                else None
            ),
        )

    def _count_admitted_lookup(
        self, counters: OffloadCounters, request: Request, num_external_tokens: int
    ) -> None:
        """Count the admitted request's last lookup once (observability)."""
        req_status = self._req_status.get(request.request_id)
        if req_status is None:
            return
        shrink_tokens = req_status.lookup_shrink
        req_status.lookup_shrink = None
        counters.count_admitted_lookup(
            shrink_tokens,
            num_external_tokens,
            request.num_tokens
            - req_status.num_locally_computed_tokens
            - num_external_tokens,
        )

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        block_size_factor = self.config.block_size_factor
        store_jobs: dict[int, TransferJob] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            skip_once = getattr(self, "_skip_external_once", None)
            if skip_once:
                skip_once.discard(req_id)
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()
            req = req_status.req
            if self.config.cpu_draft_offload == "window":
                self._prune_draft_history(req_status)

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                req_status.update_block_id_groups(new_block_id_groups)
                # Fence new blocks against in-flight stores.
                if self._block_id_to_pending_jobs:
                    new_blocks_flat = [
                        self._block_identity(group_idx, bid)
                        for group_idx, new_blocks in enumerate(new_block_id_groups)
                        for bid in new_blocks
                    ]
                    if not self._block_id_to_pending_jobs.keys().isdisjoint(
                        new_blocks_flat
                    ):
                        self._current_batch_jobs_to_flush.update(
                            jid
                            for bid in new_blocks_flat
                            for jid in self._block_id_to_pending_jobs.get(bid, ())
                        )

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            # with async scheduling, some tokens may be missing
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)
            max_offload_tokens = req_status.max_offload_tokens
            if max_offload_tokens is not None:
                num_offloadable_tokens = min(num_offloadable_tokens, max_offload_tokens)

            # Skip decode-phase blocks: clamp to the prompt length so only
            # prefill (prompt) blocks become eligible for store. next_stored_idx
            # never advances past this boundary, so decode blocks are never
            # queued in this or any later step.
            if self.config.offload_prompt_only:
                num_offloadable_tokens = min(
                    num_offloadable_tokens, req.num_prompt_tokens
                )

            # Filter out blocks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            # group_idx -> offloaded block indices selected for draft groups
            # under cpu.draft_offload "window" / "none" (ascending).
            draft_selected: dict[int, list[int]] = {}
            for group_config in self.config.kv_group_configs:
                group_state = req_status.group_states[group_config.group_idx]
                if group_config.requires_exact_boundary_source:
                    continue
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                if (
                    group_config.draft_window_tokens is not None
                    and self.config.cpu_draft_offload != "all"
                ):
                    selected = self._select_draft_blocks(
                        req_status, group_config, start_block_idx, num_blocks
                    )
                    draft_selected[group_config.group_idx] = selected
                    new_offload_keys.extend(
                        group_state.offload_keys[b] for b in selected
                    )
                    continue
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                # For each block to offload, take the last corresponding GPU block.
                # e.g. if block size factor is 3 and GPU block IDs are
                # 1 5 6 7 2 4 9 3 8 then we'll take blocks 6 4 8.
                # We will use these GPU blocks to determine if the block needs
                # offloading, or (if the GPU block ID is 0) this block should
                # be skipped due to sliding window attention / SSM.
                # We know that if a block is skipped, then all the previous blocks
                # are skipped as well. This is why we take the last of each block.
                offload_block_ids = group_state.block_ids[
                    start_block_idx * block_size_factor
                    + block_size_factor
                    - 1 : num_blocks * block_size_factor : block_size_factor
                ]
                assert len(offload_keys) == len(offload_block_ids)

                alignment_block_count = group_config.alignment_block_count
                tail = group_config.sliding_window_size_in_blocks

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue
                    # Skip SWA blocks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing `tail` blocks are reachable by
                    # _sliding_window_lookup. For DeepSeek V4 with 100K
                    # tokens this reduces SWA stores by ~78%.
                    if alignment_block_count is not None:
                        assert tail is not None
                        abs_block_idx = start_block_idx + key_idx
                        pos_in_segment = abs_block_idx % alignment_block_count
                        if pos_in_segment < alignment_block_count - tail:
                            continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self._prepare_store(new_offload_keys, req_status)
            if store_output is None:
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            # Offered (stored now or already resident): no longer deferred.
            for group_idx, selected in draft_selected.items():
                deferred = req_status.draft_deferred.get(group_idx)
                if deferred:
                    deferred.difference_update(selected)
                if selected:
                    req_status.draft_reofferable.setdefault(group_idx, set()).update(
                        selected
                    )

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = [0] * self.config.num_kv_cache_groups
            block_indices: list[int] = [0] * self.config.num_kv_cache_groups
            src_block_ids: list[int] = []
            sliding_window_block_ids: list[BlockIdentity] = []
            non_sliding_window_block_ids: list[BlockIdentity] = []
            for group_config in self.config.kv_group_configs:
                group_state = req_status.group_states[group_config.group_idx]
                if group_config.requires_exact_boundary_source:
                    continue
                is_sliding_window = (
                    group_config.sliding_window_size_in_blocks is not None
                )
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                selected_block_idxs = draft_selected.get(group_config.group_idx)
                for offloaded_block_idx in (
                    range(
                        start_block_idx,
                        min(num_blocks, len(group_state.offload_keys)),
                    )
                    if selected_block_idxs is None
                    else selected_block_idxs
                ):
                    offload_key = group_state.offload_keys[offloaded_block_idx]
                    if offload_key not in keys_to_store:
                        continue

                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            # skipped blocks cannot appear after non-skipped blocks
                            assert start_gpu_block_idx is None
                            continue
                        elif start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        identity = self._block_identity(
                            group_config.group_idx, block_id
                        )
                        if is_sliding_window:
                            sliding_window_block_ids.append(identity)
                        else:
                            non_sliding_window_block_ids.append(identity)

                group_sizes[group_config.group_idx] = num_group_blocks
                block_indices[group_config.group_idx] = start_gpu_block_idx or 0
                group_state.next_stored_block_idx = num_blocks

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, transfer_spec=(src_spec, dst_spec)
            )

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

        return store_jobs

    def _draft_window_blocks(
        self, req_status: RequestOffloadState, group_config: GroupOffloadConfig
    ) -> set[int]:
        """Offloaded block indices of a draft group that a prefix hit at one of
        this request's state boundaries (or its junction) needs: the
        ``sliding_window_size_in_blocks`` blocks before each boundary, as
        _sliding_window_lookup requires them."""
        boundaries = set(req_status.draft_window_boundaries)
        junction = getattr(req_status.req, "shared_prefix_boundary", 0)
        if isinstance(junction, int) and junction > 0:
            boundaries.add(junction)
        window = group_config.sliding_window_size_in_blocks
        assert window is not None
        blocks: set[int] = set()
        for boundary in boundaries:
            end = boundary // group_config.offloaded_block_size
            blocks.update(range(max(0, end - window), end))
        return blocks

    def _select_draft_blocks(
        self,
        req_status: RequestOffloadState,
        group_config: GroupOffloadConfig,
        start_block_idx: int,
        num_blocks: int,
    ) -> list[int]:
        """Select the blocks of a draft sliding-window group to offer for
        store under cpu.draft_offload "window" or "none".

        The draft KV cannot be recomputed by the drafter (it is projected from
        target hidden states), so a prefix hit needs the draft window before
        the hit point in the tier. "none" offers nothing. "window" offers the
        window before each boundary whose recurrent state the request handed
        to the host tier, and before its junction. A boundary is offered when
        its chunk is scheduled, after earlier chunks already passed part of
        its window, so blocks passed by the store frontier are deferred and
        offered once a boundary needs them, as long as the request still
        holds them: SlidingWindowManager frees blocks behind the window of the
        processed tokens and the pool may reuse them at once, so a later store
        could read another request's data.

        Every non-null block the frontier passes is offered, remains deferred,
        or is counted in num_skipped_draft_blocks (when it leaves the window,
        the request is preempted or it finishes). After a CPU cache reset,
        still-held offered / loaded blocks may be reconsidered and offered
        again; a block already deferred keeps its accounting obligation.
        """
        group_idx = group_config.group_idx
        group_state = req_status.group_states[group_idx]
        factor = self.config.block_size_factor
        first = max(req_status.draft_frontier.get(group_idx, 0), start_block_idx)
        passed: list[int] = []
        if num_blocks > first:
            req_status.draft_frontier[group_idx] = num_blocks
            # As in _build_store_jobs: the last GPU block of each offloaded
            # block is zero (null) when the request does not hold the block.
            last_block_ids = group_state.block_ids[
                first * factor + factor - 1 : num_blocks * factor : factor
            ]
            assert len(last_block_ids) == num_blocks - first
            passed = [
                first + i for i, block_id in enumerate(last_block_ids) if block_id != 0
            ]
        if self.config.cpu_draft_offload == "none":
            self.num_skipped_draft_blocks += len(passed)
            return []

        deferred = req_status.draft_deferred.setdefault(group_idx, set())
        deferred.update(passed)
        if not deferred:
            return []
        first_held = self._first_held_draft_block(req_status, group_config)
        wanted = self._draft_window_blocks(req_status, group_config)
        selected: list[int] = []
        for block_idx in sorted(deferred):
            if block_idx < first_held:
                deferred.discard(block_idx)
                self.num_skipped_draft_blocks += 1
            elif block_idx < num_blocks and block_idx in wanted:
                selected.append(block_idx)
        return selected

    def _prune_draft_history(self, req_status: RequestOffloadState) -> None:
        # Pending steps can roll back the optimistic computed-token count.
        # Retire accepted records only behind the processed-token boundary
        # used by the core to release sliding-window blocks. The source-read
        # check remains stricter and uses the optimistic boundary below.
        processed_tokens = max(
            0, req_status.req.num_computed_tokens - req_status.req.num_in_flight_tokens
        )
        live_boundaries: set[int] = set()
        for group_config in self.config.kv_group_configs:
            if group_config.draft_window_tokens is None:
                continue
            first_held = self._first_held_draft_block(
                req_status, group_config, num_computed_tokens=processed_tokens
            )
            offered = req_status.draft_reofferable.get(group_config.group_idx)
            if offered:
                # These offers are already settled: retiring their records
                # must not count them as skipped.
                offered.intersection_update({b for b in offered if b >= first_held})
            live_boundaries.update(
                boundary
                for boundary in req_status.draft_window_boundaries
                if boundary // group_config.offloaded_block_size > first_held
            )
        # A past boundary whose whole draft window is gone cannot need a
        # late store. Keep boundaries overlapping any draft group's window.
        req_status.draft_window_boundaries.intersection_update(live_boundaries)

    def _first_held_draft_block(
        self,
        req_status: RequestOffloadState,
        group_config: GroupOffloadConfig,
        *,
        num_computed_tokens: int | None = None,
    ) -> int:
        # SlidingWindowManager.remove_skipped_blocks frees on processed tokens,
        # which never exceed num_computed_tokens here. The connector mirror can
        # still contain non-null IDs for those freed (and possibly reused) pages.
        window_tokens = group_config.draft_window_tokens
        assert window_tokens is not None
        if num_computed_tokens is None:
            num_computed_tokens = req_status.req.num_computed_tokens
        num_skipped_tokens = max(0, num_computed_tokens - window_tokens + 1)
        return cdiv(
            num_skipped_tokens // group_config.gpu_block_size,
            self.config.block_size_factor,
        )

    def _drop_draft_deferred(self, req_status: RequestOffloadState) -> None:
        """Count deferred draft blocks as skipped once the request releases
        them (finished or preempted)."""
        for deferred in getattr(req_status, "draft_deferred", {}).values():
            self.num_skipped_draft_blocks += len(deferred)
            deferred.clear()
        # Accepted blocks have no unsettled obligation. Once ownership ends,
        # a later CPU reset must not requeue their old physical sources.
        getattr(req_status, "draft_reofferable", {}).clear()
        getattr(req_status, "draft_window_boundaries", set()).clear()

    def offload_counter_deltas(self) -> dict[str, dict[str, int]]:
        """Stored blocks by kind, lookup shrink tokens by cause and draft
        window use of hits since the previous call (see OffloadCounters)."""
        counters = getattr(self, "_offload_counters", None)
        return counters.deltas() if counters is not None else {}

    def skipped_draft_blocks_delta(self) -> int:
        """Draft blocks skipped by cpu.draft_offload since the previous call."""
        total = getattr(self, "num_skipped_draft_blocks", 0)
        reported = getattr(self, "_reported_skipped_draft_blocks", 0)
        self._reported_skipped_draft_blocks = total
        return total - reported

    def _queue_boundary_state_offloads(self, scheduler_output: SchedulerOutput) -> None:
        """Move this step's exact Mamba boundary-state hand-offs onto requests."""
        handoffs = scheduler_output.boundary_state_offloads
        if not handoffs:
            return
        for req_id, entries in handoffs.items():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            pending = req_status.pending_boundary_offloads
            for group_idx, block_id, boundary_tokens in entries:
                group_config = self._group_config_by_idx.get(group_idx)
                if (
                    group_config is None
                    or not group_config.requires_exact_boundary_source
                    or block_id == 0
                ):
                    continue
                # A boundary re-offered (e.g. after preemption and resume)
                # supersedes an older pending offer for the same state.
                pending[:] = [
                    entry
                    for entry in pending
                    if entry[0] != group_idx or entry[2] != boundary_tokens
                ]
                pending.append((group_idx, block_id, boundary_tokens))

    def _drop_pending_boundary_offloads(
        self, req_status: RequestOffloadState, reason: str
    ) -> None:
        pending = req_status.pending_boundary_offloads
        if not pending:
            return
        logger.warning(
            "Request %s: dropping %d Mamba boundary state(s) at %s tokens "
            "never accepted by the host tier (%s)",
            req_status.req.request_id,
            len(pending),
            [entry[2] for entry in pending],
            reason,
        )
        pending.clear()

    def _build_boundary_state_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        """Build stores from exact committed Mamba boundary-state blocks.

        A hand-off is tried only in the step the core offers it. The core
        does not pin the source block for the connector: once the step ends,
        the request may free it (e.g. a draft group's sliding window moves on)
        and the block pool may give it to another request. A store job created
        now is covered by ``_block_id_to_pending_jobs``, which flushes it
        before the block is reused; a later retry would not be, and would
        store another request's data under this request's key. So a hand-off
        that the host tier cannot accept this step (its state slots are pinned
        by in-flight loads, stores or a filesystem cascade) is dropped and
        counted. That boundary is then recomputed on a later prefix hit.
        """
        self._queue_boundary_state_offloads(scheduler_output)

        store_jobs: dict[int, TransferJob] = {}
        num_groups = self.config.num_kv_cache_groups
        record_draft_windows = getattr(
            self.config, "cpu_draft_offload", "all"
        ) == "window" and any(
            group.draft_window_tokens is not None
            for group in self.config.kv_group_configs
        )
        for req_id, req_status in self._req_status.items():
            pending = req_status.pending_boundary_offloads
            if not pending:
                continue
            req_status.update_offload_keys()
            req = req_status.req
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            num_offloadable_tokens = min(
                req.num_computed_tokens + num_scheduled_tokens,
                req.num_tokens,
            )
            if req_status.max_offload_tokens is not None:
                num_offloadable_tokens = min(
                    num_offloadable_tokens, req_status.max_offload_tokens
                )
            if self.config.offload_prompt_only:
                num_offloadable_tokens = min(
                    num_offloadable_tokens, req.num_prompt_tokens
                )

            rejected: list[int] = []
            for entry in pending:
                group_idx, block_id, boundary_tokens = entry
                group_config = self._group_config_by_idx[group_idx]
                # Neither the offload cap nor the block alignment changes
                # later: these offers can never be stored.
                if (
                    boundary_tokens > num_offloadable_tokens
                    or boundary_tokens % group_config.offloaded_block_size != 0
                ):
                    continue

                offload_block_idx = (
                    boundary_tokens // group_config.offloaded_block_size - 1
                )
                group_state = req_status.group_states[group_idx]
                if offload_block_idx >= len(group_state.offload_keys):
                    continue
                offload_key = group_state.offload_keys[offload_block_idx]
                store_output = self._prepare_store([offload_key], req_status)
                if store_output is None:
                    rejected.append(boundary_tokens)
                    continue
                if record_draft_windows:
                    # Stored now or already resident: keep its draft window.
                    req_status.draft_window_boundaries.add(boundary_tokens)
                keys_to_store = set(store_output.keys_to_store)
                if not keys_to_store:
                    continue
                assert keys_to_store == {offload_key}

                req_status.stored_state_blocks.setdefault(group_idx, set()).add(
                    offload_block_idx
                )
                job_id = self._generate_job_id()
                req_status.transfer_jobs.add(job_id)
                identity = self._block_identity(group_idx, block_id)
                self._block_id_to_pending_jobs.setdefault(identity, set()).add(job_id)
                self._jobs[job_id] = TransferJobStatus(
                    req_id=req_id,
                    pending_count=self.config.num_workers,
                    keys=keys_to_store,
                    is_store=True,
                    sliding_window_block_ids=[identity],
                )

                group_sizes = [0] * num_groups
                group_sizes[group_idx] = 1
                block_indices = [0] * num_groups
                block_indices[group_idx] = (
                    boundary_tokens // group_config.gpu_block_size - 1
                )
                src_spec = GPULoadStoreSpec(
                    [block_id],
                    group_sizes=group_sizes,
                    block_indices=block_indices,
                )
                store_jobs[job_id] = TransferJob(
                    req_id=req_id,
                    transfer_spec=(src_spec, store_output.store_spec),
                )
            pending.clear()
            if rejected:
                self.num_rejected_boundary_offers += len(rejected)
                logger.info(
                    "Request %s: host tier busy, dropped Mamba boundary "
                    "state(s) at %s tokens (%d dropped since start)",
                    req_id,
                    rejected,
                    self.num_rejected_boundary_offers,
                )

        return store_jobs

    def _cpu_tier_managers(self) -> list[Any]:
        """Leaf CPU-tier managers, unwrapping group pools and tiering."""
        stack, leaves = [self.manager], []
        while stack:
            m = stack.pop()
            if hasattr(m, "managers"):  # GroupedCPUOffloadingManager
                stack.extend(m.managers.values())
                continue
            leaves.append(getattr(m, "primary_tier", m))  # TieringOffloadingManager
        return leaves

    def cpu_tier_eviction_deltas(self) -> dict[str, int]:
        """CPU-tier evicted blocks by reason since the previous call.

        Read-only: sums the managers' cumulative per-reason counters and
        returns the increase, so a counter fed with the deltas matches the
        cumulative totals.
        """
        totals: dict[str, int] = {}
        for m in self._cpu_tier_managers():
            for reason, n in getattr(m, "evicted_blocks_by_reason", {}).items():
                totals[reason] = totals.get(reason, 0) + n
        reported: dict[str, int] | None = getattr(self, "_reported_evictions", None)
        if reported is None:
            reported = {}
            self._reported_evictions = reported
        deltas = {r: n - reported.get(r, 0) for r, n in totals.items()}
        reported.update(totals)
        return {r: d for r, d in deltas.items() if d > 0}

    def _cpu_tier_counters(self) -> tuple[int, int]:
        """(stored, evicted) CPU-tier block counts, summed over group pools."""
        stack = [self.manager]
        stored = evicted = 0
        while stack:
            m = stack.pop()
            if hasattr(m, "managers"):  # GroupedCPUOffloadingManager
                stack.extend(m.managers.values())
                continue
            m = getattr(m, "primary_tier", m)  # TieringOffloadingManager
            stored += getattr(m, "num_stored_blocks", 0)
            evicted += getattr(m, "num_evicted_blocks", 0)
        return stored, evicted

    def _maybe_log_counters(self) -> None:
        now = time.monotonic()
        last = getattr(self, "_counters_logged_at", None)
        if last is None:
            # Objects built without __init__ (tests) start counting now.
            self._counters_logged_at = now
            return
        if now - last < 60.0:
            return
        self._counters_logged_at = now
        stored, evicted = self._cpu_tier_counters()
        residency_suffix = ""
        if getattr(self, "_dependency_index", None) is not None:
            self.cpu_residency_snapshot(consume=False)
            totals = self._residency_totals
            residency_suffix = " cpu_resident_blocks=" + str(sum(totals.values()))
            residency_suffix += " cpu_residency=" + str(totals)
        counters = getattr(self, "_offload_counters", None)
        counter_suffix = ""
        if counters is not None:
            by_kind = counters.stored_blocks
            shrink = counters.lookup_shrink_tokens
            needed = counters.hit_draft_window_needed
            counter_suffix = (
                f" stored_blocks_by_kind=target:{by_kind['target']},"
                f"draft:{by_kind['draft']},state:{by_kind['state']} "
                f"lookup_shrink_tokens=draft_window:{shrink['draft_window']},"
                f"recurrent_state:{shrink['recurrent_state']},"
                f"sliding_window:{shrink['sliding_window']},"
                f"page_align:{shrink['page_align']} "
                f"hit_draft_window_needed=true:{needed['true']},false:{needed['false']}"
            )
        logger.info(
            "Offload counters: demoted_states=%d cpu_stored_blocks=%d "
            "cpu_evicted_blocks=%d rejected_boundary_offers=%d rule=%s "
            "cpu_draft_blocks_skipped=%d cpu.draft_offload=%s%s%s",
            getattr(self, "num_demoted_states", 0),
            stored,
            evicted,
            # Mamba boundary-state hand-offs the host tier could not accept.
            getattr(self, "num_rejected_boundary_offers", 0),
            self.config.superseded_state_rule,
            # Draft-model blocks not offloaded (cpu.draft_offload).
            getattr(self, "num_skipped_draft_blocks", 0),
            getattr(self.config, "cpu_draft_offload", "all"),
            counter_suffix,
            residency_suffix,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._maybe_log_counters()
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            # Preemption returns the request's blocks to the pool, so
            # boundary states not yet handed to a store job are gone.
            self._drop_pending_boundary_offloads(req_status, "preempted")
            # A request preempted outside schedule() (reset_prefix_cache with
            # reset_running_requests) is reported one call later. With no
            # transfer left, that call may already have re-admitted it with a
            # new load into freshly allocated blocks. The load reads no freed
            # block, and workers start it only after this step's flushes.
            if req_status.transfer_jobs and req_status.transfer_jobs.issubset(
                self._current_batch_load_jobs
            ):
                # Lookup already cleared the old admission's draft records;
                # preserve the window registered by this fresh load.
                continue
            self._drop_draft_deferred(req_status)
            if not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # If all tracked requests are finished, flush all pending jobs
        # (both store and load) - there might not be a future scheduler
        # step to trigger their completion.
        if self._req_status and all(
            rs.req.is_finished() for rs in self._req_status.values()
        ):
            self._current_batch_jobs_to_flush.update(self._jobs.keys())

        # The worker zeroes newly allocated pages before this step's stores
        # run. A page can be allocated while an earlier store still reads it
        # without passing through the new-block fence in _build_store_jobs: a
        # request parked for an async load is not scheduled, and groups that
        # are not offloaded (e.g. the QSA compressor ring) are not fenced by
        # update_state_after_alloc. Fence every page announced for zeroing.
        new_block_ids_to_zero = getattr(scheduler_output, "new_block_ids_to_zero", None)
        if new_block_ids_to_zero and self._block_id_to_pending_jobs:
            for block_id in new_block_ids_to_zero:
                segments = getattr(self, "_block_segments", None)
                identity = (
                    (segments.primary, block_id) if segments is not None else block_id
                )
                self._current_batch_jobs_to_flush.update(
                    self._block_id_to_pending_jobs.get(identity, ())
                )

        boundary_store_jobs = self._build_boundary_state_store_jobs(scheduler_output)
        normal_store_jobs = self._build_store_jobs(scheduler_output)
        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=boundary_store_jobs | normal_store_jobs,
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        completed_receive_reqs = set()
        completed_jobs = dict(meta.completed_jobs)
        host_hits = self.host_kv_hits
        manages_receives = host_hits is not None and host_hits.history_pool is not None
        if manages_receives:
            assert host_hits is not None and host_hits.history_pool is not None
            table = host_hits.history_pool.host_pages
            for attempt, (completed, failed) in meta.receive_results.items():
                status = self._jobs.get(attempt)
                if status is not None and status.receive_txn is not None:
                    status.receive_txn.complete(completed, failed, table.all_ranks)
            # Counts remain the ABI of ordinary transfer jobs. ReceiveTxn
            # ignores them, including duplicates of an already retired job.
            for job_id in list(completed_jobs):
                status = self._jobs.get(job_id)
                if job_id in meta.receive_results or (
                    status is not None and status.receive_txn is not None
                ):
                    del completed_jobs[job_id]
            for job_id, status in self._jobs.items():
                txn = status.receive_txn
                if (
                    txn is not None
                    and txn.completed_ranks == table.all_ranks
                    and table.watermark >= txn.step
                ):
                    status.failed = bool(txn.failed_ranks)
                    completed_jobs[job_id] = self.config.num_workers
        for job_id, count in completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                self._complete_stale_load(job_id, count, job_id in meta.failed_jobs)
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            if job_id in meta.failed_jobs:
                job_status.failed = True
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store and job_status.failed:
                # The tier never received valid data: drop the keys rather
                # than publishing them as loadable.
                self.manager.complete_store(
                    job_status.keys, req_status.req_context, success=False
                )
            elif job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
                counters = getattr(self, "_offload_counters", None)
                if counters is not None:
                    counters.count_stored(job_status.keys)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(job_status.keys)
                if job_status.failed:
                    # A worker could not read the keys back: the GPU blocks
                    # hold no valid prefix. The scheduler recomputes.
                    self._abandon_load(job_status.req_id, "transfer failed")
                if job_status.receive_txn is not None:
                    assert host_hits is not None
                    hook = self.value_restore_hook
                    if hook is not None and not job_status.failed:
                        # value arbitration: a finally admitted, complete receive; its
                        # NEW_LOAD entries did the index H2D.
                        hook(job_status.receive_txn)
                    host_hits.retire_receive(job_status.receive_txn)
                    completed_receive_reqs.add(job_status.req_id)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                identities = set(job_status.sliding_window_block_ids or ())
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    identities.update(job_status.non_sliding_window_block_ids or ())
                self._remove_pending_job(job_id, identities)

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if not req_status.transfer_jobs and req_status.req.is_finished():
                del self._req_status[job_status.req_id]
        if manages_receives:
            # A worker may finish before other ranks or the step watermark.
            # Emit one request completion only after the transaction retires.
            connector_output.finished_recving = completed_receive_reqs

    def _complete_stale_load(self, job_id: int, count: int, failed: bool) -> None:
        """Account a worker report of a load job discarded by reset_cache.

        The reset dropped the job's bookkeeping, not its outcome: the request
        is still promoted by finished_recving once every rank is done, so a
        failure on any rank must void the external hit exactly as it would
        without the reset.
        """
        stale_loads = getattr(self, "_stale_load_jobs", None)
        if not stale_loads:
            return
        job_status = stale_loads.get(job_id)
        if job_status is None:
            return
        job_status.failed |= failed
        job_status.pending_count -= count
        if job_status.pending_count > 0:
            return
        del stale_loads[job_id]
        if job_status.failed:
            self._abandon_load(job_status.req_id, "transfer failed after a reset")

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        if getattr(self, "_recompute_failed_loads", False):
            self._skip_external_once.discard(request.request_id)
            self._failed_loads.discard(request.request_id)
        req_status = self._req_status.get(request.request_id)

        req_context = (
            req_status.req_context if req_status else _create_req_context(request)
        )
        self.manager.on_request_finished(req_context)

        if req_status is None:
            return False, None
        if self.config.demote_superseded_states:
            self._demote_superseded_states(req_status)
        if getattr(self.manager, "state_latest_only", False) is True:
            self._drop_superseded_state_writes(req_status)
        # Blocks are freed right after this call; a store for a still
        # pending boundary offer could not be issued before block reuse.
        self._drop_pending_boundary_offloads(req_status, "request finished")
        self._drop_draft_deferred(req_status)
        if not req_status.transfer_jobs:
            del self._req_status[request.request_id]
            return False, None
        # Pending stores will outlive the request's block ownership.
        # Register them so future block reuse triggers a flush.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            if job_status.receive_txn is not None:
                job_status.receive_txn.cancelled = True
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        return False, None

    # Replay boundaries (resend and extension) sit within this many blocks of
    # the request's last full block. Older request-owned checkpoints can be
    # superseded, but prefix hits and shared junctions still serve siblings.
    _KEEP_TAIL_STATE_BLOCKS = 3

    def _hit_states_of_others(self, req_status: RequestOffloadState) -> set[OffloadKey]:
        """Prefix-hit state keys of every other tracked request (exact-boundary
        recurrent groups): checkpoints those requests resume from."""
        keys: set[OffloadKey] = set()
        for other in getattr(self, "_req_status", {}).values():
            if other is req_status:
                continue
            for group_config in self.config.kv_group_configs:
                if group_config.sliding_window_size_in_blocks != 1:
                    continue
                if not group_config.requires_exact_boundary_source:
                    continue
                group_state = other.group_states[group_config.group_idx]
                hit = group_state.num_hit_blocks
                if 0 < hit <= len(group_state.offload_keys):
                    keys.add(group_state.offload_keys[hit - 1])
        return keys

    def _demote_superseded_states(self, req_status: RequestOffloadState) -> None:
        """Demote the recurrent states this request made obsolete (soft: the
        CPU tier evicts them first; nothing is deleted).

        ``superseded_state_rule`` (OffloadingSpec) selects the states:
          request        own states in [hit, last block - 3); the last block
                         counts decode output (historical behaviour)
          request_fixed  the same, with the last block taken from the prompt:
                         decode-time states are never offloaded
                         (offload_prompt_only), so counting output demoted the
                         prompt-end replay states the next turn resumes from
          extend         request_fixed + the state at the hit point, once this
                         request stored a newer state past it: a continued
                         conversation no longer needs its previous end state
        The junction (shared prefix) is never demoted. Under request_fixed and
        extend, neither is the prefix-hit state of another active request
        (it may still need it, e.g. after preemption), and extend never
        demotes this prompt's resend checkpoint: an identical resend is capped
        at prompt_length - 1 and resumes from it (one block earlier with an
        EAGLE/MTP shift).
        """
        rule = getattr(self.config, "superseded_state_rule", "request")
        keys: list[OffloadKey] = []
        in_use = set() if rule == "request" else self._hit_states_of_others(req_status)
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks != 1:
                continue
            if not group_config.requires_exact_boundary_source:
                continue
            group_state = req_status.group_states[group_config.group_idx]
            group_keys = group_state.offload_keys
            keep = self._KEEP_TAIL_STATE_BLOCKS
            if rule == "request":
                last = len(group_keys)
            else:
                last = min(
                    len(group_keys),
                    req_status.req.num_prompt_tokens
                    // group_config.offloaded_block_size,
                )
            # Hits include GPU and external prefixes. The junction is already
            # on the usable state boundary, including the EAGLE/MTP shift.
            junction = req_status.req.shared_prefix_boundary
            hit = group_state.num_hit_blocks

            def is_junction(block_idx: int, gc=group_config, jn=junction) -> bool:
                return bool(jn) and (block_idx + 1) * gc.offloaded_block_size == jn

            demote = [b for b in range(hit, last - keep) if not is_junction(b)]
            if rule == "extend" and hit > 0:
                stored = req_status.stored_state_blocks.get(group_config.group_idx)
                # An identical resend is capped at prompt_length - 1 and
                # resumes from this prompt's resend boundary; with an
                # EAGLE/MTP shift the usable state is one block earlier.
                resend = (req_status.req.num_prompt_tokens - 1) // (
                    group_config.offloaded_block_size
                ) - 1
                resend_blocks = {resend}
                if getattr(self, "_eagle_state_shift", None) is not None:
                    resend_blocks.add(resend - 1)
                if (
                    stored
                    and max(stored) >= hit
                    and hit - 1 < len(group_keys)
                    and hit - 1 not in resend_blocks
                    and not is_junction(hit - 1)
                ):
                    demote.append(hit - 1)
            keys.extend(group_keys[b] for b in demote if group_keys[b] not in in_use)
        if keys:
            self.num_demoted_states = getattr(self, "num_demoted_states", 0) + len(keys)
            self.manager.demote(keys, req_status.req_context)

    def has_pending_persistence(self) -> bool:
        """Delayed secondary-tier writes are queued or in flight: the idle
        engine wakes about once a second to service them (EngineCoreProc)."""
        pending = getattr(self.manager, "has_pending_delayed_writes", None)
        # host SSD persistence: the host-page SSD queue (attached by EngineCore) also
        # counts.
        host_ssd = getattr(self, "host_page_ssd", None)
        return (bool(pending()) if pending is not None else False) or (
            host_ssd is not None and host_ssd.has_pending()
        )

    def persistence_step(self) -> None:
        """Idle engine, scheduler thread: write due rows, poll completions."""
        step = getattr(self.manager, "delayed_write_step", None)
        if step is not None:
            step()
        host_ssd = getattr(self, "host_page_ssd", None)
        if host_ssd is not None:
            host_ssd.step()

    def _drop_superseded_state_writes(self, req_status: RequestOffloadState) -> None:
        """state_latest_only (secondary tiers with delayed writes): of the
        recurrent states this request stored, only the newest is written to
        the secondary tier; the older ones (tail / resend checkpoints the CPU
        tier keeps on purpose) and the state at the hit point it extended are
        dropped from the pending write queues. CPU residency is unchanged.
        The junction and the prefix-hit states of other active requests are
        never dropped."""
        in_use = self._hit_states_of_others(req_status)
        junction = req_status.req.shared_prefix_boundary
        keys: list[OffloadKey] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks != 1:
                continue
            if not group_config.requires_exact_boundary_source:
                continue
            stored = req_status.stored_state_blocks.get(group_config.group_idx)
            if not stored:
                continue
            group_keys = req_status.group_states[group_config.group_idx].offload_keys
            hit = req_status.group_states[group_config.group_idx].num_hit_blocks
            latest = max(stored)
            older = {b for b in stored if b != latest}
            if 0 < hit <= latest:
                older.add(hit - 1)
            for b in sorted(older):
                if (
                    bool(junction)
                    and (b + 1) * group_config.offloaded_block_size == junction
                ):
                    continue
                if b < len(group_keys) and group_keys[b] not in in_use:
                    keys.append(group_keys[b])
        drop = getattr(self.manager, "drop_pending_secondary_writes", None)
        if keys and drop is not None:
            drop(keys)

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            block_hashes = [get_offload_block_hash(key) for key in event.keys]
            if event.removed:
                yield BlockRemoved(block_hashes=block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=0,
                    medium=event.medium,
                    lora_name=None,
                )

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored blocks."""

        if any(status.receive_txn is not None for status in self._jobs.values()):
            raise RuntimeError("Receive transactions must retire before cache reset")
        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress for active requests. Keep draft scan frontiers:
        # rewinding them can recount settled skips, especially when optimistic
        # computed tokens roll back after speculative-token rejection.
        finished_req_ids = []
        for req_id, status in self._req_status.items():
            status.transfer_jobs.clear()
            if status.req.is_finished():
                finished_req_ids.append(req_id)
            elif self.config.cpu_draft_offload == "window":
                for group_config in self.config.kv_group_configs:
                    group_idx = group_config.group_idx
                    if group_config.draft_window_tokens is None:
                        continue
                    offered = status.draft_reofferable.get(group_idx)
                    if not offered:
                        continue
                    first_held = self._first_held_draft_block(status, group_config)
                    status.draft_deferred.setdefault(group_idx, set()).update(
                        b for b in offered if b >= first_held
                    )
                    offered.clear()
                # Existing deferred blocks keep their accounting obligation;
                # previously accepted held blocks start a new one. The selector
                # checks the held boundary again before issuing any source read.
            for group_state in status.group_states:
                group_state.next_stored_block_idx = 0
        for req_id in finished_req_ids:
            del self._req_status[req_id]
        self._skip_external_once.intersection_update(self._req_status)
        self._failed_loads.intersection_update(self._req_status)

        # A load still in flight keeps its failure path: remember its request
        # so a failed completion after the reset is abandoned, not promoted.
        if getattr(self, "_recompute_failed_loads", False):
            stale_loads = self.__dict__.setdefault("_stale_load_jobs", {})
            for job_id, job_status in self._jobs.items():
                if not job_status.is_store:
                    stale_loads[job_id] = job_status

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
