# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity plan and joint admission for the long-context (Active Host KV) mode.

In long-context mode the compressed-sparse (QSA) block space is split into two
domains: a GPU *state segment* (recurrent GDN/PLE states and drafter windows)
and a *history segment* whose block IDs address the compressed index and ring
on the GPU and the main QSA K/V in a host pool. A GPU hot-page pool caches main
K/V pages for the attention gathers. This module only does the bookkeeping:

* ``LongContextGeometry`` - per-block byte costs and per-request demands,
  derived from the resolved KV cache groups (see
  ``kv_cache_utils.get_long_context_geometry``).
* ``LongContextCapacityPlan`` - how a GPU budget, a host pool and a hot pool
  split into domain capacities, and whether one ``max_model_len`` request fits.
* ``LongContextAdmissionLedger`` - per-domain usage, kept exact by observing
  the block pool's reference-count transitions, and the admission test used by
  ``KVCacheManager.allocate_slots``. A request that does not fit makes
  ``allocate_slots`` return None, so the scheduler queues or preempts it under
  its existing rules; nothing here raises at runtime.

Physically the GPU KV cache is split into two block spaces
(``KVCacheConfig.block_segments``): the state segment and the history segment
each have their own block pool, null block, free queue, prefix-cache map and
eviction, and their own tensors on every worker. Those two domains are
therefore enforced by the pools themselves; the ledger only enforces the host
history pool and the hot-page pool.

Main K/V placement (``LongContextKVConfig.main_kv_placement``):

* ``host`` (default): every history block ID owns one host page per QSA layer
  on each worker (``HostHistoryStore``), addressed by the same ID as the
  block's index pages; a history block costs index pages plus the host
  store's per-block GPU metadata, and the hot-page pool is real GPU memory.
  Because pages are addressed directly, the history segment can never hold
  more IDs than the host pool has pages: its usable blocks are capped at
  ``host_history_blocks - 1`` (one page backs the null block), and ring
  blocks take a page too.
* ``gpu``: the main K/V pages stay in the GPU history segment next to the
  index (the host-history integration split before the host store existed, kept as the
  same-plan
  reference arm). A history block costs index + main K/V bytes; the hot-page
  and host pools are admission budgets only.

Block IDs, hashes and the scheduler side are the same for both.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config.kernel import LongContextKVConfig
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)

GiB = 1 << 30

# Block domains. A history block of the main-KV group counts in both the GPU
# history segment (its index pages) and the host pool (its main K/V pages).
STATE = 0
GPU_HISTORY = 1
HOST = 2
DOMAIN_NAMES = ("GPU state segment", "GPU history segment", "host history pool")
HOT_POOL_NAME = "GPU hot-page pool"

# Physical block spaces (``KVCacheConfig.block_segments``) in long-context mode.
STATE_SEGMENT = 0
HISTORY_SEGMENT = 1
SEGMENT_NAMES = ("state", "history")
# Index of the hot-page pool in blocked-admission counters.
HOT = 3

# Local tokens of newest history pinned in the hot pool per request (host KV pool), and
# the groups they can touch when not aligned to a group boundary.
HOT_TAIL_TOKENS = 32


def _fmt_gib(num_bytes: float) -> str:
    return f"{num_bytes / GiB:.2f} GiB"


def _ceil_gib(num_bytes: float) -> float:
    """Round a byte count up to a GiB value with two decimals."""
    return cdiv(int(num_bytes) * 100, GiB) / 100


@dataclass(frozen=True)
class LongContextGeometry:
    """Per-block costs and per-request demands of one deployment.

    Byte counts are per worker (GPU) unless named otherwise; the host pool is
    the sum over ``num_workers`` worker shards.
    """

    block_tokens: int
    """Global tokens addressed by one history block ID."""
    num_workers: int
    """Workers holding a main-K/V shard (TP x PP ranks, including DCP ranks)."""
    main_kv_bytes_per_block: int
    """Main QSA K/V bytes of one history block on one worker (all layers)."""
    index_bytes_per_block: int
    """GPU bytes of one history block on one worker: compressed index pages,
    shared with the compressor ring."""
    state_bytes_per_block: int
    """GPU bytes of one state block ID on one worker: one owner-shaped page
    for every physical main-KV owner that hosts a state or drafter layer,
    plus the standalone drafter pages."""
    unified_bytes_per_block: int
    """Bytes of one block in today's unified pool, for reference."""
    state_blocks_per_request: int
    """Worst-case state block IDs of one request (GDN, PLE, drafter)."""
    ring_blocks_per_request: int
    """History block IDs a request holds for its compressor ring(s)."""
    max_model_len: int
    """Context limit the startup check must cover."""
    hot_group_tokens: int
    """Local tokens per physical hot slot, currently four."""
    hot_row_groups: int
    """Distinct groups one query row can select (top-k groups plus the
    uncompressed tail)."""
    decode_rows_per_request: int
    """Query rows of one request in a decode/verify step (1 + draft tokens)."""
    group_domains: tuple[tuple[int, ...], ...]
    """Domains charged by each KV cache group, in group order."""
    dcp_world_size: int = 1
    hot_layers: int = 1
    hot_max_write_rows: int = 1
    hot_query_heads: int = 1
    hot_head_dim: int = 256
    hot_kv_heads: int = 1
    dcp_interleave: int = 1
    hot_read_rows: int | None = 256
    logical_index_bytes_per_block: int = 0
    index_page_strides: tuple[int, ...] = ()
    """Physical GPU index page stride per layer (P_pad), host placement only."""
    state_inflight_blocks: int = 0
    """Extra state IDs one request holds while an asynchronous frame is in
    flight (one per aligned recurrent group). A segment for one request
    needs them, or that request preempts itself forever."""
    """Host ABI index bytes, excluding GPU page padding (host placement only)."""

    @property
    def shared_slots_per_block(self) -> int:
        """k for the shared hot region shared region: hot slots one index page holds, or
        0
        when the index pages cannot alias one hot slot grid per layer."""
        strides = self.index_page_strides
        slot = self.hot_slot_bytes // max(self.hot_layers, 1)
        if (
            not strides
            or len(strides) != self.hot_layers
            or len(set(strides)) != 1
            or slot <= 0
            or self.hot_slot_bytes % self.hot_layers
            or strides[0] % slot
        ):
            return 0
        return strides[0] // slot

    def shared_slot_metadata_bytes(self, max_num_seqs: int) -> int:
        """Per-slot device metadata of the shared region (slot owner, age,
        gather scratch, identity hot table), charged per history ID as k x."""
        layers = self.hot_layers
        return 12 * layers + 16 + 4 * max_num_seqs

    @property
    def local_block_tokens(self) -> int:
        """Rank-local main K/V tokens of one history block."""
        return self.block_tokens // self.dcp_world_size

    @property
    def host_metadata_bytes_per_block(self) -> int:
        """Per-ID hot maps, wanted map, readiness bytes and generation."""
        from vllm.models.qwen4_exp.nvidia.host_kv.budget import history_metadata_bytes

        return sum(
            history_metadata_bytes(self.local_block_tokens, self.hot_layers).values()
        )

    @property
    def host_bytes_per_block(self) -> int:
        """Host bytes of one history block summed over all workers."""
        return (
            self.main_kv_bytes_per_block + self.logical_index_bytes_per_block
        ) * self.num_workers

    @property
    def hot_tail_groups(self) -> int:
        return cdiv(HOT_TAIL_TOKENS, self.hot_group_tokens) + 1

    def history_blocks(self, num_tokens: int) -> int:
        """Main-KV history block IDs for ``num_tokens`` of context."""
        return cdiv(num_tokens, self.block_tokens)

    def hot_reserve_groups(self, num_tokens: int) -> int:
        """Hot-pool groups one request reserves (per layer, per worker).

        A hot group holds ``hot_group_tokens`` rank-local tokens. Every group
        a decode/verify step selects must be resident while the attention
        reads it, and the newest tokens stay pinned. A request can never need
        more groups than its own local history has.
        """
        from vllm.models.qwen4_exp.nvidia.host_kv.planning import max_local_tokens

        groups = cdiv(
            max_local_tokens(num_tokens, self.dcp_world_size, self.dcp_interleave),
            self.hot_group_tokens,
        )
        bound = self.decode_rows_per_request * min(self.hot_row_groups, groups) + min(
            self.hot_tail_groups, groups
        )
        return min(groups, bound)

    @property
    def hot_slot_bytes(self) -> int:
        """All layers' payload for one worker-local four-token slot."""
        return cdiv(
            self.main_kv_bytes_per_block * self.dcp_world_size * self.hot_group_tokens,
            self.block_tokens,
        )

    def hot_memory_bytes(self, groups: int, max_num_seqs: int = 1) -> dict[str, int]:
        """Fixed HostHistoryStore allocations per worker, including shared scratch.

        History-ID metadata is charged in history_bytes_per_block. Attention
        workspaces and CUDA graph pools outside this store are separate;
        allocator rounding is not a logical tensor allocation.
        """
        from vllm.models.qwen4_exp.nvidia.host_kv.budget import (
            history_fixed_memory_bytes,
        )

        parts = history_fixed_memory_bytes(
            slots=groups,
            num_layers=self.hot_layers,
            slot_bytes=self.hot_slot_bytes,
            max_requests=max_num_seqs,
            selection_width=4 * (self.hot_row_groups - 1) + 3,
            max_write_rows=self.hot_max_write_rows,
            row_groups=self.hot_row_groups,
            tail_groups=self.hot_tail_groups,
            read_rows=self.hot_read_rows,
        )
        payload = parts.pop("hot")
        workspace = sum(
            parts.pop(name) for name in ("indices", "staging", "staging_slots")
        )
        return {
            "payload": payload,
            "metadata": sum(parts.values()),
            "workspace": workspace,
        }

    def hot_groups_for_bytes(self, num_bytes: int, max_num_seqs: int = 1) -> int:
        """Largest slot count covered by the full per-worker device budget."""
        low, high = 0, num_bytes // self.hot_slot_bytes
        while low < high:
            mid = (low + high + 1) // 2
            if self.hot_bytes_for_groups(mid, max_num_seqs) <= num_bytes:
                low = mid
            else:
                high = mid - 1
        return low

    def hot_bytes_for_groups(self, groups: int, max_num_seqs: int = 1) -> int:
        return sum(self.hot_memory_bytes(groups, max_num_seqs).values())


@dataclass(frozen=True)
class LongContextCapacityPlan:
    """Domain capacities for one GPU budget and the configured pools."""

    geometry: LongContextGeometry
    max_num_seqs: int
    gpu_budget_bytes: int
    """KV cache memory of one worker that the GPU domains split."""
    host_pool_bytes: int
    """Host history pool summed over all workers."""
    hot_pool_bytes: int
    staging_bytes: int
    state_blocks: int
    """Usable state block IDs (a null block is charged separately)."""
    gpu_history_blocks: int
    """Usable GPU history block IDs (after the null block)."""
    host_history_blocks: int
    hot_pool_groups: int
    prewarm_hot_pool: bool = True
    main_kv_on_gpu: bool = False
    """Main QSA K/V pages stay in the GPU history segment (addressed by the
    history block ID, next to the index pages) instead of the host pool
    (``main_kv_placement="gpu"``); the hot pool then takes no GPU memory and
    only budgets admission."""
    hot_pool_source: Literal["manual", "auto"] = "manual"
    """Whether the hot pool was explicitly sized or took the remaining budget."""
    hot_slots_per_block: int = 0
    """shared hot region shared region: k hot slots alias each DONATED history ID's
    index
    page; 0 keeps the static hot pool. ``hot_pool_bytes`` then holds only the
    region's fixed scratch, and ``hot_pool_groups`` the largest region."""
    hot_pool_min_groups: int = 0
    """Guaranteed slot floor. The static static host allocation pool may exceed this
    floor; shared hot region
    can grow/shrink the shared hot region without changing this contract."""

    def hot_floor_groups(self, reserved_groups: int) -> int:
        """shared hot region's shared-region floor; static host allocation still uses
        static physical allocation."""
        return max(self.hot_pool_min_groups, reserved_groups)

    @property
    def gpu_state_bytes(self) -> int:
        return (self.state_blocks + 1) * self.geometry.state_bytes_per_block

    @property
    def history_bytes_per_block(self) -> int:
        """GPU bytes of one history block ID on one worker."""
        geo = self.geometry
        return (
            geo.index_bytes_per_block
            + (
                geo.main_kv_bytes_per_block
                if self.main_kv_on_gpu
                else geo.host_metadata_bytes_per_block
            )
            + self.hot_slots_per_block
            * geo.shared_slot_metadata_bytes(self.max_num_seqs)
        )

    @property
    def gpu_history_budget_blocks(self) -> int:
        """Usable history IDs the GPU budget alone would allow."""
        return max(self.gpu_history_budget_bytes // self.history_bytes_per_block - 1, 0)

    @property
    def host_page_blocks(self) -> int:
        """Usable history IDs the host pool backs with a page (host
        placement: every ID, the null block included, owns one page)."""
        return max(self.host_history_blocks - 1, 0)

    @property
    def gpu_hot_pool_bytes(self) -> int:
        """GPU memory the hot-page pool takes (none while main K/V is on GPU)."""
        return 0 if self.main_kv_on_gpu else self.hot_pool_bytes

    @property
    def gpu_history_budget_bytes(self) -> int:
        """GPU bytes left for the history segment (may be negative)."""
        return (
            self.gpu_budget_bytes
            - self.gpu_state_bytes
            - self.gpu_hot_pool_bytes
            - self.staging_bytes
        )

    @property
    def gpu_history_bytes(self) -> int:
        return (self.gpu_history_blocks + 1) * self.history_bytes_per_block

    @property
    def history_demand_blocks(self) -> int:
        """History and ring IDs for max_num_seqs full-length requests."""
        geo = self.geometry
        blocks = self.max_num_seqs * (
            geo.history_blocks(geo.max_model_len) + geo.ring_blocks_per_request
        )
        return blocks if self.main_kv_on_gpu else min(blocks, self.host_page_blocks)

    @property
    def gpu_remaining_bytes(self) -> int:
        return self.gpu_history_budget_bytes - self.gpu_history_bytes

    def segment_num_blocks(self) -> tuple[int, int]:
        """Blocks of the (state, history) segments, null blocks included."""
        return self.state_blocks + 1, self.gpu_history_blocks + 1

    def group_segments(self) -> tuple[int, ...]:
        """Physical segment of each KV cache group."""
        return tuple(
            STATE_SEGMENT if STATE in domains else HISTORY_SEGMENT
            for domains in self.geometry.group_domains
        )

    def capacity(self, domain: int) -> int:
        return (self.state_blocks, self.gpu_history_blocks, self.host_history_blocks)[
            domain
        ]

    def max_concurrency(self, num_tokens: int) -> dict[str, int]:
        """Arithmetic domain bounds for decode. Physical concurrency is unverified."""
        geo = self.geometry
        history = geo.history_blocks(num_tokens)
        return {
            DOMAIN_NAMES[STATE]: self.state_blocks
            // max(geo.state_blocks_per_request, 1),
            DOMAIN_NAMES[GPU_HISTORY]: self.gpu_history_blocks
            // (history + geo.ring_blocks_per_request),
            DOMAIN_NAMES[HOST]: self.host_history_blocks // history,
            HOT_POOL_NAME: (
                self.gpu_history_blocks
                // (
                    history
                    + geo.ring_blocks_per_request
                    + cdiv(geo.hot_reserve_groups(num_tokens), self.hot_slots_per_block)
                )
                if self.hot_slots_per_block
                else self.hot_pool_groups // geo.hot_reserve_groups(num_tokens)
            ),
            "max_num_seqs": self.max_num_seqs,
        }

    def fits(self, num_tokens: int) -> bool:
        return min(self.max_concurrency(num_tokens).values()) >= 1

    def max_fitting_len(self) -> int:
        """Largest context one request can have in every domain (0 if none)."""
        if not self.fits(1):
            return 0
        low, high = 1, self.geometry.max_model_len
        while low < high:
            mid = (low + high + 1) // 2
            if self.fits(mid):
                low = mid
            else:
                high = mid - 1
        return low

    def startup_problems(self) -> list[str]:
        """Why one ``max_model_len`` request cannot be held, with suggestions."""
        geo = self.geometry
        max_len = geo.max_model_len
        history = geo.history_blocks(max_len)
        problems: list[str] = []
        ring = geo.ring_blocks_per_request
        if not self.main_kv_on_gpu and self.host_page_blocks < history + ring:
            problems.append(
                f"{DOMAIN_NAMES[HOST]} holds {self.host_history_blocks} pages "
                f"(one per history block ID, the null block included) but one "
                f"{max_len:,}-token request needs {history + ring} IDs "
                f"({history} history + {ring} ring, {geo.block_tokens} tokens "
                f"each) plus the null page; set host_pool_gib >= "
                f"{_ceil_gib((history + ring + 1) * geo.host_bytes_per_block):.2f} "
                f"(or host_pool_tokens >= {(history + ring + 1) * geo.block_tokens})."
            )
        elif self.host_history_blocks < history:
            problems.append(
                f"{DOMAIN_NAMES[HOST]} holds {self.host_history_blocks} history "
                f"blocks but one {max_len:,}-token request needs {history} "
                f"({geo.block_tokens} tokens each); set host_pool_gib >= "
                f"{_ceil_gib(history * geo.host_bytes_per_block):.2f} "
                f"(or host_pool_tokens >= {history * geo.block_tokens})."
            )
        state_need = geo.state_blocks_per_request + geo.state_inflight_blocks
        if self.state_blocks < state_need:
            problems.append(
                f"{DOMAIN_NAMES[STATE]} has {self.state_blocks} state blocks but "
                f"one request needs {state_need} ({geo.state_blocks_per_request} "
                f"plus {geo.state_inflight_blocks} while an asynchronous frame "
                f"is in flight); set state_blocks >= {state_need}."
            )
        hot_need = geo.hot_reserve_groups(max_len)
        auto_hot_shortfall = False
        if self.hot_pool_source == "auto":
            # Reserve a full verify step even when history is short or the
            # host pool caps the number of simultaneous full-length requests.
            step_need = self.hot_pool_min_groups
            auto_hot_shortfall = self.hot_pool_groups < step_need
            if auto_hot_shortfall:
                hot_bytes = geo.hot_bytes_for_groups(step_need, self.max_num_seqs)
                budget_need = (
                    self.gpu_state_bytes
                    + (self.history_demand_blocks + 1) * self.history_bytes_per_block
                    + self.staging_bytes
                    + hot_bytes
                )
                problems.append(
                    f"{HOT_POOL_NAME} (auto) holds {self.hot_pool_groups} groups "
                    f"per layer but max_num_seqs={self.max_num_seqs} verify "
                    f"working sets need {step_need} rank-local groups "
                    f"({hot_bytes} bytes, manual hot_pool_gib >= "
                    f"{_ceil_gib(hot_bytes):.2f}). After state, demand-sized "
                    f"history and staging, auto needs a GPU KV budget >= "
                    f"{budget_need} bytes ({_ceil_gib(budget_need):.2f} GiB); "
                    "lower max_num_seqs, max_model_len, prefill_staging_gib "
                    "or state_blocks, or raise gpu_memory_utilization."
                )
        if self.hot_pool_groups < hot_need:
            hot_gib = _ceil_gib(geo.hot_bytes_for_groups(hot_need, self.max_num_seqs))
            problems.append(
                f"{HOT_POOL_NAME} holds {self.hot_pool_groups} groups per layer "
                f"but one request reserves {hot_need}; set hot_pool_gib >= "
                f"{hot_gib:.2f}."
            )
        if not self.main_kv_on_gpu:
            from vllm.models.qwen4_exp.nvidia.host_kv.planning import history_groups

            groups = history_groups(max_len, geo.dcp_world_size, geo.dcp_interleave)
            pass_need = min(groups, geo.hot_row_groups + geo.hot_tail_groups) + (
                self.max_num_seqs - 1
            ) * min(groups, geo.hot_tail_groups)
            if self.hot_pool_groups < pass_need:
                pass_gib = _ceil_gib(
                    geo.hot_bytes_for_groups(pass_need, self.max_num_seqs)
                )
                problems.append(
                    f"{HOT_POOL_NAME} holds {self.hot_pool_groups} groups per "
                    f"layer but one query row with {self.max_num_seqs} request "
                    f"tails needs {pass_need}; set hot_pool_gib >= "
                    f"{pass_gib:.2f}."
                )
        gpu_need = (history + geo.ring_blocks_per_request + 1) * (
            self.history_bytes_per_block
        )
        if self.gpu_history_budget_blocks < history + geo.ring_blocks_per_request:
            left = self.gpu_history_budget_bytes
            problems.append(
                f"{DOMAIN_NAMES[GPU_HISTORY]} has {_fmt_gib(max(left, 0))} per GPU "
                f"after state {_fmt_gib(self.gpu_state_bytes)}, hot pool "
                f"{_fmt_gib(self.gpu_hot_pool_bytes)} and staging "
                f"{_fmt_gib(self.staging_bytes)} (of "
                f"{_fmt_gib(self.gpu_budget_bytes)}), but one request needs "
                f"{_fmt_gib(gpu_need)}; free "
                f"{_fmt_gib(gpu_need - left)} by lowering hot_pool_gib, "
                f"prefill_staging_gib or state_blocks, or raise "
                f"gpu_memory_utilization."
            )
        if self.hot_slots_per_block:
            k = self.hot_slots_per_block
            floor = cdiv(max(self.hot_pool_min_groups, hot_need), k)
            need = history + geo.ring_blocks_per_request + floor
            if self.gpu_history_blocks < need:
                problems.append(
                    f"{DOMAIN_NAMES[GPU_HISTORY]} shared region holds "
                    f"{self.gpu_history_blocks} IDs ({k} hot slots each) but one "
                    f"{max_len:,}-token request needs {history} history + "
                    f"{geo.ring_blocks_per_request} ring IDs and a hot floor of "
                    f"{floor} IDs; lower hot_pool_gib/hot_pool_ratio, "
                    "prefill_staging_gib or state_blocks, or raise "
                    "gpu_memory_utilization."
                )
        if problems and not auto_hot_shortfall:
            fitting = self.max_fitting_len()
            if fitting:
                problems.append(
                    f"With these pools the largest max_model_len is {fitting:,}."
                )
        return problems

    def describe(self) -> str:
        geo = self.geometry
        max_len = geo.max_model_len
        concurrency = self.max_concurrency(max_len)
        main_kv = (
            "main K/V on GPU in the history segment, hot pool budgets admission only"
            if self.main_kv_on_gpu
            else "main K/V in the host pool"
        )
        return (
            f"long-context KV plan ({main_kv}): GPU budget "
            f"{_fmt_gib(self.gpu_budget_bytes)}"
            f"/GPU = state {self.state_blocks}+1 blocks "
            f"({_fmt_gib(self.gpu_state_bytes)}) + hot pool "
            f"{self.hot_pool_groups} groups/layer "
            f"({_fmt_gib(self.gpu_hot_pool_bytes)}) + staging "
            f"{_fmt_gib(self.staging_bytes)} + history {self.gpu_history_blocks}"
            f"+1 blocks ({_fmt_gib(self.gpu_history_bytes)}); host pool "
            f"{self.host_history_blocks} blocks "
            f"({self.host_history_blocks * geo.block_tokens:,} tokens, "
            f"{_fmt_gib(self.host_history_blocks * geo.host_bytes_per_block)} "
            f"over {geo.num_workers} workers); requests of {max_len:,} tokens "
            "arithmetic domain bounds (physical concurrency unverified): "
            + ", ".join(f"{name} {count}" for name, count in concurrency.items())
            + f"; per-GPU bytes budget={self.gpu_budget_bytes}, "
            f"state={self.gpu_state_bytes}, history={self.gpu_history_bytes}, "
            f"hot={self.gpu_hot_pool_bytes}, staging={self.staging_bytes}, "
            f"remaining={self.gpu_remaining_bytes}, "
            f"hot_pool_source={self.hot_pool_source}, "
            f"hot_pool_min_groups={self.hot_pool_min_groups}, "
            f"hot_admission_bytes={self.hot_pool_bytes}"
        )


def plan_long_context_capacity(
    settings: LongContextKVConfig,
    geometry: LongContextGeometry,
    gpu_budget_bytes: int,
    max_num_seqs: int,
    *,
    main_kv_on_gpu: bool | None = None,
) -> LongContextCapacityPlan:
    """Split one worker's KV budget and the configured pools into domains.

    ``main_kv_on_gpu`` keeps the main K/V pages in the GPU history segment
    (see ``LongContextCapacityPlan.main_kv_on_gpu``); None follows
    ``settings.main_kv_placement``."""
    settings.validate_settings()
    if main_kv_on_gpu is None:
        main_kv_on_gpu = settings.main_kv_placement == "gpu"
    if main_kv_on_gpu:
        # The reference arm retains its original admission sizing and never
        # allocates the host reader scratch.
        geometry = replace(geometry, hot_read_rows=None)
    if settings.host_pool_tokens is not None:
        host_blocks = settings.host_pool_tokens // geometry.block_tokens
    else:
        assert settings.host_pool_gib is not None
        host_blocks = int(settings.host_pool_gib * GiB) // (
            geometry.host_bytes_per_block
        )
    host_bytes = host_blocks * geometry.host_bytes_per_block

    staging = int(settings.prefill_staging_gib * GiB)

    state_blocks = (
        settings.state_blocks
        if settings.state_blocks is not None
        else max(
            max_num_seqs * geometry.state_blocks_per_request,
            geometry.state_blocks_per_request + geometry.state_inflight_blocks,
        )
    )
    plan = LongContextCapacityPlan(
        geometry=geometry,
        max_num_seqs=max_num_seqs,
        gpu_budget_bytes=gpu_budget_bytes,
        host_pool_bytes=host_bytes,
        hot_pool_bytes=0,
        staging_bytes=staging,
        state_blocks=state_blocks,
        gpu_history_blocks=0,
        host_history_blocks=host_blocks,
        hot_pool_groups=0,
        prewarm_hot_pool=settings.prewarm_hot_pool,
        main_kv_on_gpu=main_kv_on_gpu,
    )
    auto = settings.hot_pool_gib == "auto" or (
        settings.hot_pool_gib is None and settings.hot_pool_ratio is None
    )
    if auto:
        # State first, then just the history IDs the workload can use (each
        # host-placed ID needs a host page), then staging, then the hot pool.
        history_blocks = plan.history_demand_blocks
        remaining = (
            gpu_budget_bytes
            - plan.gpu_state_bytes
            - (history_blocks + 1) * plan.history_bytes_per_block
            - staging
        )
        hot_groups = geometry.hot_groups_for_bytes(max(remaining, 0), max_num_seqs)
    elif isinstance(settings.hot_pool_gib, (float, int)):
        hot_groups = geometry.hot_groups_for_bytes(
            int(settings.hot_pool_gib * GiB), max_num_seqs
        )
    else:
        assert settings.hot_pool_ratio is not None
        hot_tokens = int(
            host_blocks
            * geometry.block_tokens
            * settings.hot_pool_ratio
            / geometry.dcp_world_size
        )
        hot_groups = hot_tokens // geometry.hot_group_tokens
    floor = (
        max_num_seqs
        * (
            geometry.decode_rows_per_request * geometry.hot_row_groups
            + geometry.hot_tail_groups
            if main_kv_on_gpu
            else geometry.hot_reserve_groups(geometry.max_model_len)
        )
        if auto
        else hot_groups
    )
    # Manual settings guarantee a minimum, rather than imposing an upper
    # limit. Static static host allocation reserves at least the verification working
    # set too.
    if not auto and not main_kv_on_gpu:
        from vllm.models.qwen4_exp.nvidia.host_kv.planning import history_groups

        groups = history_groups(
            geometry.max_model_len, geometry.dcp_world_size, geometry.dcp_interleave
        )
        pass_need = min(groups, geometry.hot_row_groups + geometry.hot_tail_groups) + (
            max_num_seqs - 1
        ) * min(groups, geometry.hot_tail_groups)
        hot_groups = max(
            floor, geometry.hot_reserve_groups(geometry.max_model_len), pass_need
        )
    plan = replace(
        plan,
        hot_pool_groups=hot_groups,
        hot_pool_bytes=geometry.hot_bytes_for_groups(hot_groups, max_num_seqs),
        hot_pool_source="auto" if auto else "manual",
        hot_pool_min_groups=floor,
    )
    if not main_kv_on_gpu and _shared_region_enabled(geometry):
        return _shared_region_plan(plan, auto)
    if auto and not main_kv_on_gpu:
        return replace(plan, gpu_history_blocks=history_blocks)
    # GPU placement retains the existing physical history allocation. Its
    # auto hot pool uses the demand-sized remainder only for admission and
    # does not reserve or allocate a HostHistoryStore on the GPU.
    history_blocks = plan.gpu_history_budget_blocks
    if not main_kv_on_gpu:
        # Every history ID owns a host page: never more IDs than pages.
        history_blocks = min(history_blocks, plan.host_page_blocks)
    return replace(plan, gpu_history_blocks=history_blocks)


# shared hot region shared hot region for host placement whenever the index pages can
# alias
# hot slots. False (or VLLM_SM70_HOST_KV_STATIC_HOT_POOL=1) keeps the page ownership
# static pool, the rollback and comparison arm.
SHARED_HOT_REGION = True


def _shared_region_enabled(geometry: LongContextGeometry) -> bool:
    import vllm.envs as envs

    return (
        SHARED_HOT_REGION
        and not envs.VLLM_SM70_HOST_KV_STATIC_HOT_POOL
        and geometry.shared_slots_per_block > 0
    )


def shared_region_fixed_bytes(geometry: LongContextGeometry, max_num_seqs: int) -> int:
    """Fixed device bytes of the shared region: everything of the static
    store that is neither slot payload nor per-slot metadata."""
    from vllm.models.qwen4_exp.nvidia.host_kv.budget import (
        SHARED_PER_SLOT,
        history_fixed_memory_bytes,
    )

    parts = history_fixed_memory_bytes(
        # Any slot count beyond 256 * 512 bills the full victim-count tiles.
        slots=256 * 512,
        num_layers=geometry.hot_layers,
        slot_bytes=geometry.hot_slot_bytes,
        max_requests=max_num_seqs,
        selection_width=4 * (geometry.hot_row_groups - 1) + 3,
        max_write_rows=geometry.hot_max_write_rows,
        row_groups=geometry.hot_row_groups,
        tail_groups=geometry.hot_tail_groups,
        read_rows=geometry.hot_read_rows,
    )
    return sum(size for name, size in parts.items() if name not in SHARED_PER_SLOT)


def _shared_region_plan(
    plan: LongContextCapacityPlan, auto: bool
) -> LongContextCapacityPlan:
    """Give the whole remaining budget to history IDs; the hot region is the
    DONATED part of their index pages, above a guaranteed floor (slots).

    Manual values keep their converted slot count as the floor. Auto keeps
    no static floor: the floor is the admitted requests' verify working sets.
    """
    geo = plan.geometry
    k = geo.shared_slots_per_block
    plan = replace(
        plan,
        hot_slots_per_block=k,
        hot_pool_bytes=shared_region_fixed_bytes(geo, plan.max_num_seqs),
        hot_pool_min_groups=0 if auto else plan.hot_pool_min_groups,
    )
    blocks = plan.gpu_history_budget_blocks
    return replace(plan, gpu_history_blocks=blocks, hot_pool_groups=blocks * k)


def validate_long_context_plan(plan: LongContextCapacityPlan) -> None:
    """Refuse to start when one ``max_model_len`` request cannot be held."""
    problems = plan.startup_problems()
    if problems:
        raise ValueError(
            "Long-context KV mode (kernel_config.long_context_kv) cannot hold "
            "one request of max_model_len="
            f"{plan.geometry.max_model_len:,}:\n  - " + "\n  - ".join(problems)
        )


@dataclass
class LongContextAdmissionLedger:
    """Exact per-domain usage plus the hot-pool reservations.

    The block pool reports reference-count transitions: a block becomes used
    when allocated or when a prefix hit touches it from the free queue, and
    unused when its last reference is freed. Each block is tagged with the
    domains of the KV cache group that allocated it, so shared prefix blocks
    count once and evictable cached blocks count as free, as in the pool.
    """

    plan: LongContextCapacityPlan
    num_blocks: int
    physical_segments: bool = False
    """The state and GPU history domains are separate block pools that
    enforce themselves; the ledger only observes the history pool and admits
    against the host and hot-page pools."""
    in_use: list[int] = field(default_factory=lambda: [0, 0, 0])
    hot_used: int = 0
    blocked: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    """Admission refusals per domain (state, GPU history, host, hot)."""
    untracked_allocations: int = 0
    capacity: list[int] = field(init=False, default_factory=list)
    _hot: dict[str, int] = field(default_factory=dict)
    _block_domains: list[tuple[int, ...]] = field(default_factory=list)
    _scope: tuple[int, ...] | None = None
    elastic_pool: Any = None
    """shared hot region shared region (a HistoryBlockPool with slots per block): HOST
    is
    answered by the page table and HOT by the GPU history pool's floor."""

    def __post_init__(self) -> None:
        self._block_domains = [()] * self.num_blocks
        self.capacity = [self.plan.capacity(d) for d in (STATE, GPU_HISTORY, HOST)]

    @property
    def group_domains(self) -> tuple[tuple[int, ...], ...]:
        return self.plan.geometry.group_domains

    # -- block pool observer -------------------------------------------------
    def enter_group(self, group_id: int | None) -> None:
        """Attribute the next allocations to one KV cache group."""
        self._scope = None if group_id is None else self.group_domains[group_id]

    def on_allocated(self, blocks: Iterable[KVCacheBlock]) -> None:
        scope = self._scope
        if scope is None:
            # Never expected: every allocation runs inside a group scope.
            blocks = list(blocks)
            self.untracked_allocations += len(blocks)
            logger.warning_once(
                "Long-context admission saw a block allocation outside a KV "
                "cache group scope; it is not charged to any domain."
            )
            for block in blocks:
                self._block_domains[block.block_id] = ()
            return
        for block in blocks:
            self._block_domains[block.block_id] = scope
            for domain in scope:
                self.in_use[domain] += 1

    def on_acquired(self, block: KVCacheBlock) -> None:
        """A free (evictable) block got its first reference again."""
        domains = self._block_domains[block.block_id]
        if not domains and self._scope is not None:
            domains = self._block_domains[block.block_id] = self._scope
        for domain in domains:
            self.in_use[domain] += 1

    def on_released(self, blocks: Iterable[KVCacheBlock]) -> None:
        """Blocks whose last reference was just freed."""
        for block in blocks:
            for domain in self._block_domains[block.block_id]:
                self.in_use[domain] -= 1

    # -- admission ------------------------------------------------------------
    @property
    def ledger_domains(self) -> tuple[int, ...]:
        """Block domains this ledger admits against (the others are pools)."""
        if self.physical_segments:
            return (HOST,)
        return (STATE, GPU_HISTORY, HOST)

    def free(self, domain: int) -> int:
        return self.capacity[domain] - self.in_use[domain]

    def domain_demand(self, per_group_blocks: Sequence[int]) -> list[int]:
        demand = [0, 0, 0]
        for group_id, blocks in enumerate(per_group_blocks):
            if blocks:
                for domain in self.group_domains[group_id]:
                    demand[domain] += blocks
        return demand

    def hot_tokens(self, request: Request) -> int:
        return min(
            self.plan.geometry.max_model_len,
            request.num_prompt_tokens + request.max_tokens,
        )

    def blocking_domain(
        self,
        request: Request,
        per_group_blocks: Sequence[int],
        reserved_per_group: Sequence[int] | None = None,
    ) -> int | None:
        """The first domain without room for this step, or None if all fit."""
        demand = self.domain_demand(per_group_blocks)
        if reserved_per_group is not None:
            reserved = self.domain_demand(reserved_per_group)
            demand = [used + future for used, future in zip(demand, reserved)]
        pool = self.elastic_pool
        for domain in self.ledger_domains:
            if pool is not None and domain == HOST:
                # The candidate's evictable hit IDs are in the block demand,
                # but they keep their own pages: only fresh pages need room.
                if demand[domain] - pool.host_kept_free() > (
                    pool.host_reclaimable_after()
                ):
                    return domain
                continue
            if demand[domain] > self.free(domain):
                return domain
        need = self.plan.geometry.hot_reserve_groups(self.hot_tokens(request))
        additional = max(need - self._hot.get(request.request_id, 0), 0)
        if pool is not None:
            # The floor after this commit must stay DONATED next to the new
            # IDs: U_nonreclaimable + dU + R_gpu + ceil(F / k) <= A.
            floor = self.plan.hot_floor_groups(self.hot_used + additional)
            extra = pool.floor_blocks(floor) - pool.floor_blocks()
            if extra > 0 and demand[GPU_HISTORY] + extra > pool.get_num_free_blocks():
                return HOT
            return None
        if self.hot_used + additional > self.plan.hot_pool_groups:
            return HOT
        return None

    def admit(
        self,
        request: Request,
        per_group_blocks: Sequence[int],
        reserved_per_group: Sequence[int] | None = None,
        *,
        reason: str = "allocation",
    ) -> bool:
        domain = self.blocking_domain(request, per_group_blocks, reserved_per_group)
        if domain is None:
            return True
        self.blocked[domain] += 1
        name = HOT_POOL_NAME if domain == HOT else DOMAIN_NAMES[domain]
        logger.debug(
            "Long-context admission blocked: request=%s reason=%s domain=%s "
            "demand=%s reserved=%s",
            request.request_id,
            reason,
            name,
            per_group_blocks,
            reserved_per_group,
        )
        logger.info_once(
            "Long-context admission is holding requests back: the %s is full "
            "(requests wait or are preempted; none is rejected).",
            name,
        )
        return False

    def commit(self, request: Request) -> None:
        """Reserve through completion, including any appended streaming input."""
        previous = self._hot.get(request.request_id, 0)
        groups = max(
            previous, self.plan.geometry.hot_reserve_groups(self.hot_tokens(request))
        )
        self._hot[request.request_id] = groups
        self.hot_used += groups - previous
        if self.elastic_pool is not None:
            self.elastic_pool.set_hot_floor(self.plan.hot_floor_groups(self.hot_used))

    def release(self, request_id: str) -> None:
        self.hot_used -= self._hot.pop(request_id, 0)
        if self.elastic_pool is not None:
            self.elastic_pool.set_hot_floor(self.plan.hot_floor_groups(self.hot_used))

    def snapshot(self) -> dict[str, tuple[int, int]]:
        """(used, capacity) per domain, for logs and tests."""
        out = {
            DOMAIN_NAMES[d]: (self.in_use[d], self.capacity[d])
            for d in (STATE, GPU_HISTORY, HOST)
        }
        out[HOT_POOL_NAME] = (
            (self.hot_used, self.elastic_pool.hot_slots())
            if self.elastic_pool is not None
            else (self.hot_used, self.plan.hot_pool_groups)
        )
        return out
