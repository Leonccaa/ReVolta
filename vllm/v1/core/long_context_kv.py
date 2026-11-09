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

The physical allocator is not split yet: until it is, the unified pool still
holds every block and is checked as before, and the domain capacities act as an
additional admission constraint.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

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
# Index of the hot-page pool in blocked-admission counters.
HOT = 3

# Tokens of the newest history pinned in the hot pool per request (host KV pool), and
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
    """Workers holding a distinct main-K/V shard (TP x DCP x PP ranks)."""
    main_kv_bytes_per_block: int
    """Main QSA K/V bytes of one history block on one worker (all layers)."""
    index_bytes_per_block: int
    """GPU bytes of one history block on one worker: compressed index pages,
    shared with the compressor ring."""
    state_bytes_per_block: int
    """GPU bytes of one state block ID on one worker (every owner page plus
    standalone drafter pages)."""
    unified_bytes_per_block: int
    """Bytes of one block in today's unified pool, for reference."""
    state_blocks_per_request: int
    """Worst-case state block IDs of one request (GDN, PLE, drafter)."""
    ring_blocks_per_request: int
    """History block IDs a request holds for its compressor ring(s)."""
    max_model_len: int
    """Context limit the startup check must cover."""
    hot_group_tokens: int
    """Tokens per hot-page group (the indexer compression ratio)."""
    hot_row_groups: int
    """Distinct groups one query row can select (top-k groups plus the
    uncompressed tail)."""
    decode_rows_per_request: int
    """Query rows of one request in a decode/verify step (1 + draft tokens)."""
    group_domains: tuple[tuple[int, ...], ...]
    """Domains charged by each KV cache group, in group order."""

    @property
    def host_bytes_per_block(self) -> int:
        """Host bytes of one history block summed over all workers."""
        return self.main_kv_bytes_per_block * self.num_workers

    @property
    def hot_tail_groups(self) -> int:
        return cdiv(HOT_TAIL_TOKENS, self.hot_group_tokens) + 1

    def history_blocks(self, num_tokens: int) -> int:
        """Main-KV history block IDs for ``num_tokens`` of context."""
        return cdiv(num_tokens, self.block_tokens)

    def hot_reserve_groups(self, num_tokens: int) -> int:
        """Hot-pool groups one request reserves (per layer, per worker).

        Every group a decode/verify step selects must be resident while the
        attention reads it, and the newest tokens stay pinned. A request can
        never need more groups than its own history has.
        """
        groups = cdiv(max(num_tokens, 1), self.hot_group_tokens)
        bound = self.decode_rows_per_request * min(self.hot_row_groups, groups) + min(
            self.hot_tail_groups, groups
        )
        return min(groups, bound)

    def hot_groups_for_bytes(self, num_bytes: int) -> int:
        """Hot-pool groups (per layer) that ``num_bytes`` per worker hold."""
        return (num_bytes * self.block_tokens) // (
            self.main_kv_bytes_per_block * self.hot_group_tokens
        )

    def hot_bytes_for_groups(self, groups: int) -> int:
        return cdiv(
            groups * self.main_kv_bytes_per_block * self.hot_group_tokens,
            self.block_tokens,
        )


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

    @property
    def gpu_state_bytes(self) -> int:
        return (self.state_blocks + 1) * self.geometry.state_bytes_per_block

    @property
    def gpu_history_budget_bytes(self) -> int:
        """GPU bytes left for the history segment (may be negative)."""
        return (
            self.gpu_budget_bytes
            - self.gpu_state_bytes
            - self.hot_pool_bytes
            - self.staging_bytes
        )

    def capacity(self, domain: int) -> int:
        return (self.state_blocks, self.gpu_history_blocks, self.host_history_blocks)[
            domain
        ]

    def max_concurrency(self, num_tokens: int) -> dict[str, int]:
        """Requests of ``num_tokens`` each domain could hold at once."""
        geo = self.geometry
        history = geo.history_blocks(num_tokens)
        return {
            DOMAIN_NAMES[STATE]: self.state_blocks
            // max(geo.state_blocks_per_request, 1),
            DOMAIN_NAMES[GPU_HISTORY]: self.gpu_history_blocks
            // (history + geo.ring_blocks_per_request),
            DOMAIN_NAMES[HOST]: self.host_history_blocks // history,
            HOT_POOL_NAME: self.hot_pool_groups // geo.hot_reserve_groups(num_tokens),
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
        if self.host_history_blocks < history:
            problems.append(
                f"{DOMAIN_NAMES[HOST]} holds {self.host_history_blocks} history "
                f"blocks but one {max_len:,}-token request needs {history} "
                f"({geo.block_tokens} tokens each); set host_pool_gib >= "
                f"{_ceil_gib(history * geo.host_bytes_per_block):.2f} "
                f"(or host_pool_tokens >= {history * geo.block_tokens})."
            )
        if self.state_blocks < geo.state_blocks_per_request:
            problems.append(
                f"{DOMAIN_NAMES[STATE]} has {self.state_blocks} state blocks but "
                f"one request needs {geo.state_blocks_per_request}; set "
                f"state_blocks >= {geo.state_blocks_per_request}."
            )
        hot_need = geo.hot_reserve_groups(max_len)
        if self.hot_pool_groups < hot_need:
            problems.append(
                f"{HOT_POOL_NAME} holds {self.hot_pool_groups} groups per layer "
                f"but one request reserves {hot_need}; set hot_pool_gib >= "
                f"{_ceil_gib(geo.hot_bytes_for_groups(hot_need)):.2f}."
            )
        gpu_need = (history + geo.ring_blocks_per_request + 1) * (
            geo.index_bytes_per_block
        )
        if self.gpu_history_blocks < history + geo.ring_blocks_per_request:
            left = self.gpu_history_budget_bytes
            problems.append(
                f"{DOMAIN_NAMES[GPU_HISTORY]} has {_fmt_gib(max(left, 0))} per GPU "
                f"after state {_fmt_gib(self.gpu_state_bytes)}, hot pool "
                f"{_fmt_gib(self.hot_pool_bytes)} and staging "
                f"{_fmt_gib(self.staging_bytes)} (of "
                f"{_fmt_gib(self.gpu_budget_bytes)}), but one request needs "
                f"{_fmt_gib(gpu_need)}; free "
                f"{_fmt_gib(gpu_need - left)} by lowering hot_pool_gib, "
                f"prefill_staging_gib or state_blocks, or raise "
                f"gpu_memory_utilization."
            )
        if problems:
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
        history_bytes = (self.gpu_history_blocks + 1) * geo.index_bytes_per_block
        return (
            f"long-context KV plan: GPU budget {_fmt_gib(self.gpu_budget_bytes)}"
            f"/GPU = state {self.state_blocks}+1 blocks "
            f"({_fmt_gib(self.gpu_state_bytes)}) + hot pool "
            f"{self.hot_pool_groups} groups/layer "
            f"({_fmt_gib(self.hot_pool_bytes)}) + staging "
            f"{_fmt_gib(self.staging_bytes)} + history {self.gpu_history_blocks}"
            f"+1 blocks ({_fmt_gib(history_bytes)}); host pool "
            f"{self.host_history_blocks} blocks "
            f"({self.host_history_blocks * geo.block_tokens:,} tokens, "
            f"{_fmt_gib(self.host_history_blocks * geo.host_bytes_per_block)} "
            f"over {geo.num_workers} workers); requests of {max_len:,} tokens "
            f"held at once: "
            + ", ".join(f"{name} {count}" for name, count in concurrency.items())
        )


def plan_long_context_capacity(
    settings: LongContextKVConfig,
    geometry: LongContextGeometry,
    gpu_budget_bytes: int,
    max_num_seqs: int,
) -> LongContextCapacityPlan:
    """Split one worker's KV budget and the configured pools into domains."""
    settings.validate_settings()
    if settings.host_pool_tokens is not None:
        host_blocks = settings.host_pool_tokens // geometry.block_tokens
    else:
        assert settings.host_pool_gib is not None
        host_blocks = int(settings.host_pool_gib * GiB) // (
            geometry.host_bytes_per_block
        )
    host_bytes = host_blocks * geometry.host_bytes_per_block

    if settings.hot_pool_gib is not None:
        hot_groups = geometry.hot_groups_for_bytes(int(settings.hot_pool_gib * GiB))
    else:
        from vllm.config.kernel import LONG_CONTEXT_DEFAULT_HOT_POOL_RATIO

        ratio = (
            settings.hot_pool_ratio
            if settings.hot_pool_ratio is not None
            else LONG_CONTEXT_DEFAULT_HOT_POOL_RATIO
        )
        hot_tokens = int(host_blocks * geometry.block_tokens * ratio)
        hot_groups = hot_tokens // geometry.hot_group_tokens
    hot_bytes = geometry.hot_bytes_for_groups(hot_groups)
    staging = int(settings.prefill_staging_gib * GiB)

    state_blocks = (
        settings.state_blocks
        if settings.state_blocks is not None
        else max_num_seqs * geometry.state_blocks_per_request
    )
    plan = LongContextCapacityPlan(
        geometry=geometry,
        max_num_seqs=max_num_seqs,
        gpu_budget_bytes=gpu_budget_bytes,
        host_pool_bytes=host_bytes,
        hot_pool_bytes=hot_bytes,
        staging_bytes=staging,
        state_blocks=state_blocks,
        gpu_history_blocks=0,
        host_history_blocks=host_blocks,
        hot_pool_groups=hot_groups,
        prewarm_hot_pool=settings.prewarm_hot_pool,
    )
    left = plan.gpu_history_budget_bytes
    history_blocks = max(left // geometry.index_bytes_per_block - 1, 0)
    return replace(plan, gpu_history_blocks=history_blocks)


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
    in_use: list[int] = field(default_factory=lambda: [0, 0, 0])
    hot_used: int = 0
    blocked: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    """Admission refusals per domain (state, GPU history, host, hot)."""
    untracked_allocations: int = 0
    capacity: list[int] = field(init=False, default_factory=list)
    _hot: dict[str, int] = field(default_factory=dict)
    _block_domains: list[tuple[int, ...]] = field(default_factory=list)
    _scope: tuple[int, ...] | None = None

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
        self, request: Request, per_group_blocks: Sequence[int]
    ) -> int | None:
        """The first domain without room for this step, or None if all fit."""
        demand = self.domain_demand(per_group_blocks)
        for domain in (STATE, GPU_HISTORY, HOST):
            if demand[domain] > self.free(domain):
                return domain
        if request.request_id not in self._hot:
            need = self.plan.geometry.hot_reserve_groups(self.hot_tokens(request))
            if self.hot_used + need > self.plan.hot_pool_groups:
                return HOT
        return None

    def admit(self, request: Request, per_group_blocks: Sequence[int]) -> bool:
        domain = self.blocking_domain(request, per_group_blocks)
        if domain is None:
            return True
        self.blocked[domain] += 1
        name = HOT_POOL_NAME if domain == HOT else DOMAIN_NAMES[domain]
        logger.info_once(
            "Long-context admission is holding requests back: the %s is full "
            "(requests wait or are preempted; none is rejected).",
            name,
        )
        return False

    def commit(self, request: Request) -> None:
        """Reserve hot-pool room once the request holds blocks."""
        if request.request_id in self._hot:
            return
        groups = self.plan.geometry.hot_reserve_groups(self.hot_tokens(request))
        self._hot[request.request_id] = groups
        self.hot_used += groups

    def release(self, request_id: str) -> None:
        self.hot_used -= self._hot.pop(request_id, 0)

    def snapshot(self) -> dict[str, tuple[int, int]]:
        """(used, capacity) per domain, for logs and tests."""
        out = {
            DOMAIN_NAMES[d]: (self.in_use[d], self.capacity[d])
            for d in (STATE, GPU_HISTORY, HOST)
        }
        out[HOT_POOL_NAME] = (self.hot_used, self.plan.hot_pool_groups)
        return out
