# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-group prefix-hit contract (opt-in, ``qsa_cross_group_hit_check``).

A hybrid model restores one logical prefix from several KV cache groups: full
attention keeps every block before the hit, a recurrent (Mamba ``align``)
group keeps the one state block that ends at the hit, and a sliding-window
group keeps the blocks covering its window before the hit. Scratch groups
(compressor ring, tail) are never part of a hit. A prefix hit of ``L`` tokens
is complete only when every group holds what it needs at exactly ``L``.

The GPU prefix cache and the CPU/SSD offloading connector already compute a
common hit with fixed-point lookups. This module states the requirement once,
checks the final answer of both paths against it, and turns any incomplete
answer into a miss instead of reporting it. It also lets a future owner of
storage outside these pools (an Active Host KV history pool) register a
residency probe, so that a hit stops before the first block whose payload that
owner no longer holds or has not made readable.

Block identity for probes is the request's chain hash of the last
``hash_block_size`` block inside the group block. It is the same value the
offloading connector keys its blocks with, so one probe answers for both the
GPU prefix cache and the external tier.
"""

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    SinkFullAttentionManager,
    SlidingWindowManager,
)

logger = init_logger(__name__)


def cross_group_hit_check_enabled(vllm_config: object) -> bool:
    """``kernel_config.qsa_cross_group_hit_check``; only a real True enables."""
    kernel_config = getattr(vllm_config, "kernel_config", None)
    return getattr(kernel_config, "qsa_cross_group_hit_check", False) is True


def load_failure_recompute_enabled(vllm_config: object) -> bool:
    """Abandon a broken external KV load and recompute instead of crashing.

    On with the cross-group hit contract or the long-context mode. Then a load
    plan that turns out incomplete after the hit was committed, a key that is
    no longer readable when the load is prepared, or a failed transfer voids
    the request's external hit: its blocks are released and it is scheduled
    again from its local prefix hit (or from token 0), with the external
    store skipped for that retry. Other requests are not affected. Off, the
    existing asserts and ``KVHitContractError`` stay as they are.
    """
    if cross_group_hit_check_enabled(vllm_config):
        return True
    kernel_config = getattr(vllm_config, "kernel_config", None)
    long_context = getattr(kernel_config, "long_context_kv", None)
    return getattr(long_context, "enabled", False) is True


class HitNeed(enum.Enum):
    """What one KV cache group must hold for a prefix hit ending at ``L``."""

    PREFIX = "prefix"  # every block before L
    STATE = "state"  # the block that ends exactly at L
    WINDOW = "window"  # the last window_tokens tokens before L
    NONE = "none"  # never part of a hit (scratch ring / tail)
    UNCHECKED = "unchecked"  # hit semantics this contract does not model


@dataclass(frozen=True)
class GroupHitRule:
    group_id: int
    need: HitNeed
    # Tokens per block of this group, in the coordinate system of the path
    # (global scheduler tokens for the GPU cache, offloaded block for the
    # connector).
    block_size: int
    window_tokens: int = 0

    def required_blocks(self, end_tokens: int, start_tokens: int = 0) -> range:
        """Block indices that must hold valid data for a hit ending at
        ``end_tokens``. Blocks wholly below ``start_tokens // block_size`` are
        served by another path (the local GPU hit) and are excluded.

        A hit may end inside a full-attention or sliding-window block (the
        connector serves groups of different block sizes without a recurrent
        page grid); that block is then required too. Window retention is
        measured in tokens before converting to blocks. A recurrent state must
        end exactly at the hit, which
        ``KVHitContract.check`` enforces before calling this.
        """
        if end_tokens <= 0 or self.need in (HitNeed.NONE, HitNeed.UNCHECKED):
            return range(0)
        full = end_tokens // self.block_size
        end = -(-end_tokens // self.block_size)
        if self.need is HitNeed.PREFIX:
            low = 0
        elif self.need is HitNeed.STATE:
            low = full - 1
        else:
            low = max(0, end_tokens - self.window_tokens) // self.block_size
        return range(max(low, start_tokens // self.block_size), end)


# (group_id, block_index, chain_hash, gpu_block_id or None) -> resident and
# readable. ``gpu_block_id`` is given for GPU prefix-cache hits only.
ResidencyProbe = Callable[[int, int, BlockHash, int | None], bool]
# Chain hash of a group block by its index; None when the request has not
# hashed that far yet.
ChainHashFn = Callable[[GroupHitRule, int], BlockHash | None]


class KVHitContractError(RuntimeError):
    """A load plan would restore an incomplete prefix. Raised only after the
    lookup-time check passed, i.e. on an internal inconsistency."""


@dataclass
class HitContractStats:
    # Hits reported by a lookup that the contract turned into misses.
    rejected_hits: int = 0
    # Lookups whose search limit a residency probe lowered.
    probe_capped: int = 0
    # Load plans rejected after the lookup-time check (internal errors).
    load_plan_errors: int = 0
    last_reason: str | None = field(default=None, repr=False)


class KVHitContract:
    """Rules for one lookup path plus optional residency probes."""

    def __init__(
        self, rules: Sequence[GroupHitRule], path: str, hash_block_size: int = 1
    ):
        self.path = path
        self.hash_block_size = hash_block_size
        self.rules: dict[int, GroupHitRule] = {r.group_id: r for r in rules}
        self._probes: dict[int, ResidencyProbe] = {}
        self.stats = HitContractStats()

    # --- probes -----------------------------------------------------------

    def register_probe(
        self,
        group_id: int,
        probe: ResidencyProbe,
        *,
        block_size: int | None = None,
    ) -> None:
        """Make a hit depend on ``probe`` for ``group_id``.

        A group the path does not manage itself (e.g. history held only in a
        host pool, which the connector neither stores nor loads) is added as a
        PREFIX group of ``block_size`` tokens.
        """
        span = self.rules[group_id].block_size if group_id in self.rules else block_size
        if span is not None and span % self.hash_block_size:
            raise ValueError("Probe block_size must be a multiple of hash_block_size")
        if group_id not in self.rules:
            if block_size is None or block_size <= 0:
                raise ValueError(
                    f"group {group_id} is not managed by the {self.path} path; "
                    "pass its block_size to add it as a probe-only group"
                )
            self.rules[group_id] = GroupHitRule(group_id, HitNeed.PREFIX, block_size)
        elif self.rules[group_id].need in (HitNeed.NONE, HitNeed.UNCHECKED):
            raise ValueError(
                f"group {group_id} is not checked by the {self.path} contract"
            )
        self._probes[group_id] = probe

    @staticmethod
    def _probe_ready(
        probe: ResidencyProbe,
        group_id: int,
        index: int,
        block_hash: BlockHash,
        block_id: int | None,
    ) -> bool:
        try:
            return probe(group_id, index, block_hash, block_id)
        except Exception:
            logger.debug(
                "KV residency probe failed for group %d block %d",
                group_id,
                index,
                exc_info=True,
            )
            return False

    @property
    def has_probes(self) -> bool:
        return bool(self._probes)

    def probe_cap(
        self, chain_hash: ChainHashFn, start_tokens: int, max_tokens: int
    ) -> int:
        """Largest length <= ``max_tokens`` such that every probed PREFIX group
        holds all its blocks from ``start_tokens`` on. Other probed groups
        are checked on the final hit by ``check``."""
        cap = max_tokens
        for group_id, probe in self._probes.items():
            rule = self.rules[group_id]
            if rule.need is not HitNeed.PREFIX:
                continue
            for index in range(start_tokens // rule.block_size, cap // rule.block_size):
                block_hash = chain_hash(rule, index)
                if block_hash is None or not self._probe_ready(
                    probe, group_id, index, block_hash, None
                ):
                    cap = min(cap, index * rule.block_size)
                    break
        if cap < max_tokens:
            self.stats.probe_capped += 1
        return cap

    # --- checks -----------------------------------------------------------

    def check(
        self,
        end_tokens: int,
        start_tokens: int,
        block_ok: Callable[[GroupHitRule, int], bool | None],
        chain_hash: ChainHashFn,
        gpu_block_id: Callable[[GroupHitRule, int], int | None] | None = None,
    ) -> str | None:
        """Return why a hit ending at ``end_tokens`` (extending one that ends
        at ``start_tokens``) is incomplete, or None when it is complete.

        ``block_ok(rule, index)`` answers whether the path itself holds block
        ``index`` of that group (None: the path does not manage the group).
        """
        if end_tokens <= start_tokens:
            return None
        for rule in self.rules.values():
            if rule.need in (HitNeed.NONE, HitNeed.UNCHECKED):
                continue
            if rule.need is HitNeed.STATE and end_tokens % rule.block_size:
                return (
                    f"group {rule.group_id}: hit {end_tokens} is not on its "
                    f"{rule.block_size}-token state grid"
                )
            probe = self._probes.get(rule.group_id)
            for index in rule.required_blocks(end_tokens, start_tokens):
                held = block_ok(rule, index)
                if held is False:
                    return (
                        f"group {rule.group_id} ({rule.need.value}): block {index} "
                        f"missing for a hit of {end_tokens}"
                    )
                if held is None and probe is None:
                    return f"group {rule.group_id}: unmanaged group without a probe"
                if probe is not None:
                    block_hash = chain_hash(rule, index)
                    block_id = (
                        None if gpu_block_id is None else gpu_block_id(rule, index)
                    )
                    if block_hash is None or not self._probe_ready(
                        probe, rule.group_id, index, block_hash, block_id
                    ):
                        return (
                            f"group {rule.group_id}: block {index} not resident "
                            f"in its probed store for a hit of {end_tokens}"
                        )
        return None

    def reject(self, reason: str, request_id: str) -> None:
        self.stats.rejected_hits += 1
        self.stats.last_reason = reason
        # Reasons carry token counts; keep the warning itself one per path.
        logger.warning_once(
            "Cross-group hit contract (%s) rejected an incomplete prefix hit; "
            "it is treated as a miss (reasons at debug level).",
            self.path,
        )
        logger.debug(
            "Cross-group hit contract (%s) rejected request %s: %s",
            self.path,
            request_id,
            reason,
        )


# --- GPU prefix cache -------------------------------------------------------


def local_hit_rules(
    managers: Sequence[SingleTypeKVCacheManager],
) -> list[GroupHitRule]:
    """Rules for the GPU prefix cache from the coordinator's managers."""
    rules: list[GroupHitRule] = []
    for group_id, manager in enumerate(managers):
        spec = manager.kv_cache_spec
        if not spec.prefix_cacheable:
            need = HitNeed.NONE
        elif type(manager) in (FullAttentionManager, SinkFullAttentionManager):
            need = HitNeed.PREFIX
        elif type(manager) is MambaManager:
            need = HitNeed.STATE
        elif type(manager) is SlidingWindowManager:
            need = HitNeed.WINDOW
        else:
            need = HitNeed.UNCHECKED
        window_tokens = 0
        if need is HitNeed.WINDOW:
            window_tokens = spec.sliding_window - 1  # type: ignore[attr-defined]
        rules.append(GroupHitRule(group_id, need, manager.block_size, window_tokens))
    unchecked = [r.group_id for r in rules if r.need is HitNeed.UNCHECKED]
    if unchecked:
        logger.info(
            "Cross-group hit contract does not model KV cache group(s) %s; "
            "their hits follow the coordinator unchanged.",
            unchecked,
        )
    return rules


def chain_hash_from_request(
    block_hashes: Sequence[BlockHash], hash_block_size: int
) -> ChainHashFn:
    """Chain hash of a group block: the request hash of its last
    ``hash_block_size`` block."""

    def _hash(rule: GroupHitRule, index: int) -> BlockHash | None:
        factor = rule.block_size // hash_block_size
        position = (index + 1) * factor - 1
        if factor <= 0 or position >= len(block_hashes):
            return None
        return block_hashes[position]

    return _hash


def settle_local_hit(
    contract: KVHitContract,
    request_id: str,
    blocks_by_group: tuple[list[KVCacheBlock], ...],
    hit_tokens: int,
    chain_hash: ChainHashFn,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    """Check a GPU prefix-cache hit; trim full-attention groups to it.

    The hybrid coordinator truncates only its first full-attention spec group
    to the final hit, so a second full-attention spec (e.g. a split main KV /
    index pair) can return blocks beyond the hit. They are dropped here. An
    incomplete hit becomes a miss; the blocks were not touched yet.
    """

    def _block(rule: GroupHitRule, index: int) -> KVCacheBlock | None:
        if rule.group_id >= len(blocks_by_group):
            return None
        blocks = blocks_by_group[rule.group_id]
        return blocks[index] if index < len(blocks) else None

    def _block_ok(rule: GroupHitRule, index: int) -> bool | None:
        if rule.group_id >= len(blocks_by_group):
            return None  # probe-only group: its probe decides
        block = _block(rule, index)
        return block is not None and not block.is_null and block.block_hash is not None

    def _block_id(rule: GroupHitRule, index: int) -> int | None:
        block = _block(rule, index)
        return None if block is None else block.block_id

    reason = contract.check(hit_tokens, 0, _block_ok, chain_hash, _block_id)
    if reason is not None:
        contract.reject(reason, request_id)
        return tuple([] for _ in blocks_by_group), 0
    for rule in contract.rules.values():
        if rule.need is HitNeed.PREFIX and rule.group_id < len(blocks_by_group):
            del blocks_by_group[rule.group_id][hit_tokens // rule.block_size :]
    return blocks_by_group, hit_tokens


def connector_hit_rules(
    group_needs: Mapping[int, tuple[HitNeed, int, int]],
) -> list[GroupHitRule]:
    """Rules for an external tier: ``group_id -> (need, block_size, window)``."""
    return [
        GroupHitRule(group_id, need, block_size, window)
        for group_id, (need, block_size, window) in group_needs.items()
    ]
