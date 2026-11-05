# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side observability counters of the offloading connector.

Read-only bookkeeping: nothing here is consulted by any store, load, lookup or
eviction decision. The scheduler accumulates cumulative totals and the
connector reports their increase in each ``get_kv_connector_stats`` call (see
``metrics.py`` for the Prometheus series and docs/kv-offload.md for their
definitions).
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from operator import itemgetter
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, MambaSpec
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        GroupOffloadConfig,
    )


class LookupShrink:
    """How far the sparse (sliding-window and recurrent-state) groups of one
    ``_lookup`` lower the hit below the full-attention reach.

    ``reach`` is the hit length (absolute tokens) when the first sparse group
    is examined: the full-attention prefix, capped at the request length minus
    one and kept on the Mamba page grid. Every later decrease of the hit is
    attributed to the sparse group being examined, except the EAGLE/MTP state
    shift and rounding a sparse group's hit down to the Mamba page grid, which
    are ``page_align``. A lookup ending without a hit is served at the local
    hit, and the remaining decrease goes to the group examined last. So the
    causes add up to ``reach - served`` by construction.
    """

    __slots__ = ("_causes", "reach", "_current", "_cause", "tokens")

    def __init__(self, causes: Mapping[int, str]):
        self._causes = causes
        self.reach: int | None = None
        self._current = 0
        self._cause = ""
        self.tokens: dict[str, int] = {}

    def enter(self, group_idx: int, hit_tokens: int) -> None:
        """A sparse group is about to be looked up at ``hit_tokens``."""
        if self.reach is None:
            self.reach = self._current = hit_tokens
        else:
            self.lower(hit_tokens)
        self._cause = self._causes[group_idx]

    def lower(self, hit_tokens: int, cause: str | None = None) -> None:
        """The hit is now ``hit_tokens``; a decrease goes to ``cause``
        (default: the sparse group being examined)."""
        if self.reach is None or hit_tokens >= self._current:
            return
        key = cause or self._cause
        self.tokens[key] = self.tokens.get(key, 0) + self._current - hit_tokens
        self._current = hit_tokens


class OffloadCounters:
    """Cumulative totals: completed CPU-tier stores by kind (target / draft /
    state), lookup shrink tokens by cause (draft_window / recurrent_state /
    sliding_window / page_align) and external hits by whether fewer tokens
    than the draft window remain to prefill (needed true / false)."""

    def __init__(
        self,
        kind_by_group: Mapping[int, str],
        shrink_cause_by_group: Mapping[int, str],
        has_full_attention: bool,
        draft_window_tokens: int | None,
    ):
        self.kind_by_group = dict(kind_by_group)
        # An offload key is the block hash followed by a group suffix; count
        # completed stores by suffix (C-level) and map each suffix to a kind.
        self._group_suffix = itemgetter(slice(-len(make_offload_key(b"", 0)), None))
        self._kind_by_suffix = {
            make_offload_key(b"", idx): kind for idx, kind in kind_by_group.items()
        }
        self.shrink_cause_by_group = dict(shrink_cause_by_group)
        # A shrink needs a full-attention reach and a sparse group to cut it.
        self.tracks_lookup_shrink = has_full_attention and bool(shrink_cause_by_group)
        # Largest recognised draft window (tokens); None without a draft group.
        self.draft_window_tokens = draft_window_tokens
        self.stored_blocks: Counter[str] = Counter()
        self.lookup_shrink_tokens: Counter[str] = Counter()
        self.hit_draft_window_needed: Counter[str] = Counter()
        self._reported: dict[str, Counter[str]] = {}

    def new_lookup(self) -> LookupShrink | None:
        if not self.tracks_lookup_shrink:
            return None
        return LookupShrink(self.shrink_cause_by_group)

    def count_stored(self, keys: Iterable[OffloadKey]) -> None:
        kinds = self._kind_by_suffix
        stored = self.stored_blocks
        for suffix, n in Counter(map(self._group_suffix, keys)).items():
            stored[kinds.get(suffix, "target")] += n

    def count_admitted_lookup(
        self,
        shrink_tokens: Mapping[str, int] | None,
        num_external_tokens: int,
        num_tokens_after_hit: int,
    ) -> None:
        """Count a lookup when its request is admitted (once per admission,
        like the external prefix cache query counters)."""
        if shrink_tokens:
            self.lookup_shrink_tokens.update(shrink_tokens)
        window = self.draft_window_tokens
        if num_external_tokens > 0 and window is not None:
            needed = num_tokens_after_hit < window
            self.hit_draft_window_needed["true" if needed else "false"] += 1

    def totals(self) -> dict[str, Counter[str]]:
        return {
            "cpu_stored_blocks": self.stored_blocks,
            "lookup_shrink_tokens": self.lookup_shrink_tokens,
            "hit_draft_window_needed": self.hit_draft_window_needed,
        }

    def deltas(self) -> dict[str, dict[str, int]]:
        """Positive increase of every labelled total since the previous call,
        keyed like ``totals``; empty entries are left out."""
        out: dict[str, dict[str, int]] = {}
        for name, totals in self.totals().items():
            reported = self._reported.setdefault(name, Counter())
            delta = {
                label: n - reported[label]
                for label, n in totals.items()
                if n > reported[label]
            }
            if delta:
                reported.update(delta)
                out[name] = delta
        return out


def offload_group_kind(group: "GroupOffloadConfig", kv_spec: KVCacheSpec) -> str:
    """Semantic kind shared by completed-store and residency statistics.

    A recurrent group's cache mode affects dependency deletion support, not
    its statistics kind. Draft recognition remains the scheduler's decision.
    """
    if group.draft_window_tokens is not None:
        return "draft"
    return "state" if isinstance(kv_spec, MambaSpec) else "target"


def build_offload_counters(
    kv_group_configs: Iterable["GroupOffloadConfig"], kv_cache_config: KVCacheConfig
) -> OffloadCounters:
    """Classify the offloaded KV cache groups.

    Stored kind: ``draft`` for a group cpu.draft_offload recognises as a
    speculative drafter's sliding window (``draft_window_tokens`` set),
    ``state`` for recurrent-state (Mamba / GDN) groups, ``target`` for every
    other group: full attention, a target model's own sliding window, and
    EAGLE/MTP groups that are not such a sliding window (an MTP layer with full
    attention is stored and looked up like the target's full attention).
    Shrink cause, for the sparse groups only: ``draft_window``,
    ``recurrent_state``, or ``sliding_window`` for any other sliding window.
    """
    kind_by_group: dict[int, str] = {}
    shrink_cause_by_group: dict[int, str] = {}
    has_full_attention = False
    draft_windows: list[int] = []
    for group in kv_group_configs:
        idx = group.group_idx
        kind = offload_group_kind(
            group, kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        )
        kind_by_group[idx] = kind
        if group.draft_window_tokens is not None:
            draft_windows.append(group.draft_window_tokens)
        if group.sliding_window_size_in_blocks is None:
            has_full_attention = True
        elif group.draft_window_tokens is not None:
            shrink_cause_by_group[idx] = "draft_window"
        else:
            shrink_cause_by_group[idx] = (
                "recurrent_state" if kind == "state" else "sliding_window"
            )
    return OffloadCounters(
        kind_by_group,
        shrink_cause_by_group,
        has_full_attention,
        max(draft_windows) if draft_windows else None,
    )
