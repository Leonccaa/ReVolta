# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Delayed write-through from the CPU primary tier to a secondary tier.

A block stored to the CPU tier is normally cascaded to every secondary tier at
once (TieringOffloadingManager.complete_store). Many recurrent-state rows are
superseded by the next checkpoint of the same conversation within a minute,
so writing them is wasted device endurance. With a delay the manager keeps
only the keys of stored blocks for ``write_delay_s`` seconds, WITHOUT pinning
the CPU blocks (they stay ordinary resident rows and the CPU policy may evict
them), and at the deadline writes only blocks that are still resident, ready
and not demoted. Block IDs are resolved again by key at that point, so a
reused CPU slot is never written under an old key.

Also:
  * a rolling 24 h written-bytes count persisted under the tier root; above a
    threshold a WARNING is logged (at most hourly) and a counter is raised.
    Writes are never refused (Leon 2026-11-05: log only);
  * explicit drops (``drop``), used for "state latest only".
"""

import json
import os
import time
from collections import Counter, deque
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)

VOLUME_FILE = ".write-volume.json"
DROP_REASONS = ("superseded", "evicted", "latest")


class WriteVolume:
    """Bytes written over a rolling window, in fixed buckets, persisted.

    The file survives restarts so a restart does not reset the count. A
    missing or unreadable file starts empty (logged); a write failure only
    logs, it never blocks the write path.
    """

    def __init__(
        self,
        path: str,
        warn_bytes: int,
        window_s: int = 86_400,
        bucket_s: int = 3_600,
        persist_interval_s: float = 60.0,
        clock: Callable[[], float] = time.time,
    ):
        if warn_bytes < 0:
            raise ValueError("write_warn_bytes_24h must be >= 0")
        if window_s <= 0 or bucket_s <= 0 or window_s % bucket_s:
            raise ValueError("volume window must be a positive multiple of the bucket")
        self.path = path
        self.warn_bytes = int(warn_bytes)
        self.window_s = window_s
        self.bucket_s = bucket_s
        self.persist_interval_s = persist_interval_s
        self.clock = clock
        self._buckets: dict[int, int] = {}
        self._dirty = False
        self._persisted_at: float | None = None
        self._load()

    def _bucket(self, now: float) -> int:
        return int(now // self.bucket_s) * self.bucket_s

    def _prune(self, now: float) -> None:
        oldest = self._bucket(now) - self.window_s + self.bucket_s
        for start in [s for s in self._buckets if s < oldest]:
            del self._buckets[start]

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                raw = json.load(f)
            if raw.get("version") != 1 or raw.get("bucket_s") != self.bucket_s:
                raise ValueError("unsupported volume file")
            self._buckets = {int(k): int(v) for k, v in raw["buckets"].items()}
        except FileNotFoundError:
            self._buckets = {}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Write volume %s unreadable (%s); starting empty", self.path, exc
            )
            self._buckets = {}
        self._prune(self.clock())

    def used(self, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        self._prune(now)
        return sum(self._buckets.values())

    def over(self, now: float | None = None) -> bool:
        return self.warn_bytes > 0 and self.used(now) > self.warn_bytes

    def charge(self, nbytes: int, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        b = self._bucket(now)
        self._buckets[b] = self._buckets.get(b, 0) + int(nbytes)
        self._dirty = True
        if (
            self._persisted_at is None
            or now - self._persisted_at >= self.persist_interval_s
        ):
            self.persist(now)

    def persist(self, now: float | None = None) -> None:
        if not self._dirty:
            return
        now = self.clock() if now is None else now
        self._prune(now)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "version": 1,
                        "bucket_s": self.bucket_s,
                        "buckets": {str(k): v for k, v in self._buckets.items()},
                    },
                    f,
                )
            os.replace(tmp, self.path)
            self._dirty = False
            self._persisted_at = now
        except OSError as exc:
            logger.warning("Write volume %s not persisted: %s", self.path, exc)


@dataclass
class DelayedBatch:
    """Keys of one cascaded store, kept in original (prefix-chain) order."""

    due: float
    keys: list[OffloadKey]
    req_context: ReqContext
    seq: int = 0


@dataclass
class DelayedWriteQueue:
    """FIFO of pending cascades for one secondary tier (no CPU pinning)."""

    delay_s: float
    page_bytes: int
    volume: WriteVolume | None = None
    latest_only: bool = False
    clock: Callable[[], float] = time.monotonic
    stats: Counter = field(default_factory=Counter)
    _batches: deque = field(default_factory=deque)
    _pending: Counter = field(default_factory=Counter)
    _drops: dict = field(default_factory=dict)
    _seq: int = 0
    _warned_at: float | None = None

    def __post_init__(self) -> None:
        if self.delay_s < 0:
            raise ValueError("write_delay_s must be >= 0")
        if self.page_bytes <= 0:
            raise ValueError("page_bytes must be > 0")

    @property
    def depth(self) -> int:
        """Pending keys (a key queued twice counts twice)."""
        return sum(self._pending.values())

    def enqueue(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        now: float | None = None,
    ) -> None:
        keys = list(keys)
        if not keys:
            return
        now = self.clock() if now is None else now
        self._seq += 1
        self._batches.append(
            DelayedBatch(now + self.delay_s, keys, req_context, self._seq)
        )
        self._pending.update(keys)
        self.stats["enqueued"] += len(keys)

    def drop(self, keys: Iterable[OffloadKey], reason: str = "latest") -> int:
        """Mark pending keys so they are not written (all pending copies)."""
        assert reason in DROP_REASONS
        marked = 0
        for key in keys:
            if self._pending.get(key) and key not in self._drops:
                self._drops[key] = reason
                marked += 1
        return marked

    def _pop(self, batch: DelayedBatch) -> list[tuple[OffloadKey, str | None]]:
        out = []
        for key in batch.keys:
            reason = self._drops.get(key)
            self._pending[key] -= 1
            if self._pending[key] <= 0:
                del self._pending[key]
                self._drops.pop(key, None)
            out.append((key, reason))
        return out

    def take_due(self, now: float | None = None) -> list[DelayedBatch]:
        now = self.clock() if now is None else now
        due = []
        while self._batches and self._batches[0].due <= now:
            due.append(self._batches.popleft())
        return due

    def take_all_newest_first(self) -> list[DelayedBatch]:
        """At shutdown: everything still pending, the most recent store first
        (the conversations active at shutdown are the likeliest to resume)."""
        batches = list(reversed(self._batches))
        self._batches.clear()
        return batches

    def select(
        self,
        batch: DelayedBatch,
        is_ready_resident: Callable[[OffloadKey], bool],
        is_demoted: Callable[[OffloadKey], bool],
    ) -> list[OffloadKey]:
        """Keys of a batch that should be written now, original order kept."""
        live = []
        for key, mark in self._pop(batch):
            if mark is not None:
                self.stats["dropped_" + mark] += 1
            elif not is_ready_resident(key):
                self.stats["dropped_evicted"] += 1
            elif is_demoted(key):
                self.stats["dropped_superseded"] += 1
            else:
                live.append(key)
        return live

    def account(self, keys: list[OffloadKey], now_wall: float | None = None) -> None:
        """Count one submitted write job; warn (hourly at most) above the
        rolling 24 h threshold. Never refuses the write."""
        self.stats["written"] += len(keys)
        if self.volume is None:
            return
        now_wall = self.volume.clock() if now_wall is None else now_wall
        self.volume.charge(len(keys) * self.page_bytes, now_wall)
        if self.volume.over(now_wall):
            self.stats["over_threshold_jobs"] += 1
            if self._warned_at is None or now_wall - self._warned_at >= 3600:
                self._warned_at = now_wall
                self.stats["over_threshold_warnings"] += 1
                logger.warning(
                    "Secondary tier wrote %d bytes in the last 24 h, above the "
                    "%d byte warning threshold (writes continue)",
                    self.volume.used(now_wall),
                    self.volume.warn_bytes,
                )

    def summary(self) -> dict[str, int]:
        out = {k: int(v) for k, v in self.stats.items()}
        out["depth"] = self.depth
        if self.volume is not None:
            out["written_bytes_24h"] = self.volume.used()
            out["warn_bytes_24h"] = self.volume.warn_bytes
        return out
