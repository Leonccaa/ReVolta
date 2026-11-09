# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-only store accounting. No probes, payload reads or device sync.

Counters update at existing lifecycle transitions. Export is a delta batch at
most once per minute; request state dies with its Request, samples are bounded.
CPU/SSD residency here does not claim global (GPU-inclusive) serviceability.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)
SOURCES = ("gpu", "host_reuse", "host_load", "ssd", "compute")
TTFT_SOURCES = ("gpu", "host", "ssd", "compute", "mixed")
WAIT_REASONS = ("ssd_prefetch", "connector", "receive", "capacity", "budget")
CPU_SAMPLE_STRIDE = 17


@dataclass(slots=True)
class PrefixAttempt:
    attempt: int
    lookup: set[str] = field(default_factory=set)
    admitted: bool = False
    restored: bool = False
    failed: bool = False
    local: int = 0
    external: int = 0
    local_queries: int = 0
    external_queries: int = 0
    coverage_end: int = 0
    queued: float = 0.0
    scheduled: float = 0.0
    first_token: bool = False
    receive_start: float = 0.0
    prefetch_start: float = 0.0
    prefetch_seconds: float = 0.0
    receive_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    # Disjoint MAIN-token coverage. STATE/DRAFT never add token coverage.
    intervals: list[tuple[int, int, str]] = field(default_factory=list)
    ssd_handles: set[Any] = field(default_factory=set)


def _finish_wait(state: PrefixAttempt, phase: str, now: float) -> None:
    start_attr = phase + "_start"
    start = getattr(state, start_attr)
    if start:
        seconds = max(0.0, now - start)
        total_attr = phase + "_seconds"
        setattr(state, total_attr, getattr(state, total_attr) + seconds)
        if not state.scheduled:
            state.queue_wait_seconds += seconds
        setattr(state, start_attr, 0.0)


def cover(intervals: list[tuple[int, int, str]], start: int, end: int, source: str):
    """Assign a unique half-open interval; new physical evidence wins overlaps."""
    assert source in SOURCES
    if end <= start:
        return
    out = []
    for lo, hi, old_source in intervals:
        if hi <= start or lo >= end:
            out.append((lo, hi, old_source))
        else:
            if lo < start:
                out.append((lo, start, old_source))
            if hi > end:
                out.append((end, hi, old_source))
    out.append((start, end, source))
    intervals.clear()
    for lo, hi, label in sorted(out):
        if intervals and intervals[-1][1] == lo and intervals[-1][2] == label:
            intervals[-1] = (intervals[-1][0], hi, label)
        else:
            intervals.append((lo, hi, label))


class StoreMetrics:
    def __init__(self):
        self.counters: Counter[str] = Counter()
        self.cpu_ns: deque[int] = deque(maxlen=8192)
        self.observations: deque[tuple[str, str, float]] = deque(maxlen=4096)
        self.next_export = 0.0
        self.previous: dict[str, int] = {}

    def state(self, request: Any) -> PrefixAttempt:
        state = request.store_attempt
        if state is None or state.attempt != request.prefix_cache_attempt:
            state = request.store_attempt = PrefixAttempt(
                request.prefix_cache_attempt,
                queued=state.queued if state is not None else 0.0,
                first_token=state.first_token if state is not None else False,
                scheduled=state.scheduled if state is not None else 0.0,
                prefetch_seconds=state.prefetch_seconds if state is not None else 0.0,
                receive_seconds=state.receive_seconds if state is not None else 0.0,
                queue_wait_seconds=state.queue_wait_seconds
                if state is not None
                else 0.0,
                prefetch_start=state.prefetch_start if state is not None else 0.0,
            )
        return state

    def queued(self, request: Any, now: float) -> None:
        self.state(request).queued = now

    def begin_lookup(self, request: Any) -> None:
        """A fresh admission try, including a retry within the same step."""
        request.prefix_cache_attempt += 1
        state = request.store_attempt
        # The empty queued record has no job or observation tied to its ID.
        if state is not None and not (state.lookup or state.admitted or state.failed):
            state.attempt = request.prefix_cache_attempt

    def lookup(self, request: Any, tier: str, tokens: int, hits: int) -> None:
        if getattr(request, "skip_reading_prefix_cache", False):
            return
        state = self.state(request)
        if tier not in state.lookup:
            state.lookup.add(tier)
            self.counters[f"prefix_lookup_{tier}_queries"] += tokens
            self.counters[f"prefix_lookup_{tier}_hits"] += hits

    def retry(self, request: Any, reason: str, now: float) -> None:
        assert reason in WAIT_REASONS
        self.counters[f"admission_retry_{reason}"] += 1
        state = self.state(request)
        if reason == "ssd_prefetch" and not state.prefetch_start:
            state.prefetch_start = now

    def prefetch_ready(self, request: Any, now: float) -> None:
        state = self.state(request)
        _finish_wait(state, "prefetch", now)

    def admit(
        self,
        request: Any,
        local: int,
        external: int,
        host: bool,
        now: float,
        *,
        local_query: bool = True,
        external_query: bool = True,
    ):
        state = self.state(request)
        if state.admitted:
            return
        state.admitted = True
        state.local, state.external = local, external
        state.local_queries = request.num_tokens if local_query else 0
        state.external_queries = request.num_tokens - local if external_query else 0
        # A full hit still replays the final token to produce logits. Keep
        # that token in compute, without counting an extra prompt token.
        state.coverage_end = max(
            0, min(local + external, request.num_prompt_tokens, request.num_tokens - 1)
        )
        self.counters["prefix_admitted_local_queries"] += state.local_queries
        self.counters["prefix_admitted_local_hits"] += local
        self.counters["prefix_admitted_external_queries"] += state.external_queries
        self.counters["prefix_admitted_external_hits"] += external
        _finish_wait(state, "prefetch", now)
        cover(state.intervals, 0, request.num_prompt_tokens, "compute")
        cover(
            state.intervals,
            0,
            min(local, state.coverage_end),
            "host_reuse" if host else "gpu",
        )
        cover(
            state.intervals,
            local,
            state.coverage_end,
            "host_load",
        )
        if external:
            state.receive_start = now
        else:
            self.restored(request, now)

    def restored(self, request: Any, now: float) -> None:
        state = self.state(request)
        if state.restored or state.failed or not state.admitted:
            return
        state.restored = True
        self.counters["prefix_restored_local_queries"] += state.local_queries
        self.counters["prefix_restored_local_hits"] += state.local
        self.counters["prefix_restored_external_queries"] += state.external_queries
        self.counters["prefix_restored_external_hits"] += state.external
        _finish_wait(state, "receive", now)

    def fail(self, request: Any, now: float | None = None) -> None:
        state = self.state(request)
        if now is not None:
            _finish_wait(state, "prefetch", now)
            _finish_wait(state, "receive", now)
        if not state.failed:
            state.failed = True
            self.counters["prefix_restore_failed"] += 1
            cover(state.intervals, state.local, request.num_prompt_tokens, "compute")

    def preempt(self, request: Any, now: float) -> None:
        state = self.state(request)
        _finish_wait(state, "prefetch", now)
        _finish_wait(state, "receive", now)
        state.failed = True
        self.counters["prefix_attempt_preempted"] += 1

    def complete_receive(
        self,
        request: Any,
        attempt: PrefixAttempt | None,
        *,
        txn: Any = None,
        block_tokens: int = 0,
        ok: bool = True,
        now: float,
    ) -> None:
        # A job owns the actual attempt object, not just a reusable request ID.
        # Preemption may advance the generation before the next lookup creates
        # its replacement object. Neither kind of late receipt may touch it.
        if (
            attempt is None
            or request.store_attempt is not attempt
            or request.prefix_cache_attempt != attempt.attempt
        ):
            self.counters["prefix_receipt_stale"] += 1
            return
        finished = getattr(request, "is_finished", None)
        if finished is not None and finished() and not attempt.first_token:
            _finish_wait(attempt, "prefetch", now)
            _finish_wait(attempt, "receive", now)
            attempt.failed = True
            self.counters["prefix_restore_cancelled"] += 1
            return
        if not ok:
            self.fail(request, now)
        elif txn is not None:
            self.received(request, txn, block_tokens, now)
        else:
            self.restored(request, now)

    def received(self, request: Any, txn: Any, block_tokens: int, now: float):
        state = self.state(request)
        if txn.cancelled or txn.failed_ranks:
            self.fail(request, now)
            return
        for entry in txn.entries:
            lo = entry.logical_index * block_tokens
            hi = min(
                lo + block_tokens,
                request.num_prompt_tokens,
                state.coverage_end,
            )
            # Only this request's verified prefetch holds attest an SSD read.
            # persisted_epoch by itself never supplies request read lineage.
            source = (
                "host_reuse" if entry.source.value == "reuse_local" else "host_load"
            )
            if entry.handle in state.ssd_handles:
                source = "ssd"
            cover(state.intervals, lo, hi, source)
        self.restored(request, now)

    def observe(self, phase: str, source: str, seconds: float) -> None:
        assert source in TTFT_SOURCES
        if len(self.observations) == self.observations.maxlen:
            self.counters["observation_samples_dropped"] += 1
        self.observations.append((phase, source, max(0.0, seconds)))

    def first_token(self, request: Any, now: float) -> None:
        state = self.state(request)
        if state.first_token:
            return
        state.first_token = True
        amounts: Counter[str] = Counter()
        for lo, hi, source in state.intervals:
            amounts[source] += hi - lo
        for source, tokens in amounts.items():
            self.counters[f"prompt_tokens_{source}"] += tokens
        cached = {
            "host" if s.startswith("host_") else s
            for s, n in amounts.items()
            if n and s != "compute"
        }
        source = (
            next(iter(cached)) if len(cached) == 1 else "mixed" if cached else "compute"
        )
        self.observe("ttft", source, now - state.queued)
        if state.scheduled:
            self.observe(
                "queue",
                source,
                state.scheduled - state.queued - state.queue_wait_seconds,
            )
            self.observe("scheduled_to_token", source, now - state.scheduled)
        self.observe("ssd_prefetch", source, state.prefetch_seconds)
        self.observe("receive", source, state.receive_seconds)

    def step_cpu(self, ns: int) -> None:
        if len(self.cpu_ns) == self.cpu_ns.maxlen:
            self.counters["cpu_samples_rotated"] += 1
        self.cpu_ns.append(ns)

    def export(
        self,
        now: float,
        *,
        pool: Any = None,
        ssd: Any = None,
        arbiter: Any = None,
        manager: Any = None,
        ledger: Any = None,
    ) -> dict[str, Any] | None:
        if now < self.next_export:
            return None
        self.next_export = now + 60.0
        gauges: dict[str, float] = {"step_cpu_sample_stride": CPU_SAMPLE_STRIDE}
        totals: dict[str, int] = {}
        if pool is not None:
            table = pool.host_pages
            gauges.update(
                {f"host_pages_{s.value}": n for s, n in table.state_counts.items()}
            )
            allocated = len(table.pages) - table._first - table.num_free()
            gauges["host_persisted_ratio"] = (
                table.persisted_pages / allocated if allocated else 0.0
            )
            gauges["host_persisted_pages"] = table.persisted_pages
            if ssd is not None:
                gauges["host_dirty_bytes"] = (
                    allocated - table.persisted_pages
                ) * ssd.page_bytes
            elif ledger is not None:
                gauges["host_dirty_bytes"] = (
                    allocated - table.persisted_pages
                ) * ledger.plan.geometry.host_bytes_per_block
            gauges.update(
                host_available=table.available(),
                retirement_heap=len(table._pending),
                retirement_cleanup_heap=len(table._retiring),
                retirement_heap_stale=len(table._pending) - len(table._pending_live),
                retirement_cleanup_stale=len(table._retiring)
                - len(table._retiring_live),
                collect_work_max=table.collect_work_max,
            )
            totals["heap_compactions"] = table.heap_compactions
            if pool.elastic:
                snap = pool.arbiter_snapshot()
                gauges.update(
                    gpu_inuse=snap.inuse,
                    gpu_cached=snap.cached,
                    gpu_donated=pool.num_donated(),
                    gpu_held=len(pool.held),
                    gpu_floor=pool.floor_slots,
                    gpu_t_idx=pool.index_cache_target,
                    gpu_hot_capacity_slots=pool.hot_slots(),
                )
        from vllm.v1.kv_offload.unified.manager import UnifiedStoreManager

        if isinstance(manager, UnifiedStoreManager):
            snapshot = manager.snapshot()
            gauges.update(
                {
                    f"unified_{k}": v
                    for k, v in snapshot.items()
                    if not k.startswith("dma_")
                }
            )
            totals.update(
                {f"unified_{k}": v for k, v in snapshot.items() if k.startswith("dma_")}
            )
            totals.update({f"unified_{k}": v for k, v in manager.events.items()})
            totals.update({f"unit_{k}": v for k, v in manager.units.events.items()})
            totals.update(
                {f"reservation_{k}": v for k, v in manager.ledger.events.items()}
            )
        if ledger is not None:
            gauges["gpu_hot_committed_groups"] = ledger.hot_used
        if ssd is not None:
            totals.update({f"host_ssd_{k}": v for k, v in ssd.stats_counter.items()})
            totals["host_ssd_verified_load_bytes"] = (
                ssd.stats_counter["prefetch_verified"] * ssd.page_bytes
            )
            delayed = ssd._queue.summary()
            totals.update(
                {
                    f"host_delayed_{k}": v
                    for k, v in delayed.items()
                    if k.startswith("dropped_")
                }
            )
            gauges.update(
                host_ssd_jobs=len(ssd._jobs), host_delayed_depth=delayed["depth"]
            )
        if manager is not None:
            fn = getattr(manager, "delayed_write_stats", None)
            if fn is not None:
                for tier, stats in fn().items():
                    totals.update(
                        {
                            f"delayed_{tier}_{k}": v
                            for k, v in stats.items()
                            if k.startswith("dropped_")
                        }
                    )
        if arbiter is not None:
            totals.update(
                {f"arbiter_{k}": v for k, v in arbiter.decision_counts.items()}
            )
        for name, value in totals.items():
            self.counters[name] += max(0, value - self.previous.get(name, 0))
        self.previous = totals
        cpu = sorted(self.cpu_ns)
        gauges["step_cpu_samples"] = len(cpu)
        if cpu:
            for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
                i = (len(cpu) - 1) * q
                lo = int(i)
                gauges[f"step_cpu_seconds_{name}"] = (
                    cpu[lo] + (cpu[min(lo + 1, len(cpu) - 1)] - cpu[lo]) * (i - lo)
                ) / 1e9
        result = dict(
            counters=dict(self.counters),
            gauges=gauges,
            observations=list(self.observations),
        )
        self.counters.clear()
        self.observations.clear()
        self.cpu_ns.clear()
        logger.info("Store stats: counters=%s gauges=%s", result["counters"], gauges)
        return result


class StoreMetricsProm:
    def __init__(
        self,
        counter_cls: Any,
        gauge_cls: Any,
        histogram_cls: Any,
        labelnames: list[str],
        values: dict[int, list[object]],
    ):
        self.values = values
        self.events = counter_cls(
            name="vllm:store_events",
            documentation=(
                "Store lifecycle event deltas (lookup/admitted/restored are distinct)."
            ),
            labelnames=labelnames + ["event"],
        )
        self.state = gauge_cls(
            name="vllm:store_state",
            documentation=(
                "Store state; persisted means successfully written incarnation, "
                "not current SSD recoverability."
            ),
            labelnames=labelnames + ["metric"],
            multiprocess_mode="mostrecent",
        )
        self.latencies = {
            phase: histogram_cls(
                name=f"vllm:store_{phase}_seconds",
                documentation=(
                    f"Scheduler observed {phase}; no device synchronization. "
                    "Host reuse/load share the host TTFT label."
                ),
                labelnames=labelnames + ["source"],
                buckets=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
            )
            for phase in (
                "ttft",
                "queue",
                "scheduled_to_token",
                "ssd_prefetch",
                "receive",
            )
        }

    def observe(self, payload: dict[str, Any], engine: int) -> None:
        labels = self.values[engine]
        for event, count in payload["counters"].items():
            self.events.labels(*labels, event).inc(count)
        for name, value in payload["gauges"].items():
            self.state.labels(*labels, name).set(value)
        for phase, source, seconds in payload["observations"]:
            self.latencies[phase].labels(*labels, source).observe(seconds)
