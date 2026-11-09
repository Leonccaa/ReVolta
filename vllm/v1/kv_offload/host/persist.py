# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD persistence for v2 host pages: delayed write, write-back, final flush.

Design D-B: a published READY page is queued for a delayed write without a
pin. A page that is about to be evicted unwritten gets one immediate write
when net headroom allows; otherwise it is evicted and counted. At engine
shutdown ``final_flush`` writes the remaining pages, most recent first,
within a best-effort time budget. SSD prefetch is bounded by global and
per-request quotas.

All controller methods run on the scheduler thread. An IO thread only reads
or writes a row while that row holds an SSD IO owner in HostPageTable, so a
row is never recycled under a transfer. SSD owner ids are negative.
"""

import hashlib
import os
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.host.page_table import (
    HostPage,
    HostPageTable,
    PageHandle,
    PageState,
)
from vllm.v1.kv_offload.host.ssd import HostKVLayout, _HostTier, _WriteRateLimiter
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.delayed_write import (
    VOLUME_FILE,
    DelayedWriteQueue,
    WriteVolume,
)
from vllm.v1.kv_offload.tiering.fs.config import host_kv_ssd_config

logger = init_logger(__name__)

# Bump whenever the persisted row format changes. Old files then never hit
# (no cross-version migration, Leon 2026-11-07).
HOST_PAGE_ABI = "host-page-v2.1:row=workers*[main_kv|logical_index]"
# The engine-core process is SIGKILLed --shutdown-timeout after SIGTERM.
# Every shutdown step must finish this long before that point.
SHUTDOWN_MARGIN_S = 5.0


@dataclass
class FlushReport:
    """``candidates = completed + failed + dropped_unsubmitted +
    in_flight_at_deadline`` and ``in_flight_at_deadline = late_completed +
    late_failed + stuck``. Late results finished after the budget, while the
    IO grace period drained writes that had already started."""

    candidates: int = 0
    completed: int = 0
    failed: int = 0
    dropped_unsubmitted: int = 0
    in_flight_at_deadline: int = 0
    late_completed: int = 0
    late_failed: int = 0
    stuck: int = 0
    other_io_stuck: int = 0
    planned_budget_s: float = 0.0
    flush_elapsed_s: float = 0.0
    deadline_hit: bool = False
    skipped: str | None = None
    quarantined: bool = False


@dataclass(frozen=True)
class FlushPlan:
    """Budget and absolute deadline shared with the ordinary fs drain."""

    budget_s: float
    deadline: float | None
    t0: float | None
    reason: str


def plan_final_flush(
    configured_s: float,
    shutdown_timeout: float,
    deadline: float | None,
    now: float,
    t0: float | None = None,
) -> FlushPlan:
    """budget = min(configured, deadline - now).

    ``deadline`` is the engine's shared shutdown deadline (t0 +
    --shutdown-timeout - 5 s), also used by the page ownership drain and the fs drain.
    Without it nothing kills the process, so only the configured budget
    applies.
    """
    if shutdown_timeout <= 0:
        return FlushPlan(0.0, None, t0, "shutdown_timeout=0 skips final flush")
    if deadline is None:
        return FlushPlan(configured_s, None, t0, "no shutdown deadline")
    budget = min(configured_s, deadline - now)
    if budget <= 0:
        return FlushPlan(0.0, deadline, t0, "shutdown window exhausted")
    return FlushPlan(budget, deadline, t0, "ok")


def shared_shutdown_deadline(t0: float | None, shutdown_timeout: float) -> float | None:
    if t0 is None or shutdown_timeout <= 0:
        return None
    return t0 + shutdown_timeout - SHUTDOWN_MARGIN_S


def shutdown_window_warning(
    flush_s: float, fs_drain_s: float, shutdown_timeout: float
) -> str | None:
    if shutdown_timeout <= 0:
        return "--shutdown-timeout 0 skips the host KV SSD final flush"
    if flush_s + fs_drain_s + SHUTDOWN_MARGIN_S > shutdown_timeout:
        return (
            f"host KV SSD flush {flush_s:g} s + fs drain {fs_drain_s:g} s + "
            f"{SHUTDOWN_MARGIN_S:g} s margin exceeds --shutdown-timeout "
            f"{shutdown_timeout:g} s; the flush will be truncated"
        )
    return None


@dataclass
class _HostSpec:
    """The parts of OffloadingSpec that the revolta_fs namespace reads."""

    vllm_config: Any
    kv_cache_config: Any
    persistent_layout: dict
    block_size_factor: int = 1


@dataclass(eq=False)
class _Job:
    job_id: int
    handle: PageHandle
    tier_key: OffloadKey
    loading: bool
    submitted: bool = False
    flush: bool = False


@dataclass
class _Round:
    started: float
    loading: set[PageHandle] = field(default_factory=set)
    holds: list[tuple[PageHandle, int]] = field(default_factory=list)


class HostPageSsd:
    """SSD controller over the scheduler's HostPageTable and row memory."""

    def __init__(
        self,
        table: HostPageTable,
        view: memoryview,
        spec: Any,
        layout: HostKVLayout | dict,
        options: dict,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        if view.ndim != 2 or view.shape is None or view.shape[0] != len(table.pages):
            raise ValueError("Host SSD view must be [host pages, row bytes]")
        self.table = table
        self.options = options
        self.clock = clock
        self.page_bytes = view.nbytes // view.shape[0]
        self._window = options["write_threads"]
        self._max_pending = options["max_pending_blocks"]
        self._jobs: dict[int, _Job] = {}
        self._page_jobs: dict[int, int] = {}
        self._submit: deque[int] = deque()
        self._deferred: OrderedDict[PageHandle, None] = OrderedDict()
        self._inflight_stores = 0
        self._inflight_loads = 0
        self._next_job = 0
        self._next_lease = 0
        self._salts: dict[int, bytes] = {}
        self._rounds: dict[str, _Round] = {}
        self._waiters: dict[PageHandle, set[str]] = {}
        self._flushing = False
        self._report: FlushReport | None = None
        self._phase = "run"
        self._closed = False
        self.quarantined = False
        self.stats_counter: Counter[str] = Counter()
        self._ctx = ReqContext(req_id="host-kv-v2")
        layout_dict = asdict(layout) if is_dataclass(layout) else dict(layout)
        self._spec = _HostSpec(
            spec.vllm_config,
            spec.kv_cache_config,
            {
                **(getattr(spec, "persistent_layout", None) or {}),
                "host_kv_v2": {
                    "abi": HOST_PAGE_ABI,
                    "layout": layout_dict,
                    "page_bytes": self.page_bytes,
                    "io": "buffered-host-page-v2",
                },
            },
            getattr(spec, "block_size_factor", 1),
        )
        self._limiter = _WriteRateLimiter(options["write_bytes_per_second"])
        self._tier = _HostTier(
            self._limiter,
            offloading_spec=self._spec,
            primary_kv_view=view,
            tier_type="revolta_fs",
            root_dir=options["root_dir"],
            max_size_bytes=options["max_size_bytes"],
            min_free_bytes=options.get("min_free_bytes"),
            max_files=options["max_files"],
            content_manifest_path=options.get("content_manifest_path"),
            content_verify=options.get("content_verify", "stat"),
            n_read_threads=1,
            n_write_threads=self._window,
            # Jobs never expire while their rows are in use; this controller
            # bounds queued work itself, so tier backpressure never skips.
            store_timeout_s=0,
            drain_timeout_s=0,
            max_inflight_store_blocks=self._max_pending + self._window,
            write_delay_s=0,
        )
        volume = None
        try:
            volume = WriteVolume.shared(
                os.path.join(os.path.realpath(options["root_dir"]), VOLUME_FILE),
                options["write_warn_bytes_24h"],
            )
        except ValueError as exc:
            logger.warning("Host KV SSD write volume not shared: %s", exc)
        # Queue items are physical handles, not keys: an evicted incarnation
        # can never be confused with a later page of the same hash.
        self._queue: Any = DelayedWriteQueue(
            delay_s=float(options["write_delay_s"]),
            page_bytes=self.page_bytes,
            volume=volume,
            clock=clock,
        )
        if self.enabled:
            table.persistence = self

    # --- state -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return not self._closed and not self._tier.disabled

    def stats(self) -> dict[str, int]:
        out = {f"delayed_{k}": v for k, v in self._queue.summary().items()}
        out.update(self.stats_counter)
        out["jobs"] = len(self._jobs)
        out["deferred"] = len(self._deferred)
        return out

    def has_pending(self) -> bool:
        return bool(self._jobs or self._queue.depth or self._deferred or self._rounds)

    def owns_io(self, owner_id: int) -> bool:
        return owner_id in self._jobs

    def _page(self, handle: PageHandle) -> HostPage | None:
        page = self.table.pages[handle.page_id]
        return page if page.handle == handle else None

    def _writable(self, page: HostPage) -> bool:
        return (
            page.valid
            and page.state == PageState.READY
            and page.key is not None
            and page.handle.page_id >= 1
            and not self.table.is_persisted(page)
            and page.handle.page_id not in self._page_jobs
        )

    def _resident(self, handle: PageHandle) -> bool:
        page = self._page(handle)
        return page is not None and page.valid and page.state == PageState.READY

    def _tier_key(self, key: OffloadKey) -> OffloadKey:
        """Epoch 0 is stable across restarts. A reset gets a process-unique
        salt, so pre- and post-reset files can never hit each other."""
        epoch = self.table.ssd_epoch
        if epoch == 0:
            return key
        salt = self._salts.setdefault(epoch, os.urandom(16))
        digest = hashlib.sha256(salt + bytes(key)).digest()
        return make_offload_key(digest, get_offload_group_idx(key))

    def _store_jobs(self) -> int:
        return len(self._jobs) - self._inflight_loads

    # --- jobs ------------------------------------------------------------

    def _new_job(self, page: HostPage, loading: bool, front: bool = False) -> _Job:
        assert page.key is not None
        self._next_job -= 1
        job = _Job(self._next_job, page.handle, self._tier_key(page.key), loading)
        self.table.start_io(page.handle, job.job_id)
        self._jobs[job.job_id] = job
        self._page_jobs[page.handle.page_id] = job.job_id
        if loading:
            self._submit_to_tier(job)
        elif front:
            self._submit.appendleft(job.job_id)
        else:
            self._submit.append(job.job_id)
        return job

    def _submit_to_tier(self, job: _Job) -> None:
        meta = JobMetadata(
            job_id=job.job_id,
            keys=[job.tier_key],
            block_ids=np.array([job.handle.page_id], dtype=np.int64),
            is_promotion=job.loading,
            req_context=self._ctx,
        )
        try:
            if job.loading:
                self._tier.submit_load(meta)
                self._inflight_loads += 1
            else:
                self._tier.submit_store(meta)
                self._inflight_stores += 1
                self._queue.account([job.tier_key])
        except Exception:
            logger.exception("Host KV SSD job %d submission failed", job.job_id)
            self.stats_counter["submit_failures"] += 1
            self._finish(job, False)
            return
        job.submitted = True

    def _pump(self) -> None:
        while self._submit and self._inflight_stores < self._window:
            job = self._jobs[self._submit.popleft()]
            self._submit_to_tier(job)

    def _finish(self, job: _Job, success: bool) -> None:
        self._jobs.pop(job.job_id, None)
        if self._page_jobs.get(job.handle.page_id) == job.job_id:
            del self._page_jobs[job.handle.page_id]
        if job.submitted:
            if job.loading:
                self._inflight_loads -= 1
            else:
                self._inflight_stores -= 1
        if job.loading:
            ok = self.table.complete_load(job.handle, success)
            self.stats_counter["prefetch_verified" if ok else "prefetch_failed"] += 1
            for req_id in self._waiters.pop(job.handle, set()):
                round_ = self._rounds.get(req_id)
                if round_ is None or job.handle not in round_.loading:
                    continue
                round_.loading.discard(job.handle)
                if ok:
                    # The waiter's hold keeps the new READY page until its
                    # lookup pins it, it is cancelled, or T_pf passes.
                    self._next_lease -= 1
                    self.table.pin(job.handle, self._next_lease)
                    round_.holds.append((job.handle, self._next_lease))
        elif success and self.table.mark_persisted(job.handle):
            self.stats_counter["pages_written"] += 1
        else:
            self.stats_counter[
                "store_failures" if job.submitted else "store_cancelled"
            ] += 1
        report = self._report
        if job.flush and report is not None:
            if self._phase == "budget":
                if success:
                    report.completed += 1
                else:
                    report.failed += 1
            elif self._phase == "grace":
                if success:
                    report.late_completed += 1
                else:
                    report.late_failed += 1
        self.table.finish_io(job.handle, job.job_id)

    def _poll(self) -> None:
        for result in self._tier.get_finished_jobs():
            job = self._jobs.get(result.job_id)
            if job is not None:
                self._finish(job, result.success)

    # --- PersistenceHooks ------------------------------------------------

    def on_published(self, handle: PageHandle) -> None:
        if self.enabled and not self._flushing:
            self._queue.enqueue([handle], self._ctx)

    def try_writeback(self, page: HostPage, headroom_before: int) -> bool:
        """Before eviction: one immediate write (still rate limited) when the
        queue has room and net headroom pays for the page. Never blocks."""
        if not self.enabled or self._flushing or not self._writable(page):
            return False
        if self._store_jobs() >= self._max_pending:
            self.stats_counter["writeback_queue_full"] += 1
            return False
        # The page is reclaimable now; the write makes it non-reclaimable.
        if headroom_before < 1:
            self.stats_counter["writeback_no_headroom"] += 1
            return False
        self._new_job(page, False, front=True)
        self.stats_counter["writeback_submitted"] += 1
        self._pump()
        return True

    def on_evict(self, page: HostPage) -> None:
        if page.state != PageState.READY or page.key is None:
            return
        if self.table.is_persisted(page):
            self.stats_counter["evicted_persisted"] += 1
            return
        self.stats_counter["evict_unwritten"] += 1
        self._queue.drop([page.handle], "evicted")
        self._deferred.pop(page.handle, None)

    def _discard_queue(self) -> None:
        for batch in self._queue.take_all_newest_first():
            self._queue._pop(batch)

    def on_reset(self) -> None:
        self._discard_queue()
        self._deferred.clear()
        for job_id in list(self._submit):
            self._finish(self._jobs[job_id], False)
        self._submit.clear()
        for req_id in list(self._rounds):
            self.cancel(req_id)

    # --- per step --------------------------------------------------------

    def step(self) -> None:
        """Each scheduler step before admission, and while idle.

        Work per call is bounded by the free store slots: deferred pages
        wait in order and are not rescanned while the queue is full.
        """
        if self._closed or self._flushing:
            return
        self._poll()
        self._expire_prefetch(self.clock())
        for batch in self._queue.take_due():
            for handle in self._queue.select(batch, self._resident, lambda h: False):
                self._deferred[handle] = None
        headroom: int | None = None
        while self._deferred and self._store_jobs() < self._max_pending:
            handle = next(iter(self._deferred))
            del self._deferred[handle]
            page = self._page(handle)
            if page is None or not self._writable(page):
                if page is not None and self.table.is_persisted(page):
                    self.stats_counter["already_persisted"] += 1
                continue
            cost = int(self.table.reclaimable(page))
            if headroom is None:
                headroom = self.table.net_headroom()
            if headroom < cost:
                # Retry next step, first in line .
                self._deferred[handle] = None
                self._deferred.move_to_end(handle, last=False)
                self.stats_counter["deferred_no_headroom"] += 1
                break
            self._new_job(page, False)
            headroom -= cost
        if self._deferred and self._store_jobs() >= self._max_pending:
            self.stats_counter["deferred_queue_full"] += 1
        self._pump()

    # --- prefetch quota ---------------------------------------------------

    def prefetch(self, req_id: str, keys: Iterable[OffloadKey]) -> list[OffloadKey]:
        """One prefetch round per request per T_pf, in chain order.

        Connector lookup calls this for
        the first missing keys,
        then pins ``holds(req_id)`` as NEW_LOAD sources and calls
        ``release_holds``. No hit is reported before verification.
        """
        if not self.enabled or self._flushing:
            return []
        now = self.clock()
        self._expire_prefetch(now)
        if req_id in self._rounds:
            self.stats_counter["prefetch_round_limited"] += 1
            return []
        round_ = _Round(now)
        issued: list[OffloadKey] = []
        headroom: int | None = None
        per_req = self.options["prefetch_max_per_request"]
        for key in keys:
            handle = self.table.lookup(key)
            if handle is not None:
                page = self.table.get(handle)
                if (
                    page.state == PageState.LOADING
                    and page.handle.page_id in self._page_jobs
                    and len(round_.loading) < per_req
                ):
                    round_.loading.add(handle)
                    self._waiters.setdefault(handle, set()).add(req_id)
                continue
            if len(round_.loading) >= per_req:
                self.stats_counter["prefetch_request_quota"] += 1
                break
            if self._inflight_loads >= self.options["prefetch_max_inflight"]:
                self.stats_counter["prefetch_global_quota"] += 1
                break
            try:
                if self._tier.lookup(self._tier_key(key)) is not True:
                    break
            except OSError:
                self.stats_counter["prefetch_lookup_failures"] += 1
                break
            if headroom is None:
                headroom = self.table.net_headroom()
            if headroom < 1:
                self.stats_counter["prefetch_no_headroom"] += 1
                break
            handle = self.table.reserve(loading_key=key, writeback=False)
            if handle is None:
                break
            headroom -= 1
            self._new_job(self.table.get(handle), loading=True)
            round_.loading.add(handle)
            self._waiters.setdefault(handle, set()).add(req_id)
            issued.append(key)
        if round_.loading:
            self._rounds[req_id] = round_
            self.stats_counter["prefetch_issued"] += len(issued)
        return issued

    def holds(self, req_id: str) -> list[PageHandle]:
        round_ = self._rounds.get(req_id)
        return [] if round_ is None else [h for h, _ in round_.holds]

    def release_holds(self, req_id: str) -> None:
        round_ = self._rounds.get(req_id)
        if round_ is None:
            return
        for handle, lease in round_.holds:
            self.table.release(handle, lease)
        round_.holds.clear()

    def cancel(self, req_id: str) -> None:
        """Withdraw a request's waiting; physical reads finish on their own."""
        round_ = self._rounds.pop(req_id, None)
        if round_ is None:
            return
        for handle in round_.loading:
            waiters = self._waiters.get(handle)
            if waiters is not None:
                waiters.discard(req_id)
        for handle, lease in round_.holds:
            self.table.release(handle, lease)

    def _expire_prefetch(self, now: float) -> None:
        timeout = self.options["prefetch_timeout_s"]
        for req_id, round_ in list(self._rounds.items()):
            if now - round_.started >= timeout:
                if round_.loading or round_.holds:
                    self.stats_counter["prefetch_timeouts"] += 1
                self.cancel(req_id)

    # --- shutdown --------------------------------------------------------

    def final_flush(
        self,
        budget_s: float | None = None,
        deadline: float | None = None,
        io_grace_s: float | None = None,
    ) -> FlushReport:
        """Shutdown step 4: write unpersisted READY pages, newest first.

        The budget is best effort. At its end no new page is submitted and
        queued writes are cancelled; writes already started may drain for the
        IO grace period. A write that still does not finish quarantines the
        region (no unmap, no unlink) instead of hanging shutdown.
        """
        budget = (
            self.options["shutdown_flush_timeout_s"] if budget_s is None else budget_s
        )
        grace = (
            self.options["shutdown_io_grace_s"] if io_grace_s is None else io_grace_s
        )
        start = self.clock()
        report = FlushReport(planned_budget_s=max(0.0, budget))
        if self._closed:
            report.skipped = "closed"
            return report
        flush_deadline = start + max(0.0, budget)
        hard = None
        if deadline is not None:
            flush_deadline = min(flush_deadline, deadline)
            hard = deadline + SHUTDOWN_MARGIN_S / 2
        grace_deadline = flush_deadline + grace
        if hard is not None:
            grace_deadline = min(grace_deadline, hard)
        self._flushing = True
        self._report = report
        self._phase = "budget"
        self._poll()
        self._discard_queue()
        self._deferred.clear()
        for req_id in list(self._rounds):
            self.cancel(req_id)
        if flush_deadline <= start or not self.enabled:
            report.skipped = "no budget" if self.enabled else "disabled"
        else:
            self._limiter.unthrottle()
            rank: dict[int, int] = {}
            for page_id in reversed(self.table._lru):
                page = self.table.pages[page_id]
                if not (
                    page.valid
                    and page.state == PageState.READY
                    and page.key is not None
                    and not self.table.is_persisted(page)
                ):
                    continue
                rank[page_id] = len(rank)
                job_id = self._page_jobs.get(page_id)
                if job_id is not None:
                    self._jobs[job_id].flush = True
                else:
                    self._new_job(page, False).flush = True
            report.candidates = len(rank)
            self._submit = deque(
                sorted(
                    self._submit,
                    key=lambda j: rank.get(self._jobs[j].handle.page_id, -1),
                )
            )
            while True:
                self._poll()
                self._pump()
                if not any(job.flush for job in self._jobs.values()):
                    break
                if self.clock() >= flush_deadline:
                    report.deadline_hit = True
                    break
                time.sleep(0.002)
        # Budget over: cancel writes that have not started.
        self._phase = "cancel"
        for job_id in list(self._submit):
            job = self._jobs[job_id]
            if job.flush:
                report.dropped_unsubmitted += 1
            self._finish(job, False)
        self._submit.clear()
        report.in_flight_at_deadline = sum(job.flush for job in self._jobs.values())
        self._phase = "grace"
        while self._jobs and self.clock() < grace_deadline:
            self._poll()
            time.sleep(0.002)
        self._poll()
        report.stuck = sum(job.flush for job in self._jobs.values())
        report.other_io_stuck = len(self._jobs) - report.stuck
        if self._jobs:
            self.quarantined = True
        report.quarantined = self.quarantined
        report.flush_elapsed_s = self.clock() - start
        self._phase = "done"
        self._report = None
        return report

    def close(self, io_grace_s: float | None = None) -> None:
        """Shutdown step 5: join started IO, or quarantine if it never ends."""
        if self._closed:
            return
        self._flushing = True
        for job_id in list(self._submit):
            self._finish(self._jobs[job_id], False)
        self._submit.clear()
        for req_id in list(self._rounds):
            self.cancel(req_id)
        grace = (
            self.options["shutdown_io_grace_s"] if io_grace_s is None else io_grace_s
        )
        end = self.clock() + grace
        while self._jobs and self.clock() < end:
            self._poll()
            time.sleep(0.002)
        self._poll()
        self._closed = True
        if self._queue.volume is not None:
            self._queue.volume.persist()
        if self.table.persistence is self:
            self.table.persistence = None
        if self._jobs:
            self.quarantined = True
            # Stop the pool without joining: a hung writer keeps its row and
            # the region stays mapped (transfer regression). Queued tasks are dropped.
            pool = getattr(self._tier, "_pool", None)
            if pool is not None:
                pool.shutdown(wait=False)
            self._limiter.stopped.set()
            logger.error(
                "Host KV SSD IO did not finish; quarantining %d rows "
                "(region kept mapped and linked)",
                len(self._jobs),
            )
            return
        self._limiter.stopped.set()
        self._tier.shutdown()
        # Drop the row export so the scheduler can unmap the region.
        self._tier._primary_kv_view = None  # type: ignore[assignment]
        logger.info("Host KV SSD stats: %s", self.stats())


def create_host_page_ssd(
    vllm_config: Any,
    kv_cache_config: Any,
    table: HostPageTable,
    region: Any,
    layout: Any,
) -> HostPageSsd | None:
    """None unless kv_connector_extra_config["host_kv.ssd"]["enabled"]."""
    options = host_kv_ssd_config(vllm_config.kv_transfer_config)
    if options is None:
        return None
    identity = {
        "num_workers": layout.num_workers,
        "main_bytes": layout.main_bytes,
        "index_bytes": list(layout.index_bytes),
    }
    ssd = HostPageSsd(
        table,
        region.create_kv_memoryview(),
        _HostSpec(vllm_config, kv_cache_config, {}),
        identity,
        options,
    )
    tiers = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
        "secondary_tiers", []
    )
    fs_drain = max(
        (
            float(t.get("drain_timeout_s", 30.0))
            for t in tiers
            if isinstance(t, dict) and t.get("type") in ("fs", "revolta_fs")
        ),
        default=0.0,
    )
    warning = shutdown_window_warning(
        options["shutdown_flush_timeout_s"],
        fs_drain,
        getattr(vllm_config, "shutdown_timeout", 30),
    )
    if warning is not None:
        logger.warning(warning)
    if not ssd.enabled:
        ssd.close(io_grace_s=0)
        return None
    return ssd


def run_final_flush(
    ssd: HostPageSsd, plan: FlushPlan | None, now: float
) -> FlushReport:
    """Truncate the plan to the time left after the ordinary fs drain."""
    if plan is None:
        report = ssd.final_flush()
    elif plan.budget_s <= 0:
        logger.warning("Host KV SSD final flush skipped: %s", plan.reason)
        report = ssd.final_flush(budget_s=0.0, deadline=plan.deadline)
        report.skipped = plan.reason
    else:
        budget = plan.budget_s
        if plan.deadline is not None:
            budget = min(budget, plan.deadline - now)
        if budget <= 0:
            logger.warning(
                "Host KV SSD final flush skipped: shared shutdown window used"
            )
        report = ssd.final_flush(budget_s=max(0.0, budget), deadline=plan.deadline)
    end = ssd.clock()
    total = None if plan is None or plan.t0 is None else end - plan.t0
    logger.info(
        "Host KV SSD final flush: candidates=%d completed=%d failed=%d "
        "dropped_unsubmitted=%d in_flight_at_deadline=%d (late ok=%d "
        "late failed=%d stuck=%d) planned=%.2fs flush=%.2fs shutdown=%s "
        "skipped=%s quarantined=%s",
        report.candidates,
        report.completed,
        report.failed,
        report.dropped_unsubmitted,
        report.in_flight_at_deadline,
        report.late_completed,
        report.late_failed,
        report.stuck,
        report.planned_budget_s,
        report.flush_elapsed_s,
        "n/a" if total is None else f"{total:.2f}s",
        report.skipped,
        report.quarantined,
    )
    return report
