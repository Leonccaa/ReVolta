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

import os
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from queue import Empty, SimpleQueue
from typing import Any

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.host.page_table import (
    HostPage,
    HostPageTable,
    PageHandle,
    PageKind,
    PageState,
)
from vllm.v1.kv_offload.host.ssd import (
    HostKVLayout,
    _HostTier,
    _WriteRateLimiter,
    host_page_tier_key,
)
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
HOST_PAGE_ABI = "host-page-v2.2:row=workers*[main_kv|logical_index]:key=sha256"
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
    guaranteed_completed_bytes: int = 0
    best_effort_completed_bytes: int = 0
    unfinished_bytes: int = 0


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
        return FlushPlan(0.0, deadline, t0, "shutdown_timeout=0 skips final flush")
    if deadline is None:
        return FlushPlan(configured_s, None, t0, "no shutdown deadline")
    budget = min(configured_s, deadline - now)
    if budget <= 0:
        return FlushPlan(0.0, deadline, t0, "shutdown window exhausted")
    return FlushPlan(budget, deadline, t0, "ok")


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
    keys: set[OffloadKey] = field(default_factory=set)


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
        self._expired_rounds: set[str] = set()
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
        self.unit_store: Any = None
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
        self._component_tier: _HostTier | None = None
        self._component_salt = b""
        self._losses: SimpleQueue[tuple[str, tuple[OffloadKey, PageHandle] | None]] = (
            SimpleQueue()
        )
        self._file_evidence: dict[str, tuple[OffloadKey, PageHandle]] = {}
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
            external_eviction=table.value_policy is not None,
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

    def enable_components(self, layout: dict) -> None:
        """unified host budgeting versioned component namespace; MAIN serialization
        stays unchanged.

        One controller, queue, IO limit, row export and root budget. The two
        file mappers isolate the new component ABI from legacy MAIN files.
        unit SSD transactions will replace the shared cache's file-level eviction
        authority.
        """
        if self._component_tier is not None:
            return
        options = self.options
        if self.enabled:
            self._tier._cache.external_eviction = True
        spec = _HostSpec(
            self._spec.vllm_config,
            self._spec.kv_cache_config,
            {
                **self._spec.persistent_layout,
                "unified_components": {"abi": "state-draft-row-v1", "layout": layout},
            },
        )
        self._component_tier = _HostTier(
            self._limiter,
            offloading_spec=spec,
            primary_kv_view=self._tier._primary_kv_view,
            tier_type="revolta_fs",
            root_dir=options["root_dir"],
            max_size_bytes=options["max_size_bytes"],
            min_free_bytes=options.get("min_free_bytes"),
            max_files=options["max_files"],
            content_manifest_path=options.get("content_manifest_path"),
            content_verify=options.get("content_verify", "stat"),
            n_read_threads=1,
            n_write_threads=self._window,
            store_timeout_s=0,
            drain_timeout_s=0,
            max_inflight_store_blocks=self._max_pending + self._window,
            write_delay_s=0,
            external_eviction=True,
        )
        if self.enabled:
            self._tier._cache.loss_listeners.append(self._file_lost)

    def _file_lost(self, path: str) -> None:
        # Called under the FS cache lock. Snapshot the immutable proof before
        # a later same-path write can publish a different physical epoch.
        # Graph mutations still happen only on the scheduler thread.
        self._losses.put((path, self._file_evidence.get(path)))

    def _tier_for_key(self, key: OffloadKey):
        if (
            self._component_tier is not None
            and self.table.kind_for_key(key) != PageKind.MAIN
        ):
            return self._component_tier
        return self._tier

    def _poll_losses(self) -> None:
        while True:
            try:
                path, evidence = self._losses.get_nowait()
            except Empty:
                break
            if evidence is None or self._file_evidence.get(path) != evidence:
                continue
            del self._file_evidence[path]
            key, incarnation = evidence
            if self.unit_store is not None:
                self.unit_store.invalidate_key(key)
            policy = self.table.value_policy
            if policy is not None:
                policy.ssd_lost(key, incarnation)
            if self.unit_store is not None:
                self.unit_store.file_lost(key)
            page = self._page(incarnation)
            if page is not None and page.key == key:
                self.table.unpersist(incarnation)
            self.stats_counter["ssd_evidence_lost"] += 1

    # --- state -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return not self._closed and not self._tier.disabled

    def stats(self) -> dict[str, int]:
        out = {f"delayed_{k}": v for k, v in self._queue.summary().items()}
        out.update(self.stats_counter)
        out["jobs"] = len(self._jobs)
        out["deferred"] = len(self._deferred)
        if self.unit_store is not None:
            out.update(self.unit_store.snapshot())
        return out

    def has_pending(self) -> bool:
        return bool(
            self._jobs
            or self._queue.depth
            or self._deferred
            or self._rounds
            or self.unit_store is not None
            and (
                self.unit_store.writes
                or self.unit_store.prefetches
                or self.unit_store.due
            )
        )

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
        """Fixed-length SSD key for a host page key.

        A host page's block hash concatenates the hashes of every hash block
        it covers (98 x 32 bytes for a 1568-token page), far longer than a
        file name may be. The SSD key is a SHA-256 of the full key instead;
        the file's xattr binds that digest. Epoch 0 is stable across
        restarts. A reset gets a process-unique salt, so pre- and post-reset
        files can never hit each other."""
        if self.unit_store is not None:
            restored = self.unit_store.tier_key(key)
            if restored is not None:
                return restored
        epoch = self.table.ssd_epoch
        salt = b"" if epoch == 0 else self._salts.setdefault(epoch, os.urandom(16))
        if (
            self._component_tier is not None
            and self.table.kind_for_key(key) != PageKind.MAIN
        ):
            salt += self._component_salt
        return host_page_tier_key(key, salt)

    def _store_jobs(self) -> int:
        return (
            len(self._jobs)
            - self._inflight_loads
            + (len(self.unit_store.manifest_jobs) if self.unit_store is not None else 0)
        )

    # --- jobs ------------------------------------------------------------

    def _new_job(self, page: HostPage, loading: bool, front: bool = False) -> _Job:
        assert page.key is not None
        self._next_job -= 1
        job = _Job(self._next_job, page.handle, self._tier_key(page.key), loading)
        if self._report is not None and not loading:
            job.flush = True
            self._report.candidates += 1
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
        if self.unit_store is not None and not self.unit_store.prepare_job(job):
            self._finish(job, False)
            return
        meta = JobMetadata(
            job_id=job.job_id,
            keys=[job.tier_key],
            block_ids=np.array([job.handle.page_id], dtype=np.int64),
            is_promotion=job.loading,
            req_context=self._ctx,
        )
        key = self.table.get(job.handle).key
        assert key is not None
        try:
            if job.loading:
                self._tier_for_key(key).submit_load(meta)
                self._inflight_loads += 1
            else:
                self._tier_for_key(key).submit_store(meta)
                self._inflight_stores += 1
        except Exception:
            logger.exception("Host KV SSD job %d submission failed", job.job_id)
            self.stats_counter["submit_failures"] += 1
            self._finish(job, False)
            return
        job.submitted = True

    def _pump(self) -> None:
        limit = self._window
        if self.unit_store is not None:
            limit = min(limit, self.unit_store.credit.q_eff // self.page_bytes)
            if limit == 0:
                while self._submit:
                    self._finish(self._jobs[self._submit.popleft()], False)
                return
        while self._submit and self._inflight_stores < limit:
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
        file_present = success
        if success and self._component_tier is not None:
            page = self._page(job.handle)
            if (
                page is not None
                and page.valid
                and page.key is not None
                and job.tier_key == self._tier_key(page.key)
            ):
                tier = self._tier_for_key(page.key)
                path = tier.file_mapper.get_file_name(job.tier_key)
                # A later file job may have evicted this completed store
                # before the scheduler polled its receipt. A notification
                # without installed evidence must not be forgotten.
                key = page.key

                def remember_file() -> None:
                    self._file_evidence[path] = (key, job.handle)

                file_present = not tier.disabled and tier._cache.lookup(
                    path, remember_file
                )
        if self.unit_store is not None:
            page = self._page(job.handle)
            if page is not None and page.key is not None:
                self.unit_store.invalidate_key(page.key)
        if job.loading:
            ok = self.table.complete_load(job.handle, success)
            if ok and not file_present:
                self.table.unpersist(job.handle)
                if self.table.value_policy is not None:
                    page = self.table.get(job.handle)
                    assert page.key is not None
                    self.table.value_policy.ssd_lost(page.key, job.handle)
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
        elif success:
            # Charge completed stores only (delayed_written and the shared
            # 24 h volume). A store of a page invalidated meanwhile still
            # wrote its file: charged, not recorded as persisted.
            self._queue.account([job.tier_key])
            if file_present and self.table.mark_persisted(job.handle):
                self.stats_counter["pages_written"] += 1
            else:
                self.stats_counter["store_failures"] += 1
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
        if self.unit_store is not None:
            self.unit_store.job_done(job, success)

    def _poll(self) -> None:
        self._poll_losses()
        tiers = [self._tier]
        if self._component_tier is not None:
            tiers.append(self._component_tier)
        for tier in tiers:
            for result in tier.get_finished_jobs():
                job = self._jobs.get(result.job_id)
                if job is not None:
                    self._finish(job, result.success)
                elif self.unit_store is not None:
                    self.unit_store.manifest_done(result.job_id, result.success)
        self._poll_losses()

    # --- PersistenceHooks ------------------------------------------------

    def on_published(self, handle: PageHandle) -> None:
        if self.unit_store is not None:
            # A PRIVATE frame may already have appeared in a snapshot. Its
            # writable wait starts at publication, not private allocation.
            self.unit_store.dirty_since[handle] = self.clock()
        if self.enabled and not self._flushing:
            self._queue.enqueue([handle], self._ctx)

    def try_writeback(self, page: HostPage, headroom_before: int) -> bool:
        """Before eviction: one immediate write (still rate limited) when the
        queue has room and net headroom pays for the page. Never blocks."""
        if self.unit_store is not None and page.key is not None:
            component = self.unit_store.units.components.get(page.key)
            variants = (
                []
                if component is None
                else [
                    self.unit_store.units.units[uid]
                    for uid in component.refs
                    if uid.ahead is not None
                ]
            )
            if variants:
                return any(
                    self.unit_store.start_write(u, headroom_before)
                    for u in sorted(variants, key=lambda u: u.last_use)
                )
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
        self._file_evidence.clear()
        for job_id in list(self._submit):
            self._finish(self._jobs[job_id], False)
        self._submit.clear()
        for req_id in set(self._rounds) | self._expired_rounds:
            self.cancel(req_id)

    def reset_components(self, handles: list[PageHandle]) -> None:
        """Connector-only reset cannot invalidate MAIN's active bindings/ABI."""
        self._component_salt = os.urandom(16)
        self._queue.drop(handles, "evicted")
        self.stats_counter["component_resets"] += 1
        self.stats_counter["component_reset_pages"] += len(handles)
        for handle in handles:
            self._deferred.pop(handle, None)
        for path, (key, _) in tuple(self._file_evidence.items()):
            if self.table.kind_for_key(key) != PageKind.MAIN:
                del self._file_evidence[path]
        for job_id in tuple(self._submit):
            page = self._page(self._jobs[job_id].handle)
            if page is not None and page.kind != PageKind.MAIN:
                self._submit.remove(job_id)
                self._finish(self._jobs[job_id], False)

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
        if self.unit_store is not None:
            self.unit_store.step()
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
            if (
                self.unit_store is not None
                and page.key in self.unit_store.units.components
            ):
                component = self.unit_store.units.components[page.key]
                variants = [
                    self.unit_store.units.units[uid]
                    for uid in component.refs
                    if uid.ahead is not None
                ]
                if variants:
                    for unit in variants:
                        self.unit_store.start_write(unit)
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

    def prefetch(
        self, req_id: str, keys: Iterable[OffloadKey], *, extend: bool = False
    ) -> list[OffloadKey]:
        """One prefetch round per request per T_pf, in chain order.

        The connector lookup calls this for the first keys missing from the
        page table and waits while ``waiting(req_id)``. Verified pages are
        READY and held for the request; its lookup pins them as NEW_LOAD
        sources, then calls ``cancel`` to drop the round and its holds. No
        hit is reported before verification.
        """
        if not self.enabled or self._flushing:
            return []
        keys = tuple(keys)
        now = self.clock()
        self._expire_prefetch(now)
        if req_id in self._expired_rounds:
            return []
        if self.unit_store is not None:
            result = self.unit_store.prefetch(req_id, keys)
            if result is not None:
                return result
        now = self.clock()
        self._expire_prefetch(now)
        if req_id in self._expired_rounds:
            return []
        if req_id in self._rounds and not extend:
            self.stats_counter["prefetch_round_limited"] += 1
            return []
        round_ = self._rounds.get(req_id, _Round(now))
        issued: list[OffloadKey] = []
        headroom: int | None = None
        per_req = self.options["prefetch_max_per_request"]
        for key in keys:
            if (
                self.unit_store is not None
                and self.table.kind_for_key(key) != PageKind.MAIN
                and key not in self.unit_store.files
            ):
                break
            if key in round_.keys:
                continue
            handle = self.table.lookup(key)
            if handle is not None:
                page = self.table.get(handle)
                if (
                    page.state == PageState.LOADING
                    and page.handle.page_id in self._page_jobs
                    and len(round_.keys) < per_req
                ):
                    round_.loading.add(handle)
                    round_.keys.add(key)
                    self._waiters.setdefault(handle, set()).add(req_id)
                continue
            if len(round_.keys) >= per_req:
                self.stats_counter["prefetch_request_quota"] += 1
                break
            if self._inflight_loads >= self.options["prefetch_max_inflight"]:
                self.stats_counter["prefetch_global_quota"] += 1
                break
            try:
                if self._tier_for_key(key).lookup(self._tier_key(key)) is not True:
                    break
            except OSError:
                self.stats_counter["prefetch_lookup_failures"] += 1
                break
            if headroom is None:
                headroom = self.table.net_headroom()
            if headroom < 1 and self.table.reservation_ledger is None:
                self.stats_counter["prefetch_no_headroom"] += 1
                break
            ledger = self.table.reservation_ledger
            temporary = None
            if ledger is None:
                handle = self.table.reserve(loading_key=key, writeback=False)
            else:
                temporary = ledger.reserve_frames(
                    1, "prefetch", loading_key=key, writeback=False
                )
                handle = temporary[0] if temporary else None
            if handle is None:
                break
            headroom -= 1
            self._new_job(self.table.get(handle), loading=True)
            if temporary is not None:
                assert ledger is not None
                ledger.release_temporary(temporary)
            round_.loading.add(handle)
            round_.keys.add(key)
            self._waiters.setdefault(handle, set()).add(req_id)
            issued.append(key)
        if round_.loading:
            self._rounds[req_id] = round_
            self.stats_counter["prefetch_issued"] += len(issued)
        return issued

    def poll(self) -> None:
        """Scheduler thread, before a lookup: publish finished reads and
        expire prefetch rounds older than T_pf."""
        if self._closed or self._flushing:
            return
        self._poll()
        self._expire_prefetch(self.clock())
        if self.unit_store is not None:
            self.unit_store.advance_prefetches()
            self.unit_store.advance_writes()

    def waiting(self, req_id: str) -> bool:
        """The request's prefetch round still has reads in flight. False
        once every read has finished, failed, or the round expired."""
        round_ = self._rounds.get(req_id)
        return (round_ is not None and bool(round_.loading)) or (
            self.unit_store is not None and self.unit_store.waiting(req_id)
        )

    def holds(self, req_id: str) -> list[PageHandle]:
        round_ = self._rounds.get(req_id)
        handles = [] if round_ is None else [h for h, _ in round_.holds]
        if self.unit_store is not None:
            for uid in self.unit_store.requests.get(req_id, ()):
                txn = self.unit_store.prefetches.get(uid)
                if txn is not None:
                    handles.extend(
                        txn.handles[k]
                        for k in txn.sources
                        if self.table.is_ready(txn.handles[k])
                    )
        return handles

    def release_holds(self, req_id: str) -> None:
        round_ = self._rounds.get(req_id)
        if round_ is None:
            return
        for handle, lease in round_.holds:
            self.table.release(handle, lease)
        round_.holds.clear()

    def cancel(self, req_id: str) -> None:
        """Withdraw a request's waiting; physical reads finish on their own."""
        self._expired_rounds.discard(req_id)
        if self.unit_store is not None:
            self.unit_store.cancel(req_id)
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
                if self._component_tier is not None:
                    # A multi-call unified lookup keeps its original deadline.
                    # Its final answer/cancellation opens the next episode.
                    self._expired_rounds.add(req_id)

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
        if self.unit_store is not None:
            return self.unit_store.final_flush(budget_s, deadline, io_grace_s)
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
        if deadline is not None:
            flush_deadline = min(flush_deadline, deadline)
        grace_deadline = flush_deadline + grace
        if deadline is not None:
            grace_deadline = min(grace_deadline, deadline)
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

    def close(
        self, io_grace_s: float | None = None, *, deadline: float | None = None
    ) -> None:
        """Shutdown step 5: join started IO, or quarantine if it never ends."""
        if self._closed:
            return
        self._flushing = True
        if self.unit_store is not None:
            self.unit_store.begin_close()
        for job_id in list(self._submit):
            self._finish(self._jobs[job_id], False)
        self._submit.clear()
        for req_id in list(self._rounds):
            self.cancel(req_id)
        grace = (
            self.options["shutdown_io_grace_s"] if io_grace_s is None else io_grace_s
        )
        end = self.clock() + grace
        if deadline is not None:
            end = min(end, deadline)
        while (
            self._jobs or self.unit_store is not None and self.unit_store.manifest_jobs
        ) and self.clock() < end:
            self._poll()
            time.sleep(0.002)
        self._poll()
        if self.unit_store is not None:
            self.unit_store.retire()
        self._closed = True
        if self._queue.volume is not None:
            if deadline is None:
                self._queue.volume.persist()
            else:
                from vllm.v1.kv_offload.host.shutdown import wait_until

                wait_until(self._queue.volume.persist, deadline, "host-ssd-volume")
        if self.table.persistence is self:
            self.table.persistence = None
        if self._jobs or self.unit_store is not None and self.unit_store.manifest_jobs:
            self.quarantined = True
            # Stop the pool without joining: a hung writer keeps its row and
            # the region stays mapped (transfer regression). Queued tasks are dropped.
            for tier in (self._tier, self._component_tier):
                pool = getattr(tier, "_pool", None)
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
        if self._component_tier is not None:
            cache = self._tier._cache
            if self._file_lost in cache.loss_listeners:
                cache.loss_listeners.remove(self._file_lost)
            if deadline is not None:
                self._component_tier.set_shutdown_deadline(deadline)
            self._component_tier.shutdown()
            self._component_tier._primary_kv_view = None  # type: ignore[assignment]
        if deadline is None:
            self._tier.shutdown()
        else:
            from vllm.v1.kv_offload.host.shutdown import wait_until

            self._tier.set_shutdown_deadline(deadline)
            wait_until(self._tier.shutdown, deadline, "host-ssd-close")
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
    if ssd.unit_store is not None:
        logger.info(
            "Unified SSD flush guaranteed_completed_bytes=%d "
            "best_effort_written_bytes=%d unfinished_bytes=%d",
            report.guaranteed_completed_bytes,
            report.best_effort_completed_bytes,
            report.unfinished_bytes,
        )
    return report
