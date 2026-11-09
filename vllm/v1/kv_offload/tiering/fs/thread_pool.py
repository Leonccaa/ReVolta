# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_lock",
        "deadline",
        "_published",
        "_unsafe",
    )

    def __init__(
        self, job_id: JobId, n_tasks: int, deadline: float | None = None
    ) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._lock = threading.Lock()
        # Monotonic time after which the job is reported failed (None: never).
        self.deadline = deadline
        # The job's result was published (by its last task or by the deadline).
        self._published = False
        # Tasks inside a section that must finish before the job may expire
        # (see begin_unsafe).
        self._unsafe = 0

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(self, success: bool) -> tuple[bool, bool]:
        """Returns (publish now, success). The first publisher wins: a job
        already reported failed at its deadline is not reported again when
        its last, late task finishes."""
        with self._lock:
            self._completed += 1
            if not success:
                self._success = False
            if self._completed == self._n_tasks and not self._published:
                self._published = True
                return True, self._success
            return False, self._success

    def expire(self, now: float) -> bool:
        """Mark the job failed if its deadline passed, it is unpublished and no
        task is inside an unsafe section; return whether the caller must
        publish the failure. Publishing releases the job's source buffers."""
        with self._lock:
            if (
                self._published
                or self.deadline is None
                or now < self.deadline
                or self._unsafe
            ):
                return False
            self._published = True
            self._success = False
            return True

    @property
    def expired(self) -> bool:
        with self._lock:
            return self._published

    @property
    def finished(self) -> bool:
        """All physical tasks have returned, including late expired tasks."""
        with self._lock:
            return self._completed == self._n_tasks

    def begin_unsafe(self) -> bool:
        """Enter a section during which the job must not expire (e.g. while a
        task snapshots its source buffer). Returns False if the job already
        expired: its buffers may be reused, so the task must not touch them."""
        with self._lock:
            if self._published:
                return False
            self._unsafe += 1
            return True

    def end_unsafe(self) -> None:
        with self._lock:
            self._unsafe -= 1


_current = threading.local()


def current_job() -> "JobState | None":
    """The JobState of the task running on this pool thread, if any."""
    return getattr(_current, "state", None)


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        thread_name_prefix: str = "fs_secondary_tier",
        store_timeout_s: float | None = None,
        drain_timeout_s: float = 0.0,
    ) -> None:
        """
        Args:
            store_timeout_s: report a store job failed this many seconds after
                it was enqueued if it has not finished (None: never). Its
                threads may still be writing; only use it when a late write
                cannot be mistaken for a complete page (e.g. pages carry a
                checksum computed before the write).
            drain_timeout_s: at shutdown, keep running queued store tasks for
                up to this many seconds before dropping the rest (0: drop
                queued tasks immediately, the historical behaviour).
        """
        self._store_timeout_s = store_timeout_s
        self._drain_timeout_s = drain_timeout_s
        self._timed: dict[JobId, JobState] = {}
        self.num_timed_out = 0
        self._timed_lock = threading.Lock()
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._active_stores = 0
        self._store_blocks: dict[JobId, int] = {}
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool]] = deque()

        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""
        state = JobState(job_id, n_tasks)
        with self._condition:
            for fn in tasks:
                self._load_q.append((fn, state))
            self._condition.notify(n_tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
        n_blocks: int = 0,
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        deadline = (
            None
            if self._store_timeout_s is None
            else time.monotonic() + self._store_timeout_s
        )
        state = JobState(job_id, n_tasks, deadline)
        if deadline is not None:
            with self._timed_lock:
                self._timed[job_id] = state
        with self._condition:
            if n_blocks:
                self._store_blocks[job_id] = n_blocks
            for fn in tasks:
                self._store_q.append((fn, state))
            self._condition.notify(n_tasks)

    def get_finished(self) -> list[tuple[JobId, bool]]:
        if self._timed:
            self._expire_overdue()
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def _expire_overdue(self) -> None:
        now = time.monotonic()
        with self._timed_lock:
            overdue = [
                (job_id, state)
                for job_id, state in self._timed.items()
                if state.expire(now)
            ]
            for job_id, _ in overdue:
                del self._timed[job_id]
        for job_id, _ in overdue:
            logger.warning("Store job %s exceeded its I/O deadline", job_id)
            self.num_timed_out += 1
            self._finished_q.append((job_id, False))
        if overdue:
            with self._condition:
                self._condition.notify_all()

    @property
    def pending_store_blocks(self) -> int:
        """Physical queued/active pages, independent of published completion."""
        with self._condition:
            return sum(self._store_blocks.values())

    def retry_allowed(self) -> bool:
        state = current_job()
        return not self._stop and (
            state is None
            or (
                not state.expired
                and (state.deadline is None or time.monotonic() < state.deadline)
            )
        )

    def wait_for_retry(self, delay: float) -> bool:
        """Wait until retry time, expiry or stop, within the job's budget."""
        state = current_job()
        delay = min(delay, 1.0)
        if state is not None and state.deadline is not None:
            delay = min(delay, max(0.0, state.deadline - time.monotonic()))
        with self._condition:
            self._condition.wait_for(lambda: not self.retry_allowed(), timeout=delay)
            return self.retry_allowed()

    @property
    def stopping(self) -> bool:
        """Grouped tasks must stop between pages after the drain deadline."""
        with self._condition:
            return self._stop

    def shutdown(self, wait: bool = True, deadline: float | None = None) -> None:
        if self._drain_timeout_s > 0:
            # Let queued stores run for a bounded time; loads are dropped
            # (nothing will consume their results).
            drain_deadline = time.monotonic() + self._drain_timeout_s
            if deadline is not None:
                drain_deadline = min(drain_deadline, deadline)
            with self._condition:
                self._load_q.clear()
            while time.monotonic() < drain_deadline:
                with self._condition:
                    if not self._store_q and not self._active_stores:
                        break
                time.sleep(0.01)
        with self._condition:
            self._stop = True
            dropped = len(self._store_q)
            self._load_q.clear()
            for _, state in self._store_q:
                if state.job_id in self._store_blocks:
                    state.task_done(False)
                    if state.finished:
                        self._store_blocks.pop(state.job_id, None)
            self._store_q.clear()
            self._condition.notify_all()
        if dropped and self._drain_timeout_s > 0:
            logger.warning("Dropped %d queued store tasks at shutdown", dropped)
        if wait:
            for t in self._threads:
                timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                t.join(timeout=timeout)
            if any(t.is_alive() for t in self._threads):
                raise TimeoutError(
                    "Tiering drain deadline reached; IO still owns backing"
                )

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                if self._stop:
                    return
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                selected = primary if primary else secondary
                is_store = selected is self._store_q
                task, state = selected.popleft()
                if is_store:
                    self._active_stores += 1
            if state.deadline is not None and state.expired:
                # Reported failed at its deadline: its buffers may be reused.
                state.task_done(False)
                if is_store:
                    with self._condition:
                        self._active_stores -= 1
                        if state.finished:
                            self._store_blocks.pop(state.job_id, None)
                continue
            _current.state = state
            try:
                task()
                job_finished, success = state.task_done(True)
            except Exception as exc:
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success = state.task_done(False)
            finally:
                _current.state = None
                if is_store:
                    with self._condition:
                        self._active_stores -= 1
                        if state.finished:
                            self._store_blocks.pop(state.job_id, None)
                        self._condition.notify_all()

            if job_finished:
                if state.deadline is not None:
                    with self._timed_lock:
                        self._timed.pop(state.job_id, None)
                self._finished_q.append((state.job_id, success))
