# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.core.value_arbiter import NUM_HOT_COUNTERS
from vllm.v1.kv_offload.host.page_table import SealTicket
from vllm.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    transfer_spec: TransferSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None


@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed transfer jobs.

    Each worker reports {job_id: 1} for newly completed transfer jobs
    (load or store). aggregate() sums counts across workers within a step.
    The scheduler accumulates across steps and processes
    a transfer completion only when count reaches num_workers.
    """

    completed_jobs: dict[int, int] = field(default_factory=dict)
    # Completed jobs that failed on a worker (counted like completed_jobs).
    # Reported only when load_failure_recompute_enabled; a failed job is also
    # in completed_jobs so the scheduler releases its bookkeeping as usual.
    failed_jobs: dict[int, int] = field(default_factory=dict)
    seal_results: dict[SealTicket, tuple[int, int, int]] = field(default_factory=dict)
    completed_steps: dict[int, int] = field(default_factory=dict)
    # value arbitration hot-side arbitration counters: step -> (rank mask, summed
    # values),
    # one step late (see host_kv/shadow.py).
    value_counters: dict[int, tuple[int, tuple[int, ...]]] = field(default_factory=dict)
    # ReceiveTxn attempts use rank masks across steps and duplicate outputs.
    receive_results: dict[int, tuple[int, int]] = field(default_factory=dict)

    def mark_completed(self, job_id: int) -> None:
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1

    def mark_failed(self, job_id: int) -> None:
        """Record that this worker's part of a completed job failed."""
        self.failed_jobs[job_id] = 1

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v
        failed = dict(self.failed_jobs)
        for job_id, v in other.failed_jobs.items():
            failed[job_id] = failed.get(job_id, 0) + v

        seals = dict(self.seal_results)
        for ticket, masks in other.seal_results.items():
            old = seals.get(ticket, (0, 0, 0))
            seals[ticket] = (old[0] | masks[0], old[1] | masks[1], old[2] | masks[2])
        steps = dict(self.completed_steps)
        for step, ranks in other.completed_steps.items():
            steps[step] = steps.get(step, 0) | ranks
        counters = dict(self.value_counters)
        for step, (ranks, values) in other.value_counters.items():
            old_ranks, old_values = counters.get(step, (0, (0,) * len(values)))
            invalid = (
                bool(old_ranks & ranks)
                or len(values) != NUM_HOT_COUNTERS
                or len(old_values) != NUM_HOT_COUNTERS
                or any(v < 0 for v in (*old_values, *values))
            )
            # An empty tuple is an invalid report for this entire step. It
            # remains invalid through later aggregation and the arbiter holds.
            counters[step] = (
                old_ranks | ranks,
                () if invalid else tuple(a + b for a, b in zip(old_values, values)),
            )
        receives = dict(self.receive_results)
        for attempt, (completed, failed_ranks) in other.receive_results.items():
            prior = receives.get(attempt, (0, 0))
            receives[attempt] = (prior[0] | completed, prior[1] | failed_ranks)
        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            failed_jobs=failed,
            seal_results=seals,
            completed_steps=steps,
            value_counters=counters,
            receive_results=receives,
        )
