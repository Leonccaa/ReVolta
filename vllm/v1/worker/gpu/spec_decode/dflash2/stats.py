# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-slot lookup export. Only the worker owner touches CUDA and pinned data."""

import threading
import time

import torch

from vllm.v1.spec_decode.lookup_stats import (
    VECTOR_SIZE,
    LookupStatsMessage,
    LookupStatsRegistration,
    LookupStatsSnapshot,
)


class LookupStatsExporter:
    def __init__(self, registration: LookupStatsRegistration, device: torch.device):
        self.registration = registration
        self.counters = torch.zeros(VECTOR_SIZE, dtype=torch.uint64, device=device)
        self.snapshot = torch.empty_like(self.counters)
        self.pinned = torch.empty(
            VECTOR_SIZE, dtype=torch.uint64, device="cpu", pin_memory=True
        )
        self.snapshot_ready = torch.cuda.Event()
        self.export_done = torch.cuda.Event()
        self.export_stream = torch.cuda.Stream(device)
        self.device = device
        self.lock = threading.Lock()
        self.producer = 0
        self.sequence = 0
        self.covered = 0
        self.busy = False
        self.frozen = False
        self.quiescent = False
        self.receipt_callback_finished = False
        self.activity = 0
        self.completed_activity = -1
        self.snapshot_activity = -1
        self.last_snapshot: LookupStatsSnapshot | None = None
        self.next_export = time.monotonic() + 1.0
        self.capture_time = 0.0
        self.snapshot_covered = 0

    def begin_activity(self) -> None:
        if self.frozen:
            raise RuntimeError("Lookup statistics producer is retired")
        self.activity += 1
        self.quiescent = False

    def submitted(self) -> None:
        if self.frozen:
            raise RuntimeError("Lookup statistics producer is retired")
        self.producer += 1
        self.quiescent = False

    def service(
        self,
        *,
        force: bool = False,
        deadline: float | None = None,
        _final_checkpoint: bool = False,
    ) -> LookupStatsMessage:
        payload = None
        if self.lock.acquire(blocking=False):
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    return self.message()
                if self.busy and self.export_done.query():
                    if deadline is not None and time.monotonic() >= deadline:
                        return self.message()
                    # Detach all CPU values before the pinned slot can be reused.
                    totals = tuple(int(value) for value in self.pinned.tolist())
                    if deadline is not None and time.monotonic() >= deadline:
                        return self.message()
                    self.sequence += 1
                    self.covered = self.snapshot_covered
                    payload = LookupStatsSnapshot(
                        self.sequence, self.covered, totals, self.capture_time
                    )
                    self.last_snapshot = payload
                    self.busy = False
                    self.completed_activity = self.snapshot_activity
                    self.quiescent = (
                        self.covered == self.producer
                        and self.completed_activity == self.activity
                    )
                now = time.monotonic()
                if (
                    not self.busy
                    and (
                        self.covered < self.producer
                        or (_final_checkpoint and not self.quiescent)
                    )
                    and (force or now >= self.next_export)
                    and (deadline is None or now < deadline)
                ):
                    self.snapshot.copy_(self.counters, non_blocking=True)
                    self.snapshot_ready.record(torch.cuda.current_stream(self.device))
                    self.snapshot_covered = self.producer
                    self.snapshot_activity = self.activity
                    self.capture_time = time.time()
                    self.export_stream.wait_event(self.snapshot_ready)
                    with torch.cuda.stream(self.export_stream):
                        self.pinned.copy_(self.snapshot, non_blocking=True)
                        self.export_done.record(self.export_stream)
                    self.busy = True
                    self.quiescent = False
                    self.next_export = now + 1.0
            finally:
                self.lock.release()
        return self.message(payload)

    def message(self, payload: LookupStatsSnapshot | None = None) -> LookupStatsMessage:
        reg = self.registration
        return LookupStatsMessage(reg.owner, reg.generation, self.producer, payload)

    def cpu_receipt(self) -> tuple[LookupStatsMessage, bool]:
        # UniProc shutdown must not query an event or wait for this lock.
        if not self.lock.acquire(blocking=False):
            self.receipt_callback_finished = False
            return self.message(), False
        try:
            self.receipt_callback_finished = True
            self.frozen = True
            return self.message(self.last_snapshot), self.quiescent and not self.busy
        finally:
            self.lock.release()

    def drain(self, deadline: float) -> tuple[LookupStatsMessage, bool]:
        self.frozen = True
        while time.monotonic() < deadline:
            # Only final Multiproc drain may obtain a new completion proof
            # without a new producer. Normal service and UniProc CPU receipts
            # retain their existing behavior and never initiate this checkpoint.
            self.service(force=True, deadline=deadline, _final_checkpoint=True)
            if (
                time.monotonic() < deadline
                and self.quiescent
                and not self.busy
                and self.covered == self.producer
            ):
                return self.message(self.last_snapshot), True
            time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))
        return self.message(self.last_snapshot), False


def create_exporter(config, device: torch.device) -> LookupStatsExporter | None:
    registration = getattr(config, "_lookup_stats_registration", None)
    rank = getattr(config, "_lookup_stats_rank", None)
    if registration is None or rank != registration.output_rank:
        return None
    return LookupStatsExporter(registration, device)


def drain_worker(
    worker,
    *,
    stats_deadline_monotonic: float,
    expected_generation: str,
    expected_owner: str,
):
    from vllm.v1.spec_decode.lookup_stats import LookupStatsDrainReply

    reg = getattr(worker.vllm_config, "_lookup_stats_registration", None)

    def reply(message=None, quiescent=False, error=None):
        return LookupStatsDrainReply(
            worker.rank,
            message,
            quiescent,
            error,
            "" if reg is None else reg.owner,
            "" if reg is None else reg.generation,
        )

    if (
        time.monotonic() >= stats_deadline_monotonic
        or reg is None
        or reg.generation != expected_generation
        or reg.owner != expected_owner
    ):
        return reply(error="deadline_or_identity")
    runner = worker.model_runner
    exporter = getattr(
        getattr(getattr(runner, "speculator", None), "_native_lookup", None),
        "stats",
        None,
    )
    if worker.rank != reg.output_rank:
        return reply(quiescent=True)
    if exporter is None:
        return reply(error="missing_exporter")
    try:
        message, quiescent = exporter.drain(stats_deadline_monotonic)
        if time.monotonic() >= stats_deadline_monotonic:
            return reply(message, error="deadline")
        return reply(message, quiescent)
    except Exception as error:
        return reply(error=repr(error))


def begin_worker_activity(worker) -> None:
    from vllm.v1.spec_decode.lookup_quarantine import isolated

    if isolated(worker):
        raise RuntimeError("Lookup statistics worker is shutting down")
    native = getattr(
        getattr(worker.model_runner, "speculator", None), "_native_lookup", None
    )
    exporter = getattr(native, "stats", None)
    if exporter is not None:
        exporter.begin_activity()


def attach_empty_output(runner, output):
    from dataclasses import replace

    native = getattr(getattr(runner, "speculator", None), "_native_lookup", None)
    exporter = getattr(native, "stats", None)
    if exporter is None or output is None:
        return output
    # no_forward may return the shared EMPTY_MODEL_RUNNER_OUTPUT singleton.
    return replace(output, lookup_stats=exporter.service())
