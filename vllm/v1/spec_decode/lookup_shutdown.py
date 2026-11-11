# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Terminal lookup statistics receipts, bounded by the engine's shared deadline."""

import threading
import time
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.host.shutdown import remaining, wait_until
from vllm.v1.spec_decode.lookup_quarantine import (
    _hold_closing,
    _release_closing,
    quarantine,
)
from vllm.v1.spec_decode.lookup_stats import (
    LookupStatsAccumulator,
    LookupStatsDrainReply,
    LookupStatsReceipt,
    merge_output,
)

logger = init_logger(__name__)


class LookupStatsShutdown:
    def __init__(self, engine: Any):
        self.engine = engine
        self.lock = threading.Lock()
        self.started = False
        self.deadline: float | None = None
        self.receipt: LookupStatsReceipt | None = None
        # Retain the sole response consumer even after the control wait expires.
        self.callback_finished = threading.Event()
        self.prerequisite_finished = threading.Event()
        self.prerequisite_finished.set()
        self.closing_owners: tuple[Any, ...] = ()

    def protect_uniproc(self) -> None:
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        executor = self.engine.model_executor
        if isinstance(executor, UniProcExecutor):
            wrapper = executor.driver_worker
            worker = getattr(wrapper, "worker", None)
            runner = getattr(worker, "model_runner", None)
            exporter = getattr(
                getattr(getattr(runner, "speculator", None), "_native_lookup", None),
                "stats",
                None,
            )
            self.closing_owners = (
                self.engine,
                executor,
                wrapper,
                worker,
                runner,
                exporter,
            )
            _hold_closing(*self.closing_owners)

    def drain_batches(self, deadline: float) -> None:
        engine = self.engine
        queue = getattr(engine, "batch_queue", None)
        while queue:
            future, scheduled, execute_future = queue[-1]
            if not remaining(deadline):
                raise TimeoutError("Lookup batch drain reached the shared deadline")
            self.prerequisite_finished.clear()

            def consume(pending=future):
                try:
                    return pending.result()
                finally:
                    self.prerequisite_finished.set()

            output = wait_until(consume, deadline, "lookup-batch-output")
            if output is None:
                wait_until(execute_future.result, deadline, "lookup-batch-execute")
                raise RuntimeError("Lookup shutdown has an unsuccessful model batch")
            merge_output(engine, output)
            engine.scheduler.update_from_output(scheduled, output)
            queue.pop()

    def finish(self, prerequisite_error: str | None = None) -> LookupStatsReceipt:
        if self.receipt is not None:
            return self.receipt
        engine = self.engine
        state: LookupStatsAccumulator = engine.lookup_stats
        executor = engine.model_executor
        shared = engine.shutdown_deadline()
        assert shared is not None
        if self.deadline is None:
            self.deadline = min(time.monotonic() + 5.0, shared)
        deadline = self.deadline
        reason = (
            "deadline_exhausted" if prerequisite_error is None else prerequisite_error
        )
        complete = False
        isolated = False
        retired = False
        callback_finished = self.prerequisite_finished.is_set()
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        is_uniproc = isinstance(executor, UniProcExecutor)
        if is_uniproc:
            # No callback, RPC, event query or new device access is permitted.
            wrapper = executor.driver_worker
            worker = getattr(wrapper, "worker", None)
            runner = getattr(worker, "model_runner", None)
            exporter = getattr(
                getattr(getattr(runner, "speculator", None), "_native_lookup", None),
                "stats",
                None,
            )
            if (
                prerequisite_error is None
                and remaining(deadline)
                and exporter is not None
            ):
                message, quiescent = exporter.cpu_receipt()
                callback_finished = callback_finished and getattr(
                    exporter, "receipt_callback_finished", quiescent
                )
                before = state.rejected
                state.merge(message)
                complete = (
                    quiescent
                    and state.rejected == before
                    and state.covered == message.producer
                    and state.producer == message.producer
                )
                reason = "complete" if complete else "cpu_record_incomplete"
            else:
                quiescent = False
                callback_finished = False
            if not quiescent:
                isolated = True
                # Retaining owners transitively retains every graph, stream,
                # buffer, external reference and pending output owned by them.
                quarantine(reason, engine, executor, wrapper, worker, runner, exporter)
            else:
                _release_closing(*self.closing_owners)
        elif prerequisite_error is None and remaining(deadline):

            def collect():
                try:
                    return executor.collective_rpc(
                        "drain_lookup_stats",
                        timeout=remaining(deadline),
                        kwargs={
                            "stats_deadline_monotonic": deadline,
                            "expected_generation": state.registration.generation,
                            "expected_owner": state.registration.owner,
                        },
                    )
                finally:
                    self.callback_finished.set()

            try:
                replies = wait_until(collect, deadline, "lookup-final-rpc")
                if not remaining(deadline):
                    raise TimeoutError("lookup final reply exceeded absolute deadline")
                complete = self.accept_replies(replies)
                reason = "complete" if complete else "rank_or_coverage_incomplete"
            except Exception as error:
                reason = f"drain_failed/{type(error).__name__}"
                callback_finished = (
                    callback_finished and self.callback_finished.is_set()
                )
        if not complete and not is_uniproc:
            # Multiproc collective_rpc checks this before every enqueue. Ray is
            # only an interface contract in this version, with no runtime claim.
            executor.is_failed = True
            executor._lookup_stats_retired = True
            retired = True
        reg = state.registration
        self.receipt = LookupStatsReceipt(
            reg.owner,
            reg.generation,
            state.producer,
            state.covered,
            state.sequence,
            state.totals,
            state.timestamp,
            complete,
            reason,
            engine._shutdown_requested_at,
            deadline,
            isolated,
            callback_finished,
            retired,
        )
        logger.info("Lookup statistics final receipt %s", self.receipt)
        return self.receipt

    def accept_replies(self, replies: list[LookupStatsDrainReply]) -> bool:
        engine = self.engine
        state = engine.lookup_stats
        ranks = [reply.rank for reply in replies]
        required = set(range(engine.vllm_config.parallel_config.world_size))
        if set(ranks) != required or len(ranks) != len(required):
            return False
        reg = state.registration
        if any(
            reply.error
            or not reply.quiescent
            or reply.owner != reg.owner
            or reply.generation != reg.generation
            for reply in replies
        ):
            return False
        owner = next(
            reply for reply in replies if reply.rank == state.registration.output_rank
        )
        if owner.message is None:
            return False
        before = state.rejected
        state.merge(owner.message)
        return (
            state.rejected == before
            and state.covered == state.producer
            and state.covered == owner.message.producer
        )
