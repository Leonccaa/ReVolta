# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cumulative native-lookup transport and independently merged watermarks."""

import math
from typing import Any

import msgspec

BUCKETS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128)
VECTOR_SIZE = 4 + len(BUCKETS) + 1 + 1
ZERO = (0,) * VECTOR_SIZE


class LookupStatsRegistration(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    engine_index: int
    owner: str
    generation: str
    output_rank: int


class LookupStatsSnapshot(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    sequence: int
    covered: int
    totals: tuple[int, ...]
    timestamp: float


class LookupStatsMessage(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    owner: str
    generation: str
    producer: int
    snapshot: LookupStatsSnapshot | None = None


class LookupStatsDrainReply(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    rank: int
    message: LookupStatsMessage | None
    quiescent: bool
    error: str | None = None
    owner: str = ""
    generation: str = ""


class LookupStatsReceipt(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    owner: str
    generation: str
    producer: int
    covered: int
    sequence: int
    totals: tuple[int, ...]
    timestamp: float
    complete: bool
    reason: str
    requested_at: float
    deadline: float
    isolated: bool = False
    callback_finished: bool = True
    transport_retired: bool = False


def enabled(config: Any, log_stats: bool) -> bool:
    spec = getattr(config, "speculative_config", None)
    resolved = getattr(spec, "resolved_dflash2_lookup", None)
    return bool(
        log_stats
        and resolved is not None
        and resolved.mode == "gpu_head"
        and resolved.collect_stats
    )


class LookupStatsAccumulator:
    """An instance must be registered through startup, never through a message."""

    def __init__(self, registration: LookupStatsRegistration):
        self.registration = registration
        self.producer = 0
        self.sequence = 0
        self.covered = 0
        self.totals = ZERO
        self.timestamp = 0.0
        self.rejected = 0

    @property
    def pending(self) -> int:
        return int(self.producer > self.covered)

    def merge(self, message: LookupStatsMessage) -> tuple[int, ...]:
        reg = self.registration
        if (
            message.owner != reg.owner
            or message.generation != reg.generation
            or type(message.producer) is not int
            or message.producer < 0
        ):
            self.rejected += 1
            return ZERO
        # A missing or duplicate snapshot still carries new producer evidence.
        self.producer = max(self.producer, message.producer)
        snapshot = message.snapshot
        if snapshot is None:
            return ZERO
        if type(snapshot.sequence) is not int or snapshot.sequence < 1:
            self.rejected += 1
            return ZERO
        if snapshot.sequence <= self.sequence:
            return ZERO
        values = snapshot.totals
        if (
            type(snapshot.sequence) is not int
            or snapshot.sequence <= 0
            or type(snapshot.covered) is not int
            or not self.covered <= snapshot.covered <= self.producer
            or len(values) != VECTOR_SIZE
            or any(
                type(value) is not int or value < old
                for value, old in zip(values, self.totals)
            )
            or not values[3] <= values[2] <= values[1] <= values[0]
            or sum(values[4:-1]) != values[1]
            or values[-1] < values[1]
            or not math.isfinite(snapshot.timestamp)
            or snapshot.timestamp < self.timestamp
        ):
            self.rejected += 1
            return ZERO
        delta = tuple(value - old for value, old in zip(values, self.totals))
        if not _consistent_counts(values) or not _consistent_counts(delta):
            self.rejected += 1
            return ZERO
        self.sequence, self.covered = snapshot.sequence, snapshot.covered
        self.totals, self.timestamp = values, snapshot.timestamp
        return delta


def _consistent_counts(values: tuple[int, ...]) -> bool:
    lower_bounds = (1,) + tuple(bound + 1 for bound in BUCKETS)
    minimum = sum(count * lower for count, lower in zip(values[4:-1], lower_bounds))
    maximum = sum(count * upper for count, upper in zip(values[4:-2], BUCKETS))
    return (
        values[3] <= values[2] <= values[1] <= values[0]
        and sum(values[4:-1]) == values[1]
        and values[-1] >= minimum
        and (values[-2] != 0 or values[-1] <= maximum)
    )


def register_frontend(config: Any, registration: LookupStatsRegistration) -> None:
    registrations = getattr(config, "_lookup_stats_registrations", None)
    if registrations is None:
        registrations = {}
        config._lookup_stats_registrations = registrations
    registrations[registration.engine_index] = registration


def initialize_owner(config: Any, log_stats: bool) -> LookupStatsAccumulator | None:
    from uuid import uuid4

    from vllm.v1.spec_decode.lookup_quarantine import require_reusable_context

    require_reusable_context()
    if not enabled(config, log_stats):
        # A reused configuration must not retain an earlier startup identity.
        config._lookup_stats_registration = None
        config._lookup_stats_registrations = {}
        return None
    parallel = config.parallel_config
    index = parallel.data_parallel_rank
    rank = parallel.world_size - (
        parallel.tensor_parallel_size * parallel.prefill_context_parallel_size
    )
    registration = LookupStatsRegistration(
        index, f"engine-{index}/rank-{rank}", uuid4().hex, rank
    )
    config._lookup_stats_registration = registration
    register_frontend(config, registration)
    return LookupStatsAccumulator(registration)


def set_worker_rank(config: Any, rank: int) -> None:
    config._lookup_stats_rank = rank


def merge_output(engine: Any, output: Any) -> None:
    accumulator = getattr(engine, "lookup_stats", None)
    message = getattr(output, "lookup_stats", None)
    if accumulator is not None and message is not None:
        accumulator.merge(message)
