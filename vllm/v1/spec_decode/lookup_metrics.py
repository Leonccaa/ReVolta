# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Front-end lookup metrics, with registration supplied by the startup handshake."""

from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from vllm.v1.spec_decode.lookup_stats import (
    BUCKETS,
    ZERO,
    LookupStatsAccumulator,
    LookupStatsMessage,
)

PREFIX = "vllm:dflash2_lookup_"
COUNTERS = ("eligible_rounds", "candidate_rounds", "gate_pass_rounds", "adopted_rounds")


class LookupStatsFrontend:
    def __init__(self, config: Any):
        self.config = config
        self.states: dict[int, LookupStatsAccumulator] = {}

    def observe(
        self, message: LookupStatsMessage, engine: int
    ) -> tuple[int, ...] | None:
        registrations = getattr(self.config, "_lookup_stats_registrations", {})
        registration = registrations.get(engine)
        if registration is None:
            return None
        state = self.states.get(engine)
        if state is None or state.registration != registration:
            state = self.states[engine] = LookupStatsAccumulator(registration)
        return state.merge(message)

    def log(self, log_fn, prefix: str = "") -> None:
        for engine, state in self.states.items():
            eligible, candidate, gate, adopted = state.totals[:4]
            ratio = f"{100 * adopted / eligible:.2f}%" if eligible else "no samples"
            log_fn(
                "%sLookup engine=%d eligible=%d candidate=%d gate=%d adopted=%d "
                "adoption=%s pending=%d producer=%d covered=%d sequence=%d "
                "last_export=%.3f",
                prefix,
                engine,
                eligible,
                candidate,
                gate,
                adopted,
                ratio,
                state.pending,
                state.producer,
                state.covered,
                state.sequence,
                state.timestamp,
            )


class LookupStatsProm(LookupStatsFrontend):
    def __init__(self, config: Any, labels: list[str], values: dict[int, list[object]]):
        super().__init__(config)
        self.metrics: dict[int, tuple] = {}
        self.watermarks: dict[int, tuple[int, int, int, float]] = {}
        # An off instance creates no new metric series.
        if not getattr(config, "_lookup_stats_registrations", {}):
            return
        counters = [
            Counter(
                PREFIX + name, "Native lookup cumulative " + name, labelnames=labels
            )
            for name in COUNTERS
        ]
        histogram = Histogram(
            PREFIX + "match_length",
            "Candidate suffix match length",
            labelnames=labels,
            buckets=BUCKETS,
        )
        gauges = [
            Gauge(
                PREFIX + name,
                "Native lookup " + name,
                labelnames=labels,
                multiprocess_mode="mostrecent",
            )
            for name in (
                "stats_pending",
                "stats_producer",
                "stats_covered",
                "stats_last_export_timestamp_seconds",
            )
        ]
        for engine, label_values in values.items():
            self.metrics[engine] = (
                [counter.labels(*label_values) for counter in counters],
                histogram.labels(*label_values),
                [gauge.labels(*label_values) for gauge in gauges],
            )

    def observe(
        self, message: LookupStatsMessage, engine: int
    ) -> tuple[int, ...] | None:
        delta = super().observe(message, engine)
        if delta is None or engine not in self.metrics:
            return delta
        counters, histogram, gauges = self.metrics[engine]
        if delta != ZERO:
            for counter, count in zip(counters, delta[:4]):
                if count:
                    counter.inc(count)
            # ValueClass implements the library's disjoint histogram buckets,
            # including multiprocess storage. Apply each integer delta once.
            for bucket, count in zip(histogram._buckets, delta[4:-1]):
                if count:
                    bucket.inc(count)
            histogram._sum.inc(delta[-1])
        state = self.states[engine]
        current = (state.pending, state.producer, state.covered, state.timestamp)
        previous = self.watermarks.get(engine, (None, None, None, None))
        for gauge, value, old in zip(gauges, current, previous):
            if value != old:
                gauge.set(value)
        self.watermarks[engine] = current
        return delta
