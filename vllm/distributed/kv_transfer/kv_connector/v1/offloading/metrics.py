# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.worker.worker import TransferType

logger = init_logger(__name__)

# Key in ``OffloadingConnectorStats.data`` for CPU-tier evictions reported by
# the scheduler side: a list of {"reason": str, "blocks": int} entries.
CPU_EVICTIONS_KEY = "cpu_evicted_blocks"
# Key in ``OffloadingConnectorStats.data`` for speculative draft-model blocks
# the scheduler did not offload (kv_connector_extra_config "cpu.draft_offload"):
# a list of block counts.
CPU_DRAFT_SKIPPED_KEY = "cpu_draft_blocks_skipped"
CPU_RESIDENT_KEY = "cpu_resident_blocks"


@dataclass
class OffloadingOperationMetrics:
    op_size: int
    op_time: float


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Transfer ops per transfer type; CPU_EVICTIONS_KEY holds eviction
        # entries ({"reason", "blocks"}) from the scheduler side.
        self.data: dict[str, list[Any]] = {}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                if k == CPU_RESIDENT_KEY or k not in self.data:
                    self.data[k] = v
                else:
                    accumulator = self.data[k]
                    assert isinstance(accumulator, list)
                    accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        return_dict: dict[str, int | float] = {}
        for transfer_type, ops_list in self.data.items():
            assert isinstance(ops_list, list)
            if transfer_type == CPU_EVICTIONS_KEY:
                for entry in ops_list:
                    key = f"{CPU_EVICTIONS_KEY}_{entry['reason']}"
                    return_dict[key] = return_dict.get(key, 0) + entry["blocks"]
                continue
            if transfer_type == CPU_DRAFT_SKIPPED_KEY:
                return_dict[CPU_DRAFT_SKIPPED_KEY] = sum(ops_list)
                continue
            if transfer_type == CPU_RESIDENT_KEY:
                for entry in ops_list[-1:]:
                    for row in entry:
                        key = f"{CPU_RESIDENT_KEY}_{row['kind']}_{row['status']}"
                        return_dict[key] = row["blocks"]
                continue
            total_bytes = 0
            total_time = 0.0
            for op in ops_list:
                assert isinstance(op, dict)
                total_bytes += op["op_size"]
                total_time += op["op_time"]
            return_dict[f"{transfer_type}_total_bytes"] = total_bytes
            return_dict[f"{transfer_type}_total_time"] = total_time
        return return_dict

    def is_empty(self) -> bool:
        return not self.data

    def record_evictions(self, blocks_by_reason: dict[str, int]) -> None:
        entries = self.data.setdefault(CPU_EVICTIONS_KEY, [])
        for reason, blocks in blocks_by_reason.items():
            entries.append({"reason": reason, "blocks": blocks})

    def record_skipped_draft_blocks(self, blocks: int) -> None:
        self.data.setdefault(CPU_DRAFT_SKIPPED_KEY, []).append(blocks)

    def record_residency(self, totals: dict[tuple[str, str], int]) -> None:
        self.data[CPU_RESIDENT_KEY] = [
            [
                {"kind": kind, "status": status, "blocks": blocks}
                for (kind, status), blocks in totals.items()
            ]
        ]

    def record_transfer(self, num_bytes: int, time: float, transfer_type: TransferType):
        src, dst = transfer_type
        transfer_type_key = src + "_to_" + dst
        op = OffloadingOperationMetrics(num_bytes, time)
        if transfer_type_key in self.data:
            self.data[transfer_type_key].append(op)
        else:
            self.data[transfer_type_key] = [op]


class OffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        # (engine_idx, transfer_type) -> (metric with bounded labels)
        self.histogram_transfer_size: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_bytes: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_transfer_time: dict[tuple[int, str], PromMetricT] = {}
        buckets = [  # In bytes
            1e6,
            5e6,
            10e6,
            20e6,
            40e6,
            60e6,
            80e6,
            100e6,
            150e6,
            200e6,
        ]

        self._counter_kv_bytes = self._counter_cls(
            name="vllm:kv_offload_total_bytes",
            documentation="Number of bytes offloaded by KV connector",
            labelnames=labelnames + ["transfer_type"],
        )

        self._counter_kv_transfer_time = self._counter_cls(
            name="vllm:kv_offload_total_time",
            documentation="Total time measured by all KV offloading operations",
            labelnames=labelnames + ["transfer_type"],
        )

        self.counter_cpu_evicted_blocks: dict[tuple[int, str], PromMetricT] = {}
        self._counter_cpu_evicted_blocks = self._counter_cls(
            name="vllm:kv_offload_cpu_evicted_blocks",
            documentation=(
                "Blocks evicted from the CPU offload tier, by reason: capacity "
                "(policy eviction), superseded (evicted after being demoted as "
                "a superseded state), reset (dropped by a cache reset)."
            ),
            labelnames=labelnames + ["reason"],
        )

        self.counter_cpu_draft_blocks_skipped: dict[int, PromMetricT] = {}
        # Register lazily: disabled configurations expose exactly the old metrics.
        self._gauge_cpu_resident_blocks: PromMetricT | None = None
        self._counter_cpu_draft_blocks_skipped = self._counter_cls(
            name="vllm:kv_offload_cpu_draft_blocks_skipped",
            documentation=(
                "Speculative draft-model KV blocks not offloaded to the CPU "
                "tier because of kv_connector_extra_config cpu.draft_offload."
            ),
            labelnames=labelnames,
        )

        self._histogram_transfer_size = self._histogram_cls(
            name="vllm:kv_offload_size",
            documentation="Histogram of KV offload transfer size, in bytes.",
            buckets=buckets[:],
            labelnames=labelnames + ["transfer_type"],
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """
        Observe transfer statistics from the new data structure.
        transfer_stats_data is expected to be a dict where:
        - keys are transfer type strings (e.g., "cpu_to_gpu", "gpu_to_cpu")
        - values are lists of OffloadingOperationMetrics objects
        """

        for transfer_type, ops in transfer_stats_data.items():
            if transfer_type == CPU_EVICTIONS_KEY:
                self._observe_evictions(ops, engine_idx)
                continue
            if transfer_type == CPU_DRAFT_SKIPPED_KEY:
                self._observe_draft_skipped(ops, engine_idx)
                continue
            if transfer_type == CPU_RESIDENT_KEY:
                if self._gauge_cpu_resident_blocks is None:
                    self._gauge_cpu_resident_blocks = self._gauge_cls(
                        name="vllm:kv_offload_cpu_resident_blocks",
                        documentation=(
                            "Resident CPU blocks by kind and conservative usability."
                        ),
                        labelnames=self._labelnames + ["kind", "status"],
                    )
                for snapshot in ops[-1:]:
                    for row in snapshot:
                        self._gauge_cpu_resident_blocks.labels(
                            *(
                                self.per_engine_labelvalues[engine_idx]
                                + [row["kind"], row["status"]]
                            )
                        ).set(row["blocks"])
                continue
            # Cache:
            if (engine_idx, transfer_type) not in self.histogram_transfer_size:
                self.histogram_transfer_size[(engine_idx, transfer_type)] = (
                    self._histogram_transfer_size.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )
                self.counter_kv_bytes[(engine_idx, transfer_type)] = (
                    self._counter_kv_bytes.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )
                self.counter_kv_transfer_time[(engine_idx, transfer_type)] = (
                    self._counter_kv_transfer_time.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )

            # Process ops:
            assert isinstance(ops, list)
            for op in ops:  # ops is a list of serialized OffloadingOperationMetrics
                assert isinstance(op, dict)
                # Observe size histogram
                self.histogram_transfer_size[(engine_idx, transfer_type)].observe(
                    op["op_size"]
                )

                # Increment byte and time counters
                self.counter_kv_bytes[(engine_idx, transfer_type)].inc(op["op_size"])

                self.counter_kv_transfer_time[(engine_idx, transfer_type)].inc(
                    op["op_time"]
                )

    def _observe_evictions(self, entries: list[Any], engine_idx: int) -> None:
        for entry in entries:
            key = (engine_idx, entry["reason"])
            if key not in self.counter_cpu_evicted_blocks:
                self.counter_cpu_evicted_blocks[key] = (
                    self._counter_cpu_evicted_blocks.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [entry["reason"]])
                    )
                )
            self.counter_cpu_evicted_blocks[key].inc(entry["blocks"])

    def _observe_draft_skipped(self, counts: list[Any], engine_idx: int) -> None:
        if engine_idx not in self.counter_cpu_draft_blocks_skipped:
            self.counter_cpu_draft_blocks_skipped[engine_idx] = (
                self._counter_cpu_draft_blocks_skipped.labels(
                    *self.per_engine_labelvalues[engine_idx]
                )
            )
        for count in counts:
            self.counter_cpu_draft_blocks_skipped[engine_idx].inc(count)
