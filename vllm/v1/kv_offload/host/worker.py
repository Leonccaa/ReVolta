# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join STATE/WINDOW DMA with the ReceiveTxn's logical index copies."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.kv_cache_view import scatter_page_payload
from vllm.v1.kv_offload.host.receive import HostReceiveSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryStore


@dataclass
class _IndexCopy:
    stream: torch.cuda.Stream
    start_event: torch.Event
    event: torch.Event
    staging: torch.Tensor | None = None
    ordinary_result: TransferResult | None = None
    failed: bool = False
    num_bytes: int = 0


class HostReceiveHandler(OffloadingHandler):
    """One completion proves that every group on this rank has retired.

    Partially submitted index copies retain their stream and allocations.
    wait/drain must prove retirement of both handlers before failure reporting.
    """

    def __init__(
        self,
        ordinary: OffloadingHandler,
        store: "HostHistoryStore",
        index_views: list[torch.Tensor],
    ):
        self.ordinary = ordinary
        self.store = store
        self.index_views = index_views
        self._copies: dict[int, _IndexCopy] = {}
        self._drained_results: list[TransferResult] = []

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        source, target = spec
        assert isinstance(source, HostReceiveSpec)
        copy = _IndexCopy(
            torch.cuda.Stream(),
            torch.Event(enable_timing=True),
            torch.Event(enable_timing=True),
        )
        self._copies[job_id] = copy
        try:
            if not self.ordinary.transfer_async(job_id, (source.source, target)):
                raise RuntimeError("STATE/WINDOW receive submission failed")
            copy.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy.stream):
                copy.start_event.record(copy.stream)
                if len(self.index_views) != len(self.store.index_payloads):
                    raise ValueError("Receive index members differ from the host ABI")
                new_loads = source.txn.new_loads
                if new_loads and self.index_views:
                    # One stream-ordered scratch page per job bounds the peak
                    # independently of the length of the restored prefix.
                    copy.staging = torch.empty(
                        max(p.shape[1] for p in self.store.index_payloads),
                        dtype=torch.uint8,
                        device=self.index_views[0].device,
                    )
                for entry in new_loads:
                    for view, payload in zip(
                        self.index_views, self.store.index_payloads
                    ):
                        if view.shape[0] % self.store.shape.num_blocks:
                            raise ValueError(
                                "Index kernel pages do not divide history IDs"
                            )
                        assert copy.staging is not None
                        packed = copy.staging[: payload.shape[1]]
                        packed.copy_(payload[entry.handle.page_id], non_blocking=True)
                        scatter_page_payload(
                            packed,
                            view,
                            entry.hid,
                            subdivisions=view.shape[0] // self.store.shape.num_blocks,
                        )
                        copy.num_bytes += packed.numel()
                copy.event.record(copy.stream)
        except Exception:
            copy.failed = True
            raise
        return True

    def wait(self, job_ids: set[int]) -> None:
        self.ordinary.wait(job_ids)
        for job_id in job_ids:
            copy = self._copies.get(job_id)
            if copy is not None:
                copy.stream.synchronize()

    def get_finished(self) -> list[TransferResult]:
        results = self._drained_results
        for result in self.ordinary.get_finished():
            copy = self._copies.get(result.job_id)
            if copy is None:
                results.append(result)
            else:
                copy.ordinary_result = result
                copy.failed |= not result.success
        for job_id, copy in list(self._copies.items()):
            if copy.failed:
                try:
                    self.wait({job_id})
                except Exception:
                    continue
            elif copy.ordinary_result is None:
                continue
            else:
                try:
                    if not copy.event.query():
                        continue
                except Exception:
                    copy.failed = True
                    try:
                        self.wait({job_id})
                    except Exception:
                        continue
            ordinary_bytes = (
                copy.ordinary_result.transfer_size or 0
                if copy.ordinary_result is not None
                else 0
            )
            # Existing transfer metrics sum component device time. The two
            # streams can overlap, so this is distinct from request latency.
            index_time = (
                copy.start_event.elapsed_time(copy.event) * 1e-3
                if copy.num_bytes and not copy.failed
                else 0.0
            )
            ordinary_time = (
                copy.ordinary_result.transfer_time or 0.0
                if copy.ordinary_result is not None
                else 0.0
            )
            results.append(
                TransferResult(
                    job_id,
                    success=not copy.failed,
                    transfer_size=ordinary_bytes + copy.num_bytes,
                    transfer_time=ordinary_time + index_time,
                    transfer_type=("CPU", "GPU"),
                )
            )
            del self._copies[job_id]
        self._drained_results = []
        return results

    def drain(self) -> None:
        self.wait(set(self._copies))
        # The ordinary handler's drain consumes its event queue. Harvest its
        # receipts first and retain them across another handler's drain failure.
        results = self.get_finished()
        self._drained_results.extend(results)
        self.ordinary.drain()
        if self._copies:
            raise RuntimeError("Host receive drain left incomplete copies")

    def shutdown(self) -> None:
        self.ordinary.shutdown()
        self._copies.clear()
        self._drained_results.clear()
        self.index_views.clear()
