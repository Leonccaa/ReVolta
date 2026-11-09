# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.v1.kv_offload.host.page_table import HostKVStep
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def handle_preemptions(self, scheduler_output: "SchedulerOutput") -> None:
        """Submit deferred stores and finish those whose source block is reused.

        Called before the step zeroes new blocks, precopies Mamba state or
        writes KV; pre_forward is too late for that.
        """
        pass

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        pass

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass

    def seal_host_step(self) -> None:
        pass

    def close_host_kv(self) -> None:
        pass


class ActiveKVConnector(KVConnector):
    def __init__(
        self, vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
    ):
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        # TODO: support cross_layers_kv_cache
        # (see https://github.com/vllm-project/vllm/pull/27743)
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._disabled = False
        # Metadata whose preemptions were already handled this step.
        self._handled_metadata: object | None = None
        self._host_step: HostKVStep | None = None
        self._host_store = None
        self._host_indices: list[torch.Tensor] = []
        self._host_event = None
        self._host_sealed = False
        self._host_seal_failed = False
        self._host_drain_output: KVConnectorOutput | None = None
        self._host_drain_complete = False
        # Connector worker owns the registered store, including test doubles.
        worker = getattr(self.kv_connector, "connector_worker", None)
        if worker is not None and getattr(worker, "_host_history", None) is not None:
            self._host_store = worker._host_history
            config = worker.spec.kv_cache_config
            names = set(config.host_kv_layers)
            group_id = worker._host_group_id
            self._host_indices = [
                kv_caches_dict[name]
                for name in config.kv_cache_groups[group_id].layer_names
                if name not in names
            ]

    def handle_preemptions(self, scheduler_output: "SchedulerOutput") -> None:
        # Deferred stores read blocks written by the previous step. A block
        # freed since then may be handed out again in this step; the
        # connector lists those stores in jobs_to_flush, and they must
        # complete before this step zeroes, precopies or overwrites the block,
        # or the CPU copy holds zeros (MRv1 calls this before _update_states).
        if self._disabled:
            return
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self._handled_metadata = kv_connector_metadata

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        if self._handled_metadata is not kv_connector_metadata:
            self.kv_connector.handle_preemptions(kv_connector_metadata)
        self._handled_metadata = None
        self._host_step = getattr(scheduler_output, "host_kv", None)
        shadow = getattr(getattr(self, "_host_store", None), "shadow", None)
        epoch = getattr(self._host_step, "arbiter_epoch", None)
        if shadow is not None and epoch is not None:
            # Forward stream, before this step's gathers (outside graphs).
            shadow.set_window(epoch)
        self._host_sealed = False
        self._host_seal_failed = False
        self._host_event = None
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)

        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())

    def seal_host_step(self) -> None:
        step = self._host_step
        if step is None or self._host_sealed:
            return
        assert self._host_store is not None
        # Submission or context errors propagate. Only a successfully retired
        # stream may report a seal-only failure and preserve local GPU hashes.
        try:
            self._host_store.seal(step.seals, self._host_indices)
        except Exception:
            event = self._host_store.fence()
            if event is not None:
                event.synchronize()
            self._host_seal_failed = True
        self._host_event = self._host_store.fence()
        self._host_sealed = True

    def _host_receipt(self, output: KVConnectorOutput) -> None:
        step = getattr(self, "_host_step", None)
        if step is None:
            return
        from vllm.distributed import get_world_group
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            OffloadingWorkerMetadata,
        )

        if self._host_event is not None:
            self._host_event.synchronize()
        elif self._host_store is not None:
            # No-forward and aborted producer steps still need a physical
            # retirement fence before a rank can advance the watermark.
            event = self._host_store.fence()
            if event is not None:
                event.synchronize()
        meta = output.kv_connector_worker_meta
        if meta is None:
            meta = OffloadingWorkerMetadata()
        assert isinstance(meta, OffloadingWorkerMetadata)
        rank = 1 << get_world_group().rank_in_group
        for ticket in step.seals:
            compute_failed = 0 if self._host_sealed else rank
            seal_failed = rank if self._host_seal_failed else compute_failed
            meta.seal_results[ticket] = (rank, seal_failed, compute_failed)
        meta.completed_steps[step.step] = rank
        shadow = getattr(getattr(self, "_host_store", None), "shadow", None)
        if shadow is not None:
            lagged = shadow.harvest(step.step)
            if lagged is not None:
                meta.value_counters[lagged[0]] = (rank, tuple(lagged[1]))
        output.kv_connector_worker_meta = meta
        self._host_step = None
        self._host_event = None

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        if self._disabled:
            return None

        output = KVConnectorOutput()
        if wait_for_save:
            self.kv_connector.wait_for_save()
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(finished_req_ids)
        )
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        self._host_receipt(output)
        self.kv_connector.clear_connector_metadata()
        return output

    def drain_host_kv(self) -> KVConnectorOutput:
        output = getattr(self, "_host_drain_output", None)
        if output is None:
            output = KVConnectorOutput()
            self._host_drain_output = output
        if getattr(self, "_host_drain_complete", False):
            return output
        # Keep receipts across a failed drain on this or another rank. Parent
        # shutdown consumes them only after every rank has acknowledged.
        self._host_receipt(output)
        worker = getattr(self.kv_connector, "connector_worker", None)
        assert worker is not None
        sending, receiving, meta = worker.drain()
        output.finished_sending, output.finished_recving = sending, receiving
        if meta is not None:
            previous = output.kv_connector_worker_meta
            output.kv_connector_worker_meta = (
                meta if previous is None else previous.aggregate(meta)
            )
        self._host_drain_complete = True
        return output

    def close_host_kv(self) -> None:
        self._host_store = None
        self._host_indices.clear()

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        finished_req_ids = scheduler_output.finished_req_ids
        kv_connector_output = self.post_forward(finished_req_ids, wait_for_save=False)
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    def set_disabled(self, disabled: bool) -> None:
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(vllm_config, kv_caches_dict)
