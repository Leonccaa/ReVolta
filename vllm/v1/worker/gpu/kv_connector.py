# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

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

    def seal_host_step(self, after_draft: bool = False) -> None:
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
        # MTP draft pages ride in the host row. The drafter writes this step's
        # rows after sampling, so their seal follows the proposal and its
        # receipt is reported by the next receipt (or the drain).
        self._seal_after_draft = False
        # A draft main K/V that lives in the host store is read and written by
        # the proposal, so the step cannot retire before the proposal does.
        self._draft_in_host = False
        self._seal_event = None
        self._late_seals: tuple[list[Any], bool, bool, Any, int | None] | None = None
        # Connector worker owns the registered store, including test doubles.
        worker = getattr(self.kv_connector, "connector_worker", None)
        if worker is not None and getattr(worker, "_host_history", None) is not None:
            self._host_store = worker._host_history
            config = worker.spec.kv_cache_config
            names = set(config.host_kv_layers)
            group_id = worker._host_group_id
            members = [
                name
                for name in config.kv_cache_groups[group_id].layer_names
                if name not in names
            ]
            self._host_indices = [kv_caches_dict[name] for name in members]
            self._seal_after_draft = any("mtp" in n.split(".") for n in members)
            self._draft_in_host = any("mtp" in n.split(".") for n in names)

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
        self._seal_event = None
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)

        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())

    def seal_host_step(self, after_draft: bool = False) -> None:
        step = self._host_step
        if step is None or self._host_sealed:
            return
        assert self._host_store is not None
        late = getattr(self, "_seal_after_draft", False)
        if late != after_draft:
            if not after_draft:
                # The target forward's host reads retire at this fence; the
                # seal itself follows the draft proposal.
                self._host_event = self._host_store.fence()
            return
        # Submission or context errors propagate. Only a successfully retired
        # stream may report a seal-only failure and preserve local GPU hashes.
        try:
            self._host_store.seal(step.seals, self._host_indices)
        except Exception:
            event = self._host_store.fence()
            if event is not None:
                event.synchronize()
            self._host_seal_failed = True
        if late:
            self._seal_event = self._host_store.fence()
        else:
            self._host_event = self._host_store.fence()
        self._host_sealed = True

    def _report_seals(self, meta, rank, tickets, sealed, seal_failed) -> None:
        from vllm.distributed import get_world_group

        for ticket in tickets:
            compute_failed = 0 if sealed else rank
            failed = rank if seal_failed else compute_failed
            meta.seal_results[ticket] = (rank, failed, compute_failed)
            payloads = getattr(self._host_store, "index_payloads", None)
            if not failed and isinstance(payloads, list):
                meta.seal_bytes[ticket] = {
                    get_world_group().rank_in_group: sum(
                        p.shape[1] * p.element_size() for p in payloads
                    )
                }

    def _host_receipt(self, output: KVConnectorOutput, final: bool = False) -> None:
        step = getattr(self, "_host_step", None)
        late = getattr(self, "_late_seals", None)
        if step is None and late is None:
            return
        from vllm.distributed import get_world_group
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            OffloadingWorkerMetadata,
        )

        if step is not None and self._host_event is not None:
            self._host_event.synchronize()
        elif step is not None and self._host_store is not None:
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
        if late is not None:
            # The previous step's seal, queued after its draft proposal; with
            # the draft in the host store, that step retires here as well.
            tickets, sealed, seal_failed, event, late_step = late
            if event is not None:
                event.synchronize()
            self._report_seals(meta, rank, tickets, sealed, seal_failed)
            if late_step is not None:
                meta.completed_steps[late_step] = rank
            self._late_seals = None
        if step is None:
            output.kv_connector_worker_meta = meta
            return
        # A host draft reads and writes pages until the after-draft fence: its
        # step retires with the next receipt instead of waiting here.
        draft_step = (
            getattr(self, "_draft_in_host", False)
            and getattr(self, "_seal_event", None) is not None
            and not final
        )
        if getattr(self, "_seal_after_draft", False) and not final:
            self._late_seals = (
                step.seals,
                self._host_sealed,
                self._host_seal_failed,
                self._seal_event,
                step.step if draft_step else None,
            )
        else:
            event = getattr(self, "_seal_event", None)
            if event is not None:
                event.synchronize()
            self._report_seals(
                meta, rank, step.seals, self._host_sealed, self._host_seal_failed
            )
        if not draft_step:
            meta.completed_steps[step.step] = rank
        shadow = getattr(getattr(self, "_host_store", None), "shadow", None)
        if shadow is not None:
            lagged = shadow.harvest(step.step)
            if lagged is not None:
                meta.value_counters[lagged[0]] = (rank, tuple(lagged[1]))
        output.kv_connector_worker_meta = meta
        store = getattr(self, "_host_store", None)
        hot_metrics = getattr(store, "hot_metrics", None)
        if hot_metrics is not None:
            assert store is not None
            report = hot_metrics.harvest(step.step, store.slot_owner[0])
            if report is not None:
                meta.hot_metrics[get_world_group().rank_in_group] = report
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
        self._host_receipt(output, final=True)
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
