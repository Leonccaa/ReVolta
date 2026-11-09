# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from dataclasses import replace
from math import prod
from typing import TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    OffloadingSpec,
)
from vllm.v1.kv_offload.host.receive import HostReceiveSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingWorker,
    TransferSpec,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryStore


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()

        self.kv_connector_stats = OffloadingConnectorStats()
        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        self._unified_generations: dict[int, int] = {}
        self._unified_payload_bytes: dict[int, dict[str, int]] = {}
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()
        self._failed_submissions: set[int] = set()
        self._pending_failed_submissions: set[int] = set()
        self._host_history: HostHistoryStore | None = None
        self._host_group_id: int | None = None
        self._host_load_blocks: dict[int, list[int]] = {}
        self._host_index_views: list[torch.Tensor] = []
        self._receive_attempts: dict[int, int] = {}
        # Report failed transfers to the scheduler (which recomputes a failed
        # load and drops a failed store) instead of asserting.
        from vllm.v1.core.kv_cache_hit_contract import (
            load_failure_recompute_enabled,
        )

        self._report_failures: bool = load_failure_recompute_enabled(spec.vllm_config)

    def _register_handlers(self, kv_caches: CanonicalKVCaches):
        for src_cls, dst_cls, handler in self.spec.get_handlers(kv_caches):
            if self._host_history is not None and dst_cls is GPULoadStoreSpec:
                from vllm.v1.kv_offload.host.worker import HostReceiveHandler

                handler = HostReceiveHandler(
                    handler, self._host_history, self._host_index_views
                )
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        num_blocks = self.spec.kv_cache_config.num_blocks
        # Separate block spaces (long-context mode): a layer's tensor holds its
        # own space's blocks; the transfer handlers index it by that space's
        # block IDs, which the scheduler records per group.
        from vllm.v1.kv_cache_interface import kv_tensor_num_blocks

        layer_num_blocks = {
            name: kv_tensor_num_blocks(self.spec.kv_cache_config, tensor)
            for tensor in self.spec.kv_cache_config.kv_cache_tensors
            for name in tensor.shared_by
        }
        physical_strides: dict[str, int] = {}
        for tensor in self.spec.kv_cache_config.kv_cache_tensors:
            stride = getattr(tensor, "page_stride_bytes", None)
            if stride is not None:
                physical_strides.update({name: stride for name in tensor.shared_by})
        # Packed attention layers split each physical block of one tensor,
        # interleaved kernel block by kernel block, so they transfer as one
        # whole physical page registered once, by the first member.
        from vllm.v1.attention.kv_cache_view import packed_kv_members

        packed = packed_kv_members(self.spec.kv_cache_config)
        host_names = set(getattr(self.spec.kv_cache_config, "host_kv_layers", ()))
        transfer_names = self.spec.kv_cache_config.connector_transfer_layers()

        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        abi: dict[
            str, tuple[int, tuple[str, ...], str, tuple[int, ...], tuple[int, ...]]
        ] = {}
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                if layer_name not in transfer_names:
                    continue
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                member_stride = physical_strides.get(
                    layer_name, layer_kv_cache_spec.page_size_bytes
                )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert isinstance(layer_kv_cache, torch.Tensor)
                    member_index, member_count = packed.get(layer_name, (0, 1))
                    # Only the first packed member's view starts the storage.
                    assert (layer_kv_cache.storage_offset() == 0) == (member_index == 0)

                    from vllm.v1.attention.kv_cache_view import KVPhysicalPage
                    from vllm.v1.kv_cache_interface import KVQuantMode

                    owner = KVPhysicalPage(member_stride, member_count, member_index)
                    page = owner.page_size_bytes
                    tensors_per_block[layer_name] = (
                        owner.raw_pages(
                            layer_kv_cache, layer_num_blocks.get(layer_name, num_blocks)
                        ),
                    )
                    page_size_bytes[layer_name] = page
                    unpadded_page_size_bytes[layer_name] = owner.transfer_size(
                        layer_kv_cache,
                        layer_num_blocks.get(layer_name, num_blocks),
                        layer_kv_cache_spec.real_page_size_bytes,
                        planar=(
                            layer_kv_cache_spec.kv_quant_mode == KVQuantMode.INT8_PLANAR
                        ),
                        padded=layer_kv_cache_spec.page_size_padded is not None,
                    )
                    if layer_name in physical_strides:
                        # Remaining canonical members keep their physical
                        # transfer ABI. G0 is excluded from this registration.
                        unpadded_page_size_bytes[layer_name] = page
                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    state_tensors = kv_caches[layer_name]
                    assert isinstance(state_tensors, list)

                    # re-construct the raw (num_blocks, page_size) tensor
                    # from the first state tensor
                    assert len(state_tensors) > 0
                    first_state_tensor = state_tensors[0]
                    assert first_state_tensor.storage_offset() == 0
                    tensor = (
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=first_state_tensor.device,
                        )
                        .set_(first_state_tensor.untyped_storage())
                        .view(
                            (
                                layer_num_blocks.get(layer_name, num_blocks),
                                member_stride,
                            )
                        )
                    )
                    tensors_per_block[layer_name] = (tensor,)

                    page_size_bytes[layer_name] = member_stride
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes

                else:
                    raise NotImplementedError

        if getattr(self.spec, "unified_host", False):
            for group in self.spec.kv_cache_config.kv_cache_groups:
                sizes = group.kv_cache_spec
                for name in group.layer_names:
                    if name not in tensors_per_block:
                        continue
                    size = (
                        sizes.kv_cache_specs[name]
                        if isinstance(sizes, UniformTypeKVCacheSpecs)
                        else sizes
                    )
                    raw = kv_caches[name]
                    if isinstance(raw, list):
                        abi[name] = (
                            1,
                            tuple(str(t.dtype) for t in raw),
                            "none",
                            tuple(t.storage_offset() * t.element_size() for t in raw),
                            tuple(prod(t.shape[1:]) * t.element_size() for t in raw),
                        )
                    else:
                        blocks = layer_num_blocks.get(name, num_blocks)
                        # Registered physical owners cover all interleaved
                        # kernel subdivisions, including packed members.
                        rows = (
                            raw.shape[1]
                            if raw.ndim == 5 and raw.shape[0] == 2
                            else raw.shape[0]
                            if raw.ndim >= 3
                            else blocks
                        )
                        subdivision = (
                            rows // blocks
                            if rows >= blocks and rows % blocks == 0
                            else 1
                        )
                        mode = getattr(size, "kv_quant_mode", None)
                        abi[name] = (
                            subdivision,
                            (str(raw.dtype),),
                            getattr(mode, "name", "none"),
                            (0,),
                            (unpadded_page_size_bytes[name],),
                        )

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in self.spec.kv_cache_config.kv_cache_tensors:
            # Filter to layers that were actually processed above.
            # _get_kv_cache_config_deepseek_v4 emits KVCacheTensor entries for
            # every (tuple_idx, page_size) slot; slots where no group has a
            # layer at that index produce an empty shared_by (reserved memory
            # with no corresponding model layer).
            tensor_layer_names = [
                n for n in kv_cache_tensor.shared_by if n in tensors_per_block
            ]
            if not tensor_layer_names:
                continue

            # verify all layers in the group reference the exact same tensors
            assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1
            assert (
                len({tensors_per_block[n][0].data_ptr() for n in tensor_layer_names})
                == 1
            )
            assert (
                len({tensors_per_block[n][0].stride() for n in tensor_layer_names}) == 1
            )

            # pick the first layer to represent the group
            first_layer_name = tensor_layer_names[0]
            for tensor in tensors_per_block[first_layer_name]:
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=page_size_bytes[first_layer_name],
                    )
                )

                curr_tensor_idx = len(block_tensors) - 1
                for layer_name in tensor_layer_names:
                    if packed.get(layer_name, (0, 1))[0] > 0:
                        # The first member's reference covers the whole page.
                        continue
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                            subdivision=abi.get(layer_name, (1, (), "none", (), ()))[0],
                            dtypes=abi.get(layer_name, (1, (), "none", (), ()))[1],
                            scale=abi.get(layer_name, (1, (), "none", (), ()))[2],
                            member_offsets=abi.get(layer_name, (1, (), "none", (), ()))[
                                3
                            ],
                            member_lengths=abi.get(layer_name, (1, (), "none", (), ()))[
                                4
                            ],
                        )
                    )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        if host_names:
            context = self.spec.vllm_config.compilation_config.static_forward_context
            stores = {context[name].host_history.store for name in host_names}
            if len(stores) != 1:
                raise ValueError("Host main K/V must use one history store per worker")
            (store,) = stores
            group_ids = [
                index
                for index, group in enumerate(self.spec.kv_cache_config.kv_cache_groups)
                if not host_names.isdisjoint(group.layer_names)
            ]
            if len(group_ids) != 1:
                raise ValueError("Host main K/V must belong to one transfer group")
            (self._host_group_id,) = group_ids
            self._host_history = store
            if getattr(self.spec, "unified_host", False):
                from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

                assert isinstance(self.spec, CPUOffloadingSpec)
                self.spec.unified_region = store.region
            self._host_index_views = [
                kv_caches[name]
                for name in self.spec.kv_cache_config.kv_cache_groups[
                    self._host_group_id
                ].layer_names
                if name not in host_names
            ]
            if any(
                not isinstance(view, torch.Tensor) for view in self._host_index_views
            ):
                raise ValueError("Host index members require tensor views")

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._register_handlers(canonical_kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        # verify that num_blocks is at physical position 0 in the cross-layers
        # tensor layout.
        test_shape = attn_backend.get_kv_cache_shape(
            num_blocks=1234, block_size=16, num_kv_heads=1, head_size=256
        )
        num_blocks_logical_dim = test_shape.index(1234) + 1
        physical_to_logical = attn_backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
        num_blocks_physical_dim = physical_to_logical.index(num_blocks_logical_dim)
        assert num_blocks_physical_dim == 0

        kv_cache_groups = self.spec.kv_cache_config.kv_cache_groups
        assert len(kv_cache_groups) == 1
        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        num_layers = len(kv_cache_groups[0].layer_names)
        page_size_bytes = kv_cache_spec.page_size_bytes * num_layers

        assert kv_cache.storage_offset() == 0
        storage = kv_cache.untyped_storage()
        assert len(storage) % page_size_bytes == 0
        num_blocks = len(storage) // page_size_bytes
        tensor = (
            torch.tensor(
                [],
                dtype=torch.int8,
                device=kv_cache.device,
            )
            .set_(storage)
            .view(num_blocks, page_size_bytes)
        )
        kv_cache_tensor = CanonicalKVCacheTensor(
            tensor=tensor, page_size_bytes=page_size_bytes
        )
        # in cross layers layout, there's currently only a single group
        kv_cache_data_ref = CanonicalKVCacheRef(
            tensor_idx=0, page_size_bytes=page_size_bytes
        )
        canonical_kv_caches = CanonicalKVCaches(
            tensors=[kv_cache_tensor], group_data_refs=[[kv_cache_data_ref]]
        )

        self._register_handlers(canonical_kv_caches)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            self._submit_transfer(job_id, transfer_spec)
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            # Failed loads retain their destination allocation until receive
            # completion. A cancelled request's terminal flush can defer these
            # drains to get_finished without exposing those blocks to compute.
            pending_loads = self._pending_failed_submissions.intersection(
                self._load_jobs
            )
            self.worker.wait(kv_connector_metadata.jobs_to_flush - pending_loads)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            self._submit_transfer(job_id, transfer_spec)
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            if entry.generation is not None:
                self._remember_unified_job(job_id, entry)
            self._load_jobs[job_id] = entry.req_id
            source = entry.transfer_spec[0]
            if isinstance(source, HostReceiveSpec):
                self._receive_attempts[job_id] = source.txn.attempt_id
                self._host_load_blocks[job_id] = [e.hid for e in source.txn.new_loads]
            self._submit_transfer(job_id, entry.transfer_spec)

    def _remember_unified_job(self, job_id, entry) -> None:
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        assert isinstance(self.spec, CPUOffloadingSpec)
        assert self.spec._handlers is not None
        self._unified_generations[job_id] = entry.generation
        gpu = next(s for s in entry.transfer_spec if isinstance(s, GPULoadStoreSpec))
        amounts: dict[str, int] = {}
        for desc in self.spec._handlers.transfer_layout:
            group = self.spec.kv_cache_config.kv_cache_groups[desc.group]
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()))
            kind = "state" if isinstance(spec, MambaSpec) else "draft"
            amounts[kind] = (
                amounts.get(kind, 0) + desc.length * gpu.group_sizes[desc.group]
            )
        self._unified_payload_bytes[job_id] = amounts

    def _submit_transfer(self, job_id: int, transfer_spec: TransferSpec) -> None:
        try:
            success = self.worker.transfer_async(job_id, transfer_spec)
        except Exception:
            if not getattr(self, "_report_failures", False):
                raise
            success = False
        if success:
            return
        if not getattr(self, "_report_failures", False):
            assert success
        # Drain any submitted part before the scheduler can reuse destinations
        # or release source pins. wait must also accept an unsubmitted job ID.
        try:
            self.worker.wait({job_id})
        except Exception:
            # The handler still owns the fence. Retry it from get_finished,
            # retaining the scheduler's source pins and destination blocks.
            self._pending_failed_submissions.add(job_id)
            logger.warning(
                "KV transfer job %d is still awaiting drain", job_id, exc_info=True
            )
            return
        self._failed_submissions.add(job_id)

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if entry.generation is not None:
                self._remember_unified_job(job_id, entry)
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            self._unsubmitted_store_jobs.append((job_id, entry.transfer_spec))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Returns:
            tuple of (finished_sending, finished_recving). Stores never
            emit finished_sending — the scheduler tracks store completion
            via kv_connector_worker_meta.completed_jobs and fences any
            block reuse via jobs_to_flush. Loads still emit
            finished_recving so the base scheduler can resume requests
            blocked on remote KV (and free aborted-during-load reqs).
        """
        finished_recving: set[str] = set()
        pending: set[int] = getattr(self, "_pending_failed_submissions", set())
        for job_id in list(pending):
            try:
                self.worker.wait({job_id})
            except Exception:
                continue
            pending.remove(job_id)
            self._failed_submissions.add(job_id)
        failed_submissions: set[int] = getattr(self, "_failed_submissions", set())
        for transfer_result in self.worker.get_finished():
            job_id = transfer_result.job_id
            if job_id in failed_submissions or job_id in pending:
                # Report this drained, partially submitted job once below.
                continue
            # getattr: tests may build this object without __init__.
            if not transfer_result.success and getattr(self, "_report_failures", False):
                logger.warning("KV offloading transfer job %d failed", job_id)
                self._connector_worker_meta.mark_failed(job_id)
            else:
                # job failures are not supported without the recompute path
                assert transfer_result.success
            host_history = getattr(self, "_host_history", None)
            if host_history is not None:
                blocks = self._host_load_blocks.pop(job_id, [])
                if transfer_result.success:
                    # All group copies on this worker retired. Only NEW_LOAD
                    # IDs become ready. REUSE_LOCAL keeps its original state.
                    # The scheduler still waits for every rank and the step.
                    try:
                        host_history.mark_ready(blocks)
                    except Exception:
                        if not self._report_failures:
                            raise
                        transfer_result.success = False
                        self._connector_worker_meta.mark_failed(job_id)
            if (
                transfer_result.transfer_time
                and transfer_result.transfer_size is not None
                and transfer_result.transfer_type is not None
            ):
                self.kv_connector_stats.record_transfer(
                    num_bytes=transfer_result.transfer_size,
                    time=transfer_result.transfer_time,
                    transfer_type=transfer_result.transfer_type,
                )

            self._connector_worker_meta.mark_completed(job_id)
            self._receive_receipt(
                job_id, transfer_result.success, transfer_result.transfer_size
            )
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        for job_id in failed_submissions:
            if getattr(self, "_host_history", None) is not None:
                self._host_load_blocks.pop(job_id, None)
            self._connector_worker_meta.mark_failed(job_id)
            self._connector_worker_meta.mark_completed(job_id)
            self._receive_receipt(job_id, False)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        failed_submissions.clear()

        return set(), finished_recving

    def _receive_receipt(
        self, job_id: int, success: bool, nbytes: int | None = None
    ) -> None:
        generation = getattr(self, "_unified_generations", {}).pop(job_id, None)
        if generation is not None:
            from vllm.distributed import get_world_group

            rank = get_world_group().rank_in_group
            mask = 1 << rank
            amounts = self._unified_payload_bytes.pop(job_id, {})
            if nbytes is not None and success:
                # The completed copy count proves these bytes. G0 index H2D
                # is the remainder added by HostReceiveHandler; host UVA
                # traffic is not included. Failed partial DMA stays unknown.
                expected = sum(amounts.values())
                if amounts and expected <= nbytes:
                    amounts["main"] = nbytes - expected
                else:
                    amounts = {"unknown": nbytes}
                for kind, amount in amounts.items():
                    self._connector_worker_meta.job_bytes[
                        job_id, generation, rank, kind
                    ] = amount
            self._connector_worker_meta.job_results[job_id, generation] = (
                mask,
                0 if success else mask,
            )
        attempt = getattr(self, "_receive_attempts", {}).pop(job_id, None)
        if attempt is None:
            return
        from vllm.distributed import get_world_group

        rank = 1 << get_world_group().rank_in_group
        self._connector_worker_meta.receive_results[attempt] = (
            rank,
            0 if success else rank,
        )

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def drain(self) -> tuple[set[str], set[str], OffloadingWorkerMetadata | None]:
        """Finish deferred submissions and all handlers without releasing backing."""
        self.start_kv_transfers(OffloadingConnectorMetadata({}, {}))
        for handler in self.worker.handlers:
            handler.drain()
        sending, receiving = self.get_finished(set())
        if self._load_jobs or self._pending_failed_submissions:
            raise RuntimeError("Connector drain left unfinished transfer jobs")
        return sending, receiving, self.build_connector_worker_meta()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """

        if self.kv_connector_stats.is_empty():
            return None
        # Clear stats for next iteration
        kv_connector_stats = self.kv_connector_stats
        self.kv_connector_stats = OffloadingConnectorStats()
        return kv_connector_stats

    def shutdown(self) -> None:
        self.worker.shutdown()
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._failed_submissions.clear()
        self._pending_failed_submissions.clear()
        getattr(self, "_host_load_blocks", {}).clear()
        getattr(self, "_receive_attempts", {}).clear()
        getattr(self, "_host_index_views", []).clear()
        self._host_history = None
        self._connector_worker_meta = OffloadingWorkerMetadata()
