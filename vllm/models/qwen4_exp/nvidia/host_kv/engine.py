# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind a worker's host history store to its QSA layers (long-context mode).

``KVCacheConfig.host_kv_layers`` names the QSA layers whose main K/V has no
GPU tensor (``main_kv_placement="host"``). The model runner calls
``build_host_history_store`` once, after the KV cache tensors exist and
before any CUDA graph capture; every such layer then carries
``layer.host_history`` (a ``HostHistoryLayer``) and the QSA forward reads and
writes through it. ``begin_blocks`` must see every new history block ID
before the step that first writes it (the runner forwards the scheduler's
``new_block_ids_to_zero``).
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)

from .history import TAIL_GROUPS, HostHistoryShape, HostHistoryStore, hot_alias_view

logger = init_logger(__name__)


def shared_index_tensors(
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor] | None,
    num_layers: int,
) -> list[torch.Tensor]:
    """The index tensors whose pages carry the shared hot region, one per
    host layer in layer order (the padded, explicit-stride index tensors)."""
    tensors = [
        tensor
        for tensor in kv_cache_config.kv_cache_tensors
        if tensor.page_stride_bytes is not None
    ]
    if kv_caches is None or len(tensors) != num_layers:
        raise RuntimeError(
            "The shared hot region needs one padded index tensor per host layer"
        )
    views = []
    for tensor in tensors:
        view = kv_caches[tensor.shared_by[0]]
        if view.untyped_storage().nbytes() != tensor.size:
            raise RuntimeError("Index tensor storage differs from its plan")
        views.append(view)
    if len({v.untyped_storage().data_ptr() for v in views}) != num_layers:
        raise RuntimeError("Shared hot region layers must not share index pages")
    return views


def build_host_history_store(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    forward_context: dict[str, Any],
    kernel_block_sizes: list[int],
    device: torch.device,
    kv_caches: dict[str, torch.Tensor] | None = None,
) -> HostHistoryStore | None:
    """Allocate the store for ``host_kv_layers`` and bind it to the layers."""
    names = list(getattr(kv_cache_config, "host_kv_layers", None) or [])
    if not names:
        return None
    plan = kv_cache_config.long_context_plan
    if plan is None or plan.main_kv_on_gpu:
        raise RuntimeError("Host main K/V layers need a host-placement plan")
    group_ids = {
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        for name in group.layer_names
        if name in names
    }
    if len(group_ids) != 1:
        raise RuntimeError("Host main K/V layers must form one KV cache group")
    (group_id,) = group_ids
    if not kv_cache_config.in_primary_block_space(group_id):
        raise RuntimeError("Host main K/V must use the history block space")
    group_spec = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
    specs = [
        group_spec.kv_cache_specs[name]
        if isinstance(group_spec, UniformTypeKVCacheSpecs)
        else group_spec
        for name in names
    ]
    spec = specs[0]
    if not isinstance(spec, AttentionSpec) or any(other != spec for other in specs):
        raise RuntimeError("Host main K/V layers need one attention page geometry")
    layers = [forward_context[name] for name in names]
    kernel_block = kernel_block_sizes[group_id]
    storage_block = spec.storage_block_size
    if storage_block % kernel_block:
        raise RuntimeError("Host main K/V kernel blocks must divide a page")

    parallel = vllm_config.parallel_config
    sharded = bool(getattr(layers[0], "qsa_dcp_sharded", False))
    dcp_world, dcp_rank = 1, 0
    if sharded:
        from vllm.distributed import get_dcp_group

        group = get_dcp_group()
        dcp_world, dcp_rank = group.world_size, group.rank_in_group
    widths = {layer.indexer.output_width for layer in layers}
    if len(widths) != 1:
        raise RuntimeError("Host main K/V layers need one selection width")
    from vllm.config.compilation import CUDAGraphMode

    from .planning import validate_graph_capacity

    # MRv2 resolves/rounds the actual graph sizes before binding this store.
    # Check that collection, not just max_num_seqs * verify width.
    compilation = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None) or CUDAGraphMode.NONE
    read_rows = plan.geometry.hot_read_rows
    assert read_rows is not None
    validate_graph_capacity(
        read_rows,
        (getattr(compilation, "cudagraph_capture_sizes", None) or ())
        if mode.decode_mode() == CUDAGraphMode.FULL
        else (),
        vllm_config.scheduler_config.max_num_seqs,
        plan.geometry.decode_rows_per_request,
        mixed_full=mode.mixed_mode() == CUDAGraphMode.FULL,
    )
    shape = HostHistoryShape(
        num_blocks=kv_cache_config.num_blocks,
        kernel_blocks_per_block=storage_block // kernel_block,
        kernel_block_tokens=kernel_block,
        num_layers=len(names),
        kv_heads=spec.num_kv_heads,
        head_dim=spec.head_size,
        dtype=spec.dtype,
        hot_slots=(
            kv_cache_config.num_blocks * plan.hot_slots_per_block
            if plan.hot_slots_per_block
            else plan.hot_pool_groups
        ),
        slots_per_block=plan.hot_slots_per_block,
        max_requests=vllm_config.scheduler_config.max_num_seqs,
        selection_width=widths.pop(),
        max_write_rows=vllm_config.scheduler_config.max_num_batched_tokens,
        row_groups=plan.geometry.hot_row_groups,
        tail_groups=TAIL_GROUPS,
        dcp_world_size=dcp_world,
        dcp_rank=dcp_rank,
        interleave=parallel.cp_kv_cache_interleave_size if sharded else 1,
        # One claim per selected column: robust to any selection layout; the
        # per-group claim (host KV pool perf) is an optimization left for GPU tuning.
        qsa_group_columns=0,
        read_rows=read_rows,
        max_context_tokens=plan.geometry.max_model_len,
        decode_rows=plan.geometry.decode_rows_per_request,
    )
    # Check before allocating (and again before publishing layer bindings).
    fixed_bytes = sum(shape.fixed_device_memory_bytes().values())
    if fixed_bytes > plan.gpu_hot_pool_bytes:
        raise ValueError(
            f"Host history fixed allocations need {fixed_bytes} device bytes, "
            f"exceeding the hot-pool reservation {plan.gpu_hot_pool_bytes}"
        )
    from vllm.distributed import get_world_group
    from vllm.v1.kv_offload.host.region import host_region_layout

    layout = host_region_layout(kv_cache_config, vllm_config.instance_id)
    hot_views = None
    if plan.hot_slots_per_block:
        hot_views = [
            hot_alias_view(index, shape)
            for index in shared_index_tensors(kv_cache_config, kv_caches, len(names))
        ]
    store = HostHistoryStore(
        shape,
        names,
        device=device,
        region_layout=layout,
        rank=get_world_group().rank_in_group,
        hot_views=hot_views,
    )
    if store.fixed_device_memory_bytes() != shape.fixed_device_memory_bytes():
        raise ValueError(
            "Host history fixed allocation accounting differs from its shape"
        )
    for name, layer in zip(names, layers):
        layer.host_history = store.layer(name)
    logger.info(
        "Long-context host memory: %d physical pages, %d history IDs, %d layers "
        "(%.2f GiB logical shard, %.2f GiB registered region), "
        "hot pool %d groups per layer, %d "
        "query rows per gather pass.",
        layout.num_pages,
        shape.num_blocks,
        shape.num_layers,
        layout.num_pages * layout.worker_bytes / (1 << 30),
        layout.num_pages * layout.worker_bytes * layout.num_workers / (1 << 30),
        shape.hot_slots,
        shape.rows_per_pass,
    )
    return store
