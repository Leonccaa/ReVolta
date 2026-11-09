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

from .history import TAIL_GROUPS, HostHistoryShape, HostHistoryStore

logger = init_logger(__name__)


def build_host_history_store(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    forward_context: dict[str, Any],
    kernel_block_sizes: list[int],
    device: torch.device,
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
    shape = HostHistoryShape(
        num_blocks=kv_cache_config.num_blocks,
        kernel_blocks_per_block=storage_block // kernel_block,
        kernel_block_tokens=kernel_block,
        num_layers=len(names),
        kv_heads=spec.num_kv_heads,
        head_dim=spec.head_size,
        dtype=spec.dtype,
        hot_slots=plan.hot_pool_groups,
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
    )
    # Check before allocating (and again before publishing layer bindings).
    fixed_bytes = sum(shape.fixed_device_memory_bytes().values())
    if fixed_bytes > plan.gpu_hot_pool_bytes:
        raise ValueError(
            f"Host history fixed allocations need {fixed_bytes} device bytes, "
            f"exceeding the hot-pool reservation {plan.gpu_hot_pool_bytes}"
        )
    store = HostHistoryStore(shape, names, device=device)
    if store.fixed_device_memory_bytes() != shape.fixed_device_memory_bytes():
        raise ValueError(
            "Host history fixed allocation accounting differs from its shape"
        )
    for name, layer in zip(names, layers):
        layer.host_history = store.layer(name)
    logger.info(
        "Long-context main K/V in host memory: %d history blocks x %d layers "
        "(%.2f GiB pinned on this worker), hot pool %d groups per layer, %d "
        "query rows per gather pass.",
        shape.num_blocks,
        shape.num_layers,
        shape.num_blocks * shape.page_bytes / (1 << 30),
        shape.hot_slots,
        shape.rows_per_pass,
    )
    return store
