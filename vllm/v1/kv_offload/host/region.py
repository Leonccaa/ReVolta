# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical host-page ABI, independent of padded GPU index allocations."""

from dataclasses import dataclass, replace
from pathlib import Path

from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


@dataclass(frozen=True)
class HostRegionLayout:
    instance_id: str
    num_pages: int
    num_workers: int
    main_bytes: int
    index_bytes: tuple[int, ...]

    @property
    def worker_bytes(self) -> int:
        return self.main_bytes + sum(self.index_bytes)

    def open(self, rank: int | None) -> SharedOffloadRegion:
        return SharedOffloadRegion(
            f"host_v2_{self.instance_id}",
            self.num_pages * self.num_workers * self.worker_bytes,
            self.num_pages,
            rank,
            self.num_workers,
            self.worker_bytes,
            scheduler_only_unlink=True,
        )


class HostRegionOwner:
    """Parent-only unlink responsibility, established before worker startup.

    Owning the name does not map, populate or register the backing. Local
    worker mappings close independently. Call cleanup only after every worker
    access has retired, or after startup rollback confirmed worker exit.
    """

    def __init__(self, layout: HostRegionLayout):
        self.mmap_path = f"/dev/shm/vllm_offload_host_v2_{layout.instance_id}.mmap"

    def cleanup(self) -> None:
        Path(self.mmap_path).unlink(missing_ok=True)


def host_region_layout(config: KVCacheConfig, instance_id: str) -> HostRegionLayout:
    cached = getattr(config, "_host_region_layout", None)
    if isinstance(cached, HostRegionLayout):
        return replace(cached, instance_id=instance_id)
    plan = config.long_context_plan
    assert plan is not None and config.host_kv_layers
    # Only main-group logical index members belong in the seal payload. Ring
    # aliases share their allocation, but are never host history pages.
    names = set(config.host_kv_layers)
    groups = [g for g in config.kv_cache_groups if names.intersection(g.layer_names)]
    if len(groups) != 1:
        raise ValueError("Host pages require one main KV group")
    group = groups[0]
    spec = group.kv_cache_spec
    sizes = tuple(
        (
            spec.kv_cache_specs[name]
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else spec
        ).page_size_bytes
        for name in group.layer_names
        if name not in names
    )
    worker_bytes = config.host_kv_page_size_bytes + sum(sizes)
    num_pages = plan.host_pool_bytes // (worker_bytes * plan.geometry.num_workers)
    if num_pages < 2:
        raise ValueError("Host pool needs ZERO_PAGE and at least one data page")
    return HostRegionLayout(
        instance_id,
        num_pages,
        plan.geometry.num_workers,
        config.host_kv_page_size_bytes,
        sizes,
    )
