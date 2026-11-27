# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""get_manager removes the CPU offload region name right after mapping (#51)."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

pytestmark = pytest.mark.cpu_test
WORLD = 2
BLOCKS = 4
CPU_PAGE = 4096


def _path(instance_id: str) -> str:
    return f"/dev/shm/vllm_offload_{instance_id}.mmap"


def _region(instance_id: str, rank: int) -> SharedOffloadRegion:
    # Same constructor the spec uses on workers (create_handlers).
    return SharedOffloadRegion(
        instance_id=instance_id,
        total_size_bytes=CPU_PAGE * WORLD * BLOCKS,
        num_blocks=BLOCKS,
        rank=rank,
        num_workers=WORLD,
        cpu_page_size=CPU_PAGE,
    )


def _spec(instance_id: str, partition_by_group: bool):
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    spec = object.__new__(TieringOffloadingSpec)
    spec.partition_by_group = partition_by_group
    spec.unified_host = False
    spec._manager = None
    spec._scheduler_mmap = None
    spec.vllm_config = SimpleNamespace(
        instance_id=instance_id,
        parallel_config=SimpleNamespace(world_size=WORLD),
        kv_events_config=None,
    )
    spec.cpu_page_size_per_worker = CPU_PAGE
    spec.num_blocks = BLOCKS
    spec.eviction_policy = "lru"
    spec.secondary_tier_configs = []
    spec.extra_config = {}
    if partition_by_group:
        spec.cpu_group_page_sizes = {0: CPU_PAGE, 1: CPU_PAGE // 2}
        spec.cpu_group_num_blocks = {g: BLOCKS for g in (0, 1)}
    return spec


@pytest.mark.parametrize("partition_by_group", [False, True])
def test_get_manager_unlinks_regions_immediately(partition_by_group):
    """After get_manager the /dev/shm names are gone and the worker-side
    mappings still read data the scheduler side wrote."""
    instance_id = f"unlink-mgr-{uuid4().hex}"
    spec = _spec(instance_id, partition_by_group)
    if partition_by_group:
        workers = [spec._create_group_regions(rank) for rank in range(WORLD)]
        paths = [_path(f"{instance_id}_g{group}") for group in (0, 1)]
    else:
        workers = [
            {0: _region(instance_id, rank)}
            for rank in range(WORLD)  # type: ignore[dict-item]
        ]
        paths = [_path(instance_id)]
    try:
        manager = spec.get_manager()
        try:
            for path in paths:
                assert not Path(path).exists()
            for group in (0, 1) if partition_by_group else (0,):
                page = (
                    spec.cpu_group_page_sizes[group] if partition_by_group else CPU_PAGE
                )
                row_stride = page * WORLD
                tier = (
                    manager.managers[group].primary_tier
                    if partition_by_group
                    else manager.primary_tier
                )
                view = tier.get_kv_memoryview().cast("B")
                # Scheduler -> worker, one direction, block 1.
                start = row_stride
                view[start : start + page] = bytes([group + 1]) * page
                for rank in range(WORLD):
                    raw = bytes(workers[rank][group].mmap_obj[start : start + page])
                    assert raw == bytes([group + 1]) * page
        finally:
            manager.shutdown()
    finally:
        for regions in workers:
            for region in regions.values():
                region.cleanup()
        for path in paths:
            Path(path).unlink(missing_ok=True)
