# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingSpec: Spec for multi-tier KV cache offloading.

This spec creates a TieringOffloadingManager with a CPU primary tier
and configurable secondary tiers (e.g., Storage, Network).

Configuration via kv_connector_extra_config:
  - cpu_bytes_to_use: (required) Bytes to allocate for CPU primary tier
  - block_size: (optional) Block size for offloaded blocks (default: GPU block size)
  - cpu_group_partition: (optional) "auto" (default) gives every cache group its
    own pool and tier pages when the model has a recurrent group and more than
    one prefix-cacheable group; groups with a narrower cache block then get more
    slots so all groups cover the same tokens. "off" keeps one shared row per
    key, "force" also partitions multi-group models without a recurrent group.
  - eviction_policy: (optional) Primary tier eviction policy: "lru" or
    "arc" (default: "lru")
  - secondary_tiers: (optional) List of secondary tier configurations
    Each secondary tier config is a dict with:
      - type: (required) Type of secondary tier (e.g., "example", "storage", "network")
      - Additional tier-specific parameters are passed directly to the tier
        constructor. See each tier's documentation for supported parameters.

Example configuration:
{
    "cpu_bytes_to_use": 10737418240,  # 10 GB
    "block_size": 16,
    "eviction_policy": "lru",
    "secondary_tiers": [
        {
            "type": "example",
            "custom_param": 67
        }
    ]
}
"""

from contextlib import ExitStack

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.manager import GroupedCPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

logger = init_logger(__name__)


class TieringOffloadingSpec(CPUOffloadingSpec):
    """
    Spec for multi-tier KV cache offloading.

    Creates a TieringOffloadingManager with:
    - Primary tier: CPU (LRU or ARC eviction policy)
    - Secondary tiers: Configurable via extra_config

    The CPU primary tier has direct GPU access and serves as the gateway for
    all GPU↔offload operations. Secondary tiers cannot directly access GPU
    memory and must transfer data through the primary tier.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        # Redeclare for mypy: parent sets this but `--follow-imports skip` hides it
        self._manager: OffloadingManager | None = None

        # Parse secondary tier configurations
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        self.persistent_layout: dict | None = None
        if self.partition_by_group:
            parallel = vllm_config.parallel_config
            # Decode context parallelism needs nothing extra: every rank
            # registers and moves only its own pages, one row slice per
            # worker, and group spans are already global block sizes.
            if (
                parallel.pipeline_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.nnodes != 1
            ):
                raise ValueError(
                    "Grouped tiering currently requires single-node TP, "
                    "optionally with decode context parallelism"
                )
            backend = vllm_config.attention_config.backend
            if backend is None:
                raise ValueError(
                    "Grouped tiering requires an explicit attention backend"
                )
            group_tokens = dict(
                sorted(getattr(self, "cpu_group_offloaded_tokens", {}).items())
            )
            self.persistent_layout = {
                "version": "grouped-worker-interleaved-v1",
                "config_hash": vllm_config.compute_hash(),
                "attention_backend": str(backend),
                "model_revision": vllm_config.model_config.revision,
                "group_pages": self.cpu_group_page_sizes,
                "tensors": [
                    {
                        "page_size": tensor.size // kv_cache_config.num_blocks,
                        "shared_by": list(tensor.shared_by),
                    }
                    for tensor in kv_cache_config.kv_cache_tensors
                ],
            }
            if len(set(group_tokens.values())) > 1:
                # Groups now write pages of different sizes, so their files must
                # not share the v1 namespace where every key paid one row. The
                # equal-span (e.g. MTP4) dict stays byte-for-byte identical.
                self.persistent_layout["version"] = "grouped-worker-interleaved-v2"
                self.persistent_layout["group_tokens"] = group_tokens

        from vllm.v1.attention.kv_cache_view import (
            has_planar_kv_cache,
            kv_storage_metadata,
        )

        if has_planar_kv_cache(kv_cache_config):
            if self.persistent_layout is None:
                self.persistent_layout = {"config_hash": vllm_config.compute_hash()}
            self.persistent_layout["kv_storage"] = kv_storage_metadata(
                kv_cache_config, vllm_config.cache_config.cache_dtype
            )
            if any(tier.get("type") == "fs" for tier in self.secondary_tier_configs):
                raise ValueError(
                    "INT8 v1 SSD requires aligned FS framing; raw O_DIRECT pages "
                    "are unsupported. Use CPU offload until FS framing is integrated."
                )

        # Scheduler-side mmap (rank=None); kept for cleanup
        self._scheduler_mmap: SharedOffloadRegion | None = None

    def _create_group_regions(self, rank: int | None) -> dict[int, SharedOffloadRegion]:
        """Use the same group names and row geometry on scheduler and workers."""
        world_size = self.vllm_config.parallel_config.world_size
        regions: dict[int, SharedOffloadRegion] = {}
        try:
            for group, page_size in self.cpu_group_page_sizes.items():
                num_blocks = self.cpu_group_num_blocks[group]
                regions[group] = SharedOffloadRegion(
                    instance_id=f"{self.vllm_config.instance_id}_g{group}",
                    total_size_bytes=page_size * world_size * num_blocks,
                    num_blocks=num_blocks,
                    rank=rank,
                    num_workers=world_size,
                    cpu_page_size=page_size,
                )
        except Exception:
            for region in regions.values():
                region.cleanup()
            raise
        return regions

    def _tier_thread_summary(self) -> str:
        """Configured per-tier I/O threads, for the startup log only."""
        parts = []
        for config in self.secondary_tier_configs:
            if not isinstance(config, dict):
                continue
            read = config.get("n_read_threads")
            write = config.get("n_write_threads")
            if read is None and write is None:
                parts.append(f"{config.get('type', '?')}=default threads")
                continue
            parts.append(f"{config.get('type', '?')}={read or 0}r+{write or 0}w")
        return "; ".join(parts) if parts else "no secondary tier"

    def _get_grouped_manager(self, enable_events: bool) -> OffloadingManager:
        if int(self.extra_config.get("store_threshold", 0)) >= 2:
            raise ValueError(
                "store_threshold is not supported for TieringOffloadingSpec"
            )
        world_size = self.vllm_config.parallel_config.world_size
        thread_summary = self._tier_thread_summary()
        # ExitStack unwinds tiers before primary views, including partial setup.
        with ExitStack() as cleanup:
            regions = self._create_group_regions(None)
            for region in regions.values():
                cleanup.callback(region.cleanup)
            managers: dict[int, OffloadingManager] = {}
            for group, region in regions.items():
                primary = CPUPrimaryTierOffloadingManager(
                    num_blocks=self.cpu_group_num_blocks[group],
                    cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                    enable_events=enable_events,
                    mmap_region=region,
                )
                cleanup.callback(primary.shutdown)
                tiers = []
                for config in self.secondary_tier_configs:
                    tier = SecondaryTierFactory.create_secondary_tier(
                        config, primary.get_kv_memoryview(), self
                    )
                    cleanup.callback(tier.shutdown)
                    tiers.append(tier)
                managers[group] = TieringOffloadingManager(
                    primary_tier=primary,
                    secondary_tiers=tiers,
                    enable_events=enable_events,
                )
                logger.info(
                    "Grouped tier group %d: page %d B per worker x %d slots x %d "
                    "workers = %.2f GiB primary tier, %d tokens per slot, %d "
                    "secondary tier(s) (%s).",
                    group,
                    self.cpu_group_page_sizes[group],
                    self.cpu_group_num_blocks[group],
                    world_size,
                    self.cpu_group_page_sizes[group]
                    * world_size
                    * self.cpu_group_num_blocks[group]
                    / 2**30,
                    getattr(self, "cpu_group_offloaded_tokens", {}).get(group, 0),
                    len(tiers),
                    thread_summary,
                )
            manager = GroupedCPUOffloadingManager(managers)
            cleanup.pop_all()
            return manager

    @override
    def get_manager(self) -> OffloadingManager:
        """
        Get the TieringOffloadingManager.

        Creates a TieringOffloadingManager with:
        - Primary tier: CPU (LRU or ARC)
        - Secondary tiers: As configured in extra_config

        Returns:
            TieringOffloadingManager instance
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            if self.partition_by_group:
                self._manager = self._get_grouped_manager(enable_events)
                return self._manager

            # Create scheduler-side SharedOffloadRegion (rank=None) so the
            # primary tier can eagerly create a memoryview over _base.
            world_size = self.vllm_config.parallel_config.world_size
            scheduler_mmap = SharedOffloadRegion(
                instance_id=self.vllm_config.instance_id,
                total_size_bytes=self.cpu_page_size_per_worker
                * world_size
                * self.num_blocks,
                num_blocks=self.num_blocks,
                rank=None,
                num_workers=world_size,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
            self._scheduler_mmap = scheduler_mmap

            # Create primary tier (CPU-based)
            # The unpartitioned tier stores a full worker KV row for each
            # offload key. Each group can address that row with its own token
            # block size; the scheduler and transfer handler retain the group
            # geometry separately. DCP-sharded attention and replicated
            # recurrent groups therefore do not require equal block sizes.
            primary_tier = CPUPrimaryTierOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
                mmap_region=scheduler_mmap,
            )

            # Create secondary tiers
            primary_kv_view = primary_tier.get_kv_memoryview()
            secondary_tiers = []
            for i, tier_config in enumerate(self.secondary_tier_configs):
                try:
                    tier = SecondaryTierFactory.create_secondary_tier(
                        tier_config, primary_kv_view, self
                    )
                    secondary_tiers.append(tier)
                    logger.info(
                        "Created secondary tier #%d (%s)",
                        i,
                        tier.tier_type,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create secondary tier from config %s: %s",
                        tier_config,
                        e,
                    )
                    raise

            # Create TieringOffloadingManager. GPU↔CPU transfers use the inherited
            # get_handlers(); secondary tier transfers are handled by the
            # secondary tier managers and need no additional handlers here.
            tiering_manager = TieringOffloadingManager(
                primary_tier=primary_tier,
                secondary_tiers=secondary_tiers,
                enable_events=enable_events,
            )
            if int(self.extra_config.get("store_threshold", 0)) >= 2:
                raise ValueError(
                    "store_threshold is not supported for TieringOffloadingSpec"
                )
            self._manager = tiering_manager

            logger.info(
                "Created TieringOffloadingManager with primary tier "
                "(%s, %s blocks) and %s secondary tier(s)",
                self.eviction_policy,
                self.num_blocks,
                len(secondary_tiers),
            )

        return self._manager

    @override
    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuGpuOffloadingHandlers:
        world_size = self.vllm_config.parallel_config.world_size
        rank = torch.accelerator.current_device_index()
        if self.partition_by_group:
            return CpuGpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
                group_page_sizes=self.cpu_group_page_sizes,
                group_mmap_regions=self._create_group_regions(rank),
                group_num_blocks=self.cpu_group_num_blocks,
            )
        worker_mmap = SharedOffloadRegion(
            instance_id=self.vllm_config.instance_id,
            total_size_bytes=self.cpu_page_size_per_worker
            * world_size
            * self.num_blocks,
            num_blocks=self.num_blocks,
            rank=rank,
            num_workers=world_size,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            mmap_region=worker_mmap,
        )
