# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Collection, Iterator, Mapping

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


def mamba_state_slots(
    num_token_blocks: int,
    block_size: int,
    alignment_tokens: int,
    mamba_spec: MambaSpec,
    retention_interval: int | None,
    reference_tokens: int,
) -> int:
    """Recurrent-state slots needed alongside ``num_token_blocks`` token slots.

    The demand is read off the GPU prefix cache's ``reachable_block_mask``:
    for a request of ``reference_tokens`` tokens, count the boundary states
    the mask retains (its replay boundaries plus periodic checkpoints), allow
    one shared-prefix junction per request, and multiply by the number of such
    requests the token pool can hold. Mamba ``align`` hand-offs only carry
    hashed states, so the host tier never holds more states than the GPU
    mask admits. Dense retention keeps every boundary and therefore one
    state slot per token slot.
    """
    if retention_interval is None:
        return num_token_blocks
    blocks_per_request = max(1, cdiv(reference_tokens, block_size))
    mask = MambaManager.reachable_block_mask(
        0,
        blocks_per_request,
        alignment_tokens,
        mamba_spec,
        retention_interval,
        (max(reference_tokens - 1, 0), reference_tokens),
    )
    states_per_request = blocks_per_request if mask is None else sum(mask)
    # A request may additionally pin one shared-prefix junction state.
    states_per_request += 1
    requests_in_pool = max(1, cdiv(num_token_blocks, blocks_per_request))
    return min(num_token_blocks, requests_in_pool * states_per_request)


def group_slot_counts(
    spans: Mapping[int, int],
    recurrent_groups: Collection[int],
    token_slots: int,
    reference_span: int,
    state_slots: Callable[[int, int], int] | None = None,
) -> dict[int, int]:
    """Per-group CPU slot counts that keep one token coverage across groups.

    ``token_slots`` counts blocks of ``reference_span`` tokens, i.e. the
    offloaded block of the widest token group. A group whose own offloaded
    block covers ``span`` tokens gets ``token_slots * reference_span // span``
    slots, so every token group holds the same number of tokens. When all
    spans equal ``reference_span`` the count is exactly ``token_slots``, which
    is the single-``num_blocks`` sizing the grouped layout used before group
    spans could differ. Recurrent groups replace their count with
    ``state_slots(group, own_slots)``, the boundary states the prefix-cache
    retention mask needs at that same coverage.
    """
    if reference_span <= 0:
        raise ValueError(f"reference_span must be positive, got {reference_span}")
    counts: dict[int, int] = {}
    for group, span in spans.items():
        if span <= 0:
            raise ValueError(f"group {group} has a non-positive token span {span}")
        slots = max(1, token_slots * reference_span // span)
        if state_slots is not None and group in recurrent_groups:
            slots = state_slots(group, slots)
        counts[group] = slots
    return counts


class CPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        cacheable_groups = {
            i: group
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if group.kv_cache_spec.prefix_cacheable
        }
        mamba_specs = {}
        for i, group in cacheable_groups.items():
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()))
            if isinstance(spec, MambaSpec):
                mamba_specs[i] = spec
        # Rollback knob and the phase-3 widening: "auto" partitions whenever a
        # recurrent group exists (the historical trigger), "off" always keeps
        # the row-per-key layout, "force" also partitions multi-group models
        # without a recurrent group.
        partition_mode = str(self.extra_config.get("cpu_group_partition", "auto"))
        if partition_mode not in ("auto", "force", "off"):
            raise ValueError(
                "cpu_group_partition must be 'auto', 'force' or 'off', got "
                f"{partition_mode!r}"
            )
        # Private per-group pools charge each group its own page instead of one
        # shared row per key. Group token spans may differ (a DFlash drafter
        # block is a divisor of the target block), so slot counts are solved per
        # span. Keep the existing layout for single groups and no-recurrent models.
        self.partition_by_group = (
            partition_mode != "off"
            and kv_cache_config.num_blocks > 0
            and len(cacheable_groups) > 1
            and bool(mamba_specs or partition_mode == "force")
        )
        # Mamba state slots follow the prefix-cache retention policy (see
        # CacheConfig.prefix_cache_retention_interval): sparse retention keeps
        # only reachable boundary states, so the state pools are sized from the
        # retention mask for a reference request length instead of one slot
        # per token block. The reference defaults to the model's max length;
        # workloads dominated by shorter prompts can lower it.
        self.retention_interval = (
            vllm_config.cache_config.prefix_cache_retention_interval
        )
        self.state_slots_reference_tokens = int(
            self.extra_config.get(
                "mamba_state_slots_reference_tokens",
                vllm_config.model_config.max_model_len,
            )
        )
        if self.state_slots_reference_tokens <= 0:
            raise ValueError("mamba_state_slots_reference_tokens must be positive")

        self.cpu_group_page_sizes: dict[int, int] = {}
        self.cpu_group_num_blocks: dict[int, int] = {}
        self.cpu_group_offloaded_tokens: dict[int, int] = {}
        world_size = vllm_config.parallel_config.world_size
        if self.partition_by_group:
            mamba_groups: set[int] = set(mamba_specs)
            # Tokens covered by one offloaded block of each group, the same
            # quantity the connector calls ``offloaded_block_size``.
            spans = {
                i: self.gpu_block_size[i] * self.block_size_factor
                for i in cacheable_groups
            }
            # Every recurrent group is sized from its own MambaSpec, but they
            # must agree on one geometry: the connector hands states over at a
            # single mamba align size and the retention mask counts blocks of
            # that spec, so mixed geometry has to fail loudly instead of
            # silently sizing every state pool from the last spec seen.
            geometries = {
                (
                    spans[i],
                    spec.block_size * self.block_size_factor,
                    spec.mamba_cache_mode,
                )
                for i, spec in mamba_specs.items()
            }
            if len(geometries) > 1:
                raise ValueError(
                    "Grouped CPU offload needs one state slot geometry across "
                    "all recurrent groups (offloaded token span, mamba "
                    "block_size x block_size_factor and mamba_cache_mode); got "
                    f"{sorted(geometries)}."
                )
            for i, group in cacheable_groups.items():
                layer_names = set(group.layer_names)
                # Scheduler specs can replace a heterogeneous UniformType spec
                # by its largest layer spec. Physical tensor sizes/shared_by are
                # preserved and give identical budgets on scheduler and workers.
                page_size = sum(
                    tensor.size // kv_cache_config.num_blocks
                    for tensor in kv_cache_config.kv_cache_tensors
                    if not layer_names.isdisjoint(tensor.shared_by)
                )
                self.cpu_group_page_sizes[i] = page_size * self.block_size_factor
            # Physical storage contains only that group's tensors, rather than
            # every shared GPU tensor for each independently allocated group key.
            self.cpu_page_size_per_worker = sum(self.cpu_group_page_sizes.values())
            token_groups = [
                i for i in self.cpu_group_page_sizes if i not in mamba_groups
            ]
            token_page = sum(self.cpu_group_page_sizes[i] for i in token_groups)
            state_page = sum(self.cpu_group_page_sizes[i] for i in mamba_groups)
            budget_per_worker = int(cpu_bytes_to_use) // world_size
            # The pool covers ``num_blocks`` blocks of the widest token group;
            # every other group gets the slot count for the same token count.
            reference_span = max((spans[i] for i in token_groups), default=0) or max(
                spans.values()
            )

            def state_slots(group: int, num_token_blocks: int) -> int:
                spec = mamba_specs[group]
                block_size = spec.block_size * self.block_size_factor
                return mamba_state_slots(
                    num_token_blocks,
                    block_size,
                    block_size,
                    spec,
                    self.retention_interval,
                    self.state_slots_reference_tokens,
                )

            def slot_counts(num_token_blocks: int) -> dict[int, int]:
                return group_slot_counts(
                    spans, mamba_groups, num_token_blocks, reference_span, state_slots
                )

            def bytes_per_worker(num_token_blocks: int) -> int:
                return sum(
                    self.cpu_group_page_sizes[group] * slots
                    for group, slots in slot_counts(num_token_blocks).items()
                )

            # Largest token-block count whose token pages plus the state slots
            # the retention mask demands fit the per-worker budget. The search
            # starts from the bytes a reference-span block costs when no group
            # needs extra slots, which is the legacy estimate for equal spans.
            num_blocks = 0
            if token_page + state_page > 0:
                per_reference_block = sum(
                    page * (reference_span // spans[group])
                    for group, page in self.cpu_group_page_sizes.items()
                )
                if per_reference_block <= 0:
                    per_reference_block = max(bytes_per_worker(1), 1)
                num_blocks = budget_per_worker // per_reference_block
                while bytes_per_worker(num_blocks + 1) <= budget_per_worker:
                    num_blocks += 1
                while (
                    num_blocks > 0 and bytes_per_worker(num_blocks) > budget_per_worker
                ):
                    num_blocks -= 1
            self.num_blocks = num_blocks
            self.cpu_group_offloaded_tokens = dict(spans)
            self.cpu_group_num_blocks = slot_counts(num_blocks)
            bytes_per_token = (
                world_size
                * sum(
                    self.cpu_group_page_sizes[group] / spans[group]
                    for group in token_groups
                )
                if token_groups
                else 0.0
            )
            logger.info(
                "Grouped CPU offload pool: %d slots x %d tokens of the widest "
                "token group (%d tokens covered); %d recurrent group(s), %d "
                "state slots each. Token pages %d B, state pages %d B per slot "
                "set per worker, budget %d B per worker x %d workers, "
                "retention=%s, state reference tokens=%d, token-proportional "
                "cost %.0f B/token.",
                num_blocks,
                reference_span,
                num_blocks * reference_span,
                len(mamba_groups),
                min(
                    (self.cpu_group_num_blocks[g] for g in mamba_groups),
                    default=0,
                ),
                token_page,
                state_page,
                budget_per_worker,
                world_size,
                self.retention_interval,
                self.state_slots_reference_tokens,
                bytes_per_token,
            )
            for group in sorted(self.cpu_group_page_sizes):
                logger.info(
                    "Grouped CPU offload group %d: page %d B per worker, %d "
                    "slots x %d tokens per slot = %.2f GiB across %d workers "
                    "(%s group, %.0f B per covered token).",
                    group,
                    self.cpu_group_page_sizes[group],
                    self.cpu_group_num_blocks[group],
                    spans[group],
                    self.cpu_group_page_sizes[group]
                    * world_size
                    * self.cpu_group_num_blocks[group]
                    / 2**30,
                    world_size,
                    "recurrent" if group in mamba_groups else "token",
                    world_size * self.cpu_group_page_sizes[group] / spans[group],
                )
        else:
            total_gpu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            self.cpu_page_size_per_worker = (
                total_gpu_kv_bytes
                // kv_cache_config.num_blocks
                * self.block_size_factor
                if kv_cache_config.num_blocks > 0
                else 0
            )
            kv_bytes_per_offloaded_block = self.cpu_page_size_per_worker * world_size
            self.num_blocks = (
                int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
                if kv_bytes_per_offloaded_block > 0
                else 0
            )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            def create_manager(num_blocks: int) -> CPUOffloadingManager:
                return CPUOffloadingManager(
                    num_blocks=num_blocks,
                    cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                    enable_events=enable_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )

            self._manager = (
                GroupedCPUOffloadingManager(
                    {
                        i: create_manager(num_blocks)
                        for i, num_blocks in self.cpu_group_num_blocks.items()
                    }
                )
                if self.partition_by_group
                else create_manager(self.num_blocks)
            )
        return self._manager

    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuGpuOffloadingHandlers:
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            group_page_sizes=(
                self.cpu_group_page_sizes if self.partition_by_group else None
            ),
            group_num_blocks=(
                self.cpu_group_num_blocks if self.partition_by_group else None
            ),
        )

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike GPUs"
                )
            self._handlers = self.create_handlers(kv_caches)

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
