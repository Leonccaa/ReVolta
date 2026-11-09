# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-layer host authority with a bounded, request/group-keyed GPU cache.

This module owns its allocations, not scheduler block IDs. A CPU allocation is
an explicit interpreter fixture, never a pageable substitute for CUDA's pinned
pool. Construction, host inspection and admission are outside CUDA graphs;
write/gather/warm/truncate have fixed grids and never read a device scalar.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from vllm.triton_utils import triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from . import kernels


class UnsupportedHostKVConfig(ValueError):
    """Valid dimensions exceed this prototype's capacity or address limits."""


class HostKVResourceError(RuntimeError):
    """Allocation or UVA initialization failed for an admitted configuration."""


def _check_product(name: str, *sizes: int, limit: int = 2**31 - 1) -> None:
    if math.prod(sizes) > limit:
        raise UnsupportedHostKVConfig(f"{name} exceeds the supported limit {limit}")


def _check_storage(tensor: torch.Tensor, *, index_limit: int = 2**63 - 1) -> None:
    # Check bytes as well as element offsets before any pointer arithmetic.
    extent = 1 + sum((n - 1) * abs(s) for n, s in zip(tensor.shape, tensor.stride()))
    _check_product("strided tensor elements", extent, limit=index_limit)
    _check_product(
        "strided tensor bytes", extent, tensor.element_size(), limit=2**63 - 1
    )


@dataclass(frozen=True)
class LayerScales:
    """Calibrated per-layer K/V scalars, outside the dense E4M3 payload."""

    key: float = 1.0
    value: float = 1.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(x) or x <= 0 for x in (self.key, self.value)):
            raise ValueError("Host KV scales must be finite and positive")


@dataclass(frozen=True)
class HostKVConfig:
    """Fixed admission geometry and eviction policy for one layer.

    Set qsa_block_topk for selector-expanded input, with four columns per
    selected block. DCP2 interleave 1/2 compaction uses qsa_group_columns=2
    and either the full padded buffer or its narrowed prefix. Gather checks
    that each column chunk fits one local four-token hot group. Omitting
    qsa_block_topk retains the arbitrary-column capacity contract.
    """

    requests: int
    max_tokens: int
    rows: int
    selection_width: int
    max_write_rows: int
    hot_groups: int | None = None
    hot_ratio: float | None = None
    max_misses: int | None = None
    qsa_block_topk: int | None = None
    qsa_group_columns: int = 4
    eviction_policy: str = "bucket"

    @property
    def row_groups(self) -> int:
        # Arbitrary columns retain their original bound. The QSA contract is
        # opt-in because width alone cannot establish the selector's layout.
        if self.qsa_block_topk is None:
            return self.selection_width
        return self.qsa_block_topk + 1

    @property
    def groups_per_request(self) -> int:
        return triton.cdiv(self.max_tokens, 4)

    @property
    def required_groups(self) -> int:
        """Union bound in four-token groups, including unaligned tails.

        QSA expands each selected block to four columns and appends at most
        three columns from one open group. Thus 512 blocks touch at most 513
        groups per row. Rows may select disjoint groups, even for one request.
        """
        g = self.groups_per_request
        return min(
            self.requests * g,
            self.rows * min(self.row_groups, g) + self.requests * min(9, g),
        )

    def capacities(self) -> tuple[int, int]:
        dimensions = (
            self.requests,
            self.max_tokens,
            self.rows,
            self.selection_width,
            self.max_write_rows,
        )
        if any(type(n) is not int or n <= 0 for n in dimensions):
            raise ValueError("Host KV dimensions must be positive")
        if type(self.qsa_group_columns) is not int or self.qsa_group_columns not in (
            2,
            4,
        ):
            raise UnsupportedHostKVConfig("QSA group columns must be 2 or 4")
        if self.qsa_block_topk is not None:
            if (
                type(self.qsa_block_topk) is not int
                or not 1 <= self.qsa_block_topk <= 512
            ):
                raise UnsupportedHostKVConfig("QSA requires 1..512 blocks")
            widths = {4 * self.qsa_block_topk + 3}
            if self.qsa_group_columns == 2:
                widths.add((self.qsa_block_topk + 1) * 2)
            if self.selection_width not in widths:
                raise UnsupportedHostKVConfig(
                    "QSA requires the expanded or DCP2 selection width"
                )
        if self.eviction_policy not in ("bucket", "lru"):
            raise UnsupportedHostKVConfig("eviction_policy must be bucket or lru")
        for name, sizes in (
            ("requests * max_tokens", (self.requests, self.max_tokens)),
            ("rows * selection_width", (self.rows, self.selection_width)),
            ("max_write_rows", (self.max_write_rows,)),
            ("selection grid Y", (triton.cdiv(self.selection_width, 128),)),
        ):
            limit = 65535 if name == "selection grid Y" else 2**31 - 1
            _check_product(name, *sizes, limit=limit)
        if (self.hot_groups is None) == (self.hot_ratio is None):
            raise ValueError("Specify exactly one of hot_groups and hot_ratio")
        total = self.requests * self.groups_per_request
        if self.hot_ratio is not None:
            if not math.isfinite(self.hot_ratio) or not 0 < self.hot_ratio <= 1:
                raise ValueError("hot_ratio must be in (0, 1]")
            slots = math.ceil(total * self.hot_ratio)
        else:
            assert self.hot_groups is not None
            slots = self.hot_groups
        misses = self.required_groups if self.max_misses is None else self.max_misses
        if any(type(n) is not int or n <= 0 for n in (slots, misses)):
            raise ValueError("Host KV capacities must be positive integers")
        if not self.required_groups <= slots <= total:
            raise UnsupportedHostKVConfig(
                f"hot capacity must be between {self.required_groups} and {total} "
                "groups for this selection shape (including the 32-token tail); "
                "reduce the query batch or increase capacity"
            )
        if not self.required_groups <= misses <= slots:
            raise UnsupportedHostKVConfig(
                "max_misses must cover the full cold-step union bound"
            )
        _check_product("hot token coordinates", slots, 4)
        _check_product("requests * hot_groups", self.requests, slots)
        return slots, misses


class ActiveHostKV:
    """Owned host pages and GPU replicas for one layer on one device.

    ``layout_like`` is metadata only: [physical page, 2, token, KV head, dim].
    Its positive, nonoverlapping strides are preserved, including packed-layer
    holes and HND layouts. No adjacent layer is copied or owned. ``block_table``
    is immutable CPU int32 metadata with disjoint physical pages per request.
    Shared-prefix/COW and scheduler identities belong to host-history
    integration/complete-prefix contract.

    Calls must be serialized on one CUDA stream. A completed write kernel is
    followed by a ready-flag kernel on that stream. Readers cannot consume an
    uncommitted token. CPU inspection or release of this object requires the
    caller to wait for that stream; there is no hidden synchronization here.
    Consume each gather's attention before any subsequent pool mutation.
    """

    def __init__(
        self,
        config: HostKVConfig,
        layout_like: torch.Tensor,
        *,
        device: torch.device | str,
        scales: LayerScales | None = None,
        block_table: torch.Tensor | None = None,
    ) -> None:
        slots, misses = config.capacities()
        scales = LayerScales() if scales is None else scales
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("The prototype supports CUDA or the CPU interpreter")
        if device.type == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
            raise ValueError("CPU fixtures require TRITON_INTERPRET=1")
        if (
            layout_like.ndim != 5
            or layout_like.shape[1] != 2
            or layout_like.stride(-1) != 1
            or layout_like.dtype not in (torch.uint8, torch.float16)
        ):
            raise ValueError(
                "Expected a dense FP16/E4M3 [page,2,token,head,dim] carrier"
            )
        pages, _, page_tokens, heads, dim = layout_like.shape
        if min(layout_like.shape) <= 0 or page_tokens % 4:
            raise ValueError(
                "Host pages must contain a positive multiple of four tokens"
            )
        # Positive nonoverlapping strides; permit both NHD and HND, plus holes.
        extent = 1
        for stride, size in sorted(zip(layout_like.stride(), layout_like.shape)):
            if stride <= 0 or (size > 1 and stride < extent):
                raise ValueError("Host KV carrier has overlapping or invalid strides")
            extent += (size - 1) * stride
        columns = triton.cdiv(config.max_tokens, page_tokens)
        _check_storage(layout_like)
        _check_product("requests * host table columns", config.requests, columns)
        _check_product("physical page coordinates", pages)
        _check_product("group transfer elements", 8, heads, dim, limit=16384)
        for name, shape in (
            ("hot payload bytes", (slots, 8, heads, dim)),
            ("encoding scratch bytes", (config.max_write_rows, 2, heads, dim)),
        ):
            _check_product(name, *shape, layout_like.element_size(), limit=2**63 - 1)
        if block_table is None and config.requests * columns > pages:
            raise ValueError("Host carrier has too few physical pages")
        if block_table is None:
            try:
                block_table = torch.arange(config.requests * columns, dtype=torch.int32)
                block_table = block_table.reshape(config.requests, columns)
            except (RuntimeError, MemoryError) as error:
                raise HostKVResourceError(
                    "Host block table allocation failed"
                ) from error
        if (
            block_table.device.type != "cpu"
            or block_table.dtype != torch.int32
            or block_table.shape != (config.requests, columns)
        ):
            raise ValueError("block_table must be CPU int32 [requests, host pages]")
        ids = block_table.flatten().tolist()
        if min(ids) < 0 or max(ids) >= pages or len(set(ids)) != len(ids):
            raise ValueError(
                "Host block table requires distinct in-range physical pages"
            )

        self.config, self.scales = config, scales
        self.slots, self.max_misses = slots, misses
        self.page_tokens, self.columns = page_tokens, columns
        self.heads, self.dim = heads, dim
        self.is_e4m3 = layout_like.dtype == torch.uint8
        try:
            self.host = torch.empty_strided(
                layout_like.shape,
                layout_like.stride(),
                dtype=layout_like.dtype,
                device="cpu",
                pin_memory=device.type == "cuda",
            )
            self.hot = torch.empty(
                (slots, 2, 4, heads, dim), dtype=layout_like.dtype, device=device
            )
            self.device = self.hot.device
            if device.type == "cuda":
                # The existing native helper maps the entire strided storage.
                # Keep the pinned allocation alive for as long as its alias is used.
                with torch.accelerator.device_index(self.device.index):
                    self.host_device = get_accelerator_view_from_cpu_tensor(self.host)
            else:
                self.host_device = self.host
            self.block_table = block_table.to(self.device, copy=True).contiguous()
            self.hot_block_table = torch.arange(
                slots, dtype=torch.int32, device=self.device
            )
            self.hot_block_table = self.hot_block_table.repeat(config.requests, 1)
            r, t, g = config.requests, config.max_tokens, config.groups_per_request
            self.group_to_slot = torch.full(
                (r, g), -1, dtype=torch.int32, device=self.device
            )
            self.slot_owner = torch.full(
                (slots,), -1, dtype=torch.int32, device=self.device
            )
            self.age = torch.zeros(slots, dtype=torch.int64, device=self.device)
            self.clock = torch.zeros((), dtype=torch.int64, device=self.device)
            self.eviction_hand = torch.zeros((), dtype=torch.int32, device=self.device)
            self.ready = torch.zeros((r, t), dtype=torch.int32, device=self.device)
            self.claims = torch.zeros_like(self.ready)
            self.wanted = torch.zeros_like(self.group_to_slot)
            self.wanted_groups = torch.empty(
                misses, dtype=torch.int32, device=self.device
            )
            self.wanted_count = torch.zeros((), dtype=torch.int32, device=self.device)
            priority_slots = slots if config.eviction_policy == "lru" else 0
            self.priority = torch.empty(
                priority_slots, dtype=torch.int64, device=self.device
            )
            self.sorted_priority = torch.empty_like(self.priority)
            self.victim_slots = torch.empty(
                slots if config.eviction_policy == "lru" else misses,
                dtype=torch.int64 if config.eviction_policy == "lru" else torch.int32,
                device=self.device,
            )
            self.victim_tiles = min(triton.cdiv(slots, 512), 256)
            self.victim_counts = torch.empty(
                (3, self.victim_tiles), dtype=torch.int32, device=self.device
            )
            # A bounded fixed grid also works during capture and at zero misses.
            self.gather_workers = min(misses, 256)
            self.miss_groups = torch.empty(
                misses, dtype=torch.int32, device=self.device
            )
            self.miss_slots = torch.empty_like(self.miss_groups)
            self.miss_count = torch.zeros((), dtype=torch.int32, device=self.device)
            self.indices = torch.empty(
                (config.rows, config.selection_width),
                dtype=torch.int32,
                device=self.device,
            )
            self.encoded = torch.empty(
                (config.max_write_rows, 2, 1, heads, dim),
                dtype=layout_like.dtype,
                device=self.device,
            )
            self.encoding_slots = torch.arange(
                config.max_write_rows, dtype=torch.int64, device=self.device
            )
            self.k_scale = torch.tensor(
                [scales.key], dtype=torch.float32, device=self.device
            )
            self.v_scale = torch.tensor(
                [scales.value], dtype=torch.float32, device=self.device
            )
        except (RuntimeError, MemoryError) as error:
            raise HostKVResourceError(
                "Host KV allocation or UVA initialization failed"
            ) from error

    def _metadata(
        self,
        tensor: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.int32,
    ) -> None:
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != self.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Expected contiguous {dtype} metadata of shape {shape}")

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        requests: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Quantize with the original cache writer, then commit host and hot KV.

        CPU quantization is a torch reference. CUDA uses the *existing* native
        E4M3 writer with the layer scales, not a new rounding implementation.
        Source tensors and the pool must outlive queued work.
        """
        n = requests.numel()
        if (
            not 0 < n <= self.config.max_write_rows
            or key.shape != (n, self.heads, self.dim)
            or value.shape != key.shape
            or key.dtype != torch.float16
            or value.dtype != key.dtype
            or key.device != self.device
            or value.device != self.device
            or not key.is_contiguous()
            or not value.is_contiguous()
        ):
            raise ValueError(
                "write requires bounded contiguous FP16 [rows,heads,dim] KV"
            )
        self._metadata(requests, (n,))
        self._metadata(positions, (n,))
        if self.device.type == "cuda":
            from vllm._custom_ops import reshape_and_cache_flash

            k, v = self.encoded.unbind(1)
            reshape_and_cache_flash(
                key,
                value,
                k,
                v,
                self.encoding_slots[:n],
                "fp8_e4m3" if self.is_e4m3 else "auto",
                self.k_scale,
                self.v_scale,
            )
            encoded_k, encoded_v = k[:n, 0], v[:n, 0]
        elif self.is_e4m3:
            encoded_k = (
                (key.float() / self.scales.key)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            encoded_v = (
                (value.float() / self.scales.value)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
        else:
            encoded_k, encoded_v = key, value
        self.write_encoded(encoded_k, encoded_v, requests, positions)

    def write_encoded(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        requests: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Commit already encoded rows without changing any payload bit.

        Positions -1 are padding. Duplicate active coordinates in one launch
        are rejected before any payload mutation. Overwrites across launches
        are permitted and refresh a resident partial group immediately.
        """
        n = requests.numel()
        if (
            not 0 < n <= self.config.max_write_rows
            or key.shape != (n, self.heads, self.dim)
            or value.shape != key.shape
            or key.dtype != self.host.dtype
            or value.dtype != key.dtype
            or key.device != self.device
            or value.device != self.device
            or key.stride(-1) != 1
            or value.stride(-1) != 1
        ):
            raise ValueError("Invalid encoded host KV write shape, device or dtype")
        self._metadata(requests, (n,))
        self._metadata(positions, (n,))
        _check_storage(key)
        _check_storage(value)
        c = self.config
        if self.device.type == "cpu":
            active = positions != -1
            req, pos = requests[active], positions[active]
            if (
                (req < 0) | (req >= c.requests) | (pos < 0) | (pos >= c.max_tokens)
            ).any():
                raise ValueError("invalid host KV write coordinate")
            ids = req.long() * c.max_tokens + pos
            if ids.unique().numel() != ids.numel():
                raise ValueError("duplicate host KV write coordinate")
        kernels.claim_writes[(triton.cdiv(n, 128),)](
            requests,
            positions,
            self.claims,
            n,
            c.requests,
            c.max_tokens,
            128,
            debug=True,
        )
        kernels.write_rows[(n,)](
            key,
            value,
            requests,
            positions,
            self.block_table,
            self.host_device,
            self.hot,
            self.group_to_slot,
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            *self.host.stride()[:4],
            self.page_tokens,
            self.columns,
            c.groups_per_request,
            self.heads,
            self.dim,
            triton.next_power_of_2(self.heads * self.dim),
        )
        # A separate kernel boundary publishes writes only after both copies.
        kernels.commit_writes[(triton.cdiv(n, 128),)](
            requests,
            positions,
            self.ready,
            self.claims,
            n,
            c.max_tokens,
            128,
        )

    def gather(
        self,
        selection: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return stable [rows,width] hot token IDs, preserving selection order.

        The result and device miss counter are reused by the next call. Logical
        padding/future columns become -1; transfer deduplication never removes
        attention columns. Tail groups stay resident even when unselected.
        """
        c = self.config
        self._metadata(selection, (c.rows, c.selection_width))
        self._metadata(requests, (c.rows,))
        self._metadata(lengths, (c.requests,))
        self._metadata(query_positions, (c.rows,), torch.int64)
        if self.device.type == "cpu":
            # tl.device_assert is a no-op in Triton's interpreter. Validate
            # values with torch here; the CUDA branch never inspects values.
            self._cpu_validate_gather(selection, requests, lengths, query_positions)
        self._gather_metadata(selection, requests, lengths, query_positions)
        self._gather_payload()
        self._remap(selection, requests, lengths, query_positions)
        return self.indices

    def _gather_metadata(
        self,
        selection: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        self._mark_and_collect(selection, requests, lengths, query_positions)
        self._sort_victims()
        self._select_victims()
        self._update_mapping()

    def _mark_and_collect(
        self,
        selection: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        c = self.config
        kernels.clear_wanted[(triton.cdiv(self.max_misses, 256),)](
            self.wanted,
            self.wanted_groups,
            self.wanted_count,
            256,
        )
        kernels.begin_gather[(1,)](self.wanted_count, self.miss_count, self.clock)
        kernels.mark_tail[(c.requests,)](
            lengths,
            self.ready,
            self.wanted,
            self.wanted_groups,
            self.wanted_count,
            c.requests,
            c.max_tokens,
            c.groups_per_request,
            self.max_misses,
            debug=True,
        )
        grid = (c.rows, triton.cdiv(c.selection_width, 128))
        kernels.mark_selection[grid](
            selection,
            requests,
            lengths,
            query_positions,
            self.ready,
            self.wanted,
            self.wanted_groups,
            self.wanted_count,
            c.requests,
            c.max_tokens,
            c.groups_per_request,
            c.selection_width,
            self.max_misses,
            128,
            c.qsa_group_columns if c.qsa_block_topk is not None else 0,
            c.row_groups * c.qsa_group_columns,
            debug=True,
        )
        kernels.collect_misses[(triton.cdiv(self.max_misses, 256),)](
            self.wanted_groups,
            self.wanted_count,
            self.group_to_slot,
            self.age,
            self.clock,
            self.miss_groups,
            self.miss_count,
            self.max_misses,
            256,
            debug=True,
        )

    def _sort_victims(self) -> None:
        if self.config.eviction_policy != "lru":
            return
        kernels.prepare_slot_priorities[(triton.cdiv(self.slots, 256),)](
            self.wanted,
            self.slot_owner,
            self.age,
            self.clock,
            self.priority,
            self.slots,
            256,
        )
        # Stable age sorting preserves (age, slot) step-LRU without the old
        # serial O(misses * slots) reduction. CUDA uses PyTorch's parallel sort.
        # Fixed outputs are preallocated. Sort scratch is owned by PyTorch's
        # allocator / graph private pool and must be warmed before capture.
        try:
            torch.sort(
                self.priority,
                stable=True,
                out=(self.sorted_priority, self.victim_slots),
            )
        except torch.OutOfMemoryError as error:
            raise HostKVResourceError(
                "Host KV sort workspace allocation failed"
            ) from error

    def _select_victims(self) -> None:
        if self.config.eviction_policy == "lru":
            return
        kernels.count_victim_buckets[(self.victim_tiles,)](
            self.wanted,
            self.slot_owner,
            self.age,
            self.clock,
            self.miss_count,
            self.victim_counts,
            self.eviction_hand,
            self.slots,
            self.victim_tiles,
            512,
        )
        kernels.select_bucket_victims[(self.victim_tiles,)](
            self.wanted,
            self.slot_owner,
            self.age,
            self.clock,
            self.miss_count,
            self.victim_counts,
            self.victim_slots,
            self.eviction_hand,
            self.slots,
            self.victim_tiles,
            512,
            triton.next_power_of_2(self.victim_tiles),
            debug=True,
        )

    def _update_mapping(self) -> None:
        kernels.allocate_slots[(triton.cdiv(self.max_misses, 256),)](
            self.group_to_slot,
            self.slot_owner,
            self.age,
            self.clock,
            self.miss_groups,
            self.miss_slots,
            self.miss_count,
            self.victim_slots,
            self.sorted_priority,
            self.eviction_hand,
            self.max_misses,
            self.slots,
            256,
            self.config.eviction_policy == "lru",
            debug=True,
        )

    def _gather_payload(self) -> None:
        c = self.config
        kernels.gather_groups[(self.gather_workers,)](
            self.host_device,
            self.hot,
            self.block_table,
            self.ready,
            self.miss_groups,
            self.miss_slots,
            self.miss_count,
            *self.host.stride()[:4],
            self.page_tokens,
            self.columns,
            c.max_tokens,
            c.groups_per_request,
            self.heads,
            self.dim,
            triton.next_power_of_2(8 * self.heads * self.dim),
        )

    def _remap(
        self,
        selection: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        c = self.config
        grid = (c.rows, triton.cdiv(c.selection_width, 128))
        kernels.remap_selection[grid](
            selection,
            requests,
            lengths,
            query_positions,
            self.group_to_slot,
            self.indices,
            c.max_tokens,
            c.groups_per_request,
            c.selection_width,
            128,
            debug=True,
        )

    def _cpu_validate_gather(
        self,
        selection: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        c = self.config
        if ((lengths < 0) | (lengths > c.max_tokens)).any():
            raise ValueError("invalid host KV length")
        if ((requests < 0) | (requests >= c.requests)).any():
            raise ValueError("invalid host KV request")
        if ((selection < -1) | (selection >= c.max_tokens)).any():
            raise ValueError("invalid host KV selection")
        for req, length in enumerate(lengths.tolist()):
            if not self.ready[req, max(0, length - 32) : length].all():
                raise ValueError("tail KV is not committed")
        for tokens, req, qpos in zip(
            selection, requests.tolist(), query_positions.tolist()
        ):
            if c.qsa_block_topk is not None:
                limit = c.row_groups * c.qsa_group_columns
                if (tokens[limit:] >= 0).any():
                    raise ValueError("host KV QSA group layout exceeded")
                for start in range(
                    0, min(limit, c.selection_width), c.qsa_group_columns
                ):
                    chunk = tokens[start : start + c.qsa_group_columns]
                    active = chunk >= 0
                    if active.any() and (
                        chunk[0] < 0 or (chunk[active] // 4 != chunk[0] // 4).any()
                    ):
                        raise ValueError("host KV QSA group layout exceeded")
            valid = (tokens >= 0) & (tokens < lengths[req]) & (tokens <= qpos)
            if not self.ready[req, tokens[valid].long()].all():
                raise ValueError("selected KV is not committed")

    def prefetch(
        self,
        selection: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        """Warm from the prefill final-step selection, using the same shape."""
        self.gather(selection, requests, lengths, query_positions)

    warm = prefetch

    def truncate(self, lengths: torch.Tensor) -> None:
        """Retire rejected suffixes; zero lengths release request identities.

        Whole groups intersecting the retired suffix are invalidated. All old
        ready flags past each new length are cleared before identity reuse.
        Retained tokens remain the caller's responsibility (no prefix COW).
        """
        c = self.config
        self._metadata(lengths, (c.requests,))
        if (
            self.device.type == "cpu"
            and ((lengths < 0) | (lengths > c.max_tokens)).any()
        ):
            raise ValueError("invalid truncate length")
        kernels.truncate_ready[(triton.cdiv(c.requests * c.max_tokens, 128),)](
            lengths,
            self.ready,
            c.requests,
            c.max_tokens,
            128,
            debug=True,
        )
        kernels.retire_slots[(triton.cdiv(self.slots, 128),)](
            lengths,
            self.group_to_slot,
            self.slot_owner,
            self.age,
            c.groups_per_request,
            self.slots,
            128,
        )
