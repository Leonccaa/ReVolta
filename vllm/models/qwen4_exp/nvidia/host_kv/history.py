# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main QSA K/V in physical host pages, addressed through ``page_of[hid]``.

The scheduler's HostPageTable owns allocation and retirement. A worker maps
one shared region whose rows contain every rank's main KV and logical index
payload. A history ID is a binding, and a page can outlive an ID incarnation.
The worker applies binding updates on the forward stream before using an ID.

A layer's host view is ``[page, subpage, 2, kbs, H, D]``. Its strides retain
other ranks and index payload between rows. Layer views alias the registered
backing, including when subpages cannot be flattened into a uniform stride.

Reads go through a per-layer GPU hot pool of four-token groups, keyed by
``(kernel block, group)``: select -> mark wanted groups -> evict (clock
buckets, host KV pool) -> gather missing groups over UVA -> remap the selection to hot
token IDs -> the unchanged QSA attention kernels with an identity page table.
A step's query rows are processed in passes whose QSA union bound fits the
hot pool, so no selection can exceed it.

Writes encode with the cache's own writer into a staging row, then copy the
bytes to the host page and refresh a resident hot group, and a separate kernel
publishes them (``ready``). All of it is ordered on the caller's stream; a
host-side reader (SSD persistence, connectors) must wait on ``fence()``.

Calls must be serialized on one CUDA stream. Every launch has a fixed grid;
no call reads a device value on the host, so the write and read paths can be
captured in a CUDA graph. ``begin_blocks`` copies its ID list to the device
and runs before the forward pass, outside any graph.
"""

from __future__ import annotations

import math
import os
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from vllm.triton_utils import triton
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)
from vllm.v1.kv_offload.host.page_table import SealTicket
from vllm.v1.kv_offload.host.region import HostRegionLayout

from . import history_kernels as hk
from . import kernels
from .budget import (
    HISTORY_METADATA,
    QSA_ROW_GROUPS,
    READ_ROWS,
    SHARED_PER_SLOT,
    TAIL_GROUPS,
    history_fixed_memory_bytes,
)
from .pool import HostKVResourceError, UnsupportedHostKVConfig, _check_product
from .staged_write import WritePlan

# Row-group size of union windows and the masking of other windows' rows.
# Module constants only so mutation tests can show the tests depend on them.
_UNION_ROW_GROUP = 8
_MASK_OTHER_WINDOWS = True

UNION_STAT_NAMES = (
    "calls",
    "l0",
    "l1",
    "l2",
    "l3_groups",
    "windows",
    "syncs",
    "probe_overflows",
)


def tensor_storage_bytes(values: Mapping[str, object]) -> dict[str, int]:
    """Bill owned storage once, including per-layer lists and aliased views.

    A slice owns the underlying allocation's bytes, not just its numel. Walk
    explicit tensor containers only; layer objects can reference this store.
    """
    seen: set[tuple[torch.device, int]] = set()

    def size(value: object) -> int:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (value.device, storage._cdata)
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(value, (list, tuple)):
            return sum(size(item) for item in value)
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        return 0

    return {
        name: size(value)
        for name, value in values.items()
        if isinstance(value, (torch.Tensor, list, tuple, dict))
    }


@dataclass(frozen=True)
class HostHistoryShape:
    """Fixed geometry of one worker's host history store."""

    num_blocks: int
    """History block IDs, the null block 0 included."""
    kernel_blocks_per_block: int
    """Kernel blocks per history block (1 unless the backend splits pages)."""
    kernel_block_tokens: int
    """Rank-local tokens per kernel block; a multiple of four."""
    num_layers: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    """Payload dtype: uint8 (E4M3) or float16."""
    hot_slots: int
    """Hot four-token groups per layer."""
    max_requests: int
    """Rows of the attention block table in one step."""
    selection_width: int
    """Columns of a QSA selection row (``token_topk + compress_ratio - 1``)."""
    max_write_rows: int
    """Tokens written per layer per step (``max_num_batched_tokens``)."""
    row_groups: int = QSA_ROW_GROUPS
    tail_groups: int = TAIL_GROUPS
    dcp_world_size: int = 1
    dcp_rank: int = 0
    interleave: int = 1
    qsa_group_columns: int = 0
    """Columns per selected group in the selection (4, 2 after DCP2
    localization), or 0 to claim every column separately."""
    read_rows: int = READ_ROWS
    """R_cap: fixed indices scratch capacity, independent of hot slot count."""
    max_context_tokens: int | None = None
    """Startup context bound, allowing short histories below the 513-row bound."""
    decode_rows: int = 1
    """Maximum verify rows of one request; uniform decode stays one pass."""
    staged_writethrough: bool = False
    prefill_gather: str = "passes"
    """Mixed steps: ``passes`` (remap passes within R_cap) or ``union``
    (selection-union gather/selection-union gather: one union fill per window of whole
    eight-row groups, a
    page-size-4 table over the hot slots and the original selection)."""
    prefill_insert_age: str = "old"
    """Union path only: ``old`` inserts groups that only prefill rows want at
    ``clock - 9`` (first evicted), ``recent`` at ``clock``."""
    slots_per_block: int = 0
    """shared hot region shared region: k hot slots alias each history ID's index page,
    so
    ``hot_slots == num_blocks * k`` and only DONATED IDs' slots are usable.
    0 keeps the static, separately allocated hot pool."""

    def conservative_pass_geometry(self) -> tuple[int, int, list[int]]:
        """Planner inputs safe for any request assignment, without device reads.

        For short histories one row can cover an entire request. At most N
        different requests can contribute, so the bound saturates at N*G.
        Long histories retain the original numerical pass boundaries.
        """
        from .planning import history_groups

        if self.max_context_tokens is not None:
            groups = history_groups(
                self.max_context_tokens, self.dcp_world_size, self.interleave
            )
            if groups <= self.row_groups + self.tail_groups:
                tail = min(groups, self.tail_groups)
                delta = groups - tail
                return (
                    max(delta, 1),
                    tail,
                    [tail + self.max_requests * delta]
                    + [tail] * (self.max_requests - 1),
                )
        return self.row_groups, self.tail_groups, [2**63 - 1] * self.max_requests

    @property
    def num_kernel_blocks(self) -> int:
        return self.num_blocks * self.kernel_blocks_per_block

    @property
    def groups_per_kernel_block(self) -> int:
        return self.kernel_block_tokens // 4

    @property
    def groups_per_block(self) -> int:
        return self.kernel_blocks_per_block * self.groups_per_kernel_block

    @property
    def num_groups(self) -> int:
        return self.num_blocks * self.groups_per_block

    @property
    def rows_per_pass(self) -> int:
        """Query rows whose union bound fits the hot pool with every
        request's tail resident."""
        row_groups, tail, groups = self.conservative_pass_geometry()
        if sum(groups) <= self.hot_slots:
            return self.read_rows
        return min(
            self.read_rows, (self.hot_slots - self.max_requests * tail) // row_groups
        )

    @property
    def page_bytes(self) -> int:
        """Host bytes of one history block over all layers."""
        return (
            self.kernel_blocks_per_block
            * self.num_layers
            * 2
            * self.kernel_block_tokens
            * self.kv_heads
            * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )

    def fixed_device_memory_bytes(self) -> dict[str, int]:
        """Required hot-pool reservation, by tensor name (no rounding).

        The shared region bills slot payload (the index pages) and per-slot
        metadata per history ID instead."""
        parts = self._static_fixed_bytes()
        if self.slots_per_block:
            parts = {
                name: size
                for name, size in parts.items()
                if name not in SHARED_PER_SLOT
            }
        return parts

    def _static_fixed_bytes(self) -> dict[str, int]:
        return history_fixed_memory_bytes(
            slots=self.hot_slots,
            num_layers=self.num_layers,
            slot_bytes=self.num_layers
            * 8
            * self.kv_heads
            * self.head_dim
            * self.dtype.itemsize,
            max_requests=self.max_requests,
            selection_width=self.selection_width,
            max_write_rows=self.max_write_rows,
            staged_writethrough=self.staged_writethrough,
            row_groups=self.row_groups,
            tail_groups=self.tail_groups,
            read_rows=self.read_rows,
        )

    def validate(self) -> None:
        sizes = (
            self.num_blocks,
            self.kernel_blocks_per_block,
            self.kernel_block_tokens,
            self.num_layers,
            self.kv_heads,
            self.head_dim,
            self.hot_slots,
            self.max_requests,
            self.selection_width,
            self.max_write_rows,
            self.row_groups,
            self.dcp_world_size,
            self.interleave,
            self.read_rows,
            self.decode_rows,
        )
        if any(type(n) is not int or n <= 0 for n in sizes) or self.tail_groups < 0:
            raise ValueError("Host history dimensions must be positive integers")
        if self.max_context_tokens is not None and self.max_context_tokens < 1:
            raise ValueError("Host history context bound must be positive")
        if self.slots_per_block and (
            type(self.slots_per_block) is not int
            or self.slots_per_block < 0
            or self.hot_slots != self.num_blocks * self.slots_per_block
        ):
            raise ValueError("Shared hot region needs num_blocks * k slots")
        if self.kernel_block_tokens % 4:
            raise UnsupportedHostKVConfig(
                "Host history kernel blocks must hold a multiple of four tokens"
            )
        if self.dtype not in (torch.uint8, torch.float16):
            raise UnsupportedHostKVConfig(
                "Host history supports E4M3 (uint8) or FP16 main K/V"
            )
        if not 0 <= self.dcp_rank < self.dcp_world_size:
            raise ValueError("Host history DCP rank is out of range")
        if self.kernel_block_tokens % self.interleave:
            raise UnsupportedHostKVConfig("DCP interleave must divide a kernel block")
        if self.qsa_group_columns not in (0, 2, 4):
            raise UnsupportedHostKVConfig("QSA group columns must be 0, 2 or 4")
        if self.prefill_gather not in ("passes", "union"):
            raise ValueError("prefill_gather must be 'passes' or 'union'")
        if self.prefill_insert_age not in ("old", "recent"):
            raise ValueError("prefill_insert_age must be 'old' or 'recent'")
        if self.rows_per_pass < 1:
            need = self.row_groups + self.max_requests * self.tail_groups
            raise UnsupportedHostKVConfig(
                f"hot pool holds {self.hot_slots} groups per layer but one query "
                f"row with {self.max_requests} request tails needs {need}"
            )
        for name, factors in (
            ("history groups", (self.num_groups,)),
            ("ready tokens", (self.num_kernel_blocks, self.kernel_block_tokens)),
            ("hot token coordinates", (self.hot_slots, 4)),
            ("hot page table", (self.max_requests, self.hot_slots)),
            ("selection", (self.read_rows, self.selection_width)),
            ("group transfer elements", (8, self.kv_heads, self.head_dim)),
            ("selection grid Y", (triton.cdiv(self.selection_width, 128),)),
        ):
            limit = {
                "group transfer elements": 16384,
                "selection grid Y": 65535,
            }.get(name, 2**31 - 1)
            _check_product(name, *factors, limit=limit)


def _check_hot_alias(
    view: torch.Tensor, shape: HostHistoryShape, device: torch.device
) -> torch.Tensor:
    """A layer's hot slots as a uniform-stride alias of its index pages.

    The view must cover the whole index storage, start at it, and keep the
    page boundary on a slot boundary (non-divisible strides are rejected at
    planning time)."""
    s = shape
    slot_shape = (s.hot_slots, 2, 4, s.kv_heads, s.head_dim)
    storage = view.untyped_storage()
    if (
        view.shape != slot_shape
        or view.dtype != s.dtype
        or view.device != device
        or not view.is_contiguous()
        or view.storage_offset() != 0
        or view.data_ptr() != storage.data_ptr()
        or storage.nbytes() != view.numel() * view.element_size()
    ):
        raise ValueError("Shared hot region view does not alias whole index pages")
    return view


def hot_alias_view(index: torch.Tensor, shape: HostHistoryShape) -> torch.Tensor:
    """Alias ``[num_blocks, P_pad]`` index storage as ``[num_blocks*k, 2, 4,
    H, D]`` hot slots. No copy: the result shares the storage."""
    s = shape
    storage = index.untyped_storage()
    raw = torch.empty(0, dtype=torch.uint8, device=index.device)
    raw.set_(storage, 0, (storage.nbytes(),))
    slot_bytes = 8 * s.kv_heads * s.head_dim * s.dtype.itemsize
    if storage.nbytes() != s.num_blocks * s.slots_per_block * slot_bytes:
        raise ValueError(
            f"Index storage of {storage.nbytes()} bytes is not {s.num_blocks} "
            f"pages of {s.slots_per_block} hot slots"
        )
    return raw.view(s.dtype).view(s.hot_slots, 2, 4, s.kv_heads, s.head_dim)


def _allocate_host(numel: int, dtype: torch.dtype) -> torch.Tensor:
    try:
        return torch.zeros(numel, dtype=dtype, device="cpu")
    except (RuntimeError, MemoryError) as error:
        raise HostKVResourceError("Host history allocation failed") from error


def _register_host(host: torch.Tensor) -> None:
    """Pin by registration: no power-of-two rounding of a large pool."""
    nbytes = host.numel() * host.element_size()
    # Portable | mapped: every device context, with a device alias for UVA.
    result = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), nbytes, 3)
    if result.value != 0:
        raise HostKVResourceError(
            f"cudaHostRegister failed for {nbytes} host history bytes "
            f"(code={result.value})"
        )


def _unregister_host(host: torch.Tensor) -> None:
    # The finalizer holds ``host``, so the memory outlives its registration.
    torch.cuda.cudart().cudaHostUnregister(host.data_ptr())


class HostHistoryStore:
    """One worker's host pages and per-layer hot pools for the main K/V."""

    def __init__(
        self,
        shape: HostHistoryShape,
        layer_names: list[str],
        *,
        device: torch.device | str,
        debug: bool = False,
        region_layout: HostRegionLayout | None = None,
        rank: int = 0,
        value_shadow: bool = False,
        store_metrics: bool = False,
        hot_views: Sequence[torch.Tensor] | None = None,
    ) -> None:
        shape.validate()
        if (hot_views is None) != (shape.slots_per_block == 0):
            raise ValueError("Shared hot region needs one aliased view per layer")
        if len(layer_names) != shape.num_layers or len(set(layer_names)) != len(
            layer_names
        ):
            raise ValueError("Host history needs one distinct name per layer")
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Host history supports CUDA or the CPU interpreter")
        if device.type == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
            raise ValueError("CPU host history fixtures require TRITON_INTERPRET=1")
        self.shape, self.device, self.debug = shape, device, debug
        s = shape
        kb, kbs, layers = s.num_kernel_blocks, s.kernel_block_tokens, s.num_layers
        h, d = s.kv_heads, s.head_dim
        dims = (s.num_blocks, s.kernel_blocks_per_block, layers, 2, kbs, h, d)
        self.region = None
        self._registration_finalizer = None
        self.layers: list[HostHistoryLayer] = []
        self.layer_index: dict[str, int] = {}
        self.index_payloads: list[torch.Tensor] = []
        self._seal_sources: list[torch.Tensor] = []
        flat = raw = None
        try:
            if region_layout is None:
                flat = _allocate_host(math.prod(dims), s.dtype)
                self.host = flat.view(dims)
            else:
                self.region = region_layout.open(rank)
                raw = self.region.create_next_view(region_layout.main_bytes)
                # Unflatten preserves the row stride across other workers and
                # this rank's logical index payload. No reshape may copy here.
                self.host = raw.view(s.dtype).unflatten(1, dims[1:])
                self.index_payloads = [
                    self.region.create_next_view(n).view(torch.uint8)
                    for n in region_layout.index_bytes
                ]
                flat = self.region._base
            if device.type == "cuda" and self.region is None:
                _register_host(flat)
                self._registration_finalizer = weakref.finalize(
                    self, _unregister_host, flat
                )
            elif device.type == "cuda":
                from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region

                assert self.region is not None
                pin_mmap_region(self.region, flags=3)
            if device.type == "cuda":
                with torch.accelerator.device_index(device.index):
                    self.host_device = get_accelerator_view_from_cpu_tensor(self.host)
            else:
                self.host_device = self.host
            i32 = dict(dtype=torch.int32, device=device)
            self.page_of = (
                torch.arange(s.num_blocks, **i32)
                if region_layout is None
                else torch.full((s.num_blocks,), -1, **i32)
            )
            self.page_of[0] = 0
            # Control-plane binding checks for seal and diagnostic host views.
            # ReceiveTxn addresses its sources directly by physical handle.
            self.page_of_cpu = self.page_of.cpu().numpy().copy()
            self.ready = torch.zeros(kb * kbs, dtype=torch.uint8, device=device)
            # The null block is never written; reading it yields zeros.
            self.ready[: s.kernel_blocks_per_block * kbs] = 1
            self.generation = torch.zeros(s.num_blocks, **i32)
            slots = s.hot_slots
            self.hot: torch.Tensor | list[torch.Tensor]
            if hot_views is None:
                self.hot = torch.zeros(
                    (layers, slots, 2, 4, h, d), dtype=s.dtype, device=device
                )
            else:
                self.hot = [_check_hot_alias(view, s, device) for view in hot_views]
                if len(self.hot) != layers:
                    raise ValueError("Shared hot region needs one view per layer")
            self.group_to_slot = torch.full((layers, s.num_groups), -1, **i32)
            self.slot_owner = torch.full((layers, slots), -1, **i32)
            # Shared region: the null ID's slots are never hot. Every other ID
            # starts DONATED, matching the scheduler's initial roles.
            self.slot_owner[:, : s.slots_per_block] = -2
            self.available_slots = (
                slots - s.slots_per_block if s.slots_per_block else slots
            )
            self.step_passes: list[tuple[int, int]] | None = None
            self.age = torch.zeros((layers, slots), dtype=torch.int64, device=device)
            self.clock = torch.zeros(layers, dtype=torch.int64, device=device)
            self.eviction_hand = torch.zeros(layers, **i32)
            # Per-call scratch, shared by the layers (one stream, one call at
            # a time). The touched list bounds every count by the hot slots.
            self.max_misses = slots
            self.wanted = torch.zeros(s.num_groups, **i32)
            self.wanted_groups = torch.empty(slots, **i32)
            self.wanted_count = torch.zeros((), **i32)
            self.miss_groups = torch.empty(slots, **i32)
            self.miss_slots = torch.empty(slots, **i32)
            self.miss_count = torch.zeros((), **i32)
            self.victim_slots = torch.empty(slots, **i32)
            self.victim_tiles = min(triton.cdiv(slots, 512), 256)
            self.victim_counts = torch.empty((3, self.victim_tiles), **i32)
            self.gather_workers = min(slots, 256)
            self.indices = torch.empty((s.read_rows, s.selection_width), **i32)
            # Hot token IDs are global slot coordinates; every request row of
            # the identity table maps page p to hot slot p.
            self.hot_block_table = (
                torch.arange(slots, **i32).unsqueeze(0).repeat(s.max_requests, 1)
            )
            self.staging = torch.empty(
                (s.max_write_rows, 2, 1, h, d), dtype=s.dtype, device=device
            )
            self.write_plan: WritePlan | None = None
            self.write_stage = self.write_targets = None
            if s.staged_writethrough:
                self.write_stage = torch.empty(
                    (2 * s.max_write_rows, h, d), dtype=s.dtype, device=device
                )
                self.write_targets = torch.empty(
                    (s.max_write_rows, 2), dtype=torch.int64, device=device
                )
            self.staging_slots = torch.arange(
                s.max_write_rows, dtype=torch.int64, device=device
            )
            # selection-union gather union scratch is allocated on first use of a mixed
            # union
            # step and billed by ``union_gather_memory_bytes``, so the
            # ``passes`` arm allocates exactly what it did before.
            self.union_scratch: dict[str, torch.Tensor] = {}
            self.union_stats: dict[str, int] = dict.fromkeys(UNION_STAT_NAMES, 0)
            # This step's CPU batch metadata for the union path (shared hot region
            # runner).
            self.union_step: UnionStepInfo | None = None
            # value arbitration value arbitration: layer-0 shadow records and counters.
            # None
            # (arbitration off) launches nothing extra.
            self.shadow = None
            self.hot_metrics = None
            if store_metrics:
                from .shadow import HotCacheMetrics

                self.hot_metrics = HotCacheMetrics(device)
            if value_shadow:
                from .shadow import HotShadowTable

                self.shadow = HotShadowTable(s.num_groups, s.groups_per_block, device)
            self.layers = [
                HostHistoryLayer(self, index, name)
                for index, name in enumerate(layer_names)
            ]
            self.layer_index = {name: index for index, name in enumerate(layer_names)}
        except BaseException as error:
            # Drop constructor aliases before releasing the mmap buffer export.
            flat = raw = None
            self.close()
            if isinstance(error, (RuntimeError, MemoryError)):
                raise HostKVResourceError(
                    "Host history registration, metadata or UVA allocation failed"
                ) from error
            raise

    def fixed_device_memory_bytes(self) -> dict[str, int]:
        """Actual fixed tensor bytes, excluding host pages and history metadata.

        The shared region's aliased payload and per-slot arrays are billed per
        history ID and are excluded here."""
        shape = getattr(self, "shape", None)
        shared = (
            SHARED_PER_SLOT
            if shape is not None and shape.slots_per_block
            else frozenset()
        )
        return tensor_storage_bytes(
            {
                name: value
                for name, value in vars(self).items()
                if name
                not in HISTORY_METADATA
                | shared
                | {
                    "host",
                    "host_device",
                    "layers",
                    "layer_index",
                    "page_of",
                    "index_payloads",
                    "_seal_sources",
                    "union_scratch",
                    "union_stats",
                }
            }
        )

    # ----- interface for the scheduler-facing tiers (host prefix adapter)
    # -----------------

    def host_page(self, block_id: int) -> torch.Tensor:
        """Pinned CPU bytes of one history block, all layers:
        ``[ratio, layers, 2, kernel_block_tokens, H, D]``, contiguous."""
        if not 0 <= block_id < self.shape.num_blocks:
            raise IndexError(f"history block {block_id} is out of range")
        page = int(self.page_of_cpu[block_id])
        if page < 0:
            raise ValueError("History block has no host page binding")
        return self.host[page]

    def bind_pages(self, bindings: list[tuple[int, int]]) -> None:
        """Apply ordered hid/page updates on the forward stream, outside graphs."""
        if not bindings:
            return
        ids = self._block_ids([hid for hid, _ in bindings])
        values = async_tensor_h2d(
            [page for _, page in bindings],
            dtype=torch.int32,
            device=self.device,
            pin_memory=self.device.type == "cuda",
        )
        self.page_of[ids] = values
        for hid, page in bindings:
            self.page_of_cpu[hid] = page

    def seal(self, tickets: list[SealTicket], index_views: list[torch.Tensor]) -> None:
        """Queue logical index D2H before the output stream's dependency.

        Keep packed device sources alive until that step's output has retired.
        The temporary sources belong to this stream, and their allocator reuse
        is stream ordered. Host destinations remain owned by the tickets.
        """
        from vllm.v1.attention.kv_cache_view import pack_page_payload

        self._seal_sources.clear()
        for ticket in tickets:
            if int(self.page_of_cpu[ticket.hid]) != ticket.handle.page_id:
                raise ValueError("Seal refers to a stale history binding")
            if len(index_views) != len(self.index_payloads):
                raise ValueError("Seal index members differ from host ABI")
            for view, target in zip(index_views, self.index_payloads):
                if view.shape[0] % self.shape.num_blocks:
                    raise ValueError("Index kernel pages do not divide history IDs")
                packed = pack_page_payload(
                    view,
                    ticket.hid,
                    subdivisions=view.shape[0] // self.shape.num_blocks,
                )
                if packed.numel() != target.shape[1]:
                    raise ValueError("Seal payload size differs from logical ABI")
                self._seal_sources.append(packed)
                target[ticket.handle.page_id].copy_(packed, non_blocking=True)

    def close(self) -> None:
        """Called only after all workers' drain acknowledgements were collected."""
        self.layers.clear()
        self.layer_index.clear()
        self.index_payloads.clear()
        self._seal_sources.clear()
        self.host = self.host_device = None
        if self._registration_finalizer is not None:
            self._registration_finalizer()
            self._registration_finalizer = None
        if self.region is not None:
            self.region.cleanup()
            self.region = None

    def layer(self, name: str) -> HostHistoryLayer:
        return self.layers[self.layer_index[name]]

    def fence(self) -> torch.cuda.Event | None:
        """Event after every write queued so far on the current stream.

        Host pages written before it are host-visible once it completes. The
        CPU interpreter runs synchronously and returns None."""
        if self.device.type != "cuda":
            return None
        event = torch.cuda.Event()
        event.record()
        return event

    def _block_ids(self, block_ids: list[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(block_ids, torch.Tensor):
            ids = block_ids.to(device=self.device, dtype=torch.int64)
        else:
            # A fresh pinned source per call, as the KV zeroer does: a reused
            # buffer could be overwritten while its copy is in flight.
            ids = async_tensor_h2d(
                list(block_ids),
                dtype=torch.int64,
                device=self.device,
                pin_memory=self.device.type == "cuda",
            )
        n = ids.numel()
        if self.device.type == "cpu" and (
            (ids < 0).any()
            or (ids >= self.shape.num_blocks).any()
            or ids.unique().numel() != n
        ):
            raise ValueError("Host history needs distinct in-range block IDs")
        return ids

    def _set_ready(self, ids: torch.Tensor, value: int, bump: bool) -> None:
        s = self.shape
        span = s.kernel_blocks_per_block * s.kernel_block_tokens
        hk.set_block_ready[(ids.numel(), triton.cdiv(span, 1024))](
            ids, self.ready, self.generation, s.kernel_blocks_per_block,
            s.kernel_block_tokens, 1024, value, bump,
        )  # fmt: skip

    def begin_blocks(self, block_ids: list[int] | torch.Tensor) -> None:
        """New allocations of history IDs: clear readiness, retire every
        layer's hot groups of those blocks and bump their generation.

        Called with the scheduler's ``new_block_ids_to_zero`` before the
        forward pass, on the forward stream. A prefix-cache hit does not
        reallocate a block and keeps its page."""
        if len(block_ids) == 0:
            return
        ids = self._block_ids(block_ids)
        s = self.shape
        self._set_ready(ids, 0, True)
        grid = (ids.numel(), triton.cdiv(s.groups_per_block, 256), s.num_layers)
        hk.retire_blocks[grid](
            ids, self.group_to_slot, self.slot_owner, self.age, s.num_groups,
            s.hot_slots, s.groups_per_block, 256,
        )  # fmt: skip
        if self.shadow is not None:
            self.shadow.clear_blocks(ids)

    # ----- shared hot region shared region
    # --------------------------------------------------

    def apply_roles(self, donate: Sequence[int], claim: Sequence[int]) -> None:
        """Step-head role changes, on the forward stream, outside graphs.

        Order (frozen design 4.4): donate (retire the ID's own groups, then
        free its slots), claim (unmap residents, reserve), then the caller
        zeroes and begins the new IDs. The forward stream orders these after
        the previous step's reads of those slots."""
        s = self.shape
        if not s.slots_per_block:
            if donate or claim:
                raise ValueError("Static hot pool has no shared-region roles")
            return
        if set(donate) & set(claim) or 0 in donate or 0 in claim:
            raise ValueError("Invalid shared-region role update")
        kp = triton.next_power_of_2(s.slots_per_block)
        if donate:
            ids = self._block_ids(list(donate))
            grid = (ids.numel(), triton.cdiv(s.groups_per_block, 256), s.num_layers)
            hk.retire_blocks[grid](
                ids, self.group_to_slot, self.slot_owner, self.age, s.num_groups,
                s.hot_slots, s.groups_per_block, 256,
            )  # fmt: skip
            hk.set_block_slots[(ids.numel(), s.num_layers)](
                ids, self.group_to_slot, self.slot_owner, self.age, s.num_groups,
                s.hot_slots, s.slots_per_block, kp, False,
            )  # fmt: skip
        if claim:
            ids = self._block_ids(list(claim))
            if self.shadow is not None:
                # value arbitration: residents a claim unmaps are forced evictions,
                # counted
                # apart from value-driven ones; the ID's own records go.
                self.shadow.count_claims(ids, self.slot_owner[0], s.slots_per_block)
                self.shadow.clear_blocks(ids)
            hk.set_block_slots[(ids.numel(), s.num_layers)](
                ids, self.group_to_slot, self.slot_owner, self.age, s.num_groups,
                s.hot_slots, s.slots_per_block, kp, True,
            )  # fmt: skip

    def prepare_write(self, slots: Sequence[int]) -> None:
        """Plan once at step head from CPU page bindings, without a device read."""
        from .staged_write import plan_write

        if not self.shape.staged_writethrough:
            self.write_plan = None
            return
        if len(slots) > self.shape.max_write_rows:
            raise ValueError("Write plan exceeds one chunk")
        view = self.layers[0].host_view
        item = view.element_size()
        self.write_plan = plan_write(
            slots,
            self.page_of_cpu,
            self.shape.kernel_blocks_per_block,
            self.shape.kernel_block_tokens,
            self.shape.kv_heads * self.shape.head_dim * item,
            *(stride * item for stride in view.stride()[:3]),
        )
        assert self.write_targets is not None
        targets = torch.tensor(self.write_plan.targets, dtype=torch.int64).reshape(
            -1, 2
        )
        self.write_targets[: len(slots)].copy_(targets, non_blocking=True)

    def set_step_plan(
        self,
        hot_slots: int | None,
        passes: list[tuple[int, int]] | None,
        union: UnionStepInfo | None = None,
    ) -> None:
        """The scheduler's U_step and this rank's (identical) mixed-step plan:
        remap passes, or the union path's CPU batch metadata (selection-union
        gather)."""
        if hot_slots is not None and self.shape.slots_per_block:
            self.available_slots = hot_slots
        self.step_passes = passes
        self.union_step = union

    def reset_hot_state(self) -> None:
        """After profiling/capture: every non-null ID DONATED, no residency,
        no binding. Dummy runs may have written any index page or slot."""
        s = self.shape
        self.group_to_slot.fill_(-1)
        self.slot_owner.fill_(-1)
        if s.slots_per_block:
            self.slot_owner[:, : s.slots_per_block] = -2
            self.available_slots = s.hot_slots - s.slots_per_block
        self.age.zero_()
        self.clock.zero_()
        self.eviction_hand.zero_()
        self.wanted.zero_()
        if self.shadow is not None:
            self.shadow.reset()
        if self.hot_metrics is not None:
            self.hot_metrics.reset()
        self.ready.zero_()
        self.ready[: s.kernel_blocks_per_block * s.kernel_block_tokens] = 1
        if self.region is not None:
            self.page_of.fill_(-1)
            self.page_of[0] = 0
            self.page_of_cpu[:] = -1
            self.page_of_cpu[0] = 0
        self.step_passes = None
        self.union_step = None

    def mark_ready(self, block_ids: list[int] | torch.Tensor) -> None:
        """Make bound pages readable after ReceiveTxn restores their indexes.

        The host bytes must be complete and ordered before the next forward
        on this stream; the IDs must already have been begun (so no stale
        hot group of them survives)."""
        if len(block_ids) == 0:
            return
        self._set_ready(self._block_ids(block_ids), 1, False)


class HostHistoryLayer:
    """One QSA layer's view of the store: write, gather and the hot cache."""

    def __init__(self, store: HostHistoryStore, index: int, name: str) -> None:
        self.store, self.index, self.name = store, index, name
        s = store.shape
        self.host_device = store.host_device[:, :, index]
        self.host_view = store.host[:, :, index]
        self.hot = store.hot[index]
        self.key_cache, self.value_cache = self.hot.unbind(1)
        self.group_to_slot = store.group_to_slot[index]
        self.slot_owner = store.slot_owner[index]
        self.age = store.age[index]
        self.clock = store.clock[index]
        self.eviction_hand = store.eviction_hand[index]
        self.rows_per_pass = s.rows_per_pass
        self.kernel_block_tokens = s.kernel_block_tokens

    # ----- writes ---------------------------------------------------------

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        """Encode with the cache's own writer, then commit (``write_encoded``).

        CUDA uses the unchanged ``reshape_and_cache_flash`` into a one-token
        staging page, the bytes a GPU-resident cache would hold. The CPU
        interpreter uses the host KV pool torch reference encoder."""
        store, s = self.store, self.store.shape
        n = slot_mapping.shape[0]
        if n == 0:
            return
        if n > s.max_write_rows:
            raise ValueError("host history write exceeds its staging rows")
        key, value = key[:n], value[:n]
        k_stage, v_stage = store.staging[:n].unbind(1)
        if self.store.device.type == "cuda":
            from vllm._custom_ops import reshape_and_cache_flash

            reshape_and_cache_flash(
                key,
                value,
                k_stage,
                v_stage,
                store.staging_slots[:n],
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        elif s.dtype == torch.uint8:
            for source, target, scale in (
                (key, k_stage, k_scale),
                (value, v_stage, v_scale),
            ):
                target[:, 0].copy_(
                    (source.float() / scale.float())
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                )
        else:
            k_stage[:, 0].copy_(key)
            v_stage[:, 0].copy_(value)
        self.write_encoded(k_stage[:, 0], v_stage[:, 0], slot_mapping)

    def write_encoded(
        self, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Copy encoded ``[n, H, D]`` rows to their slots, then publish them.

        Slot -1 is padding (a token another DCP rank owns, or a padded row).
        A resident hot group is refreshed in the same kernel."""
        s = self.store.shape
        n = slot_mapping.shape[0]
        if n == 0:
            return
        if (
            key.shape != (n, s.kv_heads, s.head_dim)
            or value.shape != key.shape
            or key.dtype != s.dtype
            or value.dtype != s.dtype
            or key.stride(-1) != 1
            or value.stride(-1) != 1
            or slot_mapping.ndim != 1
            or slot_mapping.dtype not in (torch.int32, torch.int64)
            or not slot_mapping.is_contiguous()
        ):
            raise ValueError("Invalid host history write tensors")
        if self.store.device.type == "cpu":
            active = slot_mapping[slot_mapping >= 0]
            if (active >= s.num_kernel_blocks * s.kernel_block_tokens).any():
                raise ValueError("host history write slot is out of range")
            if active.unique().numel() != active.numel():
                raise ValueError("duplicate host history write slot")
        store = self.store
        capturing = (
            s.staged_writethrough
            and store.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        )
        plan = store.write_plan if s.staged_writethrough and not capturing else None
        if plan is not None:
            if n > len(plan.targets):
                raise ValueError("Write rows exceed the step plan")
            if self._word_writer_eligible(key, value):
                self._write_words(key, value, slot_mapping, staged=True)
            else:
                hk.stage_slots[(n,)](
                    key, value, slot_mapping, self.host_device, self.hot,
                    store.write_stage, store.write_targets,
                    self.group_to_slot, store.page_of,
                    key.stride(0), key.stride(1), value.stride(0), value.stride(1),
                    *self.host_view.stride()[:5], s.kernel_blocks_per_block,
                    s.kernel_block_tokens, s.kv_heads, s.head_dim,
                    triton.next_power_of_2(s.kv_heads * s.head_dim),
                )  # fmt: skip
            assert store.write_stage is not None
            if store.device.type == "cuda":
                from .staged_write import submit

                submit(
                    plan,
                    store.write_stage.data_ptr(),
                    self.host_view.data_ptr(),
                    torch.cuda.current_stream(store.device).cuda_stream,
                )
            else:
                # Same descriptor engine in the CPU interpreter, including pitch gaps.
                import ctypes

                for copy in plan.copies:
                    for row in range(copy.height):
                        ctypes.memmove(
                            self.host_view.data_ptr()
                            + copy.destination
                            + row * (copy.destination_pitch or copy.width),
                            store.write_stage.data_ptr()
                            + copy.source
                            + row * (copy.source_pitch or copy.width),
                            copy.width,
                        )
        elif s.staged_writethrough and self._word_writer_eligible(key, value):
            self._write_words(key, value, slot_mapping, staged=False)
        else:
            hk.write_slots[(n,)](
                key, value, slot_mapping, self.host_device, self.hot,
                self.group_to_slot, self.store.page_of,
                key.stride(0), key.stride(1), value.stride(0),
                value.stride(1), *self.host_view.stride()[:5],
                s.kernel_blocks_per_block, s.kernel_block_tokens,
                s.kv_heads, s.head_dim, triton.next_power_of_2(s.kv_heads * s.head_dim),
            )  # fmt: skip
        # A separate kernel boundary publishes rows only after both copies.
        hk.commit_slots[(triton.cdiv(n, 256),)](
            slot_mapping,
            self.store.ready,
            self.store.page_of,
            s.kernel_blocks_per_block * s.kernel_block_tokens,
            n,
            256,
        )

    def _write_words(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        *,
        staged: bool,
    ) -> None:
        """Use one word kernel for staged prefill and mixed/pure decode UVA."""
        store, s = self.store, self.store.shape
        assert store.write_stage is not None and store.write_targets is not None
        n = slots.shape[0]
        words = s.kv_heads * s.head_dim * s.dtype.itemsize // 8
        rows = max(1, 256 // triton.next_power_of_2(words))
        word_strides = tuple(
            stride * s.dtype.itemsize // 8 for stride in self.host_view.stride()[:4]
        )
        hk.write_slots_vector[(triton.cdiv(n, rows),)](
            key.view(torch.int64), value.view(torch.int64), slots,
            self.host_device.view(torch.int64), self.hot.view(torch.int64),
            store.write_stage.view(torch.int64), store.write_targets,
            self.group_to_slot, store.page_of, n,
            key.stride(0) * s.dtype.itemsize // 8,
            value.stride(0) * s.dtype.itemsize // 8,
            *word_strides,
            s.kernel_blocks_per_block, s.kernel_block_tokens, words,
            triton.next_power_of_2(words), rows, staged,
        )  # fmt: skip

    def _word_writer_eligible(self, key: torch.Tensor, value: torch.Tensor) -> bool:
        """The vector kernel needs contiguous heads and 16-byte aligned rows."""
        s = self.store.shape
        item = s.dtype.itemsize
        return (
            s.head_dim * item % 16 == 0
            and key.stride(1) == value.stride(1) == s.head_dim
            and key.stride(0) * item % 16 == value.stride(0) * item % 16 == 0
            and key.data_ptr() % 16 == value.data_ptr() % 16 == 0
            and self.host_view.data_ptr() % 16 == self.hot.data_ptr() % 16 == 0
            and all(stride * item % 16 == 0 for stride in self.host_view.stride()[:4])
        )

    # ----- reads ----------------------------------------------------------

    def gather(
        self,
        selection: torch.Tensor,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Make every selected group resident and return hot token IDs.

        ``selection`` holds rank-local token IDs (-1 padding), at most
        ``rows_per_pass`` rows; ``block_table`` is this group's kernel-block
        table and ``seq_lens`` the global sequence lengths of its rows. The
        result ``[rows, width]`` keeps every column's order, repeats and
        padding, and is valid until the next gather of any layer."""
        store, s = self.store, self.store.shape
        rows, width = selection.shape
        requests = seq_lens.shape[0]
        if (
            selection.ndim != 2
            or not 0 < rows <= s.read_rows
            or width > s.selection_width
            or selection.dtype != torch.int32
            or selection.stride(1) != 1
            or token_to_req.shape != (rows,)
            or token_to_req.dtype != torch.int32
            or not token_to_req.is_contiguous()
            or block_table.ndim != 2
            or block_table.dtype != torch.int32
            or block_table.stride(1) != 1
            or block_table.shape[0] < requests
            or not 0 < requests <= s.max_requests
            or seq_lens.ndim != 1
            or not seq_lens.is_contiguous()
        ):
            raise ValueError("Invalid host history gather tensors")
        if store.device.type == "cpu":
            self._cpu_validate(selection, token_to_req, block_table, seq_lens)
        geometry = (s.dcp_world_size, s.dcp_rank, s.interleave)
        m = store.max_misses
        kernels.clear_wanted[(triton.cdiv(m, 256),)](
            store.wanted, store.wanted_groups, store.wanted_count, 256
        )
        kernels.begin_gather[(1,)](store.wanted_count, store.miss_count, self.clock)
        hk.mark_tail[(requests,)](
            block_table, seq_lens, store.ready, store.wanted, store.wanted_groups,
            store.wanted_count, block_table.stride(0), s.kernel_block_tokens, m,
            *geometry, debug=store.debug,
        )  # fmt: skip
        grid = (rows, triton.cdiv(width, 128))
        hk.mark_selection[grid](
            selection, token_to_req, block_table, seq_lens, store.ready,
            store.wanted, store.wanted_groups, store.wanted_count,
            selection.stride(0), block_table.stride(0), width,
            s.kernel_block_tokens, m, 128, *geometry, s.qsa_group_columns,
            debug=store.debug,
        )  # fmt: skip
        self._fill_from_marks()
        out = store.indices[:rows, :width]
        hk.remap_selection[grid](
            selection, token_to_req, block_table, seq_lens, self.group_to_slot,
            out, selection.stride(0), block_table.stride(0), out.stride(0), width,
            s.kernel_block_tokens, 128, *geometry, debug=store.debug,
        )  # fmt: skip
        return out

    def _fill_from_marks(
        self, *, classed: bool = False, counters: torch.Tensor | None = None
    ) -> None:
        """Make the marked groups (Touched/Count) resident.

        Same kernels and order for the remap passes and the union windows:
        collect misses, pick victims (never a wanted slot), allocate, copy
        from the host pages through ``page_of``. ``classed`` (union path,
        ``prefill_insert_age=old``) ages groups only prefill rows want as old.
        """
        store, s = self.store, self.store.shape
        m = store.max_misses
        if classed:
            hk.collect_misses_classed[(triton.cdiv(m, 256),)](
                store.wanted_groups, store.wanted_count, store.wanted,
                self.group_to_slot, self.age, self.clock, store.miss_groups,
                store.miss_count, m, 256, debug=store.debug,
            )  # fmt: skip
        else:
            kernels.collect_misses[(triton.cdiv(m, 256),)](
                store.wanted_groups, store.wanted_count, self.group_to_slot,
                self.age, self.clock, store.miss_groups, store.miss_count, m, 256,
                debug=store.debug,
            )  # fmt: skip
        shadow = store.shadow if self.index == 0 else None
        if self.index == 0 and store.hot_metrics is not None:
            store.hot_metrics.count(store)
        if shadow is not None:
            shadow.check_misses(store, triton.cdiv(m, 256))
        kernels.count_victim_buckets[(store.victim_tiles,)](
            store.wanted, self.slot_owner, self.age, self.clock, store.miss_count,
            store.victim_counts, self.eviction_hand, s.hot_slots,
            store.victim_tiles, 512,
        )  # fmt: skip
        kernels.select_bucket_victims[(store.victim_tiles,)](
            store.wanted, self.slot_owner, self.age, self.clock, store.miss_count,
            store.victim_counts, store.victim_slots, self.eviction_hand,
            s.hot_slots, store.victim_tiles, 512,
            triton.next_power_of_2(store.victim_tiles), debug=True,
        )  # fmt: skip
        if shadow is not None:
            # Before allocation overwrites the victims' owners.
            shadow.record_victims(store, self.slot_owner, triton.cdiv(m, 256))
        kernels.allocate_slots[(triton.cdiv(m, 256),)](
            self.group_to_slot, self.slot_owner, self.age, self.clock,
            store.miss_groups, store.miss_slots, store.miss_count,
            store.victim_slots, store.victim_slots, self.eviction_hand, m,
            s.hot_slots, 256, False, debug=store.debug,
        )  # fmt: skip
        hk.gather_groups[(store.gather_workers,)](
            self.host_device, self.hot, store.ready, store.miss_groups,
            store.miss_slots, store.miss_count, store.page_of,
            *self.host_view.stride()[:5], s.kernel_blocks_per_block,
            s.kernel_block_tokens, s.kv_heads, s.head_dim,
            triton.next_power_of_2(8 * s.kv_heads * s.head_dim),
        )  # fmt: skip
        if classed:
            assert counters is not None
            hk.age_prefill_misses[(triton.cdiv(m, 256),)](
                store.miss_groups, store.miss_slots, store.miss_count,
                store.wanted, self.age, self.clock, counters, 256,
            )  # fmt: skip
        if counters is not None:
            hk.accumulate_union[(1,)](store.wanted_count, store.miss_count, counters)

    # ----- selection-union gather union windows (mixed steps)
    # --------------------------------

    def _union_buffers(self, rows: int, table_numel: int) -> dict[str, torch.Tensor]:
        """Lazily sized union scratch, shared by the layers (one stream)."""
        store = self.store
        scratch = store.union_scratch
        i32 = dict(dtype=torch.int32, device=store.device)
        groups = triton.cdiv(rows, 1)
        if "scan" not in scratch or scratch["scan"].numel() < groups + 1:
            scratch["scan"] = torch.zeros(groups + 1, **i32)
        if "counters" not in scratch:
            scratch["counters"] = torch.zeros(4, dtype=torch.int64, device=store.device)
        if "table" not in scratch or scratch["table"].numel() < table_numel:
            scratch["table"] = torch.empty(max(table_numel, 1), **i32)
        if "requests" not in scratch or scratch["requests"].numel() < rows:
            scratch["requests"] = torch.empty(rows, **i32)
        return scratch

    def _union_scan(
        self,
        mode: int,
        selection: torch.Tensor,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        req_class: torch.Tensor,
        scan: torch.Tensor,
        num_row_groups: int = 0,
        *,
        tails: bool = True,
    ) -> None:
        """One pass of ``union_tail``/``union_selection`` in ``mode``.

        ``selection`` rows start on an eight-row boundary of the step."""
        store, s = self.store, self.store.shape
        geometry = (s.dcp_world_size, s.dcp_rank, s.interleave)
        tail_code, sentinel = num_row_groups + 1, num_row_groups + 2
        if tails:
            hk.union_tail[(seq_lens.shape[0],)](
                block_table, seq_lens, store.ready, req_class, store.wanted,
                store.wanted_groups, store.wanted_count, scan, tail_code,
                sentinel, block_table.stride(0), s.kernel_block_tokens,
                store.max_misses, *geometry, mode, debug=store.debug,
            )  # fmt: skip
        rows, width = selection.shape
        if rows == 0:
            return
        hk.union_selection[(rows, triton.cdiv(width, 128))](
            selection, token_to_req, block_table, seq_lens, store.ready,
            req_class, store.wanted, store.wanted_groups, store.wanted_count,
            scan, num_row_groups, sentinel, selection.stride(0),
            block_table.stride(0), width, s.kernel_block_tokens,
            store.max_misses, 128, *geometry, mode, _UNION_ROW_GROUP,
            debug=store.debug,
        )  # fmt: skip

    def _union_clear(self) -> None:
        """Clear the previous fill's marks (a complete Touched list)."""
        store = self.store
        m = store.max_misses
        kernels.clear_wanted[(triton.cdiv(m, 256),)](
            store.wanted, store.wanted_groups, store.wanted_count, 256
        )
        store.wanted_count.zero_()

    def build_microblock_table(
        self, block_table: torch.Tensor, seq_lens: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """Page-size-4 table of the hot slots, one row per block-table row.

        Its logical capacity equals the block table's (``width * kbs`` local
        tokens), so the grouped route's synthetic positions/lengths are the
        GPU arm's. Valid until the next fill of any layer."""
        s = self.store.shape
        width = block_table.shape[1] * s.groups_per_kernel_block
        table = out[: block_table.shape[0] * width].view(block_table.shape[0], width)
        hk.build_microblock_table[
            (block_table.shape[0], triton.cdiv(width, 256))
        ](
            block_table, seq_lens, self.group_to_slot, table, seq_lens.shape[0],
            block_table.stride(0), width, s.kernel_block_tokens, 256,
            s.dcp_world_size, s.dcp_rank, s.interleave,
        )  # fmt: skip
        return table

    def _cpu_validate(
        self,
        selection: torch.Tensor,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        check_capacity: bool = True,
    ) -> None:
        """Torch checks for the interpreter, where device asserts are no-ops."""
        s = self.store.shape
        ready = self.store.ready
        world, rank, il = s.dcp_world_size, s.dcp_rank, s.interleave
        span = world * il

        def local(length: int) -> int:
            return length // span * il + min(max(length % span - rank * il, 0), il)

        def slot(req: int, token: int) -> int:
            kb = int(block_table[req, token // s.kernel_block_tokens])
            return kb * s.kernel_block_tokens + token % s.kernel_block_tokens

        lengths = [local(int(x)) for x in seq_lens.tolist()]
        groups = set()
        for req, length in enumerate(lengths):
            for token in range(max(0, length - 32), length):
                if not ready[slot(req, token)]:
                    raise ValueError("tail KV is not committed")
                groups.add(slot(req, token) // 4)
        if (token_to_req < 0).any() or (token_to_req >= len(lengths)).any():
            raise ValueError("invalid host history request row")
        for row, req in zip(selection.tolist(), token_to_req.tolist()):
            row_groups = set()
            for token in row:
                if token < 0:
                    continue
                if token >= lengths[req]:
                    raise ValueError("host history selection past length")
                if not ready[slot(req, token)]:
                    raise ValueError("selected KV is not committed")
                row_groups.add(slot(req, token) // 4)
            if len(row_groups) > s.row_groups:
                raise ValueError("host history QSA group bound exceeded")
            groups |= row_groups
        if check_capacity and len(groups) > self.store.max_misses:
            raise ValueError("host history hot capacity exceeded")
        if not check_capacity:
            # Union path: what gather_groups asserts on the device.
            ratio = s.kernel_blocks_per_block
            per_group = s.groups_per_kernel_block
            pages = self.store.page_of[
                torch.tensor(sorted({g // per_group // ratio for g in groups})).long()
            ]
            if (pages < 0).any():
                raise ValueError("unbound host page read")


def for_each_pass(
    layer: HostHistoryLayer,
    selection: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    attend,
    *,
    passes: Sequence[tuple[int, int]] | None = None,
    single_pass: bool = False,
) -> None:
    """Gather and attend query rows in passes that fit the hot pool.

    ``attend(start, end, hot_ids)`` must consume the hot cache for rows
    ``[start, end)`` before returning (on the same stream): the next pass may
    evict the groups it read. A step that fits one pass keeps the original
    row count and therefore the original split/merge plan."""
    rows = selection.shape[0]
    if single_pass:
        if rows > layer.store.shape.read_rows:
            raise ValueError("Decode rows exceed host-history R_cap")
        passes = [(0, rows)] if rows else []
    if passes is None and not single_pass:
        passes = getattr(layer.store, "step_passes", None)
        if passes is not None and (not passes or passes[-1][1] != rows):
            raise ValueError("Host-history step plan does not cover the rows")
    if passes is None:
        # No CPU batch metadata (dummy/profile runs, static pool): the
        # conservative plan, charging every request's tail, over the slots
        # available this step, without reading GPU tensors.
        from .planning import plan_passes

        s = layer.store.shape
        row_groups, tail_groups, groups = s.conservative_pass_geometry()
        passes = plan_passes(
            [rows] + [0] * (s.max_requests - 1),
            groups,
            getattr(layer.store, "available_slots", s.hot_slots),
            s.read_rows,
            row_groups=row_groups,
            tail_groups=tail_groups,
        )
    cursor = 0
    for start, end in passes:
        if (
            start != cursor
            or not start < end <= rows
            or end - start > layer.store.shape.read_rows
        ):
            raise ValueError("Invalid host-history pass intervals")
        hot_ids = layer.gather(
            selection[start:end], token_to_req[start:end], block_table, seq_lens
        )
        attend(start, end, hot_ids)
        cursor = end
    if cursor != rows:
        raise ValueError("Host-history pass intervals do not cover the query rows")


@dataclass
class UnionWindowResult:
    """What one union call did (host-side, no device read needed)."""

    level: str
    """``l0`` (CPU bound, no sync), ``l1`` (exact probe, one window) or
    ``l2`` (exact greedy windows, with ``l3`` per-row groups if any)."""
    windows: int
    syncs: int
    l3_groups: int
    spans: list[tuple[int, int, str]]
    """Executed row spans in order: ``(start, end, "union" | "rows")``."""


def for_each_union_window(
    layer: HostHistoryLayer,
    selection: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    outputs: Sequence[torch.Tensor],
    attend_union,
    attend_rows,
    *,
    capacity: int,
    fast_path: bool = False,
    req_class: torch.Tensor | None = None,
    max_window_groups: int | None = None,
) -> UnionWindowResult:
    """Attend a mixed step from the union of its selections (selection-union gather,
    selection-union gather).

    ``attend_union(table, requests, outs)`` runs the unchanged attention on
    every row with the original ``selection``, the hot K/V and the page-size-4
    ``table``, writing all rows of ``outs`` (same shapes as ``outputs``); rows
    whose request is -1 belong to another window and are discarded.
    ``attend_rows(start, end, hot_ids)`` is the remap-pass callback, used for
    an eight-row group that does not fit alone (L3); it writes ``outputs``
    rows directly.

    ``capacity`` is the step's hot capacity U, identical on every rank. Each
    window holds whole eight-row groups whose exact union, every request's
    tail counted once, is at most U. A window's table and hot slots are valid
    only until its attention is consumed; the next window rebuilds both on
    the same stream. ``fast_path`` (L0) asserts a CPU bound already proved
    that the whole step fits; otherwise one host read decides (L1) and each
    further window costs one host read (L2). Rank-local window counts may
    differ: no collective runs between windows.

    ``req_class`` holds per-request class bits (2 decode, 1 prefill) when
    ``prefill_insert_age`` is ``old``. ``max_window_groups`` caps windows (a
    test and A/B hook; any legal split gives the same bits)."""
    store, s = layer.store, layer.store.shape
    if store.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("host-history union windows synchronize; not in a graph")
    rows, width = selection.shape
    requests = seq_lens.shape[0]
    if (
        selection.ndim != 2
        or not 0 < rows <= s.max_write_rows
        or width > s.selection_width
        or selection.dtype != torch.int32
        or selection.stride(1) != 1
        or token_to_req.shape != (rows,)
        or token_to_req.dtype != torch.int32
        or not token_to_req.is_contiguous()
        or block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.stride(1) != 1
        or block_table.shape[0] < requests
        or not 0 < requests <= s.max_requests
        or seq_lens.ndim != 1
        or not seq_lens.is_contiguous()
        or not 0 < capacity <= s.hot_slots
        or (max_window_groups is not None and max_window_groups < 1)
        or any(o.shape[0] != rows for o in outputs)
    ):
        raise ValueError("Invalid host history union tensors")
    if store.device.type == "cpu":
        layer._cpu_validate(
            selection, token_to_req, block_table, seq_lens, check_capacity=False
        )
    classed = s.prefill_insert_age == "old"
    if req_class is None:
        req_class = torch.ones(requests, dtype=torch.int32, device=store.device)
    elif req_class.shape != (requests,) or req_class.dtype != torch.int32:
        raise ValueError("Invalid host history request classes")
    if not classed:
        # Bits only steer insertion age; without the policy every claim is 1.
        req_class = torch.ones(requests, dtype=torch.int32, device=store.device)
    width_mb = block_table.shape[1] * s.groups_per_kernel_block
    scratch = layer._union_buffers(rows, block_table.shape[0] * width_mb)
    scan, counters = scratch["scan"], scratch["counters"]
    stats = store.union_stats
    stats["calls"] += 1
    step = _UNION_ROW_GROUP
    groups8 = triton.cdiv(rows, step)
    single = max_window_groups is None or max_window_groups >= groups8
    args = (block_table, seq_lens, req_class, scan)

    def fill_and_attend(start: int, end: int, outs) -> None:
        # Marks for [start, end) are in place (Touched complete).
        hk.begin_fill[(1,)](store.miss_count, layer.clock)
        layer._fill_from_marks(classed=classed, counters=counters)
        table = layer.build_microblock_table(block_table, seq_lens, scratch["table"])
        if start == 0 and end == rows:
            attend_union(table, token_to_req, outs)
            return
        masked = scratch["requests"][:rows]
        masked.copy_(token_to_req)
        if _MASK_OTHER_WINDOWS:
            masked[:start].fill_(-1)
            masked[end:].fill_(-1)
        attend_union(table, masked, outs)

    layer._union_clear()
    syncs = 0
    if single:
        layer._union_scan(hk.UNION_MARK, selection, token_to_req, *args)
        if fast_path:
            if (store.debug or store.device.type == "cpu") and int(
                store.wanted_count.item()
            ) > capacity:
                raise RuntimeError("union fast path bound below the exact union")
            stats["l0"] += 1
            stats["windows"] += 1
            fill_and_attend(0, rows, outputs)
            return UnionWindowResult("l0", 1, 0, 0, [(0, rows, "union")])
        count = int(store.wanted_count.item())
        syncs += 1
        if count <= capacity:
            stats["l1"] += 1
            stats["windows"] += 1
            stats["syncs"] += syncs
            fill_and_attend(0, rows, outputs)
            return UnionWindowResult("l1", 1, syncs, 0, [(0, rows, "union")])
        # Overflow: Touched may be incomplete. Rescan the same inputs to
        # restore every claimed gid to zero; residency, ages and the clock
        # were never touched.
        stats["probe_overflows"] += 1
        layer._union_scan(hk.UNION_RESET, selection, token_to_req, *args)
        store.wanted_count.zero_()

    from .planning import union_window_groups

    temporaries = [torch.empty_like(o) for o in outputs]
    spans: list[tuple[int, int, str]] = []
    windows = l3_groups = 0
    group = 0
    while group < groups8:
        layer._union_clear()
        start = step * group
        rest_sel, rest_req = selection[start:], token_to_req[start:]
        remaining = groups8 - group
        scan[: remaining + 1].zero_()
        layer._union_scan(hk.UNION_FIRST, rest_sel, rest_req, *args, remaining)
        layer._union_scan(hk.UNION_COUNT, rest_sel, rest_req, *args, remaining)
        layer._union_scan(hk.UNION_RESET, rest_sel, rest_req, *args, remaining)
        counts = scan[: remaining + 1].tolist()
        syncs += 1
        limit = None if max_window_groups is None else max_window_groups
        taken = union_window_groups(counts[0], counts[1:], capacity, limit)
        if taken == 0:
            # L3: this eight-row group alone exceeds U; run its rows through
            # the remap passes, one row per pass (a row and every tail fit
            # the floor). Outputs are written directly.
            end = min(start + step, rows)
            for_each_pass(
                layer,
                selection[start:end],
                token_to_req[start:end],
                block_table,
                seq_lens,
                lambda a, b, ids, base=start: attend_rows(base + a, base + b, ids),
                passes=[(i, i + 1) for i in range(end - start)],
            )
            spans.append((start, end, "rows"))
            l3_groups += 1
            group += 1
            continue
        end = min(step * (group + taken), rows)
        layer._union_scan(
            hk.UNION_MARK, selection[start:end], token_to_req[start:end], *args
        )
        if store.debug or store.device.type == "cpu":
            exact = counts[0] + sum(counts[1 : taken + 1])
            if int(store.wanted_count.item()) != exact:
                raise RuntimeError("union window marks differ from its plan")
        fill_and_attend(start, end, temporaries)
        for output, temporary in zip(outputs, temporaries, strict=True):
            output[start:end].copy_(temporary[start:end])
        spans.append((start, end, "union"))
        windows += 1
        group += taken
    stats["l2"] += 1
    stats["windows"] += windows
    stats["l3_groups"] += l3_groups
    stats["syncs"] += syncs
    return UnionWindowResult("l2", windows, syncs, l3_groups, spans)


@dataclass(frozen=True)
class UnionStepInfo:
    """CPU batch metadata of one step for the union path (supplied by shared hot
    region).

    ``capacity`` is U_step, identical on every rank. ``rows_by_req`` and
    ``groups`` (from the maximum rank-local length) follow the block-table
    row order; ``decode`` marks decode-class requests (a request's last
    prefill chunk may be marked decode to keep its pages recent)."""

    capacity: int
    rows_by_req: tuple[int, ...] | None = None
    groups: tuple[int, ...] | None = None
    decode: tuple[bool, ...] | None = None


def run_union_windows(
    layer: HostHistoryLayer,
    selection: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    outputs: Sequence[torch.Tensor],
    attend_union,
    attend_rows,
) -> UnionWindowResult:
    """Model entry: derive U, the L0 decision and classes, then run windows.

    With ``store.union_step`` (this step's ``UnionStepInfo``) the L0 bound is
    exact per request. Without it the static pool's capacity and a shape
    bound are used: ``sum_q min(G_q, rg*r_q + tg) <= min(n*G_max, rg*rows +
    tg*n)``, and classes come from per-request row counts on the device."""
    from .planning import history_groups, union_fast_path

    store, s = layer.store, layer.store.shape
    rows, requests = selection.shape[0], seq_lens.shape[0]
    info: UnionStepInfo | None = getattr(store, "union_step", None)
    req_class = None
    if info is not None:
        capacity = min(info.capacity, s.hot_slots)
        fast = (
            info.rows_by_req is not None
            and info.groups is not None
            and len(info.rows_by_req) == requests
            and sum(info.rows_by_req) == rows
            and union_fast_path(
                info.rows_by_req,
                info.groups,
                capacity,
                row_groups=s.row_groups,
                tail_groups=s.tail_groups,
            )
        )
        if info.decode is not None and len(info.decode) == requests:
            req_class = torch.tensor(
                [2 if d else 1 for d in info.decode], dtype=torch.int32
            ).to(store.device, non_blocking=True)
    else:
        # Static pool, or no CPU batch metadata (dummy/profile runs): the
        # slots usable this step (shared hot region keeps reserved shared-region slots
        # out).
        capacity = min(getattr(store, "available_slots", s.hot_slots), s.hot_slots)
        bound = s.row_groups * rows + s.tail_groups * requests
        if s.max_context_tokens is not None:
            g_max = history_groups(s.max_context_tokens, s.dcp_world_size, s.interleave)
            bound = min(bound, requests * g_max)
        fast = bound <= capacity
    if req_class is None and s.prefill_insert_age == "old":
        counts = torch.zeros(requests, dtype=torch.int32, device=store.device)
        counts.index_add_(0, token_to_req.long(), torch.ones_like(token_to_req))
        req_class = torch.where(counts <= s.decode_rows, 2, 1).to(torch.int32)
    return for_each_union_window(
        layer,
        selection,
        token_to_req,
        block_table,
        seq_lens,
        outputs,
        attend_union,
        attend_rows,
        capacity=capacity,
        fast_path=fast,
        req_class=req_class,
    )
