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
            self.staging_slots = torch.arange(
                s.max_write_rows, dtype=torch.int64, device=device
            )
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
            hk.set_block_slots[(ids.numel(), s.num_layers)](
                ids, self.group_to_slot, self.slot_owner, self.age, s.num_groups,
                s.hot_slots, s.slots_per_block, kp, True,
            )  # fmt: skip

    def set_step_plan(
        self,
        hot_slots: int | None,
        passes: list[tuple[int, int]] | None,
    ) -> None:
        """The scheduler's U_step and this rank's (identical) mixed-step plan."""
        if hot_slots is not None and self.shape.slots_per_block:
            self.available_slots = hot_slots
        self.step_passes = passes

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
        self.ready.zero_()
        self.ready[: s.kernel_blocks_per_block * s.kernel_block_tokens] = 1
        if self.region is not None:
            self.page_of.fill_(-1)
            self.page_of[0] = 0
            self.page_of_cpu[:] = -1
            self.page_of_cpu[0] = 0
        self.step_passes = None

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
        kernels.collect_misses[(triton.cdiv(m, 256),)](
            store.wanted_groups, store.wanted_count, self.group_to_slot, self.age,
            self.clock, store.miss_groups, store.miss_count, m, 256,
            debug=store.debug,
        )  # fmt: skip
        kernels.count_victim_buckets[(store.victim_tiles,)](
            store.wanted, self.slot_owner, self.age, self.clock, store.miss_count,
            store.victim_counts, self.eviction_hand, s.hot_slots,
            store.victim_tiles, 512,
        )  # fmt: skip
        kernels.select_bucket_victims[(store.victim_tiles,)](
            store.wanted, self.slot_owner, self.age, self.clock, store.miss_count,
            store.victim_counts, store.victim_slots, self.eviction_hand,
            s.hot_slots, store.victim_tiles, 512,
            triton.next_power_of_2(store.victim_tiles), debug=store.debug,
        )  # fmt: skip
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
        out = store.indices[:rows, :width]
        hk.remap_selection[grid](
            selection, token_to_req, block_table, seq_lens, self.group_to_slot,
            out, selection.stride(0), block_table.stride(0), out.stride(0), width,
            s.kernel_block_tokens, 128, *geometry, debug=store.debug,
        )  # fmt: skip
        return out

    def _cpu_validate(
        self,
        selection: torch.Tensor,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
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
        if len(groups) > self.store.max_misses:
            raise ValueError("host history hot capacity exceeded")


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
