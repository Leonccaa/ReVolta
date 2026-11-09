# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine host history store: main QSA K/V addressed by history block ID.

In long-context mode with the main K/V in the host pool, every history block
ID of the scheduler's GPU history segment owns exactly one host page per QSA
layer; the same ID addresses the block's compressed-index page on the GPU.
The host allocation is made once, for every history ID (null block 0
included), so a host page lives exactly as long as its history block: it is
written while the block's request computes, kept while the block sits in the
prefix cache, and reset when the scheduler hands the ID to a new allocation
(``begin_blocks``, driven by the same new-block list that drives KV zeroing).
Nothing here allocates or evicts host pages on its own.

Layout. One pinned allocation ``[num_blocks, ratio, layers, 2, kbs, H, D]``:
a history block's pages for all layers are one contiguous byte range
(``host_page``), and each layer sees a kernel-block view
``[num_blocks * ratio, 2, kbs, H, D]`` that matches the GPU paged layout, so
the engine's slot mapping and block tables address it unchanged.

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
from dataclasses import dataclass

import torch

from vllm.triton_utils import triton
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)

from . import history_kernels as hk
from . import kernels
from .budget import (
    HISTORY_METADATA,
    QSA_ROW_GROUPS,
    TAIL_GROUPS,
    history_fixed_memory_bytes,
)
from .pool import HostKVResourceError, UnsupportedHostKVConfig, _check_product


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
        return (self.hot_slots - self.max_requests * self.tail_groups) // (
            self.row_groups
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
        """Required hot-pool reservation, by tensor name (no rounding)."""
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
        )
        if any(type(n) is not int or n <= 0 for n in sizes) or self.tail_groups < 0:
            raise ValueError("Host history dimensions must be positive integers")
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
            ("selection", (self.rows_per_pass, self.selection_width)),
            ("group transfer elements", (8, self.kv_heads, self.head_dim)),
            ("selection grid Y", (triton.cdiv(self.selection_width, 128),)),
        ):
            limit = {
                "group transfer elements": 16384,
                "selection grid Y": 65535,
            }.get(name, 2**31 - 1)
            _check_product(name, *factors, limit=limit)


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
    ) -> None:
        shape.validate()
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
        flat = _allocate_host(math.prod(dims), s.dtype)
        if device.type == "cuda":
            _register_host(flat)
            # Unregister only when the store goes away; views must not
            # outlive it.
            weakref.finalize(self, _unregister_host, flat)
        self.host = flat.view(dims)
        try:
            if device.type == "cuda":
                with torch.accelerator.device_index(device.index):
                    self.host_device = get_accelerator_view_from_cpu_tensor(self.host)
            else:
                self.host_device = self.host
            i32 = dict(dtype=torch.int32, device=device)
            self.ready = torch.zeros(kb * kbs, dtype=torch.uint8, device=device)
            # The null block is never written; reading it yields zeros.
            self.ready[: s.kernel_blocks_per_block * kbs] = 1
            self.generation = torch.zeros(s.num_blocks, **i32)
            slots = s.hot_slots
            self.hot = torch.zeros(
                (layers, slots, 2, 4, h, d), dtype=s.dtype, device=device
            )
            self.group_to_slot = torch.full((layers, s.num_groups), -1, **i32)
            self.slot_owner = torch.full((layers, slots), -1, **i32)
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
            self.indices = torch.empty((s.rows_per_pass, s.selection_width), **i32)
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
        except (RuntimeError, MemoryError) as error:
            raise HostKVResourceError(
                "Host history GPU metadata or UVA allocation failed"
            ) from error
        self.layers = [
            HostHistoryLayer(self, index, name)
            for index, name in enumerate(layer_names)
        ]
        self.layer_index = {name: index for index, name in enumerate(layer_names)}

    def fixed_device_memory_bytes(self) -> dict[str, int]:
        """Actual fixed tensor bytes, excluding host pages and history metadata."""
        return {
            name: value.numel() * value.element_size()
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor)
            and name not in HISTORY_METADATA | {"host", "host_device"}
        }

    # ----- interface for the scheduler-facing tiers (host prefix adapter)
    # -----------------

    def host_page(self, block_id: int) -> torch.Tensor:
        """Pinned CPU bytes of one history block, all layers:
        ``[ratio, layers, 2, kernel_block_tokens, H, D]``, contiguous."""
        if not 0 <= block_id < self.shape.num_blocks:
            raise IndexError(f"history block {block_id} is out of range")
        return self.host[block_id]

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

    def mark_ready(self, block_ids: list[int] | torch.Tensor) -> None:
        """Publish whole pages filled outside the forward pass (a RAM/SSD
        tier restoring main K/V into newly begun IDs, host prefix adapter).

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
        self.host_device = store.host_device[:, :, index].flatten(0, 1)
        self.host_view = store.host[:, :, index].flatten(0, 1)
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
            self.group_to_slot, key.stride(0), key.stride(1), value.stride(0),
            value.stride(1), *self.host_view.stride()[:4], s.kernel_block_tokens,
            s.kv_heads, s.head_dim, triton.next_power_of_2(s.kv_heads * s.head_dim),
        )  # fmt: skip
        # A separate kernel boundary publishes rows only after both copies.
        hk.commit_slots[(triton.cdiv(n, 256),)](slot_mapping, self.store.ready, n, 256)

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
            or not 0 < rows <= s.rows_per_pass
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
            store.miss_slots, store.miss_count, *self.host_view.stride()[:4],
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
) -> None:
    """Gather and attend query rows in passes that fit the hot pool.

    ``attend(start, end, hot_ids)`` must consume the hot cache for rows
    ``[start, end)`` before returning (on the same stream): the next pass may
    evict the groups it read. A step that fits one pass keeps the original
    row count and therefore the original split/merge plan."""
    rows = selection.shape[0]
    step = layer.rows_per_pass
    for start in range(0, rows, step):
        end = min(rows, start + step)
        hot_ids = layer.gather(
            selection[start:end], token_to_req[start:end], block_table, seq_lens
        )
        attend(start, end, hot_ids)
