# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated host KV DCP2 experiment support. No engine path imports this module."""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from vllm.models.qwen4_exp.nvidia.host_kv import (
    ActiveHostKV,
    HostKVConfig,
    LayerScales,
)
from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import (
    _localize_indices,
    qsa_dcp_local_selection_width,
    qsa_localize_dcp_indices,
)
from vllm.triton_utils import triton
from vllm.v1.attention.kv_cache_view import bind_attention_cache
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

T = TypeVar("T")


def enable_cpu_interpreter() -> None:
    """Repair Triton 3.6 interpreter comparison typing, as host KV pool tests do."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU mode requires TRITON_INTERPRET=1")
    import numpy as np
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    from vllm.triton_utils import tl

    def greater(_builder, lhs, rhs):
        return TensorHandle(np.greater(lhs.data, rhs.data), tl.int1)

    InterpreterBuilder.create_fcmpOGT = greater


class RankFailure(RuntimeError):
    """Every surviving rank receives the same phase and error list."""


class ExperimentGroup:
    """Torch collectives with the GroupCoordinator interface used by DCP.

    All ranks create contiguous pairs in the same order. The world gloo group
    is an independent control plane, including when CUDA is in an error state.
    Process/collective loss is bounded by process-group timeouts and torchrun.
    """

    def __init__(self, backend: str, timeout: int = 60) -> None:
        self.rank = dist.get_rank()
        total = dist.get_world_size()
        if total not in (1, 4):
            raise ValueError("Use one probe rank or four TP4/DCP2 ranks")
        limit = timedelta(seconds=timeout)
        self.control = dist.new_group(list(range(total)), backend="gloo", timeout=limit)
        self.world_size = min(2, total)
        self.rank_in_group = self.rank % self.world_size
        self.device_group = None
        for start in range(0, total, self.world_size):
            ranks = list(range(start, start + self.world_size))
            group = dist.new_group(ranks, backend=backend, timeout=limit)
            if self.rank in ranks:
                self.device_group = group
        self.backend = backend
        self.epoch = 0

    def agree(
        self,
        phase: str,
        error: str | None = None,
        *,
        contract: str | None = None,
    ) -> None:
        messages: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(
            messages,
            (self.epoch, phase, error, contract),
            group=self.control,
        )
        expected_epoch = self.epoch
        self.epoch += 1
        if any(
            n != expected_epoch or p != phase or e is not None or c != contract
            for n, p, e, c in messages
        ):
            raise RankFailure(f"{phase}: {messages}")

    def local(self, phase: str, fn: Callable[[], T], *, cuda: bool = False) -> T:
        error = None
        result = None
        try:
            result = fn()
            if cuda:
                torch.accelerator.synchronize()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.agree(phase, error)
        return result  # type: ignore[return-value]

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensors, tensor.contiguous(), group=self.device_group)
        return torch.cat(tensors, dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        # Identical rank-major packing to the normal communicator. Gloo lacks
        # reduce_scatter, so CPU tests use SUM all-reduce then the head slice.
        if self.backend == "gloo":
            dist.all_reduce(tensor, group=self.device_group)
            return tensor.chunk(self.world_size, dim=dim)[
                self.rank_in_group
            ].contiguous()
        packed = torch.movedim(tensor, dim, 0).contiguous()
        out = torch.empty_like(packed.chunk(self.world_size, dim=0)[0])
        dist.reduce_scatter_tensor(out, packed, group=self.device_group)
        return torch.movedim(out, 0, dim).contiguous()

    def combine(self, out: torch.Tensor, lse: torch.Tensor, backend: str):
        if self.world_size == 1:
            return out, lse
        fn = dcp_a2a_lse_reduce if backend == "a2a" else cp_lse_ag_out_rs
        return fn(out, lse, self, return_lse=True, is_lse_base_on_e=False)


@dataclass(frozen=True)
class Profile:
    tokens: int = 524287  # global length, deliberately odd
    queries: int = 8  # per request
    batch_queries: int = 2
    topk: int = 2048
    union_groups: int = 1536  # per request across all query rows
    dim: int = 256
    heads: int = 6  # TP4 local query heads, 12 after DCP query gather
    page: int = 832
    members: int = 2
    hot_ratio: float = 0.1
    e4m3: bool = True

    def config(self, world: int) -> HostKVConfig:
        if (
            self.tokens < 65
            or min(self.queries, self.batch_queries, self.topk, self.dim, self.heads)
            <= 0
            or self.queries % self.batch_queries
            or self.topk % 4
            or not self.topk // 4 <= self.union_groups <= self.topk // 4 * self.queries
            or self.union_groups > (self.tokens - self.queries - 50) // 4
            or self.page % 4
            or self.members not in (1, 2)
            or self.hot_ratio not in (0.1, 0.2)
        ):
            raise ValueError("Invalid fixed DCP2 experiment profile")
        width = qsa_dcp_local_selection_width(self.topk, 4, world, 1, self.topk + 3)
        config = HostKVConfig(
            8,
            triton.cdiv(self.tokens, world),
            8 * self.batch_queries,
            width,
            4096,
            hot_ratio=self.hot_ratio,
        )
        config.capacities()
        return config


def local_length(length: torch.Tensor, world: int, rank: int) -> torch.Tensor:
    """Count owned positions in [0,length), interleave=1, including odd tails."""
    return (length + world - 1 - rank) // world


def reset_hot(pool: ActiveHostKV) -> None:
    pool.group_to_slot.fill_(-1)
    pool.slot_owner.fill_(-1)
    pool.age.zero_()
    pool.clock.zero_()
    pool.wanted.zero_()
    pool.wanted_count.zero_()
    pool.miss_count.zero_()


def error_metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    a, b = expected.detach().cpu().contiguous(), actual.detach().cpu().contiguous()
    bits = a.view(torch.uint8).reshape(*a.shape, a.element_size())
    other = b.view(torch.uint8).reshape_as(bits)
    different = (bits != other).any(-1)
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a.float()[finite] - b.float()[finite]).abs()
    values = (
        torch.quantile(delta, torch.tensor([0.5, 0.95, 0.99])).tolist()
        if delta.numel()
        else [0.0, 0.0, 0.0]
    )
    first = different.nonzero()
    return dict(
        bitwise=not bool(different.any()),
        mismatches=int(different.sum()),
        elements=a.numel(),
        first=first[0].tolist() if first.numel() else None,
        max_abs=float(delta.max()) if delta.numel() else 0.0,
        p50_abs=values[0],
        p95_abs=values[1],
        p99_abs=values[2],
        nonfinite_mismatches=int((~finite & (a != b)).sum()),
    )


class LayerExperiment:
    """A packed GPU source, B GPU hot view, C host KV pool host pool, fixed operands.

    All data is synthetic and deterministic. We replay fixed compressed-group
    choices through the real expansion/localization kernels, without an index
    scorer or model weights. Setup seeds encoded host bytes directly, avoiding
    millions of tiny write launches. Native quantization has its host KV pool GPU gates.
    """

    def __init__(
        self,
        profile: Profile,
        group: ExperimentGroup,
        device: torch.device,
        route: str = "auto",
        combine: str = "ag_rs",
    ) -> None:
        self.p, self.group, self.device = profile, group, device
        self.route, self.combine_backend = route, combine
        self.config = c = profile.config(group.world_size)
        self.scales = LayerScales(0.375, 1.75)
        self.dtype = torch.uint8 if profile.e4m3 else torch.float16
        columns = triton.cdiv(c.max_tokens, profile.page)
        pages = c.requests * columns
        shape = (pages, 2, profile.page, 1, profile.dim)
        page_bytes = (
            2
            * profile.page
            * profile.dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        raw = torch.zeros(
            pages * profile.members * page_bytes, dtype=torch.int8, device="meta"
        )
        layout = bind_attention_cache(
            raw,
            shape=shape,
            dtype=self.dtype,
            page_size_bytes=page_bytes,
            scheduler_block_size=profile.page,
            kernel_block_size=profile.page,
            member_index=profile.members - 1,
            member_count=profile.members,
        )
        generator = torch.Generator().manual_seed(2002)
        table = torch.randperm(pages, generator=generator).int().reshape(8, columns)
        self.pool = pool = ActiveHostKV(
            c, layout, device=device, scales=self.scales, block_table=table
        )
        pool.host.zero_()
        # Each owner rank has different operands, reproducible for the oracle.
        generator.manual_seed(2400 + group.rank_in_group + 13 * (group.rank // 2))
        for req in range(8):
            payload = (
                torch.randn(
                    (columns, 2, profile.page, 1, profile.dim), generator=generator
                )
                * 0.3
            ).half()
            if profile.e4m3:
                payload[:, 0] = payload[:, 0] / self.scales.key
                payload[:, 1] = payload[:, 1] / self.scales.value
                payload = payload.to(torch.float8_e4m3fn).view(torch.uint8)
            pool.host[table[req].long()] = payload
        # Preserve the packed allocation/offset through the production binder.
        source_raw = torch.zeros(raw.numel(), dtype=torch.int8, device=device)
        self.source = bind_attention_cache(
            source_raw,
            shape=shape,
            dtype=self.dtype,
            page_size_bytes=page_bytes,
            scheduler_block_size=profile.page,
            kernel_block_size=profile.page,
            member_index=profile.members - 1,
            member_count=profile.members,
        )
        self.source.copy_(pool.host)
        pool.ready.fill_(1)
        self.b_hot = torch.empty_like(pool.hot)
        self.requests = torch.arange(
            8, dtype=torch.int32, device=device
        ).repeat_interleave(profile.batch_queries)
        self.lengths = torch.empty(8, dtype=torch.int32, device=device)
        self.positions = torch.empty(c.rows, dtype=torch.int64, device=device)
        self.local_lengths = torch.empty_like(self.lengths)
        self.local_positions = torch.empty_like(self.positions)
        self.blocks = torch.empty(
            (c.rows, profile.topk // 4), dtype=torch.int32, device=device
        )
        self.selection = torch.empty(
            (c.rows, profile.topk + 3), dtype=torch.int32, device=device
        )
        self.local = torch.empty_like(self.selection)
        self.narrow = torch.empty(
            (c.rows, c.selection_width), dtype=torch.int32, device=device
        )
        self.hot_full = torch.full_like(self.selection, -1)
        self.query = torch.empty(
            (c.rows, profile.heads, profile.dim), dtype=torch.float16, device=device
        )
        self.gate = torch.empty_like(self.query)
        self.gathered = torch.empty(
            (c.rows, profile.heads * group.world_size, profile.dim),
            dtype=torch.float16,
            device=device,
        )
        self.partial = torch.empty_like(self.gathered, dtype=torch.float32)
        self.lse = torch.empty(
            self.gathered.shape[:2], dtype=torch.float32, device=device
        )
        self.output = torch.empty_like(self.query)
        self.cpu_plan = None
        if device.type == "cpu":
            self.cpu_plan = SparseAttentionPlan(
                rows=c.rows,
                query_heads=self.gathered.shape[1],
                kv_heads=1,
                head_dim=profile.dim,
                selection_width=c.selection_width,
                device=device,
                e4m3=profile.e4m3,
                scales=self.scales,
                output_dtype=torch.float32,
            )
        # Scatter compressed groups over the full history, preserving one
        # fixed permutation per request while the churn suffix advances.
        available = (profile.tokens - profile.queries - 50) // 4
        trace_generator = torch.Generator().manual_seed(2077)
        self.group_orders = [
            torch.randperm(available, generator=trace_generator) for _ in range(8)
        ]
        self.last_grouped = 0

    def prepare(
        self, step: int, batch: int, scenario: str, *, edge: bool = False
    ) -> None:
        p, c = self.p, self.config
        shift = 0 if scenario == "warm" else step % 3
        lengths = torch.tensor(
            [p.tokens - 2 * r - shift for r in range(8)], dtype=torch.int32
        )
        positions = (
            torch.stack(
                [
                    torch.arange(p.batch_queries)
                    + int(lengths[r])
                    - p.queries
                    + batch * p.batch_queries
                    for r in range(8)
                ]
            )
            .flatten()
            .long()
        )
        choices = []
        # Stable core + deterministic rotating suffix. Churn is a trace input,
        # never asserted to equal the actual miss fraction.
        count = p.topk // 4
        stable = (
            p.union_groups
            if scenario == "warm"
            else max(0, p.union_groups - max(1, p.union_groups // 5))
        )
        for row in range(positions.numel()):
            req, query = divmod(row, p.batch_queries)
            query += batch * p.batch_queries
            available = self.group_orders[req].numel()
            union = torch.arange(p.union_groups)
            rotating = union >= stable
            union[rotating] = stable + (
                union[rotating] - stable + step * max(1, p.union_groups // 5)
            ) % (available - stable)
            selected = union[
                (torch.arange(count) + query * p.union_groups // p.queries)
                % p.union_groups
            ]
            selected = self.group_orders[req][selected]
            choices.append(selected.int())
        blocks = torch.stack(choices)
        if edge:
            # Position zero has one globally visible token, absent on rank 1.
            # The next row has no selection on either owner.
            positions[0] = 0
            if c.rows > 1:
                positions[1] = -1
        self.blocks.copy_(blocks)
        self.lengths.copy_(lengths)
        self.positions.copy_(positions)
        self.local_lengths.copy_(
            local_length(lengths, self.group.world_size, self.group.rank_in_group)
        )
        self.local_positions.copy_(
            local_length(positions + 1, self.group.world_size, self.group.rank_in_group)
            - 1
        )
        gen = torch.Generator().manual_seed(3100 + step * 7 + batch + self.group.rank)
        self.query.copy_((torch.randn(self.query.shape, generator=gen) * 0.25).half())
        self.gate.copy_(torch.randn(self.gate.shape, generator=gen).half())

    def trace_digest(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.blocks, self.positions, self.lengths):
            digest.update(tensor.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def localize(self) -> None:
        if self.device.type == "cuda":
            qsa_ops.expand_qsa_block_indices_cuda(
                self.blocks,
                self.positions,
                self.lengths,
                self.requests,
                4,
                self.p.topk,
                self.selection,
            )
            qsa_localize_dcp_indices(
                self.selection,
                self.local,
                dcp_world_size=self.group.world_size,
                dcp_rank=self.group.rank_in_group,
                interleave_size=1,
                local_block_size=self.p.page,
            )
        else:
            # Public wrappers intentionally require CUDA. Execute their exact
            # underlying kernels in interpreter mode without editing wrappers.
            qsa_ops._expand_qsa_indices_kernel[
                (self.config.rows, triton.cdiv(self.p.topk + 3, 256))
            ](
                self.blocks,
                self.positions,
                self.lengths,
                self.requests,
                self.selection,
                *self.blocks.stride(),
                *self.selection.stride(),
                self.config.rows,
                8,
                BLOCK_TOPK=self.p.topk // 4,
                COMPRESS_RATIO=4,
                TOKEN_TOPK=self.p.topk,
                OUTPUT_WIDTH=self.p.topk + 3,
                COLUMN_BLOCK=256,
            )
            _localize_indices[(self.config.rows,)](
                self.selection,
                self.local,
                self.selection.stride(0),
                self.local.stride(0),
                WIDTH=self.p.topk + 3,
                WORLD=self.group.world_size,
                RANK=self.group.rank_in_group,
                INTERLEAVE=1,
                BLOCK=triton.next_power_of_2(self.p.topk + 3),
            )
        self.narrow.copy_(self.local[:, : self.config.selection_width])

    def gather(self) -> None:
        self.pool.gather(
            self.narrow, self.requests, self.local_lengths, self.local_positions
        )
        self.hot_full.fill_(-1)
        self.hot_full[:, : self.config.selection_width].copy_(self.pool.indices)

    def mirror_and_check(self) -> None:
        """Independent GPU-source -> B hot copy, exact C bytes and ID oracle.

        Inspection is deliberately outside timers and graphs. Only occupied
        slots are read. Invalid source page padding is never compared.
        """
        pool, c = self.pool, self.config
        if not bool((self.local[:, c.selection_width :] == -1).all()):
            raise AssertionError("local width lost selected columns")
        selected = self.selection.cpu()
        local = self.local.cpu()
        for row in range(c.rows):
            global_ids = selected[row][selected[row] >= 0]
            owned = (
                global_ids[
                    global_ids % self.group.world_size == self.group.rank_in_group
                ]
                // self.group.world_size
            )
            got = local[row][local[row] >= 0]
            if not torch.equal(got, owned):
                raise AssertionError("DCP localization changed logical order")
        owners = pool.slot_owner.cpu()
        slots = (owners >= 0).nonzero().flatten()
        gids = owners[slots].long()
        req, start = gids // c.groups_per_request, gids % c.groups_per_request * 4
        token = start[:, None] + torch.arange(4)
        pages = pool.block_table.cpu()[
            req[:, None], token.clamp_max(c.max_tokens - 1) // self.p.page
        ].long()
        within = token % self.p.page
        sk, sv = self.source.unbind(1)
        target = slots.to(self.device)
        pages, within = pages.to(self.device), within.to(self.device)
        valid = (token < c.max_tokens).to(self.device)
        bk, bv = self.b_hot.unbind(1)
        bk[target] = torch.where(valid[..., None, None], sk[pages, within], 0)
        bv[target] = torch.where(valid[..., None, None], sv[pages, within], 0)
        expected = self.b_hot[target].contiguous().view(torch.uint8)
        actual = pool.hot[target].contiguous().view(torch.uint8)
        if not torch.equal(expected, actual):
            raise AssertionError("B/C raw payload differs")
        mapped = pool.indices.cpu()
        reqs = self.requests.cpu()
        for row in range(c.rows):
            ids = mapped[row]
            valid_ids = ids >= 0
            recovered = owners[(ids[valid_ids] // 4).long()]
            if not bool((recovered // c.groups_per_request == reqs[row]).all()):
                raise AssertionError("hot slot belongs to another request")
            recovered = recovered % c.groups_per_request * 4 + ids[valid_ids] % 4
            if not torch.equal(recovered, local[row, : c.selection_width][valid_ids]):
                raise AssertionError("hot remap changed logical columns")
            if not torch.equal(valid_ids, local[row, : c.selection_width] >= 0):
                raise AssertionError("hot remap changed padding")

    def attention(self, arm: str) -> None:
        if arm == "A":
            key, value = self.source.unbind(1)
            indices, table = self.local, self.pool.block_table
        else:
            key, value = (self.b_hot if arm == "B" else self.pool.hot).unbind(1)
            indices, table = self.hot_full, self.pool.hot_block_table
        if self.cpu_plan is not None:
            self.partial.copy_(
                self.cpu_plan.run(
                    self.gathered,
                    key,
                    value,
                    indices[:, : self.config.selection_width].contiguous(),
                    table,
                    self.requests,
                )
            )
            self.lse.copy_(self.cpu_plan.lse)
            return
        grouped = 0
        if self.route != "triton":
            grouped = qsa_ops.qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
                self.gathered,
                key,
                value,
                indices,
                table,
                self.requests,
                self.partial,
                self.lse,
                "fp8_e4m3" if self.p.e4m3 else "auto",
                self.scales.key,
                self.scales.value,
            )
        if self.route == "grouped" and grouped == 0:
            raise RuntimeError(
                "Requested grouped route is unavailable for this native build/profile"
            )
        self.last_grouped = grouped
        if grouped < self.config.rows:
            qsa_ops.qsa_sparse_paged_attention(
                self.gathered[grouped:],
                key,
                value,
                indices[grouped:, : self.config.selection_width],
                table,
                self.requests[grouped:],
                out=self.partial[grouped:],
                lse=self.lse[grouped:],
                kv_cache_dtype="fp8_e4m3" if self.p.e4m3 else "auto",
                k_scale=self.scales.key,
                v_scale=self.scales.value,
            )

    def full_reference(self, operands: list[tuple[torch.Tensor, ...]]):
        """Original full-row GPU plan, with source pages shared read-only.

        The working C plan uses bounded batches. Reassemble request-major
        inputs before running the original 8*queries-row A plan, so changing
        batch size cannot silently disappear from the numerical comparison.
        """
        reference = copy.copy(self)
        reference.config = replace(self.config, rows=8 * self.p.queries)
        reference.query, reference.gate, reference.local = (
            self.join_batches([batch[i] for batch in operands]) for i in range(3)
        )
        reference.requests = torch.arange(
            8, dtype=torch.int32, device=self.device
        ).repeat_interleave(self.p.queries)
        reference.gathered = torch.empty(
            (reference.config.rows, self.gathered.shape[1], self.p.dim),
            dtype=torch.float16,
            device=self.device,
        )
        reference.partial = torch.empty_like(reference.gathered, dtype=torch.float32)
        reference.lse = torch.empty(
            reference.gathered.shape[:2], dtype=torch.float32, device=self.device
        )
        reference.output = torch.empty_like(reference.query)
        if self.cpu_plan is not None:
            reference.cpu_plan = SparseAttentionPlan(
                rows=reference.config.rows,
                query_heads=self.gathered.shape[1],
                kv_heads=1,
                head_dim=self.p.dim,
                selection_width=self.config.selection_width,
                device=self.device,
                e4m3=self.p.e4m3,
                scales=self.scales,
                output_dtype=torch.float32,
            )
        return reference

    def join_batches(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        combined = torch.stack(tensors)
        # [batch, request, query-in-batch, ...] -> [request, query, ...]
        combined = combined.reshape(
            len(tensors), 8, self.p.batch_queries, *combined.shape[2:]
        )
        order = (1, 0, 2, *range(3, combined.ndim))
        return (
            combined.permute(order)
            .reshape(8 * self.p.queries, *tensors[0].shape[1:])
            .contiguous()
        )

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        merged, lse = self.group.combine(self.partial, self.lse, self.combine_backend)
        self.output.copy_(merged)
        qsa_ops._qsa_output_gate(self.output, self.gate)
        return self.output, lse
