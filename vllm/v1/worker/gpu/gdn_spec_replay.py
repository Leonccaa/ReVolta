# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Commit accepted GDN states by replaying the verifier (Model Runner V2, align mode).

With ``SpeculativeConfig.gdn_spec_state_replay`` a GDN cache group keeps two state
blocks per running request instead of ``2 + num_speculative_tokens``. The verifier
starts from the running block (the column the align bookkeeping already selects),
stores no per-position state, and records its inputs (see
``QwenGatedDeltaNet._spec_state_replay_verify``). After sampling, the same launch is
repeated per layer with an index table that maps

* the last accepted position to the running block (the next step's state), and
* a prefix-cache block boundary crossed before that position to its block,

so both states are the verifier's own. The align copies then move conv windows as
before; recurrent states need only the boundary copy when the boundary is the last
accepted position (``SPEC_STATE_REPLAY`` in ``mamba_align``).
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


@dataclass
class GDNSpecReplayLayer:
    layer: torch.nn.Module
    group_id: int
    # The verifier's running blocks ([num_reqs]) and cu_seqlens ([num_reqs + 1]).
    base_state_ids: torch.Tensor
    cu_seqlens: torch.Tensor


@dataclass
class GDNSpecReplayStep:
    # The verifier launch: request and token counts (CUDA-graph padded when the
    # step replayed a FULL graph), so the replay uses the same schedule.
    num_reqs: int
    num_tokens: int
    # Batch rows of the speculative requests, in verifier order ([num_spec]).
    spec_rows: torch.Tensor
    layers: list[GDNSpecReplayLayer]
    block_tables: tuple[torch.Tensor, ...]
    # The verifier replayed a FULL CUDA graph: every row is a speculative
    # request (padded to the captured batch) and the GDN metadata tensors are
    # the builder's persistent buffers, so the replay can be graph-captured.
    full_graph: bool = False


def build_replay_state_indices(
    *,
    base_state_ids: torch.Tensor,
    num_reqs: int,
    num_columns: int,
    spec_rows: torch.Tensor,
    num_accepted: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """The replay's state-index table for one cache group.

    Args:
        base_state_ids: running block per verifier row ([>= num_reqs]); padded
            rows hold PAD_SLOT_ID and the replay skips them.
        num_columns: verifier table width; its last column is the start block.
        spec_rows: batch row of each real speculative request, in row order.
        num_accepted: tokens committed per speculative request (>= 1),
            aligned with ``spec_rows``.
        num_computed_tokens: computed tokens after this step per speculative
            request, aligned with ``spec_rows``.
        block_table: the group's block table, indexed by batch row.
    """
    device = base_state_ids.device
    base = base_state_ids[:num_reqs].to(torch.int32)
    table = torch.full(
        (num_reqs, num_columns), PAD_SLOT_ID, dtype=torch.int32, device=device
    )
    table[:, -1] = base
    num_spec = spec_rows.numel()
    if num_spec == 0:
        return table
    rows = torch.arange(num_spec, device=device)
    accepted = num_accepted.to(torch.long).clamp(min=1)
    table[rows, accepted - 1] = base[:num_spec]
    # The state after accepted position t covers (computed - accepted + 1 + t)
    # tokens. A block boundary passed strictly before the last accepted
    # position gets its own prefix-cache snapshot.
    computed = num_computed_tokens.to(torch.long)
    running = computed - accepted + 1
    aligned = (computed // block_size) * block_size
    boundary_pos = aligned - running
    has_boundary = (aligned >= running) & (boundary_pos < accepted - 1)
    # Fixed-size writes, no host sync: a row without a boundary writes the
    # running block to its last accepted column again.
    dst_cols = (aligned // block_size - 1).clamp(min=0)
    dst_blocks = block_table[spec_rows.to(torch.long), dst_cols].to(torch.int32)
    cols = torch.where(has_boundary, boundary_pos, accepted - 1)
    table[rows, cols] = torch.where(has_boundary, dst_blocks, base[:num_spec])
    return table


@triton.jit
def _replay_state_table_kernel(
    table_ptr,
    table_stride,
    base_ptr,
    base_stride,
    spec_rows_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    num_computed_ptr,
    block_table_ptr,
    block_table_stride,
    num_spec,
    block_size,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    PAD: tl.constexpr,
):
    # One program per verifier row: the same table as build_replay_state_indices.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLUMNS)
    base = tl.load(base_ptr + row * base_stride)
    vals = tl.where(cols == NUM_COLUMNS - 1, base, PAD)
    batch_row = tl.load(spec_rows_ptr + row, mask=row < num_spec, other=-1).to(tl.int64)
    # Rows past num_spec, and rows marked -1 (a graph's fixed-size spec rows),
    # keep only their start column.
    if batch_row >= 0:
        req = tl.load(idx_mapping_ptr + batch_row).to(tl.int64)
        accepted = tl.maximum(tl.load(num_sampled_ptr + batch_row).to(tl.int64), 1)
        computed = tl.load(num_computed_ptr + req).to(tl.int64)
        running = computed - accepted + 1
        aligned = (computed // block_size) * block_size
        boundary_pos = aligned - running
        has_boundary = (aligned >= running) & (boundary_pos < accepted - 1)
        dst_col = tl.maximum(aligned // block_size - 1, 0)
        dst = tl.load(
            block_table_ptr + batch_row * block_table_stride + dst_col,
            mask=has_boundary,
            other=PAD,
        )
        # The last accepted position commits to the running block; a boundary
        # passed before it gets its prefix-cache block.
        vals = tl.where(cols == accepted - 1, base, vals)
        vals = tl.where(has_boundary & (cols == boundary_pos), dst.to(vals.dtype), vals)
    tl.store(table_ptr + row * table_stride + cols, vals, mask=cols < NUM_COLUMNS)


def build_replay_state_table(
    *,
    base_state_ids: torch.Tensor,
    num_reqs: int,
    num_columns: int,
    spec_rows: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """``build_replay_state_indices`` in one kernel launch, gathers included.

    ``spec_rows`` index the batch (``idx_mapping``, ``num_sampled`` and
    ``block_table`` rows); ``num_computed_tokens`` is per request state.
    """
    table = torch.empty(
        (num_reqs, num_columns), dtype=torch.int32, device=base_state_ids.device
    )
    if num_reqs == 0:
        return table
    _replay_state_table_kernel[(num_reqs,)](
        table,
        table.stride(0),
        base_state_ids,
        base_state_ids.stride(0),
        spec_rows,
        idx_mapping,
        num_sampled,
        num_computed_tokens,
        block_table,
        block_table.stride(0),
        spec_rows.numel(),
        block_size,
        NUM_COLUMNS=num_columns,
        BLOCK_COLUMNS=triton.next_power_of_2(num_columns),
        PAD=PAD_SLOT_ID,
    )
    return table


def _replay_launches(
    step: GDNSpecReplayStep,
    *,
    spec_rows: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    block_size: int,
) -> None:
    tables: dict[tuple[int, int], torch.Tensor] = {}
    for entry in step.layers:
        layer = entry.layer
        key = (entry.group_id, entry.base_state_ids.data_ptr())
        table = tables.get(key)
        if table is None:
            # One launch per cache group (the torch reference took about 25).
            table = build_replay_state_table(
                base_state_ids=entry.base_state_ids,
                num_reqs=step.num_reqs,
                num_columns=layer._replay_slots + 1,
                spec_rows=spec_rows,
                idx_mapping=idx_mapping,
                num_sampled=num_sampled,
                num_computed_tokens=num_computed_tokens,
                block_table=step.block_tables[entry.group_id],
                block_size=block_size,
            )
            tables[key] = table
        layer.replay_spec_states(table, entry.cu_seqlens, step.num_tokens)


def replay_gdn_spec_states(
    step: GDNSpecReplayStep,
    *,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    block_size: int,
) -> None:
    """Commit this step's accepted recurrent states (see the module docstring)."""
    _replay_launches(
        step,
        spec_rows=step.spec_rows,
        idx_mapping=idx_mapping,
        num_sampled=num_sampled,
        num_computed_tokens=num_computed_tokens,
        block_size=block_size,
    )


def replay_graph_key(
    step: GDNSpecReplayStep, num_computed_tokens: torch.Tensor, block_size: int
) -> tuple:
    """What a captured replay graph depends on besides the per-step inputs.

    The per-layer recorded inputs and the GDN metadata tensors (running blocks,
    cu_seqlens) are persistent buffers in a FULL step; a graph replays only when
    they are the same tensors it captured.
    """
    return (
        step.num_reqs,
        step.num_tokens,
        block_size,
        num_computed_tokens.data_ptr(),
        tuple(t.data_ptr() for t in step.block_tables),
        tuple(
            (
                id(e.layer),
                e.group_id,
                e.base_state_ids.data_ptr(),
                e.base_state_ids.stride(0),
                e.cu_seqlens.data_ptr(),
            )
            for e in step.layers
        ),
    )


@dataclass
class _ReplayGraph:
    graph: "torch.cuda.CUDAGraph"
    spec_rows: torch.Tensor  # [num_reqs], -1 marks padding rows
    idx_mapping: torch.Tensor  # [num_reqs]
    num_sampled: torch.Tensor  # [num_reqs]


class GDNSpecReplayGraphs:
    """The post-sampling replay as one CUDA graph per FULL verify shape.

    Eager, the replay issues one table kernel per cache group and one verifier
    launch per GDN layer. When the target's sampling has drained the GPU queue,
    those host-side launches become the step's critical path (about 7 ms per step
    for 36 layers). A FULL step's replay depends only on persistent tensors plus
    three small per-step inputs, so it is captured once per shape, the second
    time the shape is seen (the first, eager, run compiles the kernels), and then
    replayed after copying those inputs into the graph's buffers. Mixed steps
    (PIECEWISE or eager verifier) keep the eager replay.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.graphs: dict[tuple, _ReplayGraph] = {}
        self._seen: set[tuple] = set()
        self._pool = None

    def run(
        self,
        step: GDNSpecReplayStep,
        *,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        block_size: int,
    ) -> None:
        if not (self.enabled and step.full_graph):
            replay_gdn_spec_states(
                step,
                idx_mapping=idx_mapping,
                num_sampled=num_sampled,
                num_computed_tokens=num_computed_tokens,
                block_size=block_size,
            )
            return
        key = replay_graph_key(step, num_computed_tokens, block_size)
        entry = self.graphs.get(key)
        if entry is None:
            if key not in self._seen:
                self._seen.add(key)
                replay_gdn_spec_states(
                    step,
                    idx_mapping=idx_mapping,
                    num_sampled=num_sampled,
                    num_computed_tokens=num_computed_tokens,
                    block_size=block_size,
                )
                return
            entry = self._capture(step, num_computed_tokens, block_size)
            self.graphs[key] = entry
        fill_replay_graph_inputs(
            entry.spec_rows,
            entry.idx_mapping,
            entry.num_sampled,
            spec_rows=step.spec_rows,
            idx_mapping=idx_mapping,
            num_sampled=num_sampled,
        )
        entry.graph.replay()

    def _capture(
        self,
        step: GDNSpecReplayStep,
        num_computed_tokens: torch.Tensor,
        block_size: int,
    ) -> _ReplayGraph:
        device = num_computed_tokens.device
        n = step.num_reqs
        spec_rows = torch.full((n,), -1, dtype=torch.int64, device=device)
        idx_mapping = torch.zeros(n, dtype=torch.int32, device=device)
        num_sampled = torch.ones(n, dtype=torch.int32, device=device)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # Nothing runs during capture; the caller replays the graph for this step.
        with torch.cuda.graph(graph, self._pool, capture_error_mode="thread_local"):
            _replay_launches(
                step,
                spec_rows=spec_rows,
                idx_mapping=idx_mapping,
                num_sampled=num_sampled,
                num_computed_tokens=num_computed_tokens,
                block_size=block_size,
            )
        return _ReplayGraph(graph, spec_rows, idx_mapping, num_sampled)


def fill_replay_graph_inputs(
    spec_rows_buf: torch.Tensor,
    idx_mapping_buf: torch.Tensor,
    num_sampled_buf: torch.Tensor,
    *,
    spec_rows: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
) -> None:
    """Copy this step's inputs into a replay graph's fixed-size buffers.

    ``spec_rows`` lists the real speculative rows; the rest of the buffer is -1,
    which the table kernel treats as a padding row. ``idx_mapping`` and
    ``num_sampled`` are indexed by batch row and may be shorter than the graph's
    padded batch.
    """
    num_spec = spec_rows.numel()
    spec_rows_buf.fill_(-1)
    spec_rows_buf[:num_spec].copy_(spec_rows)
    idx_mapping_buf[: idx_mapping.numel()].copy_(idx_mapping)
    num_sampled_buf[: num_sampled.numel()].copy_(num_sampled)
