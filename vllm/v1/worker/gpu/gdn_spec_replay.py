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
    if bool(has_boundary.any()):
        dst_cols = (aligned // block_size - 1).clamp(min=0)
        dst_blocks = block_table[spec_rows.to(torch.long), dst_cols].to(torch.int32)
        table[rows[has_boundary], boundary_pos[has_boundary]] = dst_blocks[has_boundary]
    return table


def replay_gdn_spec_states(
    step: GDNSpecReplayStep,
    *,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    block_size: int,
) -> None:
    """Commit this step's accepted recurrent states (see the module docstring)."""
    spec_rows = step.spec_rows
    req_indices = idx_mapping[spec_rows].to(torch.long)
    num_accepted = num_sampled[spec_rows]
    computed = num_computed_tokens[req_indices]
    tables: dict[tuple[int, int], torch.Tensor] = {}
    for entry in step.layers:
        layer = entry.layer
        key = (entry.group_id, entry.base_state_ids.data_ptr())
        table = tables.get(key)
        if table is None:
            table = build_replay_state_indices(
                base_state_ids=entry.base_state_ids,
                num_reqs=step.num_reqs,
                num_columns=layer._replay_slots + 1,
                spec_rows=spec_rows,
                num_accepted=num_accepted,
                num_computed_tokens=computed,
                block_table=step.block_tables[entry.group_id],
                block_size=block_size,
            )
            tables[key] = table
        layer.replay_spec_states(table, entry.cu_seqlens, step.num_tokens)
