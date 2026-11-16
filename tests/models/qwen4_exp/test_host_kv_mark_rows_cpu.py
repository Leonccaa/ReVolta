# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""The fused decode-mark kernel must not recompile per row count.

``ROWS`` (and the ``ROW_OFFSET`` it relays to the tail phase) used to be
``tl.constexpr``: every padded decode batch size selected a new specialization,
so the first step at that size paid a full compile inside the inference path.
They are runtime scalars now. The gates run the fused kernel over row counts
from the three specialization classes (one, a multiple of 16, neither) and
compare against the unfused two-kernel baseline it replaces.
"""

import os

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv import history_kernels as hk
from vllm.triton_utils import triton

pytestmark = pytest.mark.skip_global_cleanup

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreter = pytest.mark.skipif(
    not INTERPRETED, reason="Triton kernels need CUDA or TRITON_INTERPRET=1"
)
# Rows that used to select different specialization classes.
COUNTS = (1, 2, 3, 15, 16, 17, 33)
KBS = 32  # kernel-block tokens
REQUESTS, BLOCKS, M, WIDTH = 3, 4, 96, 8
GEOMETRY = (1, 0, 1)  # WORLD, RANK, interleave


def _inputs(rows, width):
    generator = torch.Generator().manual_seed(23)
    table = torch.arange(REQUESTS * BLOCKS, dtype=torch.int32).reshape(REQUESTS, BLOCKS)
    seq_lens = torch.full((REQUESTS,), BLOCKS * KBS, dtype=torch.int32)
    ready = torch.ones(REQUESTS * BLOCKS * KBS, dtype=torch.int8)
    selection = torch.randint(
        0, BLOCKS * KBS, (rows, width), generator=generator, dtype=torch.int32
    )
    token_to_req = torch.arange(rows, dtype=torch.int32) % REQUESTS
    return table, seq_lens, ready, selection, token_to_req


def _fresh_buffers(wanted_groups):
    return (
        torch.zeros(wanted_groups, dtype=torch.int32),
        torch.full((M,), -1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(M, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )


def _marks(wanted, touched, count, groups):
    claimed = sorted(g for g in range(groups) if int(wanted[g]) & 1)
    listed = sorted(g for g in touched[: int(count)].tolist() if g >= 0)
    return claimed, listed


@interpreter
@pytest.mark.parametrize("rows", COUNTS)
@pytest.mark.parametrize("width", (4, 8))
def test_fused_marks_match_unfused_baseline(rows, width):
    table, seq_lens, ready, selection, token_to_req = _inputs(rows, width)
    groups = REQUESTS * BLOCKS * (KBS // 4)
    grid = (rows, triton.cdiv(width, 128))

    fused_wanted, fused_touched, fused_count, _, _ = _fresh_buffers(groups)
    hk.mark_decode[(rows + REQUESTS, grid[1])](
        selection,
        token_to_req,
        table,
        seq_lens,
        ready,
        fused_wanted,
        fused_touched,
        fused_count,
        selection.stride(0),
        table.stride(0),
        width,
        KBS,
        M,
        128,
        *GEOMETRY,
        0,
        rows,
        num_warps=1,
    )

    base_wanted, base_touched, base_count, _, _ = _fresh_buffers(groups)
    hk.mark_tail[(REQUESTS,)](
        table,
        seq_lens,
        ready,
        base_wanted,
        base_touched,
        base_count,
        table.stride(0),
        KBS,
        M,
        *GEOMETRY,
        num_warps=1,
    )
    hk.mark_selection[grid](
        selection,
        token_to_req,
        table,
        seq_lens,
        ready,
        base_wanted,
        base_touched,
        base_count,
        selection.stride(0),
        table.stride(0),
        width,
        KBS,
        M,
        128,
        *GEOMETRY,
        0,
        num_warps=1,
    )

    want = _marks(base_wanted, base_touched, base_count, groups)
    got = _marks(fused_wanted, fused_touched, fused_count, groups)
    assert got == want
    assert got[0]  # the tail alone keeps nine groups per request resident
    assert int(fused_count) == len(got[1]) == int(base_count)


@interpreter
@pytest.mark.parametrize("rows", COUNTS)
def test_ordered_fused_marks_keep_tail_first(rows):
    """ORDERED splits the two phases' lists; the tail relay still uses ROWS."""
    table, seq_lens, ready, selection, token_to_req = _inputs(rows, WIDTH)
    groups = REQUESTS * BLOCKS * (KBS // 4)
    grid = (rows, triton.cdiv(WIDTH, 128))

    wanted, touched, count, sel, sel_count = _fresh_buffers(groups)
    hk.mark_decode[(rows + REQUESTS, grid[1])](
        selection,
        token_to_req,
        table,
        seq_lens,
        ready,
        wanted,
        touched,
        count,
        selection.stride(0),
        table.stride(0),
        WIDTH,
        KBS,
        M,
        128,
        *GEOMETRY,
        0,
        rows,
        Sel=sel,
        SelCount=sel_count,
        ORDERED=True,
        num_warps=1,
    )
    tail = [g for g in touched[: int(count)].tolist() if g >= 0]
    selection_list = [g for g in sel[: int(sel_count)].tolist() if g >= 0]
    assert tail and selection_list
    # Every tail entry is a group the selection phase also (or only) wants.
    assert all(int(wanted[g]) & 1 for g in tail)
    assert all(int(wanted[g]) & 2 for g in selection_list)
    # The same rows through the unfused path claim exactly the same groups.
    base_wanted, base_touched, base_count, _, _ = _fresh_buffers(groups)
    hk.mark_tail[(REQUESTS,)](
        table,
        seq_lens,
        ready,
        base_wanted,
        base_touched,
        base_count,
        table.stride(0),
        KBS,
        M,
        *GEOMETRY,
        num_warps=1,
    )
    assert _marks(base_wanted, base_touched, base_count, groups)[0] == sorted(
        set(tail) & {g for g in range(groups) if int(base_wanted[g]) & 1}
    )
    assert sorted(tail) == _marks(base_wanted, base_touched, base_count, groups)[0]
