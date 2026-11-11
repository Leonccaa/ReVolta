# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-storage DCP2 candidate/query exchange for uniform decode only."""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _pack_candidates_query(
    Blocks,
    Scores,
    Query,
    Packed,
    BS0,
    BS1,
    SS0,
    SS1,
    QS0,
    QS1,
    K: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col = tl.load(Blocks + row * BS0 + x * BS1, x < K, other=-1)
    valid = (x < K) & (col >= 0)
    bits = tl.load(Scores + row * SS0 + tl.maximum(col, 0) * SS1, valid, other=-8388608)
    tl.store(Packed + row * WIDTH + x, bits, x < K)
    tl.store(
        Packed + row * WIDTH + K + x, tl.where(valid, col * 2 + RANK, 2147483647), x < K
    )
    # Query is an int16 view: even signaling NaNs never enter an FP register.
    qword = x
    elem = qword * 2
    head = elem // DIM
    offset = row * QS0 + head * QS1 + elem % DIM
    lo = (
        tl.load(Query + offset, qword < HEADS * DIM // 2, other=0)
        .to(tl.uint16)
        .to(tl.uint32)
    )
    hi = (
        tl.load(Query + offset + 1, qword < HEADS * DIM // 2, other=0)
        .to(tl.uint16)
        .to(tl.uint32)
    )
    tl.store(
        Packed + row * WIDTH + 2 * K + qword, lo | (hi << 16), qword < HEADS * DIM // 2
    )


@triton.jit
def _unpack_query(
    Gathered,
    Query,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.program_id(1)
    word = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    valid = word < HEADS * DIM // 2
    bits = tl.load(
        Gathered + (rank * ROWS + row) * WIDTH + 2 * K + word, valid, other=0
    ).to(tl.uint32)
    dest = Query + (row * 2 * HEADS + rank * HEADS) * DIM + 2 * word
    tl.store(dest, (bits & 65535).to(tl.uint16), valid)
    tl.store(dest + 1, (bits >> 16).to(tl.uint16), valid)


class QSACoalesceWorkspace:
    """One workspace per model, shared by its serial QSA layers and graphs.

    Capacity R, local H/D, top-k K: 3*R*(2*K+H*D/2)*4 + R*2*H*D*2
    bytes. Rank-major receive storage and row-major Q remain live until the
    attention call completes. No tensor allocation occurs in exchange().
    """

    def __init__(
        self,
        rows: int,
        k: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ):
        self.rows, self.k, self.heads, self.dim = rows, k, heads, dim
        self.width = 2 * k + heads * dim // 2
        self.send = torch.empty(rows * self.width, dtype=torch.int32, device=device)
        self.receive = torch.empty(
            2 * rows * self.width, dtype=torch.int32, device=device
        )
        self.query = torch.empty((rows, 2 * heads, dim), dtype=dtype, device=device)

    def supports(self, query: torch.Tensor, rows: int) -> bool:
        return (
            0 < rows <= self.rows
            and query.shape == (rows, self.heads, self.dim)
            and query.dtype == self.query.dtype
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.device == self.send.device
            and self.dim % 2 == 0
            and query.stride(-1) == 1
        )


class QSACoalescedExchange:
    """A forward-local transaction; absence of a result means full fallback.

    Eligibility uses only replicated scheduler descriptors/configuration and
    tensor geometry, never rank-local lengths, visibility, scores or IDs.
    All workers must start with identical environment/configuration, as for
    the original sharded-index collective. No per-rank exception fallback.
    """

    def __init__(self, workspace: QSACoalesceWorkspace, query: torch.Tensor):
        self.workspace = workspace
        self.query = query
        self.gathered_query: torch.Tensor | None = None

    def exchange(
        self,
        blocks: torch.Tensor,
        logits: torch.Tensor,
        rank: int,
        gather_into: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        ws = self.workspace
        rows = self.query.shape[0]
        assert ws.supports(self.query, rows)
        assert blocks.shape == (rows, ws.k) and blocks.dtype == torch.int32
        assert logits.dtype == torch.float32 and logits.shape[0] == rows
        assert rank in (0, 1)
        send = ws.send[: rows * ws.width].view(rows, ws.width)
        receive = ws.receive[: 2 * rows * ws.width].view(2, rows, ws.width)
        _pack_candidates_query[
            (rows, triton.cdiv(max(ws.k, ws.heads * ws.dim // 2), 128))
        ](
            blocks,
            logits.view(torch.int32),
            self.query.view(torch.int16),
            send,
            *blocks.stride(),
            *logits.stride(),
            *self.query.stride()[:2],
            ws.k,
            ws.heads,
            ws.dim,
            rank,
            ws.width,
            128,
        )
        if gather_into is None:
            import torch.distributed as dist

            from vllm.distributed import get_dcp_group

            # Same concat-style collective/device group as GroupCoordinator's
            # ordinary all_gather; explicit output retains fixed scratch.
            dist.all_gather_into_tensor(
                receive.view(2 * rows, ws.width),
                send,
                group=get_dcp_group().device_group,
            )
        else:
            gather_into(receive, send)
        query = ws.query[:rows]
        _unpack_query[(rows, 2, triton.cdiv(ws.heads * ws.dim // 2, 128))](
            receive,
            query.view(torch.int16),
            rows,
            ws.k,
            ws.heads,
            ws.dim,
            ws.width,
            128,
        )
        self.gathered_query = query
        # Preserve rank/row strides; the existing merge accepts strided input.
        return receive[:, :, : 2 * ws.k]
