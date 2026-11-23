# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 native FP16 bounded top-k dispatch helper."""

from __future__ import annotations

import torch

from vllm import _custom_ops as ops


def sm70_compact_half_topk_revision() -> int:
    """Return the compiled SM70 compact half top-k revision, or 0 if absent."""
    if not hasattr(torch.ops._C, "sm70_compact_half_topk_out"):
        return 0
    if not hasattr(torch.ops._C, "sm70_compact_half_topk_wide_revision"):
        return 1
    return int(ops.sm70_compact_half_topk_wide_revision())


def sm70_compact_half_topk_partial_spec(
    rows: int,
    width: int,
    k: int,
) -> tuple[tuple[int, int], torch.dtype]:
    """Return the workspace shape and dtype required by sm70_compact_half_topk_out."""
    if width <= 65535:
        tiles = (width + 1023) // 1024
        return (rows, tiles * k), torch.uint32
    w1 = ((width + 1023) // 1024) * k
    w2 = ((w1 + 1023) // 1024) * k
    return (rows, w1 + w2), torch.int64


def sm70_compact_half_topk_ids_partial_spec(
    rows: int,
    width: int,
    k: int,
) -> tuple[tuple[int, int], torch.dtype]:
    """Return the workspace shape and dtype for sm70_compact_half_topk_ids_out."""
    if width <= 65535:
        tiles = (width + 1023) // 1024
        return (rows, tiles * k), torch.uint32
    if width <= 131072 and k == 32:
        w1 = ((width + 2047) // 2048) * k
        return (rows, w1), torch.int64
    if width <= 262144:
        w1 = ((width + 2047) // 2048) * k
        w2 = ((w1 + 2047) // 2048) * k
        return (rows, w1 + w2), torch.int64
    w1 = ((width + 1023) // 1024) * k
    w2 = ((w1 + 1023) // 1024) * k
    return (rows, w1 + w2), torch.int64


def maybe_sm70_compact_half_topk(
    logits: torch.Tensor,
    k: int,
    *,
    partial: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run lossless SM70 FP16 top-k returning (fp32 values, int64 ids) if admitted."""
    if (
        not logits.is_cuda
        or logits.dtype != torch.float16
        or logits.ndim != 2
        or logits.stride(1) != 1
    ):
        return None
    revision = sm70_compact_half_topk_revision()
    if revision <= 0:
        return None
    rows, width = int(logits.shape[0]), int(logits.shape[1])
    max_rows = 256 if revision >= 2 else 64
    if not 1 <= rows <= max_rows:
        return None
    if not 1 <= k <= 64 or width < k or width > 524288:
        return None
    if k != 64 and revision < 2:
        return None
    if torch.cuda.get_device_capability(logits.device) != (7, 0):
        return None

    partial_shape, partial_dtype = sm70_compact_half_topk_partial_spec(rows, width, k)
    if partial is None:
        partial = torch.empty(partial_shape, dtype=partial_dtype, device=logits.device)
    if values is None:
        values = torch.empty((rows, k), dtype=torch.float32, device=logits.device)
    if ids is None:
        ids = torch.empty((rows, k), dtype=torch.int64, device=logits.device)
    ops.sm70_compact_half_topk_out(logits, partial, values, ids, k)
    return values, ids
