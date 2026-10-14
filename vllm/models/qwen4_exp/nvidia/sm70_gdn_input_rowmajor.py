# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 GDN input projection for 2..48 rows on the row-major weights (SM70).

Opt-in ``VLLM_SM70_QWEN38_GDN_NO_PACK=1``: the loader does not build the
packed N32/K16 copy of ``in_proj_qkvz``/``in_proj_ba`` (725.6 MiB per TP4
rank), and :func:`gdn_input_rowmajor` serves 2..48 decode/verify rows with the
SM70 FP16 skinny dual GEMM ``fp16_skinny_gemm2_sm70_out``
(``csrc/sm70_turbomind/ops/fp16_skinny_gemm_sm70.cu``) on the original
checkpoint weights. One launch computes both projections and writes
contiguous qkv/z/b/a (no split copy).

The kernel reads each weight once per call for all rows, accumulates in FP32
and rounds to FP16 once; every reduction is fixed order (no atomics). The K
partition depends only on ``(warps_k, splits)``, and the plan below is the same
for every M, so each row's result is independent of the batch width.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

MIN_ROWS = 2
MAX_ROWS = 48  # kMaxRows of the native op
QKVZ_SHAPE = (4096, 2560)
BA_SHAPE = (24, 2560)
# Output column splits: qkvz -> qkv | z, ba -> b | a.
QKV_COLS = 2560
B_COLS = 12
_OP2 = "fp16_skinny_gemm2_sm70_out"


class SkinnyConfig(NamedTuple):
    ng: int  # 8-column groups per warp
    warps_n: int  # warps splitting the CTA's columns
    warps_k: int  # warps splitting K inside a CTA
    splits: int  # CTAs splitting K (> 1 adds a fixed-order reduce kernel)


# Plan for the fused N = 4096 + 24, K = 2560 launch; one plan for every M.
PLAN = SkinnyConfig(1, 1, 2, 1)


@lru_cache(maxsize=1)
def native_available() -> bool:
    return hasattr(torch.ops._C, _OP2)


def _is_sm70() -> bool:
    return current_platform.is_device_capability(70)


def _device_ok(t: torch.Tensor) -> bool:
    return t.is_cuda


def _launch2(
    outs: tuple[torch.Tensor, ...],
    x2d: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    workspace: torch.Tensor,
    config: SkinnyConfig,
) -> None:
    torch.ops._C.fp16_skinny_gemm2_sm70_out(*outs, x2d, w0, w1, workspace, *config)


def rows_admitted(m: int) -> bool:
    return MIN_ROWS <= m <= MAX_ROWS


def _weight_ok(weight: torch.Tensor, x: torch.Tensor, shape: tuple) -> bool:
    return bool(
        tuple(weight.shape) == shape
        and weight.dtype == torch.float16
        and _device_ok(weight)
        and weight.device == x.device
        and weight.shape[1] == x.shape[-1]
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


def admitted(
    x: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor
) -> bool:
    return bool(
        x.ndim == 2
        and rows_admitted(x.shape[0])
        and x.dtype == torch.float16
        and _device_ok(x)
        and _weight_ok(qkvz_weight, x, QKVZ_SHAPE)
        and _weight_ok(ba_weight, x, BA_SHAPE)
        and not envs.VLLM_BATCH_INVARIANT
        # FP32 accumulation only: an explicit FP16-accumulation request keeps
        # the cuBLAS route that honors it.
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
        and _is_sm70()
        and native_available()
    )


def _input_2d(x: torch.Tensor) -> torch.Tensor:
    """Unit column stride and 16-byte aligned rows, else an aligned copy."""
    aligned = (
        x.stride(-1) == 1
        and (x.shape[0] == 1 or x.stride(0) % 8 == 0)
        and x.data_ptr() % 16 == 0
    )
    return x if aligned else x.clone(memory_format=torch.contiguous_format)


def _workspace(x2d: torch.Tensor, config: SkinnyConfig, n: int) -> torch.Tensor:
    numel = config.splits * x2d.shape[0] * n if config.splits > 1 else 0
    return torch.empty((numel,), dtype=torch.float32, device=x2d.device)


def gdn_input_rowmajor(
    x: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Contiguous ``(qkv, z, b, a)`` of ``x @ qkvz.T`` / ``x @ ba.T``.

    Returns ``None`` when not admitted; the caller keeps its fallback.
    """
    if not admitted(x, qkvz_weight, ba_weight):
        return None
    m = x.shape[0]
    n0, n1 = QKVZ_SHAPE[0], BA_SHAPE[0]
    x2d = _input_2d(x)
    outs = tuple(
        x2d.new_empty((m, cols))
        for cols in (QKV_COLS, n0 - QKV_COLS, B_COLS, n1 - B_COLS)
    )
    _launch2(outs, x2d, qkvz_weight, ba_weight, _workspace(x2d, PLAN, n0 + n1), PLAN)
    logger.info_once(
        "SM70 Qwen3.8 GDN input: row-major skinny dual GEMM for 2..48 rows (plan=%s).",
        tuple(PLAN),
    )
    return outs  # type: ignore[return-value]


__all__ = [
    "MAX_ROWS",
    "MIN_ROWS",
    "PLAN",
    "SkinnyConfig",
    "admitted",
    "gdn_input_rowmajor",
    "native_available",
    "rows_admitted",
]
