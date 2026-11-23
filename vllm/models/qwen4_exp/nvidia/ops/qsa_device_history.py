# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 QSA readers for replicated draft and sharded target GPU KV."""

import importlib

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

MAX_ROWS = 20
H6_MAX_ROWS = 64
MAX_SELECTION_WIDTH = 4096


def load_qsa_device_history() -> None:
    importlib.import_module("vllm._sm70_qsa_device_C")
    _ = torch.ops.vllm_sm70_qsa_device.run_paged_dcp
    logger.info_once("SM70 QSA device history enabled for eligible DCP2 decode.")


def load_qsa_paged_fp16_h6() -> None:
    importlib.import_module("vllm._sm70_qsa_device_C")
    _ = torch.ops.vllm_sm70_qsa_device.run


def qsa_device_history_workspace(rows: int, width: int) -> list[torch.Tensor]:
    # The two H6 passes run on one stream and reuse the same split workspace.
    splits = rows * 6 * cdiv(width, 64)
    return current_workspace_manager().get_simultaneous(
        ((splits, 256), torch.float32), ((splits, 2), torch.float32)
    )


def try_qsa_paged_fp16_h6(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    scales: torch.Tensor,
    indices: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    gate: torch.Tensor | None,
) -> bool:
    if not (
        0 < q.shape[0] <= H6_MAX_ROWS
        and 0 < indices.shape[1] <= MAX_SELECTION_WIDTH
        and kv_cache.is_contiguous()
    ):
        return False
    partial, state = qsa_device_history_workspace(q.shape[0], indices.shape[1])
    torch.ops.vllm_sm70_qsa_device.run(
        q,
        kv_cache,
        scales,
        indices,
        table,
        requests,
        positions,
        lengths,
        out,
        gate.view_as(q) if gate is not None else None,
        partial,
        state,
    )
    logger.info_once(
        "Using SM70 native FP16 H6 MTP QSA reader (rows=%d, page=%d, width=%d).",
        q.shape[0],
        kv_cache.size(2),
        indices.shape[1],
    )
    return True


def qsa_device_history(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    k_scale: float,
    v_scale: float,
) -> None:
    partial, state = qsa_device_history_workspace(q.shape[0], indices.shape[1])
    torch.ops.vllm_sm70_qsa_device.run_paged_dcp(
        q, k, v, indices, table, requests, k_scale, v_scale, out, lse, partial, state
    )
