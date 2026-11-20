# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""1Cat's SM70 H6 device-history reader on ReVolta's DCP2 GPU KV pages."""

import importlib

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

MAX_ROWS = 20
MAX_SELECTION_WIDTH = 4096


def load_qsa_device_history() -> None:
    importlib.import_module("vllm._sm70_qsa_device_C")
    _ = torch.ops.vllm_sm70_qsa_device.run_paged_dcp
    logger.info_once("SM70 QSA device history enabled for eligible DCP2 decode.")


def qsa_device_history_workspace(rows: int, width: int) -> list[torch.Tensor]:
    # The two H6 passes run on one stream and reuse the same split workspace.
    splits = rows * 6 * cdiv(width, 64)
    return current_workspace_manager().get_simultaneous(
        ((splits, 256), torch.float32), ((splits, 2), torch.float32)
    )


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
