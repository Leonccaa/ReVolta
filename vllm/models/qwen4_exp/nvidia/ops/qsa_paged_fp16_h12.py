# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

H12_MAX_ROWS = 48
H12_MAX_WIDTH = 4096


def load_qsa_paged_fp16_h12() -> None:
    importlib.import_module("vllm._sm70_qsa_indexer_C")
    _ = torch.ops.vllm_sm70_qsa_indexer.paged_fp16_h12
    logger.info_once("SM70 QSA paged FP16 H12 enabled for eligible DCP2 decode.")


def qsa_paged_fp16_h12_workspace(rows: int, width: int) -> list[torch.Tensor]:
    splits = cdiv(width, 64)
    return current_workspace_manager().get_simultaneous(
        ((rows, 12, splits, 256), torch.float32),
        ((rows, 12, splits, 2), torch.float32),
    )


def qsa_paged_fp16_h12(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    partial, state = qsa_paged_fp16_h12_workspace(q.shape[0], indices.shape[1])
    torch.ops.vllm_sm70_qsa_indexer.paged_fp16_h12(
        q, k, v, indices, table, requests, out, lse, partial, state
    )
