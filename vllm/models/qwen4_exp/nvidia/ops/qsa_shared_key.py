# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability-gated shared-key scoring for small QSA verification batches."""

import importlib
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def load_operator():
    try:
        importlib.import_module("vllm._sm70_qsa_indexer_C")
    except ImportError:
        return False

    @torch.library.register_fake("vllm_sm70_qsa_indexer::run")
    def fake(
        scores,
        visible,
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        divisor,
        compress,
    ):
        return None

    if hasattr(torch.ops.vllm_sm70_qsa_indexer, "run_batched_decode"):
        torch.library.register_fake("vllm_sm70_qsa_indexer::run_batched_decode")(fake)
    return True


def shared_key_reason(q, cache, table, requests, positions, lengths):
    if not current_platform.is_device_capability(70):
        return "requires_SM70"
    if q.dtype != torch.float16 or cache.dtype != torch.float16:
        return "requires_FP16_query_and_indexer_cache"
    if not (2 <= q.shape[0] <= 8 and q.shape[1:] == (4, 128)):
        return "requires_M2_8_H4_D128"
    if table.shape[0] != 1:
        return "multiple_requests"
    if not all(x.dtype == torch.int32 for x in (table, requests, lengths)):
        return "requires_int32_request_metadata"
    if positions.dtype not in (torch.int32, torch.int64):
        return "requires_integer_positions"
    if not all(x.is_contiguous() for x in (requests, positions, lengths)):
        return "noncontiguous_row_metadata"
    if not load_operator():
        return "native_extension_unavailable"
    return None


def maybe_shared_key_scores(
    q, cache, table, requests, positions, lengths, compress, columns, divisor
):
    reason = shared_key_reason(q, cache, table, requests, positions, lengths)
    if reason is not None:
        logger.info_once("SM70 QSA shared-key scorer fallback: %s.", reason)
        return None
    scores = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    torch.ops.vllm_sm70_qsa_indexer.run(
        scores,
        visible,
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        float(divisor),
        compress,
    )
    logger.info_once("Using SM70 QSA shared-key indexer for M=%d.", q.shape[0])
    return scores, visible


def maybe_batched_decode_scores(q, cache, table, starts, positions, lengths, compress):
    """Use the MRv2 request-major starts without scanning row ownership."""
    rows, nreq = q.shape[0], table.shape[0]
    if (
        not (2 <= nreq <= 32 and rows <= 64 and 2 <= rows // nreq <= 8)
        or q.shape[1:] != (4, 128)
        or cache.shape[2:] != (1, 128)
        or starts.shape != (nreq + 1,)
        or q.dtype != torch.float16
        or cache.dtype != torch.float16
        or any(x.dtype != torch.int32 for x in (table, starts, lengths))
        or positions.dtype not in (torch.int32, torch.int64)
        or not all(x.is_contiguous() for x in (starts, positions, lengths))
        or q.stride(-1) != 1
        or cache.stride(-1) != 1
        or any(i % 8 for x in (q, cache) for i in (*x.stride()[:2], x.storage_offset()))
    ):
        return None
    if not load_operator() or not hasattr(
        torch.ops.vllm_sm70_qsa_indexer, "run_batched_decode"
    ):
        return None

    scores = torch.empty(
        (rows, table.shape[1] * cache.shape[1]), device=q.device, dtype=torch.float32
    )
    visible = torch.empty(rows, device=q.device, dtype=torch.int32)
    torch.ops.vllm_sm70_qsa_indexer.run_batched_decode(
        scores, visible, q, cache, table, starts, positions, lengths, 128**0.5, compress
    )
    logger.info_once("Using SM70 QSA batched shared-key decode scorer.")
    return scores, visible
