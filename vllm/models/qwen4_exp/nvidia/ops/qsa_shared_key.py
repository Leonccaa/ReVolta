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
def _load_operators():
    try:
        importlib.import_module("vllm._sm70_qsa_indexer_C")
    except ImportError:
        return False, None

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

    torch.library.register_fake("vllm_sm70_qsa_indexer::run_batch")(fake)
    batched = getattr(torch.ops.vllm_sm70_qsa_indexer, "run_batched_decode", None)
    if batched is not None:
        torch.library.register_fake("vllm_sm70_qsa_indexer::run_batched_decode")(fake)
    return True, batched


def load_operator():
    return _load_operators()[0]


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
    # One request's rows form a contiguous span in the QSA metadata.
    torch.ops.vllm_sm70_qsa_indexer.run_batch(
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
    """Use MRv2 request offsets after request-segments admission."""
    rows, nreq = q.shape[0], table.shape[0]
    if (
        rows > min(64, 8 * nreq)
        or starts.shape != (nreq + 1,)
        or starts.dtype != torch.int32
        or not all(x.is_contiguous() for x in (starts, positions, lengths))
        or q.stride(-1) != 1
        or cache.stride(-1) != 1
        or any(i % 8 for x in (q, cache) for i in (*x.stride()[:2], x.storage_offset()))
    ):
        return None
    _, batched = _load_operators()
    if batched is None:
        return None

    scores = torch.empty(
        (rows, table.shape[1] * cache.shape[1]), device=q.device, dtype=torch.float32
    )
    visible = torch.empty(rows, device=q.device, dtype=torch.int32)
    batched(
        scores, visible, q, cache, table, starts, positions, lengths, 128**0.5, compress
    )
    logger.info_once("Using SM70 QSA batched shared-key decode scorer.")
    return scores, visible
