# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded compact sampling for uniform-width SM70 MTP verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _TARGET_PROBE_K,
    _TARGET_TOP_K,
    DFlash2LogitsFallback,
    _compact_target_reference_rows_gpu,
    _sample_reference_requests,
    _supports_sparse_sampling_contract,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    greedy_draft_sparse_topk_rejection_sample,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)


def _is_sm70(device: torch.device) -> bool:
    return device.type == "cuda" and current_platform.is_device_capability(
        70, device.index
    )


def try_mtp_compact_target_rejection(
    model: Any,
    spec_config: SpeculativeConfig,
    draft_logits: torch.Tensor | None,
    rejection_sampler: RejectionSampler,
    sample_hidden_states: torch.Tensor,
    input_batch: InputBatch,
    grammar_output: GrammarOutput | None,
    *,
    has_lora: bool,
) -> SamplerOutput | DFlash2LogitsFallback | None:
    """Attempt compact sampling; None leaves the runner's normal route intact."""

    def dense(reason: str) -> SamplerOutput | None:
        logger.debug(
            "SM70 MTP compact target route=dense reason=%s requests=%d rows=%d",
            reason,
            input_batch.num_reqs,
            sample_hidden_states.shape[0],
        )
        return None

    if spec_config.draft_sample_method != "greedy" or draft_logits is not None:
        return dense("not_greedy_draft")
    if grammar_output is not None or input_batch.has_structured_output_reqs:
        return dense("grammar")
    if has_lora:
        return dense("lora")
    if not _is_sm70(sample_hidden_states.device):
        return dense("not_sm70")
    probe = getattr(model, "get_mtp_target_topk_with_fallback", None)
    sampler = rejection_sampler.sampler
    states = sampler.sampling_states
    if probe is None:
        return dense("unsupported_target")
    if input_batch.is_dummy_batch:
        return dense("dummy_batch")
    max_steps = rejection_sampler.num_speculative_steps
    if not 1 <= input_batch.num_reqs <= 8 or not 3 <= max_steps <= 5:
        return dense("unsupported_width")
    row_counts = np.diff(input_batch.cu_num_logits_np)
    steps = int(row_counts[0]) - 1
    if not 3 <= steps <= max_steps or not np.all(row_counts == steps + 1):
        return dense("unsupported_or_mixed_width")
    idx = input_batch.idx_mapping_np
    if np.any(states.temperature.np[idx] != np.float32(1.0)):
        return dense("temperature")
    if np.any(states.top_p.np[idx] != np.float32(0.95)):
        return dense("top_p")
    if not _supports_sparse_sampling_contract(rejection_sampler, input_batch):
        return dense("sampling_transform_or_nonstandard_rejection")

    # Probe 64 bypasses the approximate DFlash2 QPN8 selector. The callback
    # gathers the same already-computed local projection on every TP rank.
    ids, logits, fallback = probe(sample_hidden_states, _TARGET_PROBE_K)
    if fallback is None:
        return dense("projection_has_no_reference")
    expanded = input_batch.expanded_idx_mapping
    logits, ids = sort_topk_with_vocab_ties(
        logits.float(), ids, vocab_size=states.vocab_size, descending=True
    )
    reference_rows = _compact_target_reference_rows_gpu(
        logits, states.top_p.gpu, expanded, strict_top_k=True
    )
    reference_reqs = np.flatnonzero(
        np.logical_or.reduceat(reference_rows, input_batch.cu_num_logits_np[:-1])
    )
    reference_logits = None
    if reference_reqs.size:
        reference_logits = fallback()
        if reference_reqs.size == idx.size or reference_logits is None:
            logger.debug(
                "SM70 MTP compact target route=dense reason=%s requests=%d "
                "rows=%d reference_requests=%d",
                "reference_all"
                if reference_reqs.size == idx.size
                else "reference_non_gather_rank",
                idx.size,
                logits.shape[0],
                reference_reqs.size,
            )
            return DFlash2LogitsFallback(reference_logits)

    indices = input_batch.logits_indices
    sampled, num_sampled = greedy_draft_sparse_topk_rejection_sample(
        ids[:, :_TARGET_TOP_K],
        logits[:, :_TARGET_TOP_K],
        input_batch.input_ids[indices],
        input_batch.cu_num_logits,
        input_batch.positions[indices],
        input_batch.idx_mapping,
        states.top_p.gpu,
        states.seeds.gpu,
        # Keep the configured output stride shared with the dense fallback.
        # Verification and bonus are bounded by each request's actual row end.
        max_steps,
        sampler.use_fp64_gumbel,
    )
    if reference_logits is not None:
        reqs, dense_ids, dense_counts = _sample_reference_requests(
            reference_logits, reference_reqs, input_batch, rejection_sampler, None
        )
        sampled.index_copy_(0, reqs, dense_ids)
        num_sampled.index_copy_(0, reqs, dense_counts)
    compact_reqs = idx.size - reference_reqs.size
    logger.info_once("Using SM70 MTP compact target sampling (guarded K%d).", steps)
    logger.debug(
        "SM70 MTP compact target route=%s compact_requests=%d compact_rows=%d "
        "reference_requests=%d",
        "mixed" if reference_reqs.size else "compact",
        compact_reqs,
        compact_reqs * (steps + 1),
        reference_reqs.size,
    )
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
    )
