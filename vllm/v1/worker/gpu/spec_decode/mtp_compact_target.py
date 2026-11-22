# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact sampling for uniform-width SM70 MTP verification."""

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
    _pack_reference_requests,
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
    device_only: bool = False,
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
    if not 1 <= input_batch.num_reqs <= 8 or not 3 <= max_steps <= 6:
        return dense("unsupported_width")
    row_counts = np.diff(input_batch.cu_num_logits_np)
    steps = int(row_counts[0]) - 1
    if not np.all(row_counts == steps + 1):
        return dense("unsupported_or_mixed_width")
    # Adaptive single-request K2 uses the original configured sampler stride.
    # Keep the previously covered K3–K5 and cap ranges for other batches.
    single_k2 = steps == 2 and input_batch.num_reqs == 1
    if not single_k2 and not 3 <= steps <= max_steps <= 5:
        return dense("unsupported_or_mixed_width")
    idx = input_batch.idx_mapping_np
    if np.any(states.temperature.np[idx] != np.float32(1.0)):
        return dense("temperature")
    if np.any(states.top_p.np[idx] != np.float32(0.95)):
        return dense("top_p")
    if not _supports_sparse_sampling_contract(rejection_sampler, input_batch):
        return dense("sampling_transform_or_nonstandard_rejection")

    reasoning_eos = sampler.reasoning_eos
    guarded = reasoning_eos.active_requests(idx) if reasoning_eos is not None else []
    if len(guarded) == input_batch.num_reqs:
        return dense("reasoning_eos_all")

    # Probe 64 bypasses the approximate DFlash2 QPN8 selector. The callback
    # gathers the same already-computed local projection on every TP rank.
    ids, logits, fallback = probe(sample_hidden_states, _TARGET_PROBE_K)
    if fallback is None:
        return dense("projection_has_no_reference")
    expanded = input_batch.expanded_idx_mapping
    logits, ids = sort_topk_with_vocab_ties(
        logits.float(), ids, vocab_size=states.vocab_size, descending=True
    )
    if device_only:
        reference_reqs = np.asarray(guarded, dtype=np.int64)
        cutoff_count = 0
    else:
        reference_rows = _compact_target_reference_rows_gpu(
            logits, states.top_p.gpu, expanded, strict_top_k=True
        )
        cutoffs = np.flatnonzero(
            np.logical_or.reduceat(reference_rows, input_batch.cu_num_logits_np[:-1])
        )
        reference_reqs = np.union1d(cutoffs, guarded) if guarded else cutoffs
        cutoff_count = cutoffs.size
    reference_logits = None
    packed_metadata = None
    if reference_reqs.size:
        # Multiproc broadcasts the ordered batch; all ranks derive these
        # cutoffs from the same gathered Top64 and request sampling params.
        # Reasoning EOS can add rows from rank-local state.
        # Restrict to all-gather so every rank receives the packed logits.
        if (
            reference_reqs.size < idx.size
            and reasoning_eos is None
            and fallback.processor.use_all_gather
        ):
            packed_metadata = _pack_reference_requests(input_batch, reference_reqs)
            reference_logits = fallback.gather_rows(packed_metadata[0])
        else:
            reference_logits = fallback()
        if reference_reqs.size == idx.size or reference_logits is None:
            logger.debug(
                "SM70 MTP compact target route=dense reason=%s requests=%d "
                "rows=%d cutoff_requests=%d reasoning_eos_requests=%d",
                "reference_all"
                if reference_reqs.size == idx.size
                else "reference_non_gather_rank",
                idx.size,
                logits.shape[0],
                cutoff_count,
                len(guarded),
            )
            return DFlash2LogitsFallback(reference_logits)

    if not device_only:
        ids = ids[:, :_TARGET_TOP_K]
        logits = logits[:, :_TARGET_TOP_K]
    indices = input_batch.logits_indices
    sampled, num_sampled = greedy_draft_sparse_topk_rejection_sample(
        ids,
        logits,
        input_batch.input_ids[indices],
        input_batch.cu_num_logits,
        input_batch.positions[indices],
        input_batch.idx_mapping,
        states.top_p.gpu,
        states.seeds.gpu,
        # Keep the configured output stride shared with the dense fallback.
        # The kernel bounds verification and the bonus by each actual row end.
        max_steps,
        sampler.use_fp64_gumbel,
        target_top_k=_TARGET_TOP_K if device_only else None,
    )
    if reference_logits is not None:
        reqs, dense_ids, dense_counts = _sample_reference_requests(
            reference_logits,
            reference_reqs,
            input_batch,
            rejection_sampler,
            None,
            packed_metadata=packed_metadata,
        )
        sampled.index_copy_(0, reqs, dense_ids)
        num_sampled.index_copy_(0, reqs, dense_counts)
    compact_reqs = idx.size - reference_reqs.size
    logger.info_once(
        "Using SM70 MTP compact target sampling (%s K%d).",
        "device" if device_only else "guarded",
        steps,
    )
    logger.debug(
        "SM70 MTP compact target route=%s compact_requests=%d compact_rows=%d "
        "reference_requests=%d cutoff_requests=%d reasoning_eos_requests=%d",
        "mixed" if reference_reqs.size else "compact",
        compact_reqs,
        compact_reqs * (steps + 1),
        reference_reqs.size,
        cutoff_count,
        len(guarded),
    )
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
    )
