# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    gumbel_block_argmax,
    gumbel_noised_argmax,
    tl_rand32,
)


@triton.jit
def _compute_block_max_and_sumexp(logits):
    block_max = tl.max(logits, axis=0)
    block_sumexp = tl.where(
        block_max > float("-inf"),
        tl.sum(tl.exp(logits - block_max)),
        0.0,
    )
    return block_max, block_sumexp


@triton.jit
def _compute_block_max_and_sumexp_excl(logits, BLOCK_SIZE: tl.constexpr):
    # The sum of exponentials without one max element, whose exp(0) = 1
    # would round a small remainder away in FP32. Elements tied with the max
    # add exactly 1 each, so they are counted instead of summed.
    block_max = tl.max(logits, axis=0)
    below = tl.sum(tl.where(logits < block_max, tl.exp(logits - block_max), 0.0))
    num_ties = tl.sum((logits == block_max).to(tl.int32))
    block_sumexp_excl = tl.where(
        block_max > float("-inf"), below + (num_ties - 1).to(tl.float32), 0.0
    )
    return block_max, block_sumexp_excl


@triton.jit
def _compute_global_max_and_correction(
    local_max_ptr,
    local_max_stride,
    local_sumexp_excl_ptr,
    local_sumexp_excl_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    # logsumexp = global_max + correction. Block sums exclude each block's
    # max; only the single global max is excluded from the total, so the
    # other blocks' maxes are added back.
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_excl_ptr + logit_idx * local_sumexp_excl_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    global_max, max_block = tl.max(maxes, axis=0, return_indices=True)
    scaled = tl.where(
        blocks == max_block,
        sumexps,
        (1.0 + sumexps) * tl.exp(maxes - global_max),
    )
    scaled = tl.where(blocks_mask, scaled, 0.0)
    correction = tldevice.log1p(tl.sum(scaled))
    correction = tl.where(global_max > float("-inf"), correction, 0.0)
    return global_max, correction


@triton.jit
def _compute_single_lse_from_excl(
    local_max_ptr,
    local_max_stride,
    local_sumexp_excl_ptr,
    local_sumexp_excl_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    # One FP32 logsumexp, max + log(sum), from the same block stats. Used
    # only by the second-tier residual, so it is computed only on fallback.
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_excl_ptr + logit_idx * local_sumexp_excl_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    global_max = tl.max(maxes, axis=0)
    total = tl.sum(
        tl.where(blocks_mask, (1.0 + sumexps) * tl.exp(maxes - global_max), 0.0)
    )
    return global_max + tl.log(total)


@triton.jit
def _log_softmax(logits, global_max, correction):
    # (logits - max) - correction, not logits - (max + correction): adding a
    # small correction to a large max rounds it away in FP32. A fully masked
    # row (max = -inf) gives NaN, which the residual treats as empty.
    return (logits - global_max) - correction


@triton.jit
def _compute_block_stats_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [4, num_logits, num_blocks] block stats: target max, target sum, draft
    # max, draft sum (sums exclude one block max)
    block_stats_ptr,
    block_stats_plane_stride,
    block_stats_row_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    target_local_max_ptr = block_stats_ptr
    target_local_sumexp_ptr = block_stats_ptr + block_stats_plane_stride
    draft_local_max_ptr = block_stats_ptr + 2 * block_stats_plane_stride
    draft_local_sumexp_ptr = block_stats_ptr + 3 * block_stats_plane_stride
    target_local_max_stride = block_stats_row_stride
    target_local_sumexp_stride = block_stats_row_stride
    draft_local_max_stride = block_stats_row_stride
    draft_local_sumexp_stride = block_stats_row_stride
    logit_idx = tl.program_id(0)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)

    if draft_step_idx >= num_speculative_steps:
        # Bonus token. Max/argmax and summed exponentials are not needed.
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0:
        # Greedy sampling. Only the target max/argmax are needed.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr
            + logit_idx * target_local_argmax_stride
            + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        # Get local target max and summed exponentials.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        # Sums exclude one block max; see _compute_global_max_and_correction.
        target_max, target_sumexp = _compute_block_max_and_sumexp_excl(
            target_logits, BLOCK_SIZE
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            target_max,
        )
        tl.store(
            target_local_sumexp_ptr
            + logit_idx * target_local_sumexp_stride
            + block_idx,
            target_sumexp,
        )
        if HAS_DRAFT_LOGITS:
            # Get local draft max and summed exponentials.
            draft_logits = tl.load(
                draft_logits_ptr
                + req_state_idx * draft_logits_stride_0
                + draft_step_idx * draft_logits_stride_1
                + block_offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            draft_max, draft_sumexp = _compute_block_max_and_sumexp_excl(
                draft_logits, BLOCK_SIZE
            )
            tl.store(
                draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
                draft_max,
            )
            tl.store(
                draft_local_sumexp_ptr
                + logit_idx * draft_local_sumexp_stride
                + block_idx,
                draft_sumexp,
            )


@triton.jit
def _rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [2, num_reqs, 2]: (max, correction) of the rejected position's
    # logsumexp, target then draft
    rejected_logsumexp_ptr,
    rejected_logsumexp_plane_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [4, num_logits, num_blocks] block stats: target max, target sum, draft
    # max, draft sum (sums exclude one block max)
    block_stats_ptr,
    block_stats_plane_stride,
    block_stats_row_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    target_local_max_ptr = block_stats_ptr
    target_local_sumexp_ptr = block_stats_ptr + block_stats_plane_stride
    draft_local_max_ptr = block_stats_ptr + 2 * block_stats_plane_stride
    draft_local_sumexp_ptr = block_stats_ptr + 3 * block_stats_plane_stride
    target_local_max_stride = block_stats_row_stride
    target_local_sumexp_stride = block_stats_row_stride
    draft_local_max_stride = block_stats_row_stride
    draft_local_sumexp_stride = block_stats_row_stride
    target_rejected_logsumexp_ptr = rejected_logsumexp_ptr
    draft_rejected_logsumexp_ptr = (
        rejected_logsumexp_ptr + rejected_logsumexp_plane_stride
    )
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    rejected_step = 0
    target_max = 0.0
    target_correction = 0.0
    draft_max = 0.0
    draft_correction = 0.0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            if temp == 0.0:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
                target_blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
                target_blocks_mask = target_blocks < vocab_num_blocks
                target_local_max = tl.load(
                    target_local_max_ptr
                    + logit_idx * target_local_max_stride
                    + target_blocks,
                    mask=target_blocks_mask,
                    other=float("-inf"),
                )
                # NaNs can send the block argmax into the padded region.
                target_local_max = tl.where(
                    target_local_max != target_local_max,
                    float("-inf"),
                    target_local_max,
                )
                max_target_block_idx = tl.argmax(target_local_max, axis=0)
                target_argmax = tl.load(
                    target_local_argmax_ptr
                    + logit_idx * target_local_argmax_stride
                    + max_target_block_idx
                ).to(tl.int64)

                if SYNTHETIC_MODE:
                    pos = tl.load(pos_ptr + logit_idx)
                    u = tl_rand32(seed, pos, includes_zero=False)
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= u < rate
                else:
                    accepted &= target_argmax == draft_sampled
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    draft_sampled if accepted else target_argmax,
                )
            else:
                target_logit = tl.load(
                    target_logits_ptr + logit_idx * target_logits_stride + draft_sampled
                ).to(tl.float32)
                target_max, target_correction = _compute_global_max_and_correction(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                target_log_prob = _log_softmax(
                    target_logit, target_max, target_correction
                )
                pos = tl.load(pos_ptr + logit_idx)
                u = tl_rand32(seed, pos, includes_zero=False)
                if HAS_DRAFT_LOGITS:
                    draft_logit = tl.load(
                        draft_logits_ptr
                        + req_state_idx * draft_logits_stride_0
                        + i * draft_logits_stride_1
                        + draft_sampled
                    ).to(tl.float32)
                    draft_max, draft_correction = _compute_global_max_and_correction(
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        logit_idx,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                    )
                    draft_log_prob = _log_softmax(
                        draft_logit, draft_max, draft_correction
                    )
                else:
                    # One-hot draft: q(draft_token) = 1, log_q = 0.
                    draft_log_prob = 0

                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted &= u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted &= target_log_prob > tl.log(u) + draft_log_prob
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)
    tl.store(target_rejected_logsumexp_ptr + 2 * req_idx, target_max)
    tl.store(target_rejected_logsumexp_ptr + 2 * req_idx + 1, target_correction)
    tl.store(draft_rejected_logsumexp_ptr + 2 * req_idx, draft_max)
    tl.store(draft_rejected_logsumexp_ptr + 2 * req_idx + 1, draft_correction)


@triton.jit
def _stable_residual_logits(
    target_logits,
    draft_logits,
    target_max,
    target_correction,
    draft_max,
    draft_correction,
):
    # Residual max(p(x) - q(x), 0) in log form:
    # log(exp(a) - exp(b)) = a + log(-expm1(b - a)) if b < a.
    # Log probs keep (max, correction) apart, and expm1 keeps 1 - exp(b - a)
    # positive when exp(b - a) rounds to 1. b = -inf gives a; a = -inf with
    # b >= a, and NaN, fail b - a < 0 and give -inf.
    target_log_probs = _log_softmax(target_logits, target_max, target_correction)
    draft_log_probs = _log_softmax(draft_logits, draft_max, draft_correction)
    delta = draft_log_probs - target_log_probs
    return tl.where(
        delta < 0.0,
        target_log_probs + tl.log(-tldevice.expm1(delta)),
        float("-inf"),
    ).to(tl.float32)


@triton.jit
def _single_lse_residual_logits(target_logits, draft_logits, target_lse, draft_lse):
    # The previous residual arithmetic, with one FP32 logsumexp:
    # log(exp(a) - exp(b)) = a + log1p(-exp(b - a)) if exp(b - a) < 1.
    # Kept as a second tier: it can keep a residual that the stable form
    # loses (for example two tied target tokens), and vice versa.
    target_log_probs = target_logits - target_lse
    draft_log_probs = draft_logits - draft_lse
    ratio = tl.exp(draft_log_probs - target_log_probs)
    return tl.where(
        ratio < 1.0,
        target_log_probs + tldevice.log1p(-ratio),
        float("-inf"),
    ).to(tl.float32)


@triton.jit
def _resample_kernel(
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [2, num_reqs, 2]: (max, correction) of the rejected position's
    # logsumexp, target then draft
    rejected_logsumexp_ptr,
    rejected_logsumexp_plane_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    target_rejected_logsumexp_ptr = rejected_logsumexp_ptr
    draft_rejected_logsumexp_ptr = (
        rejected_logsumexp_ptr + rejected_logsumexp_plane_stride
    )
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. No resampling needed because
        # the target argmax is already in the sampled tensor.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    # Compute the residual logits to resample the rejected token from.
    if is_bonus:
        # Bonus token (no rejections). Directly use the target logits.
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + resample_idx * draft_logits_stride_1
            + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        residual_logits = _stable_residual_logits(
            target_logits,
            draft_logits,
            tl.load(target_rejected_logsumexp_ptr + 2 * req_idx),
            tl.load(target_rejected_logsumexp_ptr + 2 * req_idx + 1),
            tl.load(draft_rejected_logsumexp_ptr + 2 * req_idx),
            tl.load(draft_rejected_logsumexp_ptr + 2 * req_idx + 1),
        )
    else:
        # One-hot draft. The residual is just the target distribution with
        # the rejected draft token probability zeroed out.
        rejected_draft_token = tl.load(draft_sampled_ptr + resample_token_idx + 1)
        residual_logits = tl.where(
            block != rejected_draft_token,
            target_logits,
            float("-inf"),
        ).to(tl.float32)

    # Resample the rejected/bonus token.
    value, idx = gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,  # processed_logits_ptr
        0,  # processed_logits_stride
        0,  # processed_logits_col_stride
        None,  # processed_logits_col_ptr
        vocab_size,
        IS_DRAFTING=False,
        APPLY_TEMPERATURE=False,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + block_idx,
        token_id,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _row_gumbel_argmax(
    target_logits_ptr,
    target_logits_stride,
    token_idx,
    # Draft row of the rejected step; read only with SINGLE_LSE_RESIDUAL.
    draft_row_ptr,
    target_lse,
    draft_lse,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
    SINGLE_LSE_RESIDUAL: tl.constexpr,
):
    # One program walks the whole vocabulary, sampling either the target or
    # its single-logsumexp residual. The noise is keyed by token id, so this
    # equals a per-block Gumbel argmax followed by a first-max reduction over
    # blocks. Returns (token, noised value); value -inf means empty.
    if USE_FP64:
        best_value = tl.full((), float("-inf"), tl.float64)
    else:
        best_value = tl.full((), float("-inf"), tl.float32)
    best_token = tl.full((), 0, tl.int64)
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        block = block_start + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            target_logits_ptr + token_idx * target_logits_stride + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        if SINGLE_LSE_RESIDUAL:
            draft_logits = tl.load(
                draft_row_ptr + block, mask=mask, other=float("-inf")
            ).to(tl.float32)
            logits = _single_lse_residual_logits(
                logits, draft_logits, target_lse, draft_lse
            )
        value, idx = gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seed_ptr,
            pos_ptr,
            None,  # processed_logits_ptr
            0,  # processed_logits_stride
            0,  # processed_logits_col_stride
            None,  # processed_logits_col_ptr
            vocab_size,
            IS_DRAFTING=False,
            APPLY_TEMPERATURE=False,
            USE_FP64=USE_FP64,
        )
        # Strictly greater keeps the first block on ties and skips NaN.
        better = value > best_value
        best_token = tl.where(better, block_start + idx, best_token)
        best_value = tl.where(better, value, best_value)
    return best_token, best_value


@triton.jit
def _insert_resampled_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    # [num_logits, V], read only with HAS_FALLBACK
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [max_num_reqs, num_speculative_steps, V], read only with HAS_FALLBACK
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [4, num_logits, num_blocks] block stats (target max, target sum, draft
    # max, draft sum), read only with HAS_FALLBACK
    block_stats_ptr,
    block_stats_plane_stride,
    block_stats_row_stride,
    vocab_num_blocks,
    # [2] int64, written only with HAS_COUNT
    fallback_count_ptr,
    vocab_size,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
    HAS_FALLBACK: tl.constexpr,
    FALLBACK_BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. The target argmax is already
        # in the sampled tensor.
        return

    # Insert the resampled token.
    block = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = block < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    # Keep NaN block maxima from selecting padding before the index load.
    resampled_local_max = tl.where(
        resampled_local_max != resampled_local_max,
        float("-inf"),
        resampled_local_max,
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled = tl.load(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + resampled_max_block_idx,
    )
    if HAS_FALLBACK:
        # FP32 can round a real but tiny residual max(p - q, 0) to zero on
        # every token (e.g. a grammar that allows one token, with p and q
        # equal to the last bit on it). The argmax above then picks token 0,
        # outside the target support. Such rows are rare; redraw them here,
        # without an extra launch, from (1) the single-logsumexp residual,
        # which keeps some residuals the stable form loses, and otherwise
        # (2) the target distribution. The count is an integer, so other
        # rows run no new float math.
        num_residual_blocks = tl.sum(
            (resampled_local_max > float("-inf")).to(tl.int32), axis=0
        )
        if num_residual_blocks == 0:
            found = False
            if not is_bonus:
                second, second_value = _row_gumbel_argmax(
                    target_logits_ptr,
                    target_logits_stride,
                    resample_token_idx,
                    draft_logits_ptr
                    + req_state_idx * draft_logits_stride_0
                    + num_sampled * draft_logits_stride_1,
                    _compute_single_lse_from_excl(
                        block_stats_ptr,
                        block_stats_row_stride,
                        block_stats_ptr + block_stats_plane_stride,
                        block_stats_row_stride,
                        resample_token_idx,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                    ),
                    _compute_single_lse_from_excl(
                        block_stats_ptr + 2 * block_stats_plane_stride,
                        block_stats_row_stride,
                        block_stats_ptr + 3 * block_stats_plane_stride,
                        block_stats_row_stride,
                        resample_token_idx,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                    ),
                    expanded_idx_mapping_ptr,
                    temp_ptr,
                    seed_ptr,
                    pos_ptr,
                    vocab_size,
                    BLOCK_SIZE=FALLBACK_BLOCK_SIZE,
                    USE_FP64=USE_FP64,
                    SINGLE_LSE_RESIDUAL=True,
                )
                found = second_value > float("-inf")
                if found:
                    resampled = second
                    if HAS_COUNT:
                        tl.atomic_add(fallback_count_ptr, 1)
            if not found:
                resampled, _ = _row_gumbel_argmax(
                    target_logits_ptr,
                    target_logits_stride,
                    resample_token_idx,
                    draft_logits_ptr,
                    0.0,
                    0.0,
                    expanded_idx_mapping_ptr,
                    temp_ptr,
                    seed_ptr,
                    pos_ptr,
                    vocab_size,
                    BLOCK_SIZE=FALLBACK_BLOCK_SIZE,
                    USE_FP64=USE_FP64,
                    SINGLE_LSE_RESIDUAL=False,
                )
                if HAS_COUNT:
                    tl.atomic_add(fallback_count_ptr + 1, 1)
    tl.store(
        sampled_ptr + req_idx * sampled_stride + num_sampled,
        resampled,
    )


@triton.jit
def _dflash2_compact_target_row(
    target_ids_ptr,
    target_logits_ptr,
    target_stride,
    row,
    temperature,
    top_p,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LIMIT_K: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOP_K
    token_ids = tl.load(
        target_ids_ptr + row * target_stride + offsets,
        mask=mask,
        other=0,
    ).to(tl.int64)
    logits = tl.load(
        target_logits_ptr + row * target_stride + offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if temperature != 1.0:
        logits = logits / temperature

    if LIMIT_K < TOP_K:
        cutoff = tl.load(target_logits_ptr + row * target_stride + LIMIT_K - 1).to(
            tl.float32
        )
        if temperature != 1.0:
            cutoff = cutoff / temperature
        # Keep every member of a tied top-k boundary within the admitted
        # shortlist. The host guard rejects truncated boundary ties.
        mask &= logits >= cutoff
        logits = tl.where(mask, logits, float("-inf"))

    # get_topk_tokens_and_logits returns scores in descending order. Apply
    # nucleus filtering only within that exact top-k support, matching the
    # target's top-k-first sampling contract without materializing V columns.
    row_max = tl.max(logits, axis=0)
    unnormalized = tl.where(mask, tl.exp(logits - row_max), 0.0)
    normalizer = tl.sum(unnormalized, axis=0)
    probs = unnormalized / normalizer
    cumulative_before = tl.cumsum(probs, axis=0) - probs
    keep = mask & ((top_p >= 1.0) | (cumulative_before < top_p))
    logits = tl.where(keep, logits, float("-inf"))

    kept_max = tl.max(logits, axis=0)
    kept_sumexp = tl.sum(tl.where(keep, tl.exp(logits - kept_max), 0.0), axis=0)
    logsumexp = kept_max + tl.log(kept_sumexp)
    return token_ids, logits, keep, logsumexp


@triton.jit
def _dflash2_sparse_topk_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_logits, target_top_k]
    target_ids_ptr,
    target_logits_ptr,
    target_stride,
    # [max_num_reqs, num_speculative_steps, draft_top_k]
    draft_ids_ptr,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    # [num_logits]
    draft_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temperature_ptr,
    top_p_ptr,
    seed_ptr,
    # [num_logits]
    pos_ptr,
    num_speculative_steps: tl.constexpr,
    TARGET_TOP_K: tl.constexpr,
    TARGET_BLOCK_K: tl.constexpr,
    TARGET_LIMIT_K: tl.constexpr,
    DRAFT_TOP_K: tl.constexpr,
    DRAFT_BLOCK_K: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """DFlash2 chain rejection on compact target/draft supports.

    The target distribution is zero outside its top-k/top-p support and the
    DFlash2 proposal is zero outside selector top-k. Therefore both the
    acceptance ratio and ``relu(p - q)`` recovery distribution are exact on
    these compact sets; scanning or storing a full-vocabulary row is unnecessary.
    """
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    top_p = tl.load(top_p_ptr + req_state_idx).to(tl.float32)
    seed = tl.load(seed_ptr + req_state_idx)

    target_offsets = tl.arange(0, TARGET_BLOCK_K)
    target_mask = target_offsets < TARGET_TOP_K
    draft_offsets = tl.arange(0, DRAFT_BLOCK_K)
    draft_mask = draft_offsets < DRAFT_TOP_K

    accepted = True
    accepted_steps = tl.zeros((), dtype=tl.int32)
    for step in range(num_speculative_steps):
        if accepted and step < num_tokens - 1:
            row = start_idx + step
            target_ids, target_logits, target_keep, target_lse = (
                _dflash2_compact_target_row(
                    target_ids_ptr,
                    target_logits_ptr,
                    target_stride,
                    row,
                    temperature,
                    top_p,
                    TOP_K=TARGET_TOP_K,
                    BLOCK_K=TARGET_BLOCK_K,
                    LIMIT_K=TARGET_LIMIT_K,
                )
            )
            proposed = tl.load(draft_sampled_ptr + row + 1).to(tl.int64)
            target_proposed_logit = tl.max(
                tl.where(
                    target_keep & (target_ids == proposed),
                    target_logits,
                    float("-inf"),
                ),
                axis=0,
            )

            draft_base = (
                draft_logits_ptr
                + req_state_idx * draft_stride_0
                + step * draft_stride_1
            )
            draft_id_base = (
                draft_ids_ptr + req_state_idx * draft_stride_0 + step * draft_stride_1
            )
            draft_ids = tl.load(
                draft_id_base + draft_offsets,
                mask=draft_mask,
                other=0,
            ).to(tl.int64)
            draft_logits = tl.load(
                draft_base + draft_offsets,
                mask=draft_mask,
                other=float("-inf"),
            ).to(tl.float32)
            draft_max, draft_sumexp = _compute_block_max_and_sumexp(draft_logits)
            draft_lse = draft_max + tl.log(draft_sumexp)
            draft_proposed_logit = tl.max(
                tl.where(
                    draft_mask & (draft_ids == proposed),
                    draft_logits,
                    float("-inf"),
                ),
                axis=0,
            )

            pos = tl.load(pos_ptr + row)
            uniform = tl_rand32(seed, pos, includes_zero=False)
            accepted &= (target_proposed_logit - target_lse) > (
                tl.log(uniform) + draft_proposed_logit - draft_lse
            )
            tl.store(
                sampled_ptr + req_idx * sampled_stride + step,
                proposed,
            )
            accepted_steps += accepted.to(tl.int32)

    resample_step = accepted_steps
    resample_row = start_idx + resample_step
    target_ids, target_logits, target_keep, target_lse = _dflash2_compact_target_row(
        target_ids_ptr,
        target_logits_ptr,
        target_stride,
        resample_row,
        temperature,
        top_p,
        TOP_K=TARGET_TOP_K,
        BLOCK_K=TARGET_BLOCK_K,
        LIMIT_K=TARGET_LIMIT_K,
    )
    is_bonus = resample_row == end_idx - 1
    residual_logits = target_logits
    if not is_bonus:
        draft_base = (
            draft_logits_ptr
            + req_state_idx * draft_stride_0
            + resample_step * draft_stride_1
        )
        draft_id_base = (
            draft_ids_ptr
            + req_state_idx * draft_stride_0
            + resample_step * draft_stride_1
        )
        draft_ids = tl.load(
            draft_id_base + draft_offsets,
            mask=draft_mask,
            other=0,
        ).to(tl.int64)
        draft_logits = tl.load(
            draft_base + draft_offsets,
            mask=draft_mask,
            other=float("-inf"),
        ).to(tl.float32)
        draft_max, draft_sumexp = _compute_block_max_and_sumexp(draft_logits)
        draft_lse = draft_max + tl.log(draft_sumexp)

        id_match = target_ids[:, None] == draft_ids[None, :]
        draft_logits_at_target = tl.max(
            tl.where(
                id_match & target_mask[:, None] & draft_mask[None, :],
                draft_logits[None, :],
                float("-inf"),
            ),
            axis=1,
        )
        target_log_probs = target_logits - target_lse
        draft_log_probs = draft_logits_at_target - draft_lse
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            target_keep & (ratio < 1.0),
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        )

    pos = tl.load(pos_ptr + resample_row)
    _, selected_offset = gumbel_noised_argmax(
        residual_logits,
        target_ids,
        target_keep,
        seed,
        pos,
        temperature,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    selected_token = tl.load(
        target_ids_ptr + resample_row * target_stride + selected_offset
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + resample_step,
        selected_token,
    )
    tl.store(num_sampled_ptr + req_idx, resample_step + 1)


def dflash2_sparse_topk_rejection_sample(
    target_topk_ids: torch.Tensor,
    target_topk_logits: torch.Tensor,
    draft_topk_ids: torch.Tensor,
    draft_topk_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    use_fp64: bool = False,
    target_top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact DFlash2 rejection on compact top-k distributions.

    Callers must enforce the no-penalty/no-logit-bias contract and provide
    target candidates after a global top-k merge. Draft scores must be the
    temperature-applied scores used to draw the corresponding proposal.
    With target_top_k smaller than the shortlist, callers must retain all
    cutoff ties and order equal target scores by descending vocabulary ID.
    """
    if target_topk_ids.ndim != 2 or target_topk_logits.ndim != 2:
        raise ValueError("target top-k tensors must be two-dimensional")
    if target_topk_ids.shape != target_topk_logits.shape:
        raise ValueError("target top-k ids/logits shapes must match")
    if draft_topk_ids.ndim != 3 or draft_topk_logits.ndim != 3:
        raise ValueError("draft top-k tensors must be three-dimensional")
    if draft_topk_ids.shape != draft_topk_logits.shape:
        raise ValueError("draft top-k ids/logits shapes must match")
    if draft_topk_ids.shape[1] != num_speculative_steps:
        raise ValueError("draft top-k step count does not match configuration")
    if target_topk_ids.shape[0] != draft_sampled.shape[0]:
        raise ValueError("target rows and sampled-token rows must match")
    if not 0 < target_topk_ids.shape[1] <= 64:
        raise ValueError("target top-k width must be in [1, 64]")
    if target_top_k is None:
        target_top_k = target_topk_ids.shape[1]
    if not 0 < target_top_k <= target_topk_ids.shape[1]:
        raise ValueError("target cutoff must be within the candidate width")
    if not 0 < draft_topk_ids.shape[2] <= 64:
        raise ValueError("draft top-k width must be in [1, 64]")

    # The cutoff probe retains 21 columns before slicing to top-20. The kernel
    # already accepts a row stride, so preserve that view and avoid two copies.
    if (
        target_topk_ids.stride(-1) != 1
        or target_topk_logits.stride(-1) != 1
        or target_topk_ids.stride(0) != target_topk_logits.stride(0)
    ):
        target_topk_ids = target_topk_ids.contiguous()
        target_topk_logits = target_topk_logits.contiguous()
    draft_topk_ids = draft_topk_ids.contiguous()
    draft_topk_logits = draft_topk_logits.contiguous()
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = draft_sampled.new_empty(
        num_reqs,
        num_speculative_steps + 1,
        dtype=torch.int64,
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    _dflash2_sparse_topk_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_topk_ids,
        target_topk_logits,
        target_topk_ids.stride(0),
        draft_topk_ids,
        draft_topk_logits,
        draft_topk_ids.stride(0),
        draft_topk_ids.stride(1),
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        temperature,
        top_p,
        seed,
        pos,
        num_speculative_steps=num_speculative_steps,
        TARGET_TOP_K=target_topk_ids.shape[1],
        TARGET_BLOCK_K=triton.next_power_of_2(target_topk_ids.shape[1]),
        TARGET_LIMIT_K=target_top_k,
        DRAFT_TOP_K=draft_topk_ids.shape[2],
        DRAFT_BLOCK_K=triton.next_power_of_2(draft_topk_ids.shape[2]),
        USE_FP64=use_fp64,
        num_warps=1,
    )
    return sampled, num_sampled


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    # [2] int64; if given, counts rows whose stable residual was empty and
    # that were redrawn from [0] the single-logsumexp residual or [1] the
    # target distribution.
    fallback_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    has_draft_logits = draft_logits is not None

    # Without draft logits the kernels get None and zero strides; every use
    # is behind HAS_DRAFT_LOGITS / HAS_FALLBACK.
    draft_strides = draft_logits.stride() if draft_logits is not None else (0, 0, 0)

    # Compute the block-level logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.int64
    )
    # One backing tensor for the four FP32 block stats (target max, target
    # sum, draft max, draft sum), passed as one pointer and two strides.
    block_stats = target_logits.new_empty(
        4, num_logits, vocab_num_blocks, dtype=torch.float32
    )
    _compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        block_stats,
        block_stats.stride(0),
        block_stats.stride(1),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_strides[0],
        draft_strides[1],
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    # (max, correction) of the rejected position's logsumexp, target then
    # draft.
    rejected_logsumexp = target_logits.new_empty(2, num_reqs, 2, dtype=torch.float32)
    _rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        rejected_logsumexp,
        rejected_logsumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        block_stats,
        block_stats.stride(0),
        block_stats.stride(1),
        draft_sampled,
        draft_logits,
        draft_strides[0],
        draft_strides[1],
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        synthetic_conditional_rates,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        rejected_logsumexp,
        rejected_logsumexp.stride(0),
        draft_logits,
        draft_strides[0],
        draft_strides[1],
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
    )

    # Insert the resampled tokens into the output sampled.
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        target_logits,
        target_logits.stride(0),
        seed,
        pos,
        draft_logits,
        draft_strides[0],
        draft_strides[1],
        block_stats,
        block_stats.stride(0),
        block_stats.stride(1),
        vocab_num_blocks,
        fallback_count,
        vocab_size,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
        HAS_FALLBACK=has_draft_logits,
        FALLBACK_BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        USE_FP64=use_fp64,
        HAS_COUNT=fallback_count is not None,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        # The kernel reads num_sampled and then increments it in place, with
        # no barrier in between. With several warps, a later warp can load the
        # incremented count, take a different early-return branch, and leave
        # the cross-warp argmax with unwritten partials. One warp makes the
        # read-before-write order hold for the whole program.
        num_warps=1,
    )
    return sampled, num_sampled
