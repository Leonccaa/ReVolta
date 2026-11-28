# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded QPN8 vocabulary search for the original FP16 MTP target head.

Adapted from 1CatAI v1.5.4 ``qwen3_5_mtp_bounded_head.py`` (top-1 path).

For original weights W and QPN8 effective FP16 weights Q, Cauchy-Schwarz
bounds the score difference by ||x|| ||W-Q||. The stored coefficient also
covers arbitrary FP32 accumulation order and FP16 input subnormal flushing
for both projections. FP64 preparation rounds these positive coefficients
upward. Runtime bounds include input-norm and coarse-score rounding.

Candidates are selected by their upper bounds and scored from original weights.
A winner must strictly exceed every discarded upper bound across TP ranks.
Other rows execute the complete original-weight projection before the normal
global argmax.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm import _sm70_ops as sm70_ops
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
KEEP = 128
# On V100 the bounded search beats the dense head up to 12 rows; at 24 rows
# it is already slower even when every row certifies.
_MAX_BOUNDED_ROWS = 12


@triton.jit
def _score_id_keys(values, offsets, WideIds: tl.constexpr):
    # Signed zeros share a rank, and NaNs precede finite scores.
    bits = values.to(tl.uint16, bitcast=True).to(tl.uint32)
    ordered = tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
    ordered = tl.where(values == 0, 0x8000, ordered)
    ordered = tl.where(values != values, 0xFFFF, ordered)
    if WideIds:
        keys = ((ordered.to(tl.int64) - 0x8000) << 32) | (
            0xFFFFFFFF - offsets.to(tl.int64)
        )
    else:
        keys = ((ordered.to(tl.int32) - 0x8000) << 16) | (65535 - offsets).to(tl.int32)
    return keys


@triton.jit
def _topk_stage(
    Src,
    Dst,
    Count: tl.constexpr,
    Stages: tl.constexpr,
    Keep: tl.constexpr,
    Block: tl.constexpr,
    WideIds: tl.constexpr,
):
    row, stage = tl.program_id(0), tl.program_id(1)
    offsets = stage * Block + tl.arange(0, Block)
    sentinel: tl.constexpr = -9223372036854775808 if WideIds else -2147483648
    keys = tl.load(Src + row * Count + offsets, offsets < Count, sentinel)
    selected = tl.topk(keys, Keep)
    lane = tl.arange(0, Keep)
    tl.store(Dst + row * Stages * Keep + stage * Keep + lane, selected)


@triton.jit
def _topk_merge(
    Scratch,
    Ids,
    Count: tl.constexpr,
    Keep: tl.constexpr,
    Block: tl.constexpr,
    WideIds: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, Block)
    sentinel: tl.constexpr = -9223372036854775808 if WideIds else -2147483648
    keys = tl.load(Scratch + row * Count + offsets, offsets < Count, sentinel)
    keys = tl.topk(keys, Keep)
    id_mask: tl.constexpr = 0xFFFFFFFF if WideIds else 65535
    ids = id_mask - (keys & id_mask)
    lane = tl.arange(0, Keep)
    tl.store(Ids + row * Keep + lane, ids.to(tl.int32))


@triton.jit
def _input_norms(X, Norms, Width: tl.constexpr, Block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, Block)
    x = tl.load(X + row * Width + col, col < Width, 0).to(tl.float32)
    tiny = tl.where(tl.abs(x) < 0.00006103515625, x, 0.0)
    # 1 + 2^-10 covers even a serial 5120-term positive FP32 sum,
    # followed by sqrt and rounding. Nonfinite inputs propagate into the
    # bound, which then forces the dense path.
    norm = tl.sqrt(tl.sum(x * x, 0)) * 1.0009765625
    subnorm = tl.sqrt(tl.sum(tiny * tiny, 0)) * 1.0009765625
    tl.store(Norms + row * 2, norm)
    tl.store(Norms + row * 2 + 1, subnorm)


@triton.jit
def _compute_upper_bound(coarse, coefficient, norm_sum, xnorm, subnorm):
    error = (
        xnorm * coefficient
        + subnorm * norm_sum
        + tl.abs(coarse) * 0.0009765625
        + 0.000000059604644775390625
    )
    error = error * 1.0000038146972656
    upper = (coarse + error).to(tl.float16)
    bits = upper.to(tl.uint16, bitcast=True)
    next_bits = tl.where(upper < 0, bits - 1, bits + 1)
    next_bits = tl.where(upper == 0, 1, next_bits)
    next_bits = tl.where(upper == float("inf"), 0x7C00, next_bits)
    next_upper = next_bits.to(tl.uint16).to(tl.float16, bitcast=True)
    valid = (
        (coarse == coarse)
        & (tl.abs(coarse) != float("inf"))
        & (error == error)
        & (error != float("inf"))
    )
    return tl.where(valid, next_upper, float("inf"))


@triton.jit
def _upper_topk_tiles(
    Coarse,
    Coeff,
    Norms,
    Scratch,
    Vocab: tl.constexpr,
    Tiles: tl.constexpr,
    Keep: tl.constexpr,
    Block: tl.constexpr,
    WideIds: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * Block + tl.arange(0, Block)
    mask = col < Vocab
    coarse = tl.load(Coarse + row * Vocab + col, mask, 0).to(tl.float32)
    coefficient = tl.load(Coeff + col * 2, mask, 0)
    norm_sum = tl.load(Coeff + col * 2 + 1, mask, 0)
    xnorm, subnorm = tl.load(Norms + row * 2), tl.load(Norms + row * 2 + 1)
    upper = _compute_upper_bound(coarse, coefficient, norm_sum, xnorm, subnorm)
    values = tl.where(mask, upper, -float("inf"))
    keys = _score_id_keys(values, col, WideIds)
    sentinel: tl.constexpr = -9223372036854775808 if WideIds else -2147483648
    keys = tl.where(mask, keys, sentinel)
    selected = tl.topk(keys, Keep)
    lane = tl.arange(0, Keep)
    tl.store(Scratch + row * Tiles * Keep + tile * Keep + lane, selected)


@triton.jit
def _finish_packet(
    Ids,
    Scores,
    Coarse,
    Coeff,
    Norms,
    Packet,
    Vocab: tl.constexpr,
    Keep: tl.constexpr,
    Start: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, Block)
    score = tl.load(Scores + row * Keep + lane, lane < Keep, other=-float("inf")).to(
        tl.float32
    )
    ids = tl.load(Ids + row * Keep + lane, lane < Keep, other=2147483647)
    nan = score != score
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    best = tl.max(tl.where(nan, -float("inf"), score), 0)
    winner = tl.where(has_nan, nan, score == best)
    best_id = tl.min(tl.where(winner, ids, 2147483647), 0)
    best = tl.where(has_nan, float("nan"), best)
    last = tl.load(Ids + row * Keep + Keep - 1)
    coarse = tl.load(Coarse + row * Vocab + last).to(tl.float32)
    coefficient = tl.load(Coeff + last * 2)
    norm_sum = tl.load(Coeff + last * 2 + 1)
    xnorm, subnorm = tl.load(Norms + row * 2), tl.load(Norms + row * 2 + 1)
    cutoff = _compute_upper_bound(coarse, coefficient, norm_sum, xnorm, subnorm).to(
        tl.float32
    )
    tl.store(Packet + row * 3, best)
    tl.store(Packet + row * 3 + 1, (best_id + Start).to(tl.float32))
    tl.store(Packet + row * 3 + 2, cutoff)


@triton.jit
def _certify(Gathered, Flags, Ranks: tl.constexpr):
    # Certified rows keep each rank's candidate winner; the following global
    # top-1 exchange selects the certified winner among them.
    row = tl.program_id(0)
    ranks = tl.arange(0, Ranks)
    value = tl.load(Gathered + row * Ranks * 3 + ranks * 3)
    cutoff = tl.load(Gathered + row * Ranks * 3 + ranks * 3 + 2)
    valid = (
        tl.sum(((value != value) | (tl.abs(value) == float("inf"))).to(tl.int32), 0)
        == 0
    )
    flag = valid & (tl.max(value, 0) > tl.max(cutoff, 0))
    tl.store(Flags + row, flag.to(tl.int32))


@triton.jit
def _masked_max_tiles(
    Dense,
    Flags,
    Scratch,
    Vocab: tl.constexpr,
    Tiles: tl.constexpr,
    WideIds: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    if tl.load(Flags + row) == 0:
        col = tile * 1024 + tl.arange(0, 1024)
        value = tl.load(Dense + row * Vocab + col, col < Vocab, -float("inf"))
        keys = _score_id_keys(value, col, WideIds)
        sentinel: tl.constexpr = -9223372036854775808 if WideIds else -2147483648
        keys = tl.where(col < Vocab, keys, sentinel)
        tl.store(Scratch + row * Tiles + tile, tl.max(keys, 0))


@triton.jit
def _masked_max_finish(
    Dense,
    Flags,
    Scratch,
    Packet,
    Output,
    Vocab: tl.constexpr,
    Tiles: tl.constexpr,
    Start: tl.constexpr,
    Block: tl.constexpr,
    WideIds: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(Flags + row) != 0:
        value = tl.load(Packet + row * 3)
        token = tl.load(Packet + row * 3 + 1)
    else:
        offsets = tl.arange(0, Block)
        sentinel: tl.constexpr = -9223372036854775808 if WideIds else -2147483648
        key = tl.max(
            tl.load(Scratch + row * Tiles + offsets, offsets < Tiles, sentinel), 0
        )
        id_mask: tl.constexpr = 0xFFFFFFFF if WideIds else 65535
        local_id = id_mask - (key & id_mask)
        value = tl.load(Dense + row * Vocab + local_id).to(tl.float32)
        token = (local_id + Start).to(tl.float32)
    tl.store(Output + row * 2, value)
    tl.store(Output + row * 2 + 1, token)


def _prepare_proposal(weight: torch.Tensor):
    """Compute the exact QPN8 effective weights and upward-rounded coefficients."""
    vocab, width = weight.shape
    quant = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty(vocab, 1, device=weight.device, dtype=torch.float32)
    coefficients = torch.empty(vocab, 2, device=weight.device, dtype=torch.float32)
    epsilon = 2**-23
    gamma = (2 * width * epsilon) / (1 - 2 * width * epsilon)
    for begin in range(0, vocab, 512):
        original = weight[begin : begin + 512]
        scale = original.float().abs().amax(-1, keepdim=True) / 448
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = (original.float() / scale).to(torch.float8_e4m3fn)
        represented = ((codes.float() / 256) * (scale * 256).half().float()).half()
        quant[begin : begin + 512], scales[begin : begin + 512] = codes, scale
        residual = (original.double() - represented.double()).norm(dim=-1)
        norm_sum = original.double().norm(dim=-1) + represented.double().norm(dim=-1)
        subnorm_sum = torch.where(original.abs() < 2**-14, original, 0).double().norm(
            dim=-1
        ) + torch.where(represented.abs() < 2**-14, represented, 0).double().norm(
            dim=-1
        )
        exact = torch.stack((residual + gamma * norm_sum + subnorm_sum, norm_sum), -1)
        rounded = torch.nextafter(
            exact.float(), torch.full_like(exact.float(), float("inf"))
        )
        coefficients[begin : begin + 512] = rounded
    codes, scales = sm70_ops.fp8_qpn8_prepare_sm70(quant, scales)
    return codes, scales, coefficients


def _proposal(x, codes, scales, coefficients):
    rows, vocab = x.size(0), codes.numel() // x.size(1)
    coarse = torch.empty(rows, vocab, device=x.device, dtype=torch.float16)
    for start in range(0, rows, 40):
        stop = min(start + 40, rows)
        chunk_rows = stop - start
        prefetch = vocab < 65536 and chunk_rows <= (8 if x.size(1) == 2560 else 16)
        split_k = 8 if (x.size(1) == 2560 and chunk_rows <= 16) else 16
        nacc = 1 if (vocab >= 65536 and chunk_rows <= 16) else 2
        sm70_ops.fp8_qpn8_gemm_sm70_out(
            coarse[start:stop],
            x[start:stop],
            codes,
            scales,
            split_k,
            nacc,
            True,
            prefetch,
        )
    norms = torch.empty(rows, 2, device=x.device, dtype=torch.float32)
    norm_block = max(1024, triton.next_power_of_2(x.size(1)))
    _input_norms[(rows,)](
        x, norms, x.size(1), norm_block, num_warps=4 if norm_block <= 4096 else 8
    )
    wide_ids = vocab > 65536
    tile_block = 1024
    tiles = triton.cdiv(vocab, tile_block)
    count = tiles * KEEP
    scratch = torch.empty(
        rows,
        count,
        device=x.device,
        dtype=torch.int64 if wide_ids else torch.int32,
    )
    ids = torch.empty(rows, KEEP, device=x.device, dtype=torch.int32)
    _upper_topk_tiles[(rows, tiles)](
        coarse,
        coefficients,
        norms,
        scratch,
        vocab,
        tiles,
        KEEP,
        tile_block,
        wide_ids,
        num_warps=8,
    )
    while count > 1024:
        stages = triton.cdiv(count, 1024)
        reduced = torch.empty(rows, stages * KEEP, device=x.device, dtype=scratch.dtype)
        _topk_stage[(rows, stages)](
            scratch, reduced, count, stages, KEEP, 1024, wide_ids, num_warps=8
        )
        scratch = reduced
        count = stages * KEEP
    merge_block = max(64, triton.next_power_of_2(count))
    _topk_merge[(rows,)](
        scratch,
        ids,
        count,
        KEEP,
        merge_block,
        wide_ids,
        num_warps=8,
    )
    return coarse, norms, ids


def _gather_bounded_packet(packet: torch.Tensor) -> torch.Tensor:
    tp_group = get_tp_group()
    world_size = tp_group.world_size
    ca = getattr(tp_group.device_communicator, "ca_comm", None)
    if ca is not None and (
        not torch.cuda.is_current_stream_capturing() or ca._IS_CAPTURING
    ):
        rows = packet.size(0)
        stacked = packet.new_zeros((world_size, rows, 4))
        stacked[tp_group.rank_in_group, :, :3].copy_(packet)
        reduced = ca.custom_all_reduce(stacked)
        if reduced is not None:
            return reduced[:, :, :3].movedim(0, 1).reshape(rows, world_size * 3)
    return tensor_model_parallel_all_gather(packet, dim=-1)


def _bounded_pair(x, weight, codes, scales, coefficients, start):
    x = x.contiguous()
    rows, vocab = x.size(0), weight.size(0)
    coarse, norms, ids = _proposal(x, codes, scales, coefficients)
    scores = torch.empty(rows, KEEP, device=x.device, dtype=torch.float16)
    torch.ops._C.sm70_fp16_candidate_scores_out(scores, x, weight, ids, True)
    packet = torch.empty(rows, 3, device=x.device, dtype=torch.float32)
    _finish_packet[(rows,)](
        ids,
        scores,
        coarse,
        coefficients,
        norms,
        packet,
        vocab,
        KEEP,
        start,
        max(32, triton.next_power_of_2(KEEP)),
        num_warps=2,
    )
    gathered = _gather_bounded_packet(packet)
    flags = torch.empty(rows, device=x.device, dtype=torch.int32)
    _certify[(rows,)](gathered, flags, gathered.size(1) // 3, num_warps=1)
    # Uncertified rows recompute the complete original-weight projection.
    dense = coarse
    torch.ops._C.sm70_fp16_masked_head_out(dense, x, weight, flags, True)
    tiles = triton.cdiv(vocab, 1024)
    wide_ids = vocab > 65536
    scratch = torch.empty(
        rows, tiles, device=x.device, dtype=torch.int64 if wide_ids else torch.int32
    )
    output = torch.empty(rows, 2, device=x.device, dtype=torch.float32)
    _masked_max_tiles[(rows, tiles)](
        dense, flags, scratch, vocab, tiles, wide_ids, num_warps=4
    )
    _masked_max_finish[(rows,)](
        dense,
        flags,
        scratch,
        packet,
        output,
        vocab,
        tiles,
        start,
        triton.next_power_of_2(tiles),
        wide_ids,
        num_warps=2,
    )
    return output


class MTPBoundedTargetHead(nn.Module):
    """Preserve the dense head while owning the QPN8 search weights."""

    @torch.inference_mode()
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head
        self.shard_indices = head.shard_indices
        self.start = self.shard_indices.org_vocab_start_index
        codes, scales, coefficients = _prepare_proposal(head.weight.detach())
        # The draft QPN8 head quantizes the same weight identically; let it
        # reuse these codes instead of keeping a second FP8 copy.
        head.weight._sm70_shared_qpn8 = (codes, scales)
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer("coefficients", coefficients, persistent=False)

    @property
    def weight(self):
        return self.head.weight

    @property
    def quant_method(self):
        return self

    def apply(self, layer, x, bias=None):
        return self.head.quant_method.apply(self.head, x, bias)

    def maybe_get_sm70_lm_head_top1(self, hidden_states, bias=None):
        return self.head.maybe_get_sm70_lm_head_top1(hidden_states, bias)

    def maybe_get_sm70_lm_head_top1_pair(self, hidden_states, bias=None):
        # Larger batches keep the dense head, which is faster there.
        if bias is not None or not 1 <= hidden_states.shape[0] <= _MAX_BOUNDED_ROWS:
            return None
        return _bounded_pair(
            hidden_states,
            self.weight,
            self.codes,
            self.scales,
            self.coefficients,
            self.start,
        )


def prepare_mtp_bounded_target_head(
    head: nn.Module, config: VllmConfig
) -> MTPBoundedTargetHead | None:
    weight = head.weight
    shard = head.shard_indices
    speculative = config.speculative_config
    if (
        not current_platform.is_device_capability(70)
        or weight.dtype != torch.float16
        or weight.shape[1] not in (2560, 5120)
        or weight.shape[0] % 32
        or shard.num_org_vocab_padding
        or shard.num_added_elements
        or not isinstance(head.quant_method, UnquantizedEmbeddingMethod)
        or envs.VLLM_BATCH_INVARIANT
        or config.lora_config is not None
        or config.parallel_config.tensor_parallel_size == 1
        or speculative is None
        or speculative.method != "mtp"
        or speculative.draft_sample_method == "probabilistic"
    ):
        return None
    prepared = MTPBoundedTargetHead(head)
    logger.info_once(
        "SM70 MTP bounded target head prepared with original FP16 scoring "
        "and device-side full-vocabulary fallback."
    )
    return prepared
