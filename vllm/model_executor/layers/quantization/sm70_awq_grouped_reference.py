# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch references for the SM70 grouped small-M AWQ MoE decode op.

Device-agnostic (CPU or CUDA) and independent of the native extension:

* the prepared TurboMind U4 bank layout used by the QPN M1 and grouped ops
  ([E, K, N/8] int32 words, N32 tiles x K8 rows x 32 columns, nibble order
  (0, 4, 1, 5, 2, 6, 3, 7)) with compact 3-byte (FP16 scale, uint8 zero) or
  4-byte (FP16 scale, FP16 bias) group-32 metadata;
* dequantization with the native FP16 boundary: fp16(q * s + fp16(-z * s)),
  one rounding (the hfma2 result), not (q - z) * s;
* the routing plan written by the op (ascending expert, ascending route;
  packs of at most `cap` rows over one row list sorted by expert);
* the MoE decode itself: FP64 dots, FP16 gate/up, SiLU in FP32 from the FP16
  gate, FP16 product, FP16 per-route W2 output, then a router-order FP32 fma
  reduction rounded once to FP16 (the QPN M1 boundaries).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

NIBBLE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
GROUP = 32
TOP_K = 10
HIDDEN = 2560
INTERMEDIATE = 160
EXPERTS = 512
MAX_TOKENS = 48
MAX_ROUTES = MAX_TOKENS * TOP_K
ROW_CAP = 48  # rows per pack at most (6 m8n8k4 row blocks)
# Plan layout (kPlan* in awq_grouped_decode_sm70.cu): [0] pack count, then
# per pack its expert, row count and first row, then one route per row.
PLAN_EXPERTS = 8
PLAN_COUNTS = PLAN_EXPERTS + MAX_ROUTES
PLAN_STARTS = PLAN_COUNTS + MAX_ROUTES
PLAN_ROWS = PLAN_STARTS + MAX_ROUTES
PLAN_INTS = PLAN_ROWS + MAX_ROUTES


def pack_cap(num_tokens: int, pack_rows: int = 0) -> int:
    """Rows per pack the op uses: 8 * ceil(M / 8), capped by pack_rows."""
    row_blocks = (num_tokens + 7) // 8
    if pack_rows:
        row_blocks = min(row_blocks, pack_rows // 8)
    return 8 * row_blocks


def pack_tm_bank(
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    compact: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one expert into the prepared bank layout.

    codes: [K, N] ints in 0..15; zeros: [K/32, N] ints in 0..15;
    scales: [K/32, N] FP16. Returns (weight [K, N/8] int32, metadata
    [K/32, N, 3] uint8 if compact else [K/32, N] int32).
    """
    k, n = codes.shape
    assert k % 8 == 0 and n % 32 == 0 and k % GROUP == 0
    values = codes.long().reshape(k // 8, 8, n // 32, 32).permute(2, 0, 3, 1)
    packed = torch.zeros((n // 32, k // 8, 32), dtype=torch.int64, device=codes.device)
    for logical, physical in enumerate(NIBBLE_ORDER):
        packed |= values[..., logical] << (4 * physical)
    # Two's-complement wrap into int32 storage.
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    weight = packed.to(torch.int32).reshape(k, n // 8)
    scale_bits = scales.contiguous().view(torch.int16).long() & 0xFFFF
    if compact:
        metadata = torch.stack(
            (scale_bits & 255, scale_bits >> 8, zeros.long() & 255), -1
        ).to(torch.uint8)
    else:
        bias = (-zeros.to(torch.float16) * scales).to(torch.float16)
        bias_bits = bias.contiguous().view(torch.int16).long() & 0xFFFF
        word = scale_bits | (bias_bits << 16)
        word = torch.where(word >= 2**31, word - 2**32, word)
        metadata = word.to(torch.int32)
    return weight, metadata


def decode_tm_bank(weight: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
    """Decode one prepared expert bank to its logical FP16 [K, N] matrix."""
    k, words = weight.shape
    n = words * 8
    tiles = weight.long().reshape(n // 32, k // 8, 32) & 0xFFFFFFFF
    codes = torch.empty(
        (n // 32, k // 8, 32, 8), dtype=torch.int64, device=weight.device
    )
    for logical, physical in enumerate(NIBBLE_ORDER):
        codes[..., logical] = (tiles >> (4 * physical)) & 0xF
    codes = codes.permute(1, 3, 0, 2).reshape(k, n)
    if metadata.dtype == torch.uint8:
        assert metadata.shape == (k // GROUP, n, 3)
        scale_bits = metadata[..., 0].long() | (metadata[..., 1].long() << 8)
        scale_bits = torch.where(scale_bits >= 2**15, scale_bits - 2**16, scale_bits)
        scales = scale_bits.to(torch.int16).view(torch.float16)
        zeros = metadata[..., 2].to(torch.float16)
        bias = (-zeros * scales).to(torch.float16)
    else:
        assert metadata.shape == (k // GROUP, n) and metadata.dtype == torch.int32
        word = metadata.long() & 0xFFFFFFFF
        lo, hi = word & 0xFFFF, word >> 16
        lo = torch.where(lo >= 2**15, lo - 2**16, lo)
        hi = torch.where(hi >= 2**15, hi - 2**16, hi)
        scales = lo.to(torch.int16).view(torch.float16)
        bias = hi.to(torch.int16).view(torch.float16)
    group = torch.arange(k, device=weight.device) // GROUP
    # q * s and the FP16 bias are exact in FP32 here, so one FP16 rounding
    # reproduces the native hfma2 result.
    return (codes.float() * scales[group].float() + bias[group].float()).to(
        torch.float16
    )


def grouped_plan_reference(ids: torch.Tensor, cap: int) -> torch.Tensor:
    """The int32 plan the native op writes for `ids` ([M, 10]) and `cap`.

    Only the meaningful prefix is defined: entries past the pack count and
    past the routed rows are left as -1 here (the native buffer is not
    cleared).
    """
    flat = ids.reshape(-1).tolist()
    plan = torch.full((PLAN_INTS,), -1, dtype=torch.int32)
    by_expert: dict[int, list[int]] = {}
    for route, expert in enumerate(flat):
        if 0 <= expert < EXPERTS:
            by_expert.setdefault(int(expert), []).append(route)
    pack = 0
    row = 0
    for expert in sorted(by_expert):
        rows = by_expert[expert]
        for start in range(0, len(rows), cap):
            chunk = rows[start : start + cap]
            plan[PLAN_EXPERTS + pack] = expert
            plan[PLAN_COUNTS + pack] = len(chunk)
            plan[PLAN_STARTS + pack] = row
            for i, route in enumerate(chunk):
                plan[PLAN_ROWS + row + i] = route
            row += len(chunk)
            pack += 1
    plan[0] = pack
    return plan


def plan_pack_routes(plan: torch.Tensor, pack: int) -> list[int]:
    """Route indices of one pack of a (native or reference) plan."""
    count = int(plan[PLAN_COUNTS + pack])
    start = int(plan[PLAN_STARTS + pack])
    return plan[PLAN_ROWS + start : PLAN_ROWS + start + count].tolist()


def plan_prefix_mask(plan: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the plan entries that grouped_plan_reference defines."""
    mask = torch.zeros(PLAN_INTS, dtype=torch.bool)
    packs = int(plan[0])
    mask[0] = True
    mask[PLAN_EXPERTS : PLAN_EXPERTS + packs] = True
    mask[PLAN_COUNTS : PLAN_COUNTS + packs] = True
    mask[PLAN_STARTS : PLAN_STARTS + packs] = True
    rows = int(plan[PLAN_COUNTS : PLAN_COUNTS + packs].sum()) if packs else 0
    mask[PLAN_ROWS : PLAN_ROWS + rows] = True
    return mask


def _fma_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """FP32 fmaf(a, b, c): exact FP64 product and sum, one FP32 rounding."""
    return (a.double() * b.double() + c.double()).float()


def awq_grouped_decode_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: Callable[[int], torch.Tensor],
    w2: Callable[[int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MoE decode with the QPN M1 FP16 boundaries.

    x: [M, 2560] FP16; topk_ids: [M, 10] int; topk_weights: [M, 10] FP32;
    w13(e) -> decoded [2560, 320] FP16 (gate/up interleaved columns);
    w2(e) -> decoded [160, 2560] FP16. Invalid ids (outside 0..511)
    contribute zero. Returns (out [M, 2560] FP16, intermediate [M*10, 160]
    FP16 with zero rows for invalid routes).
    """
    m = x.shape[0]
    device = x.device
    ids = topk_ids.reshape(-1).to(torch.int64)
    routes = ids.numel()
    intermediate = torch.zeros(
        (routes, INTERMEDIATE), dtype=torch.float16, device=device
    )
    route_out = torch.zeros((routes, HIDDEN), dtype=torch.float16, device=device)
    tokens = torch.arange(routes, device=device) // TOP_K
    for expert in sorted({int(e) for e in ids.tolist() if 0 <= int(e) < EXPERTS}):
        index = (ids == expert).nonzero().flatten()
        rows = x[tokens[index]].double()
        gate_up = rows @ w13(expert).double()
        # FP64 dot rounded to FP32 then FP16 (the kernel rounds an FP32 sum).
        gate = gate_up[:, 0::2].float().to(torch.float16)
        up = gate_up[:, 1::2].float().to(torch.float16)
        gate_f = gate.float()
        silu = (gate_f / (1.0 + torch.exp(-gate_f))).to(torch.float16)
        act = (silu.float() * up.float()).to(torch.float16)
        intermediate[index] = act
        route_out[index] = (
            (act.double() @ w2(expert).double()).float().to(torch.float16)
        )
    weights = topk_weights.reshape(m, TOP_K).float()
    values = route_out.view(m, TOP_K, HIDDEN).float()
    acc = torch.zeros((m, HIDDEN), dtype=torch.float32, device=device)
    for slot in range(TOP_K):
        acc = _fma_f32(weights[:, slot : slot + 1], values[:, slot], acc)
    return acc.to(torch.float16), intermediate


def random_bank(
    k: int,
    n: int,
    experts: list[int] | tuple[int, ...],
    compact: bool = True,
    seed: int = 0,
    device: str | torch.device = "cpu",
    scale_range: tuple[float, float] = (0.001, 0.011),
    num_experts: int = EXPERTS,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """A [num_experts, ...] bank with random contents for `experts` only.

    Returns (weight, metadata, decoded) with decoded[e] = [K, N] FP16.
    Other experts stay zero (weight and metadata), which decodes to 0.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.zeros((num_experts, k, n // 8), dtype=torch.int32, device=device)
    metadata = torch.zeros(
        (num_experts, k // GROUP, n, 3) if compact else (num_experts, k // GROUP, n),
        dtype=torch.uint8 if compact else torch.int32,
        device=device,
    )
    decoded = {}
    lo, hi = scale_range
    for expert in experts:
        codes = torch.randint(0, 16, (k, n), generator=generator)
        zeros = torch.randint(0, 16, (k // GROUP, n), generator=generator)
        scales = (torch.rand((k // GROUP, n), generator=generator) * (hi - lo) + lo).to(
            torch.float16
        )
        w, meta = pack_tm_bank(codes, zeros, scales, compact)
        weight[expert].copy_(w)
        metadata[expert].copy_(meta)
        decoded[int(expert)] = decode_tm_bank(w, meta).to(device)
    return weight, metadata, decoded


def build_random_awq_layer(seed: int = 4242, device: str = "cuda") -> torch.nn.Module:
    """A Qwen3.8 TP4 AWQ MoE layer (E=512, hidden 2560, intermediate 160,
    g32) on random checkpoint tensors, prepared by the production
    AWQSM70MoEMethod.process_weights_after_loading. Load-time switches
    (compact metadata, grouped/active-grouped admission) follow the
    environment at call time. The method is kept as ``layer._r2_method``.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.awq_sm70_moe import (
        AWQSM70MoEMethod,
    )

    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(
        tp_size=4, experts_per_token=TOP_K, has_bias=False
    )
    layer.global_num_experts = EXPERTS
    layer.local_num_experts = EXPERTS
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    method = AWQSM70MoEMethod(4, GROUP, True, layer)
    with torch.device(device):
        method.create_weights(layer, EXPERTS, HIDDEN, INTERMEDIATE, torch.float16)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _, param in layer.named_parameters():
        if param.dtype == torch.int32:
            param.data.copy_(
                torch.randint(
                    -(2**31),
                    2**31 - 1,
                    param.shape,
                    dtype=torch.int32,
                    device=device,
                    generator=generator,
                )
            )
        else:
            param.data.copy_(
                torch.rand(param.shape, device=device, generator=generator) * 0.01
                + 0.001
            )
    method.process_weights_after_loading(layer)
    layer._r2_method = method
    return layer
