# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent CPU/CUDA oracle for raw-E4M3 grouped NVFP4.

No native decoder or FP8 instructions. FP64 dots deliberately avoid imitating
an implementation-specific MMA association. Boundaries match the SM70 W4A16
reference: FP16 weights, gate/up, activated gate, product and per-route W2;
router slots are joined with FP32 FMA in original order, then rounded to FP16.
"""

from collections.abc import Callable

import torch

MAX_TOKENS = 40
MAX_ROUTES = MAX_TOKENS * 10
PLAN_EXPERTS = 8
PLAN_COUNTS = PLAN_EXPERTS + MAX_ROUTES
PLAN_STARTS = PLAN_COUNTS + MAX_ROUTES
PLAN_ROWS = PLAN_STARTS + MAX_ROUTES
PLAN_INTS = PLAN_ROWS + MAX_ROUTES
NIBBLE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def decode_e4m3(codes: torch.Tensor) -> torch.Tensor:
    """All 256 E4M3FN encodings, including signed zero/subnormals/NaN."""
    bits = codes.to(torch.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    magnitude = torch.where(
        exponent == 0,
        mantissa.float() * 2.0**-9,
        (1.0 + mantissa.float() / 8) * torch.exp2(exponent.float() - 7),
    )
    magnitude = torch.where((bits & 127) == 127, torch.nan, magnitude)
    return torch.where((bits & 128) != 0, -magnitude, magnitude)


def effective_scales(
    codes: torch.Tensor, globals_: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """One bank or expert of prepared g16 scales, FP32 multiply then FP16 RNE.

    Prepared W13 uses a global slot per physical N32 tile when interleaved,
    otherwise per half of N. W2 has one global. Layout is inherited from
    nvfp4_expand_raw_scales_sm70_out; no persistent expanded bank is needed
    by the new native op.
    """
    n = codes.shape[-1]
    columns = torch.arange(n, device=codes.device)
    if globals_.shape[-1] == 2:
        slots = (columns // 32) % 2 if interleaved else (columns >= n // 2).long()
    else:
        slots = torch.zeros(n, dtype=torch.long, device=codes.device)
    factors = globals_[..., slots].unsqueeze(-2)
    return (decode_e4m3(codes) * factors.float()).half()


def pack_tm_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical [K,N] E2M1 nibbles into a TurboMind/QPN bank [K,N/8]."""
    k, n = codes.shape
    values = codes.long().reshape(k // 8, 8, n // 32, 32).permute(2, 0, 3, 1)
    packed = torch.zeros((n // 32, k // 8, 32), dtype=torch.int64, device=codes.device)
    for logical, physical in enumerate(NIBBLE_ORDER):
        packed |= values[..., logical] << (4 * physical)
    return packed.to(torch.int32).reshape(k, n // 8)


def decode_tm_bank(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Prepared single expert to FP16 [K,N], independent of native code."""
    k, words = weight.shape
    n = words * 8
    tiles = weight.long().reshape(n // 32, k // 8, 32) & 0xFFFFFFFF
    codes = torch.stack([(tiles >> (4 * i)) & 15 for i in NIBBLE_ORDER], -1)
    codes = codes.permute(1, 3, 0, 2).reshape(k, n)
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
    values = magnitudes[codes & 7] * torch.where((codes & 8) != 0, -1, 1)
    return (values.float() * scales.repeat_interleave(16, dim=0).float()).half()


def grouped_plan_reference(ids: torch.Tensor) -> torch.Tensor:
    """Stable CSR, packs of eight, invalid expert IDs omitted."""
    plan = torch.full((PLAN_INTS,), -1, dtype=torch.int32)
    by_expert: dict[int, list[int]] = {}
    for route, expert in enumerate(ids.reshape(-1).tolist()):
        if 0 <= expert < 512:
            by_expert.setdefault(expert, []).append(route)
    pack, row = 0, 0
    for expert in sorted(by_expert):
        routes = by_expert[expert]
        for start in range(0, len(routes), 8):
            chunk = routes[start : start + 8]
            plan[PLAN_EXPERTS + pack] = expert
            plan[PLAN_COUNTS + pack] = len(chunk)
            plan[PLAN_STARTS + pack] = row
            plan[PLAN_ROWS + row : PLAN_ROWS + row + len(chunk)] = torch.tensor(chunk)
            row += len(chunk)
            pack += 1
    plan[0] = pack
    return plan


def moe_reference(
    x: torch.Tensor,
    ids: torch.Tensor,
    topk: torch.Tensor,
    w13: Callable[[int], torch.Tensor],
    w2: Callable[[int], torch.Tensor],
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 independent dots, FP16 stage boundaries, router-order FP32 FMA.

    w13 and w2 return logical decoded [H,2I] and [I,H] matrices. This also
    accepts small shapes for CPU unit tests. Invalid IDs contribute zero.
    """
    m, hidden = x.shape
    flat = ids.reshape(-1).long()
    active = sorted({e for e in flat.tolist() if 0 <= e < 512})
    if not active:
        return torch.zeros_like(x), torch.empty(flat.numel(), 0)
    width = w2(active[0]).shape[0]
    mid = torch.zeros((flat.numel(), width), dtype=torch.float16, device=x.device)
    routed = torch.zeros((flat.numel(), hidden), dtype=torch.float16, device=x.device)
    for expert in active:
        routes = (flat == expert).nonzero().flatten()
        gu = (x[routes // ids.shape[1]].double() @ w13(expert).double()).float().half()
        gate, up = (gu[:, 0::2], gu[:, 1::2]) if interleaved else gu.chunk(2, -1)
        gf = gate.float()
        activated = (gf / (1 + torch.exp(-gf))).half()
        mid[routes] = (activated.float() * up.float()).half()
        routed[routes] = (mid[routes].double() @ w2(expert).double()).float().half()
    acc = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        values = routed.reshape(m, ids.shape[1], hidden)[:, slot]
        # FP16 * FP32 is exact in FP64; one cast gives FP32 fmaf rounding.
        acc = (
            values.double() * topk[:, slot : slot + 1].double() + acc.double()
        ).float()
    return acc.half(), mid
