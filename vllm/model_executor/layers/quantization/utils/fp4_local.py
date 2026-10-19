# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time CPU arithmetic for FP4-Local (never quantizes activations)."""

import torch


def dequantize_mxfp8_cpu(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """E4M3 weights times row-major E8M0 block32 scales, rounded once to FP16."""
    if (
        weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.uint8
        or weight.ndim != 2
        or weight.shape[1] % 32
        or scale.shape != (weight.shape[0], weight.shape[1] // 32)
    ):
        raise ValueError("Invalid MXFP8 E4M3/E8M0 block32 geometry")
    weight, scale = weight.cpu(), scale.cpu()
    if bool((scale == 255).any()):
        raise ValueError("MXFP8 E8M0 NaN scale")
    result = torch.empty(weight.shape, dtype=torch.float16, device="cpu")
    for start in range(0, weight.shape[0], 256):
        values = (
            weight[start : start + 256].float().reshape(-1, weight.shape[1] // 32, 32)
        )
        exponents = scale[start : start + 256].to(torch.int32) - 127
        result[start : start + 256] = (
            torch.ldexp(values, exponents[..., None])
            .reshape(-1, weight.shape[1])
            .half()
        )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("MXFP8 weights are not finite in FP16")
    return result


def dequantize_nvfp4_cpu(
    weight: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Independent W4A16 E2M1 x E4M3 block16 x FP32 multiplier reference."""
    if (
        weight.dtype != torch.uint8
        or weight.ndim != 2
        or scale.dtype != torch.float8_e4m3fn
        or weight.shape[1] % 8
        or scale.shape != (weight.shape[0], weight.shape[1] // 8)
        or global_scale.numel() != 1
    ):
        raise ValueError("Invalid NVFP4 block16 geometry")
    weight = weight.cpu()
    codes = torch.stack((weight & 15, weight >> 4), -1).reshape(weight.shape[0], -1)
    # E2M1 has signed zero and magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6.
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = lut[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    return (
        values
        * scale.cpu().float().repeat_interleave(16, -1)
        * global_scale.cpu().float().reshape(())
    ).half()
