# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless packing of *prepared* SM70 AWQ metadata (no numeric conversion).

Each pair of consecutive TurboMind metadata records occupies five bytes:
{s0_lo, s0_hi, s1_lo, s1_hi, z0 | z1 << 4}. Pair along the prepared N axis,
after AWQ nibble ordering, gate/up interleaving and TurboMind conversion.
Only even N is supported; the exact TP4 lane needs no padding.
"""

import torch


def pack_int4_zero(metadata: torch.Tensor) -> torch.Tensor:
    """[... ,G,N,3] uint8 -> [... ,G,N/2,5] uint8, CPU or CUDA.

    The caller must supply native AWQ prepare output: its zeros are already
    masked to 0..15. Avoid a device-to-host range check during model loading.
    """
    if (
        metadata.dtype != torch.uint8
        or metadata.ndim < 3
        or metadata.shape[-1] != 3
        or metadata.shape[-2] % 2
    ):
        raise ValueError("int4 zero requires uint8 [...,G,even N,3]")
    packed = torch.empty(
        (*metadata.shape[:-2], metadata.shape[-2] // 2, 5),
        dtype=torch.uint8,
        device=metadata.device,
    )
    packed[..., :2] = metadata[..., 0::2, :2]
    packed[..., 2:4] = metadata[..., 1::2, :2]
    packed[..., 4] = metadata[..., 0::2, 2] | (metadata[..., 1::2, 2] << 4)
    return packed
