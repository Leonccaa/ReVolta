# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def rotary_phases(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = positions.float()[:, None] * inv_freq
    return angles.cos(), angles.sin()


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Interleaved even/odd rotation for the single text position axis."""
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    return (
        torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
        .flatten(-2)
        .to(x.dtype)
    )
