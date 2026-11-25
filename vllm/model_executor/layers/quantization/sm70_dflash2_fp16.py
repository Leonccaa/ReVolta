# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed draft projections adapted from 1Cat-vLLM #892."""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)

# Local (N, K) shards of the Flash-Next DFlash2 draft. All five shapes
# admit the tile-32, eight-warp instantiation from the imported kernel.
_GEOMETRY = {
    (1536, 2560): (32, 8),  # QKV
    (2560, 1024): (32, 8),  # O
    (3840, 2560): (32, 8),  # gate/up
    (2560, 1920): (32, 8),  # down
    (640, 2560): (32, 8),  # replicated convolution coefficients
}


def pack_dflash2_fp16_weight(weight: torch.Tensor, tile: int) -> torch.Tensor:
    n, k = weight.shape
    slots = torch.arange(tile, device=weight.device)
    columns = (
        (slots & 3) | ((slots & 12) << 1) | ((slots & 16) >> 2)
        if tile == 32
        else (slots & 3) | ((slots & 4) << 1) | ((slots & 8) >> 1)
    )
    return (
        weight.view(n // tile, tile, k // 16, 16)
        .permute(0, 2, 1, 3)
        .index_select(2, columns)
        .contiguous()
    )


def prepare_dflash2_fp16_m8(layer: torch.nn.Module) -> None:
    weight = layer.weight
    geometry = _GEOMETRY.get(tuple(weight.shape))
    if geometry is None or layer.bias is not None:
        return
    tile, warps = geometry
    packed = pack_dflash2_fp16_weight(weight.detach(), tile)
    layer.register_buffer("_sm70_dflash2_fp16_packed", packed, persistent=False)
    layer._sm70_dflash2_fp16_geometry = geometry
    logger.info_once(
        "SM70 DFlash2 draft FP16 M8 enabled: N%d/K%d, tile=%d, warps=%d.",
        *weight.shape,
        tile,
        warps,
    )


def apply_dflash2_fp16_m8(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    weight = layer.weight
    # A dynamic draft trace may start outside M8. Let the native op select
    # M8 at runtime and use the ordinary dense product for its other rows.
    if not torch.compiler.is_compiling() and x.shape[0] != 8:
        return None
    output = x.new_empty((x.shape[0], weight.shape[0]))
    tile, warps = layer._sm70_dflash2_fp16_geometry
    ops.sm70_dflash2_fp16_dispatch_out(
        output, x.contiguous(), layer._sm70_dflash2_fp16_packed, weight, tile, warps
    )
    return output


if hasattr(torch.ops._C, "sm70_dflash2_fp16_dispatch_out"):

    @torch.library.register_fake("_C::sm70_dflash2_fp16_dispatch_out")
    def _fake_dispatch(output, input, packed, weight, tile, warps) -> None:
        return None
