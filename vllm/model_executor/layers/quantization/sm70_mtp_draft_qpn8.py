# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from 1Cat-vLLM commit 1fa9c438bc.
"""Channel-QPN8 projections for the SM70 native MTP drafter.

The drafter's BF16 checkpoint layer runs as FP16 TurboMind GEMMs, re-reading
every weight byte at each of its draft steps. Its decoder projections are
requantized once to channel FP8 and run with native QPN8 kernels.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm import _sm70_ops
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)

_MARK = "_sm70_mtp_draft_qpn8"
_READY = "_sm70_mtp_draft_qpn8_ready"
_SUFFIXES = frozenset(
    {
        "fc",
        "fc_embedding",
        "fc_hidden",
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    }
)


def mark_mtp_draft_qpn8(model: nn.Module, *, max_capture_rows: int = 0) -> int:
    """Admit the drafter's unquantized decoder projections before loading."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    # Medium prefills and every graph shape use the native row chunks.
    max_native_rows = max(128, max_capture_rows)
    count = 0
    for module in model.modules():
        if (
            isinstance(module, LinearBase)
            and isinstance(module.quant_method, UnquantizedLinearMethod)
            and getattr(module, "prefix", "").rsplit(".", 1)[-1] in _SUFFIXES
            and getattr(module, "bias", None) is None
        ):
            setattr(module, _MARK, max_native_rows)
            count += 1
    return count


@torch.no_grad()
def prepare_mtp_draft_qpn8(layer: nn.Module) -> bool:
    if not getattr(layer, _MARK, False):
        return False
    weight = layer.weight
    if weight.dtype != torch.float16 or not current_platform.is_device_capability(70):
        return False
    n, k = weight.shape
    if n % 32 or k % 128:
        return False
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
        _SM70_CHANNEL_FP8_QPN8_CONFIGS,
    )
    from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
        prepare_channel_qpn8_weight,
    )

    codes, scales = prepare_channel_qpn8_weight(weight)
    split_k, nacc, prefetch = _SM70_CHANNEL_FP8_QPN8_CONFIGS.get(
        (k, n), (8 if k % 256 else 16, 2, False)
    )
    replace_parameter(layer, "weight", codes)
    layer.register_buffer("_sm70_mtp_draft_qpn8_scales", scales, persistent=False)
    layer._sm70_mtp_draft_qpn8_shape = (n, k)
    layer._sm70_mtp_draft_qpn8_config = (split_k, nacc, prefetch)
    layer._unquantized_fused_silu_and_mul = apply_mtp_draft_qpn8_fused_silu
    setattr(layer, _READY, True)
    logger.info_once(
        "SM70 MTP drafter projections use channel-QPN8 (FP16 weights released)."
    )
    return True


def apply_mtp_draft_qpn8(layer: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    if not getattr(layer, _READY, False):
        return None
    n, k = layer._sm70_mtp_draft_qpn8_shape
    split_k, nacc, prefetch = layer._sm70_mtp_draft_qpn8_config
    x_2d = x.reshape(-1, k).contiguous()
    out = x_2d.new_empty((x_2d.shape[0], n))
    _sm70_ops.fp8_qpn8_draft_dispatch_sm70_out(
        out,
        x_2d,
        layer.weight,
        layer._sm70_mtp_draft_qpn8_scales,
        split_k,
        nacc,
        prefetch,
        False,
        layer._sm70_mtp_draft_qpn8,
    )
    return out.reshape(*x.shape[:-1], n)


def apply_mtp_draft_qpn8_fused_silu(
    layer: nn.Module, x: torch.Tensor
) -> torch.Tensor | None:
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
        _SM70_CHANNEL_FP8_QPN8_GATED_CONFIG,
    )

    n, k = layer._sm70_mtp_draft_qpn8_shape
    if n % 64:
        return None
    split_k, nacc, prefetch = _SM70_CHANNEL_FP8_QPN8_GATED_CONFIG
    x_2d = x.reshape(-1, k).contiguous()
    out_features = n // 2
    out = x_2d.new_empty((x_2d.shape[0], out_features))
    _sm70_ops.fp8_qpn8_draft_dispatch_sm70_out(
        out,
        x_2d,
        layer.weight,
        layer._sm70_mtp_draft_qpn8_scales,
        split_k,
        nacc,
        prefetch,
        True,
        layer._sm70_mtp_draft_qpn8,
    )
    return out.reshape(*x.shape[:-1], out_features)
