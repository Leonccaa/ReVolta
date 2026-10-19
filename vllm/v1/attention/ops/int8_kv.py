# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write-only reference for the v1 planar INT8 cache (no reader fallback)."""

import torch

from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import INT8


def reference_int8_kv_write(
    key: torch.Tensor,
    value: torch.Tensor,
    view: KVCacheView,
    slot_mapping: torch.Tensor,
) -> None:
    """Independent of launch/batch shape; unique valid slots are required.

    Intended for CPU tests. Invalid slots (negative or beyond capacity) do not
    write either plane. Input rows beyond slot_mapping are graph padding.
    """
    if (
        view.codec is not INT8
        or key.ndim != 3
        or key.shape != value.shape
        or key.shape[1:] != view.key.shape[2:]
        or slot_mapping.ndim != 1
        or slot_mapping.numel() > key.shape[0]
    ):
        raise ValueError("Invalid INT8 writer shapes/codec")
    assert view.key_scale is not None and view.value_scale is not None
    capacity = view.key.shape[0] * view.key.shape[1]
    slots = slot_mapping.tolist()
    valid = [s for s in slots if 0 <= s < capacity]
    if len(valid) != len(set(valid)):
        raise ValueError("Concurrent INT8 writes require unique valid slots")
    for row, slot in enumerate(slots):
        if not 0 <= slot < capacity:
            continue
        page, token = divmod(slot, view.key.shape[1])
        for src, dst, scales in (
            (key, view.key, view.key_scale),
            (value, view.value, view.value_scale),
        ):
            payload, scale = INT8.encode(src[row])
            dst[page, token].copy_(payload)
            scales[page, token].copy_(scale)
