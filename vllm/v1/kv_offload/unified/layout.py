# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit transport descriptions over overlapping rank-frame views.

Offsets describe backing, lengths describe payload, and completed worker jobs
report DMA. A dtype/scale description is an ABI declaration, not sampled data.
Packed attention stays an opaque whole physical owner; Mamba keeps its unpadded
prefix. Neither is reinterpreted by this module.
"""

from dataclasses import dataclass

from vllm.v1.kv_offload.base import CanonicalKVCaches


@dataclass(frozen=True)
class TensorTransfer:
    group: int
    tensor: int
    offset: int
    length: int
    subdivision: int
    dtypes: tuple[str, ...]
    scale: str
    gpu_stride: int
    cpu_stride: int
    member_offsets: tuple[int, ...]
    member_lengths: tuple[int, ...]


def transfer_layout(
    caches: CanonicalKVCaches, frame_stride: int, worker_bytes: int
) -> tuple[TensorTransfer, ...]:
    offsets = []
    cursor = 0
    for tensor in caches.tensors:
        offsets.append(cursor)
        cursor += tensor.page_size_bytes
    if cursor > worker_bytes:
        raise ValueError("Component tensors exceed the unified rank frame")
    result = []
    for group, refs in enumerate(caches.group_data_refs):
        for ref in refs:
            tensor = caches.tensors[ref.tensor_idx]
            if not 0 < ref.page_size_bytes <= tensor.page_size_bytes:
                raise ValueError("Invalid component payload length")
            if ref.subdivision < 1 or (
                ref.member_offsets
                and (
                    len(ref.member_offsets) != len(ref.member_lengths)
                    or len(ref.member_offsets) != len(ref.dtypes)
                    or any(
                        offset < 0
                        or length <= 0
                        or offset + length > ref.page_size_bytes
                        for offset, length in zip(
                            ref.member_offsets, ref.member_lengths
                        )
                    )
                )
            ):
                raise ValueError("Invalid tensor member description")
            result.append(
                TensorTransfer(
                    group,
                    ref.tensor_idx,
                    offsets[ref.tensor_idx],
                    ref.page_size_bytes,
                    ref.subdivision,
                    ref.dtypes,
                    ref.scale,
                    tensor.tensor.stride(0) * tensor.tensor.element_size(),
                    frame_stride,
                    tuple(offsets[ref.tensor_idx] + x for x in ref.member_offsets),
                    ref.member_lengths,
                )
            )
    return tuple(result)
