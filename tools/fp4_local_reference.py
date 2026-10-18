# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent CPU oracle for FP4-Local; no vLLM or CUDA imports.

CSF layout ported literally by slab/row/column from the codec contract in
local-inference-lab/vllm @ 13b13ad54b and vllm-project/vllm PR #60646
(825b240). This deliberately does not call the production vector decoder.
"""

import numpy as np
import torch


def csf_reference_bytes(fixed, exceptions):
    stream = fixed.cpu().numpy()
    slabs, width = stream.shape
    columns = (width // 16 - 1) * 2
    result = bytearray(slabs * 16 * columns)
    for slab in range(slabs):
        for row in range(16):
            base = int(stream[slab, row])
            for column in range(columns):
                packed = int(stream[slab, 16 + row * (columns // 2) + column // 2])
                code = (packed >> (4 * (column % 2))) & 15
                result[(slab * 16 + row) * columns + column] = base + code
    for word in exceptions.cpu().numpy().reshape(-1):
        word = int(word)
        result[word & 0xFFFFFF] = word >> 24
    return (
        torch.frombuffer(result, dtype=torch.uint8).clone().reshape(slabs * 16, columns)
    )


def e4m3_bytes(raw):
    raw = np.asarray(raw, dtype=np.uint8)
    exp, mant = (raw >> 3) & 15, raw & 7
    value = np.where(
        exp == 0,
        mant.astype(np.float32) / 512,
        (1 + mant.astype(np.float32) / 8) * np.exp2(exp.astype(np.float32) - 7),
    )
    return np.where(raw & 128, -value, value)


def nvfp4_reference(weight, raw_scales, global_scale):
    packed = weight.cpu().numpy()
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    magnitude = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    values = magnitude[codes & 7] * np.where(codes & 8, np.float32(-1), np.float32(1))
    scales = np.repeat(e4m3_bytes(raw_scales), 16, axis=-1)
    return torch.from_numpy(
        ((values * scales) * np.float32(global_scale)).astype(np.float16)
    )
