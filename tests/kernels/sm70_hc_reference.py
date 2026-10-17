# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU packing helpers for the SM70 HC batch-reuse layout tests.

Copied verbatim from benchmarks/kernels/benchmark_sm70_hc_batch_reuse.py at
69ff71dc82, before benchmarks/ was removed from ReVolta.
"""


def pack_weight(weight):
    """[4, hidden, 320] -> warp-contiguous branch/K fragments, no conversion."""
    if weight.ndim != 3 or weight.shape[0] != 4 or weight.shape[2] != 320:
        raise ValueError("HC up weight must have shape [4, hidden, 320]")
    hidden = weight.shape[1]
    if hidden not in (640, 2560):
        raise ValueError("Only the replicated or TP4 hidden shape is supported")
    return (
        weight.reshape(4, hidden // 8, 8, 20, 2, 8)
        .permute(1, 3, 4, 0, 2, 5)
        .contiguous()
    )


def pack_down_weight(weight, tile_n=32):
    """Replicated/sharded HC down -> warp fragments without dtype conversion."""
    if weight.ndim != 2 or weight.shape[1] != 10240 or weight.shape[0] not in (88, 336):
        raise ValueError("HC down weight must have shape [336 or 88, 10240]")
    if tile_n not in (16, 32):
        raise ValueError("HC down tile must have 16 or 32 columns")
    padded_n = 352 if weight.shape[0] == 336 else 96
    padded = weight.new_zeros(padded_n, 10240)
    padded[: weight.shape[0]].copy_(weight)
    return (
        padded.reshape(padded_n // tile_n, tile_n, 640, 2, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
