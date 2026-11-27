# FlashInfer-SM70 primitives

This directory holds Volta (SM70) WMMA primitives in FlashInfer's layout conventions and one kernel built on them. The ReVolta serving path for Qwen3.8 Flash-Next does not use them. The experimental `FLASHINFER_SM70` attention backend (`vllm/v1/attention/backends/flashinfer_sm70.py`) is separate: it runs on Flash-V100 metadata and kernels and does not use this directory either.

| Path | Contents |
| --- | --- |
| `include/flashinfer/attention/sm70/volta_mma.cuh` | Native Volta WMMA fragment types and helpers |
| `csrc/h3_noncausal_sm70.cu` | FP16 non-causal attention for head dimension 128, built as the extension `vllm._h3_flashinfer_C` |
| `patches/` | Placeholder for a FlashInfer patch series; it contains no patches |

## WMMA primitives

`volta_mma.cuh` exposes fragment types for A, QK-B, PV-B and the FP32 accumulator. Its load, init/load-accumulator, MMA-update and store helpers let a caller keep one accumulator fragment in registers across a complete reduction.

The two update operations keep FlashInfer's `m16n16k16` logical layouts:

| Operation | Matrix layout | Function |
| --- | --- | --- |
| QK | Q row-major times K-transpose through a col-major B view | `mma_sync_m16n16k16_row_col_f16f16f32` |
| PV | P row-major times V row-major | `mma_sync_m16n16k16_row_row_f16f16f32` |

QK expects the physical K tile as row-major `[N, K]`; the row/col WMMA view then computes `Q[M, K] * K[N, K]^T`. PV expects V as row-major `[K, N]`. The fragment overloads update `AccumulatorFragment` without touching C memory. The pointer overloads remain as functional controls: `kInit` starts at zero and `kInplaceUpdate` loads C, but every call also stores C, so do not use them in a multi-K hot loop.

The implementation uses `nvcuda::wmma` when compiled for `sm_70`, which emits Volta `HMMA.884` instead of recreating an `m8n8k4` raw-HMMA decomposition.

## Non-causal attention kernel

`noncausal_attention(..., backend="FLASHINFER_SM70")` in `vllm/model_executor/layers/sm70_attention.py` calls the kernel. It takes contiguous BSHD FP16 tensors with matching query and key lengths and head dimension 128. If the extension is not in the wheel, the wrapper compiles it with `torch.utils.cpp_extension`. The inherited MiniMax H3 video-generation code is its only caller; that model is outside the ReVolta support scope.

`tools/prepare-flashinfer-sm70.sh` clones the pinned FlashInfer `v0.6.13` source, applies any patches in `patches/` and copies `include/` over it. The ReVolta build does not run it.

## See also

- [Building ReVolta](../BUILDING.md)
- [Support and known limits](../docs/known-limits.md)
