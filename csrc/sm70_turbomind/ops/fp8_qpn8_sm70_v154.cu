// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The QPN8 execution layout is derived from dnv2003/v100-skinny (MIT) and
// its block-scale adaptation in haohervchb/sglang-V100. Automatic operator
// dispatch is restricted to accepted SM70 tensor/layout contracts and can be
// disabled with VLLM_SM70_FP8_QPN8=0.
// See LICENSE.v100-skinny in this directory for the retained MIT notice.

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <mutex>

#include "activation_pack_sm70.cuh"
#include "../../ops.h"

namespace {

__device__ __forceinline__ void fp8x8_to_half2x4_fast(uint2 q, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x3f803f80u;
  unsigned permuted[4];
  permuted[0] = __byte_perm(q.x, q.y, 0x0400);
  permuted[1] = __byte_perm(q.x, q.y, 0x0501);
  permuted[2] = __byte_perm(q.x, q.y, 0x0602);
  permuted[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned value =
        ((permuted[i] << 8) & kSign) | ((permuted[i] << 7) & kExponentMantissa);
    out[i] = *reinterpret_cast<const half2*>(&value);
  }
}

__device__ __forceinline__ int qpn8_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

#define VLLM_SM70_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// M32-class two-phase GEMM with a templated row-tile count: RowTiles=4 covers
// M=17..32 (four independent 8-row accumulator tiles), RowTiles=2 covers
// M=9..16. Keeping all logical split-K warps resident would require 64 KiB of
// static reduction storage for split-16. Instead, half as many physical warps
// execute the original logical warp ranges in two ordered phases. The compact
// first-half sum lets the final reduction retain the exact
// p0 + ... + p(SplitK-1) order. Each projection uses
// (SplitK / 2 + 1) * RowTiles * 1 KiB of reduction storage and reuses B over
// all RowTiles * 8 rows.
// BlockScaled=true switches the scale operand from the channel layout
// [N] (K-invariant, loaded once per column) to the block layout
// [K/128, N_total/32] consumed in-loop every eight k16-groups, mirroring the
// M<=8 packed kernel's reload pattern. false preserves the original channel
// semantics bit-for-bit.
template <int SplitK, bool PackedInput = false, bool Gated = false,
          bool FullRows = false, bool BlockScaled = false, int RowTiles = 4,
          int KUnroll = 0>
__global__ __launch_bounds__(
    32 * (SplitK / 2) * (Gated ? 2 : 1),
    (32 * (SplitK / 2) * (Gated ? 2 : 1) == 128)
        ? 3
        : 2) void fp8_qpn8_m32_twophase_sm70_kernel(const uint8_t* __restrict__ codes,
                                                    const half* __restrict__ channel_scales,
                                                    const half* __restrict__ input,
                                                    half* __restrict__ output,
                                                    half* __restrict__ k16_output,
                                                    int n, int k, int m) {
  static_assert(SplitK == 8 || SplitK == 12 || SplitK == 16,
                "M32 two-phase QPN8 supports split-8, split-12 or split-16");
  static_assert(RowTiles == 2 || RowTiles == 4 || RowTiles == 5,
                "two-phase QPN8 supports two, four or five row tiles");
  static_assert(KUnroll == 0 || KUnroll == 1 || KUnroll == 2 || KUnroll == 4,
                "M32 two-phase QPN8 K-unroll must be 0 (auto), 1, 2 or 4");
  if constexpr (FullRows) m = RowTiles * 8;
  constexpr int kPhysicalWarps = SplitK / 2;
  constexpr int kRowTiles = RowTiles;
  constexpr int kOutputElements = kRowTiles * 256;
  // With four or five row tiles, unrolling four K groups exceeds 128 registers
  // and prevents a second CTA from residing on an SM70 SM.
  constexpr int kKUnroll =
      RowTiles >= 4 ? 1 : (KUnroll > 0 ? KUnroll : (PackedInput ? 1 : 4));
  constexpr int kProjections = Gated ? 2 : 1;
  constexpr int kThreads = 32 * kPhysicalWarps * kProjections;
  constexpr int kPartialSlots = (kOutputElements + kThreads - 1) / kThreads;
  constexpr int kUpSmemSlots = (RowTiles == 5 && Gated && SplitK == 8) ? 2 : 0;
  constexpr int kUpRegSlots =
      (RowTiles == 5 && Gated) ? (kPartialSlots - kUpSmemSlots) : 1;
  __shared__ float reduction_storage[kProjections]
                                    [kPhysicalWarps + (RowTiles <= 4 ? 1 : 0)]
                                    [kOutputElements];
  __shared__ float first_sum_smem[RowTiles == 5 ? kOutputElements : 1];
  __shared__ float
      first_sum_up_smem[kUpSmemSlots > 0 ? kUpSmemSlots * kThreads : 1];

  const int lane = threadIdx.x & 31;
  const int warp =
      Gated ? (threadIdx.x >> 5) % kPhysicalWarps : threadIdx.x >> 5;
  const int projection = Gated ? (threadIdx.x >> 5) / kPhysicalWarps : 0;
  const int tile = blockIdx.x + projection * (n / 32);
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_warp = groups_k16 / SplitK;
  const uint4* code_ptr = reinterpret_cast<const uint4*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  // Block scales reuse the [K/128, N_total/32] layout of the M<=8 packed
  // kernel: one half per (scale_group, tile) pair, broadcast into a half2 for
  // the weight dequant. With Gated, both projections' codes live in one tile
  // stream, so the scale row stride spans both projections' tiles. Channel
  // scales keep their per-column K-invariant semantics.
  const int scale_row_tiles = (Gated ? 2 : 1) * (n >> 5);
  const half* block_scale_ptr = channel_scales + tile;
  half loaded_scale;
  int loaded_scale_group = -1;
  half next_scale = __float2half(0.0f);
  half2 scale2;
  if constexpr (BlockScaled) {
    loaded_scale = __float2half(0.0f);
  } else {
    loaded_scale = __ldg(channel_scales + tile * 32 + qpn8_col_from_lane(lane));
  }
  scale2 = __halves2half2(loaded_scale, loaded_scale);
  float first_sum_up[kUpRegSlots] = {};

#pragma unroll
  for (int phase = 0; phase < 2; ++phase) {
    float accum[kRowTiles][2][8];
#pragma unroll
    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
      for (int chain = 0; chain < 2; ++chain) {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          accum[row_tile][chain][index] = 0.0f;
        }
      }
    }

    const int logical_warp = warp + phase * kPhysicalWarps;
    const int group_begin = logical_warp * groups_per_warp;
    constexpr bool kPrefetch = FullRows;
    uint4 next_codes = make_uint4(0, 0, 0, 0);
    if constexpr (kPrefetch) {
      next_codes = __ldcs(code_ptr + static_cast<size_t>(group_begin) * 32);
    }
    if constexpr (BlockScaled) {
      loaded_scale_group = -1;
      next_scale = __float2half(0.0f);
      if constexpr (kPrefetch) {
        next_scale =
            __ldg(block_scale_ptr +
                  static_cast<size_t>(group_begin >> 3) * scale_row_tiles);
      }
    }
#pragma unroll kKUnroll
    for (int group = group_begin; group < group_begin + groups_per_warp;
         ++group) {
      if constexpr (BlockScaled) {
        const int scale_group = group >> 3;
        if (scale_group != loaded_scale_group) {
          if constexpr (kPrefetch) {
            loaded_scale = next_scale;
            const int next_sg = scale_group + 1;
            if ((next_sg << 3) < group_begin + groups_per_warp) {
              next_scale =
                  __ldg(block_scale_ptr +
                        static_cast<size_t>(next_sg) * scale_row_tiles);
            }
          } else {
            loaded_scale =
                __ldg(block_scale_ptr +
                      static_cast<size_t>(scale_group) * scale_row_tiles);
          }
          loaded_scale_group = scale_group;
          scale2 = __halves2half2(loaded_scale, loaded_scale);
        }
      }
      const uint4 packed =
          kPrefetch ? next_codes
                    : __ldcs(code_ptr + static_cast<size_t>(group) * 32);
      if constexpr (kPrefetch) {
        if (group + 1 < group_begin + groups_per_warp) {
          next_codes = __ldcs(code_ptr + static_cast<size_t>(group + 1) * 32);
        }
      }
      half2 weights[8];
      fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
      fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights + 4);
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        weights[index] = __hmul2(weights[index], scale2);
      }

      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
      for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
        uint4 input01 = make_uint4(0, 0, 0, 0);
        uint4 input23 = make_uint4(0, 0, 0, 0);
        const int input_row_idx = row_tile * 8 + row;
        if (input_row_idx < m) {
          const half* input_row =
              PackedInput
                  ? input +
                        (static_cast<size_t>(group) * m + input_row_idx) * 16
                  : input + static_cast<size_t>(input_row_idx) * k + group * 16;
          input01 = *reinterpret_cast<const uint4*>(input_row);
          input23 = *reinterpret_cast<const uint4*>(input_row + 8);
        }
        const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
        const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][1], a0[2], a0[3], b[2], b[3]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][0], a1[0], a1[1], b[4], b[5]);
        VLLM_SM70_MMA_8N8K4(accum[row_tile][1], a1[2], a1[3], b[6], b[7]);
      }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][1][index];
        const int output_row =
            row_tile * 8 + (index & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col =
            (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
        reduction_storage[projection][warp]
                         [output_row * 32 + quadpair * 8 + output_col] =
                             accum[row_tile][0][index];
      }
    }
    __syncthreads();

    auto reduce_slot = [&](int element, int slot) {
      float value = 0.0f, up = 0.0f;
      if (phase != 0) {
        if constexpr (RowTiles == 5) {
          value = first_sum_smem[element];
          if constexpr (Gated) {
            up = slot < kUpSmemSlots ? first_sum_up_smem[element]
                                     : first_sum_up[slot - kUpSmemSlots];
          }
        } else {
          value = reduction_storage[0][kPhysicalWarps][element];
          if constexpr (Gated)
            up = reduction_storage[1][kPhysicalWarps][element];
        }
      }
#pragma unroll
      for (int k_warp = 0; k_warp < kPhysicalWarps; ++k_warp) {
        value += reduction_storage[0][k_warp][element];
        if constexpr (Gated) up += reduction_storage[1][k_warp][element];
      }
      if (phase == 0) {
        if constexpr (RowTiles == 5) {
          first_sum_smem[element] = value;
          if constexpr (Gated) {
            if (slot < kUpSmemSlots) {
              first_sum_up_smem[element] = up;
            } else {
              first_sum_up[slot - kUpSmemSlots] = up;
            }
          }
        } else {
          reduction_storage[0][kPhysicalWarps][element] = value;
          if constexpr (Gated)
            reduction_storage[1][kPhysicalWarps][element] = up;
        }
      } else {
        const int output_row = element >> 5;
        const int output_col = element & 31;
        if (output_row < m) {
          if constexpr (Gated) value = (value / (1.0f + __expf(-value))) * up;
          const half result = __float2half(value);

          output[static_cast<size_t>(output_row) * n + blockIdx.x * 32 +
                 output_col] = result;
          if constexpr (Gated) {
            if (k16_output != nullptr) {
              k16_output[vllm::sm70::k16_half_index(
                  output_row, blockIdx.x * 32 + output_col, m)] = result;
            }
          }
        }
      }
    };
    if constexpr (RowTiles == 5) {
      // Constant indices keep M40 first-phase partials in registers.
#pragma unroll
      for (int slot = 0; slot < kPartialSlots; ++slot) {
        const int element = threadIdx.x + slot * kThreads;
        if (element < kOutputElements) reduce_slot(element, slot);
      }
    } else {
      // Preserve the measured M32 reduction schedule.
      for (int element = threadIdx.x; element < kOutputElements;
           element += blockDim.x) {
        reduce_slot(element, element / kThreads);
      }
    }
    __syncthreads();
  }
}
template <int SplitK, bool PackedInput = false, bool Gated = false,
          bool BlockScaled = false, int RowTiles = 4, int KUnroll = 0,
          bool FullRows = false>
void launch_fp8_qpn8_m32_twophase_sm70(const uint8_t* codes,
                                       const half* channel_scales,
                                       const half* input, half* output, int n,
                                       int k, int m, cudaStream_t stream,
                                       half* k16_output = nullptr) {
  constexpr int kPhysicalWarps = SplitK / 2;
  if constexpr (!FullRows && RowTiles == 4) {
    if (m > RowTiles * 8 && m <= (RowTiles + 1) * 8) {
      launch_fp8_qpn8_m32_twophase_sm70<SplitK, PackedInput, Gated, BlockScaled,
                                        RowTiles + 1, KUnroll>(
          codes, channel_scales, input, output, n, k, m, stream, k16_output);
      return;
    }
  }
  auto kernel =
      fp8_qpn8_m32_twophase_sm70_kernel<SplitK, PackedInput, Gated, FullRows,
                                        false, BlockScaled, RowTiles, KUnroll>;
  if constexpr (!FullRows) {
    // A full row-tile set drops the per-group row guard; the math per row is
    // identical, so this only fires when the batch fills the tile set.
    if (m == RowTiles * 8) {
      kernel = fp8_qpn8_m32_twophase_sm70_kernel<SplitK, PackedInput, Gated,
                                                 true, false, BlockScaled,
                                                 RowTiles, KUnroll>;
    }
  }
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  kernel<<<(n / 32), (32 * kPhysicalWarps * (Gated ? 2 : 1)), 0, stream>>>(
      codes, channel_scales, input, output, k16_output, n, k, m);
}

}  // namespace

void fp8_qpn8_gemm_v154_sm70_out(torch::Tensor out, torch::Tensor input,
                                 torch::Tensor codes,
                                 torch::Tensor group_scales, int64_t split_k,
                                 int64_t accumulator_chains, bool fast_decoder,
                                 bool prefetch_codes) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() && codes.is_cuda() &&
                  group_scales.is_cuda(),
              "fp8_qpn8_gemm_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "fp8_qpn8_gemm_sm70_out: input and output must be float16");
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8,
              "fp8_qpn8_gemm_sm70_out: codes must be uint8");
  TORCH_CHECK(group_scales.scalar_type() == torch::kFloat16,
              "fp8_qpn8_gemm_sm70_out: group scales must be float16");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && group_scales.dim() == 2,
              "fp8_qpn8_gemm_sm70_out: expected 2D tensors");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous() &&
                  codes.is_contiguous() && group_scales.is_contiguous(),
              "fp8_qpn8_gemm_sm70_out: tensors must be contiguous");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == codes.get_device() &&
                  input.get_device() == group_scales.get_device(),
              "fp8_qpn8_gemm_sm70_out: tensors must share one device");

  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = out.size(1);
  TORCH_CHECK(m >= 1 && m <= 40,
              "fp8_qpn8_gemm_sm70_out: M must be in [1, 40]");
  TORCH_CHECK(out.size(0) == m, "fp8_qpn8_gemm_sm70_out: output M mismatch");
  TORCH_CHECK(n > 0 && n % 32 == 0,
              "fp8_qpn8_gemm_sm70_out: N must be a positive multiple of 32");
  const bool channel_scales =
      group_scales.size(0) == 1 && group_scales.size(1) == n;
  const bool block_scales =
      group_scales.size(0) == k / 128 && group_scales.size(1) == n / 32;
  TORCH_CHECK(channel_scales ? (k > 0 && k % 16 == 0) : (k > 0 && k % 128 == 0),
              "fp8_qpn8_gemm_sm70_out: channel scales require K%16=0; "
              "block scales require K%128=0");
  TORCH_CHECK(codes.numel() == n * k,
              "fp8_qpn8_gemm_sm70_out: packed code size mismatch");
  TORCH_CHECK(channel_scales || block_scales,
              "fp8_qpn8_gemm_sm70_out: scale shape mismatch");
  TORCH_CHECK(m <= 8 || channel_scales,
              "fp8_qpn8_gemm_sm70_out: M=9..40 requires channel scales");
  TORCH_CHECK(split_k == 4 || split_k == 8 || split_k == 12 || split_k == 16 ||
                  split_k == 32,
              "fp8_qpn8_gemm_sm70_out: unsupported split_k");
  TORCH_CHECK((k / 16) % split_k == 0,
              "fp8_qpn8_gemm_sm70_out: K/16 must be divisible by split_k");
  TORCH_CHECK(m <= 8 || split_k <= 16,
              "fp8_qpn8_gemm_sm70_out: M=9..40 does not support split_k 32");
  TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
              "fp8_qpn8_gemm_sm70_out: accumulator_chains must be 1 or 2");
  TORCH_CHECK(!prefetch_codes || fast_decoder,
              "fp8_qpn8_gemm_sm70_out: prefetch experiment requires the "
              "fast decoder");
  const bool exact_shape_split = split_k == 12;
  TORCH_CHECK(!exact_shape_split ||
                  (accumulator_chains == 2 && fast_decoder && !prefetch_codes),
              "fp8_qpn8_gemm_sm70_out: split_k 12 requires the "
              "fast decoder, two accumulator chains, and no prefetch");
  TORCH_CHECK(
      m <= 16 || ((split_k == 12 || split_k == 16) && accumulator_chains == 2 &&
                  fast_decoder && !prefetch_codes),
      "fp8_qpn8_gemm_sm70_out: M=17..40 requires split_k 12/16, "
      "the fast decoder, two accumulator chains, and no prefetch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr =
      reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>());
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

  if (m > 16) {
    // Amortize the extra activation kernel only over large dense projections.
    // A full M32 tile also benefits on short-K/split-12 projections. Partial
    // row tiles retain their established unroll tradeoff.
    if (n >= 2048 && ((split_k == 16 && k >= 4096) ||
                      ((m % 8 == 0) && split_k == 12 && k >= 1536))) {
      torch::Tensor packed_input;
      auto* packed_ptr =
          vllm::sm70::k16_packed_input(input, packed_input, stream);
      if (split_k == 12) {
        launch_fp8_qpn8_m32_twophase_sm70<12, true>(
            code_ptr, scale_ptr, packed_ptr, output_ptr, static_cast<int>(n),
            static_cast<int>(k), static_cast<int>(m), stream);
      } else {
        launch_fp8_qpn8_m32_twophase_sm70<16, true>(
            code_ptr, scale_ptr, packed_ptr, output_ptr, static_cast<int>(n),
            static_cast<int>(k), static_cast<int>(m), stream);
      }
    } else if (split_k == 12) {
      launch_fp8_qpn8_m32_twophase_sm70<12>(
          code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n),
          static_cast<int>(k), static_cast<int>(m), stream);
    } else {
      launch_fp8_qpn8_m32_twophase_sm70<16>(
          code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n),
          static_cast<int>(k), static_cast<int>(m), stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  fp8_qpn8_gemm_sm70_out(out, input, codes, group_scales, split_k,
                         accumulator_chains, fast_decoder, prefetch_codes);
}

// D2 (SP-43): native block-scale [K/128, N_total/32] consumption for M=9..32
// on the M32-class two-phase kernel. M=17..32 dispatches RowTiles=4, M=9..16
// RowTiles=2; anything else is rejected. The caller supplies QPN8-packed codes
// and block scales (as produced by fp8_qpn8_prepare_sm70) and either a
// scattered [M, K] or pre-packed [K/16, M, 16] fp16 activation.
void fp8_qpn8_block_native_sm70_out(torch::Tensor out, torch::Tensor input,
                                    torch::Tensor codes,
                                    torch::Tensor group_scales, int64_t split_k,
                                    int64_t k_unroll, bool packed_input,
                                    bool gated) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() && codes.is_cuda() &&
                  group_scales.is_cuda(),
              "fp8_qpn8_block_native_sm70_out: tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "fp8_qpn8_block_native_sm70_out: input and output must be "
              "float16");
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8,
              "fp8_qpn8_block_native_sm70_out: codes must be uint8");
  TORCH_CHECK(group_scales.scalar_type() == torch::kFloat16,
              "fp8_qpn8_block_native_sm70_out: group scales must be float16");
  TORCH_CHECK(out.dim() == 2 && group_scales.dim() == 2,
              "fp8_qpn8_block_native_sm70_out: expected 2D out/scales");
  TORCH_CHECK(input.dim() == (packed_input ? 3 : 2),
              "fp8_qpn8_block_native_sm70_out: input must be [M, K] or packed "
              "[K/16, M, 16]");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous() &&
                  codes.is_contiguous() && group_scales.is_contiguous(),
              "fp8_qpn8_block_native_sm70_out: tensors must be contiguous");
  TORCH_CHECK(input.get_device() == out.get_device() &&
                  input.get_device() == codes.get_device() &&
                  input.get_device() == group_scales.get_device(),
              "fp8_qpn8_block_native_sm70_out: tensors must share one device");

  const int64_t m = packed_input ? input.size(1) : input.size(0);
  const int64_t k =
      packed_input ? input.size(0) * input.size(2) : input.size(1);
  const int64_t n = out.size(1);
  const int64_t n_total = gated ? 2 * n : n;
  TORCH_CHECK(m >= 9 && m <= 40,
              "fp8_qpn8_block_native_sm70_out: M must be in [9, 40], got ", m);
  TORCH_CHECK(out.size(0) == m,
              "fp8_qpn8_block_native_sm70_out: output M mismatch");
  TORCH_CHECK(n > 0 && n % 32 == 0,
              "fp8_qpn8_block_native_sm70_out: N must be a positive multiple "
              "of 32");
  TORCH_CHECK(k > 0 && k % 128 == 0,
              "fp8_qpn8_block_native_sm70_out: K must be a positive multiple "
              "of 128");
  TORCH_CHECK(
      !packed_input || (input.size(0) == k / 16 && input.size(2) == 16),
      "fp8_qpn8_block_native_sm70_out: packed input must be [K/16, M, 16]");
  TORCH_CHECK(
      group_scales.size(0) == k / 128 && group_scales.size(1) == n_total / 32,
      "fp8_qpn8_block_native_sm70_out: expected block scales "
      "[K/128, N_total/32]");
  TORCH_CHECK(codes.numel() == n_total * k,
              "fp8_qpn8_block_native_sm70_out: code size mismatch");
  TORCH_CHECK(split_k == 8 || split_k == 12 || split_k == 16,
              "fp8_qpn8_block_native_sm70_out: split_k must be 8, 12, or 16");
  TORCH_CHECK((k / 16) % split_k == 0,
              "fp8_qpn8_block_native_sm70_out: invalid split_k for K");
  TORCH_CHECK(k_unroll == 0 || k_unroll == 1 || k_unroll == 2 || k_unroll == 4,
              "fp8_qpn8_block_native_sm70_out: k_unroll must be 0 (auto), 1, "
              "2, or 4");
  const bool auto_pack =
      !packed_input &&
      (gated || (k_unroll == 0 && (split_k == 12 || split_k == 16)));
  const bool effective_packed = packed_input || auto_pack;
  const int unroll =
      k_unroll == 0 ? (effective_packed ? 1 : 4) : static_cast<int>(k_unroll);
  TORCH_CHECK(effective_packed || unroll != 1,
              "fp8_qpn8_block_native_sm70_out: k_unroll 1 requires packed "
              "input");
  TORCH_CHECK(!gated || split_k == 8,
              "fp8_qpn8_block_native_sm70_out: gated route is split-8");
  TORCH_CHECK(!gated || unroll == 1,
              "fp8_qpn8_block_native_sm70_out: gated route uses the packed "
              "unroll-1 variant");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* code_ptr = codes.data_ptr<uint8_t>();
  const auto* scale_ptr =
      reinterpret_cast<const half*>(group_scales.data_ptr<at::Half>());
  torch::Tensor packed_holder, twin;
  const half* input_ptr =
      auto_pack ? vllm::sm70::k16_packed_input(input, packed_holder, stream)
                : reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  half* k16_output = gated ? vllm::sm70::k16_twin_for(out, twin) : nullptr;

  TORCH_CHECK(!effective_packed || unroll == 1,
              "fp8_qpn8_block_native_sm70_out: packed route uses unroll 1");
  const int row_tiles = m > 16 ? 4 : 2;
#define VLLM_QPN8_BLOCK_NATIVE_CALL(SPLIT, UROLL, RT, PK, GT)          \
  launch_fp8_qpn8_m32_twophase_sm70<SPLIT, PK, GT, true, RT, UROLL>(   \
      code_ptr, scale_ptr, input_ptr, output_ptr, static_cast<int>(n), \
      static_cast<int>(k), static_cast<int>(m), stream, k16_output)

  if (gated) {
    // Frozen GDN gate/up route: split-8, packed, two projections per block.
    if (row_tiles == 4) {
      VLLM_QPN8_BLOCK_NATIVE_CALL(8, 1, 4, true, true);
    } else {
      VLLM_QPN8_BLOCK_NATIVE_CALL(8, 1, 2, true, true);
    }
  } else if (effective_packed) {
    TORCH_CHECK(split_k != 8,
                "fp8_qpn8_block_native_sm70_out: non-gated packed route "
                "supports split_k 12 or 16");
    if (row_tiles == 4) {
      if (split_k == 12) {
        VLLM_QPN8_BLOCK_NATIVE_CALL(12, 1, 4, true, false);
      } else {
        VLLM_QPN8_BLOCK_NATIVE_CALL(16, 1, 4, true, false);
      }
    } else {
      if (split_k == 12) {
        VLLM_QPN8_BLOCK_NATIVE_CALL(12, 1, 2, true, false);
      } else {
        VLLM_QPN8_BLOCK_NATIVE_CALL(16, 1, 2, true, false);
      }
    }
  } else {
    TORCH_CHECK(split_k == 12 || split_k == 16,
                "fp8_qpn8_block_native_sm70_out: non-gated scattered route "
                "supports split_k 12 or 16");
    if (row_tiles == 4) {
      if (unroll == 2) {
        if (split_k == 12) {
          VLLM_QPN8_BLOCK_NATIVE_CALL(12, 2, 4, false, false);
        } else {
          VLLM_QPN8_BLOCK_NATIVE_CALL(16, 2, 4, false, false);
        }
      } else {
        if (split_k == 12) {
          VLLM_QPN8_BLOCK_NATIVE_CALL(12, 4, 4, false, false);
        } else {
          VLLM_QPN8_BLOCK_NATIVE_CALL(16, 4, 4, false, false);
        }
      }
    } else {
      if (unroll == 2) {
        if (split_k == 12) {
          VLLM_QPN8_BLOCK_NATIVE_CALL(12, 2, 2, false, false);
        } else {
          VLLM_QPN8_BLOCK_NATIVE_CALL(16, 2, 2, false, false);
        }
      } else {
        if (split_k == 12) {
          VLLM_QPN8_BLOCK_NATIVE_CALL(12, 4, 2, false, false);
        } else {
          VLLM_QPN8_BLOCK_NATIVE_CALL(16, 4, 2, false, false);
        }
      }
    }
  }
#undef VLLM_QPN8_BLOCK_NATIVE_CALL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (k16_output != nullptr) vllm::sm70::k16_publish(out, twin);
}

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "fp8_qpn8_gemm_v154_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor group_scales, int split_k, int accumulator_chains, "
      "bool fast_decoder, bool prefetch_codes) -> ()");
  m.def(
      "fp8_qpn8_block_native_sm70_out(Tensor(a!) out, Tensor input, Tensor "
      "codes, "
      "Tensor group_scales, int split_k, int k_unroll=0, bool "
      "packed_input=False, "
      "bool gated=False) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("fp8_qpn8_gemm_v154_sm70_out", &fp8_qpn8_gemm_v154_sm70_out);
  m.impl("fp8_qpn8_block_native_sm70_out", &fp8_qpn8_block_native_sm70_out);
}
