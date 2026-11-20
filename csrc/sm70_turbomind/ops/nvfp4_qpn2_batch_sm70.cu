// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Cooperative QPN2 batch kernels from 1Cat v1.5.4 (#1518). The entry point
// accepts the existing unbundled QPN2 layouts and uses no native policy.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include <torch/library.h>

#include "activation_pack_sm70.cuh"
#include "nvfp4_qpn2_layout.cuh"
#include "qpn_pair_sm70.cuh"

namespace {

__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

__device__ __forceinline__ half2 nvfp4_effective_scale(uint8_t value,
                                                       float global_scale) {
  // Match TurboMind's W4A16 weights: multiply the exact E4M3 group scale
  // by the FP32 global scale before rounding once to FP16. Rounding the
  // global factor first changes the model's weights on every decode step.
  const half raw = __low2half(fp8e4m3_to_half2(value));
  const half scaled =
      __float2half_rn(__fmul_rn(__half2float(raw), global_scale));
  return __halves2half2(scaled, scaled);
}

template <bool RebiasedScale = false>
__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const half2 raw = *reinterpret_cast<half2*>(&values[index]);
    if constexpr (RebiasedScale) {
      // The admitted scale range makes this power-of-two rescaling exact,
      // including FP16 subnormals. Both paths round the final product once.
      output[index] = __hmul2(raw, __hmul2(scale, __float2half2_rn(16384.0f)));
    } else {
      const half2 code = __hmul2(raw, __float2half2_rn(16384.0f));
      output[index] = __hmul2(code, scale);
    }
  }
}

// Batch kernels share the native/TurboMind reader without a second layout.
template <bool TurboMindLayout, bool RebiasedScale = false>
struct Nvfp4PairReader {
  [[maybe_unused]] static constexpr bool kFp4 = true;
  Nvfp4Qpn2CodeReader<TurboMindLayout> codes;
  const uint8_t* scales;
  float global_scale;

  __device__ Nvfp4PairReader(const uint8_t* w, const void* s, int tile,
                             int groups, int lane, float scale)
      : codes(w, tile, groups, lane),
        scales(static_cast<const uint8_t*>(s) +
               static_cast<size_t>(tile) * groups * 32 + lane),
        global_scale(scale) {}

  __device__ __forceinline__ void load(int group, half2* weights) const {
    const uint2 q = codes.load(group);
    const half2 scale = nvfp4_effective_scale(
        __ldg(scales + static_cast<size_t>(group) * 32), global_scale);
    dequant_e2m1x8<RebiasedScale>(q.x, scale, weights);
    dequant_e2m1x8<RebiasedScale>(q.y, scale, weights + 4);
  }
};

#define VLLM_SM70_QPN2_MMA(C, A0, A1, B0, B1)                       \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// Single-phase zero-spill cooperative batch GEMM for M=9..64 (16- or 32-col).
template <int Split, int NAcc, int RowTiles, bool TurboMindLayout,
          int TileColumns, int MinBlocks, bool RebiasedScale>
__global__
__launch_bounds__(Split* TileColumns, MinBlocks) void nvfp4_qpn2_coop_batch_sm70_kernel(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const half* __restrict__ input, half* __restrict__ output, int n, int k,
    int m, float global_scale) {
  static_assert(RowTiles >= 2 && RowTiles <= 5);
  static_assert(TileColumns == 16 || TileColumns == 32);
  constexpr int QuadsPerTile = TileColumns / 8;
  __shared__ float partial[Split][8 * TileColumns];

  const int lane = threadIdx.x & 31;
  const int qp = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int warp = (threadIdx.x >> 5) * (32 / TileColumns) + qp / QuadsPerTile;
  const int row_base = blockIdx.x * RowTiles * 8;
  const int output_tile = blockIdx.y;
  const int tile = output_tile * TileColumns / 32;
  const int mapped_lane =
      (lane & 19) |
      ((qp % QuadsPerTile + (output_tile % (32 / TileColumns)) * QuadsPerTile)
       << 2);
  const int groups = k >> 4;
  const int per_warp = groups / Split;
  const int begin = warp * per_warp;
  const int end = begin + per_warp;

  const Nvfp4PairReader<TurboMindLayout, RebiasedScale> reader(
      codes, scales, tile, groups, mapped_lane, global_scale);

  float accum[RowTiles][NAcc][8] = {};
#pragma unroll 1
  for (int group = begin; group < end; ++group) {
    half2 weights[8];
    reader.load(group, weights);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
    const half* a_group = input + static_cast<size_t>(group) * m * 16;
#pragma unroll
    for (int tr = 0; tr < RowTiles; ++tr) {
      const int row = row_base + tr * 8 + local_row;
      const int safe_row = row < m ? row : 0;
      const half* a = a_group + safe_row * 16;
      const uint4 a01 = __ldg(reinterpret_cast<const uint4*>(a));
      const uint4 a23 = __ldg(reinterpret_cast<const uint4*>(a + 8));
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&a01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&a23);
      VLLM_SM70_QPN2_MMA(accum[tr][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_QPN2_MMA(accum[tr][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_QPN2_MMA(accum[tr][0], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_QPN2_MMA(accum[tr][1 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int tr = 0; tr < RowTiles; ++tr) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (NAcc == 2) accum[tr][0][i] += accum[tr][1][i];
      const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partial[warp][r * TileColumns + (qp % QuadsPerTile) * 8 + c] =
          accum[tr][0][i];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 8 * TileColumns; e += blockDim.x) {
      const int row = row_base + tr * 8 + e / TileColumns;
      if (row < m) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < Split; ++w) sum += partial[w][e];
        output[static_cast<size_t>(row) * n + output_tile * TileColumns +
               e % TileColumns] = __float2half(sum);
      }
    }
    if (tr + 1 < RowTiles) __syncthreads();
  }
}

// Paired two-tile (64-column GEMM or 32+32 gated-SiLU) zero-spill batch kernel.
// Issues two independent 256-byte weight loads per warp to saturate HBM2 MLP
// while sharing K16 activation broadcasts from L1 across both tiles.
template <int Split, int RowTiles, bool Gated, bool TurboMindLayout,
          int MinBlocks, bool RebiasedScale>
__global__
__launch_bounds__(Split * 32, MinBlocks) void nvfp4_qpn2_pair_batch_sm70_kernel(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const half* __restrict__ input, half* __restrict__ output,
    half* __restrict__ k16_output, int width, int k, int m,
    float global_scale) {
  static_assert(RowTiles >= 2 && RowTiles <= 5);
  __shared__ float partial[2][Split][256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int qp = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.x * RowTiles * 8;
  const int out_pair = blockIdx.y;
  const int total_tiles = width / 32;
  const int tile0 = Gated ? out_pair : 2 * out_pair;
  const int tile1 = Gated ? out_pair + total_tiles : 2 * out_pair + 1;
  const bool has_tile1 = Gated || (tile1 < total_tiles);
  const int safe_tile1 = has_tile1 ? tile1 : tile0;
  const int groups = k >> 4;
  const int per_warp = groups / Split;
  const int begin = warp * per_warp;
  const int end = begin + per_warp;

  const Nvfp4PairReader<TurboMindLayout, RebiasedScale> r0(
      codes, scales, tile0, groups, lane, global_scale);
  const Nvfp4PairReader<TurboMindLayout, RebiasedScale> r1(
      codes, scales, safe_tile1, groups, lane, global_scale);

  float accum[2][RowTiles][8] = {};
#pragma unroll 1
  for (int group = begin; group < end; ++group) {
    half2 w0[8], w1[8];
    r0.load(group, w0);
    r1.load(group, w1);
    const unsigned* b0 = reinterpret_cast<const unsigned*>(w0);
    const unsigned* b1 = reinterpret_cast<const unsigned*>(w1);
    const half* a_group = input + static_cast<size_t>(group) * m * 16;
#pragma unroll
    for (int tr = 0; tr < RowTiles; ++tr) {
      const int row = row_base + tr * 8 + local_row;
      const int safe_row = row < m ? row : 0;
      const half* a = a_group + safe_row * 16;
      const uint4 a01 = __ldg(reinterpret_cast<const uint4*>(a));
      const uint4 a23 = __ldg(reinterpret_cast<const uint4*>(a + 8));
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&a01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&a23);
      VLLM_SM70_QPN2_MMA(accum[0][tr], a0[0], a0[1], b0[0], b0[1]);
      VLLM_SM70_QPN2_MMA(accum[1][tr], a0[0], a0[1], b1[0], b1[1]);
      VLLM_SM70_QPN2_MMA(accum[0][tr], a0[2], a0[3], b0[2], b0[3]);
      VLLM_SM70_QPN2_MMA(accum[1][tr], a0[2], a0[3], b1[2], b1[3]);
      VLLM_SM70_QPN2_MMA(accum[0][tr], a1[0], a1[1], b0[4], b0[5]);
      VLLM_SM70_QPN2_MMA(accum[1][tr], a1[0], a1[1], b1[4], b1[5]);
      VLLM_SM70_QPN2_MMA(accum[0][tr], a1[2], a1[3], b0[6], b0[7]);
      VLLM_SM70_QPN2_MMA(accum[1][tr], a1[2], a1[3], b1[6], b1[7]);
    }
  }

#pragma unroll
  for (int tr = 0; tr < RowTiles; ++tr) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partial[p][warp][r * 32 + qp * 8 + c] = accum[p][tr][i];
      }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += blockDim.x) {
      const int row = row_base + tr * 8 + e / 32;
      if (row < m) {
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int w = 0; w < Split; ++w) {
          s0 += partial[0][w][e];
          s1 += partial[1][w][e];
        }
        if constexpr (Gated) {
          const half gh = __float2half(s0);
          const half uh = __float2half(s1);
          const float gf = __half2float(gh);
          const half result = __hmul(__float2half(gf / (1.0f + expf(-gf))), uh);
          const int col = tile0 * 32 + e % 32;
          output[static_cast<size_t>(row) * width + col] = result;
          if (k16_output != nullptr) {
            k16_output[vllm::sm70::k16_half_index(row, col, m)] = result;
          }
        } else {
          output[static_cast<size_t>(row) * width + tile0 * 32 + e % 32] =
              __float2half(s0);
          if (has_tile1) {
            output[static_cast<size_t>(row) * width + tile1 * 32 + e % 32] =
                __float2half(s1);
          }
        }
      }
    }
    if (tr + 1 < RowTiles) __syncthreads();
  }
}

// Narrow 16-column paired gated fallback when hidden / 32 < SM count.
template <int Split, int RowTiles, bool TurboMindLayout, int MinBlocks,
          bool RebiasedScale>
__global__
__launch_bounds__(Split * 16, MinBlocks) void nvfp4_qpn2_paired_gated_narrow_sm70_kernel(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const half* __restrict__ input, half* __restrict__ output,
    half* __restrict__ k16_output, int hidden, int k, int m,
    float global_scale) {
  static_assert(RowTiles >= 3 && RowTiles <= 5);
  __shared__ float partial[2][Split][128];

  const int lane = threadIdx.x & 31;
  const int qp = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int warp = (threadIdx.x >> 5) * 2 + qp / 2;
  const int row_base = blockIdx.x * RowTiles * 8;
  const int output_tile = blockIdx.y;
  const int tile0 = output_tile / 2;
  const int tile1 = tile0 + hidden / 32;
  const int mapped_lane =
      (lane & 19) | (((qp % 2) + (output_tile % 2) * 2) << 2);
  const int groups = k >> 4;
  const int per_warp = groups / Split;
  const int begin = warp * per_warp;
  const int end = begin + per_warp;

  const Nvfp4PairReader<TurboMindLayout, RebiasedScale> r0(
      codes, scales, tile0, groups, mapped_lane, global_scale);
  const Nvfp4PairReader<TurboMindLayout, RebiasedScale> r1(
      codes, scales, tile1, groups, mapped_lane, global_scale);

  float accum[2][RowTiles][8] = {};
#pragma unroll 1
  for (int group = begin; group < end; ++group) {
    half2 w0[8], w1[8];
    r0.load(group, w0);
    r1.load(group, w1);
    const unsigned* b0 = reinterpret_cast<const unsigned*>(w0);
    const unsigned* b1 = reinterpret_cast<const unsigned*>(w1);
    const half* a_group = input + static_cast<size_t>(group) * m * 16;
#pragma unroll
    for (int tr = 0; tr < RowTiles; ++tr) {
      const int row = row_base + tr * 8 + local_row;
      const int safe_row = row < m ? row : 0;
      const half* a = a_group + safe_row * 16;
      const uint4 a01 = __ldg(reinterpret_cast<const uint4*>(a));
      const uint4 a23 = __ldg(reinterpret_cast<const uint4*>(a + 8));
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&a01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&a23);
      VLLM_SM70_QPN2_MMA(accum[0][tr], a0[0], a0[1], b0[0], b0[1]);
      VLLM_SM70_QPN2_MMA(accum[1][tr], a0[0], a0[1], b1[0], b1[1]);
      VLLM_SM70_QPN2_MMA(accum[0][tr], a0[2], a0[3], b0[2], b0[3]);
      VLLM_SM70_QPN2_MMA(accum[1][tr], a0[2], a0[3], b1[2], b1[3]);
      VLLM_SM70_QPN2_MMA(accum[0][tr], a1[0], a1[1], b0[4], b0[5]);
      VLLM_SM70_QPN2_MMA(accum[1][tr], a1[0], a1[1], b1[4], b1[5]);
      VLLM_SM70_QPN2_MMA(accum[0][tr], a1[2], a1[3], b0[6], b0[7]);
      VLLM_SM70_QPN2_MMA(accum[1][tr], a1[2], a1[3], b1[6], b1[7]);
    }
  }

#pragma unroll
  for (int tr = 0; tr < RowTiles; ++tr) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partial[p][warp][r * 16 + (qp % 2) * 8 + c] = accum[p][tr][i];
      }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 128; e += blockDim.x) {
      const int row = row_base + tr * 8 + e / 16;
      if (row < m) {
        float gate = 0.0f, up = 0.0f;
#pragma unroll
        for (int w = 0; w < Split; ++w) {
          gate += partial[0][w][e];
          up += partial[1][w][e];
        }
        const half gh = __float2half(gate);
        const half uh = __float2half(up);
        const float gf = __half2float(gh);
        const half result = __hmul(__float2half(gf / (1.0f + expf(-gf))), uh);
        const int col = output_tile * 16 + e % 16;
        output[static_cast<size_t>(row) * hidden + col] = result;
        if (k16_output != nullptr) {
          k16_output[vllm::sm70::k16_half_index(row, col, m)] = result;
        }
      }
    }
    if (tr + 1 < RowTiles) __syncthreads();
  }
}

template <int RowTiles, bool TurboMindLayout>
void launch_qpn2_coop_batch(const uint8_t* codes, const uint8_t* scales,
                            const half* packed_input, half* output, int n,
                            int k, int m, float global_scale,
                            cudaStream_t stream) {
  const int row_blocks = (m + RowTiles * 8 - 1) / (RowTiles * 8);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const bool rebias =
      global_scale > 0.0f && global_scale <= 65504.0f / (16384.0f * 480.0f);
  const int pairs = (n + 63) / 64;
  const bool can_split16 = ((k >> 4) % 16 == 0);
  if (pairs * row_blocks >= (sms * 3) / 4) {
    const dim3 grid(row_blocks, pairs);
    if constexpr (RowTiles == 5) {
      // RowTiles == 5 holds 80 FP32 accumulators (~140 regs); 256 threads with
      // MinBlocks = 1 avoids all register spills while keeping 2x weight MLP.
      if (rebias) {
        nvfp4_qpn2_pair_batch_sm70_kernel<8, RowTiles, false, TurboMindLayout,
                                          1, true>
            <<<grid, 256, 0, stream>>>(codes, scales, packed_input, output,
                                       nullptr, n, k, m, global_scale);
      } else {
        nvfp4_qpn2_pair_batch_sm70_kernel<8, RowTiles, false, TurboMindLayout,
                                          1, false>
            <<<grid, 256, 0, stream>>>(codes, scales, packed_input, output,
                                       nullptr, n, k, m, global_scale);
      }
    } else if (can_split16) {
      if (rebias) {
        nvfp4_qpn2_pair_batch_sm70_kernel<16, RowTiles, false, TurboMindLayout,
                                          1, true>
            <<<grid, 512, 0, stream>>>(codes, scales, packed_input, output,
                                       nullptr, n, k, m, global_scale);
      } else {
        nvfp4_qpn2_pair_batch_sm70_kernel<16, RowTiles, false, TurboMindLayout,
                                          1, false>
            <<<grid, 512, 0, stream>>>(codes, scales, packed_input, output,
                                       nullptr, n, k, m, global_scale);
      }
    } else {
      if (rebias) {
        nvfp4_qpn2_pair_batch_sm70_kernel<8, RowTiles, false, TurboMindLayout,
                                          2, true>
            <<<grid, 256, 0, stream>>>(codes, scales, packed_input, output,
                                       nullptr, n, k, m, global_scale);
      } else {
        nvfp4_qpn2_pair_batch_sm70_kernel<8, RowTiles, false, TurboMindLayout,
                                          2, false>
            <<<grid, 256, 0, stream>>>(codes, scales, packed_input, output,
                                       nullptr, n, k, m, global_scale);
      }
    }
  } else if ((n / 32) * row_blocks >= sms) {
    const dim3 grid(row_blocks, n / 32);
    if (can_split16) {
      if (rebias) {
        nvfp4_qpn2_coop_batch_sm70_kernel<16, 1, RowTiles, TurboMindLayout, 32,
                                          1, true><<<grid, 512, 0, stream>>>(
            codes, scales, packed_input, output, n, k, m, global_scale);
      } else {
        nvfp4_qpn2_coop_batch_sm70_kernel<16, 1, RowTiles, TurboMindLayout, 32,
                                          1, false><<<grid, 512, 0, stream>>>(
            codes, scales, packed_input, output, n, k, m, global_scale);
      }
    } else {
      if (rebias) {
        nvfp4_qpn2_coop_batch_sm70_kernel<8, 1, RowTiles, TurboMindLayout, 32,
                                          2, true><<<grid, 256, 0, stream>>>(
            codes, scales, packed_input, output, n, k, m, global_scale);
      } else {
        nvfp4_qpn2_coop_batch_sm70_kernel<8, 1, RowTiles, TurboMindLayout, 32,
                                          2, false><<<grid, 256, 0, stream>>>(
            codes, scales, packed_input, output, n, k, m, global_scale);
      }
    }
  } else {
    constexpr int MinBlocks = (RowTiles == 5) ? 3 : 4;
    const dim3 grid(row_blocks, n / 16);
    if (rebias) {
      nvfp4_qpn2_coop_batch_sm70_kernel<8, 2, RowTiles, TurboMindLayout, 16,
                                        MinBlocks, true>
          <<<grid, 128, 0, stream>>>(codes, scales, packed_input, output, n, k,
                                     m, global_scale);
    } else {
      nvfp4_qpn2_coop_batch_sm70_kernel<8, 2, RowTiles, TurboMindLayout, 16,
                                        MinBlocks, false>
          <<<grid, 128, 0, stream>>>(codes, scales, packed_input, output, n, k,
                                     m, global_scale);
    }
  }
}

template <int RowTiles, bool TurboMindLayout>
void launch_qpn2_paired_gated_batch(const uint8_t* codes, const uint8_t* scales,
                                    const half* packed_input, half* output,
                                    half* k16_output, int hidden, int k, int m,
                                    float global_scale, cudaStream_t stream) {
  const int row_blocks = (m + RowTiles * 8 - 1) / (RowTiles * 8);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const bool rebias =
      global_scale > 0.0f && global_scale <= 65504.0f / (16384.0f * 480.0f);
  if ((hidden / 32) * row_blocks < sms) {
    constexpr int MinBlocks = (RowTiles <= 4) ? 4 : 3;
    const dim3 grid(row_blocks, hidden / 16);
    if (rebias) {
      nvfp4_qpn2_paired_gated_narrow_sm70_kernel<8, RowTiles, TurboMindLayout,
                                                 MinBlocks, true>
          <<<grid, 128, 0, stream>>>(codes, scales, packed_input, output,
                                     k16_output, hidden, k, m, global_scale);
    } else {
      nvfp4_qpn2_paired_gated_narrow_sm70_kernel<8, RowTiles, TurboMindLayout,
                                                 MinBlocks, false>
          <<<grid, 128, 0, stream>>>(codes, scales, packed_input, output,
                                     k16_output, hidden, k, m, global_scale);
    }
  } else {
    constexpr int MinBlocks = (RowTiles == 5) ? 1 : 2;
    const dim3 grid(row_blocks, hidden / 32);
    if (rebias) {
      nvfp4_qpn2_pair_batch_sm70_kernel<8, RowTiles, true, TurboMindLayout,
                                        MinBlocks, true>
          <<<grid, 256, 0, stream>>>(codes, scales, packed_input, output,
                                     k16_output, hidden, k, m, global_scale);
    } else {
      nvfp4_qpn2_pair_batch_sm70_kernel<8, RowTiles, true, TurboMindLayout,
                                        MinBlocks, false>
          <<<grid, 256, 0, stream>>>(codes, scales, packed_input, output,
                                     k16_output, hidden, k, m, global_scale);
    }
  }
}

template <int TileRows, bool TurboMindLayout>
void launch_batch(const uint8_t* codes, const uint8_t* scales,
                  const half* input, half* out, half* twin, int n, int k, int m,
                  float global_scale, bool gated, cudaStream_t stream) {
  if (gated) {
    launch_qpn2_paired_gated_batch<TileRows, TurboMindLayout>(
        codes, scales, input, out, twin, n, k, m, global_scale, stream);
  } else {
    launch_qpn2_coop_batch<TileRows, TurboMindLayout>(
        codes, scales, input, out, n, k, m, global_scale, stream);
  }
}

template <bool TurboMindLayout>
void run_batch(torch::Tensor out, torch::Tensor input, torch::Tensor codes,
               torch::Tensor scales, float global_scale, bool gated,
               cudaStream_t stream) {
  const int m = input.size(0), k = input.size(1), n = out.size(1);
  if (m <= 16 && (gated || n % 64 == 0)) {
    vllm::sm70::k16_clear();
    if (gated) {
      vllm::sm70::launch_qpn_pair_m16<Nvfp4PairReader<TurboMindLayout>, 8,
                                      true>(out, input, codes, scales,
                                            global_scale, stream);
    } else {
      vllm::sm70::launch_qpn_pair_m16<Nvfp4PairReader<TurboMindLayout>, 8,
                                      false>(out, input, codes, scales,
                                             global_scale, stream);
    }
    return;
  }

  torch::Tensor packed, twin;
  const auto* x = vllm::sm70::k16_packed_input(input, packed, stream);
  const auto* w = codes.data_ptr<uint8_t>();
  const auto* s = scales.data_ptr<uint8_t>();
  auto* y = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  half* k16_out = gated ? vllm::sm70::k16_twin_for(out, twin) : nullptr;
  if (m <= 16) {
    launch_batch<2, TurboMindLayout>(w, s, x, y, k16_out, n, k, m, global_scale,
                                     gated, stream);
  } else if (m <= 24 || (m > 40 && m <= 48)) {
    launch_batch<3, TurboMindLayout>(w, s, x, y, k16_out, n, k, m, global_scale,
                                     gated, stream);
  } else if (m <= 32 || m > 48) {
    launch_batch<4, TurboMindLayout>(w, s, x, y, k16_out, n, k, m, global_scale,
                                     gated, stream);
  } else {
    launch_batch<5, TurboMindLayout>(w, s, x, y, k16_out, n, k, m, global_scale,
                                     gated, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (k16_out) vllm::sm70::k16_publish(out, twin);
}

void nvfp4_qpn2_batch_sm70_out(torch::Tensor out, torch::Tensor input,
                               torch::Tensor codes, torch::Tensor scales,
                               double global_scale, bool gated,
                               bool turbomind_layout) {
  TORCH_CHECK(
      input.is_cuda() && input.dim() == 2 && out.dim() == 2 &&
          input.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf &&
          codes.scalar_type() == at::kByte && scales.scalar_type() == at::kByte,
      "Expected CUDA FP16 matrices and unbundled uint8 weights/scales");
  for (const auto& tensor : {out, input, codes, scales})
    TORCH_CHECK(tensor.device() == input.device() && tensor.is_contiguous(),
                "QPN2 tensors must be contiguous and on the same device");

  const int64_t m = input.size(0), k = input.size(1), n = out.size(1);
  const int64_t weight_n = n * (gated ? 2 : 1);
  TORCH_CHECK(m >= 9 && m <= 64 && out.size(0) == m && k >= 512 &&
                  k % 128 == 0 && n >= 1024 && n % 32 == 0,
              "QPN2 batch requires M9..64, K >= 512 and N >= 1024");
  TORCH_CHECK(
      codes.numel() == weight_n * k / 2 && scales.numel() == weight_n * k / 16,
      "QPN2 packed tensor size mismatch");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "QPN2 batch requires SM70");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (turbomind_layout) {
    run_batch<true>(out, input, codes, scales, global_scale, gated, stream);
  } else {
    run_batch<false>(out, input, codes, scales, global_scale, gated, stream);
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "nvfp4_qpn2_batch_sm70_out(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, bool gated, "
      "bool turbomind_layout=False) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("nvfp4_qpn2_batch_sm70_out", &nvfp4_qpn2_batch_sm70_out);
}
