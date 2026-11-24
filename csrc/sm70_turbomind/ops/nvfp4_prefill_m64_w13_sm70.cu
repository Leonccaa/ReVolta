// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// M64 W13 from 1Cat's SM70 NVFP4 prefill v6, adapted to the interleaved
// gate/up scales and the existing ordered W2 routing.
// Source: 1CatAI/1Cat-vLLM, 85f162568be414db04629ddd1cb879b73ed40f23.

#include <algorithm>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {

// ============================================================================
// Common SM70 MMA, Conflict-Free Shared Memory & Fast Math Helpers
// ============================================================================

#define FN_MMA_884_RC(C, A0, A1, B0, B1)                            \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// Algebraically conflict-free 128-bit shared memory indexing (+1 uint4 per 4
// rows): Zero bank conflicts on A stores, B stores, A MMA multicast loads, and
// B MMA loads!
__device__ __forceinline__ int smem_idx17(int row, int slot4) {
  const int swz = slot4 ^ (row & 2) ^ ((row >> 3) & 1);
  return (row >> 2) * 17 + (row & 3) * 4 + swz;
}

__device__ __forceinline__ uint4 decode_e2m1x8_u4(unsigned packed,
                                                  half2 scale) {
  constexpr unsigned sign = 0x80008000u, em = 0x0e000e00u;
  const unsigned v0 = ((packed << 12) & sign) | ((packed << 9) & em);
  const unsigned v1 = ((packed << 8) & sign) | ((packed << 5) & em);
  const unsigned v2 = ((packed << 4) & sign) | ((packed << 1) & em);
  const unsigned v3 = (packed & sign) | ((packed >> 3) & em);
  const half2 h0 = __hmul2(*reinterpret_cast<const half2*>(&v0), scale);
  const half2 h1 = __hmul2(*reinterpret_cast<const half2*>(&v1), scale);
  const half2 h2 = __hmul2(*reinterpret_cast<const half2*>(&v2), scale);
  const half2 h3 = __hmul2(*reinterpret_cast<const half2*>(&v3), scale);
  return make_uint4(*reinterpret_cast<const unsigned*>(&h0),
                    *reinterpret_cast<const unsigned*>(&h1),
                    *reinterpret_cast<const unsigned*>(&h2),
                    *reinterpret_cast<const unsigned*>(&h3));
}

__device__ __forceinline__ void decode_e2m1x8(unsigned packed, half2 scale,
                                              half2* out) {
  *reinterpret_cast<uint4*>(out) = decode_e2m1x8_u4(packed, scale);
}

__device__ __forceinline__ half nvfp4_fast_hmma_scale(uint8_t scale_code,
                                                      float gs_x256) {
  const unsigned code = scale_code;
  const unsigned short bits = static_cast<unsigned short>(
      ((code & 0x80U) << 8U) | ((code & 0x7fU) << 7U));
  const half prepared =
      __float2half_rn(__half2float(__ushort_as_half(bits)) * gs_x256);
  return __hmul(prepared, __ushort_as_half(0x7400U));
}

// ============================================================================
// Part 1: Counting-Sort MoE Permute + Weighted Scatter + Tile Scheduler
// ============================================================================

constexpr int kMoEExperts = 512;
constexpr int kMoETopK = 10;

struct alignas(16) MoETileDesc {
  int32_t expert;
  int32_t start;
  int32_t count;
  int32_t pad;
};

__global__ __launch_bounds__(256) void moe_block_hist_kernel(
    const int32_t* __restrict__ topk_ids, int32_t* __restrict__ block_counts,
    int slots_per_blk, int total_slots) {
  __shared__ int32_t s_cnt[kMoEExperts];
  const int tx = threadIdx.x;
  s_cnt[tx] = 0;
  s_cnt[tx + 256] = 0;
  __syncthreads();

  const int start = blockIdx.x * slots_per_blk;
  const int end = min(start + slots_per_blk, total_slots);
  for (int idx = start + tx; idx < end; idx += 256) {
    atomicAdd(&s_cnt[topk_ids[idx]], 1);
  }
  __syncthreads();

  int32_t* dst = block_counts + static_cast<int>(blockIdx.x) * kMoEExperts;
  dst[tx] = s_cnt[tx];
  dst[tx + 256] = s_cnt[tx + 256];
}

// Convert block counts into local offsets and emit the W13 expert tiles.
__global__ __launch_bounds__(512) void moe_block_scan_schedule_kernel(
    int32_t* __restrict__ block_counts, int num_blks,
    int32_t* __restrict__ offsets, MoETileDesc* __restrict__ tiles,
    int32_t* __restrict__ num_tiles) {
  __shared__ int32_t s_warp_c[16];
  __shared__ int32_t s_warp_tc[16];

  const int tx = threadIdx.x;  // 0..511
  const int lane = tx & 31;
  const int warp = tx >> 5;

  int c = 0;
  for (int b = 0; b < num_blks; ++b) {
    int32_t* row = block_counts + b * kMoEExperts;
    const int cnt = row[tx];
    row[tx] = c;
    c += cnt;
  }

  int scan_c = c;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int nc = __shfl_up_sync(0xffffffffu, scan_c, d);
    if (lane >= d) scan_c += nc;
  }
  if (lane == 31) s_warp_c[warp] = scan_c;
  __syncthreads();
  if (warp == 0) {
    int wc = (lane < 16) ? s_warp_c[lane] : 0;
#pragma unroll
    for (int d = 1; d < 16; d <<= 1) {
      const int nc = __shfl_up_sync(0xffffffffu, wc, d);
      if (lane >= d) wc += nc;
    }
    if (lane < 16) s_warp_c[lane] = wc;
  }
  __syncthreads();

  const int excl_c = ((warp > 0) ? s_warp_c[warp - 1] : 0) + scan_c - c;
  offsets[tx] = excl_c;
  if (tx == kMoEExperts - 1) {
    offsets[kMoEExperts] = excl_c + c;
  }

  constexpr int cta_m = 64;
  const int tc = (c + cta_m - 1) / cta_m;
  int scan_tc = tc;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int ntc = __shfl_up_sync(0xffffffffu, scan_tc, d);
    if (lane >= d) scan_tc += ntc;
  }
  if (lane == 31) s_warp_tc[warp] = scan_tc;
  __syncthreads();
  if (warp == 0) {
    int wtc = (lane < 16) ? s_warp_tc[lane] : 0;
#pragma unroll
    for (int d = 1; d < 16; d <<= 1) {
      const int ntc = __shfl_up_sync(0xffffffffu, wtc, d);
      if (lane >= d) wtc += ntc;
    }
    if (lane < 16) s_warp_tc[lane] = wtc;
  }
  __syncthreads();

  const int excl_tc = ((warp > 0) ? s_warp_tc[warp - 1] : 0) + scan_tc - tc;
  if (tx == kMoEExperts - 1) *num_tiles = excl_tc + tc;
  for (int tile = 0; tile < tc; ++tile) {
    const int remaining = c - tile * cta_m;
    tiles[excl_tc + tile] =
        MoETileDesc{tx, excl_c + tile * cta_m, min(remaining, cta_m), 0};
  }
}

__global__ __launch_bounds__(256) void moe_block_scatter_kernel(
    const int32_t* __restrict__ topk_ids, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ block_counts,
    int32_t* __restrict__ sorted_token_idx, int32_t* __restrict__ inverse,
    int slots_per_blk, int total_slots) {
  __shared__ int32_t s_cnt[kMoEExperts];
  __shared__ int32_t s_base[kMoEExperts];
  const int tx = threadIdx.x;
  const int32_t* blk_row =
      block_counts + static_cast<int>(blockIdx.x) * kMoEExperts;
  s_cnt[tx] = 0;
  s_cnt[tx + 256] = 0;
  s_base[tx] = offsets[tx] + blk_row[tx];
  s_base[tx + 256] = offsets[tx + 256] + blk_row[tx + 256];
  __syncthreads();

  const int start = blockIdx.x * slots_per_blk;
  const int end = min(start + slots_per_blk, total_slots);
  for (int idx = start + tx; idx < end; idx += 256) {
    const int expert = topk_ids[idx];
    const int dst = s_base[expert] + atomicAdd(&s_cnt[expert], 1);
    sorted_token_idx[dst] = idx / kMoETopK;
    inverse[idx] = dst;
  }
}

#define FN_SWIGLU_ELEM_OUT(G_VAL, U_VAL, R_IDX, C_IDX)           \
  do {                                                           \
    const half gate_h = __float2half_rn(G_VAL);                  \
    const half up_h = __float2half_rn(U_VAL);                    \
    const float gf = __half2float(gate_h);                       \
    const half act_h = __float2half_rn(gf / (1.0f + expf(-gf))); \
    smem.s_out[R_IDX][C_IDX] = __hmul(act_h, up_h);              \
  } while (0)
#define FN_STAGE_SWIGLU_8_LIT(ACC, ROW_BASE, WN)                       \
  do {                                                                 \
    if ((ROW_BASE) < m_count) {                                        \
      const int r_b = (ROW_BASE) + (lane & 1) + ((lane & 16) ? 4 : 0); \
      const int c_b = (WN) * 16 + quad * 4 + ((lane >> 1) & 1);        \
      FN_SWIGLU_ELEM_OUT((ACC)[0], (ACC)[1], r_b, c_b);                \
      FN_SWIGLU_ELEM_OUT((ACC)[2], (ACC)[3], r_b + 2, c_b);            \
      FN_SWIGLU_ELEM_OUT((ACC)[4], (ACC)[5], r_b, c_b + 2);            \
      FN_SWIGLU_ELEM_OUT((ACC)[6], (ACC)[7], r_b + 2, c_b + 2);        \
    }                                                                  \
  } while (0)
__global__ __launch_bounds__(640, 1) void fn_nvfp4_w13_m64_n320_kernel(
    const half* __restrict__ x,               // [M, 2560]
    const uint32_t* __restrict__ weights,     // [512, 2560, 40]
    const uint8_t* __restrict__ scale_codes,  // [512, 160, 320]
    const float* __restrict__ global_scales,  // [512, 2]
    const int32_t* __restrict__ sorted_token_idx,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    half* __restrict__ out)  // [M * 10, 160]
{
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;

  constexpr int kHidden = 2560;
  constexpr int kRawN = 320;
  constexpr int kInter = 160;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHidden / kStepK;  // 80

  struct SharedW13M64N320 {
    union {
      struct {
        uint4 s_A[2][32 * 17];  // [2][64 rows, 64 halves] = 17,408 B (2
                                // K-stages/buf!)
        uint4 s_B[2][80 * 17];  // [2][320 cols, 32 halves] = 43,520 B
      } mainloop;               // 60,928 B + 256 B = 61,184 B <= 96 KiB!
      half s_out[64][168];
    };
    int32_t s_tok[64];
  };
  extern __shared__ __align__(16) unsigned char w13_n320_raw[];
  SharedW13M64N320& smem = *reinterpret_cast<SharedW13M64N320*>(w13_n320_raw);
  int32_t* s_tok = smem.s_tok;

  const int tx = threadIdx.x;    // 0..639
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..19
  const int warp_m = warp / 10;  // 0..1 -> 32 rows each = 64 rows
  const int warp_n = warp % 10;  // 0..9 -> 32 raw cols (16 SwiGLU cols) each

  if (tx < 64) {
    smem.s_tok[tx] = (tx < m_count) ? sorted_token_idx[m_start + tx] : -1;
  }
  __syncthreads();

  // All 512 threads (16 warps!) gather a full 128-byte cache line (64 halves =
  // 8 uint4s) per row of x ONCE every 2 stages (512+ HMMA cycles of HBM latency
  // hiding)!
  const int a_r0 = tx >> 3;      // 0..63 for tx < 512
  const int a_c8 = tx & 7;       // 0..7 -> 64 halves (128 bytes)!
  const int a_half = a_c8 >> 2;  // 0 for cols 0..31, 1 for cols 32..63
  const int a_s0 = a_c8 & 3;     // 0..3 within 32-half stage
  const int a_smem_idx = (tx < 512) ? smem_idx17(a_half * 64 + a_r0, a_s0) : 0;
  const half* x_ptr0 = nullptr;
  if (tx < 512) {
    const int tok0 = s_tok[a_r0];
    if (tok0 >= 0) {
      x_ptr0 = x + static_cast<size_t>(tok0) * kHidden + a_c8 * 8;
    }
  }

  const int sub_b = warp_m;  // 0..1
  const int col320_b = warp_n * 32 + lane;
  const int b_store_idx0 = smem_idx17(col320_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col320_b, sub_b * 2 + 1);
  const int t32_b = warp_n;  // 0..9

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kHidden * (kRawN / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kHidden / 16) * kRawN;
  // TurboMind's packed W13 alternates gate and up columns.
  const float gs_b_x256 =
      __ldg(global_scales + static_cast<size_t>(expert) * 2 + (col320_b & 1)) *
      256.0f;

  {
    const uint4 av0 = x_ptr0 ? __ldg(reinterpret_cast<const uint4*>(x_ptr0))
                             : make_uint4(0, 0, 0, 0);
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
    const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);

    if (tx < 512) {
      smem.mainloop.s_A[0][a_smem_idx] = av0;
    }
    const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
    const half2 scale2 = __halves2half2(s_h, s_h);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};

#pragma unroll 2
  for (int pair_st = 0; pair_st < (kNumStages / 2); ++pair_st) {
    const int cur_a = pair_st & 1;
    const int nxt_a = cur_a ^ 1;

    uint4 av_next = make_uint4(0, 0, 0, 0);
    if (pair_st + 1 < (kNumStages / 2) && x_ptr0) {
      av_next =
          __ldg(reinterpret_cast<const uint4*>(x_ptr0 + (pair_st + 1) * 64));
    }

#pragma unroll
    for (int sub_st = 0; sub_st < 2; ++sub_st) {
      const int st = pair_st * 2 + sub_st;
      const int cur_b = sub_st;
      const int nxt_b = sub_st ^ 1;
      const int a_row_off = sub_st * 64;

      uint32_t w0 = 0, w1 = 0;
      uint8_t sc = 0;
      if (st + 1 < kNumStages) {
        const int g = (st + 1) * 2 + sub_b;
        const size_t w_off =
            (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
        const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
        w0 = __ldg(exp_w + w_off);
        w1 = __ldg(exp_w + w_off + 32);
        sc = __ldg(exp_sc + s_idx);
      }

      if (m_base < m_count) {
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int sl0 = sub * 2;
          const int sl1 = sub * 2 + 1;
          const uint4 b0 = smem.mainloop.s_B[cur_b][smem_idx17(b_col, sl0)];
          const uint4 b1 = smem.mainloop.s_B[cur_b][smem_idx17(b_col, sl1)];
          uint4 a_lo =
              smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0, sl0)];
          uint4 a_hi =
              smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0, sl1)];
          FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 8, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 8, sl1)];
          FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 16, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 16, sl1)];
          FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 24, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 24, sl1)];
          FN_MMA_884_RC(acc3, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc3, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc3, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc3, a_hi.z, a_hi.w, b1.z, b1.w);
        }
      }

      if (sub_st == 1 && pair_st + 1 < (kNumStages / 2) && tx < 512) {
        smem.mainloop.s_A[nxt_a][a_smem_idx] = av_next;
      }
      if (st + 1 < kNumStages) {
        const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
        const half2 scale2 = __halves2half2(s_h, s_h);
        alignas(16) half2 dec0[4], dec1[4];
        decode_e2m1x8(w0, scale2, dec0);
        decode_e2m1x8(w1, scale2, dec1);
        smem.mainloop.s_B[nxt_b][b_store_idx0] =
            *reinterpret_cast<const uint4*>(dec0);
        smem.mainloop.s_B[nxt_b][b_store_idx1] =
            *reinterpret_cast<const uint4*>(dec1);
      }
      __syncthreads();
    }
  }

  FN_STAGE_SWIGLU_8_LIT(acc0, m_base + 0, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc1, m_base + 8, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc2, m_base + 16, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc3, m_base + 24, warp_n);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int vec_id = tx + i * 640;
    const int out_r = vec_id / 20;
    const int out_c8 = (vec_id % 20) * 8;
    if (out_r < m_count) {
      *reinterpret_cast<uint4*>(
          out + static_cast<size_t>(m_start + out_r) * kInter + out_c8) =
          *reinterpret_cast<const uint4*>(&smem.s_out[out_r][out_c8]);
    }
  }
}

void w13_prefill(torch::Tensor intermediate, torch::Tensor offsets,
                 torch::Tensor inverse, torch::Tensor sorted_tokens,
                 const torch::Tensor& x, const torch::Tensor& ids,
                 const torch::Tensor& weight, const torch::Tensor& scales,
                 const torch::Tensor& globals) {
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = x.size(0);
  const int slots = rows * kMoETopK;
  const int rows_per_block = std::max(256, ((rows + 63) / 64 + 31) & ~31);
  const int blocks = (rows + rows_per_block - 1) / rows_per_block;
  const int slots_per_block = rows_per_block * kMoETopK;
  const int max_tiles = (slots + 63) / 64 + kMoEExperts;
  auto counts = torch::empty({blocks, kMoEExperts}, ids.options());
  auto tiles = torch::empty({max_tiles, 4}, ids.options());
  auto num_tiles = torch::empty({1}, ids.options());
  auto* schedule = reinterpret_cast<MoETileDesc*>(tiles.data_ptr());
  constexpr int smem = 61184;
  static thread_local int configured_device = -1;
  if (configured_device != x.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m64_n320_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m64_n320_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured_device = x.get_device();
  }
  moe_block_hist_kernel<<<blocks, 256, 0, stream>>>(ids.data_ptr<int32_t>(),
                                                    counts.data_ptr<int32_t>(),
                                                    slots_per_block, slots);
  moe_block_scan_schedule_kernel<<<1, kMoEExperts, 0, stream>>>(
      counts.data_ptr<int32_t>(), blocks, offsets.data_ptr<int32_t>(), schedule,
      num_tiles.data_ptr<int32_t>());
  moe_block_scatter_kernel<<<blocks, 256, 0, stream>>>(
      ids.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(), sorted_tokens.data_ptr<int32_t>(),
      inverse.data_ptr<int32_t>(), slots_per_block, slots);
  fn_nvfp4_w13_m64_n320_kernel<<<max_tiles, 640, smem, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const uint32_t*>(weight.data_ptr()),
      scales.data_ptr<uint8_t>(), globals.data_ptr<float>(),
      sorted_tokens.data_ptr<int32_t>(), schedule,
      num_tiles.data_ptr<int32_t>(),
      reinterpret_cast<half*>(intermediate.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "nvfp4_prefill_m64_w13_sm70_out(Tensor(a!) intermediate, Tensor(b!) "
      "offsets, "
      "Tensor(c!) inverse, Tensor(d!) sorted_tokens, Tensor x, Tensor ids, "
      "Tensor weight, Tensor scales, Tensor globals) -> ()");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("nvfp4_prefill_m64_w13_sm70_out", &w13_prefill);
}
