// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// GDN prefill kernels from 1CatAI/1Cat-vLLM commit
// 85f162568be414db04629ddd1cb879b73ed40f23,
// csrc/sm70_turbomind/ops/flashnext_prefill_sm70.cu.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vllm_fn_sm70 {

#define FN_MMA_884_RC(C, A0, A1, B0, B1)                            \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

#define FN_MMA_884_RR(C, A0, A1, B0, B1)                            \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

#define FN_MMA_884_CR(C, A0, A1, B0, B1)                            \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

namespace gdn_sm70 {

constexpr int kGdnHg = 4;
constexpr int kGdnH = 12;
constexpr int kGdnK = 128;
constexpr int kGdnV = 128;
constexpr int kGdnS = 64;
constexpr float kLog2E = 1.4426950408889634f;

// Single-launch Fused (chunk_local_cumsum + HMMA K@K^T + FP32 64x64 Block-16/32
// Unit Lower-Triangular Inverse + A_bar + HMMA Q@K^T P) across all 80 V100 SMs!
// Eliminates chunk_local_cumsum + tilelang_kkt_solve_kernel + 5.4 GB
// intermediate a_mat HBM read/write + duplicate K HBM read! 35,328 B static
// smem <= 48 KiB -> 2 CTAs/SM (16 warps/SM), 0 spills!
template <typename BetaT, typename IdxT>
__global__ __launch_bounds__(256, 2) void fn_gdn_cumsum_kkt_prep_ap_kernel(
    const half* __restrict__ q,      // [1, T, 4, 128]
    const half* __restrict__ k,      // [1, T, 4, 128]
    half* __restrict__ a_bar_out,    // [1, T, 12, 64]
    half* __restrict__ p_out,        // [1, T, 12, 64]
    float* __restrict__ g_out,       // [1, T, 12]
    const float* __restrict__ g_in,  // [1, T, 12] (raw un-cumsum'd g)
    const BetaT* __restrict__ b,     // [1, T, 12]
    const int32_t* __restrict__ cu_seqlens,
    const IdxT* __restrict__ chunk_indices,  // [num_chunks, 2]
    float scale) {
  const int bc = blockIdx.x;
  const int bh = blockIdx.y;  // 0..11
  const int bhg = bh / (kGdnH / kGdnHg);
  const int batch_idx = static_cast<int>(chunk_indices[bc * 2 + 0]);
  const int chunk_idx = static_cast<int>(chunk_indices[bc * 2 + 1]);
  const int seq_start = cu_seqlens[batch_idx];
  const int seq_end = cu_seqlens[batch_idx + 1];
  const int left = seq_start + chunk_idx * kGdnS;
  const int valid_len = min(kGdnS, max(0, seq_end - left));
  if (valid_len <= 0) return;

  union FusedPrepSmem {
    struct {
      half s_q[kGdnS][136];  // 17,408 B
      half s_k[kGdnS][136];  // 17,408 B -> 34,816 B
    } qk;
    struct {
      float s_A[kGdnS][65];  // 16,640 B (odd stride 65 = 0 bank conflicts!)
      half s_p[kGdnS][72];   //  9,216 B -> 25,856 B
    } inv;
  };
  __shared__ __align__(16) FusedPrepSmem smem;
  __shared__ float s_g[kGdnS];
  __shared__ float s_b[kGdnS];

  const int tx = threadIdx.x;  // 0..255
  const int lane = tx & 31;
  const int warp = tx >> 5;  // 0..7

  // 1. Warp 0 computes inclusive chunk_local_cumsum(g) + stores g_out & s_g;
  //    Warp 1 loads beta into s_b; all 8 warps load Q and K ONCE into smem!
  if (warp == 0) {
    const int r0 = lane;
    const int r1 = lane + 32;
    float gv0 = (r0 < valid_len)
                    ? __ldg(g_in + static_cast<int64_t>(left + r0) * kGdnH + bh)
                    : 0.0f;
    float gv1 = (r1 < valid_len)
                    ? __ldg(g_in + static_cast<int64_t>(left + r1) * kGdnH + bh)
                    : 0.0f;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float n0 = __shfl_up_sync(0xffffffffu, gv0, d);
      const float n1 = __shfl_up_sync(0xffffffffu, gv1, d);
      if (lane >= d) {
        gv0 += n0;
        gv1 += n1;
      }
    }
    const float sum0 = __shfl_sync(0xffffffffu, gv0, 31);
    gv1 += sum0;
    s_g[r0] = gv0;
    s_g[r1] = gv1;
    if (r0 < valid_len) {
      g_out[static_cast<int64_t>(left + r0) * kGdnH + bh] = gv0;
    }
    if (r1 < valid_len) {
      g_out[static_cast<int64_t>(left + r1) * kGdnH + bh] = gv1;
    }
  } else if (warp == 1) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int r = lane + u * 32;
      float bv = 0.0f;
      if (r < valid_len) {
        if constexpr (std::is_same_v<BetaT, float>) {
          bv = __ldg(b + static_cast<int64_t>(left + r) * kGdnH + bh);
        } else {
          bv = __half2float(
              __ldg(b + static_cast<int64_t>(left + r) * kGdnH + bh));
        }
      }
      s_b[r] = bv;
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int idx = tx + i * 256;
    const int r = idx >> 4;
    const int c8 = (idx & 15) * 8;
    uint4 qv = make_uint4(0, 0, 0, 0);
    uint4 kv = make_uint4(0, 0, 0, 0);
    if (r < valid_len) {
      const int64_t base =
          (static_cast<int64_t>(left + r) * kGdnHg + bhg) * kGdnK + c8;
      qv = __ldg(reinterpret_cast<const uint4*>(q + base));
      kv = __ldg(reinterpret_cast<const uint4*>(k + base));
    }
    *reinterpret_cast<uint4*>(&smem.qk.s_q[r][c8]) = qv;
    *reinterpret_cast<uint4*>(&smem.qk.s_k[r][c8]) = kv;
  }
  __syncthreads();

  // 2. Simultaneously compute Q @ K^T AND K @ K^T on Volta Tensor Cores while
  //    s_q and s_k are resident in shared memory (reusing kb0, kb1 loads)!
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int rq = warp * 8 + mma_row;
  const int ck0 = quad * 8 + mma_row;
  const int ck1 = 32 + ck0;
  float p0[8] = {}, p1[8] = {};
  float k0[8] = {}, k1[8] = {};

#pragma unroll
  for (int k_off = 0; k_off < kGdnK; k_off += 8) {
    const uint4 qa = *reinterpret_cast<const uint4*>(&smem.qk.s_q[rq][k_off]);
    const uint4 ka = *reinterpret_cast<const uint4*>(&smem.qk.s_k[rq][k_off]);
    const uint4 kb0 = *reinterpret_cast<const uint4*>(&smem.qk.s_k[ck0][k_off]);
    const uint4 kb1 = *reinterpret_cast<const uint4*>(&smem.qk.s_k[ck1][k_off]);
    FN_MMA_884_RC(p0, qa.x, qa.y, kb0.x, kb0.y);
    FN_MMA_884_RC(p0, qa.z, qa.w, kb0.z, kb0.w);
    FN_MMA_884_RC(p1, qa.x, qa.y, kb1.x, kb1.y);
    FN_MMA_884_RC(p1, qa.z, qa.w, kb1.z, kb1.w);
    FN_MMA_884_RC(k0, ka.x, ka.y, kb0.x, kb0.y);
    FN_MMA_884_RC(k0, ka.z, ka.w, kb0.z, kb0.w);
    FN_MMA_884_RC(k1, ka.x, ka.y, kb1.x, kb1.y);
    FN_MMA_884_RC(k1, ka.z, ka.w, kb1.z, kb1.w);
  }
  __syncthreads();  // Done with s_q and s_k!

  // 3. Stage BOTH P -> smem.inv.s_p AND initial (I - strictLower(b * K@K^T))
  //    -> smem.inv.s_A BEFORE the next barrier!
  const int r_off = ((lane & 16) ? 4 : 0) | (lane & 1);
  const int r0 = warp * 8 + r_off;
  const int r2 = r0 + 2;
  const int c_sub = quad * 8 + (((lane >> 1) & 1) << 1);
  const float gr0 = s_g[r0];
  const float gr2 = s_g[r2];
  const float br0 = s_b[r0];
  const float br2 = s_b[r2];

  auto stage_pk_half = [&](const float (&pa)[8], const float (&ka)[8],
                           int c_half) {
    const int cb = c_half + c_sub;
    const float gc0 = s_g[cb];
    const float gc1 = s_g[cb + 1];
    const float gc4 = s_g[cb + 4];
    const float gc5 = s_g[cb + 5];
    *reinterpret_cast<half2*>(&smem.inv.s_p[r0][cb]) = __floats2half2_rn(
        (r0 >= cb) ? pa[0] * (scale * exp2f((gr0 - gc0) * kLog2E)) : 0.0f,
        (r0 >= cb + 1) ? pa[1] * (scale * exp2f((gr0 - gc1) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.inv.s_p[r2][cb]) = __floats2half2_rn(
        (r2 >= cb) ? pa[2] * (scale * exp2f((gr2 - gc0) * kLog2E)) : 0.0f,
        (r2 >= cb + 1) ? pa[3] * (scale * exp2f((gr2 - gc1) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.inv.s_p[r0][cb + 4]) = __floats2half2_rn(
        (r0 >= cb + 4) ? pa[4] * (scale * exp2f((gr0 - gc4) * kLog2E)) : 0.0f,
        (r0 >= cb + 5) ? pa[5] * (scale * exp2f((gr0 - gc5) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.inv.s_p[r2][cb + 4]) = __floats2half2_rn(
        (r2 >= cb + 4) ? pa[6] * (scale * exp2f((gr2 - gc4) * kLog2E)) : 0.0f,
        (r2 >= cb + 5) ? pa[7] * (scale * exp2f((gr2 - gc5) * kLog2E)) : 0.0f);

    auto init_a = [&](int r, int c, float br, float kv) -> float {
      if (r == c) return 1.0f;
      if (r > c && r < valid_len && c < valid_len) return -br * kv;
      return 0.0f;
    };
    smem.inv.s_A[r0][cb + 0] = init_a(r0, cb + 0, br0, ka[0]);
    smem.inv.s_A[r0][cb + 1] = init_a(r0, cb + 1, br0, ka[1]);
    smem.inv.s_A[r2][cb + 0] = init_a(r2, cb + 0, br2, ka[2]);
    smem.inv.s_A[r2][cb + 1] = init_a(r2, cb + 1, br2, ka[3]);
    smem.inv.s_A[r0][cb + 4] = init_a(r0, cb + 4, br0, ka[4]);
    smem.inv.s_A[r0][cb + 5] = init_a(r0, cb + 5, br0, ka[5]);
    smem.inv.s_A[r2][cb + 4] = init_a(r2, cb + 4, br2, ka[6]);
    smem.inv.s_A[r2][cb + 5] = init_a(r2, cb + 5, br2, ka[7]);
  };
  stage_pk_half(p0, k0, 0);
  stage_pk_half(p1, k1, 32);
  __syncthreads();

  // Flush P to global memory using coalesced 128-bit stores
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tx + i * 256;
    const int r = idx >> 3;
    const int c0 = (idx & 7) * 8;
    if (r < valid_len) {
      *reinterpret_cast<uint4*>(
          p_out + (static_cast<int64_t>(left + r) * kGdnH + bh) * kGdnS + c0) =
          *reinterpret_cast<const uint4*>(&smem.inv.s_p[r][c0]);
    }
  }

  // 4. Exact FP32 64x64 Block-16/32 Unit Lower-Triangular Inverse in
  // smem.inv.s_A: Step 4a: Warps 0..3 invert the four 16x16 diagonal blocks in
  // parallel!
  if (warp < 4) {
    const int base = warp * 16;
    for (int i = 1; i < 16; ++i) {
      const int r = base + i;
      const float a_rc = (lane < i) ? smem.inv.s_A[r][base + lane] : 0.0f;
      float sum = 0.0f;
      for (int m = 1; m < i; ++m) {
        const float neg_l_rm = __shfl_sync(0xffffffffu, a_rc, m);
        if (lane < m) {
          sum = __fmaf_rn(neg_l_rm, smem.inv.s_A[base + m][base + lane], sum);
        }
      }
      if (lane < i) {
        smem.inv.s_A[r][base + lane] = a_rc + sum;
      }
      __syncwarp(0xffffffffu);
    }
  }
  __syncthreads();

  // Step 4b: Off-diagonal 16x16 blocks A_10 = A_11 * (-L_10) * A_00 and
  //          A_32 = A_33 * (-L_32) * A_22 across all 256 threads!
  {
    const int lr = tx >> 4;  // 0..15
    const int lc = tx & 15;  // 0..15
    float t10 = 0.0f, t32 = 0.0f;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
      t10 = __fmaf_rn(smem.inv.s_A[16 + lr][m], smem.inv.s_A[m][lc], t10);
      t32 = __fmaf_rn(smem.inv.s_A[48 + lr][32 + m],
                      smem.inv.s_A[32 + m][32 + lc], t32);
    }
    __syncthreads();
    smem.inv.s_A[16 + lr][lc] = t10;
    smem.inv.s_A[48 + lr][32 + lc] = t32;
    __syncthreads();

    float a10 = 0.0f, a32 = 0.0f;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
      a10 = __fmaf_rn(smem.inv.s_A[16 + lr][16 + m], smem.inv.s_A[16 + m][lc],
                      a10);
      a32 = __fmaf_rn(smem.inv.s_A[48 + lr][48 + m],
                      smem.inv.s_A[48 + m][32 + lc], a32);
    }
    __syncthreads();
    smem.inv.s_A[16 + lr][lc] = a10;
    smem.inv.s_A[48 + lr][32 + lc] = a32;
    __syncthreads();
  }

  // Step 4c: Bottom-left 32x32 quadrant M_10 = M_1 * (-L_bottom_left) * M_0!
  {
    const int lr = tx >> 3;           // 0..31
    const int c_base = (tx & 7) * 4;  // 0, 4, ..., 28
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
#pragma unroll
    for (int m = 0; m < 32; ++m) {
      const float lhs = smem.inv.s_A[32 + lr][m];
      t0 = __fmaf_rn(lhs, smem.inv.s_A[m][c_base + 0], t0);
      t1 = __fmaf_rn(lhs, smem.inv.s_A[m][c_base + 1], t1);
      t2 = __fmaf_rn(lhs, smem.inv.s_A[m][c_base + 2], t2);
      t3 = __fmaf_rn(lhs, smem.inv.s_A[m][c_base + 3], t3);
    }
    __syncthreads();
    smem.inv.s_A[32 + lr][c_base + 0] = t0;
    smem.inv.s_A[32 + lr][c_base + 1] = t1;
    smem.inv.s_A[32 + lr][c_base + 2] = t2;
    smem.inv.s_A[32 + lr][c_base + 3] = t3;
    __syncthreads();

    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
#pragma unroll
    for (int m = 0; m < 32; ++m) {
      const float lhs = smem.inv.s_A[32 + lr][32 + m];
      m0 = __fmaf_rn(lhs, smem.inv.s_A[32 + m][c_base + 0], m0);
      m1 = __fmaf_rn(lhs, smem.inv.s_A[32 + m][c_base + 1], m1);
      m2 = __fmaf_rn(lhs, smem.inv.s_A[32 + m][c_base + 2], m2);
      m3 = __fmaf_rn(lhs, smem.inv.s_A[32 + m][c_base + 3], m3);
    }
    __syncthreads();
    smem.inv.s_A[32 + lr][c_base + 0] = m0;
    smem.inv.s_A[32 + lr][c_base + 1] = m1;
    smem.inv.s_A[32 + lr][c_base + 2] = m2;
    smem.inv.s_A[32 + lr][c_base + 3] = m3;
    __syncthreads();
  }

  // 5. Scale A -> A_bar and write coalesced 128-bit uint4 directly to HBM!
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tx + i * 256;
    const int r = idx >> 3;
    const int c0 = (idx & 7) * 8;
    if (r < valid_len) {
      const float gr = s_g[r];
      uint4 av;
      half* ah = reinterpret_cast<half*>(&av);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int c = c0 + j;
        const float gf =
            (r >= c && c < valid_len) ? exp2f((gr - s_g[c]) * kLog2E) : 0.0f;
        ah[j] = __float2half_rn(smem.inv.s_A[r][c] * gf * s_b[c]);
      }
      *reinterpret_cast<uint4*>(
          a_bar_out + (static_cast<int64_t>(left + r) * kGdnH + bh) * kGdnS +
          c0) = av;
    }
  }
}

struct GdnRecurrentSmem {
  half s_k[kGdnS][136];  // 17,408 B
  half s_q[kGdnS][136];  // 17,408 B
  half s_ap[kGdnS][72];  //  9,216 B (holds A_bar, then P)
  half s_h[kGdnK][36];   //  9,216 B
  half s_v[kGdnS][36];   //  4,608 B (holds V -> u -> vn)
  half s_vd[kGdnS][36];  //  4,608 B
  float s_gexp[kGdnS];   //    256 B
  float s_grev[kGdnS];   //    256 B -> Total = 62,976 B (<= 64 KB on 80 SMs!)
};

template <bool MODE_96CTA>
__global__ __launch_bounds__(256, 1) void fn_gdn_recurrent_kernel(
    const half* __restrict__ q,      // [1, T, 4, 128]
    const half* __restrict__ k,      // [1, T, 4, 128]
    const half* __restrict__ v,      // [1, T, 12, 128]
    const half* __restrict__ a_bar,  // [1, T, 12, 64]
    const half* __restrict__ p_mat,  // [1, T, 12, 64]
    const float* __restrict__ g,     // [1, T, 12]
    float* __restrict__ h0_state,    // [slots, 12, 128, 128]
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ state_indices,
    const bool* __restrict__ has_initial_state,
    half* __restrict__ o_out,    // [1, T, 12, 128]
    float* __restrict__ ht_out,  // [B, 12, 128, 128] or nullptr
    float scale, bool use_initial_state, bool state_layout_vlk,
    bool inplace_final_state, bool store_final_state) {
  extern __shared__ __align__(16) char raw_smem[];
  GdnRecurrentSmem& smem = *reinterpret_cast<GdnRecurrentSmem*>(raw_smem);

  const int cta_id = blockIdx.x;
  const int bb = blockIdx.y;
  int bh = 0, col_start = 0, bv = 16;
  if constexpr (MODE_96CTA) {
    bh = cta_id >> 3;  // 0..11
    col_start = (cta_id & 7) * 16;
    bv = 16;
  } else {
    // Exact 80-CTA decomposition across all 80 V100 SMs (zero idle SMs!):
    // Heads 0..7: 7 CTAs each (24, 24, 16, 16, 16, 16, 16) = 56 CTAs
    // Heads 8..11: 6 CTAs each (24, 24, 24, 24, 16, 16)    = 24 CTAs
    if (cta_id < 56) {
      bh = cta_id / 7;
      const int s = cta_id % 7;
      if (s < 2) {
        col_start = s * 24;
        bv = 24;
      } else {
        col_start = 48 + (s - 2) * 16;
        bv = 16;
      }
    } else {
      const int rem = cta_id - 56;
      bh = 8 + rem / 6;
      const int s = rem % 6;
      if (s < 4) {
        col_start = s * 24;
        bv = 24;
      } else {
        col_start = 96 + (s - 4) * 16;
        bv = 16;
      }
    }
  }
  const int bhg = bh / (kGdnH / kGdnHg);
  const int seq_start = cu_seqlens[bb];
  const int seq_end = cu_seqlens[bb + 1];
  const int seq_len = seq_end - seq_start;
  const int num_iters = (seq_len + kGdnS - 1) / kGdnS;
  const int state_slot = (state_indices != nullptr) ? state_indices[bb] : bb;
  const bool init_valid = use_initial_state && (has_initial_state == nullptr ||
                                                has_initial_state[bb]);

  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;  // 0..7
  const int quad = (lane >> 2) & 3;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int r_off = ((lane & 16) ? 4 : 0) | (lane & 1);
  const int c_b = quad * 8 + (((lane >> 1) & 1) << 1);
  const int vn_col = quad * 8 + ((lane & 16) ? 4 : 0);
  const int vk = lane & 3;

  float h0[8] = {}, h1[8] = {};
  if (init_valid) {
    const float* h_src =
        h0_state +
        (static_cast<int64_t>(state_slot) * kGdnH + bh) * (kGdnK * kGdnV);
    auto load_h_tile = [&](float (&acc)[8], int kb) {
      const int kr0 = kb + r_off;
      const int kr2 = kr0 + 2;
      auto rd = [&](int kr, int vc) -> float {
        if (vc >= bv) return 0.0f;
        const int gc = col_start + vc;
        return state_layout_vlk ? __ldg(h_src + gc * kGdnK + kr)
                                : __ldg(h_src + kr * kGdnV + gc);
      };
      acc[0] = rd(kr0, c_b);
      acc[1] = rd(kr0, c_b + 1);
      acc[2] = rd(kr2, c_b);
      acc[3] = rd(kr2, c_b + 1);
      acc[4] = rd(kr0, c_b + 4);
      acc[5] = rd(kr0, c_b + 5);
      acc[6] = rd(kr2, c_b + 4);
      acc[7] = rd(kr2, c_b + 5);
    };
    load_h_tile(h0, warp * 16);
    load_h_tile(h1, warp * 16 + 8);
  }

  if (num_iters <= 0) return;

  // Software-pipelined chunk prefetch registers (~46 regs out of 127 free regs
  // at 1 CTA/SM -> <= 185 regs <= 255 max, ZERO spills, 0.0 bit-exact diff!):
  // Overlaps chunk (i_s + 1)'s global K, Q, A_bar, V, g loads with Step 4's
  // 80 Tensor Core HMMA instructions!
  uint4 kv_pre[4], qv_pre[4], av_pre[2];
  uint4 vv_pre = make_uint4(0, 0, 0, 0);
  float g0_pre = 0.0f, g1_pre = 0.0f;

  auto prefetch_chunk_inputs = [&](int chunk_left, int chunk_valid) {
    if (warp == 0) {
      const int r0 = lane;
      const int r1 = lane + 32;
      g0_pre = __ldg(
          g +
          static_cast<int64_t>(chunk_left + min(r0, chunk_valid - 1)) * kGdnH +
          bh);
      g1_pre = __ldg(
          g +
          static_cast<int64_t>(chunk_left + min(r1, chunk_valid - 1)) * kGdnH +
          bh);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = tx + i * 256;
      const int r = idx >> 4;
      const int c8 = (idx & 15) * 8;
      kv_pre[i] = make_uint4(0, 0, 0, 0);
      qv_pre[i] = make_uint4(0, 0, 0, 0);
      if (r < chunk_valid) {
        const int64_t base =
            (static_cast<int64_t>(chunk_left + r) * kGdnHg + bhg) * kGdnK + c8;
        kv_pre[i] = __ldg(reinterpret_cast<const uint4*>(k + base));
        qv_pre[i] = __ldg(reinterpret_cast<const uint4*>(q + base));
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tx + i * 256;
      const int r = idx >> 3;
      const int c8 = (idx & 7) * 8;
      av_pre[i] = make_uint4(0, 0, 0, 0);
      if (r < chunk_valid) {
        av_pre[i] = __ldg(reinterpret_cast<const uint4*>(
            a_bar +
            (static_cast<int64_t>(chunk_left + r) * kGdnH + bh) * kGdnS + c8));
      }
    }
    {
      const int r = tx >> 2;
      const int c8 = (tx & 3) * 8;
      vv_pre = make_uint4(0, 0, 0, 0);
      if (r < chunk_valid && c8 < bv) {
        vv_pre = __ldg(reinterpret_cast<const uint4*>(
            v + (static_cast<int64_t>(chunk_left + r) * kGdnH + bh) * kGdnV +
            col_start + c8));
      }
    }
  };

  prefetch_chunk_inputs(seq_start, min(kGdnS, seq_end - seq_start));

  for (int i_s = 0; i_s < num_iters; ++i_s) {
    const int left = seq_start + i_s * kGdnS;
    const int valid_len = min(kGdnS, seq_end - left);

    // 1. Stage prefetched K, Q, A_bar, V, g + H_old into smem in ONE barrier!
    if (warp == 0) {
      const int r0 = lane;
      const int r1 = lane + 32;
      const float g_last = __shfl_sync(0xffffffffu, g1_pre, 31);
      smem.s_gexp[r0] = exp2f(g0_pre * kLog2E);
      smem.s_gexp[r1] = exp2f(g1_pre * kLog2E);
      smem.s_grev[r0] =
          (r0 < valid_len) ? exp2f((g_last - g0_pre) * kLog2E) : 0.0f;
      smem.s_grev[r1] =
          (r1 < valid_len) ? exp2f((g_last - g1_pre) * kLog2E) : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = tx + i * 256;
      const int r = idx >> 4;
      const int c8 = (idx & 15) * 8;
      *reinterpret_cast<uint4*>(&smem.s_k[r][c8]) = kv_pre[i];
      *reinterpret_cast<uint4*>(&smem.s_q[r][c8]) = qv_pre[i];
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tx + i * 256;
      const int r = idx >> 3;
      const int c8 = (idx & 7) * 8;
      *reinterpret_cast<uint4*>(&smem.s_ap[r][c8]) = av_pre[i];
    }

    {
      const int r = tx >> 2;
      const int c8 = (tx & 3) * 8;
      const uint2* vu2 = reinterpret_cast<const uint2*>(&vv_pre);
      *reinterpret_cast<uint2*>(&smem.s_v[r][c8]) = vu2[0];
      *reinterpret_cast<uint2*>(&smem.s_v[r][c8 + 4]) = vu2[1];
    }

    // Stage H_old ([128, bv]) into s_h
    auto stage_h = [&](const float (&acc)[8], int kb) {
      const int kr0 = kb + r_off;
      const int kr2 = kr0 + 2;
      *reinterpret_cast<half2*>(&smem.s_h[kr0][c_b]) =
          __floats2half2_rn(acc[0], acc[1]);
      *reinterpret_cast<half2*>(&smem.s_h[kr2][c_b]) =
          __floats2half2_rn(acc[2], acc[3]);
      *reinterpret_cast<half2*>(&smem.s_h[kr0][c_b + 4]) =
          __floats2half2_rn(acc[4], acc[5]);
      *reinterpret_cast<half2*>(&smem.s_h[kr2][c_b + 4]) =
          __floats2half2_rn(acc[6], acc[7]);
    };
    stage_h(h0, warp * 16);
    stage_h(h1, warp * 16 + 8);
    __syncthreads();  // Barrier #1

    // 2. u = V - g_exp * (K @ H_old), while prefetching P in registers!
    uint4 p_pre0 = make_uint4(0, 0, 0, 0);
    uint4 p_pre1 = make_uint4(0, 0, 0, 0);
    {
      const int idx0 = tx;
      const int idx1 = tx + 256;
      const int pr0 = idx0 >> 3, pc0 = (idx0 & 7) * 8;
      const int pr1 = idx1 >> 3, pc1 = (idx1 & 7) * 8;
      if (pr0 < valid_len) {
        p_pre0 = __ldg(reinterpret_cast<const uint4*>(
            p_mat + (static_cast<int64_t>(left + pr0) * kGdnH + bh) * kGdnS +
            pc0));
      }
      if (pr1 < valid_len) {
        p_pre1 = __ldg(reinterpret_cast<const uint4*>(
            p_mat + (static_cast<int64_t>(left + pr1) * kGdnH + bh) * kGdnS +
            pc1));
      }
    }

    const int sb = warp * 8;
    const int sr0 = sb + r_off;
    const int sr2 = sr0 + 2;
    float u_acc[8] = {};
#pragma unroll
    for (int k_off = 0; k_off < kGdnK; k_off += 8) {
      const uint4 ka =
          *reinterpret_cast<const uint4*>(&smem.s_k[sb + mma_row][k_off]);
      const uint2 hb0 =
          *reinterpret_cast<const uint2*>(&smem.s_h[k_off + vk][vn_col]);
      const uint2 hb1 =
          *reinterpret_cast<const uint2*>(&smem.s_h[k_off + 4 + vk][vn_col]);
      FN_MMA_884_RR(u_acc, ka.x, ka.y, hb0.x, hb0.y);
      FN_MMA_884_RR(u_acc, ka.z, ka.w, hb1.x, hb1.y);
    }
    {
      const float ge0 = smem.s_gexp[sr0];
      const float ge2 = smem.s_gexp[sr2];
      const half2 v01 = *reinterpret_cast<const half2*>(&smem.s_v[sr0][c_b]);
      const half2 v23 = *reinterpret_cast<const half2*>(&smem.s_v[sr2][c_b]);
      const half2 v45 =
          *reinterpret_cast<const half2*>(&smem.s_v[sr0][c_b + 4]);
      const half2 v67 =
          *reinterpret_cast<const half2*>(&smem.s_v[sr2][c_b + 4]);
      *reinterpret_cast<half2*>(&smem.s_v[sr0][c_b]) =
          __floats2half2_rn(__half2float(v01.x) - ge0 * u_acc[0],
                            __half2float(v01.y) - ge0 * u_acc[1]);
      *reinterpret_cast<half2*>(&smem.s_v[sr2][c_b]) =
          __floats2half2_rn(__half2float(v23.x) - ge2 * u_acc[2],
                            __half2float(v23.y) - ge2 * u_acc[3]);
      *reinterpret_cast<half2*>(&smem.s_v[sr0][c_b + 4]) =
          __floats2half2_rn(__half2float(v45.x) - ge0 * u_acc[4],
                            __half2float(v45.y) - ge0 * u_acc[5]);
      *reinterpret_cast<half2*>(&smem.s_v[sr2][c_b + 4]) =
          __floats2half2_rn(__half2float(v67.x) - ge2 * u_acc[6],
                            __half2float(v67.y) - ge2 * u_acc[7]);
    }
    __syncthreads();  // Barrier #2

    // 3. vd = A_bar @ u, then store vd -> s_vd, vn -> s_v, and P -> s_ap!
    float vd_acc[8] = {};
#pragma unroll
    for (int k_off = 0; k_off < kGdnS; k_off += 8) {
      const uint4 aa =
          *reinterpret_cast<const uint4*>(&smem.s_ap[sb + mma_row][k_off]);
      const uint2 ub0 =
          *reinterpret_cast<const uint2*>(&smem.s_v[k_off + vk][vn_col]);
      const uint2 ub1 =
          *reinterpret_cast<const uint2*>(&smem.s_v[k_off + 4 + vk][vn_col]);
      FN_MMA_884_RR(vd_acc, aa.x, aa.y, ub0.x, ub0.y);
      FN_MMA_884_RR(vd_acc, aa.z, aa.w, ub1.x, ub1.y);
    }
    __syncthreads();  // Ensure all warps finished reading s_ap and s_v!
    {
      const float grv0 = smem.s_grev[sr0];
      const float grv2 = smem.s_grev[sr2];
      *reinterpret_cast<half2*>(&smem.s_vd[sr0][c_b]) =
          __floats2half2_rn(vd_acc[0], vd_acc[1]);
      *reinterpret_cast<half2*>(&smem.s_vd[sr2][c_b]) =
          __floats2half2_rn(vd_acc[2], vd_acc[3]);
      *reinterpret_cast<half2*>(&smem.s_vd[sr0][c_b + 4]) =
          __floats2half2_rn(vd_acc[4], vd_acc[5]);
      *reinterpret_cast<half2*>(&smem.s_vd[sr2][c_b + 4]) =
          __floats2half2_rn(vd_acc[6], vd_acc[7]);

      *reinterpret_cast<half2*>(&smem.s_v[sr0][c_b]) =
          __floats2half2_rn(vd_acc[0] * grv0, vd_acc[1] * grv0);
      *reinterpret_cast<half2*>(&smem.s_v[sr2][c_b]) =
          __floats2half2_rn(vd_acc[2] * grv2, vd_acc[3] * grv2);
      *reinterpret_cast<half2*>(&smem.s_v[sr0][c_b + 4]) =
          __floats2half2_rn(vd_acc[4] * grv0, vd_acc[5] * grv0);
      *reinterpret_cast<half2*>(&smem.s_v[sr2][c_b + 4]) =
          __floats2half2_rn(vd_acc[6] * grv2, vd_acc[7] * grv2);

      const int idx0 = tx, idx1 = tx + 256;
      *reinterpret_cast<uint4*>(&smem.s_ap[idx0 >> 3][(idx0 & 7) * 8]) = p_pre0;
      *reinterpret_cast<uint4*>(&smem.s_ap[idx1 >> 3][(idx1 & 7) * 8]) = p_pre1;
    }
    __syncthreads();  // Barrier #3

    // Prefetch chunk (i_s + 1)'s K, Q, A_bar, V, g BEFORE Step 4's 80 HMMAs!
    if (i_s + 1 < num_iters) {
      const int next_left = left + kGdnS;
      prefetch_chunk_inputs(next_left, min(kGdnS, seq_end - next_left));
    }

    // 4. O = scale * g_exp * (Q @ H_old) + P @ vd, AND H_new = g_last * H_old +
    // K^T @ vn!
    float o_acc[8] = {};
#pragma unroll
    for (int k_off = 0; k_off < kGdnK; k_off += 8) {
      const uint4 qa =
          *reinterpret_cast<const uint4*>(&smem.s_q[sb + mma_row][k_off]);
      const uint2 hb0 =
          *reinterpret_cast<const uint2*>(&smem.s_h[k_off + vk][vn_col]);
      const uint2 hb1 =
          *reinterpret_cast<const uint2*>(&smem.s_h[k_off + 4 + vk][vn_col]);
      FN_MMA_884_RR(o_acc, qa.x, qa.y, hb0.x, hb0.y);
      FN_MMA_884_RR(o_acc, qa.z, qa.w, hb1.x, hb1.y);
    }
    {
      const float sge0 = scale * smem.s_gexp[sr0];
      const float sge2 = scale * smem.s_gexp[sr2];
      o_acc[0] *= sge0;
      o_acc[1] *= sge0;
      o_acc[4] *= sge0;
      o_acc[5] *= sge0;
      o_acc[2] *= sge2;
      o_acc[3] *= sge2;
      o_acc[6] *= sge2;
      o_acc[7] *= sge2;
    }
#pragma unroll
    for (int k_off = 0; k_off < kGdnS; k_off += 8) {
      const uint4 pa =
          *reinterpret_cast<const uint4*>(&smem.s_ap[sb + mma_row][k_off]);
      const uint2 vdb0 =
          *reinterpret_cast<const uint2*>(&smem.s_vd[k_off + vk][vn_col]);
      const uint2 vdb1 =
          *reinterpret_cast<const uint2*>(&smem.s_vd[k_off + 4 + vk][vn_col]);
      FN_MMA_884_RR(o_acc, pa.x, pa.y, vdb0.x, vdb0.y);
      FN_MMA_884_RR(o_acc, pa.z, pa.w, vdb1.x, vdb1.y);
    }
    // Emit O directly to global HBM
    {
      auto emit_o2 = [&](int sr, int cb_off, float v0, float v1) {
        if (sr < valid_len && cb_off < bv) {
          *reinterpret_cast<half2*>(
              o_out + (static_cast<int64_t>(left + sr) * kGdnH + bh) * kGdnV +
              col_start + cb_off) = __floats2half2_rn(v0, v1);
        }
      };
      emit_o2(sr0, c_b, o_acc[0], o_acc[1]);
      emit_o2(sr2, c_b, o_acc[2], o_acc[3]);
      emit_o2(sr0, c_b + 4, o_acc[4], o_acc[5]);
      emit_o2(sr2, c_b + 4, o_acc[6], o_acc[7]);
    }

    // Update H_new = g_exp[63] * H_old + K^T @ vn
    {
      const float g_last = smem.s_gexp[kGdnS - 1];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        h0[e] *= g_last;
        h1[e] *= g_last;
      }
      const int am = (lane & 16) ? 4 : 0;
      const int kb0 = warp * 16 + am;
      const int kb1 = warp * 16 + 8 + am;
#pragma unroll
      for (int k_off = 0; k_off < kGdnS; k_off += 4) {
        const uint2 vnb =
            *reinterpret_cast<const uint2*>(&smem.s_v[k_off + vk][vn_col]);
        const uint2 ka0 =
            *reinterpret_cast<const uint2*>(&smem.s_k[k_off + vk][kb0]);
        const uint2 ka1 =
            *reinterpret_cast<const uint2*>(&smem.s_k[k_off + vk][kb1]);
        FN_MMA_884_CR(h0, ka0.x, ka0.y, vnb.x, vnb.y);
        FN_MMA_884_CR(h1, ka1.x, ka1.y, vnb.x, vnb.y);
      }
    }
    __syncthreads();  // Barrier #4 before next chunk overwrites smem
  }

  if (store_final_state) {
    float* dst =
        inplace_final_state
            ? (h0_state + (static_cast<int64_t>(state_slot) * kGdnH + bh) *
                              (kGdnK * kGdnV))
            : (ht_out +
               (static_cast<int64_t>(bb) * kGdnH + bh) * (kGdnK * kGdnV));
    auto store_h_tile = [&](const float (&acc)[8], int kb) {
      const int kr0 = kb + r_off;
      const int kr2 = kr0 + 2;
      auto wr = [&](int kr, int vc, float val) {
        if (vc < bv) {
          const int gc = col_start + vc;
          if (state_layout_vlk) {
            dst[gc * kGdnK + kr] = val;
          } else {
            dst[kr * kGdnV + gc] = val;
          }
        }
      };
      wr(kr0, c_b, acc[0]);
      wr(kr0, c_b + 1, acc[1]);
      wr(kr2, c_b, acc[2]);
      wr(kr2, c_b + 1, acc[3]);
      wr(kr0, c_b + 4, acc[4]);
      wr(kr0, c_b + 5, acc[5]);
      wr(kr2, c_b + 4, acc[6]);
      wr(kr2, c_b + 5, acc[7]);
    };
    store_h_tile(h0, warp * 16);
    store_h_tile(h1, warp * 16 + 8);
  }
}

void gdn_chunk_fwd_fused_sm70_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a_bar_buf,
    torch::Tensor p_buf, torch::Tensor g_cs_buf, torch::Tensor g_raw,
    torch::Tensor b, torch::Tensor initial_state, torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices, std::optional<torch::Tensor> state_indices,
    std::optional<torch::Tensor> has_initial_state, torch::Tensor out,
    std::optional<torch::Tensor> final_state, double scale,
    bool use_initial_state, bool state_layout_vlk, bool inplace_final_state,
    bool store_final_state) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int cur_dev = q.device().index();
  const int num_chunks = static_cast<int>(chunk_indices.size(0));
  const int real_batch = static_cast<int>(cu_seqlens.size(0)) - 1;
  if (real_batch <= 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();

  static bool configured_gdn_fused_devs[16] = {};
  constexpr size_t kRecSmem = sizeof(GdnRecurrentSmem);
  if (cur_dev >= 0 && cur_dev < 16 && !configured_gdn_fused_devs[cur_dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_cumsum_kkt_prep_ap_kernel<float, int32_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_cumsum_kkt_prep_ap_kernel<float, int64_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_cumsum_kkt_prep_ap_kernel<half, int32_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_cumsum_kkt_prep_ap_kernel<half, int64_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_recurrent_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kRecSmem));
    configured_gdn_fused_devs[cur_dev] = true;
  }

  if (num_chunks > 0) {
    dim3 prep_grid(num_chunks, kGdnH);
    if (b.scalar_type() == at::kFloat) {
      if (chunk_indices.scalar_type() == at::kInt) {
        fn_gdn_cumsum_kkt_prep_ap_kernel<float, int32_t>
            <<<prep_grid, 256, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr()),
                reinterpret_cast<const half*>(k.data_ptr()),
                reinterpret_cast<half*>(a_bar_buf.data_ptr()),
                reinterpret_cast<half*>(p_buf.data_ptr()),
                g_cs_buf.data_ptr<float>(), g_raw.data_ptr<float>(),
                b.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),
                chunk_indices.data_ptr<int32_t>(), static_cast<float>(scale));
      } else {
        fn_gdn_cumsum_kkt_prep_ap_kernel<float, int64_t>
            <<<prep_grid, 256, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr()),
                reinterpret_cast<const half*>(k.data_ptr()),
                reinterpret_cast<half*>(a_bar_buf.data_ptr()),
                reinterpret_cast<half*>(p_buf.data_ptr()),
                g_cs_buf.data_ptr<float>(), g_raw.data_ptr<float>(),
                b.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),
                chunk_indices.data_ptr<int64_t>(), static_cast<float>(scale));
      }
    } else {
      if (chunk_indices.scalar_type() == at::kInt) {
        fn_gdn_cumsum_kkt_prep_ap_kernel<half, int32_t>
            <<<prep_grid, 256, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr()),
                reinterpret_cast<const half*>(k.data_ptr()),
                reinterpret_cast<half*>(a_bar_buf.data_ptr()),
                reinterpret_cast<half*>(p_buf.data_ptr()),
                g_cs_buf.data_ptr<float>(), g_raw.data_ptr<float>(),
                reinterpret_cast<const half*>(b.data_ptr()),
                cu_seqlens.data_ptr<int32_t>(),
                chunk_indices.data_ptr<int32_t>(), static_cast<float>(scale));
      } else {
        fn_gdn_cumsum_kkt_prep_ap_kernel<half, int64_t>
            <<<prep_grid, 256, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr()),
                reinterpret_cast<const half*>(k.data_ptr()),
                reinterpret_cast<half*>(a_bar_buf.data_ptr()),
                reinterpret_cast<half*>(p_buf.data_ptr()),
                g_cs_buf.data_ptr<float>(), g_raw.data_ptr<float>(),
                reinterpret_cast<const half*>(b.data_ptr()),
                cu_seqlens.data_ptr<int32_t>(),
                chunk_indices.data_ptr<int64_t>(), static_cast<float>(scale));
      }
    }
  }

  const int32_t* st_idx_ptr =
      (state_indices.has_value() && state_indices->defined())
          ? state_indices->data_ptr<int32_t>()
          : nullptr;
  const bool* has_init_ptr =
      (has_initial_state.has_value() && has_initial_state->defined())
          ? has_initial_state->data_ptr<bool>()
          : nullptr;
  float* ht_ptr = (final_state.has_value() && final_state->defined())
                      ? final_state->data_ptr<float>()
                      : nullptr;

  fn_gdn_recurrent_kernel<false>
      <<<dim3(80, real_batch), 256, kRecSmem, stream>>>(
          reinterpret_cast<const half*>(q.data_ptr()),
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a_bar_buf.data_ptr()),
          reinterpret_cast<const half*>(p_buf.data_ptr()),
          g_cs_buf.data_ptr<float>(), initial_state.data_ptr<float>(),
          cu_seqlens.data_ptr<int32_t>(), st_idx_ptr, has_init_ptr,
          reinterpret_cast<half*>(out.data_ptr()), ht_ptr,
          static_cast<float>(scale), use_initial_state, state_layout_vlk,
          inplace_final_state, store_final_state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gdn_sm70

}  // namespace vllm_fn_sm70

TORCH_LIBRARY(vllm_sm70_fn_prefill, m) {
  m.def(
      "gdn_chunk_fwd_fused_sm70(Tensor q, Tensor k, Tensor v, "
      "Tensor(a!) a_bar_buf, Tensor(b!) p_buf, Tensor(c!) g_cs_buf, "
      "Tensor g_raw, Tensor b, Tensor(d!) initial_state, Tensor cu_seqlens, "
      "Tensor chunk_indices, Tensor? state_indices, Tensor? has_initial_state, "
      "Tensor(e!) out, Tensor(f!)? final_state, float scale, "
      "bool use_initial_state, bool state_layout_vlk, "
      "bool inplace_final_state, bool store_final_state) -> ()");
  m.impl("gdn_chunk_fwd_fused_sm70", torch::kCUDA,
         &vllm_fn_sm70::gdn_sm70::gdn_chunk_fwd_fused_sm70_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "SM70 GDN prefill operators";
}
