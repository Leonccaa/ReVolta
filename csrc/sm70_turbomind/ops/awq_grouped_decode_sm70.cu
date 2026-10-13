// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Qwen3.8 TP4 native-g32 AWQ grouped small-M MoE decode (M = 1..20 rows).
//
// The multi-row grouped dataflow follows nvfp4_grouped_decode_sm70.cu; the
// AWQ U4 decoder and 3-byte/4-byte metadata readers follow
// awq_qpn_m1_sm70.cu, whose quadpair-N m8n8k4 dataflow is derived from
// dnv2003/v100-skinny (MIT, see adjacent LICENSE.v100-skinny). The prepared
// TurboMind banks are read in place: no repack and no second weight copy.
//
// Three launches per MoE layer and no floating-point atomics:
//   1. plan: groups the M*10 routes by expert (ascending expert id, ascending
//      route inside an expert) into packs of at most 8*RB rows, written to a
//      fixed-size int32 buffer. One warp assigns ordinals with match.any,
//      so the plan is deterministic and contains no atomics.
//   2. W13 + SwiGLU: grid (10 N32 tiles, M*10 pack slots). A CTA reads one
//      tile of one active expert once and multiplies every row routed to it:
//      RB m8n8k4 row blocks share each decoded B fragment. K is split across
//      warps; FP32 partials are reduced in a fixed order in shared memory,
//      gate/up are rounded to FP16, SiLU is evaluated in FP32 from the FP16
//      gate and rounded, then multiplied by up in FP16 (the QPN M1 boundary).
//   3. W2 + weighted unroute: 80 CTAs, each owning 32 output columns for all
//      tokens. Warps walk the packs and read each expert's W2 slice once; FP16
//      route outputs stay in shared memory and, after one barrier, each token
//      is reduced over its 10 router slots in router order with fmaf in FP32
//      and rounded once to FP16 (the QPN M1 W2 boundary).
//
// With split=16 the per-route arithmetic (K assignment per warp, MMA chain,
// partial-sum order, FP16 boundaries, router-order fmaf) is the same as
// awq_moe_qpn_m1_sm70_out, so every row is expected to match a separate M=1
// QPN call bitwise (batch invariance). split=8 changes only the FP32 split-K
// association. Neither is bitwise equal to the TurboMind serial split-K route.

#include <torch/all.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <cstdint>

namespace {

constexpr int kK = 2560;   // hidden size (W13 K, W2 N)
constexpr int kN13 = 320;  // interleaved gate/up columns per TP4 rank
constexpr int kK2 = 160;   // intermediate size per TP4 rank (W2 K)
constexpr int kN2 = 2560;  // W2 output columns
constexpr int kExperts = 512;
constexpr int kTopK = 10;
constexpr int kMaxTokens = 20;
constexpr int kMaxRoutes = kMaxTokens * kTopK;  // 200
constexpr int kRowCap = 24;                     // 3 m8n8k4 row blocks
constexpr int kTiles13 = kN13 / 32;             // 10
constexpr int kTiles2 = kN2 / 32;               // 80

// int32 plan layout (fixed size, graph capturable):
//   [0]                         number of packs
//   [kPlanExperts + p]          expert id of pack p
//   [kPlanCounts + p]           rows in pack p (1..cap)
//   [kPlanRows + p*kRowCap + i] route index (token*10 + slot) of row i
constexpr int kPlanExperts = 8;
constexpr int kPlanCounts = kPlanExperts + kMaxRoutes;
constexpr int kPlanRows = kPlanCounts + kMaxRoutes;
constexpr int kPlanInts = kPlanRows + kMaxRoutes * kRowCap;  // 5208
static_assert(kPlanInts == 5208, "keep vllm/_sm70_ops.py in sync");

// Metadata formats, as in awq_qpn_m1_sm70.cu: 0 = compact 3-byte scalar
// reader, 1 = compact 3-byte warp-cooperative reader, 2 = 4-byte scale+bias.
template <int Format>
__device__ __forceinline__ uint32_t read_awq_stats(const uint8_t* stats,
                                                   int group, int tile, int col,
                                                   int n) {
  uint32_t bits = 0;
  if constexpr (Format == 2) {
    return __ldg(reinterpret_cast<const uint32_t*>(stats) + group * n +
                 tile * 32 + col);
  } else if constexpr (Format == 1) {
    const int lane = threadIdx.x & 31;
    const auto* words = reinterpret_cast<const uint32_t*>(
        stats + (static_cast<size_t>(group) * n + tile * 32) * 3);
    const uint32_t word = lane < 24 ? __ldg(words + lane) : 0;
    const int byte = col * 3;
    const int shift = (byte & 3) * 8;
    const uint32_t lo = __shfl_sync(0xffffffffu, word, byte >> 2);
    const uint32_t hi =
        __shfl_sync(0xffffffffu, word, (byte >> 2) + (shift > 8));
    bits = lo >> shift;
    if (shift > 8) bits |= hi << (32 - shift);
  } else {
    const auto* record =
        stats + (static_cast<size_t>(group) * n + tile * 32 + col) * 3;
    bits = static_cast<uint32_t>(__ldg(record)) |
           (static_cast<uint32_t>(__ldg(record + 1)) << 8) |
           (static_cast<uint32_t>(__ldg(record + 2)) << 16);
  }
  const half scale = __ushort_as_half(static_cast<uint16_t>(bits));
  const half zero = __int2half_rn(static_cast<int>((bits >> 16) & 0xff));
  const half bias = __hmul(__hneg(zero), scale);
  return (bits & 0xffffu) |
         (static_cast<uint32_t>(__half_as_ushort(bias)) << 16);
}

// Exact TurboMind U4 -> half conversion with the native FP16 bias boundary
// (identical to awq_qpn_m1_sm70.cu; do not substitute (q - zero) * scale).
__device__ __forceinline__ void dequant_awq_u4x8(uint32_t packed,
                                                 uint32_t stats,
                                                 half2* decoded) {
  uint32_t h[4];
  const uint32_t upper = __byte_perm(packed, 0, 0x4321);
  constexpr uint32_t lut = (0xf0 & 0xcc) | 0xaa;
  constexpr uint32_t bottom_mask = 0x000f000f;
  constexpr uint32_t top_mask = 0x00f000f0;
  constexpr uint32_t magic0 = 0x64006400;
  constexpr uint32_t magic1 = 0x54005400;
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[0])
      : "r"(packed), "n"(bottom_mask), "n"(magic0), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[1])
      : "r"(packed), "n"(top_mask), "n"(magic1), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[2])
      : "r"(upper), "n"(bottom_mask), "n"(magic0), "n"(lut));
  asm("lop3.b32 %0, %1, %2, %3, %4;"
      : "=r"(h[3])
      : "r"(upper), "n"(top_mask), "n"(magic1), "n"(lut));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[0]) : "r"(h[0]), "r"(magic0));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[1]) : "r"(h[1]), "r"(magic1));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[2]) : "r"(h[2]), "r"(magic0));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(h[3]) : "r"(h[3]), "r"(magic1));
  const half scale = __ushort_as_half(static_cast<uint16_t>(stats));
  const half bias = __ushort_as_half(static_cast<uint16_t>(stats >> 16));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    decoded[i] =
        __hfma2(*reinterpret_cast<const half2*>(&h[i]),
                __halves2half2(scale, scale), __halves2half2(bias, bias));
  }
}

#define GROUPED_MMA(C, A0, A1, B0, B1)                              \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};"                                  \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// One K16 step for every active row block. `nrb` is CTA-uniform; inactive A
// rows contribute zeros, which leaves the other rows' results unchanged.
template <int RB>
__device__ __forceinline__ void mma_row_blocks(float (&accum)[RB][8],
                                               const half* const (&a)[RB],
                                               const bool (&on)[RB], int nrb,
                                               int k16, const half2* decoded) {
  const auto* b = reinterpret_cast<const uint32_t*>(decoded);
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
      uint4 lo = make_uint4(0, 0, 0, 0);
      uint4 hi = make_uint4(0, 0, 0, 0);
      if (on[rb]) {
        lo = *reinterpret_cast<const uint4*>(a[rb] + k16 * 16);
        hi = *reinterpret_cast<const uint4*>(a[rb] + k16 * 16 + 8);
      }
      GROUPED_MMA(accum[rb], lo.x, lo.y, b[0], b[1]);
      GROUPED_MMA(accum[rb], lo.z, lo.w, b[2], b[3]);
      GROUPED_MMA(accum[rb], hi.x, hi.y, b[4], b[5]);
      GROUPED_MMA(accum[rb], hi.z, hi.w, b[6], b[7]);
    }
  }
}

// m8n8k4 f32 accumulator element i of this lane -> (row, column in quadpair).
__device__ __forceinline__ int acc_row(int i, int lane) {
  return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}
__device__ __forceinline__ int acc_col(int i, int lane) {
  return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}

// Deterministic, atomic-free routing plan. blockDim.x == kExperts.
__global__ void __launch_bounds__(kExperts)
    grouped_plan_kernel(const int32_t* __restrict__ ids, int routes, int cap,
                        int32_t* __restrict__ plan) {
  __shared__ int s_id[kMaxRoutes];
  __shared__ int s_ordinal[kMaxRoutes];
  __shared__ int s_count[kExperts];
  __shared__ int s_first[kExperts];
  __shared__ int s_warp_sum[kExperts / 32];
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  s_count[t] = 0;
  if (t < kMaxRoutes) {
    int id = t < routes ? ids[t] : -1;
    if (id < 0 || id >= kExperts) id = -1;  // invalid routes contribute zero
    s_id[t] = id;
  }
  __syncthreads();
  if (warp == 0) {
    for (int base = 0; base < routes; base += 32) {
      const int r = base + lane;
      const int id = r < routes ? s_id[r] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, id);
      const unsigned lower = peers & ((1u << lane) - 1u);
      int ordinal = -1;
      if (id >= 0) ordinal = s_count[id] + __popc(lower);
      __syncwarp();
      // The lowest lane of each id group advances that expert's count.
      if (id >= 0 && lower == 0) s_count[id] += __popc(peers);
      if (r < routes) s_ordinal[r] = ordinal;
      __syncwarp();
    }
  }
  __syncthreads();
  const int count = s_count[t];
  const int npacks = (count + cap - 1) / cap;
  int inclusive = npacks;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int value = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += value;
  }
  if (lane == 31) s_warp_sum[warp] = inclusive;
  __syncthreads();
  int warp_base = 0;
  for (int w = 0; w < warp; ++w) warp_base += s_warp_sum[w];
  const int first = warp_base + inclusive - npacks;
  s_first[t] = first;
  for (int p = 0; p < npacks; ++p) {
    plan[kPlanExperts + first + p] = t;
    plan[kPlanCounts + first + p] = min(cap, count - p * cap);
  }
  if (t == kExperts - 1) plan[0] = warp_base + inclusive;
  __syncthreads();
  if (t < routes) {
    const int id = s_id[t];
    if (id >= 0) {
      const int ordinal = s_ordinal[t];
      const int pack = s_first[id] + ordinal / cap;
      plan[kPlanRows + pack * kRowCap + ordinal % cap] = t;
    }
  }
}

template <int Format, int Split, int RB>
__global__ void __launch_bounds__(Split * 32)
    grouped_w13_kernel(const half* __restrict__ x,
                       const uint32_t* __restrict__ weights,
                       const uint8_t* __restrict__ metadata,
                       const int32_t* __restrict__ plan,
                       half* __restrict__ out) {
  extern __shared__ float partial[];  // [Split][RB * 8][32]
  constexpr int kRows = RB * 8;
  constexpr int kSteps = (kK / 16) / Split;
  static_assert(kSteps % 2 == 0, "warps must start on an even K16 step");
  const int pack = blockIdx.y;
  if (pack >= __ldg(plan)) return;
  const int expert = __ldg(plan + kPlanExperts + pack);
  const int count = __ldg(plan + kPlanCounts + pack);
  const int32_t* rows = plan + kPlanRows + pack * kRowCap;
  const int nrb = (count + 7) >> 3;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;

  const half* a[RB];
  bool on[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int r = rb * 8 + a_row;
    on[rb] = r < count;
    const int route = on[rb] ? __ldg(rows + r) : 0;
    a[rb] = x + static_cast<size_t>(route / kTopK) * kK;
  }
  const uint32_t* w = weights + static_cast<size_t>(expert) * (kK * kN13 / 8);
  constexpr int stats_bytes = (kK / 32) * kN13 * (Format == 2 ? 4 : 3);
  const uint8_t* st = metadata + static_cast<size_t>(expert) * stats_bytes;

  float accum[RB][8];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int i = 0; i < 8; ++i) accum[rb][i] = 0.f;
  }
#pragma unroll 2
  for (int s = 0; s < kSteps; s += 2) {
    const int group = warp * kSteps + s;  // even K16 step
    const size_t base =
        (static_cast<size_t>(tile) * (kK / 8) + group * 2) * 32 + packed_col;
    const uint32_t p0 = __ldcs(w + base);
    const uint32_t p1 = __ldcs(w + base + 32);
    const uint32_t p2 = __ldcs(w + base + 64);
    const uint32_t p3 = __ldcs(w + base + 96);
    // Both K16 halves of a g32 group reuse one metadata record.
    const uint32_t stats =
        read_awq_stats<Format>(st, group / 2, tile, packed_col, kN13);
    half2 d0[8];
    half2 d1[8];
    dequant_awq_u4x8(p0, stats, d0);
    dequant_awq_u4x8(p1, stats, d0 + 4);
    dequant_awq_u4x8(p2, stats, d1);
    dequant_awq_u4x8(p3, stats, d1 + 4);
    mma_row_blocks<RB>(accum, a, on, nrb, group, d0);
    mma_row_blocks<RB>(accum, a, on, nrb, group + 1, d1);
  }
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    if (rb < nrb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        partial[(warp * kRows + rb * 8 + acc_row(i, lane)) * 32 + quadpair * 8 +
                acc_col(i, lane)] = accum[rb][i];
      }
    }
  }
  __syncthreads();
  // Fixed split order from 0.f, as the QPN M1 reduction. Columns 2j/2j+1 of
  // the interleaved bank are gate/up of intermediate index tile*16 + j.
  for (int idx = threadIdx.x; idx < count * 16; idx += blockDim.x) {
    const int r = idx >> 4;
    const int j = idx & 15;
    float gate_sum = 0.f;
    float up_sum = 0.f;
#pragma unroll
    for (int s = 0; s < Split; ++s) {
      const float* p = partial + (s * kRows + r) * 32;
      gate_sum += p[2 * j];
      up_sum += p[2 * j + 1];
    }
    const half gate = __float2half(gate_sum);
    const half up = __float2half(up_sum);
    const float gate_f = __half2float(gate);
    const half silu = __float2half(gate_f / (1.f + expf(-gate_f)));
    out[static_cast<size_t>(__ldg(rows + r)) * kK2 + tile * 16 + j] =
        __hmul(silu, up);
  }
}

template <int Format, int RB, int Warps>
__global__ void __launch_bounds__(Warps * 32)
    grouped_w2_reduce_kernel(const half* __restrict__ intermediate,
                             const uint32_t* __restrict__ weights,
                             const uint8_t* __restrict__ metadata,
                             const int32_t* __restrict__ plan,
                             const float* __restrict__ topk,
                             half* __restrict__ out, int tokens) {
  __shared__ half values[kMaxRoutes][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int routes = tokens * kTopK;
  // Invalid routes (expert < 0 or >= 512) keep a zero contribution.
  for (int idx = threadIdx.x; idx < routes * 32; idx += blockDim.x)
    values[idx >> 5][idx & 31] = __float2half(0.f);
  __syncthreads();
  const int packs = __ldg(plan);
  const int quadpair = (lane >> 2) & 3;
  const int a_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int packed_col = quadpair * 8 + a_row;
  constexpr int stats_bytes = (kK2 / 32) * kN2 * (Format == 2 ? 4 : 3);
  for (int pack = warp; pack < packs; pack += Warps) {
    const int expert = __ldg(plan + kPlanExperts + pack);
    const int count = __ldg(plan + kPlanCounts + pack);
    const int32_t* rows = plan + kPlanRows + pack * kRowCap;
    const int nrb = (count + 7) >> 3;
    const half* a[RB];
    bool on[RB];
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      const int r = rb * 8 + a_row;
      on[rb] = r < count;
      const int route = on[rb] ? __ldg(rows + r) : 0;
      a[rb] = intermediate + static_cast<size_t>(route) * kK2;
    }
    const uint32_t* w = weights + static_cast<size_t>(expert) * (kK2 * kN2 / 8);
    const uint8_t* st = metadata + static_cast<size_t>(expert) * stats_bytes;
    float accum[RB][8];
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) accum[rb][i] = 0.f;
    }
#pragma unroll
    for (int group = 0; group < kK2 / 16; group += 2) {
      const size_t base =
          (static_cast<size_t>(tile) * (kK2 / 8) + group * 2) * 32 + packed_col;
      const uint32_t p0 = __ldcs(w + base);
      const uint32_t p1 = __ldcs(w + base + 32);
      const uint32_t p2 = __ldcs(w + base + 64);
      const uint32_t p3 = __ldcs(w + base + 96);
      const uint32_t stats =
          read_awq_stats<Format>(st, group / 2, tile, packed_col, kN2);
      half2 d0[8];
      half2 d1[8];
      dequant_awq_u4x8(p0, stats, d0);
      dequant_awq_u4x8(p1, stats, d0 + 4);
      dequant_awq_u4x8(p2, stats, d1);
      dequant_awq_u4x8(p3, stats, d1 + 4);
      mma_row_blocks<RB>(accum, a, on, nrb, group, d0);
      mma_row_blocks<RB>(accum, a, on, nrb, group + 1, d1);
    }
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      if (rb < nrb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = rb * 8 + acc_row(i, lane);
          if (r < count) {
            // Per-route FP16 materialization, as in the QPN M1 W2 kernel.
            values[__ldg(rows + r)][quadpair * 8 + acc_col(i, lane)] =
                __float2half(accum[rb][i]);
          }
        }
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < tokens * 32; idx += blockDim.x) {
    const int token = idx >> 5;
    const int c = idx & 31;
    float weighted = 0.f;
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      // Original router order and FP32 fmaf, then one FP16 rounding.
      weighted = fmaf(__ldg(topk + token * kTopK + slot),
                      __half2float(values[token * kTopK + slot][c]), weighted);
    }
    out[static_cast<size_t>(token) * kN2 + tile * 32 + c] =
        __float2half(weighted);
  }
}

template <int Format, int Split, int RB>
void launch_w13(const at::Tensor& input, const at::Tensor& w13,
                const at::Tensor& s13, const at::Tensor& plan,
                at::Tensor& intermediate, int routes, cudaStream_t stream) {
  constexpr size_t smem = static_cast<size_t>(Split) * RB * 8 * 32 * 4;
  static_assert(smem <= 48 * 1024, "W13 partials exceed default smem");
  grouped_w13_kernel<Format, Split, RB>
      <<<dim3(kTiles13, routes), Split * 32, smem, stream>>>(
          reinterpret_cast<const half*>(input.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w13.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s13.const_data_ptr()),
          plan.const_data_ptr<int32_t>(),
          reinterpret_cast<half*>(intermediate.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Format, int RB>
void launch_w2(const at::Tensor& intermediate, const at::Tensor& w2,
               const at::Tensor& s2, const at::Tensor& plan,
               const at::Tensor& topk, at::Tensor& out, int tokens,
               cudaStream_t stream) {
  // One row block keeps few accumulators: use 32 warps per column tile.
  constexpr int kWarps = RB == 1 ? 32 : 16;
  grouped_w2_reduce_kernel<Format, RB, kWarps>
      <<<kTiles2, kWarps * 32, 0, stream>>>(
          reinterpret_cast<const half*>(intermediate.const_data_ptr()),
          reinterpret_cast<const uint32_t*>(w2.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(s2.const_data_ptr()),
          plan.const_data_ptr<int32_t>(), topk.const_data_ptr<float>(),
          reinterpret_cast<half*>(out.mutable_data_ptr()), tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int W13Format, int W2Format, int RB>
void launch_all(const at::Tensor& input, const at::Tensor& w13,
                const at::Tensor& s13, const at::Tensor& w2,
                const at::Tensor& s2, const at::Tensor& topk,
                const at::Tensor& plan, at::Tensor& intermediate,
                at::Tensor& out, int tokens, int split, cudaStream_t stream) {
  const int routes = tokens * kTopK;
  if (split == 8) {
    launch_w13<W13Format, 8, RB>(input, w13, s13, plan, intermediate, routes,
                                 stream);
  } else {
    launch_w13<W13Format, 16, RB>(input, w13, s13, plan, intermediate, routes,
                                  stream);
  }
  launch_w2<W2Format, RB>(intermediate, w2, s2, plan, topk, out, tokens,
                          stream);
}

template <int W13Format, int W2Format>
void dispatch_rb(int rb, const at::Tensor& input, const at::Tensor& w13,
                 const at::Tensor& s13, const at::Tensor& w2,
                 const at::Tensor& s2, const at::Tensor& topk,
                 const at::Tensor& plan, at::Tensor& intermediate,
                 at::Tensor& out, int tokens, int split, cudaStream_t stream) {
  switch (rb) {
    case 1:
      launch_all<W13Format, W2Format, 1>(input, w13, s13, w2, s2, topk, plan,
                                         intermediate, out, tokens, split,
                                         stream);
      break;
    case 2:
      launch_all<W13Format, W2Format, 2>(input, w13, s13, w2, s2, topk, plan,
                                         intermediate, out, tokens, split,
                                         stream);
      break;
    default:
      launch_all<W13Format, W2Format, 3>(input, w13, s13, w2, s2, topk, plan,
                                         intermediate, out, tokens, split,
                                         stream);
      break;
  }
}

void check_tensor(const at::Tensor& tensor, at::ScalarType dtype,
                  at::IntArrayRef shape, const at::Device& device,
                  const char* name, uintptr_t alignment = 16) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device,
              "awq_grouped_decode: ", name,
              " must be CUDA on the input device");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.sizes() == shape &&
                  tensor.is_contiguous(),
              "awq_grouped_decode: ", name, " dtype/shape/contiguity mismatch");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % alignment == 0,
      "awq_grouped_decode: ", name, " alignment mismatch");
}

void awq_moe_grouped_decode_sm70_out(at::Tensor out, at::Tensor intermediate,
                                     at::Tensor plan, const at::Tensor& input,
                                     const at::Tensor& w13,
                                     const at::Tensor& s13,
                                     const at::Tensor& w2, const at::Tensor& s2,
                                     const at::Tensor& ids,
                                     const at::Tensor& topk, int64_t split) {
  const auto device = input.device();
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= kMaxTokens,
      "awq_grouped_decode: input must be [1..20, 2560]");
  const int tokens = static_cast<int>(input.size(0));
  const int routes = tokens * kTopK;
  TORCH_CHECK(split == 0 || split == 8 || split == 16,
              "awq_grouped_decode: split must be 0 (default 16), 8 or 16");
  check_tensor(input, at::kHalf, {tokens, kK}, device, "input");
  check_tensor(out, at::kHalf, {tokens, kN2}, device, "output");
  check_tensor(intermediate, at::kHalf, {routes, kK2}, device, "intermediate");
  TORCH_CHECK(plan.is_cuda() && plan.device() == device &&
                  plan.scalar_type() == at::kInt && plan.is_contiguous() &&
                  plan.numel() >= kPlanInts,
              "awq_grouped_decode: plan must be contiguous CUDA int32 with at "
              "least ",
              kPlanInts, " elements");
  check_tensor(w13, at::kInt, {kExperts, kK, kN13 / 8}, device, "W13");
  check_tensor(w2, at::kInt, {kExperts, kK2, kN2 / 8}, device, "W2");
  const bool compact = s13.scalar_type() == at::kByte;
  if (compact) {
    check_tensor(s13, at::kByte, {kExperts, kK / 32, kN13, 3}, device,
                 "W13 metadata");
    check_tensor(s2, at::kByte, {kExperts, kK2 / 32, kN2, 3}, device,
                 "W2 metadata");
  } else {
    check_tensor(s13, at::kInt, {kExperts, kK / 32, kN13}, device,
                 "W13 metadata");
    check_tensor(s2, at::kInt, {kExperts, kK2 / 32, kN2}, device,
                 "W2 metadata");
  }
  check_tensor(ids, at::kInt, {tokens, kTopK}, device, "expert IDs", 4);
  check_tensor(topk, at::kFloat, {tokens, kTopK}, device, "router weights", 4);
  for (const auto* tensor : {&input, &w13, &s13, &w2, &s2, &ids, &topk}) {
    at::assert_no_overlap(out, *tensor);
    at::assert_no_overlap(intermediate, *tensor);
    at::assert_no_overlap(plan, *tensor);
  }
  at::assert_no_overlap(out, intermediate);
  at::assert_no_overlap(out, plan);
  at::assert_no_overlap(intermediate, plan);
  const c10::cuda::CUDAGuard guard(device);
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "awq_grouped_decode requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Each token routes to an expert at most once, so an expert has at most
  // `tokens` rows: ceil(tokens / 8) row blocks read its weights exactly once.
  const int rb = (tokens + 7) / 8;
  grouped_plan_kernel<<<1, kExperts, 0, stream>>>(
      ids.const_data_ptr<int32_t>(), routes, rb * 8,
      plan.mutable_data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int w13_split = split == 8 ? 8 : 16;
  if (compact) {
    dispatch_rb<1, 0>(rb, input, w13, s13, w2, s2, topk, plan, intermediate,
                      out, tokens, w13_split, stream);
  } else {
    dispatch_rb<2, 2>(rb, input, w13, s13, w2, s2, topk, plan, intermediate,
                      out, tokens, w13_split, stream);
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "awq_moe_grouped_decode_sm70_out(Tensor(a!) out, Tensor(b!) "
      "intermediate, Tensor(c!) plan, Tensor input, Tensor w13, Tensor s13, "
      "Tensor w2, Tensor s2, Tensor ids, Tensor topk, int split) -> ()");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("awq_moe_grouped_decode_sm70_out", &awq_moe_grouped_decode_sm70_out);
}
