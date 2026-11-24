// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
// Direct device E4M3/FP16 history: FP16 QK operands, FP32 QK/PV,
// probability and merge. No protected-hot ownership or staging is required.
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/extension.h>

namespace device_qsa {
constexpr int D = 256, H = 6, C = 64, PAD = 264;
struct Args {
  const half* q;
  const unsigned char* history;
  const float* scales;
  const int* indices;
  const int* table;
  const int* requests;
  const int64_t* positions;
  const int* lengths;
  const half* gate;
  const half* hot;      // [hot_tokens, 2, D] FP16
  const int* resolved;  // [M, width] hot token, -1 host, -2 invalid
  const int32_t* page_slots;
  const int32_t* tags;
  int hot_pages;
  int64_t page_slots_len;
  half* out;
  float* out_f32;
  float* lse;
  float* partial;
  float* state;
  int M, blocks, page, width, table_width, num_requests, splits, tiles;
  // merge over one window (FP16 history): window-normalized FP32 rows
  // and per-head (max, sum) instead of the gated output.
  float* wout;
  float* wstate;
  int64_t qs0, qs1, is0, ts0, os0, os1, gs0, gs1, rs0, ls0;
  // GPU DCP uses separate strided K/V planes and per-layer scales.
  const unsigned char* paged_v;
  int64_t kb, kt, vb, vt;
  float k_scale, v_scale;
};

__device__ __forceinline__ float fast_exp2(float x) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

__device__ __forceinline__ float e4m3(unsigned c) {
  const unsigned sign = (c & 128u) << 24;
  const unsigned magnitude = c & 127u;
  if (magnitude == 127u) return __uint_as_float(sign | 0x7fc00000u);
  const unsigned normal = (magnitude << 20) + (120u << 23);
  const float subnormal = float(magnitude & 7u) * 0x1p-9f;
  const unsigned bits = magnitude < 8u ? __float_as_uint(subnormal) : normal;
  return __uint_as_float(bits | sign);
}

__device__ __forceinline__ void mma(float (&d)[8], uint32_t a0, uint32_t a1,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// B row-major: lane holds B[k = lane & 3][n = (lane & 16 ? 4 : 0) + i] of its
// quad pair's 8 columns (i = 0..3, pairs in b0, b1); A and D as in mma().
// lut: the 256 E4M3 code values (e4m3(code)), in shared memory.
__device__ __forceinline__ uint4 decode8_raw(const uint2 packed, float scale,
                                             const float* lut) {
  uint4 result;
  unsigned* words = reinterpret_cast<unsigned*>(&result);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned p = j < 2 ? packed.x : packed.y;
    const unsigned code0 = (p >> (16 * (j & 1))) & 255u;
    const unsigned code1 = (p >> (16 * (j & 1) + 8)) & 255u;
    const half lo = __float2half_rn(__fmul_rn(lut[code0], scale));
    const half hi = __float2half_rn(__fmul_rn(lut[code1], scale));
    words[j] =
        unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi)) << 16);
  }
  return result;
}

// Read one 8-value chunk of token K (kv=0) or V (kv=1): from the protected
// hot cache when resolved there, else from the authoritative history, which
// may be pinned host memory mapped into the device address space (UVA).
template <bool HOT>
__device__ __forceinline__ uint4 load_kv(const Args& a, int64_t slot, int hot,
                                         int kv, int dim) {
  if (slot < 0) return make_uint4(0, 0, 0, 0);
  if constexpr (HOT) {
    if (hot >= 0) {
      return __ldg(reinterpret_cast<const uint4*>(
          a.hot + (int64_t(hot) * 2 + kv) * D + dim));
    }
  }
  const int block = slot / a.page, off = slot % a.page;
  const int64_t element = ((int64_t(block) * 2 + kv) * a.page + off) * D + dim;
  return __ldg(reinterpret_cast<const uint4*>(
      reinterpret_cast<const half*>(a.history) + element));
}

// Keep the FP16 kernel names used by existing profiler readouts.
template <bool, bool HOT, bool = false>
__global__ __launch_bounds__(256) void partial(Args a) {
  const int row = blockIdx.y, split = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[8][PAD];
  __shared__ __align__(16) half kv[C][PAD];
  __shared__ float scores[4][8][C];
  __shared__ int64_t slots[C];
  __shared__ int hots[C];
  __shared__ float alpha[8];
  const int req = a.requests[row];
  const int64_t position = a.positions[row];
  const int length = req >= 0 && req < a.num_requests ? a.lengths[req] : 0;
  const int head = t / 32, d8 = (t % 32) * 8;
  const uint4 qv = head < H ? __ldg(reinterpret_cast<const uint4*>(
                                  a.q + row * a.qs0 + head * a.qs1 + d8))
                            : make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(&query[head][d8]) = qv;
  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  constexpr int kChunks = C * 32 / 256;
  // Running softmax state of head `warp` (FP32), and this thread's
  // dimension t of every head's unnormalized output. With one tile per CTA
  // this is exactly the single-tile arithmetic.
  float running_max = __int_as_float(0xff800000), running_sum = 0.f;
  float result[H] = {};
  for (int tile = 0; tile < a.tiles; ++tile) {
    const int base = (split * a.tiles + tile) * C;
    if (base >= a.width) break;
    __syncthreads();  // the previous tile is done with slots, kv and scores
    if (t < C) {
      const int col = base + t;
      const int logical = col < a.width ? a.indices[row * a.is0 + col] : -1;
      int64_t slot = -1;
      if (req >= 0 && req < a.num_requests && logical >= 0 &&
          logical <= position && logical < length &&
          logical / a.page < a.table_width) {
        const int physical = a.table[req * a.ts0 + logical / a.page];
        if (physical >= 0 && physical < a.blocks)
          slot = int64_t(physical) * a.page + logical % a.page;
      }
      slots[t] = slot;
      int hot = HOT && slot >= 0 ? a.resolved[row * a.rs0 + col] : -1;
      if (HOT && hot == -1 && slot >= 0 && a.page_slots != nullptr &&
          a.tags != nullptr) {
        const int64_t p4 = slot >> 2;
        if (p4 < a.page_slots_len) {
          const int32_t reg = a.page_slots[p4];
          if (reg >= 0 && reg < a.hot_pages &&
              a.tags[reg] == static_cast<int32_t>(p4)) {
            hot = (reg << 2) | static_cast<int>(slot & 3);
          }
        }
      }
      hots[t] = hot;
    }
    __syncthreads();
    for (int v = t; v < C * 32; v += 256) {
      const int token = v / 32, dim = (v % 32) * 8;
      *reinterpret_cast<uint4*>(&kv[token][dim]) =
          load_kv<HOT>(a, slots[token], hots[token], 0, dim);
    }
    __syncthreads();
    // Issue V loads ahead of QK and softmax so global/PCIe latency overlaps
    // tensor-core QK and score reduction (including FP16 hot-cache readers).
    uint4 raw[kChunks];
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
      raw[i] = load_kv<HOT>(a, slots[token], hots[token], 1, dim);
    }
    float acc[8] = {};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int k = kw * 64 + s * 16;
      const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
      const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
      const uint4 k0 = *reinterpret_cast<const uint4*>(&kv[nr][k]);
      const uint4 k1 = *reinterpret_cast<const uint4*>(&kv[nr][k + 8]);
      mma(acc, q0.x, q0.y, k0.x, k0.y);
      mma(acc, q0.z, q0.w, k0.z, k0.w);
      mma(acc, q1.x, q1.y, k1.x, k1.y);
      mma(acc, q1.z, q1.w, k1.z, k1.w);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                      ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
      scores[kw][h][col] = acc[e];
    }
    __syncthreads();
    for (int v = t; v < 8 * C; v += 256) {
      const int h = v / C, k = v % C;
      const float sum =
          ((scores[0][h][k] + scores[1][h][k]) + scores[2][h][k]) +
          scores[3][h][k];
      scores[0][h][k] =
          slots[k] >= 0 ? sum * (0.0625f * 1.4426950408889634f) : -1e20f;
    }
    __syncthreads();
    float maximum = fmaxf(scores[0][warp][lane], scores[0][warp][lane + 32]);
#pragma unroll
    for (int s = 16; s; s >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
    maximum = fmaxf(running_max, maximum);
    const float p0 =
        slots[lane] >= 0 ? fast_exp2(scores[0][warp][lane] - maximum) : 0.f;
    const float p1 = slots[lane + 32] >= 0
                         ? fast_exp2(scores[0][warp][lane + 32] - maximum)
                         : 0.f;
    scores[0][warp][lane] = p0;
    scores[0][warp][lane + 32] = p1;
    float sum = p0 + p1;
#pragma unroll
    for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    const float scale = fast_exp2(running_max - maximum);
    running_sum = __fmaf_rn(running_sum, scale, sum);
    running_max = maximum;
    if (lane == 0) alpha[warp] = scale;
    __syncthreads();
    // The shared key tile is retired. Reuse its storage for values, keeping
    // FP32 probabilities and PV FMAs instead of rounding probabilities to
    // half.
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
      *reinterpret_cast<uint4*>(&kv[token][dim]) = raw[i];
    }
    __syncthreads();
    // Thread t owns dimension t of every head: one value load serves all
    // six heads. Each output keeps its sequential FP32 FMA chain over k.
#pragma unroll
    for (int h = 0; h < H; ++h) result[h] *= alpha[h];
#pragma unroll 4
    for (int k = 0; k < C; ++k) {
      const float value = __half2float(kv[k][t]);
#pragma unroll
      for (int h = 0; h < H; ++h)
        result[h] = __fmaf_rn(scores[0][h][k], value, result[h]);
    }
  }
  if (lane == 0 && warp < H) {
    a.state[((row * H + warp) * a.splits + split) * 2] = running_max;
    a.state[((row * H + warp) * a.splits + split) * 2 + 1] = running_sum;
  }
#pragma unroll
  for (int h = 0; h < H; ++h)
    a.partial[((row * H + h) * a.splits + split) * D + t] = result[h];
}

// One (row, head): combine the split states and partial outputs. Shared
// scratch: weights/denom_parts[64], scalars[2].
template <int NH = H, bool DEVICE_PAGED = false>
__device__ __forceinline__ void merge_head(const Args& a, int row, int head,
                                           int t, float* weights,
                                           float* denom_parts, float* scalars) {
  const int64_t state_base = int64_t(row * NH + head) * a.splits * 2;
  if (t < 32) {
    const int s0 = t, s1 = t + 32;
    const float m0 = s0 < a.splits && a.state[state_base + s0 * 2 + 1] > 0
                         ? a.state[state_base + s0 * 2]
                         : -1e20f;
    const float m1 = s1 < a.splits && a.state[state_base + s1 * 2 + 1] > 0
                         ? a.state[state_base + s1 * 2]
                         : -1e20f;
    float maximum = fmaxf(m0, m1);
#pragma unroll
    for (int s = 16; s; s >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
    if (!t) scalars[0] = maximum;
  }
  __syncthreads();
  if (t < 64) {
    const bool valid = t < a.splits && a.state[state_base + t * 2 + 1] > 0;
    const float weight =
        valid ? fast_exp2(a.state[state_base + t * 2] - scalars[0]) : 0.f;
    weights[t] = weight;
    denom_parts[t] = valid ? weight * a.state[state_base + t * 2 + 1] : 0.f;
  }
  __syncthreads();
  if (t < 32) {
    float sum = denom_parts[t] + denom_parts[t + 32];
#pragma unroll
    for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    if (!t) scalars[1] = sum;
  }
  __syncthreads();
  float numerator = 0.f;
  for (int s = 0; s < a.splits; ++s)
    numerator = __fmaf_rn(weights[s],
                          a.partial[((row * NH + head) * a.splits + s) * D + t],
                          numerator);
  const float denominator = scalars[1];
  float result = denominator > 0 ? __fdiv_rn(numerator, denominator) : 0.f;
  if constexpr (DEVICE_PAGED) result *= a.v_scale;
  if (a.wout) {
    a.wout[int64_t(row * NH + head) * D + t] = result;
    if (!t) {
      a.wstate[int64_t(row * NH + head) * 2] = scalars[0];
      a.wstate[int64_t(row * NH + head) * 2 + 1] = denominator;
    }
    return;
  }
  if (a.lse && !t) {
    a.lse[row * a.ls0 + head] =
        denominator > 0.f ? (scalars[0] + log2f(denominator)) : -CUDART_INF_F;
  }
  if (a.gate) {
    result = __half2float(__float2half_rn(result));
    const float gate = __half2float(a.gate[row * a.gs0 + head * a.gs1 + t]);
    result *= __fdiv_rn(1.f, 1.f + fast_exp2(-gate * 1.4426950408889634f));
  }
  if (a.out_f32) {
    a.out_f32[row * a.os0 + head * a.os1 + t] = result;
  } else {
    a.out[row * a.os0 + head * a.os1 + t] = __float2half_rn(result);
  }
}

// Prefill over staged E4M3 history (one 64-key tile per CTA, no hot cache):
// the K and V tiles stay E4M3 in shared memory and are decoded where they are
// used, with the same arithmetic as `partial` (half(e4m3 * scale), FP32
// QK/PV). Half the tile footprint fits three CTAs per SM.
constexpr int KP = 272;  // E4M3 tile row stride (bytes)
template <bool DEVICE_PAGED = false>
__global__ __launch_bounds__(256, 3) void partial_fp8_lite(Args a) {
  const int row = blockIdx.y, split = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[8][PAD];
  __shared__ __align__(16) unsigned char tile[C][KP];
  __shared__ float scores[4][8][C];
  __shared__ int64_t slots[C];
  __shared__ float kscale[C], vscale[C];
  __shared__ float lut[256];
  lut[t] = e4m3(t);
  const int req = a.requests[row];
  if (t < C) {
    const int col = split * C + t;
    const int logical = col < a.width ? a.indices[row * a.is0 + col] : -1;
    int64_t slot = -1;
    // DCP selection is already causal in global token space.
    if (req >= 0 && req < a.num_requests && logical >= 0 &&
        logical / a.page < a.table_width &&
        (DEVICE_PAGED ||
         (logical <= a.positions[row] && logical < a.lengths[req]))) {
      const int physical = a.table[req * a.ts0 + logical / a.page];
      if (physical >= 0 && physical < a.blocks)
        slot = int64_t(physical) * a.page + logical % a.page;
    }
    slots[t] = slot;
    if constexpr (!DEVICE_PAGED) {
      kscale[t] = slot >= 0 ? __ldg(a.scales + slot * 2) : 0.f;
      vscale[t] = slot >= 0 ? __ldg(a.scales + slot * 2 + 1) : 0.f;
    }
  }
  const int head = t / 32, d8 = (t % 32) * 8;
  const uint4 qv = head < H ? __ldg(reinterpret_cast<const uint4*>(
                                  a.q + row * a.qs0 + head * a.qs1 + d8))
                            : make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(&query[head][d8]) = qv;
  __syncthreads();
  constexpr int kChunks = C * 32 / 256;
  uint2 vraw[kChunks];
#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
    const int64_t slot = slots[token];
    uint2 kraw = make_uint2(0, 0);
    vraw[i] = make_uint2(0, 0);
    if (slot >= 0) {
      const int block = slot / a.page, off = slot % a.page;
      if constexpr (DEVICE_PAGED) {
        kraw = __ldg(reinterpret_cast<const uint2*>(a.history + block * a.kb +
                                                    off * a.kt + dim));
        vraw[i] = __ldg(reinterpret_cast<const uint2*>(
            a.paged_v + block * a.vb + off * a.vt + dim));
      } else {
        const int64_t base = (int64_t(block) * 2 * a.page + off) * D + dim;
        kraw = __ldg(reinterpret_cast<const uint2*>(a.history + base));
        vraw[i] = __ldg(reinterpret_cast<const uint2*>(a.history + base +
                                                       int64_t(a.page) * D));
      }
    }
    *reinterpret_cast<uint2*>(&tile[token][dim]) = kraw;
  }
  __syncthreads();
  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const float ks = DEVICE_PAGED ? 1.f : kscale[nr];
  float acc[8] = {};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int k = kw * 64 + s * 16;
    const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
    const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
    const uint4 k0 =
        decode8_raw(*reinterpret_cast<const uint2*>(&tile[nr][k]), ks, lut);
    const uint4 k1 =
        decode8_raw(*reinterpret_cast<const uint2*>(&tile[nr][k + 8]), ks, lut);
    mma(acc, q0.x, q0.y, k0.x, k0.y);
    mma(acc, q0.z, q0.w, k0.z, k0.w);
    mma(acc, q1.x, q1.y, k1.x, k1.y);
    mma(acc, q1.z, q1.w, k1.z, k1.w);
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                    ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
    scores[kw][h][col] = acc[e];
  }
  __syncthreads();
  float qk_scale = 0.0625f * 1.4426950408889634f;
  if constexpr (DEVICE_PAGED) qk_scale *= a.k_scale;
  for (int v = t; v < 8 * C; v += 256) {
    const int h = v / C, k = v % C;
    const float sum = ((scores[0][h][k] + scores[1][h][k]) + scores[2][h][k]) +
                      scores[3][h][k];
    scores[0][h][k] = slots[k] >= 0 ? sum * qk_scale : -1e20f;
  }
  __syncthreads();
  float maximum = fmaxf(scores[0][warp][lane], scores[0][warp][lane + 32]);
#pragma unroll
  for (int s = 16; s; s >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
  const float p0 =
      slots[lane] >= 0 ? fast_exp2(scores[0][warp][lane] - maximum) : 0.f;
  const float p1 = slots[lane + 32] >= 0
                       ? fast_exp2(scores[0][warp][lane + 32] - maximum)
                       : 0.f;
  scores[0][warp][lane] = p0;
  scores[0][warp][lane + 32] = p1;
  float sum = p0 + p1;
#pragma unroll
  for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
  if (lane == 0 && warp < H) {
    a.state[((row * H + warp) * a.splits + split) * 2] = maximum;
    a.state[((row * H + warp) * a.splits + split) * 2 + 1] = sum;
  }
  __syncthreads();  // the K tile is retired
#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
    *reinterpret_cast<uint2*>(&tile[token][dim]) = vraw[i];
  }
  __syncthreads();
  float result[H] = {};
#pragma unroll 4
  for (int k = 0; k < C; ++k) {
    float value = lut[tile[k][t]];
    if constexpr (!DEVICE_PAGED)
      value = __half2float(__float2half_rn(__fmul_rn(value, vscale[k])));
#pragma unroll
    for (int h = 0; h < H; ++h)
      result[h] = __fmaf_rn(scores[0][h][k], value, result[h]);
  }
#pragma unroll
  for (int h = 0; h < H; ++h)
    a.partial[((row * H + h) * a.splits + split) * D + t] = result[h];
}

template <int NH = H, bool DEVICE_PAGED = false>
__global__ void merge_n(Args a) {
  __shared__ float weights[64], denom_parts[64], scalars[2];
  merge_head<NH, DEVICE_PAGED>(a, blockIdx.x / NH, blockIdx.x % NH, threadIdx.x,
                               weights, denom_parts, scalars);
}
}  // namespace device_qsa

void device_qsa_out(torch::Tensor q, torch::Tensor history,
                    torch::Tensor scales, torch::Tensor indices,
                    torch::Tensor table, torch::Tensor requests,
                    torch::Tensor positions, torch::Tensor lengths,
                    torch::Tensor out, std::optional<torch::Tensor> gate,
                    torch::Tensor partial, torch::Tensor state,
                    std::optional<torch::Tensor> hot = std::nullopt,
                    std::optional<torch::Tensor> resolved = std::nullopt,
                    int64_t tiles = 1,
                    std::optional<torch::Tensor> window_out = std::nullopt,
                    std::optional<torch::Tensor> window_state = std::nullopt,
                    std::optional<torch::Tensor> page_slots = std::nullopt,
                    std::optional<torch::Tensor> tags = std::nullopt,
                    std::optional<torch::Tensor> lse = std::nullopt) {
  if (q.dim() == 3 && q.size(1) > 6 && q.size(1) % 6 == 0 && !window_out) {
    const int64_t total_heads = q.size(1);
    for (int64_t h0 = 0; h0 < total_heads; h0 += 6) {
      std::optional<torch::Tensor> g_slice;
      if (gate) g_slice = gate->narrow(1, h0, 6);
      std::optional<torch::Tensor> lse_slice;
      if (lse) lse_slice = lse->narrow(1, h0, 6);
      device_qsa_out(q.narrow(1, h0, 6), history, scales, indices, table,
                     requests, positions, lengths, out.narrow(1, h0, 6),
                     g_slice, partial, state, hot, resolved, tiles, window_out,
                     window_state, page_slots, tags, lse_slice);
    }
    return;
  }
  const c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
              q.size(0) > 0 && q.size(0) <= 64 && q.size(1) == 6 &&
              q.size(2) == 256 && q.stride(2) == 1);
  TORCH_CHECK(history.scalar_type() == at::kHalf && history.dim() == 5 &&
              history.size(1) == 2 && history.size(3) == 1 &&
              history.size(4) == 256 && history.is_contiguous());
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous() &&
              scales.numel() == 2);
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.dim() == 2 &&
              indices.size(0) == q.size(0) && indices.stride(1) == 1 &&
              indices.size(1) > 0 && indices.size(1) <= 4096);
  TORCH_CHECK(table.scalar_type() == at::kInt && table.dim() == 2 &&
              table.stride(1) == 1 && requests.scalar_type() == at::kInt &&
              positions.scalar_type() == at::kLong &&
              lengths.scalar_type() == at::kInt);
  TORCH_CHECK(requests.numel() == q.size(0) && positions.numel() == q.size(0) &&
              lengths.numel() == table.size(0) && requests.is_contiguous() &&
              positions.is_contiguous() && lengths.is_contiguous());
  const bool out_is_f32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK(
      out.sizes() == q.sizes() &&
      (out.scalar_type() == at::kHalf || out_is_f32) && out.stride(2) == 1 &&
      (!gate || (gate->sizes() == q.sizes() &&
                 gate->scalar_type() == at::kHalf && gate->stride(2) == 1)));
  if (lse) {
    TORCH_CHECK(lse->device() == q.device() &&
                lse->scalar_type() == at::kFloat && lse->dim() == 2 &&
                lse->size(0) == q.size(0) && lse->size(1) == q.size(1) &&
                lse->stride(1) == 1);
  }
  for (const auto& tensor : {history, scales, indices, table, requests,
                             positions, lengths, out, partial, state})
    TORCH_CHECK(tensor.device() == q.device());
  if (gate) TORCH_CHECK(gate->device() == q.device());
  TORCH_CHECK(hot.has_value() == resolved.has_value());
  if (hot) {
    TORCH_CHECK(hot->is_cuda() && hot->device() == q.device() &&
                hot->scalar_type() == at::kHalf && hot->dim() == 3 &&
                hot->size(1) == 2 && hot->size(2) == 256 &&
                hot->is_contiguous());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(hot->data_ptr()) % 16 == 0,
                "QSA hot storage requires aligned vector loads");
    TORCH_CHECK(resolved->device() == q.device() &&
                resolved->scalar_type() == at::kInt && resolved->dim() == 2 &&
                resolved->size(0) >= q.size(0) &&
                resolved->size(1) >= indices.size(1) &&
                resolved->stride(1) == 1);
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
              q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0);
  auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 0);
  device_qsa::Args a{};
  a.q = reinterpret_cast<const half*>(q.data_ptr());
  a.history = reinterpret_cast<const unsigned char*>(history.data_ptr());
  a.indices = indices.data_ptr<int>();
  a.table = table.data_ptr<int>();
  a.requests = requests.data_ptr<int>();
  a.positions = positions.data_ptr<int64_t>();
  a.lengths = lengths.data_ptr<int>();
  a.gate = gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr;
  a.hot = hot ? reinterpret_cast<const half*>(hot->data_ptr()) : nullptr;
  a.resolved = resolved ? resolved->data_ptr<int>() : nullptr;
  a.rs0 = resolved ? resolved->stride(0) : 0;
  if (hot && page_slots && tags && page_slots->is_cuda() &&
      page_slots->device() == q.device() &&
      page_slots->scalar_type() == at::kInt && page_slots->is_contiguous() &&
      tags->is_cuda() && tags->device() == q.device() &&
      tags->scalar_type() == at::kInt && tags->is_contiguous() &&
      tags->numel() == hot->size(0) / 4) {
    a.page_slots = page_slots->data_ptr<int32_t>();
    a.tags = tags->data_ptr<int32_t>();
    a.hot_pages = static_cast<int>(tags->numel());
    a.page_slots_len = page_slots->numel();
  }
  a.out = out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr());
  a.out_f32 = out_is_f32 ? out.data_ptr<float>() : nullptr;
  a.lse = lse ? lse->data_ptr<float>() : nullptr;
  a.ls0 = lse ? lse->stride(0) : 0;
  a.partial = partial.data_ptr<float>();
  a.state = state.data_ptr<float>();
  TORCH_CHECK(window_out.has_value() == window_state.has_value());
  if (window_out) {
    TORCH_CHECK(
        window_out->scalar_type() == at::kFloat &&
            window_state->scalar_type() == at::kFloat &&
            window_out->is_contiguous() && window_state->is_contiguous() &&
            window_out->numel() == q.size(0) * 6 * 256 &&
            window_state->numel() == q.size(0) * 6 * 2,
        "QSA window merge needs FP32 [M, 6, 256] rows and [M, 6, 2] state");
  }
  a.wout = window_out ? window_out->data_ptr<float>() : nullptr;
  a.wstate = window_state ? window_state->data_ptr<float>() : nullptr;
  a.M = q.size(0);
  a.blocks = history.size(0);
  a.page = history.size(2);
  a.width = indices.size(1);
  a.table_width = table.size(1);
  a.num_requests = table.size(0);
  // Keys per CTA: `tiles` 64-key tiles with an online softmax (1: the
  // single-tile arithmetic of the decode readers).
  a.tiles = static_cast<int>(tiles);
  TORCH_CHECK(a.tiles >= 1);
  a.splits = (a.width + 64 * a.tiles - 1) / (64 * a.tiles);
  TORCH_CHECK(partial.scalar_type() == at::kFloat &&
              state.scalar_type() == at::kFloat && partial.is_contiguous() &&
              state.is_contiguous() &&
              partial.numel() >= a.M * 6 * a.splits * 256 &&
              state.numel() >= a.M * 6 * a.splits * 2);
  a.qs0 = q.stride(0);
  a.qs1 = q.stride(1);
  a.is0 = indices.stride(0);
  a.ts0 = table.stride(0);
  a.os0 = out.stride(0);
  a.os1 = out.stride(1);
  a.gs0 = gate ? gate->stride(0) : 0;
  a.gs1 = gate ? gate->stride(1) : 0;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(a.splits, a.M);
  if (hot) {
    device_qsa::partial<false, true><<<grid, 256, 0, stream>>>(a);
  } else {
    device_qsa::partial<false, false><<<grid, 256, 0, stream>>>(a);
  }
  device_qsa::merge_n<6><<<a.M * 6, 256, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Copy a contiguous run of authoritative host pages with the copy engine.
// History is a CUDA-addressable view of pinned storage; using its host alias
// avoids treating that view as an ordinary device-to-device tensor copy.
// Return false without writes for actual device history (the gather fallback).
bool stage_host_run(torch::Tensor history, torch::Tensor /*scales*/,
                    torch::Tensor out, torch::Tensor /*out_scales*/,
                    int64_t source_page, int64_t target_page, int64_t pages) {
  TORCH_CHECK(history.is_cuda() && out.is_cuda() &&
                  history.device() == out.device() && history.is_contiguous() &&
                  out.is_contiguous(),
              "QSA stage history must have contiguous same-device views");
  TORCH_CHECK(history.dim() == 5 && out.dim() == 5 && history.size(1) == 2 &&
                  history.size(2) > 0 && history.size(2) % 4 == 0 &&
                  history.size(3) == 1 && history.size(4) == 256 &&
                  history.sizes().slice(1) == out.sizes().slice(1) &&
                  history.scalar_type() == out.scalar_type() &&
                  history.scalar_type() == at::kHalf,
              "QSA stage history requires matching page-major FP16 views");
  TORCH_CHECK(source_page >= 0 && source_page <= history.size(0) &&
                  target_page >= 0 && target_page <= out.size(0) &&
                  pages >= 0 && pages <= history.size(0) - source_page &&
                  pages <= out.size(0) - target_page,
              "QSA stage page range is outside its source or destination");
  if (!pages) return true;
  const c10::cuda::CUDAGuard guard(out.device());
  cudaPointerAttributes source{}, destination{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&source, history.data_ptr()));
  if (source.type != cudaMemoryTypeHost || source.hostPointer == nullptr)
    return false;
  C10_CUDA_CHECK(cudaPointerGetAttributes(&destination, out.data_ptr()));
  TORCH_CHECK(destination.type == cudaMemoryTypeDevice,
              "QSA stage destination must be device memory");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t page_bytes = history.stride(0) * history.element_size();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      static_cast<char*>(out.data_ptr()) + target_page * page_bytes,
      static_cast<const char*>(source.hostPointer) + source_page * page_bytes,
      pages * page_bytes, cudaMemcpyHostToDevice, stream));
  return true;
}

namespace host_prep {

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(half v) { return __half2float(v); }

__device__ __forceinline__ void write_host_hot_kv(
    const uint4& val_u4, int64_t slot, int kv, int lane, int blocks,
    int page_size, int hot_pages, int64_t page_slots_len, half* history,
    const int32_t* tags, const int32_t* page_slots, half* hot) {
  if (slot < 0 || slot >= static_cast<int64_t>(blocks) * page_size) return;
  const int64_t block = slot / page_size, tok = slot % page_size;
  const int64_t hist_elem =
      ((block * 2 + kv) * page_size + tok) * 256 + lane * 8;
  const int64_t page = slot >> 2;
  const int32_t resident =
      (page >= 0 && page < page_slots_len) ? page_slots[page] : -1;
  const bool update_hot = (resident >= 0 && resident < hot_pages) &&
                          (tags[resident] == static_cast<int32_t>(page));
  const int64_t hot_elem =
      ((static_cast<int64_t>(resident) * 4 + (slot & 3)) * 2 + kv) * 256 +
      lane * 8;

  *reinterpret_cast<uint4*>(history + hist_elem) = val_u4;
  if (update_hot) {
    *reinterpret_cast<uint4*>(hot + hot_elem) = val_u4;
  }
}

template <typename CS, typename PosT, typename SlotT>
__global__ void prep_write_kernel(
    const half* __restrict__ qkv, int64_t ld_qkv, const PosT* __restrict__ pos,
    const CS* __restrict__ cos_sin, int rot, const half* __restrict__ qw,
    const half* __restrict__ kw, float eps, half* __restrict__ query_out,
    half* __restrict__ key_out, const SlotT* __restrict__ slots,
    half* __restrict__ history, const int32_t* __restrict__ tags,
    const int32_t* __restrict__ page_slots, half* __restrict__ hot,
    int32_t* __restrict__ epoch, int HQ, int blocks, int page_size,
    int hot_pages, int64_t page_slots_len) {
  const int t = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp > HQ + 1) return;

  if (warp == HQ + 1) {
    const int64_t sl = static_cast<int64_t>(slots[t]);
    const uint4 v = *reinterpret_cast<const uint4*>(
        qkv + static_cast<int64_t>(t) * ld_qkv + HQ * 512 + 256 + lane * 8);
    write_host_hot_kv(v, sl, 1, lane, blocks, page_size, hot_pages,
                      page_slots_len, history, tags, page_slots, hot);
    return;
  }

  const bool is_k = (warp == HQ);
  const half* src = qkv + static_cast<int64_t>(t) * ld_qkv +
                    (is_k ? HQ * 512 : warp * 512) + lane * 8;
  const uint4 raw = *reinterpret_cast<const uint4*>(src);
  const half* xh = reinterpret_cast<const half*>(&raw);
  const half* w = (is_k ? kw : qw) + lane * 8;
  const uint4 wraw = *reinterpret_cast<const uint4*>(w);
  const half* wh = reinterpret_cast<const half*>(&wraw);
  float x[8], ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = __half2float(xh[j]);
    ss += x[j] * x[j];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss * (1.0f / 256.0f) + eps);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = __half2float(
        __float2half_rn(x[j] * inv * (1.0f + __half2float(wh[j]))));
  }
  const int half_rot = rot / 2, d0 = lane * 8;
  float px[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int off = half_rot / 8;
    const float up = __shfl_down_sync(0xffffffffu, x[j], off);
    const float dn = __shfl_up_sync(0xffffffffu, x[j], off);
    px[j] = (d0 < half_rot) ? up : dn;
  }
  if (d0 < rot) {
    const int64_t p = static_cast<int64_t>(pos[t]);
    const CS* cs = cos_sin + p * rot;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = d0 + j;
      const bool first = d < half_rot;
      const int i = first ? d : d - half_rot;
      const float c = to_f32(cs[i]), s = to_f32(cs[half_rot + i]);
      x[j] = first ? x[j] * c - px[j] * s : x[j] * c + px[j] * s;
    }
  }
  uint4 outv;
  half* oh = reinterpret_cast<half*>(&outv);
#pragma unroll
  for (int j = 0; j < 8; ++j) oh[j] = __float2half_rn(x[j]);
  if (!is_k) {
    *reinterpret_cast<uint4*>(
        query_out + (static_cast<int64_t>(t) * HQ + warp) * 256 + lane * 8) =
        outv;
  } else {
    *reinterpret_cast<uint4*>(key_out + static_cast<int64_t>(t) * 256 +
                              lane * 8) = outv;
    if (t == 0 && lane == 0) atomicAdd(epoch, 1);
    const int64_t sl = static_cast<int64_t>(slots[t]);
    write_host_hot_kv(outv, sl, 0, lane, blocks, page_size, hot_pages,
                      page_slots_len, history, tags, page_slots, hot);
  }
}

}  // namespace host_prep

void prep_write_out(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cos_sin,
                    torch::Tensor qw, torch::Tensor kw, double eps,
                    torch::Tensor query_out, torch::Tensor key_out,
                    torch::Tensor slots, torch::Tensor history,
                    torch::Tensor scales, torch::Tensor tags,
                    torch::Tensor page_slots, torch::Tensor hot,
                    std::optional<torch::Tensor> hot_scales,
                    torch::Tensor epoch) {
  const c10::cuda::CUDAGuard guard(qkv.device());
  const int64_t M = qkv.size(0);
  if (M == 0) return;
  const int HQ = static_cast<int>(query_out.size(1));
  const int rot = static_cast<int>(cos_sin.size(1));
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kHalf &&
              qkv.dim() == 2 && M <= 64 && qkv.stride(1) == 1 &&
              qkv.size(1) >= HQ * 512 + 512);
  TORCH_CHECK(HQ >= 1 && HQ <= 24 && query_out.is_cuda() &&
              query_out.scalar_type() == at::kHalf &&
              query_out.sizes() == at::IntArrayRef({M, HQ, 256}) &&
              query_out.is_contiguous() && key_out.is_cuda() &&
              key_out.scalar_type() == at::kHalf &&
              key_out.sizes() == at::IntArrayRef({M, 1, 256}) &&
              key_out.is_contiguous());
  TORCH_CHECK(rot > 0 && rot <= 256 && rot % 16 == 0 &&
              cos_sin.is_contiguous() &&
              (cos_sin.scalar_type() == at::kFloat ||
               cos_sin.scalar_type() == at::kHalf));
  TORCH_CHECK(qw.scalar_type() == at::kHalf && qw.numel() == 256 &&
              qw.is_contiguous() && kw.scalar_type() == at::kHalf &&
              kw.numel() == 256 && kw.is_contiguous());
  TORCH_CHECK(
      pos.is_contiguous() && pos.numel() >= M &&
      (pos.scalar_type() == at::kLong || pos.scalar_type() == at::kInt));
  TORCH_CHECK(
      slots.is_contiguous() && slots.numel() >= M &&
      (slots.scalar_type() == at::kLong || slots.scalar_type() == at::kInt));
  TORCH_CHECK(history.scalar_type() == at::kHalf && history.dim() == 5 &&
              history.size(1) == 2 && history.size(3) == 1 &&
              history.size(4) == 256 && history.is_contiguous() &&
              scales.scalar_type() == at::kFloat && scales.is_contiguous());
  TORCH_CHECK(!hot_scales || !hot_scales->defined() || !hot_scales->numel());
  TORCH_CHECK(hot.is_cuda() && hot.scalar_type() == at::kHalf &&
              hot.dim() == 3 && hot.size(1) == 2 && hot.size(2) == 256 &&
              hot.is_contiguous());
  TORCH_CHECK(tags.scalar_type() == at::kInt && tags.is_contiguous() &&
              page_slots.scalar_type() == at::kInt &&
              page_slots.is_contiguous() && epoch.scalar_type() == at::kInt &&
              epoch.is_contiguous() && epoch.numel() >= 1);
  auto st = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(history.size(0));
  const int page_size = static_cast<int>(history.size(2));
  const int hot_pages = static_cast<int>(tags.numel());
  const int64_t page_slots_len = page_slots.numel();

#define LAUNCH_PREP_WRITE(CS_T, CS_P, POS_T, SLOT_T)                          \
  host_prep::prep_write_kernel<CS_T, POS_T, SLOT_T>                           \
      <<<static_cast<int>(M), 32 * (HQ + 2), 0, st>>>(                        \
          reinterpret_cast<const half*>(qkv.data_ptr()), qkv.stride(0),       \
          pos.data_ptr<POS_T>(), CS_P, rot,                                   \
          reinterpret_cast<const half*>(qw.data_ptr()),                       \
          reinterpret_cast<const half*>(kw.data_ptr()),                       \
          static_cast<float>(eps),                                            \
          reinterpret_cast<half*>(query_out.data_ptr()),                      \
          reinterpret_cast<half*>(key_out.data_ptr()),                        \
          slots.data_ptr<SLOT_T>(),                                           \
          reinterpret_cast<half*>(history.data_ptr()),                        \
          tags.data_ptr<int32_t>(), page_slots.data_ptr<int32_t>(),           \
          reinterpret_cast<half*>(hot.data_ptr()), epoch.data_ptr<int32_t>(), \
          HQ, blocks, page_size, hot_pages, page_slots_len)

#define DISPATCH_SLOTS(CS_T, CS_P, POS_T)          \
  if (slots.scalar_type() == at::kLong) {          \
    LAUNCH_PREP_WRITE(CS_T, CS_P, POS_T, int64_t); \
  } else {                                         \
    LAUNCH_PREP_WRITE(CS_T, CS_P, POS_T, int32_t); \
  }

#define DISPATCH_POS(CS_T, CS_P)        \
  if (pos.scalar_type() == at::kLong) { \
    DISPATCH_SLOTS(CS_T, CS_P, int64_t) \
  } else {                              \
    DISPATCH_SLOTS(CS_T, CS_P, int32_t) \
  }

  if (cos_sin.scalar_type() == at::kFloat) {
    DISPATCH_POS(float, cos_sin.data_ptr<float>())
  } else {
    DISPATCH_POS(half, reinterpret_cast<const half*>(cos_sin.data_ptr()))
  }
#undef DISPATCH_POS
#undef DISPATCH_SLOTS
#undef LAUNCH_PREP_WRITE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace hot_resolve {

template <typename PosT>
__global__ __launch_bounds__(256) void protect_hot_kernel(
    const int32_t* __restrict__ indices, const int32_t* __restrict__ table,
    const int32_t* __restrict__ requests, const PosT* __restrict__ positions,
    const int32_t* __restrict__ lengths, const int32_t* __restrict__ tags,
    const int32_t* __restrict__ page_slots, int32_t* __restrict__ stamps,
    const int32_t* __restrict__ epoch, int64_t* __restrict__ virtual_positions,
    int32_t* __restrict__ virtual_lengths, int32_t* __restrict__ resolved,
    int64_t* __restrict__ stats, int32_t* __restrict__ miss_queue, int width,
    int num_tiles, int table_width, int64_t table_stride, int64_t index_stride,
    int64_t resolved_stride, int num_requests, int num_blocks, int page_size,
    int hot_pages) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * 256 + threadIdx.x;
  const int32_t req = requests[row];
  const int64_t pos = static_cast<int64_t>(positions[row]);
  const bool valid_req = (req >= 0 && req < num_requests);
  const int64_t len = valid_req ? static_cast<int64_t>(lengths[req]) : 0;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int64_t visible = max(min(pos + 1, len), static_cast<int64_t>(0));
    const int64_t selected_count =
        min(visible / 4, static_cast<int64_t>(width / 4)) * 4 +
        min(visible % 4, static_cast<int64_t>(width % 4));
    virtual_positions[row] = selected_count - 1;
    virtual_lengths[row] = static_cast<int32_t>(selected_count);
  }

  int32_t out_val = -2;
  if (col < width) {
    const int32_t logical =
        indices[static_cast<int64_t>(row) * index_stride + col];
    bool valid = valid_req && logical >= 0 &&
                 static_cast<int64_t>(logical) <= pos &&
                 static_cast<int64_t>(logical) < len &&
                 (logical / page_size) < table_width;
    int32_t blk = -1;
    if (valid) {
      blk = table[static_cast<int64_t>(req) * table_stride +
                  (logical / page_size)];
      valid = (blk >= 0 && blk < num_blocks);
    }
    if (valid) {
      const int32_t token = blk * page_size + (logical % page_size);
      const int32_t page = token >> 2;
      const int32_t reg = page_slots[page];
      if (reg >= 0 && reg < hot_pages && tags[reg] == page) {
        stamps[reg] = *epoch;
        out_val = (reg << 2) | (token & 3);
      } else {
        out_val = -1;
      }
    }
    resolved[static_cast<int64_t>(row) * resolved_stride + col] = out_val;
  }

  const uint32_t miss32 =
      __ballot_sync(0xffffffffu, col < width && out_val == -1);
  const uint32_t hit32 =
      __ballot_sync(0xffffffffu, col < width && out_val >= 0);
  const int half_shift = threadIdx.x & 16;
  const uint32_t miss16 = (miss32 >> half_shift) & 0xffffu;
  const uint32_t hit16 = (hit32 >> half_shift) & 0xffffu;
  const int tile = blockIdx.x * 16 + (threadIdx.x >> 4);
  if (tile < num_tiles && (threadIdx.x & 15) == 0) {
    if (miss16 == 0u) {
      if (hit16 != 0u) {
        const int64_t counter =
            (static_cast<int64_t>(row) * num_tiles + tile) * 3;
        stats[counter] += __popc(hit16);
      }
    } else {
      const int idx = atomicAdd(&miss_queue[0], 1);
      miss_queue[2 + idx] = (row << 16) | tile;
    }
  }
}

template <typename PosT>
__global__ __launch_bounds__(128) void install_hot_misses_kernel(
    const half* __restrict__ history, half* __restrict__ hot,
    int32_t* __restrict__ tags, int32_t* __restrict__ stamps,
    const int32_t* __restrict__ epoch, int32_t* __restrict__ hands,
    int32_t* __restrict__ page_slots, int64_t* __restrict__ stats,
    int32_t* __restrict__ miss_queue, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ table, const int32_t* __restrict__ requests,
    const PosT* __restrict__ positions, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ resolved, int width, int num_tiles, int table_width,
    int64_t table_stride, int64_t index_stride, int64_t resolved_stride,
    int num_requests, int num_blocks, int page_size, int hot_pages,
    int num_hands) {
  const int total_misses =
      *reinterpret_cast<const volatile int32_t*>(&miss_queue[0]);
  if (blockIdx.x >= total_misses) return;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int32_t epoch_val = *epoch;

  for (int item = blockIdx.x; item < total_misses; item += gridDim.x) {
    const int32_t packed = miss_queue[2 + item];
    const int row = packed >> 16;
    const int tile = packed & 0xffff;
    const int col_base = tile * 16 + warp * 4;
    const int col = col_base + lane;

    const int32_t req = requests[row];
    const int64_t pos = static_cast<int64_t>(positions[row]);
    const bool valid_req = (req >= 0 && req < num_requests);
    const int64_t len = valid_req ? static_cast<int64_t>(lengths[req]) : 0;

    int32_t init_res = -2;
    bool valid = false;
    int32_t page = 0;
    int32_t off4 = 0;
    if (lane < 4 && col < width) {
      init_res = resolved[static_cast<int64_t>(row) * resolved_stride + col];
      const int32_t logical =
          indices[static_cast<int64_t>(row) * index_stride + col];
      valid = valid_req && logical >= 0 &&
              static_cast<int64_t>(logical) <= pos &&
              static_cast<int64_t>(logical) < len &&
              (logical / page_size) < table_width;
      if (valid) {
        const int32_t blk = table[static_cast<int64_t>(req) * table_stride +
                                  (logical / page_size)];
        if (blk >= 0 && blk < num_blocks) {
          const int32_t tok = blk * page_size + (logical % page_size);
          page = tok >> 2;
          off4 = tok & 3;
        } else {
          valid = false;
        }
      }
    }

    const uint32_t group_miss_mask =
        __ballot_sync(0xffffffffu, lane < 4 && init_res == -1);
    int32_t grp_hits = 0, grp_misses = 0, grp_contended = 0;

    if (group_miss_mask == 0u) {
      grp_hits = __popc(__ballot_sync(0xffffffffu, lane < 4 && init_res >= 0));
    } else {
      const int32_t first = __shfl_sync(0xffffffffu, page, 0);
      const int32_t v0 = __shfl_sync(0xffffffffu, valid ? 1 : 0, 0);
      const bool lane_ok = !valid || (page == first && off4 == lane);
      const bool aligned =
          (v0 != 0) &&
          (__ballot_sync(0xffffffffu, lane >= 4 || lane_ok) == 0xffffffffu);

      if (!aligned) {
        grp_hits =
            __popc(__ballot_sync(0xffffffffu, lane < 4 && init_res >= 0));
        grp_misses =
            __popc(__ballot_sync(0xffffffffu, lane < 4 && init_res == -1));
      } else {
        int32_t slot = -1;
        int hit_flag = 0, claim_flag = 0;
        if (lane == 0) {
          const int32_t reg = atomicAdd(&page_slots[first], 0);
          if (reg >= 0 && reg < hot_pages) {
            const int32_t t =
                *reinterpret_cast<const volatile int32_t*>(&tags[reg]);
            if (t == first && atomicCAS(&page_slots[first], reg, reg) == reg) {
              stamps[reg] = epoch_val;
              if (*reinterpret_cast<const volatile int32_t*>(&tags[reg]) ==
                      first &&
                  *reinterpret_cast<const volatile int32_t*>(
                      &page_slots[first]) == reg) {
                slot = reg;
                hit_flag = 1;
              }
            }
          }
          if (!hit_flag) {
            claim_flag = (atomicCAS(&page_slots[first], -1, -2) == -1) ? 1 : 0;
          }
        }
        const bool hit = __shfl_sync(0xffffffffu, hit_flag, 0) != 0;
        const bool claim = __shfl_sync(0xffffffffu, claim_flag, 0) != 0;
        slot = __shfl_sync(0xffffffffu, slot, 0);

        bool install = false;
        int32_t old_tag = -1;
        if (claim) {
          const uint32_t base_hand = static_cast<uint32_t>(
              ((static_cast<uint32_t>(first) * 2654435761u) >> 16) %
              static_cast<uint32_t>(num_hands));
          const uint32_t span = static_cast<uint32_t>(hot_pages / num_hands);
          const int probes_per_hand =
              max(1, min(8, static_cast<int>((span + 15u) >> 4)));
          const int hand_tries = min(num_hands, 8);
          for (int h = 0; h < hand_tries && !install; ++h) {
            const uint32_t lane_hand = (base_hand + static_cast<uint32_t>(h)) %
                                       static_cast<uint32_t>(num_hands);
            const uint32_t base = lane_hand * span;
            for (int probe_iter = 0; probe_iter < probes_per_hand && !install;
                 ++probe_iter) {
              uint32_t hand = 0;
              if (lane == 0) {
                hand = atomicAdd(reinterpret_cast<uint32_t*>(&hands[lane_hand]),
                                 16u);
              }
              hand = __shfl_sync(0xffffffffu, hand, 0);
              int32_t age = -1;
              if (lane < 16) {
                const uint32_t cand =
                    base + ((hand + static_cast<uint32_t>(lane)) % span);
                const int32_t st =
                    *reinterpret_cast<const volatile int32_t*>(&stamps[cand]);
                age = (st != epoch_val) ? (epoch_val - st) : -1;
              }
              int32_t best_age = age;
#pragma unroll
              for (int off = 8; off >= 1; off >>= 1) {
                const int32_t other =
                    __shfl_xor_sync(0xffffffffu, best_age, off);
                if (other > best_age) best_age = other;
              }
              best_age = __shfl_sync(0xffffffffu, best_age, 0);
              if (best_age >= 0) {
                const uint32_t rot = (static_cast<uint32_t>(first) * 5u) & 15u;
                const uint32_t best_mask =
                    __ballot_sync(0xffffffffu, lane < 16 && age == best_age);
                const uint32_t other_mask =
                    __ballot_sync(0xffffffffu, lane < 16 && age >= 0) &
                    ~best_mask;
                for (int pass = 0; pass < 2 && !install; ++pass) {
                  const uint32_t raw_mask = pass == 0 ? best_mask : other_mask;
                  uint32_t rem =
                      ((raw_mask >> rot) | (raw_mask << (16u - rot))) & 0xffffu;
                  while (rem != 0u && !install) {
                    const uint32_t bit = static_cast<uint32_t>(__ffs(rem) - 1);
                    rem &= (rem - 1u);
                    const uint32_t chosen = (bit + rot) & 15u;
                    const int32_t chosen_slot =
                        static_cast<int32_t>(base + ((hand + chosen) % span));
                    int won = 0;
                    if (lane == 0) {
                      const int32_t obs =
                          *reinterpret_cast<const volatile int32_t*>(
                              &stamps[chosen_slot]);
                      if (obs != epoch_val &&
                          atomicCAS(&stamps[chosen_slot], obs, epoch_val) ==
                              obs) {
                        won = 1;
                        slot = chosen_slot;
                        old_tag = *reinterpret_cast<const volatile int32_t*>(
                            &tags[chosen_slot]);
                        if (old_tag >= 0) {
                          atomicCAS(&page_slots[old_tag], chosen_slot, -1);
                        }
                      }
                    }
                    if (__shfl_sync(0xffffffffu, won, 0)) {
                      slot = __shfl_sync(0xffffffffu, slot, 0);
                      old_tag = __shfl_sync(0xffffffffu, old_tag, 0);
                      install = true;
                    }
                  }
                }
              }
            }
          }
        }

        if (install) {
          const int32_t phys_blk = (first * 4) / page_size;
          const int32_t phys_off = (first * 4) % page_size;
#pragma unroll
          for (int tok_idx = 0; tok_idx < 4; ++tok_idx) {
#pragma unroll
            for (int kv = 0; kv < 2; ++kv) {
              const half* s_ptr =
                  history +
                  ((static_cast<int64_t>(phys_blk) * 2 + kv) * page_size +
                   phys_off + tok_idx) *
                      256;
              half* d_ptr =
                  hot +
                  (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
              reinterpret_cast<uint4*>(d_ptr)[lane] =
                  reinterpret_cast<const uint4*>(s_ptr)[lane];
            }
          }
        }

        __syncwarp();
        if (lane == 0) {
          if (install) {
            __threadfence();
            atomicCAS(&tags[slot], old_tag, first);
            __threadfence();
            atomicCAS(&page_slots[first], -2, slot);
          } else if (claim) {
            atomicCAS(&page_slots[first], -2, -1);
          }
        }
        __syncwarp();

        if (lane < 4 && col < width) {
          resolved[static_cast<int64_t>(row) * resolved_stride + col] =
              valid ? ((hit || install) ? ((slot << 2) | lane) : -1) : -2;
        }
        const int valid_cnt = __popc(
            __ballot_sync(0xffffffffu, lane < 4 && col < width && valid));
        grp_hits = hit ? valid_cnt : 0;
        grp_misses = hit ? 0 : valid_cnt;
        grp_contended = (!hit && !install) ? 1 : 0;
      }
    }

    if (lane == 0) {
      const int64_t base_counter =
          (static_cast<int64_t>(row) * num_tiles + tile) * 3;
      if (grp_hits != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&stats[base_counter]),
                  static_cast<unsigned long long>(grp_hits));
      }
      if (grp_misses != 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&stats[base_counter + 1]),
            static_cast<unsigned long long>(grp_misses));
      }
      if (grp_contended != 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&stats[base_counter + 2]),
            static_cast<unsigned long long>(grp_contended));
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const int active_blocks = min(total_misses, static_cast<int>(gridDim.x));
    const int done = atomicAdd(&miss_queue[1], 1) + 1;
    if (done == active_blocks) {
      miss_queue[1] = 0;
      __threadfence();
      *reinterpret_cast<volatile int32_t*>(&miss_queue[0]) = 0;
    }
  }
}

__global__ __launch_bounds__(128) void warm_hot_staged_kernel(
    const half* __restrict__ staged, half* __restrict__ hot,
    int32_t* __restrict__ tags, int32_t* __restrict__ stamps,
    const int32_t* __restrict__ epoch, int32_t* __restrict__ hands,
    int32_t* __restrict__ page_slots, const int32_t* __restrict__ table,
    const int32_t* __restrict__ staged_table,
    const int32_t* __restrict__ lengths, int num_requests, int table_width,
    int64_t table_stride, int64_t staged_stride, int num_blocks,
    int staged_blocks, int page_size, int hot_pages, int num_hands,
    int per_req_budget) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int global_warp = blockIdx.x * 4 + warp;
  const int total_warps = gridDim.x * 4;
  const int total_items = num_requests * per_req_budget;
  const int32_t epoch_val = *epoch;
  constexpr int kSinkPages = 64;

  for (int item = global_warp; item < total_items; item += total_warps) {
    const int req = item / per_req_budget;
    const int idx_in_budget = item % per_req_budget;
    const int full_p4 = lengths[req] / 4;
    if (idx_in_budget >= full_p4) continue;

    int lp4 = idx_in_budget;
    if (full_p4 > per_req_budget && idx_in_budget >= kSinkPages) {
      lp4 = (full_p4 - (per_req_budget - kSinkPages)) +
            (idx_in_budget - kSinkPages);
    }
    const int logical0 = lp4 * 4;
    const int blk_col = logical0 / page_size;
    const int off_in_page = logical0 % page_size;
    if (blk_col >= table_width) continue;

    const int32_t phys_blk =
        table[static_cast<int64_t>(req) * table_stride + blk_col];
    const int32_t stg_blk =
        staged_table[static_cast<int64_t>(req) * staged_stride + blk_col];
    if (phys_blk < 0 || phys_blk >= num_blocks || stg_blk < 0 ||
        stg_blk >= staged_blocks) {
      continue;
    }

    const int32_t first = (phys_blk * page_size + off_in_page) >> 2;
    int32_t slot = -1;
    int hit_flag = 0, claim_flag = 0;
    if (lane == 0) {
      const int32_t reg = atomicAdd(&page_slots[first], 0);
      if (reg >= 0 && reg < hot_pages &&
          *reinterpret_cast<const volatile int32_t*>(&tags[reg]) == first) {
        stamps[reg] = epoch_val;
        hit_flag = 1;
      } else if (reg == -1) {
        claim_flag = (atomicCAS(&page_slots[first], -1, -2) == -1) ? 1 : 0;
      }
    }
    if (__shfl_sync(0xffffffffu, hit_flag, 0) != 0) continue;
    const bool claim = __shfl_sync(0xffffffffu, claim_flag, 0) != 0;
    if (!claim) continue;

    bool install = false;
    int32_t old_tag = -1;
    const uint32_t base_hand = static_cast<uint32_t>(
        ((static_cast<uint32_t>(first) * 2654435761u) >> 16) %
        static_cast<uint32_t>(num_hands));
    const uint32_t span = static_cast<uint32_t>(hot_pages / num_hands);
    const int probes_per_hand =
        max(1, min(8, static_cast<int>((span + 15u) >> 4)));
    const int hand_tries = min(num_hands, 8);
    for (int h = 0; h < hand_tries && !install; ++h) {
      const uint32_t lane_hand = (base_hand + static_cast<uint32_t>(h)) %
                                 static_cast<uint32_t>(num_hands);
      const uint32_t base = lane_hand * span;
      for (int probe_iter = 0; probe_iter < probes_per_hand && !install;
           ++probe_iter) {
        uint32_t hand = 0;
        if (lane == 0) {
          hand = atomicAdd(reinterpret_cast<uint32_t*>(&hands[lane_hand]), 16u);
        }
        hand = __shfl_sync(0xffffffffu, hand, 0);
        int32_t age = -1;
        if (lane < 16) {
          const uint32_t cand =
              base + ((hand + static_cast<uint32_t>(lane)) % span);
          const int32_t st =
              *reinterpret_cast<const volatile int32_t*>(&stamps[cand]);
          age = (st != epoch_val) ? (epoch_val - st) : -1;
        }
        int32_t best_age = age;
#pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
          const int32_t other = __shfl_xor_sync(0xffffffffu, best_age, off);
          if (other > best_age) best_age = other;
        }
        best_age = __shfl_sync(0xffffffffu, best_age, 0);
        if (best_age >= 0) {
          const uint32_t rot = (static_cast<uint32_t>(first) * 5u) & 15u;
          const uint32_t best_mask =
              __ballot_sync(0xffffffffu, lane < 16 && age == best_age);
          const uint32_t other_mask =
              __ballot_sync(0xffffffffu, lane < 16 && age >= 0) & ~best_mask;
          for (int pass = 0; pass < 2 && !install; ++pass) {
            const uint32_t raw_mask = pass == 0 ? best_mask : other_mask;
            uint32_t rem =
                ((raw_mask >> rot) | (raw_mask << (16u - rot))) & 0xffffu;
            while (rem != 0u && !install) {
              const uint32_t bit = static_cast<uint32_t>(__ffs(rem) - 1);
              rem &= (rem - 1u);
              const uint32_t chosen = (bit + rot) & 15u;
              const int32_t chosen_slot =
                  static_cast<int32_t>(base + ((hand + chosen) % span));
              int won = 0;
              if (lane == 0) {
                const int32_t obs = *reinterpret_cast<const volatile int32_t*>(
                    &stamps[chosen_slot]);
                if (obs != epoch_val &&
                    atomicCAS(&stamps[chosen_slot], obs, epoch_val) == obs) {
                  won = 1;
                  slot = chosen_slot;
                  old_tag = *reinterpret_cast<const volatile int32_t*>(
                      &tags[chosen_slot]);
                  if (old_tag >= 0) {
                    atomicCAS(&page_slots[old_tag], chosen_slot, -1);
                  }
                }
              }
              if (__shfl_sync(0xffffffffu, won, 0)) {
                slot = __shfl_sync(0xffffffffu, slot, 0);
                old_tag = __shfl_sync(0xffffffffu, old_tag, 0);
                install = true;
              }
            }
          }
        }
      }
    }

    if (install) {
#pragma unroll
      for (int tok_idx = 0; tok_idx < 4; ++tok_idx) {
#pragma unroll
        for (int kv = 0; kv < 2; ++kv) {
          const half* s_ptr =
              staged + ((static_cast<int64_t>(stg_blk) * 2 + kv) * page_size +
                        off_in_page + tok_idx) *
                           256;
          half* d_ptr =
              hot + (static_cast<int64_t>(slot * 4 + tok_idx) * 2 + kv) * 256;
          reinterpret_cast<uint4*>(d_ptr)[lane] =
              reinterpret_cast<const uint4*>(s_ptr)[lane];
        }
      }
    }

    __syncwarp();
    if (lane == 0) {
      if (install) {
        __threadfence();
        atomicCAS(&tags[slot], old_tag, first);
        __threadfence();
        atomicCAS(&page_slots[first], -2, slot);
      } else {
        atomicCAS(&page_slots[first], -2, -1);
      }
    }
    __syncwarp();
  }
}

}  // namespace hot_resolve

void resolve_hot_out(torch::Tensor history, torch::Tensor /*scales*/,
                     torch::Tensor hot, std::optional<torch::Tensor> hot_scales,
                     torch::Tensor tags, torch::Tensor stamps,
                     torch::Tensor epoch, torch::Tensor hands,
                     torch::Tensor page_slots, torch::Tensor stats,
                     torch::Tensor miss_queue, torch::Tensor indices,
                     torch::Tensor table, torch::Tensor requests,
                     torch::Tensor positions, torch::Tensor lengths,
                     torch::Tensor virtual_positions,
                     torch::Tensor virtual_lengths, torch::Tensor resolved) {
  const int rows = indices.size(0);
  const int width = indices.size(1);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(indices.device());
  TORCH_CHECK(history.scalar_type() == at::kHalf && history.dim() == 5 &&
              history.size(1) == 2 && history.size(2) % 4 == 0 &&
              history.size(3) == 1 && history.size(4) == 256 &&
              history.is_contiguous());
  TORCH_CHECK(hot.scalar_type() == at::kHalf &&
              (!hot_scales || !hot_scales->numel()));
  const int num_blocks = history.size(0);
  const int page_size = history.size(2);
  const int hot_pages = hot.size(0) / 4;
  const int num_hands = hands.numel();
  const int num_tiles = (width + 15) / 16;
  const int table_width = table.size(1);
  const int num_requests = table.size(0);
  TORCH_CHECK(miss_queue.numel() >= rows * num_tiles + 2 &&
              stats.numel() >= rows * num_tiles * 3);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 protect_grid(std::max(1, (width + 255) / 256), rows);
  const dim3 install_grid(std::min(rows * std::max(1, num_tiles), 160));

  auto launch = [&](auto pos_tag) {
    using PosT = decltype(pos_tag);
    hot_resolve::protect_hot_kernel<PosT><<<protect_grid, 256, 0, stream>>>(
        indices.data_ptr<int32_t>(), table.data_ptr<int32_t>(),
        requests.data_ptr<int32_t>(), positions.data_ptr<PosT>(),
        lengths.data_ptr<int32_t>(), tags.data_ptr<int32_t>(),
        page_slots.data_ptr<int32_t>(), stamps.data_ptr<int32_t>(),
        epoch.data_ptr<int32_t>(), virtual_positions.data_ptr<int64_t>(),
        virtual_lengths.data_ptr<int32_t>(), resolved.data_ptr<int32_t>(),
        stats.data_ptr<int64_t>(), miss_queue.data_ptr<int32_t>(), width,
        num_tiles, table_width, table.stride(0), indices.stride(0),
        resolved.stride(0), num_requests, num_blocks, page_size, hot_pages);
    hot_resolve::install_hot_misses_kernel<PosT>
        <<<install_grid, 128, 0, stream>>>(
            reinterpret_cast<const half*>(history.data_ptr()),
            reinterpret_cast<half*>(hot.data_ptr()), tags.data_ptr<int32_t>(),
            stamps.data_ptr<int32_t>(), epoch.data_ptr<int32_t>(),
            hands.data_ptr<int32_t>(), page_slots.data_ptr<int32_t>(),
            stats.data_ptr<int64_t>(), miss_queue.data_ptr<int32_t>(),
            indices.data_ptr<int32_t>(), table.data_ptr<int32_t>(),
            requests.data_ptr<int32_t>(), positions.data_ptr<PosT>(),
            lengths.data_ptr<int32_t>(), resolved.data_ptr<int32_t>(), width,
            num_tiles, table_width, table.stride(0), indices.stride(0),
            resolved.stride(0), num_requests, num_blocks, page_size, hot_pages,
            num_hands);
  };
  if (positions.scalar_type() == at::kLong) {
    launch(int64_t{});
  } else {
    TORCH_CHECK(positions.scalar_type() == at::kInt);
    launch(int32_t{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void warm_hot_staged_out(torch::Tensor staged_history,
                         torch::Tensor /*staged_scales*/, torch::Tensor hot,
                         std::optional<torch::Tensor> hot_scales,
                         torch::Tensor tags, torch::Tensor stamps,
                         torch::Tensor epoch, torch::Tensor hands,
                         torch::Tensor page_slots, torch::Tensor table,
                         torch::Tensor staged_table, torch::Tensor lengths,
                         int64_t num_blocks) {
  const int num_requests = table.size(0);
  const int staged_blocks = staged_history.size(0);
  if (num_requests <= 0 || staged_blocks <= 0) return;
  const c10::cuda::CUDAGuard guard(hot.device());
  TORCH_CHECK(staged_history.scalar_type() == at::kHalf &&
              hot.scalar_type() == at::kHalf &&
              (!hot_scales || !hot_scales->numel()));
  const int page_size = staged_history.size(2);
  const int hot_pages = hot.size(0) / 4;
  if (hot_pages <= 0) return;
  const int per_req_budget =
      std::min(staged_blocks * (page_size / 4),
               std::max(16, hot_pages / std::max(1, num_requests)));
  const int total_items = num_requests * per_req_budget;
  if (total_items <= 0) return;
  const dim3 grid(std::min((total_items + 3) / 4, 160));
  const auto stream = at::cuda::getCurrentCUDAStream();
  hot_resolve::warm_hot_staged_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const half*>(staged_history.data_ptr()),
      reinterpret_cast<half*>(hot.data_ptr()), tags.data_ptr<int32_t>(),
      stamps.data_ptr<int32_t>(), epoch.data_ptr<int32_t>(),
      hands.data_ptr<int32_t>(), page_slots.data_ptr<int32_t>(),
      table.data_ptr<int32_t>(), staged_table.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), num_requests, table.size(1), table.stride(0),
      staged_table.stride(0), static_cast<int>(num_blocks), staged_blocks,
      page_size, hot_pages, hands.numel(), per_req_budget);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace dcp2_p2p {

constexpr int kMaxM = 64;
constexpr int kHLocal = 6;
constexpr int kDim = 256;
constexpr int kQSlotWords = kMaxM * kHLocal * (kDim / 2);  // 49152 uint2
constexpr int kOSlotWords = kMaxM * kHLocal * (kDim + 1);  // 98688 uint2

__device__ __forceinline__ unsigned ldv_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void st2_tag(uint2* p, unsigned payload,
                                        unsigned tag) {
  asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" ::"l"(p),
               "r"(payload), "r"(tag)
               : "memory");
}

__device__ __forceinline__ unsigned poll2_tag(const uint2* p, unsigned tag) {
  unsigned a, b;
  const long long t0 = clock64();
  do {
    asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                 : "=r"(a), "=r"(b)
                 : "l"(p));
    if (clock64() - t0 > 4000000000LL) __trap();
  } while (b != tag);
  return a;
}

__device__ __forceinline__ void finish_epoch(unsigned* seq_state) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(seq_state + 1, 1u) == static_cast<unsigned>(gridDim.x - 1)) {
      seq_state[1] = 0u;
      __threadfence();
      atomicAdd(seq_state, 1u);
    }
  }
}

__global__ __launch_bounds__(128) void dcp2_q_prep_kernel(
    const half* __restrict__ query, int64_t qs0, int64_t qs1,
    half* __restrict__ gathered_q, const int64_t* __restrict__ positions,
    const int32_t* __restrict__ lengths, int64_t* __restrict__ local_positions,
    int32_t* __restrict__ local_lengths, uint2* __restrict__ my_buf,
    uint2* __restrict__ peer_buf, unsigned* __restrict__ seq_state, int M,
    int num_reqs, int dcp_rank, int interleave) {
  const int tx = threadIdx.x;
  const unsigned ep = ldv_u32(seq_state);
  const unsigned tag = ep + 1u;
  const int64_t slot_off = static_cast<int64_t>(ep & 1u) * kQSlotWords;
  uint2* my_slot = my_buf + slot_off;
  uint2* peer_slot = peer_buf + slot_off;

  // CTA 0 computes local_positions and local_lengths (replaces 12 PyTorch ops).
  if (blockIdx.x == 0) {
    const int cycle = 2 * interleave;
    const int rank_off = dcp_rank * interleave;
    if (tx < M) {
      int64_t p1 = positions[tx] + 1;
      if (p1 < 0) p1 = 0;
      const int64_t rem = p1 % cycle;
      int64_t extra = rem - rank_off;
      if (extra < 0) extra = 0;
      if (extra > interleave) extra = interleave;
      local_positions[tx] = (p1 / cycle) * interleave + extra - 1;
    }
    if (tx < num_reqs) {
      int32_t len = lengths[tx];
      if (len < 0) len = 0;
      const int32_t rem = len % cycle;
      int32_t extra = rem - rank_off;
      if (extra < 0) extra = 0;
      if (extra > interleave) extra = interleave;
      local_lengths[tx] = (len / cycle) * interleave + extra;
    }
  }

  const int total_pairs = M * kHLocal;
  const int my_h0 = dcp_rank * kHLocal;
  const int peer_h0 = (1 - dcp_rank) * kHLocal;

  // Phase 1: push all local query heads to peer + copy locally before polling.
  for (int idx = blockIdx.x; idx < total_pairs; idx += gridDim.x) {
    const int m = idx / kHLocal;
    const int h = idx % kHLocal;
    const uint32_t h2 = *reinterpret_cast<const uint32_t*>(
        query + static_cast<int64_t>(m) * qs0 + static_cast<int64_t>(h) * qs1 +
        tx * 2);
    *reinterpret_cast<uint32_t*>(
        gathered_q +
        (static_cast<int64_t>(m) * (2 * kHLocal) + my_h0 + h) * kDim + tx * 2) =
        h2;
    st2_tag(peer_slot + static_cast<int64_t>(idx) * (kDim / 2) + tx, h2, tag);
  }

  // Phase 2: poll peer query heads into the complementary head slice.
  for (int idx = blockIdx.x; idx < total_pairs; idx += gridDim.x) {
    const int m = idx / kHLocal;
    const int h = idx % kHLocal;
    const uint32_t peer_h2 =
        poll2_tag(my_slot + static_cast<int64_t>(idx) * (kDim / 2) + tx, tag);
    *reinterpret_cast<uint32_t*>(
        gathered_q +
        (static_cast<int64_t>(m) * (2 * kHLocal) + peer_h0 + h) * kDim +
        tx * 2) = peer_h2;
  }

  finish_epoch(seq_state);
}

__global__ __launch_bounds__(256) void dcp2_combine_gate_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_lse, const half* __restrict__ gate,
    int64_t gs0, int64_t gs1, half* __restrict__ out, int64_t os0, int64_t os1,
    uint2* __restrict__ my_buf, uint2* __restrict__ peer_buf,
    unsigned* __restrict__ seq_state, int M, int dcp_rank) {
  const int tx = threadIdx.x;
  const unsigned ep = ldv_u32(seq_state);
  const unsigned tag = ep + 1u;
  const int64_t slot_off = static_cast<int64_t>(ep & 1u) * kOSlotWords;
  uint2* my_slot = my_buf + slot_off;
  uint2* peer_slot = peer_buf + slot_off;

  const int total_pairs = M * kHLocal;
  const int my_h0 = dcp_rank * kHLocal;
  const int peer_h0 = (1 - dcp_rank) * kHLocal;

  // Phase 1: push the peer's 6 heads (partial_out + partial_lse) over NVLink.
  for (int idx = blockIdx.x; idx < total_pairs; idx += gridDim.x) {
    const int m = idx / kHLocal;
    const int h = idx % kHLocal;
    const int ph = peer_h0 + h;
    const float v =
        partial_out[(static_cast<int64_t>(m) * (2 * kHLocal) + ph) * kDim + tx];
    const int64_t base = static_cast<int64_t>(idx) * (kDim + 1);
    st2_tag(peer_slot + base + tx, __float_as_uint(v), tag);
    if (tx == 0) {
      const float l = partial_lse[static_cast<int64_t>(m) * (2 * kHLocal) + ph];
      st2_tag(peer_slot + base + kDim, __float_as_uint(l), tag);
    }
  }

  __shared__ float s_lse_peer;

  // Phase 2: poll peer's contribution to our 6 heads, merge base-2 LSE, apply
  // exact sigmoid output gate, and write FP16 output.
  for (int idx = blockIdx.x; idx < total_pairs; idx += gridDim.x) {
    const int m = idx / kHLocal;
    const int h = idx % kHLocal;
    const int mh = my_h0 + h;
    const int64_t base = static_cast<int64_t>(idx) * (kDim + 1);
    const float out_peer = __uint_as_float(poll2_tag(my_slot + base + tx, tag));
    if (tx == 0) {
      s_lse_peer = __uint_as_float(poll2_tag(my_slot + base + kDim, tag));
    }
    __syncthreads();

    const float out_local =
        partial_out[(static_cast<int64_t>(m) * (2 * kHLocal) + mh) * kDim + tx];
    const float lse_local =
        partial_lse[static_cast<int64_t>(m) * (2 * kHLocal) + mh];
    const float lse_peer = s_lse_peer;

    const bool v0 = (lse_local > -1.0e30f);
    const bool v1 = (lse_peer > -1.0e30f);
    float val = 0.0f;
    if (v0 && v1) {
      const float mx = fmaxf(lse_local, lse_peer);
      const float w0 = exp2f(lse_local - mx);
      const float w1 = exp2f(lse_peer - mx);
      val = __fdiv_rn(out_local * w0 + out_peer * w1, w0 + w1);
    } else if (v0) {
      val = out_local;
    } else if (v1) {
      val = out_peer;
    }

    // Match _qsa_output_gate(output, gate) rounding: FP16 attention output
    // multiplied by sigmoid(gate) in FP32, then cast back to FP16.
    val = __half2float(__float2half_rn(val));
    const float g = __half2float(gate[static_cast<int64_t>(m) * gs0 +
                                      static_cast<int64_t>(h) * gs1 + tx]);
    val *=
        __fdiv_rn(1.0f, 1.0f + device_qsa::fast_exp2(-g * 1.4426950408889634f));
    out[static_cast<int64_t>(m) * os0 + static_cast<int64_t>(h) * os1 + tx] =
        __float2half_rn(val);
    __syncthreads();
  }

  finish_epoch(seq_state);
}

void dcp2_q_prep_out(torch::Tensor query, torch::Tensor gathered_q,
                     torch::Tensor positions, torch::Tensor lengths,
                     torch::Tensor local_positions, torch::Tensor local_lengths,
                     int64_t my_buf_ptr, int64_t peer_buf_ptr,
                     torch::Tensor seq_state, int64_t dcp_rank,
                     int64_t interleave) {
  const c10::cuda::CUDAGuard guard(query.device());
  const int M = static_cast<int>(query.size(0));
  const int num_reqs = static_cast<int>(lengths.size(0));
  TORCH_CHECK(query.is_cuda() && query.scalar_type() == at::kHalf &&
              query.dim() == 3 && M > 0 && M <= kMaxM &&
              query.size(1) == kHLocal && query.size(2) == kDim &&
              query.stride(2) == 1);
  TORCH_CHECK(gathered_q.is_cuda() && gathered_q.scalar_type() == at::kHalf &&
              gathered_q.is_contiguous() && gathered_q.size(0) == M &&
              gathered_q.size(1) == 2 * kHLocal && gathered_q.size(2) == kDim);
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == at::kLong &&
              positions.is_contiguous() && positions.numel() >= M);
  TORCH_CHECK(lengths.is_cuda() && lengths.scalar_type() == at::kInt &&
              lengths.is_contiguous() && num_reqs > 0 && num_reqs <= 128);
  TORCH_CHECK(local_positions.is_cuda() &&
              local_positions.scalar_type() == at::kLong &&
              local_positions.is_contiguous() && local_positions.numel() >= M);
  TORCH_CHECK(
      local_lengths.is_cuda() && local_lengths.scalar_type() == at::kInt &&
      local_lengths.is_contiguous() && local_lengths.numel() >= num_reqs);
  TORCH_CHECK(seq_state.is_cuda() && seq_state.scalar_type() == at::kInt &&
              seq_state.is_contiguous() && seq_state.numel() >= 2);
  TORCH_CHECK(dcp_rank == 0 || dcp_rank == 1);
  TORCH_CHECK(interleave >= 1 && my_buf_ptr != 0 && peer_buf_ptr != 0);

  const int ctas = std::min(M * kHLocal, 80);
  const auto stream = at::cuda::getCurrentCUDAStream();
  dcp2_q_prep_kernel<<<ctas, 128, 0, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr()), query.stride(0),
      query.stride(1), reinterpret_cast<half*>(gathered_q.data_ptr()),
      positions.data_ptr<int64_t>(), lengths.data_ptr<int32_t>(),
      local_positions.data_ptr<int64_t>(), local_lengths.data_ptr<int32_t>(),
      reinterpret_cast<uint2*>(static_cast<uintptr_t>(my_buf_ptr)),
      reinterpret_cast<uint2*>(static_cast<uintptr_t>(peer_buf_ptr)),
      reinterpret_cast<unsigned*>(seq_state.data_ptr<int32_t>()), M, num_reqs,
      static_cast<int>(dcp_rank), static_cast<int>(interleave));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dcp2_combine_gate_out(torch::Tensor partial_out, torch::Tensor partial_lse,
                           torch::Tensor gate, torch::Tensor out,
                           int64_t my_buf_ptr, int64_t peer_buf_ptr,
                           torch::Tensor seq_state, int64_t dcp_rank) {
  const c10::cuda::CUDAGuard guard(partial_out.device());
  const int M = static_cast<int>(partial_out.size(0));
  TORCH_CHECK(partial_out.is_cuda() &&
              partial_out.scalar_type() == at::kFloat &&
              partial_out.is_contiguous() && partial_out.dim() == 3 && M > 0 &&
              M <= kMaxM && partial_out.size(1) == 2 * kHLocal &&
              partial_out.size(2) == kDim);
  TORCH_CHECK(partial_lse.is_cuda() &&
              partial_lse.scalar_type() == at::kFloat &&
              partial_lse.is_contiguous() && partial_lse.dim() == 2 &&
              partial_lse.size(0) == M && partial_lse.size(1) == 2 * kHLocal);
  TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == at::kHalf &&
              gate.dim() == 3 && gate.size(0) == M && gate.size(1) == kHLocal &&
              gate.size(2) == kDim && gate.stride(2) == 1);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf &&
              out.dim() == 3 && out.size(0) == M && out.size(1) == kHLocal &&
              out.size(2) == kDim && out.stride(2) == 1);
  TORCH_CHECK(seq_state.is_cuda() && seq_state.scalar_type() == at::kInt &&
              seq_state.is_contiguous() && seq_state.numel() >= 2);
  TORCH_CHECK(dcp_rank == 0 || dcp_rank == 1);
  TORCH_CHECK(my_buf_ptr != 0 && peer_buf_ptr != 0);

  const int ctas = std::min(M * kHLocal, 80);
  const auto stream = at::cuda::getCurrentCUDAStream();
  dcp2_combine_gate_kernel<<<ctas, 256, 0, stream>>>(
      partial_out.data_ptr<float>(), partial_lse.data_ptr<float>(),
      reinterpret_cast<const half*>(gate.data_ptr()), gate.stride(0),
      gate.stride(1), reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
      out.stride(1),
      reinterpret_cast<uint2*>(static_cast<uintptr_t>(my_buf_ptr)),
      reinterpret_cast<uint2*>(static_cast<uintptr_t>(peer_buf_ptr)),
      reinterpret_cast<unsigned*>(seq_state.data_ptr<int32_t>()), M,
      static_cast<int>(dcp_rank));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace dcp2_p2p

// Adapt the imported H6 reader to ReVolta's rank-local GPU KV views.
static void device_qsa_paged_dcp_out(torch::Tensor q, torch::Tensor k,
                                     torch::Tensor v, torch::Tensor indices,
                                     torch::Tensor table,
                                     torch::Tensor requests, double k_scale,
                                     double v_scale, torch::Tensor out,
                                     torch::Tensor lse, torch::Tensor partial,
                                     torch::Tensor state) {
  using namespace device_qsa;
  const c10::cuda::CUDAGuard guard(q.device());
  Args a{};
  a.M = q.size(0);
  a.width = indices.size(1);
  a.splits = (a.width + C - 1) / C;
  TORCH_CHECK(a.M > 0 && a.M <= 20 && q.size(1) == 2 * H && q.size(2) == D &&
              a.splits > 0 && a.splits <= 64);
  TORCH_CHECK(partial.numel() >= a.M * H * a.splits * D &&
              state.numel() >= a.M * H * a.splits * 2);
  a.blocks = k.size(0);
  a.page = k.size(1);
  a.table_width = table.size(1);
  a.num_requests = table.size(0);
  a.history = k.data_ptr<uint8_t>();
  a.paged_v = v.data_ptr<uint8_t>();
  a.kb = k.stride(0);
  a.kt = k.stride(1);
  a.vb = v.stride(0);
  a.vt = v.stride(1);
  a.indices = indices.data_ptr<int>();
  a.table = table.data_ptr<int>();
  a.requests = requests.data_ptr<int>();
  a.k_scale = static_cast<float>(k_scale);
  a.v_scale = static_cast<float>(v_scale);
  a.qs0 = q.stride(0);
  a.qs1 = q.stride(1);
  a.is0 = indices.stride(0);
  a.ts0 = table.stride(0);
  a.os0 = out.stride(0);
  a.os1 = out.stride(1);
  a.ls0 = lse.stride(0);
  a.partial = partial.data_ptr<float>();
  a.state = state.data_ptr<float>();
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(a.splits, a.M);
  for (int h = 0; h < 2 * H; h += H) {
    a.q = reinterpret_cast<const half*>(q.data_ptr<at::Half>()) + h * a.qs1;
    a.out_f32 = out.data_ptr<float>() + h * a.os1;
    a.lse = lse.data_ptr<float>() + h;
    partial_fp8_lite<true><<<grid, D, 0, stream>>>(a);
    merge_n<H, true><<<a.M * H, D, 0, stream>>>(a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(vllm_sm70_qsa_device, m) {
  m.def(
      "run_paged_dcp(Tensor q, Tensor k, Tensor v, Tensor indices, Tensor "
      "table, Tensor requests, float k_scale, float v_scale, Tensor(a!) out, "
      "Tensor(b!) lse, Tensor(c!) partial, Tensor(d!) state) -> ()");
  m.impl("run_paged_dcp", torch::kCUDA, &device_qsa_paged_dcp_out);
  m.def(
      "stage_host_run(Tensor history, Tensor scales, Tensor(a!) out, "
      "Tensor(b!) out_scales, int source_page, int target_page, int pages) "
      "-> bool");
  m.impl("stage_host_run", torch::kCUDA, &stage_host_run);
  m.def(
      "run(Tensor q, Tensor history, Tensor scales, Tensor indices, Tensor "
      "table, Tensor requests, Tensor positions, Tensor lengths, Tensor(a!) "
      "out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state, "
      "Tensor(d!)? lse=None) -> ()");
  m.impl("run", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
            std::optional<torch::Tensor> gate, torch::Tensor partial,
            torch::Tensor state, std::optional<torch::Tensor> lse) {
           device_qsa_out(q, history, scales, indices, table, requests,
                          positions, lengths, out, gate, partial, state,
                          std::nullopt, std::nullopt, 1, std::nullopt,
                          std::nullopt, std::nullopt, std::nullopt, lse);
         });
  // Prefill over device (staged) history: every row in `step`-row tiles that
  // reuse one partial/state workspace, launched from C++ instead of a Python
  // loop. Same kernels and arithmetic as `run` on each tile.
  m.def(
      "run_tiled(Tensor q, Tensor history, Tensor scales, Tensor indices, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "Tensor(a!) out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state, "
      "int step, int tiles=1, Tensor(d!)? lse=None) -> ()");
  m.impl("run_tiled", torch::kCUDA,
         [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
            torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
            torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
            std::optional<torch::Tensor> gate, torch::Tensor partial,
            torch::Tensor state, int64_t step, int64_t tiles,
            std::optional<torch::Tensor> lse) {
           TORCH_CHECK(step > 0 && step <= 64);
           const int64_t rows = q.size(0);
           TORCH_CHECK(indices.size(0) == rows && requests.size(0) == rows &&
                       positions.size(0) == rows && out.size(0) == rows &&
                       (!gate || gate->size(0) == rows) &&
                       (!lse || lse->size(0) == rows));
           for (int64_t start = 0; start < rows; start += step) {
             const int64_t n = std::min(step, rows - start);
             std::optional<torch::Tensor> g;
             if (gate) g = gate->narrow(0, start, n);
             std::optional<torch::Tensor> l;
             if (lse) l = lse->narrow(0, start, n);
             device_qsa_out(q.narrow(0, start, n), history, scales,
                            indices.narrow(0, start, n), table,
                            requests.narrow(0, start, n),
                            positions.narrow(0, start, n), lengths,
                            out.narrow(0, start, n), g, partial, state,
                            std::nullopt, std::nullopt, tiles, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, l);
           }
         });
  // run_tiled over one window of a staged-in-parts history (FP16 draft):
  // window-normalized FP32 rows and per-head (max, sum) for
  // qsa_merge_windows instead of the gated output.
  m.def(
      "run_tiled_window(Tensor q, Tensor history, Tensor scales, "
      "Tensor indices, Tensor table, Tensor requests, Tensor positions, "
      "Tensor lengths, Tensor(a!) partial, Tensor(b!) state, "
      "Tensor(c!) window_out, Tensor(d!) window_state, int step, "
      "int tiles=1) -> ()");
  m.impl(
      "run_tiled_window", torch::kCUDA,
      [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
         torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
         torch::Tensor positions, torch::Tensor lengths, torch::Tensor partial,
         torch::Tensor state, torch::Tensor window_out,
         torch::Tensor window_state, int64_t step, int64_t tiles) {
        TORCH_CHECK(step > 0 && step <= 64);
        const int64_t rows = q.size(0);
        TORCH_CHECK(indices.size(0) == rows && requests.size(0) == rows &&
                    positions.size(0) == rows && window_out.size(0) == rows &&
                    window_state.size(0) == rows);
        for (int64_t start = 0; start < rows; start += step) {
          const int64_t n = std::min(step, rows - start);
          // Window rows write only window_out/state; `q` stands in for out.
          device_qsa_out(
              q.narrow(0, start, n), history, scales,
              indices.narrow(0, start, n), table, requests.narrow(0, start, n),
              positions.narrow(0, start, n), lengths, q.narrow(0, start, n),
              std::nullopt, partial, state, std::nullopt, std::nullopt, tiles,
              window_out.narrow(0, start, n), window_state.narrow(0, start, n));
        }
      });
  // Host-authoritative history with a protected device hot cache: resolved
  // tokens read the hot copy, the rest read mapped host history directly.
  m.def(
      "run_hot(Tensor q, Tensor history, Tensor scales, Tensor indices, Tensor "
      "table, Tensor requests, Tensor positions, Tensor lengths, Tensor(a!) "
      "out, Tensor? gate, Tensor(b!) partial, Tensor(c!) state, Tensor hot, "
      "Tensor resolved, int tiles=1, Tensor? page_slots=None, "
      "Tensor? tags=None, Tensor(d!)? lse=None) -> ()");
  m.impl(
      "run_hot", torch::kCUDA,
      [](torch::Tensor q, torch::Tensor history, torch::Tensor scales,
         torch::Tensor indices, torch::Tensor table, torch::Tensor requests,
         torch::Tensor positions, torch::Tensor lengths, torch::Tensor out,
         std::optional<torch::Tensor> gate, torch::Tensor partial,
         torch::Tensor state, torch::Tensor hot, torch::Tensor resolved,
         int64_t tiles, std::optional<torch::Tensor> page_slots,
         std::optional<torch::Tensor> tags, std::optional<torch::Tensor> lse) {
        device_qsa_out(q, history, scales, indices, table, requests, positions,
                       lengths, out, gate, partial, state, hot, resolved, tiles,
                       std::nullopt, std::nullopt, page_slots, tags, lse);
      });
  m.def(
      "prep_write(Tensor qkv, Tensor pos, Tensor cos_sin, Tensor qw, Tensor "
      "kw, "
      "float eps, Tensor(a!) query_out, Tensor(b!) key_out, Tensor slots, "
      "Tensor(c!) history, Tensor(d!) scales, Tensor tags, Tensor page_slots, "
      "Tensor(e!) hot, Tensor(f!)? hot_scales, Tensor(g!) epoch) -> ()");
  m.impl("prep_write", torch::kCUDA, &prep_write_out);
  m.def(
      "resolve_hot(Tensor history, Tensor scales, Tensor(a!) hot, "
      "Tensor(b!)? hot_scales, Tensor(c!) tags, Tensor(d!) stamps, "
      "Tensor epoch, Tensor(e!) hands, Tensor(f!) page_slots, "
      "Tensor(g!) stats, Tensor(h!) miss_queue, Tensor indices, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "Tensor(i!) virtual_positions, Tensor(j!) virtual_lengths, "
      "Tensor(k!) resolved) -> ()");
  m.impl("resolve_hot", torch::kCUDA, &resolve_hot_out);
  m.def(
      "warm_hot_staged(Tensor staged_history, Tensor staged_scales, "
      "Tensor(a!) hot, Tensor(b!)? hot_scales, Tensor(c!) tags, "
      "Tensor(d!) stamps, Tensor epoch, Tensor(e!) hands, "
      "Tensor(f!) page_slots, Tensor table, Tensor staged_table, "
      "Tensor lengths, int num_blocks) -> ()");
  m.impl("warm_hot_staged", torch::kCUDA, &warm_hot_staged_out);
  m.def(
      "dcp2_q_prep(Tensor query, Tensor(a!) gathered_q, Tensor positions, "
      "Tensor lengths, Tensor(b!) local_positions, Tensor(c!) local_lengths, "
      "int my_buf_ptr, int peer_buf_ptr, Tensor(d!) seq_state, int dcp_rank, "
      "int interleave) -> ()");
  m.impl("dcp2_q_prep", torch::kCUDA, &dcp2_p2p::dcp2_q_prep_out);
  m.def(
      "dcp2_combine_gate(Tensor partial_out, Tensor partial_lse, Tensor gate, "
      "Tensor(a!) out, int my_buf_ptr, int peer_buf_ptr, "
      "Tensor(b!) seq_state, int dcp_rank) -> ()");
  m.impl("dcp2_combine_gate", torch::kCUDA, &dcp2_p2p::dcp2_combine_gate_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
