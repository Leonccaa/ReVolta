// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from 1Cat-vLLM v1.5.4, device_qsa::rows_fp8<2, true, true>:
// https://github.com/1CatAI/1Cat-vLLM/blob/1fa9c438bc6e593dbfdd9430b23012a82023bed8/csrc/sm70_turbomind/ops/qsa_device_history_sm70.cu#L1314
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
constexpr int D = 256, H = 6, C = 64, PAD = 264;

struct Args {
  const half* q;
  const int8_t *k, *v;
  const float *sk, *sv;
  const int *indices, *table, *requests;
  const int64_t* positions;
  const int* lengths;
  float *out, *lse;
  int blocks, page, width, table_width, num_requests;
  int64_t qs0, qs1, kb, kt, vb, vt, skb, skt, svb, svt, is0, ts0, os0, os1, ls0;
};

__device__ __forceinline__ float fast_exp2(float x) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
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
__device__ __forceinline__ void mma_rr(float (&d)[8], uint32_t a0, uint32_t a1,
                                       uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// Two signed INT8 codes (bytes `lo`, `hi` of `word`, selected by __byte_perm)
// as an exact FP16 pair: (0x6400 | (b ^ 0x80)) - 1152.0 == b for all int8 b.
__device__ __forceinline__ uint32_t int8x2_half(uint32_t word,
                                                uint32_t selector) {
  const uint32_t biased =
      __byte_perm(word, 0x64646464u, selector) ^ 0x00800080u;
  const uint32_t bias_bits = 0x64806480u;  // half2(1152.0f, 1152.0f)
  const half2 h2 = __hsub2(*reinterpret_cast<const half2*>(&biased),
                           *reinterpret_cast<const half2*>(&bias_bits));
  return *reinterpret_cast<const uint32_t*>(&h2);
}

// Eight signed INT8 codes as four exact unscaled FP16 pairs.
__device__ __forceinline__ uint4 decode8_int8_unscaled(const uint2 packed) {
  return make_uint4(
      int8x2_half(packed.x, 0x4140u), int8x2_half(packed.x, 0x4342u),
      int8x2_half(packed.y, 0x4140u), int8x2_half(packed.y, 0x4342u));
}

template <bool ResidualPV>
__global__ __launch_bounds__(256, 2) void rows_int8_planar(Args a) {
  const int row = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[8][PAD];
  // Padded rows: the QK K reads (8-row groups), the PV V reads (4 rows) and
  // the PV probability reads (8 head rows) would otherwise share banks.
  constexpr int RKP = 296, SP = C + 4;
  __shared__ __align__(16) unsigned char tile[C][RKP];
  __shared__ __align__(16) float scores[4][8][SP];
  __shared__ int64_t slots[3][C];
  __shared__ float kscale[3][C], vscale[3][C];
  __shared__ float alpha[8], denom[8];
  const int req = a.requests[row];
  const int64_t position = a.positions[row];
  const bool live = req >= 0 && req < a.num_requests;
  const int length = live ? a.lengths[req] : 0;
  const int ntiles = (a.width + C - 1) / C;
  // Locate stages (threads t < C, key t of the located tile).
  int logical = -1;
  int64_t found = -1;
  float found_k = 0.f, found_v = 0.f;
  auto stage_index = [&](int j) {
    const int col = j * C + t;
    logical = t < C && j < ntiles && col < a.width
                  ? a.indices[row * a.is0 + col]
                  : -1;
  };
  auto stage_table = [&]() {
    found = -1;
    if (live && logical >= 0 && logical <= position && logical < length &&
        logical / a.page < a.table_width) {
      const int physical = a.table[req * a.ts0 + logical / a.page];
      if (physical >= 0 && physical < a.blocks)
        found = (int64_t(physical) << 32) | (logical % a.page);
    }
  };
  auto stage_scales = [&]() {
    if (found >= 0) {
      const int block = found >> 32, off = static_cast<uint32_t>(found);
      found_k = __ldg(a.sk + block * a.skb + off * a.skt);
      found_v = __ldg(a.sv + block * a.svb + off * a.svt);
    } else {
      found_k = 0.f;
      found_v = 0.f;
    }
  };
  auto stage_store = [&](int b) {
    if (t < C) {
      slots[b][t] = found;
      kscale[b][t] = found_k;
      vscale[b][t] = found_v;
    }
    return t < C && found >= 0;
  };
  constexpr int kChunks = C * (D / 8) / 256;
  uint2 kraw[kChunks], vraw[kChunks];
  auto load = [&](uint2(&raw)[kChunks], int b, const int8_t* plane,
                  int64_t block_stride, int64_t token_stride) {
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
      const int64_t slot = slots[b][token];
      raw[i] = make_uint2(0, 0);
      if (slot >= 0) {
        const int block = slot >> 32, off = static_cast<uint32_t>(slot);
        raw[i] = __ldg(reinterpret_cast<const uint2*>(
            plane + block * block_stride + off * token_stride + dim));
      }
    }
  };
  auto store = [&](const uint2(&raw)[kChunks]) {
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 256, token = v / 32, dim = (v % 32) * 8;
      *reinterpret_cast<uint2*>(&tile[token][dim]) = raw[i];
    }
  };
  {
    const int head = t / 32, d8 = (t % 32) * 8;
    const uint4 qv = head < H ? __ldg(reinterpret_cast<const uint4*>(
                                    a.q + row * a.qs0 + head * a.qs1 + d8))
                              : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&query[head][d8]) = qv;
  }
  // Prologue: tiles 0 and 1 located, tile 0's K/V in flight.
  stage_index(0);
  stage_table();
  stage_scales();
  int count0 = __syncthreads_count(stage_store(0));
  stage_index(1);
  stage_table();
  stage_scales();
  int count1 = __syncthreads_count(stage_store(1));
  if (count0) {
    load(kraw, 0, a.k, a.kb, a.kt);
    load(vraw, 0, a.v, a.vb, a.vt);
  }
  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  float running_max = -1e20f, running_sum = 0.f;  // head `warp`
  float pv[8] = {};  // element e is head pv_head(e), dim pv_dim(e)
  const int c8 = r8;
  const int pv_base = warp * 32 + ((lane >> 2) & 3) * 8;
  auto pv_head = [&](int e) {
    return (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
  };
  auto pv_dim = [&](int e) {
    return pv_base + ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
  };
  for (int j = 0; j < ntiles; ++j) {
    const int b = j % 3, b1 = (j + 1) % 3, b2 = (j + 2) % 3;
    const bool current = count0 > 0;
    stage_index(j + 2);
    if (current) store(kraw);
    // Tile j + 1's K overlaps the rest of this tile.
    if (count1) load(kraw, b1, a.k, a.kb, a.kt);
    __syncthreads();  // K tile published
    if (current) {
      float acc[8] = {};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = kw * 64 + s * 16;
        const uint4 q0 = *reinterpret_cast<const uint4*>(&query[r8][k]);
        const uint4 q1 = *reinterpret_cast<const uint4*>(&query[r8][k + 8]);
        const uint4 k0 = decode8_int8_unscaled(
            *reinterpret_cast<const uint2*>(&tile[nr][k]));
        const uint4 k1 = decode8_int8_unscaled(
            *reinterpret_cast<const uint2*>(&tile[nr][k + 8]));
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
    }
    stage_table();
    __syncthreads();  // QK partials published, K tile retired
    if (current) {
      store(vraw);
      // Warp `warp` owns head row `warp`: reduce the four K-quarter
      // partials of its keys, then the online-softmax update.
      float s0 = ((scores[0][warp][lane] + scores[1][warp][lane]) +
                  scores[2][warp][lane]) +
                 scores[3][warp][lane];
      float s1 = ((scores[0][warp][lane + 32] + scores[1][warp][lane + 32]) +
                  scores[2][warp][lane + 32]) +
                 scores[3][warp][lane + 32];
      const bool v0 = slots[b][lane] >= 0, v1 = slots[b][lane + 32] >= 0;
      // Per-key K scale after the dot product (unscaled K operands).
      s0 = v0 ? s0 * kscale[b][lane] * (0.0625f * 1.4426950408889634f) : -1e20f;
      s1 = v1 ? s1 * kscale[b][lane + 32] * (0.0625f * 1.4426950408889634f)
              : -1e20f;
      float maximum = fmaxf(s0, s1);
#pragma unroll
      for (int s = 16; s; s >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
      maximum = fmaxf(maximum, running_max);
      const float p0 = v0 ? fast_exp2(s0 - maximum) : 0.f;
      const float p1 = v1 ? fast_exp2(s1 - maximum) : 0.f;
      // PV weights carry the per-key V scale (unscaled V operands).
      scores[0][warp][lane] = p0 * vscale[b][lane];
      scores[0][warp][lane + 32] = p1 * vscale[b][lane + 32];
      float sum = p0 + p1;
#pragma unroll
      for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
      const float scale = fast_exp2(running_max - maximum);
      running_sum = running_sum * scale + sum;
      running_max = maximum;
      if (lane == 0) alpha[warp] = scale;
    }
    // Tile j + 1's V overlaps this tile's PV and the next QK.
    if (count1) load(vraw, b1, a.v, a.vb, a.vt);
    stage_scales();
    __syncthreads();  // V tile, probabilities and alpha published
    if (current) {
#pragma unroll
      for (int e = 0; e < 8; ++e) pv[e] *= alpha[pv_head(e)];
      // Row-major V operand: 4 consecutive dims of key k + (lane & 3).
      const int vk = lane & 3, vn = pv_base + ((lane & 16) ? 4 : 0);
      float correction[8] = {};
#pragma unroll 4
      for (int k = 0; k < C; k += 4) {
        const float4 p = *reinterpret_cast<const float4*>(&scores[0][c8][k]);
        const half2 p01 = __floats2half2_rn(p.x, p.y);
        const half2 p23 = __floats2half2_rn(p.z, p.w);
        const uint32_t codes =
            *reinterpret_cast<const uint32_t*>(&tile[k + vk][vn]);
        const uint32_t v01 = int8x2_half(codes, 0x4140u);
        const uint32_t v23 = int8x2_half(codes, 0x4342u);
        mma_rr(pv, *reinterpret_cast<const uint32_t*>(&p01),
               *reinterpret_cast<const uint32_t*>(&p23), v01, v23);
        if constexpr (ResidualPV) {
          // Lift the residual before rounding. Dividing exact INT8 values
          // by 2048 is also exact in half. Sum corrections within this tile.
          const float2 rounded01 = __half22float2(p01);
          const float2 rounded23 = __half22float2(p23);
          const half2 r01 = __floats2half2_rn((p.x - rounded01.x) * 2048.f,
                                              (p.y - rounded01.y) * 2048.f);
          const half2 r23 = __floats2half2_rn((p.z - rounded23.x) * 2048.f,
                                              (p.w - rounded23.y) * 2048.f);
          const half2 inv = __float2half2_rn(1.f / 2048.f);
          const half2 v01_scaled =
              __hmul2(*reinterpret_cast<const half2*>(&v01), inv);
          const half2 v23_scaled =
              __hmul2(*reinterpret_cast<const half2*>(&v23), inv);
          mma_rr(correction, *reinterpret_cast<const uint32_t*>(&r01),
                 *reinterpret_cast<const uint32_t*>(&r23),
                 *reinterpret_cast<const uint32_t*>(&v01_scaled),
                 *reinterpret_cast<const uint32_t*>(&v23_scaled));
        }
      }
      if constexpr (ResidualPV) {
#pragma unroll
        for (int e = 0; e < 8; ++e) pv[e] += correction[e];
      }
    }
    // Publishes tile j + 2's location; tile/scores free for the next tile.
    const int count2 = __syncthreads_count(stage_store(b2));
    count0 = count1;
    count1 = count2;
  }
  if (lane == 0) denom[warp] = running_sum;
  if (lane == 0 && warp < H) {
    a.lse[row * a.ls0 + warp] =
        running_sum > 0.f ? (running_max + log2f(running_sum)) : -CUDART_INF_F;
  }
  __syncthreads();
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int head = pv_head(e);
    if (head < H) {
      a.out[row * a.os0 + head * a.os1 + pv_dim(e)] =
          denom[head] > 0.f ? __fdiv_rn(pv[e], denom[head]) : 0.f;
    }
  }
}

template <bool ResidualPV>
void rows_int8_planar_out(const torch::Tensor& q, const torch::Tensor& k,
                          const torch::Tensor& v, const torch::Tensor& sk,
                          const torch::Tensor& sv, const torch::Tensor& indices,
                          const torch::Tensor& table,
                          const torch::Tensor& requests,
                          const torch::Tensor& positions,
                          const torch::Tensor& lengths,
                          const torch::Tensor& out, const torch::Tensor& lse) {
  const c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  Args a{};
  a.q = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  a.k = k.data_ptr<int8_t>();
  a.v = v.data_ptr<int8_t>();
  a.sk = sk.data_ptr<float>();
  a.sv = sv.data_ptr<float>();
  a.indices = indices.data_ptr<int>();
  a.table = table.data_ptr<int>();
  a.requests = requests.data_ptr<int>();
  a.positions = positions.data_ptr<int64_t>();
  a.lengths = lengths.data_ptr<int>();
  a.out = out.data_ptr<float>();
  a.lse = lse.data_ptr<float>();
  a.blocks = k.size(0);
  a.page = k.size(1);
  a.width = indices.size(1);
  a.table_width = table.size(1);
  a.num_requests = table.size(0);
  a.qs0 = q.stride(0);
  a.qs1 = q.stride(1);
  a.kb = k.stride(0);
  a.kt = k.stride(1);
  a.vb = v.stride(0);
  a.vt = v.stride(1);
  a.skb = sk.stride(0);
  a.skt = sk.stride(1);
  a.svb = sv.stride(0);
  a.svt = sv.stride(1);
  a.is0 = indices.stride(0);
  a.ts0 = table.stride(0);
  a.os0 = out.stride(0);
  a.os1 = out.stride(1);
  a.ls0 = lse.stride(0);
  // As in the upstream host path, each launch reads six heads with the
  // original row strides and writes its slice of the output and LSE.
  for (int head = 0; head < q.size(1); head += H) {
    rows_int8_planar<ResidualPV><<<q.size(0), 256, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    a.q += H * a.qs1;
    a.out += H * a.os1;
    a.lse += H;
  }
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(vllm_sm70_qsa_device, m) {
  m.def(
      "rows_int8_planar(Tensor q, Tensor k, Tensor v, Tensor sk, Tensor sv, "
      "Tensor indices, Tensor table, Tensor requests, Tensor positions, "
      "Tensor lengths, Tensor(a!) out, Tensor(b!) lse) -> ()");
  m.impl("rows_int8_planar", torch::kCUDA, &rows_int8_planar_out<false>);
  // The benchmark calls this compile-time variant directly.
  m.def(
      "rows_int8_planar_residual(Tensor q, Tensor k, Tensor v, Tensor sk, "
      "Tensor sv, Tensor indices, Tensor table, Tensor requests, "
      "Tensor positions, Tensor lengths, Tensor(a!) out, Tensor(b!) lse) -> "
      "()");
  m.impl("rows_int8_planar_residual", torch::kCUDA,
         &rows_int8_planar_out<true>);
}
