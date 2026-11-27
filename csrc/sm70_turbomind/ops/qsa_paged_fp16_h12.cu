// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>

#include <type_traits>

namespace {
constexpr int D = 256, H = 6, C = 64, PAD = 264;
constexpr int H12 = 2 * H;

struct Args {
  const half* q;
  const void* k;
  const void* v;
  const int* indices;
  const int* table;
  const int* requests;
  float* out;
  float* lse;
  float* partial;
  float* state;
  int blocks, page, width, table_width, num_requests, splits;
  int64_t qs0, qs1, kb, kt, vb, vt, is0, ts0, os0, os1, ls0;
  const float* sk;
  const float* sv;
  int64_t skb, skt, svb, svt;
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

// Packed conversion from 1Cat-vLLM: bias the signed bytes into FP16
// mantissas, then subtract 1152.0 to get exact, unscaled integer values.
__device__ __forceinline__ uint32_t int8x2_half(uint32_t word,
                                                uint32_t selector) {
  const uint32_t biased =
      __byte_perm(word, 0x64646464u, selector) ^ 0x00800080u;
  const uint32_t bias_bits = 0x64806480u;
  const half2 pair = __hsub2(*reinterpret_cast<const half2*>(&biased),
                             *reinterpret_cast<const half2*>(&bias_bits));
  return *reinterpret_cast<const uint32_t*>(&pair);
}

__device__ __forceinline__ uint4 unpack_kv(uint2 packed) {
  return make_uint4(
      int8x2_half(packed.x, 0x4140u), int8x2_half(packed.x, 0x4342u),
      int8x2_half(packed.y, 0x4140u), int8x2_half(packed.y, 0x4342u));
}

__device__ __forceinline__ uint4 unpack_kv(uint4 packed) { return packed; }

template <bool INT8>
__global__ __launch_bounds__(256, 2) void partial_paged_h12(Args a) {
  using Element = std::conditional_t<INT8, int8_t, half>;
  using Vector = std::conditional_t<INT8, uint2, uint4>;
  using Probability = std::conditional_t<INT8, float, half>;
  const int row = blockIdx.y, split = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  __shared__ __align__(16) half query[2][8][PAD];
  __shared__ __align__(16) half tile[C][PAD];
  // Score storage replaces the K tile after all warps have read K. The V
  // tile reuses the same memory after both groups have formed probabilities.
  auto& scores = *reinterpret_cast<float (*)[4][8][C]>(&tile);
  __shared__ __align__(16) Probability probs[H12][C];
  __shared__ int64_t slots[C];
  __shared__ float k_scales[INT8 ? C : 1], v_scales[INT8 ? C : 1];

  bool live = false;
  if (t < C) {
    const int req = a.requests[row], col = split * C + t;
    const int pos = col < a.width ? a.indices[row * a.is0 + col] : -1;
    int64_t slot = -1;
    // The selector already applied causality before DCP localized the IDs.
    if (req >= 0 && req < a.num_requests && pos >= 0 &&
        pos / a.page < a.table_width) {
      const int page = a.table[req * a.ts0 + pos / a.page];
      if (page >= 0 && page < a.blocks) {
        const int offset = pos % a.page;
        slot = (int64_t(page) << 32) | offset;
        if constexpr (INT8) {
          k_scales[t] = __ldg(a.sk + page * a.skb + offset * a.skt);
          v_scales[t] = __ldg(a.sv + page * a.svb + offset * a.svt);
        }
      }
    }
    slots[t] = slot;
    live = slot >= 0;
  }

  float maxima[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float sums[2] = {};
  float result[H12] = {};
  if (__syncthreads_or(live)) {
    const int head = t / 32, d8 = (t % 32) * 8;
#pragma unroll
    for (int hg = 0; hg < 2; ++hg) {
      const uint4 qv =
          head < H ? __ldg(reinterpret_cast<const uint4*>(
                         a.q + row * a.qs0 + (hg * H + head) * a.qs1 + d8))
                   : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(&query[hg][head][d8]) = qv;
    }
    constexpr int chunks = C * 32 / D;
    Vector vraw[chunks];
#pragma unroll
    for (int i = 0; i < chunks; ++i) {
      const int vector = t + i * D, token = vector >> 5,
                dim = (vector & 31) * 8;
      const int64_t slot = slots[token];
      Vector kraw{};
      vraw[i] = Vector{};
      if (slot >= 0) {
        const int64_t page = slot >> 32;
        const int offset = int(slot);
        kraw = __ldg(
            reinterpret_cast<const Vector*>(static_cast<const Element*>(a.k) +
                                            page * a.kb + offset * a.kt + dim));
        vraw[i] = __ldg(
            reinterpret_cast<const Vector*>(static_cast<const Element*>(a.v) +
                                            page * a.vb + offset * a.vt + dim));
      }
      *reinterpret_cast<uint4*>(&tile[token][dim]) = unpack_kv(kraw);
    }
    __syncthreads();

    const int nt = warp & 1, kw = warp >> 1;
    const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int nr =
        nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    float acc[2][8] = {};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int d = kw * 64 + s * 16;
      const uint4 k0 = *reinterpret_cast<const uint4*>(&tile[nr][d]);
      const uint4 k1 = *reinterpret_cast<const uint4*>(&tile[nr][d + 8]);
#pragma unroll
      for (int hg = 0; hg < 2; ++hg) {
        const uint4 q0 = *reinterpret_cast<const uint4*>(&query[hg][r8][d]);
        const uint4 q1 = *reinterpret_cast<const uint4*>(&query[hg][r8][d + 8]);
        mma(acc[hg], q0.x, q0.y, k0.x, k0.y);
        mma(acc[hg], q0.z, q0.w, k0.z, k0.w);
        mma(acc[hg], q1.x, q1.y, k1.x, k1.y);
        mma(acc[hg], q1.z, q1.w, k1.z, k1.w);
      }
    }
    __syncthreads();
#pragma unroll
    for (int hg = 0; hg < 2; ++hg) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int k = nt * 32 + ((lane >> 2) & 3) * 8 +
                      ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
        scores[kw][h][k] = acc[hg][e];
      }
      __syncthreads();
      for (int v = t; v < 8 * C; v += D) {
        const int h = v / C, k = v % C;
        float sum = ((scores[0][h][k] + scores[1][h][k]) + scores[2][h][k]) +
                    scores[3][h][k];
        const bool valid = slots[k] >= 0;
        if constexpr (INT8) {
          if (valid) sum *= k_scales[k];
        }
        scores[0][h][k] =
            valid ? sum * (0.0625f * 1.4426950408889634f) : -1e20f;
      }
      __syncthreads();
      if (warp < H) {
        const float s0 = scores[0][warp][lane];
        const float s1 = scores[0][warp][lane + 32];
        float maximum = fmaxf(s0, s1);
#pragma unroll
        for (int s = 16; s; s >>= 1)
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
        const float p0 = slots[lane] >= 0 ? fast_exp2(s0 - maximum) : 0.f;
        const float p1 = slots[lane + 32] >= 0 ? fast_exp2(s1 - maximum) : 0.f;
        if constexpr (INT8) {
          probs[hg * H + warp][lane] =
              slots[lane] >= 0 ? p0 * v_scales[lane] : 0.f;
          probs[hg * H + warp][lane + 32] =
              slots[lane + 32] >= 0 ? p1 * v_scales[lane + 32] : 0.f;
        } else {
          // Match the FP16 reader's PV operand rounding.
          probs[hg * H + warp][lane] = __float2half_rn(p0);
          probs[hg * H + warp][lane + 32] = __float2half_rn(p1);
        }
        // The softmax normalizer and LSE use the unscaled FP32 probabilities.
        float sum = p0 + p1;
#pragma unroll
        for (int s = 16; s; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
        maxima[hg] = maximum;
        sums[hg] = sum;
      }
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < chunks; ++i) {
      const int vector = t + i * D, token = vector >> 5,
                dim = (vector & 31) * 8;
      *reinterpret_cast<uint4*>(&tile[token][dim]) = unpack_kv(vraw[i]);
    }
    __syncthreads();
#pragma unroll 4
    for (int k = 0; k < C; ++k) {
      const float value = __half2float(tile[k][t]);
#pragma unroll
      for (int h = 0; h < H12; ++h) {
        if constexpr (INT8)
          result[h] = __fmaf_rn(probs[h][k], value, result[h]);
        else
          result[h] = __fmaf_rn(__half2float(probs[h][k]), value, result[h]);
      }
    }
  }
  if (lane == 0 && warp < H) {
#pragma unroll
    for (int hg = 0; hg < 2; ++hg) {
      const int state = ((row * H12 + hg * H + warp) * a.splits + split) * 2;
      a.state[state] = maxima[hg];
      a.state[state + 1] = sums[hg];
    }
  }
#pragma unroll
  for (int h = 0; h < H12; ++h)
    a.partial[((row * H12 + h) * a.splits + split) * D + t] = result[h];
}

// Merge one (row, head) across the rank-local splits, emitting base-2 LSE
// for the existing DCP combine.
__global__ void merge_paged_fp16_h12(Args a) {
  const int row = blockIdx.x / H12, head = blockIdx.x % H12, t = threadIdx.x;
  __shared__ float weights[64], denom_parts[64], scalars[2];
  const int64_t state_base = int64_t(row * H12 + head) * a.splits * 2;
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
    numerator = __fmaf_rn(
        weights[s], a.partial[((row * H12 + head) * a.splits + s) * D + t],
        numerator);
  const float denominator = scalars[1];
  const float result =
      denominator > 0 ? __fdiv_rn(numerator, denominator) : 0.f;
  a.out[row * a.os0 + head * a.os1 + t] = result;
  if (!t) {
    a.lse[row * a.ls0 + head] =
        denominator > 0.f ? (scalars[0] + log2f(denominator)) : -CUDART_INF_F;
  }
}

template <bool INT8>
void launch_paged_h12(const torch::Tensor& q, const torch::Tensor& k,
                      const torch::Tensor& v, const torch::Tensor& indices,
                      const torch::Tensor& table, const torch::Tensor& requests,
                      torch::Tensor& out, torch::Tensor& lse,
                      torch::Tensor& partial, torch::Tensor& state,
                      const torch::Tensor* sk = nullptr,
                      const torch::Tensor* sv = nullptr) {
  using Element = std::conditional_t<INT8, int8_t, at::Half>;
  const c10::cuda::CUDAGuard guard(q.device());
  const int rows = q.size(0), width = indices.size(1);
  const int splits = (width + C - 1) / C;
  TORCH_CHECK(rows > 0 && rows <= 48 && q.size(1) == H12 && q.size(2) == D &&
              splits > 0 && splits <= 64);
  TORCH_CHECK(partial.numel() >= rows * H12 * splits * D &&
              state.numel() >= rows * H12 * splits * 2);
  const Args a{
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      k.data_ptr<Element>(),
      v.data_ptr<Element>(),
      indices.data_ptr<int>(),
      table.data_ptr<int>(),
      requests.data_ptr<int>(),
      out.data_ptr<float>(),
      lse.data_ptr<float>(),
      partial.data_ptr<float>(),
      state.data_ptr<float>(),
      static_cast<int>(k.size(0)),
      static_cast<int>(k.size(1)),
      width,
      static_cast<int>(table.size(1)),
      static_cast<int>(table.size(0)),
      splits,
      q.stride(0),
      q.stride(1),
      k.stride(0),
      k.stride(1),
      v.stride(0),
      v.stride(1),
      indices.stride(0),
      table.stride(0),
      out.stride(0),
      out.stride(1),
      lse.stride(0),
      INT8 ? sk->data_ptr<float>() : nullptr,
      INT8 ? sv->data_ptr<float>() : nullptr,
      INT8 ? sk->stride(0) : 0,
      INT8 ? sk->stride(1) : 0,
      INT8 ? sv->stride(0) : 0,
      INT8 ? sv->stride(1) : 0,
  };

  static thread_local int configured_device = -1;
  if (configured_device != q.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        partial_paged_h12<INT8>, cudaFuncAttributePreferredSharedMemoryCarveout,
        100));
    configured_device = q.get_device();
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  partial_paged_h12<INT8><<<dim3(splits, rows), D, 0, stream>>>(a);
  merge_paged_fp16_h12<<<rows * H12, D, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void paged_fp16_h12_out(const torch::Tensor& q, const torch::Tensor& k,
                        const torch::Tensor& v, const torch::Tensor& indices,
                        const torch::Tensor& table,
                        const torch::Tensor& requests, torch::Tensor& out,
                        torch::Tensor& lse, torch::Tensor& partial,
                        torch::Tensor& state) {
  launch_paged_h12<false>(q, k, v, indices, table, requests, out, lse, partial,
                          state);
}

void paged_int8_h12_out(const torch::Tensor& q, const torch::Tensor& k,
                        const torch::Tensor& v, const torch::Tensor& sk,
                        const torch::Tensor& sv, const torch::Tensor& indices,
                        const torch::Tensor& table,
                        const torch::Tensor& requests, torch::Tensor& out,
                        torch::Tensor& lse, torch::Tensor& partial,
                        torch::Tensor& state) {
  launch_paged_h12<true>(q, k, v, indices, table, requests, out, lse, partial,
                         state, &sk, &sv);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(vllm_sm70_qsa_indexer, m) {
  m.def(
      "paged_fp16_h12(Tensor q, Tensor k, Tensor v, Tensor indices, "
      "Tensor table, Tensor requests, Tensor(a!) out, Tensor(b!) lse, "
      "Tensor(c!) partial, Tensor(d!) state) -> ()");
  m.impl("paged_fp16_h12", torch::kCUDA, &paged_fp16_h12_out);
  m.def(
      "paged_int8_h12(Tensor q, Tensor k, Tensor v, Tensor sk, Tensor sv, "
      "Tensor indices, Tensor table, Tensor requests, Tensor(a!) out, "
      "Tensor(b!) lse, Tensor(c!) partial, Tensor(d!) state) -> ()");
  m.impl("paged_int8_h12", torch::kCUDA, &paged_int8_h12_out);
}
