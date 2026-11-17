// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Opt-in Qwen3.8 TP4 raw-E4M3 grouped decode, M=1..40 (runtime M=2..40).
// Dataflow and integer routing derived from nvfp4_grouped_decode_sm70.cu.
// The compact plan layout comes from awq_grouped_decode_sm70.cu. QPN MMA
// originates in dnv2003/v100-skinny (MIT; see adjacent LICENSE.v100-skinny).
// Three launches, fixed storage, current stream, no allocation/readback/tuning.
// K16 groups ascend inside each contiguous split. Four K4 MMA instructions
// accumulate in FP32 per group; W13 joins splits in increasing split order.
// W13 gate/up, SiLU, product and W2 per-route outputs retain FP16 boundaries.
// W2 joins router slots 0..9 with FP32 fmaf and rounds once to FP16.
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
constexpr int kExperts = 512;
constexpr int kMaxTokens = 40;
constexpr int kMaxRoutes = kMaxTokens * 10;
constexpr int kPack = 8;
constexpr int kPlanExperts = 8;
constexpr int kPlanCounts = kPlanExperts + kMaxRoutes;
constexpr int kPlanStarts = kPlanCounts + kMaxRoutes;
constexpr int kPlanRows = kPlanStarts + kMaxRoutes;
constexpr int kPlanInts = kPlanRows + kMaxRoutes;
static_assert(kPlanInts == 1608, "keep Python plan size in sync");

// Decode every E4M3 encoding without FP8 instructions. global remains FP32;
// round the product once to FP16, then apply the QPN bit-decoder's 2^14 bias.
// This is a register-local scale, never an expanded persistent scale bank.
__device__ __forceinline__ half read_scale(uint8_t code, float global) {
  const unsigned mag = code & 127u;
  unsigned bits = mag == 127u ? 0x7e00u : (mag << 7u) + 0x2000u;
  if (mag < 8u) bits = __half_as_ushort(__float2half_rn(mag * 0x1p-9f));
  bits |= (code & 128u) << 8u;
  const half raw = __ushort_as_half(bits);
  return __hmul(__float2half_rn(__fmul_rn(__half2float(raw), global)),
                __float2half_rn(16384.0f));
}

// Upstream 1Cat eb8448193a reserves groups with integer atomics. Also reserve
// a compact row span per expert so the existing plan needs only 400 rows.
__global__ void grouped_plan_kernel(const int32_t* __restrict__ ids, int routes,
                                    int32_t* __restrict__ plan,
                                    half* intermediate) {
  __shared__ int counts[kExperts];
  __shared__ int group_base[kExperts];
  __shared__ int row_base[kExperts];
  __shared__ int total_groups, total_rows;
  const int t = threadIdx.x;
  for (int e = t; e < kExperts; e += blockDim.x) counts[e] = 0;
  if (t == 0) total_groups = total_rows = 0;
  __syncthreads();
  int expert = 0, ordinal = -1;
  if (t < routes) {
    expert = ids[t];
    if (expert >= 0 && expert < kExperts) {
      ordinal = atomicAdd(counts + expert, 1);
    } else {
      for (int c = 0; c < 160; ++c)
        intermediate[t * 160 + c] = __float2half(0.f);
    }
  }
  __syncthreads();
  for (int e = t; e < kExperts; e += blockDim.x) {
    const int count = counts[e];
    if (count > 0) {
      group_base[e] = atomicAdd(&total_groups, (count + kPack - 1) / kPack);
      row_base[e] = atomicAdd(&total_rows, count);
    }
  }
  __syncthreads();
  if (t == 0) plan[0] = total_groups;
  if (ordinal >= 0) {
    const int row = row_base[expert] + ordinal;
    plan[kPlanRows + row] = t;
    if (ordinal % kPack == 0) {
      const int group = group_base[expert] + ordinal / kPack;
      plan[kPlanExperts + group] = expert;
      plan[kPlanCounts + group] = min(kPack, counts[expert] - ordinal);
      plan[kPlanStarts + group] = row;
    }
  }
}

// Strict E4M3 reconstruction from 1Cat eb8448193a. The subnormals are
// exactly representable in FP16, so their low byte is zero.
__device__ __forceinline__ half read_w13_scale(uint8_t code, float global) {
  const unsigned mag = code & 127u;
  unsigned bits = mag == 127u ? 0x7e00u : (mag << 7u) + 0x2000u;
  if (mag < 8u) bits = ((0x232221201e1c1800ULL >> (mag * 8u)) & 255u) << 8u;
  bits |= (code & 128u) << 8u;
  const half raw = __ushort_as_half(bits);
  return __hmul(__float2half_rn(__fmul_rn(__half2float(raw), global)),
                __float2half_rn(16384.0f));
}

__device__ __forceinline__ void decode(unsigned packed, half2 scale,
                                       half2* out) {
  constexpr unsigned sign = 0x80008000u, em = 0x0e000e00u;
  unsigned v[4] = {((packed << 12) & sign) | ((packed << 9) & em),
                   ((packed << 8) & sign) | ((packed << 5) & em),
                   ((packed << 4) & sign) | ((packed << 1) & em),
                   (packed & sign) | ((packed >> 3) & em)};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    out[i] = __hmul2(*reinterpret_cast<half2*>(v + i), scale);
}

#define PACKED_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int Split, bool Interleaved>
__global__ void w13_kernel(const half* x, const uint32_t* weights,
                           const uint8_t* scales, const float* globals,
                           const int32_t* plan, half* out) {
  // Split within the CTA: no floating-point atomics or global partial tensor.
  __shared__ float partial[2][Split][kPack][32];
  __shared__ half projected[2][kPack][32];
  const int group_id = blockIdx.y;
  if (group_id >= plan[0]) return;
  const int count = plan[kPlanCounts + group_id];
  const int expert = plan[kPlanExperts + group_id];
  const int32_t* rows = plan + kPlanRows + plan[kPlanStarts + group_id];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int projection = warp / Split, split = warp % Split;
  const int tile =
      Interleaved ? blockIdx.x * 2 + projection : blockIdx.x + projection * 5;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + mma_row;
  const int route = mma_row < count ? rows[mma_row] : 0;
  float accum[8] = {};
  if (expert < kExperts) {
    const uint32_t* w = weights + static_cast<size_t>(expert) * 2560 * 40;
    const uint8_t* s = scales + static_cast<size_t>(expert) * 160 * 320;
    const half* input = x + static_cast<size_t>(route / 10) * 2560;
    const int slot = Interleaved ? (col & 1) : projection;
    const float global = __ldg(globals + expert * 2 + slot);
#pragma unroll 4
    for (int g = split * (160 / Split); g < (split + 1) * (160 / Split); ++g) {
      const size_t offset =
          (static_cast<size_t>(tile) * 320 + g * 2) * 32 + col;
      const half scalar =
          read_w13_scale(__ldg(s + (g * 10 + tile) * 32 + col), global);
      const half2 scale = __halves2half2(scalar, scalar);
      half2 decoded[8];
      decode(__ldcs(w + offset), scale, decoded);
      decode(__ldcs(w + offset + 32), scale, decoded + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
      uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
      if (mma_row < count) {
        lo = *reinterpret_cast<const uint4*>(input + g * 16);
        hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
      }
      PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
      PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
      PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
      PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partial[projection][split][r][quad * 8 + c] = accum[i];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < 2 * kPack * 32; idx += blockDim.x) {
    const int p = idx / (kPack * 32), r = idx / 32 % kPack, c = idx % 32;
    // FP16 materialization is retained before SiLU, then again before the
    // multiplication. Split>1 changes FP32 association, not quantization.
    float value = 0;
#pragma unroll
    for (int s = 0; s < Split; ++s) value += partial[p][s][r][c];
    projected[p][r][c] = __float2half_rn(value);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < count * 32; idx += blockDim.x) {
    const int r = idx / 32, c = idx % 32;
    const int p = Interleaved ? c / 16 : 0;
    const int pc = Interleaved ? c % 16 * 2 : c;
    const half gate = projected[p][r][pc];
    const half up = Interleaved ? projected[p][r][pc + 1] : projected[1][r][c];
    const float gf = __half2float(gate);
    const half activated = __float2half_rn(gf / (1.0f + expf(-gf)));
    out[static_cast<size_t>(rows[r]) * 160 + blockIdx.x * 32 + c] =
        __hmul(activated, up);
  }
}

__global__ __launch_bounds__(1024) void w2_batch_reduce_kernel(
    const half* x, const uint32_t* weights, const uint8_t* scales,
    const float* globals, const float* topk, const int32_t* plan, half* out,
    int tokens) {
  __shared__ half values[kMaxRoutes][32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int col = quad * 8 + r, tile = blockIdx.x;
  // Initialize invalid/padded routes too. The barrier prevents another warp
  // from overwriting zeros before initialization has finished.
  for (int i = threadIdx.x; i < tokens * 10 * 32; i += blockDim.x)
    reinterpret_cast<half*>(values)[i] = __float2half(0.f);
  __syncthreads();
  const int groups = plan[0];
  for (int group = warp; group < groups; group += 32) {
    const int count = plan[kPlanCounts + group],
              expert = plan[kPlanExperts + group];
    const int32_t* rows = plan + kPlanRows + plan[kPlanStarts + group];
    const int route = r < count ? rows[r] : 0;
    float accum[8] = {};
    if (expert < kExperts) {
      const uint32_t* w = weights + static_cast<size_t>(expert) * 160 * 320;
      const uint8_t* s = scales + static_cast<size_t>(expert) * 10 * 2560;
      const half* input = x + static_cast<size_t>(route) * 160;
#pragma unroll
      for (int g = 0; g < 10; ++g) {
        const int offset = (tile * 20 + g * 2) * 32 + col;
        const half scalar = read_scale(__ldg(s + (g * 80 + tile) * 32 + col),
                                       __ldg(globals + expert));
        const half2 scale = __halves2half2(scalar, scalar);
        half2 decoded[8];
        decode(__ldcs(w + offset), scale, decoded);
        decode(__ldcs(w + offset + 32), scale, decoded + 4);
        const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
        uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
        if (r < count) {
          lo = *reinterpret_cast<const uint4*>(input + g * 16);
          hi = *reinterpret_cast<const uint4*>(input + g * 16 + 8);
        }
        PACKED_MMA(accum, lo.x, lo.y, b[0], b[1]);
        PACKED_MMA(accum, lo.z, lo.w, b[2], b[3]);
        PACKED_MMA(accum, hi.x, hi.y, b[4], b[5]);
        PACKED_MMA(accum, hi.z, hi.w, b[6], b[7]);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (row < count) {
        values[rows[row]][quad * 8 + c] = __float2half_rn(accum[i]);
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < tokens * 32; idx += blockDim.x) {
    const int token = idx / 32, c = idx % 32;
    float weighted = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 10; ++slot) {
      weighted = fmaf(__half2float(values[token * 10 + slot][c]),
                      __ldg(topk + token * 10 + slot), weighted);
    }
    out[token * 2560 + tile * 32 + c] = __float2half_rn(weighted);
  }
}

void check(const torch::Tensor& t, at::ScalarType dtype, at::IntArrayRef shape,
           at::Device device, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == dtype &&
                  t.sizes() == shape && t.is_contiguous(),
              "nvfp4_raw_grouped: ", name,
              " device/dtype/shape/stride mismatch");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0,
              "nvfp4_raw_grouped: ", name, " must be 16-byte aligned");
}

void run(torch::Tensor out, torch::Tensor intermediate, torch::Tensor plan,
         const torch::Tensor& x, const torch::Tensor& w13,
         const torch::Tensor& s13, const torch::Tensor& g13,
         const torch::Tensor& w2, const torch::Tensor& s2,
         const torch::Tensor& g2, const torch::Tensor& ids,
         const torch::Tensor& topk, int64_t split, bool interleaved) {
  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= kMaxTokens,
              "nvfp4_raw_grouped: expected M in [1,40]");
  const int tokens = x.size(0), routes = tokens * 10;
  const auto device = x.device();
  check(x, at::kHalf, {tokens, 2560}, device, "input");
  const c10::cuda::CUDAGuard guard(device);
  const auto* props = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(props->major == 7 && props->minor == 0,
              "nvfp4_raw_grouped: requires exact SM70");
  TORCH_CHECK(
      split == 1 || split == 2 || split == 4 || split == 5 || split == 8,
      "nvfp4_raw_grouped: split must be 1,2,4,5,8");
  check(out, at::kHalf, {tokens, 2560}, device, "output");
  check(intermediate, at::kHalf, {routes, 160}, device, "intermediate");
  check(plan, at::kInt, {kPlanInts}, device, "plan");
  check(w13, at::kInt, {512, 2560, 40}, device, "W13");
  check(s13, at::kByte, {512, 160, 320}, device, "W13 E4M3 codes");
  check(g13, at::kFloat, {512, 2}, device, "W13 FP32 globals");
  check(w2, at::kInt, {512, 160, 320}, device, "W2");
  check(s2, at::kByte, {512, 10, 2560}, device, "W2 E4M3 codes");
  check(g2, at::kFloat, {512, 1}, device, "W2 FP32 globals");
  check(ids, at::kInt, {tokens, 10}, device, "IDs");
  check(topk, at::kFloat, {tokens, 10}, device, "router weights");
  // All mutations use caller-owned, disjoint storage. Reject aliases before
  // any launch, including input/output aliases that could corrupt graph replay.
  for (const auto& dst : {out, intermediate, plan}) {
    for (const auto& src : {x, w13, s13, g13, w2, s2, g2, ids, topk})
      at::assert_no_overlap(dst, src);
  }
  at::assert_no_overlap(out, intermediate);
  at::assert_no_overlap(out, plan);
  at::assert_no_overlap(intermediate, plan);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  grouped_plan_kernel<<<1, routes <= 256 ? 256 : 512, 0, stream>>>(
      ids.data_ptr<int32_t>(), routes, plan.data_ptr<int32_t>(),
      reinterpret_cast<half*>(intermediate.data_ptr()));
#define LAUNCH(S, I)                                        \
  w13_kernel<S, I><<<dim3(5, routes), 64 * S, 0, stream>>>( \
      reinterpret_cast<const half*>(x.data_ptr()),          \
      reinterpret_cast<const uint32_t*>(w13.data_ptr()),    \
      s13.data_ptr<uint8_t>(), g13.data_ptr<float>(),       \
      plan.data_ptr<int32_t>(),                             \
      reinterpret_cast<half*>(intermediate.data_ptr()))
#define CASE(S)         \
  case S:               \
    if (interleaved) {  \
      LAUNCH(S, true);  \
    } else {            \
      LAUNCH(S, false); \
    }                   \
    break
  switch (split) {
    CASE(1);
    CASE(2);
    CASE(4);
    CASE(5);
    CASE(8);
  }
#undef CASE
#undef LAUNCH
  w2_batch_reduce_kernel<<<80, 1024, 0, stream>>>(
      reinterpret_cast<const half*>(intermediate.data_ptr()),
      reinterpret_cast<const uint32_t*>(w2.data_ptr()), s2.data_ptr<uint8_t>(),
      g2.data_ptr<float>(), topk.data_ptr<float>(), plan.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr()), tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "nvfp4_raw_grouped_decode_sm70_out(Tensor(a!) out, "
      "Tensor(b!) intermediate, Tensor(c!) plan, Tensor x, Tensor w13, "
      "Tensor s13, Tensor g13, Tensor w2, Tensor s2, Tensor g2, Tensor ids, "
      "Tensor topk, int split, bool interleaved) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("nvfp4_raw_grouped_decode_sm70_out", &run);
}
