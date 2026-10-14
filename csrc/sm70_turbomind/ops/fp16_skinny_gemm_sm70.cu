// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SM70 small-M FP16 dense GEMM: out[M, N] = x[M, K] @ w[N, K]^T (+ bias),
// M = 1..48, reading the original row-major checkpoint weight in place (no
// packed copy). Volta mma.m8n8k4 (FP16 x FP16 -> FP32), one rounding to FP16.
//
// Dataflow (the quadpair-split-K form of qwen38_dense_batch_sm70_out, widened
// to the grouped-decode row blocks of awq_grouped_decode_sm70.cu):
//   * RB = ceil(M / 8) row blocks (1..6) share every B fragment, so each
//     weight element is read once per call for all M rows.
//   * The four quadpairs of a warp own interleaved K chunks of one 64-element
//     K step: quadpair q reads elements [8q, 8q + 8) and [32 + 8q, 40 + 8q).
//     A warp load of one row (weight column or activation row) is therefore
//     64 contiguous bytes (two full sectors) instead of eight 32-byte pieces
//     of eight different rows. Each warp covers 8 * NG output columns.
//   * A CTA holds warps_n x warps_k warps: warps_n split the CTA's columns
//     (they share activation lines in L1), warps_k split K. gridDim.y = splits
//     splits K across CTAs.
//   * Partial sums are reduced in a fixed order: quadpairs 0..3 by warp
//     shuffles, then the warps_k warps in shared memory, then (splits > 1)
//     the CTA splits by a second small kernel over an FP32 workspace. No
//     atomics; the result is deterministic and CUDA-graph capturable.
//   * Bias (FP16) is added in FP32 before the single FP16 rounding.
//
// The FP32 association depends only on (K, warps_k, splits): rows are
// independent in m8n8k4 (inactive rows are zero), so a row's result does not
// depend on M, on its row block, on NG or on warps_n (batch invariance for a
// fixed K partition).
//
// A second entry point computes two weights that share the activation (e.g.
// Qwen3.8 GDN in_proj_qkvz + in_proj_ba) in one launch and writes each one's
// columns into two separate outputs split at a column (qkv|z and b|a), so the
// caller needs no split copy.

#include <torch/all.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <cstdint>
#include <optional>
#include <tuple>

namespace {

constexpr int kMaxRows = 48;
constexpr int kMaxRowBlocks = (kMaxRows + 7) / 8;  // 6
constexpr int kMaxWarps = 8;                       // warps_n * warps_k
constexpr int kMaxSplits = 64;

struct SkinnySegment {
  const half* weight;  // [n, k] row-major
  const half* bias;    // [n] or nullptr
  half* out0;          // columns [0, split)
  half* out1;          // columns [split, n)
  int n;
  int split;
  int ld0;
  int ld1;
  int cta_begin;  // first blockIdx.x of this segment
  int col_base;   // first workspace column of this segment
};

struct SkinnyParams {
  const half* x;     // [m, k], row stride ldx
  float* workspace;  // [splits, m, ntot] when splits > 1
  SkinnySegment seg0;
  SkinnySegment seg1;
  int nseg;
  int m;
  int k;
  int ldx;
  int ntot;
};

#define SKINNY_MMA(C, A0, A1, B0, B1)                               \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};"                                  \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// m8n8k4 f32 accumulator element i of this lane -> (row, column) of the
// quadpair's 8x8 tile (same mapping as awq_grouped_decode_sm70.cu).
__device__ __forceinline__ int acc_row(int i, int lane) {
  return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}
__device__ __forceinline__ int acc_col(int i, int lane) {
  return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}

__device__ __forceinline__ uint4 zero4() { return make_uint4(0, 0, 0, 0); }

// Weights are streamed once: evict-first in L1 and L2.
__device__ __forceinline__ uint4 load_weight(const half* p) {
  return __ldcs(reinterpret_cast<const uint4*>(p));
}
// Activations are re-read by every column CTA: keep the default caching.
__device__ __forceinline__ uint4 load_input(const half* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

template <int NG>
__device__ __forceinline__ void load_b(uint4 (&lo)[NG], uint4 (&hi)[NG],
                                       const half* const (&wrow)[NG],
                                       const bool (&col_ok)[NG], int lo_k,
                                       int k_end) {
  const int hi_k = lo_k + 32;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    lo[g] = (col_ok[g] && lo_k < k_end) ? load_weight(wrow[g] + lo_k) : zero4();
    hi[g] = (col_ok[g] && hi_k < k_end) ? load_weight(wrow[g] + hi_k) : zero4();
  }
}

__device__ __forceinline__ void store_output(const SkinnySegment& seg, int row,
                                             int col, float value) {
  if (seg.bias != nullptr) {
    value = __fadd_rn(value, __half2float(seg.bias[col]));
  }
  const half h = __float2half_rn(value);
  if (col < seg.split) {
    seg.out0[static_cast<size_t>(row) * seg.ld0 + col] = h;
  } else {
    seg.out1[static_cast<size_t>(row) * seg.ld1 + (col - seg.split)] = h;
  }
}

template <int NG, int RB>
__global__ void __launch_bounds__(kMaxWarps * 32)
    fp16_skinny_main_kernel(const SkinnyParams p, const int warps_n,
                            const int warps_k) {
  static_assert(NG == 1 || NG == 2 || NG == 4, "unsupported column groups");
  static_assert(RB >= 1 && RB <= kMaxRowBlocks, "unsupported row blocks");
  // [warps_k][RB * 8][cta_cols] FP32 partial sums of this CTA.
  extern __shared__ float red[];
  constexpr int kRows = RB * 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wn = warp % warps_n;
  const int wk = warp / warps_n;
  const int qp = (lane >> 2) & 3;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int cta_cols = 8 * NG * warps_n;

  const bool second =
      p.nseg > 1 && static_cast<int>(blockIdx.x) >= p.seg1.cta_begin;
  // Copy the selected segment (no dynamic indexing into kernel parameters).
  const SkinnySegment seg = second ? p.seg1 : p.seg0;
  const half* __restrict__ w = seg.weight;
  const int n = seg.n;
  const int c0 = (static_cast<int>(blockIdx.x) - seg.cta_begin) * cta_cols;

  // Contiguous K partition (16-element units) of this (CTA split, warp).
  const int steps = p.k >> 4;
  const int parts = static_cast<int>(gridDim.y) * warps_k;
  const int part = static_cast<int>(blockIdx.y) * warps_k + wk;
  const int k_begin = (part * steps / parts) * 16;
  const int k_end = ((part + 1) * steps / parts) * 16;

  const half* wrow[NG];
  bool col_ok[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int col = c0 + wn * 8 * NG + g * 8 + r8;
    col_ok[g] = col < n;
    wrow[g] = w + static_cast<size_t>(col_ok[g] ? col : 0) * p.k;
  }
  const half* xrow[RB];
  bool row_ok[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    const int row = rb * 8 + r8;
    row_ok[rb] = row < p.m;
    xrow[rb] = p.x + static_cast<size_t>(row_ok[rb] ? row : 0) * p.ldx;
  }

  float acc[NG][RB][8];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[g][rb][i] = 0.f;
    }
  }

  // B fragments of the current 64-element step; the next step's fragments
  // are requested before this step's MMAs (one step of weight prefetch).
  uint4 b_lo[NG];
  uint4 b_hi[NG];
  load_b<NG>(b_lo, b_hi, wrow, col_ok, k_begin + qp * 8, k_end);
#pragma unroll 2
  for (int e = k_begin; e < k_end; e += 64) {
    const int lo_k = e + qp * 8;
    const int hi_k = lo_k + 32;
    uint4 n_lo[NG];
    uint4 n_hi[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      n_lo[g] = zero4();
      n_hi[g] = zero4();
    }
    if (e + 64 < k_end) {  // warp-uniform
      load_b<NG>(n_lo, n_hi, wrow, col_ok, lo_k + 64, k_end);
    }
    const bool lo_in = lo_k < k_end;
    const bool hi_in = hi_k < k_end;
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      const uint4 a_lo =
          (row_ok[rb] && lo_in) ? load_input(xrow[rb] + lo_k) : zero4();
      const uint4 a_hi =
          (row_ok[rb] && hi_in) ? load_input(xrow[rb] + hi_k) : zero4();
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        SKINNY_MMA(acc[g][rb], a_lo.x, a_lo.y, b_lo[g].x, b_lo[g].y);
        SKINNY_MMA(acc[g][rb], a_lo.z, a_lo.w, b_lo[g].z, b_lo[g].w);
        SKINNY_MMA(acc[g][rb], a_hi.x, a_hi.y, b_hi[g].x, b_hi[g].y);
        SKINNY_MMA(acc[g][rb], a_hi.z, a_hi.w, b_hi[g].z, b_hi[g].w);
      }
    }
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      b_lo[g] = n_lo[g];
      b_hi[g] = n_hi[g];
    }
  }

  // Quadpair reduction in the fixed order q0 + q1 + q2 + q3 (every lane ends
  // with the sums of its quadpair-0 tile positions).
  const int base_lane = lane & ~12;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float v = __shfl_sync(0xffffffffu, acc[g][rb][i], base_lane);
#pragma unroll
        for (int q = 1; q < 4; ++q) {
          v = __fadd_rn(
              v, __shfl_sync(0xffffffffu, acc[g][rb][i], base_lane | (q << 2)));
        }
        acc[g][rb][i] = v;
      }
    }
  }
  if (qp == 0) {
#pragma unroll
    for (int g = 0; g < NG; ++g) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int row = rb * 8 + acc_row(i, lane);
          const int col = wn * 8 * NG + g * 8 + acc_col(i, lane);
          red[(wk * kRows + row) * cta_cols + col] = acc[g][rb][i];
        }
      }
    }
  }
  __syncthreads();

  // Fixed warp_k order, then either the final FP16 output (one CTA split) or
  // this split's FP32 partial (reduced by fp16_skinny_reduce_kernel).
  const bool final_output = gridDim.y == 1;
  float* __restrict__ partial =
      final_output
          ? nullptr
          : p.workspace + static_cast<size_t>(blockIdx.y) * p.m * p.ntot;
  for (int idx = threadIdx.x; idx < p.m * cta_cols; idx += blockDim.x) {
    const int row = idx / cta_cols;
    const int c = idx - row * cta_cols;
    const int col = c0 + c;
    if (col >= n) continue;
    float v = red[row * cta_cols + c];
    for (int q = 1; q < warps_k; ++q) {
      v = __fadd_rn(v, red[(q * kRows + row) * cta_cols + c]);
    }
    if (final_output) {
      store_output(seg, row, col, v);
    } else {
      partial[static_cast<size_t>(row) * p.ntot + seg.col_base + col] = v;
    }
  }
}

// Fixed-order sum over the CTA K splits, bias, single FP16 rounding.
__global__ void __launch_bounds__(256)
    fp16_skinny_reduce_kernel(const SkinnyParams p, const int splits) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= p.m * p.ntot) return;
  const int row = idx / p.ntot;
  const int gcol = idx - row * p.ntot;
  const size_t plane = static_cast<size_t>(p.m) * p.ntot;
  float v = p.workspace[idx];
  for (int s = 1; s < splits; ++s) {
    v = __fadd_rn(v, p.workspace[s * plane + idx]);
  }
  const bool second = p.nseg > 1 && gcol >= p.seg1.col_base;
  const SkinnySegment seg = second ? p.seg1 : p.seg0;
  store_output(seg, row, gcol - seg.col_base, v);
}

template <int NG, int RB>
void launch_main(const SkinnyParams& p, int warps_n, int warps_k, int ctas,
                 int splits, cudaStream_t stream) {
  const int cta_cols = 8 * NG * warps_n;
  const size_t smem =
      static_cast<size_t>(warps_k) * RB * 8 * cta_cols * sizeof(float);
  TORCH_CHECK(smem <= 48 * 1024, "fp16_skinny_gemm: reduction tile exceeds ",
              "48 KB (warps_k * rows * cta_cols)");
  fp16_skinny_main_kernel<NG, RB>
      <<<dim3(ctas, splits), warps_n * warps_k * 32, smem, stream>>>(p, warps_n,
                                                                     warps_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NG>
void dispatch_rb(int rb, const SkinnyParams& p, int warps_n, int warps_k,
                 int ctas, int splits, cudaStream_t stream) {
  static_assert(kMaxRowBlocks == 6, "add the new row-block variants");
  if constexpr (NG == 4) {
    // 4 column groups keep 32 accumulators per row block: at most 24 rows.
    TORCH_CHECK(rb <= 3, "fp16_skinny_gemm: ng=4 supports at most 24 rows");
    switch (rb) {
      case 1:
        launch_main<NG, 1>(p, warps_n, warps_k, ctas, splits, stream);
        break;
      case 2:
        launch_main<NG, 2>(p, warps_n, warps_k, ctas, splits, stream);
        break;
      default:
        launch_main<NG, 3>(p, warps_n, warps_k, ctas, splits, stream);
        break;
    }
  } else {
    switch (rb) {
      case 1:
        launch_main<NG, 1>(p, warps_n, warps_k, ctas, splits, stream);
        break;
      case 2:
        launch_main<NG, 2>(p, warps_n, warps_k, ctas, splits, stream);
        break;
      case 3:
        launch_main<NG, 3>(p, warps_n, warps_k, ctas, splits, stream);
        break;
      case 4:
        launch_main<NG, 4>(p, warps_n, warps_k, ctas, splits, stream);
        break;
      case 5:
        launch_main<NG, 5>(p, warps_n, warps_k, ctas, splits, stream);
        break;
      default:
        launch_main<NG, 6>(p, warps_n, warps_k, ctas, splits, stream);
        break;
    }
  }
}

void check_config(int64_t ng, int64_t warps_n, int64_t warps_k, int64_t splits,
                  int k, int m) {
  TORCH_CHECK(ng == 1 || ng == 2 || ng == 4,
              "fp16_skinny_gemm: ng must be 1, 2 or 4");
  TORCH_CHECK(ng != 4 || m <= 24,
              "fp16_skinny_gemm: ng=4 supports at most 24 rows");
  TORCH_CHECK(warps_n >= 1 && warps_k >= 1 && warps_n * warps_k <= kMaxWarps,
              "fp16_skinny_gemm: need 1 <= warps_n * warps_k <= ", kMaxWarps);
  TORCH_CHECK(splits >= 1 && splits <= kMaxSplits,
              "fp16_skinny_gemm: splits must be in [1, ", kMaxSplits, "]");
  TORCH_CHECK(splits * warps_k <= k / 16,
              "fp16_skinny_gemm: splits * warps_k must not exceed K / 16");
}

void check_half(const at::Tensor& t, const at::Device& device,
                const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == device, "fp16_skinny_gemm: ", name,
              " must be CUDA on the input device");
  TORCH_CHECK(t.scalar_type() == at::kHalf, "fp16_skinny_gemm: ", name,
              " must be FP16");
}

void check_weight(const at::Tensor& w, const at::Device& device, int k,
                  const char* name) {
  check_half(w, device, name);
  TORCH_CHECK(
      w.dim() == 2 && w.size(0) >= 1 && w.size(1) == k && w.is_contiguous(),
      "fp16_skinny_gemm: ", name, " must be contiguous [N, K]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.const_data_ptr()) % 16 == 0,
              "fp16_skinny_gemm: ", name, " must be 16-byte aligned");
  TORCH_CHECK(w.size(0) <= (int64_t{1} << 24), "fp16_skinny_gemm: ", name,
              " has too many rows");
}

void check_output(const at::Tensor& out, const at::Device& device, int m,
                  int64_t n, const char* name) {
  check_half(out, device, name);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == m && out.size(1) == n &&
                  out.stride(1) == 1 && out.stride(0) >= n,
              "fp16_skinny_gemm: ", name, " must be [", m, ", ", n,
              "] with unit column stride");
}

// Validates x and returns (m, k, ldx).
std::tuple<int, int, int> check_input(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2,
              "fp16_skinny_gemm: x must be a CUDA FP16 matrix");
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  TORCH_CHECK(m >= 1 && m <= kMaxRows, "fp16_skinny_gemm: x must have 1..",
              kMaxRows, " rows");
  TORCH_CHECK(k >= 16 && k % 16 == 0 && k <= (int64_t{1} << 20),
              "fp16_skinny_gemm: K must be a positive multiple of 16");
  TORCH_CHECK(x.stride(1) == 1 && (m == 1 || x.stride(0) % 8 == 0),
              "fp16_skinny_gemm: x rows must be unit-stride and 16-byte "
              "aligned (row stride a multiple of 8)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0,
              "fp16_skinny_gemm: x must be 16-byte aligned");
  const int64_t ldx = m == 1 ? k : x.stride(0);
  TORCH_CHECK(ldx >= k, "fp16_skinny_gemm: invalid x row stride");
  return {static_cast<int>(m), static_cast<int>(k), static_cast<int>(ldx)};
}

void check_workspace(const at::Tensor& ws, const at::Device& device,
                     int64_t splits, int m, int ntot) {
  if (splits <= 1) return;
  TORCH_CHECK(ws.is_cuda() && ws.device() == device &&
                  ws.scalar_type() == at::kFloat && ws.is_contiguous() &&
                  ws.numel() >= splits * m * static_cast<int64_t>(ntot),
              "fp16_skinny_gemm: workspace must be contiguous CUDA FP32 with "
              "at least splits * M * N elements");
}

void check_sm70(const at::Device& device) {
  const auto* properties = at::cuda::getDeviceProperties(device.index());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "fp16_skinny_gemm requires SM70");
}

void run(SkinnyParams& p, int64_t ng, int64_t warps_n, int64_t warps_k,
         int64_t splits) {
  const int cta_cols = static_cast<int>(8 * ng * warps_n);
  p.seg0.cta_begin = 0;
  p.seg0.col_base = 0;
  int ctas = (p.seg0.n + cta_cols - 1) / cta_cols;
  p.ntot = p.seg0.n;
  if (p.nseg > 1) {
    p.seg1.cta_begin = ctas;
    p.seg1.col_base = p.seg0.n;
    ctas += (p.seg1.n + cta_cols - 1) / cta_cols;
    p.ntot += p.seg1.n;
  }
  TORCH_CHECK(static_cast<int64_t>(p.m) * p.ntot < (int64_t{1} << 31),
              "fp16_skinny_gemm: output too large");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int rb = (p.m + 7) / 8;
  const int wn = static_cast<int>(warps_n);
  const int wk = static_cast<int>(warps_k);
  const int s = static_cast<int>(splits);
  if (ng == 1) {
    dispatch_rb<1>(rb, p, wn, wk, ctas, s, stream);
  } else if (ng == 2) {
    dispatch_rb<2>(rb, p, wn, wk, ctas, s, stream);
  } else {
    dispatch_rb<4>(rb, p, wn, wk, ctas, s, stream);
  }
  if (s > 1) {
    const int total = p.m * p.ntot;
    fp16_skinny_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(p, s);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

SkinnySegment make_segment(const at::Tensor& w, const half* bias, half* out0,
                           half* out1, int split, int ld0, int ld1) {
  SkinnySegment seg{};
  seg.weight = reinterpret_cast<const half*>(w.const_data_ptr());
  seg.bias = bias;
  seg.out0 = out0;
  seg.out1 = out1;
  seg.n = static_cast<int>(w.size(0));
  seg.split = split;
  seg.ld0 = ld0;
  seg.ld1 = ld1;
  return seg;
}

void fp16_skinny_gemm_sm70_out(at::Tensor out, const at::Tensor& x,
                               const at::Tensor& weight,
                               const std::optional<at::Tensor>& bias,
                               at::Tensor workspace, int64_t ng,
                               int64_t warps_n, int64_t warps_k,
                               int64_t splits) {
  const auto [m, k, ldx] = check_input(x);
  const auto device = x.device();
  check_config(ng, warps_n, warps_k, splits, k, m);
  check_weight(weight, device, k, "weight");
  const int n = static_cast<int>(weight.size(0));
  check_output(out, device, m, n, "out");
  const half* bias_ptr = nullptr;
  if (bias.has_value()) {
    check_half(*bias, device, "bias");
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n && bias->is_contiguous(),
                "fp16_skinny_gemm: bias must be contiguous [N]");
    bias_ptr = reinterpret_cast<const half*>(bias->const_data_ptr());
    at::assert_no_overlap(out, *bias);
  }
  check_workspace(workspace, device, splits, m, n);
  at::assert_no_overlap(out, x);
  at::assert_no_overlap(out, weight);
  if (splits > 1) {
    at::assert_no_overlap(workspace, out);
    at::assert_no_overlap(workspace, x);
    at::assert_no_overlap(workspace, weight);
  }
  const c10::cuda::CUDAGuard guard(device);
  check_sm70(device);
  auto* out_ptr = reinterpret_cast<half*>(out.mutable_data_ptr());
  const int ld = static_cast<int>(out.stride(0));
  SkinnyParams p{};
  p.x = reinterpret_cast<const half*>(x.const_data_ptr());
  p.workspace = splits > 1 ? workspace.mutable_data_ptr<float>() : nullptr;
  p.seg0 = make_segment(weight, bias_ptr, out_ptr, out_ptr, n, ld, ld);
  p.nseg = 1;
  p.m = m;
  p.k = k;
  p.ldx = ldx;
  run(p, ng, warps_n, warps_k, splits);
}

// Two weights sharing x. Weight i's first outi_a.size(1) columns go to outi_a
// and the remaining columns to outi_b.
void fp16_skinny_gemm2_sm70_out(at::Tensor out0a, at::Tensor out0b,
                                at::Tensor out1a, at::Tensor out1b,
                                const at::Tensor& x, const at::Tensor& w0,
                                const at::Tensor& w1, at::Tensor workspace,
                                int64_t ng, int64_t warps_n, int64_t warps_k,
                                int64_t splits) {
  const auto [m, k, ldx] = check_input(x);
  const auto device = x.device();
  check_config(ng, warps_n, warps_k, splits, k, m);
  check_weight(w0, device, k, "w0");
  check_weight(w1, device, k, "w1");
  const int n0 = static_cast<int>(w0.size(0));
  const int n1 = static_cast<int>(w1.size(0));
  TORCH_CHECK(out0a.dim() == 2 && out1a.dim() == 2,
              "fp16_skinny_gemm2: outputs must be matrices");
  const int split0 = static_cast<int>(out0a.size(1));
  const int split1 = static_cast<int>(out1a.size(1));
  TORCH_CHECK(split0 >= 1 && split0 < n0 && split1 >= 1 && split1 < n1,
              "fp16_skinny_gemm2: each weight needs two non-empty outputs");
  check_output(out0a, device, m, split0, "out0a");
  check_output(out0b, device, m, n0 - split0, "out0b");
  check_output(out1a, device, m, split1, "out1a");
  check_output(out1b, device, m, n1 - split1, "out1b");
  check_workspace(workspace, device, splits, m, n0 + n1);
  const at::Tensor* outs[] = {&out0a, &out0b, &out1a, &out1b};
  for (int i = 0; i < 4; ++i) {
    for (const auto* t : {&x, &w0, &w1}) at::assert_no_overlap(*outs[i], *t);
    for (int j = i + 1; j < 4; ++j) at::assert_no_overlap(*outs[i], *outs[j]);
    if (splits > 1) at::assert_no_overlap(workspace, *outs[i]);
  }
  const c10::cuda::CUDAGuard guard(device);
  check_sm70(device);
  SkinnyParams p{};
  p.x = reinterpret_cast<const half*>(x.const_data_ptr());
  p.workspace = splits > 1 ? workspace.mutable_data_ptr<float>() : nullptr;
  p.seg0 = make_segment(
      w0, nullptr, reinterpret_cast<half*>(out0a.mutable_data_ptr()),
      reinterpret_cast<half*>(out0b.mutable_data_ptr()), split0,
      static_cast<int>(out0a.stride(0)), static_cast<int>(out0b.stride(0)));
  p.seg1 = make_segment(
      w1, nullptr, reinterpret_cast<half*>(out1a.mutable_data_ptr()),
      reinterpret_cast<half*>(out1b.mutable_data_ptr()), split1,
      static_cast<int>(out1a.stride(0)), static_cast<int>(out1b.stride(0)));
  p.nseg = 2;
  p.m = m;
  p.k = k;
  p.ldx = ldx;
  run(p, ng, warps_n, warps_k, splits);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "fp16_skinny_gemm_sm70_out(Tensor(a!) out, Tensor x, Tensor weight, "
      "Tensor? bias, Tensor(b!) workspace, int ng, int warps_n, int warps_k, "
      "int splits) -> ()");
  m.def(
      "fp16_skinny_gemm2_sm70_out(Tensor(a!) out0a, Tensor(b!) out0b, "
      "Tensor(c!) out1a, Tensor(d!) out1b, Tensor x, Tensor w0, Tensor w1, "
      "Tensor(e!) workspace, int ng, int warps_n, int warps_k, int splits) "
      "-> ()");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("fp16_skinny_gemm_sm70_out", &fp16_skinny_gemm_sm70_out);
  m.impl("fp16_skinny_gemm2_sm70_out", &fp16_skinny_gemm2_sm70_out);
}
