// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, D.Skryabin
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Private copies of flash_decode_paged.cu helpers keep the legacy TU untouched.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "fused_mma.h"

namespace {
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kXQATCBlockN = 128;
constexpr int kXQATCStride = 128;
// QSA exposes selected four-token microblocks as virtual paged KV. Keep enough
// page slots for one full XQA tile at that minimum supported granularity.
constexpr int kXQATCPageIdsCapacity = kXQATCBlockN / 4;
constexpr int kXQATC256WideWarpCount = 8;
constexpr int kXQATC256WideThreads = kXQATC256WideWarpCount * kWarpSize;
constexpr int kXQATC256WideBlockM = 8;
constexpr int kXQATC256WideThreadsPerRow = kWarpSize;
constexpr int kXQATCG6DualCtaThreads = 6 * kWarpSize;
constexpr int kXQATCG6Pipeline8WarpThreads = 8 * kWarpSize;
constexpr int kXQARouteAllSeqLens = 0;
constexpr int kXQARouteShortSeqLens = -1;
constexpr int kXQARouteLongSeqLens = 1;
constexpr int kXQARouteP1024Sawtooth = 4;
constexpr int kXQARouteP256Sawtooth = 5;
constexpr int kXQARouteP1024SawtoothMid = 6;
constexpr int kXQARouteP1024SawtoothFinal = 7;
constexpr int kXQARouteRangeSeqLens = 8;
constexpr int kXQARouteWaveLongSeqLens = 9;
// The dense 256/128-half strides alias Volta shared-memory banks in the
// WMMA A/B fragment loads. Both padded strides remain 16-byte aligned.
constexpr int kXQATC256WidePaddedQStride = 264;
constexpr int kXQATC256WidePaddedKVStride = 136;
constexpr int kXQATC256WideAlignedPaddedQStride = 272;
constexpr int kXQATC256WideAlignedPaddedKVStride = 144;
constexpr int kXQATCQKPipelinePanelDim = 64;
constexpr int kXQATCQKPipelineKVStride = 72;
constexpr float kXQANegInf = -1.0e30f;

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  static constexpr int kQKStride = kKVStride;
  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    alignas(16) __half k[kXQATCBlockN * kKVStride];
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];

  __device__ __forceinline__ __half* k_buffer(int) { return reuse_kv.k; }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCQKPipelineSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  static constexpr int kQKStride = kXQATCQKPipelineKVStride;
  struct QKBuffers {
    alignas(16) __half panel[2][kXQATCBlockN * kQKStride];
  };

  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    QKBuffers qk;
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];

  __device__ __forceinline__ __half* k_buffer(int index) {
    return reuse_kv.qk.panel[index];
  }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

template <int SEQ_LEN_ROUTE>
__device__ __forceinline__ bool xqa_seq_len_route_active(
    const int seq_len, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  if constexpr (SEQ_LEN_ROUTE == kXQARouteShortSeqLens) {
    return seq_len < route_seq_len_begin;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteLongSeqLens) {
    return seq_len >= route_seq_len_begin;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024Sawtooth) {
    return (seq_len >= route_seq_len_begin && seq_len < route_seq_len_end) ||
           seq_len >= route_seq_len_final;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP256Sawtooth) {
    return seq_len < route_seq_len_begin ||
           (seq_len >= route_seq_len_end && seq_len < route_seq_len_final);
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024SawtoothMid) {
    return seq_len >= route_seq_len_begin && seq_len < route_seq_len_end;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024SawtoothFinal) {
    return seq_len >= route_seq_len_final;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteRangeSeqLens) {
    return seq_len >= route_seq_len_begin && seq_len < route_seq_len_end;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens) {
    return seq_len >= route_seq_len_begin;
  }
  return true;
}

__device__ __forceinline__ int xqa_sawtooth_partition_size(
    const int seq_len, const int p1024_mid_seq_len, const int p256_long_seq_len,
    const int p1024_final_seq_len) {
  const bool use_p1024 =
      (seq_len >= p1024_mid_seq_len && seq_len < p256_long_seq_len) ||
      seq_len >= p1024_final_seq_len;
  return use_p1024 ? 1024 : 256;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : 0.f;
  if (warp == 0) {
    val = warp_reduce_sum(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_max(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : -1.0e20f;
  if (warp == 0) {
    val = warp_reduce_max(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

constexpr int kGroupedVerifyQ8MaxQ = 8;
constexpr int kGroupedVerifyQ16MaxQ = 16;
constexpr int kGroupedVerifyMaxSupportedQ = kGroupedVerifyQ16MaxQ;
constexpr int kGroupedVerifyHeads = 6;
constexpr int kGroupedVerifyHeadDim = 256;
constexpr int kGroupedVerifyRows = 48;
constexpr int kGroupedVerifyBlockN = 32;
constexpr int kGroupedVerifyQStride = 264;
constexpr int kGroupedVerifyKVStride = 264;
constexpr int kGroupedVerifyScoreStride = 32;
constexpr int kGroupedVerifyProbStride = 40;
constexpr int kGroupedVerifyPageIdsCapacity = 16;
// One packed head group times eighty context splits maps one 512-thread CTA to
// each of V100's eighty SMs. Short sequences still reduce active_splits using
// kGroupedVerifyMinTokensPerSplit inside the captured graph.
constexpr int kGroupedVerifyQ8Splits = 80;
constexpr int kGroupedVerifyWorkspaceRows =
    kGroupedVerifyQ8MaxQ * kGroupedVerifyQ8Splits;
// A packed CTA replaces two 256-thread head-group CTAs and can occupy only one
// SM. Use 64-token chunks to expose enough short-context parallelism without
// paying one split reduction per N32 tile; longer contexts saturate at the
// unchanged eighty splits. The single-query utility variant keeps the old
// 128-token split contract for its stricter FP32-reference error envelope.
constexpr int kGroupedVerifyMinTokensPerSplit = 64;
constexpr int kGroupedVerifySingleQueryMinTokensPerSplit = 128;
constexpr int kGroupedVerifyShortContextMaxTokens = 128;
constexpr int kGroupedVerifyThreads = 512;
constexpr int kGroupedVerifyWarps = kGroupedVerifyThreads / kWarpSize;
constexpr int kGroupedVerifyQKWarps =
    (kGroupedVerifyRows / 16) * (kGroupedVerifyBlockN / 16);
constexpr int kGroupedVerifyOutputTiles =
    (kGroupedVerifyRows / 16) * (kGroupedVerifyHeadDim / 16);
constexpr int kGroupedVerifyOutputTilesPerWarp =
    kGroupedVerifyOutputTiles / kGroupedVerifyWarps;

template <int MAX_QUERY_TOKENS>
struct GroupedVerifyTraits {
  static_assert(MAX_QUERY_TOKENS == kGroupedVerifyQ8MaxQ ||
                    MAX_QUERY_TOKENS == kGroupedVerifyQ16MaxQ,
                "grouped verifier supports q8 and q16 workspaces only");
  static constexpr int kHeadsPerCta = kGroupedVerifyRows / MAX_QUERY_TOKENS;
  static constexpr int kHeadGroups = kGroupedVerifyHeads / kHeadsPerCta;
  static constexpr int kSplits = kGroupedVerifyWorkspaceRows / MAX_QUERY_TOKENS;
  static_assert(MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows,
                "grouped verifier CTA must retain 48 rows");
  static_assert(kGroupedVerifyHeads % kHeadsPerCta == 0,
                "query heads must divide evenly across grouped CTAs");
  static_assert(kSplits * MAX_QUERY_TOKENS == kGroupedVerifyWorkspaceRows,
                "q8 and q16 workspaces must have equal element counts");
};

struct alignas(256) GroupedVerifySmem {
  union {
    struct {
      alignas(16) __half q[kGroupedVerifyRows * kGroupedVerifyQStride];
      alignas(16) __half kv[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
      alignas(16) float scores[kGroupedVerifyRows * kGroupedVerifyScoreStride];
      alignas(16) __half probs[kGroupedVerifyRows * kGroupedVerifyProbStride];
    } compute;
    alignas(16) float output[kGroupedVerifyRows * kGroupedVerifyHeadDim];
  } storage;
  alignas(16) float row_max[kGroupedVerifyRows];
  alignas(16) float row_sum[kGroupedVerifyRows];
  alignas(16) float row_scale[kGroupedVerifyRows];
  alignas(16) int page_ids[kGroupedVerifyPageIdsCapacity];
  alignas(16) uint32_t sparse_token_masks[kGroupedVerifyBlockN / 4];
};

static_assert(sizeof(GroupedVerifySmem) <= 64 * 1024,
              "packed grouped verifier must fit Volta's 64 KiB opt-in budget");
static_assert(kGroupedVerifyOutputTiles % kGroupedVerifyWarps == 0,
              "output tiles must divide evenly across warps");

template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY>
__device__ __forceinline__ int grouped_verify_active_splits(
    const int total_kv) {
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  constexpr int kMinTokensPerSplit =
      SINGLE_QUERY ? kGroupedVerifySingleQueryMinTokensPerSplit
                   : kGroupedVerifyMinTokensPerSplit;
  const int active_splits =
      min(Traits::kSplits,
          max(1, (total_kv + kMinTokensPerSplit - 1) / kMinTokensPerSplit));
  if constexpr (SINGLE_QUERY) {
    return active_splits;
  }
  return total_kv <= kGroupedVerifyShortContextMaxTokens ? 1 : active_splits;
}

template <bool COMPENSATE = false>
__device__ __forceinline__ void grouped_verify_qk(
    const __half* __restrict__ shared_q, const __half* __restrict__ shared_k,
    float* __restrict__ shared_scores, const float qk_scale,
    const int active_m_tiles) {
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kGroupedVerifyQKWarps) {
    return;
  }

  const int m_tile = warp_id / (kGroupedVerifyBlockN / 16);
  if ((active_m_tiles & (1 << m_tile)) == 0) {
    return;
  }
  const int n_tile = warp_id % (kGroupedVerifyBlockN / 16);
  volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
      q_fragment;
  volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::col_major>
      k_fragment;
  volta::fragment<volta::accumulator, 16, 16, 16, float> score_fragment;
  volta::fill_fragment(score_fragment, 0.0f);
  float correction[8] = {};

#pragma unroll
  for (int k_offset = 0; k_offset < kGroupedVerifyHeadDim; k_offset += 16) {
    volta::load_matrix_sync(
        q_fragment, shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset,
        kGroupedVerifyQStride);
    // K is row-major [N, D].  The same bytes represent K^T as a col-major
    // [D, N] matrix, which is the B operand needed by Q @ K^T.
    volta::load_matrix_sync(
        k_fragment, shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset,
        kGroupedVerifyKVStride);
    if constexpr (COMPENSATE) {
      // E4M3 x FP16 products fit comfortably in FP32, but a D256 Tensor
      // Core accumulation can still lose low bits. Sum short K16 products
      // with compensated FP32 additions; explicit RN operations preserve
      // the correction under the standard fast-math build.
      volta::fragment<volta::accumulator, 16, 16, 16, float> tile_fragment;
      volta::fill_fragment(tile_fragment, 0.0f);
      volta::mma_sync(tile_fragment, q_fragment, k_fragment, tile_fragment);
#pragma unroll
      for (int i = 0; i < score_fragment.num_elements; ++i) {
        const float y = __fsub_rn(tile_fragment.x[i], correction[i]);
        const float sum = __fadd_rn(score_fragment.x[i], y);
        correction[i] = __fsub_rn(__fsub_rn(sum, score_fragment.x[i]), y);
        score_fragment.x[i] = sum;
      }
    } else {
      volta::mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);
    }
  }
#pragma unroll
  for (int i = 0; i < score_fragment.num_elements; ++i) {
    score_fragment.x[i] *= qk_scale;
  }
  volta::store_matrix_sync(
      shared_scores + m_tile * 16 * kGroupedVerifyScoreStride + n_tile * 16,
      score_fragment, kGroupedVerifyScoreStride, volta::mem_row_major);
}

__device__ __forceinline__ void grouped_verify_scale_output_fragment(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& fragment,
    const float* __restrict__ row_scale, const int tile_row_start) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const float first_scale = row_scale[tile_row_start + row];
  const float second_scale = row_scale[tile_row_start + row + 2];
  fragment.x[0] *= first_scale;
  fragment.x[1] *= first_scale;
  fragment.x[2] *= second_scale;
  fragment.x[3] *= second_scale;
  fragment.x[4] *= first_scale;
  fragment.x[5] *= first_scale;
  fragment.x[6] *= second_scale;
  fragment.x[7] *= second_scale;
}

__device__ __forceinline__ void grouped_verify_add_output_tile(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& output,
    const volta::fragment<volta::accumulator, 16, 16, 16, float>& tile,
    const float* row_scale, const int tile_row_start) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const float first_scale = row_scale[tile_row_start + row];
  const float second_scale = row_scale[tile_row_start + row + 2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float scale = (i & 2) ? second_scale : first_scale;
    output.x[i] = fmaf(output.x[i], scale, tile.x[i]);
  }
}

template <bool SPARSE_PAGE4, bool ROW_SEQLENS = false>
__device__ __forceinline__ bool grouped_verify_key_visible(
    const uint32_t* __restrict__ sparse_tile_masks, const int token_idx,
    const int query_len, const int head_idx, const int kv_idx,
    const int valid_k_rows, const int lane_or_col, const int prefix_kv_len,
    const int* row_lengths = nullptr) {
  const bool row_valid = token_idx < query_len &&
                         head_idx < kGroupedVerifyHeads &&
                         lane_or_col < valid_k_rows;
  if (!row_valid) {
    return false;
  }
  if constexpr (SPARSE_PAGE4) {
    const uint32_t token_mask =
        sparse_tile_masks[(kv_idx & (kGroupedVerifyBlockN - 1)) >> 2];
    const int mask_bit = token_idx * 4 + (kv_idx & 3);
    return (token_mask & (1u << mask_bit)) != 0;
  }
  if constexpr (ROW_SEQLENS) {
    return kv_idx < row_lengths[token_idx];
  }
  return kv_idx <= prefix_kv_len + token_idx;
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens>
__global__ void flash_attention_decode_xqa_reduce_stats_kernel(
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t stats_stride0, const int64_t stats_stride1,
    const int route_seq_len_begin, const int route_seq_len_end,
    const int route_seq_len_final) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int partition_size =
      PARTITION_SIZE == 0
          ? xqa_sawtooth_partition_size(seq_len, route_seq_len_begin,
                                        route_seq_len_end, route_seq_len_final)
          : PARTITION_SIZE;
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || num_partitions <= 0) {
    return;
  }

  extern __shared__ float max_shared[];
  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;

  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float m = max_logits[stats_base + i];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float global_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float weight =
        exp_sums[stats_base + i] * __expf(max_shared[i] - global_max);
    max_logits[stats_base + i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  if (threadIdx.x == 0) {
    // The partition sum is dead after weights have been materialized.
    exp_sums[stats_base] = global_sum;
  }
}

template <int D, int PARTITION_SIZE, int D_TILE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, typename PARTIAL_T = __half>
__global__ void flash_attention_decode_xqa_reduce_output_kernel(
    const PARTIAL_T* __restrict__ tmp_out, const float* __restrict__ weights,
    const float* __restrict__ global_sums, const int* __restrict__ seq_lens,
    __half* __restrict__ out, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  static_assert(D_TILE > 0 && D_TILE <= kWarpSize,
                "Split-reduce dimension tile must fit in one warp");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int d = blockIdx.z * D_TILE + threadIdx.x;
  if (batch_idx >= batch_size || head_idx >= num_heads_q || d >= D) {
    return;
  }

  const int64_t out_index = static_cast<int64_t>(batch_idx) * out_stride0 +
                            static_cast<int64_t>(head_idx) * out_stride1 + d;
  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int partition_size =
      PARTITION_SIZE == 0
          ? xqa_sawtooth_partition_size(seq_len, route_seq_len_begin,
                                        route_seq_len_end, route_seq_len_final)
          : PARTITION_SIZE;
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || num_partitions <= 0) {
    out[out_index] = __float2half(0.f);
    return;
  }

  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;
  const float global_sum = global_sums[stats_base];
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;

  float acc = 0.f;
  for (int i = 0; i < num_partitions; ++i) {
    acc = fmaf(weights[stats_base + i],
               static_cast<float>(
                   tmp_out[tmp_out_base +
                           static_cast<int64_t>(i) * tmp_out_stride2 + d]),
               acc);
  }
  out[out_index] = __float2half(acc * inv_global_sum);
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          typename PARTIAL_T = __half>
void launch_flash_attention_decode_xqa_split_reduce(
    at::Tensor& out, const at::Tensor& seq_lens, const at::Tensor& tmp_out,
    at::Tensor& max_logits, at::Tensor& exp_sums,
    const int launch_num_partitions, const int dim_tile, cudaStream_t stream,
    const int route_seq_len_begin = 0, const int route_seq_len_end = 0,
    const int route_seq_len_final = 0) {
  const int batch_size = out.size(0);
  const int num_heads_q = out.size(1);
  const dim3 stats_grid(batch_size, num_heads_q, 1);
  const dim3 stats_block(kThreadsPerBlock);
  const size_t stats_shared_mem =
      static_cast<size_t>(launch_num_partitions) * sizeof(float);
  flash_attention_decode_xqa_reduce_stats_kernel<PARTITION_SIZE, SEQ_LEN_ROUTE>
      <<<stats_grid, stats_block, stats_shared_mem, stream>>>(
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          seq_lens.data_ptr<int>(), batch_size, launch_num_partitions,
          num_heads_q, max_logits.stride(0), max_logits.stride(1),
          route_seq_len_begin, route_seq_len_end, route_seq_len_final);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_SPLIT_REDUCE_OUTPUT(D_TILE)                                   \
  do {                                                                       \
    const dim3 output_grid(batch_size, num_heads_q,                          \
                           (256 + D_TILE - 1) / D_TILE);                     \
    const dim3 output_block(D_TILE);                                         \
    flash_attention_decode_xqa_reduce_output_kernel<                         \
        256, PARTITION_SIZE, D_TILE, SEQ_LEN_ROUTE, PARTIAL_T>               \
        <<<output_grid, output_block, 0, stream>>>(                          \
            reinterpret_cast<const PARTIAL_T*>(tmp_out.data_ptr()),          \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),        \
            seq_lens.data_ptr<int>(),                                        \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size, \
            launch_num_partitions, num_heads_q, tmp_out.stride(0),           \
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),      \
            max_logits.stride(1), out.stride(0), out.stride(1),              \
            route_seq_len_begin, route_seq_len_end, route_seq_len_final);    \
  } while (0)

  switch (dim_tile) {
    case 16:
      LAUNCH_SPLIT_REDUCE_OUTPUT(16);
      break;
    case 32:
      LAUNCH_SPLIT_REDUCE_OUTPUT(32);
      break;
    default:
      LAUNCH_SPLIT_REDUCE_OUTPUT(8);
      break;
  }
#undef LAUNCH_SPLIT_REDUCE_OUTPUT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

#include "int8_page4.cuh"
