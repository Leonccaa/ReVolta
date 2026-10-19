// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, D.Skryabin
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// INT8-only specialization adapted from 1CatAI/1Cat-vLLM #1193/#1194.
// Compile separately so legacy templates and PTX remain unchanged.
#pragma once
#include "int8_kv_traits.cuh"
namespace int8_page4 {
template <bool PROPAGATE_NAN>
__device__ __forceinline__ float clamped_softmax_exp(float difference) {
  // Row-scaled codecs encode invalid input rows with a NaN scale. Do not
  // let fmax turn those rows into finite probabilities. Legacy arithmetic
  // retains its original clamp semantics.
  if constexpr (PROPAGATE_NAN) {
    if (isnan(difference)) return difference;
  }
  return __expf(fmaxf(difference, -80.0f));
}
template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          int KV_DTYPE = int8_kv::KV_CACHE_DTYPE_FP16,
          bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ uint4 load_xqa_tc_kv_vector(
    const void* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int copy_idx, const int panel_d_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const uint16_t* __restrict__ e4m3_lut = nullptr,
    const int8_kv::KVPageMapping mapping = {}) {
  const int row = copy_idx / panel_d_stride_uint4;
  const int vec_col = copy_idx % panel_d_stride_uint4;
  const int token_offset = tile_page_offset + kv_tile_start + row;
  static_assert(BLOCK_SIZE == 0 || BLOCK_SIZE == 4 || BLOCK_SIZE == 16 ||
                    BLOCK_SIZE == 784 || BLOCK_SIZE == 800 ||
                    BLOCK_SIZE == 1568 || BLOCK_SIZE == 1648 ||
                    BLOCK_SIZE == 3296,
                "Unsupported paged-KV block-size specialization");
  static_assert(!CONTIGUOUS_HKV1_LAYOUT || BLOCK_SIZE == 16 ||
                    BLOCK_SIZE == 800 || BLOCK_SIZE == 1568 ||
                    BLOCK_SIZE == 1648 || BLOCK_SIZE == 3296,
                "The fixed-stride Hkv=1 layout requires a specialized page");
  int logical_block;
  int block_offset;
  if constexpr (BLOCK_SIZE == 4) {
    logical_block = token_offset >> 2;
    block_offset = token_offset & 3;
  } else if constexpr (BLOCK_SIZE == 16) {
    logical_block = token_offset >> 4;
    block_offset = token_offset & 15;
  } else if constexpr (BLOCK_SIZE == 784) {
    logical_block = token_offset / 784;
    block_offset = token_offset - logical_block * 784;
  } else if constexpr (BLOCK_SIZE == 800) {
    // E4M3 long-wave partitions can span four 800-token pages. Keep the
    // constant-page specialization division-free without truncating at page 1.
    logical_block =
        (token_offset >= 800) + (token_offset >= 1600) + (token_offset >= 2400);
    block_offset = token_offset - logical_block * 800;
  } else if constexpr (BLOCK_SIZE == 1568) {
    logical_block = token_offset / 1568;
    block_offset = token_offset - logical_block * 1568;
  } else if constexpr (BLOCK_SIZE == 1648) {
    logical_block = token_offset / 1648;
    block_offset = token_offset - logical_block * 1648;
  } else if constexpr (BLOCK_SIZE == 3296) {
    logical_block = token_offset / 3296;
    block_offset = token_offset - logical_block * 3296;
  } else {
    logical_block = token_offset / block_size;
    block_offset = token_offset % block_size;
  }
  const int physical_block = page_ids[logical_block];
  int64_t physical_offset;
  static_assert(!CONTIGUOUS_HKV1_LAYOUT,
                "INT8 readers require explicit payload strides");
  {
    physical_offset = mapping.offset(physical_block, block_offset, kv_head_idx,
                                     block_stride, token_stride, head_stride) +
                      panel_offset;
  }
  return int8_kv::KVCodecTraits<KV_DTYPE>::template load_half8<E4M3_SHARED_LUT>(
      kv_cache, physical_offset, vec_col, e4m3_lut);
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = int8_kv::KV_CACHE_DTYPE_FP16,
          bool FP8_PAIR_LOAD = false, bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx = threadIdx.x,
    const uint16_t* __restrict__ e4m3_lut = nullptr,
    const int8_kv::KVPageMapping mapping = {}) {
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  if constexpr (FP8_PAIR_LOAD) {
    static_assert(int8_kv::KVCodecTraits<KV_DTYPE>::paired_byte_load,
                  "Paired XQA loads require a byte codec");
    static_assert(!E4M3_SHARED_LUT,
                  "Paired E4M3 conversion does not use the shared LUT");
    const int pair_stride = panel_d_stride_uint4 / 2;
    const int pair_count = valid_kv_tile_rows * pair_stride;
    for (int pair_idx = copy_thread_idx; pair_idx < pair_count;
         pair_idx += NUM_THREADS) {
      const int row = pair_idx / pair_stride;
      const int vec_pair = pair_idx % pair_stride;
      const int token_offset = tile_page_offset + kv_tile_start + row;
      int logical_block;
      int block_offset;
      if constexpr (BLOCK_SIZE == 4) {
        logical_block = token_offset >> 2;
        block_offset = token_offset & 3;
      } else if constexpr (BLOCK_SIZE == 16) {
        logical_block = token_offset >> 4;
        block_offset = token_offset & 15;
      } else if constexpr (BLOCK_SIZE == 784) {
        logical_block = token_offset / 784;
        block_offset = token_offset - logical_block * 784;
      } else if constexpr (BLOCK_SIZE == 800) {
        logical_block = (token_offset >= 800) + (token_offset >= 1600) +
                        (token_offset >= 2400);
        block_offset = token_offset - logical_block * 800;
      } else if constexpr (BLOCK_SIZE == 1568) {
        logical_block = token_offset / 1568;
        block_offset = token_offset - logical_block * 1568;
      } else if constexpr (BLOCK_SIZE == 1648) {
        logical_block = token_offset / 1648;
        block_offset = token_offset - logical_block * 1648;
      } else if constexpr (BLOCK_SIZE == 3296) {
        logical_block = token_offset / 3296;
        block_offset = token_offset - logical_block * 3296;
      } else {
        logical_block = token_offset / block_size;
        block_offset = token_offset % block_size;
      }
      const int physical_block = page_ids[logical_block];
      int64_t physical_offset;
      static_assert(!CONTIGUOUS_HKV1_LAYOUT,
                    "INT8 readers require explicit payload strides");
      {
        physical_offset =
            mapping.offset(physical_block, block_offset, kv_head_idx,
                           block_stride, token_stride, head_stride) +
            panel_offset;
      }
      const uint4 raw = __ldg(reinterpret_cast<const uint4*>(kv_cache) +
                              physical_offset / 16 + vec_pair);
      const int shared_offset = row * kv_smem_stride_uint4 + vec_pair * 2;
      const uint64_t raw_lo =
          static_cast<uint64_t>(raw.x) | (static_cast<uint64_t>(raw.y) << 32);
      shared_vec[shared_offset] =
          int8_kv::KVCodecTraits<KV_DTYPE>::template half8_from_packed<true>(
              raw_lo);
      const uint64_t raw_hi =
          static_cast<uint64_t>(raw.z) | (static_cast<uint64_t>(raw.w) << 32);
      shared_vec[shared_offset + 1] =
          int8_kv::KVCodecTraits<KV_DTYPE>::template half8_from_packed<true>(
              raw_hi);
    }
  } else {
    const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
    for (int copy_idx = copy_thread_idx; copy_idx < copy_count;
         copy_idx += NUM_THREADS) {
      const int row = copy_idx / panel_d_stride_uint4;
      const int vec_col = copy_idx % panel_d_stride_uint4;
      shared_vec[row * kv_smem_stride_uint4 + vec_col] =
          load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                E4M3_SHARED_LUT>(
              kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
              tile_page_offset, kv_tile_start, block_size, kv_head_idx,
              block_stride, token_stride, head_stride, panel_offset, e4m3_lut,
              mapping);
    }
  }
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = int8_kv::KV_CACHE_DTYPE_FP16,
          bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel_and_zero(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx,
    const uint16_t* __restrict__ e4m3_lut = nullptr,
    const int8_kv::KVPageMapping mapping = {}) {
  const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  constexpr int kLoadStages = 4;
  for (int copy_base = copy_thread_idx; copy_base < copy_count;
       copy_base += NUM_THREADS * kLoadStages) {
    uint4 staged[kLoadStages];
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        staged[stage] =
            load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                  E4M3_SHARED_LUT>(
                kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
                tile_page_offset, kv_tile_start, block_size, kv_head_idx,
                block_stride, token_stride, head_stride, panel_offset, e4m3_lut,
                mapping);
      }
    }
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        const int row = copy_idx / panel_d_stride_uint4;
        const int vec_col = copy_idx % panel_d_stride_uint4;
        shared_vec[row * kv_smem_stride_uint4 + vec_col] = staged[stage];
      }
    }
  }
  for (int copy_idx = copy_thread_idx + copy_count;
       copy_idx < kXQATCBlockN * panel_d_stride_uint4;
       copy_idx += NUM_THREADS) {
    const int row = copy_idx / panel_d_stride_uint4;
    const int vec_col = copy_idx % panel_d_stride_uint4;
    shared_vec[row * kv_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
}
template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM, int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          bool ALIGNED_PADDED_SMEM, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          bool E4M3_SHARED_LUT = false, typename PARTIAL_T = __half>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_partition_kernel_256_wide(
        const __half* __restrict__ q, const void* __restrict__ k_cache,
        const void* __restrict__ v_cache, PARTIAL_T* __restrict__ tmp_out,
        float* __restrict__ max_logits, float* __restrict__ exp_sums,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const int64_t v_block_stride, const int64_t v_token_stride,
        const int64_t v_head_stride, const float softmax_scale,
        const float k_scale, const float v_scale, const int route_seq_len_begin,
        const int route_seq_len_end, const int route_seq_len_final,
        const int8_kv::KVScaleView scales) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPVPanelDim = kXQATCStride;
  constexpr int kNumPVPanels = D / kPVPanelDim;
  constexpr int kQKPanelDim =
      QK_SW_PIPELINE ? kXQATCQKPipelinePanelDim : kXQATCStride;
  constexpr int kNumQKPanels = D / kQKPanelDim;
  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  // Keep softmax P in fp32 through PV for FP16 and E4M3 KV, avoiding a half
  // round-trip before the existing fp32 FMA accumulation. Preserve E5M2's
  // established half-P numerical contract. This branch is compile-time only.
  constexpr bool kKeepPfp32 = KV_DTYPE != int8_kv::KV_CACHE_DTYPE_FP8_E5M2;
  constexpr bool kPropagateNaN =
      int8_kv::KVCodecTraits<KV_DTYPE>::per_token_head;
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int pv_panel_d_stride_uint4 = kPVPanelDim / 8;
  constexpr int qk_smem_stride_uint4 =
      QK_SW_PIPELINE ? SmemLayout::kQKStride / 8 : kv_smem_stride_uint4;
  constexpr int qk_panel_d_stride_uint4 = kQKPanelDim / 8;
  constexpr int kAccumsPerThread = D / kWarpSize;
  constexpr bool kPVHalf2 = CONTIGUOUS_HKV1_LAYOUT &&
                            (BLOCK_SIZE == 800 || BLOCK_SIZE == 1568) &&
                            int8_kv::KVCodecTraits<KV_DTYPE>::quantized &&
                            int8_kv::KVCodecTraits<KV_DTYPE>::fp32_partials;
  static_assert(GROUP_SIZE == 4 || GROUP_SIZE == 6 || GROUP_SIZE == 8,
                "Wide D=256 TC XQA kernel supports q_per_kv in {4, 6, 8}");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one full softmax/PV warp");
  static_assert(
      !QK_SW_PIPELINE || (GROUP_SIZE == 6 && (NUM_THREADS == 6 * kWarpSize ||
                                              NUM_THREADS == 8 * kWarpSize)),
      "The QK software pipeline requires a six- or eight-warp G6 "
      "kernel");
  static_assert(
      !E4M3_SHARED_LUT || KV_DTYPE == int8_kv::KV_CACHE_DTYPE_FP8_E4M3,
      "The shared conversion LUT is only valid for E4M3 KV");
  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  int partition_size;
  if constexpr (PARTITION_SIZE == -1) {
    static_assert(SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens,
                  "Runtime wave partitions require the wave-long route");
    partition_size = seq_len < route_seq_len_end
                         ? 512
                         : (seq_len < route_seq_len_final ? 896 : 1664);
  } else if constexpr (PARTITION_SIZE == 0) {
    const int partition_size_begin =
        route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin;
    partition_size = seq_len < partition_size_begin ? 64 : 256;
  } else {
    partition_size = PARTITION_SIZE;
  }
  const int start_token_idx = partition_idx * partition_size;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + partition_size - 1) / partition_size;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(partition_size, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  uint16_t* e4m3_lut = nullptr;
  if constexpr (E4M3_SHARED_LUT) {
    e4m3_lut = reinterpret_cast<uint16_t*>(smem_raw + sizeof(SmemLayout));
    for (int raw = tid; raw < 256; raw += NUM_THREADS) {
      e4m3_lut[raw] =
          int8_kv::fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw));
    }
  }
  __half* sQ = smem.q;
  __half* sK = smem.k_buffer(0);
  __half* sV = smem.v_buffer();
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float out_acc[kAccumsPerThread];
#pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  int partition_start_page = 0;
  int partition_page_offset = 0;
  if constexpr (PARTITION_PAGE_IDS) {
    int partition_page_count;
    if constexpr (BLOCK_SIZE == 800) {
      partition_start_page = start_token_idx / 800;
      partition_page_offset = start_token_idx - partition_start_page * 800;
      partition_page_count = (partition_page_offset + part_tokens + 799) / 800;
    } else {
      partition_start_page = start_token_idx / block_size;
      partition_page_offset =
          start_token_idx - partition_start_page * block_size;
      partition_page_count =
          (partition_page_offset + part_tokens + block_size - 1) / block_size;
    }
    for (int idx = tid; idx < partition_page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[partition_start_page + idx]);
    }
    __syncthreads();
  }

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    int start_page = partition_start_page;
    int tile_page_offset;
    int page_count = 0;
    if constexpr (PARTITION_PAGE_IDS) {
      tile_page_offset = partition_page_offset + block_n * kXQATCBlockN;
    } else if constexpr (BLOCK_SIZE == 16) {
      start_page = tile_token_start >> 4;
      tile_page_offset = tile_token_start & 15;
      page_count = (tile_page_offset + valid_k_rows + 15) >> 4;
    } else if constexpr (BLOCK_SIZE == 784) {
      start_page = tile_token_start / 784;
      tile_page_offset = tile_token_start - start_page * 784;
      page_count = (tile_page_offset + valid_k_rows + 783) / 784;
    } else if constexpr (BLOCK_SIZE == 800) {
      start_page = tile_token_start / 800;
      tile_page_offset = tile_token_start - start_page * 800;
      page_count = 1 + (tile_page_offset + valid_k_rows > 800);
    } else if constexpr (BLOCK_SIZE == 1568) {
      start_page = tile_token_start / 1568;
      tile_page_offset = tile_token_start - start_page * 1568;
      page_count = (tile_page_offset + valid_k_rows + 1567) / 1568;
    } else {
      start_page = tile_token_start / block_size;
      tile_page_offset = tile_token_start - start_page * block_size;
      page_count =
          (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    }

    if constexpr (!PARTITION_PAGE_IDS) {
      for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
        smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
      }
      __syncthreads();
    }

    for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
         kv_tile_start += kXQATCBlockN) {
      const int valid_kv_tile_rows =
          min(kXQATCBlockN, valid_k_rows - kv_tile_start);
      // The QK fragments die before softmax/PV. This permits the compiler to
      // reuse their registers for the memory-latency-sensitive PV phase.
      volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::row_major>
          qk_a_frag;
      volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::col_major>
          qk_b_frag;
      volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
          qk_acc_frag;
      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        volta::fill_fragment(qk_acc_frag, 0.0f);
      }

      if constexpr (QK_SW_PIPELINE) {
        constexpr int kConsumerWarps = kXQATCBlockN / WMMA_N;
        constexpr int kProducerThreads =
            NUM_THREADS - kConsumerWarps * kWarpSize;
        const int producer_tid = tid - kConsumerWarps * kWarpSize;
        if (producer_tid >= 0) {
          load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                        kProducerThreads, KV_DTYPE,
                                        E4M3_SHARED_LUT>(
              smem.k_buffer(0), k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, 0, producer_tid, e4m3_lut,
              BLOCK_SIZE == 4 ? scales.pages : int8_kv::KVPageMapping{});
        }
        __syncthreads();

#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          if (producer_tid >= 0 && panel_idx + 1 < kNumQKPanels) {
            load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                          kProducerThreads, KV_DTYPE,
                                          E4M3_SHARED_LUT>(
                smem.k_buffer((panel_idx + 1) & 1), k_cache, smem.page_ids,
                valid_kv_tile_rows, qk_panel_d_stride_uint4,
                qk_smem_stride_uint4, tile_page_offset, kv_tile_start,
                block_size, kv_head_idx, k_block_stride, k_token_stride,
                k_head_stride, panel_offset + kQKPanelDim, producer_tid,
                e4m3_lut,
                BLOCK_SIZE == 4 ? scales.pages : int8_kv::KVPageMapping{});
          }
          if (warp_id < kConsumerWarps) {
            const int tile_n = warp_id * WMMA_N;
            const __half* sK_panel = smem.k_buffer(panel_idx & 1);
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag,
                  sK_panel + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      } else {
#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                               KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
              sK, k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, panel_offset, threadIdx.x,
              e4m3_lut,
              BLOCK_SIZE == 4 ? scales.pages : int8_kv::KVPageMapping{});
          for (int idx = tid + valid_kv_tile_rows * qk_panel_d_stride_uint4;
               idx < kXQATCBlockN * qk_panel_d_stride_uint4;
               idx += NUM_THREADS) {
            const int row = idx / qk_panel_d_stride_uint4;
            const int vec_col = idx % qk_panel_d_stride_uint4;
            reinterpret_cast<uint4*>(sK)[row * qk_smem_stride_uint4 + vec_col] =
                make_uint4(0, 0, 0, 0);
          }
          __syncthreads();

          if (warp_id < (kXQATCBlockN / WMMA_N)) {
            const int tile_n = warp_id * WMMA_N;
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag, sK + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      }

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
#pragma unroll
        for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
          qk_acc_frag.x[i] *= (KV_DTYPE == int8_kv::KV_CACHE_DTYPE_FP16 ||
                               int8_kv::KVCodecTraits<KV_DTYPE>::per_token_head)
                                  ? softmax_scale
                                  : softmax_scale * k_scale;
        }
        volta::store_matrix_sync(sS + kv_tile_start + warp_id * WMMA_N,
                                 qk_acc_frag, kXQATCBlockN,
                                 volta::mem_row_major);
      }
      __syncthreads();
    }

    if constexpr (int8_kv::KVCodecTraits<KV_DTYPE>::per_token_head) {
      int8_kv::scale_attention_tile<KV_DTYPE>(
          sS, GROUP_SIZE, valid_k_rows, kXQATCBlockN, scales.key, smem.page_ids,
          tile_page_offset, block_size, kv_head_idx);
      __syncthreads();
    }

    if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      const unsigned mask = 0xffffffffu;
      float* sS_row_f = sS + row * kXQATCBlockN;
      __half* sP_row_h = sP + row * kXQATCBlockN;
      float* sP_row_f = sS + row * kXQATCBlockN;
      const int vec_cols = valid_k_rows >> 2;
      const int tail_start = vec_cols << 2;
      const int vec_col = thread_in_row;

      float thread_max = kXQANegInf;
      __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
      __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
      }

      const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
      const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
      const float new_max = fmaxf(old_max, row_max);
      const float exp_diff = __expf(old_max - new_max);

      float thread_sum = 0.f;
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        const float e0 = clamped_softmax_exp<kPropagateNaN>(v4.x - new_max);
        const float e1 = clamped_softmax_exp<kPropagateNaN>(v4.y - new_max);
        const float e2 = clamped_softmax_exp<kPropagateNaN>(v4.z - new_max);
        const float e3 = clamped_softmax_exp<kPropagateNaN>(v4.w - new_max);
        thread_sum += (e0 + e1) + (e2 + e3);
        if constexpr (kKeepPfp32) {
          reinterpret_cast<float4*>(sP_row_f)[vec_col] =
              make_float4(e0, e1, e2, e3);
        } else {
          packed_exp0 = __float22half2_rn(make_float2(e0, e1));
          packed_exp1 = __float22half2_rn(make_float2(e2, e3));
        }
      }

#pragma unroll
      for (int c = tail_start + thread_in_row; c < kXQATCBlockN;
           c += kXQATC256WideThreadsPerRow) {
        const float v = (c < valid_k_rows) ? sS_row_f[c] : kXQANegInf;
        const float e = clamped_softmax_exp<kPropagateNaN>(v - new_max);
        thread_sum += (c < valid_k_rows) ? e : 0.0f;
        if constexpr (kKeepPfp32) {
          sP_row_f[c] = (c < valid_k_rows) ? e : 0.0f;
        } else {
          sP_row_h[c] =
              (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
        }
      }

#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
      }

      const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
      const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);

      if (thread_in_row == 0) {
        row_sum_reg = exp_diff * old_sum + row_sum;
        row_max_reg = new_max;
      }
      if constexpr (!kKeepPfp32) {
        __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
        if (vec_col < vec_cols) {
          const int base_offset = vec_col * 2;
          sP_half2[base_offset] = packed_exp0;
          sP_half2[base_offset + 1] = packed_exp1;
        }
      }

      if (block_n > 0) {
#pragma unroll
        for (int i = 0; i < kAccumsPerThread; ++i) {
          out_acc[i] *= exp_diff;
        }
      }
    }
    __syncthreads();

    // Scale PV weights only after the unscaled softmax sum is recorded.
    // The same FP32 P tile is reused by every V panel.
    if constexpr (int8_kv::KVCodecTraits<KV_DTYPE>::per_token_head) {
      int8_kv::scale_attention_tile<KV_DTYPE>(
          sS, GROUP_SIZE, valid_k_rows, kXQATCBlockN, scales.value,
          smem.page_ids, tile_page_offset, block_size, kv_head_idx);
      __syncthreads();
    }

    for (int panel_idx = 0; panel_idx < kNumPVPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPVPanelDim;
      for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
           kv_tile_start += kXQATCBlockN) {
        const int valid_kv_tile_rows =
            min(kXQATCBlockN, valid_k_rows - kv_tile_start);
        load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                             KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
            sV, v_cache, smem.page_ids, valid_kv_tile_rows,
            pv_panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
            kv_tile_start, block_size, kv_head_idx, v_block_stride,
            v_token_stride, v_head_stride, panel_offset, threadIdx.x, e4m3_lut,
            BLOCK_SIZE == 4 ? scales.pages : int8_kv::KVPageMapping{});
        for (int idx = tid + valid_kv_tile_rows * pv_panel_d_stride_uint4;
             idx < kXQATCBlockN * pv_panel_d_stride_uint4; idx += NUM_THREADS) {
          const int row = idx / pv_panel_d_stride_uint4;
          const int vec_col = idx % pv_panel_d_stride_uint4;
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const __half* sP_row = sP + row * kXQATCBlockN + kv_tile_start;
          const float* sP_row_f = sS + row * kXQATCBlockN + kv_tile_start;
#pragma unroll
          for (int token = 0; token < kXQATCBlockN; ++token) {
            if (token >= valid_kv_tile_rows) {
              break;
            }
            float prob;
            if constexpr (kKeepPfp32) {
              prob = sP_row_f[token];
            } else {
              prob = __half2float(sP_row[token]);
            }
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
            if constexpr (kPVHalf2) {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / (2 * kWarpSize));
                   ++d_iter) {
                const int local_d2 = lane_id + d_iter * kWarpSize;
                const float2 value = __half22float2(
                    reinterpret_cast<const __half2*>(sV_row)[local_d2]);
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter * 2;
                out_acc[acc_idx] = fmaf(prob, value.x, out_acc[acc_idx]);
                out_acc[acc_idx + 1] =
                    fmaf(prob, value.y, out_acc[acc_idx + 1]);
              }
            } else {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / kWarpSize);
                   ++d_iter) {
                const int local_d = lane_id + d_iter * kWarpSize;
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter;
                out_acc[acc_idx] =
                    fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      smem.row_max[row] = row_max_reg;
      smem.row_sum[row] = row_sum_reg;
    }
  }
  __syncthreads();

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum = smem.row_sum[row];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    PARTIAL_T* tmp_out_ptr =
        tmp_out + static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
        static_cast<int64_t>(head_idx) * tmp_out_stride1 +
        static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    const float output_scale =
        int8_kv::KVCodecTraits<KV_DTYPE>::output_scale(inv_row_sum, v_scale);
    if constexpr (kPVHalf2) {
#pragma unroll
      for (int acc_idx = 0; acc_idx < kAccumsPerThread; ++acc_idx) {
        constexpr int kAccumsPerPanel = kPVPanelDim / kWarpSize;
        const int panel_idx = acc_idx / kAccumsPerPanel;
        const int panel_acc_idx = acc_idx % kAccumsPerPanel;
        const int d_iter = panel_acc_idx / 2;
        const int component = panel_acc_idx & 1;
        const int d = panel_idx * kPVPanelDim + d_iter * (2 * kWarpSize) +
                      thread_in_row * 2 + component;
        const float value = out_acc[acc_idx] * output_scale;
        if constexpr (std::is_same_v<PARTIAL_T, float>) {
          tmp_out_ptr[d] = value;
        } else {
          tmp_out_ptr[d] = __float2half(value);
        }
      }
    } else {
      for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
        const float value = out_acc[d / kWarpSize] * output_scale;
        if constexpr (std::is_same_v<PARTIAL_T, float>) {
          tmp_out_ptr[d] = value;
        } else {
          tmp_out_ptr[d] = __float2half(value);
        }
      }
    }
    if (thread_in_row == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = smem.row_max[row];
      exp_sums[stats_index] = row_sum;
    }
  }
}

template <int PAGE_SIZE, bool FIXED_LAYOUT, int CODEC, bool PAIRED>
__device__ __forceinline__ void load_grouped_kv_panel(
    __half* shared, const void* cache, const int* pages, int rows,
    int panel_stride, int shared_stride, int first_token, int page_size,
    int64_t block_stride, int64_t token_stride, int64_t head_stride,
    bool aligned_pair, const int8_kv::KVPageMapping mapping = {}) {
  if constexpr (PAIRED) {
    if (aligned_pair) {
      load_xqa_tc_kv_panel<PAGE_SIZE, FIXED_LAYOUT, kGroupedVerifyThreads,
                           CODEC, true>(
          shared, cache, pages, rows, panel_stride, shared_stride, first_token,
          0, page_size, 0, block_stride, token_stride, head_stride, 0,
          threadIdx.x, nullptr, mapping);
      return;
    }
  }
  load_xqa_tc_kv_panel<PAGE_SIZE, FIXED_LAYOUT, kGroupedVerifyThreads, CODEC>(
      shared, cache, pages, rows, panel_stride, shared_stride, first_token, 0,
      page_size, 0, block_stride, token_stride, head_stride, 0, threadIdx.x,
      nullptr, mapping);
}

template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE = 0,
          bool SINGLE_QUERY = false, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool STAGE_PARTITION_PAGE_IDS = false,
          int KV_DTYPE = int8_kv::KV_CACHE_DTYPE_FP8_E5M2,
          bool SPARSE_PAGE4 = false, typename PARTIAL_T = __half,
          bool ROW_SEQLENS = false, bool COMPENSATE_P = false,
          bool PAIRED_BYTE_LOAD = false>
__global__
__launch_bounds__(kGroupedVerifyThreads, 1) void flash_attention_grouped_verify_e5m2_partial_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, PARTIAL_T* __restrict__ partial_out,
    float* __restrict__ partial_lse, const int query_len,
    const int max_num_blocks, const int page_block_size,
    const int64_t k_block_stride, const int64_t k_token_stride,
    const int64_t k_head_stride, const int64_t v_block_stride,
    const int64_t v_token_stride, const int64_t v_head_stride,
    const float qk_scale, const float v_scale,
    const uint32_t* __restrict__ sparse_token_masks = nullptr,
    const int num_groups = 1, const int* row_lengths = nullptr,
    const int8_kv::KVScaleView scales = {}, const bool stage_page_ids = true,
    const bool aligned_pair = true) {
  static_assert(!ROW_SEQLENS || !SPARSE_PAGE4,
                "explicit row lengths apply to dense causal groups only");
  static_assert(!COMPENSATE_P || SPARSE_PAGE4 ||
                    (ROW_SEQLENS && std::is_same_v<PARTIAL_T, float>),
                "probability compensation requires sparse output or "
                "explicit-row FP32 groups");
  constexpr bool kRowScales = int8_kv::KVCodecTraits<KV_DTYPE>::per_token_head;
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  const int head_group = blockIdx.x;
  const int split_id = blockIdx.y;
  const int group_idx = blockIdx.z;
  if (head_group >= Traits::kHeadGroups || split_id >= Traits::kSplits ||
      group_idx >= num_groups || query_len <= 0 ||
      query_len > MAX_QUERY_TOKENS) {
    return;
  }
  if constexpr (ROW_SEQLENS) {
    row_lengths += static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS;
  }

  if constexpr (!SPARSE_PAGE4) {
    partial_out += static_cast<int64_t>(group_idx) * Traits::kSplits *
                   MAX_QUERY_TOKENS * kGroupedVerifyHeads *
                   kGroupedVerifyHeadDim;
    partial_lse += static_cast<int64_t>(group_idx) * Traits::kSplits *
                   MAX_QUERY_TOKENS * kGroupedVerifyHeads *
                   (ROW_SEQLENS ? 2 : 1);
  }

  int total_kv = seq_lens[group_idx];
  if constexpr (ROW_SEQLENS) {
    total_kv = 0;
    for (int i = 0; i < query_len; ++i)
      total_kv = max(total_kv, row_lengths[i]);
  }
  if (total_kv <= 0) {
    if constexpr (SPARSE_PAGE4) {
      constexpr int kGroupOutputElements =
          kGroupedVerifyQ8MaxQ * kGroupedVerifyHeads * kGroupedVerifyHeadDim;
      const int64_t group_output_offset =
          static_cast<int64_t>(group_idx) * kGroupOutputElements;
      for (int idx = threadIdx.x; idx < kGroupOutputElements;
           idx += kGroupedVerifyThreads) {
        partial_out[group_output_offset + idx] = __float2half_rn(0.0f);
      }
      constexpr int kGroupLseElements =
          kGroupedVerifyQ8MaxQ * kGroupedVerifyHeads;
      if (threadIdx.x < kGroupLseElements) {
        partial_lse[static_cast<int64_t>(group_idx) * kGroupLseElements +
                    threadIdx.x] = kXQANegInf;
      }
    }
    return;
  }
  const int active_splits =
      SPARSE_PAGE4
          ? 1
          : grouped_verify_active_splits<MAX_QUERY_TOKENS, SINGLE_QUERY>(
                total_kv);
  if (split_id >= active_splits) {
    return;
  }
  const int total_tiles =
      (total_kv + kGroupedVerifyBlockN - 1) / kGroupedVerifyBlockN;
  const int base_tiles = total_tiles / active_splits;
  const int extra_tiles = total_tiles - base_tiles * active_splits;
  const int split_tile_start =
      split_id * base_tiles + min(split_id, extra_tiles);
  const int split_tiles = base_tiles + (split_id < extra_tiles ? 1 : 0);
  const int split_start = split_tile_start * kGroupedVerifyBlockN;
  const int split_end =
      min(total_kv, split_start + split_tiles * kGroupedVerifyBlockN);
  const int prefix_kv_len = max(0, total_kv - query_len);
  const int head_start = head_group * Traits::kHeadsPerCta;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  extern __shared__ char grouped_verify_smem_raw[];
  GroupedVerifySmem& smem =
      *reinterpret_cast<GroupedVerifySmem*>(grouped_verify_smem_raw);
  __half* shared_q = smem.storage.compute.q;
  __half* shared_kv = smem.storage.compute.kv;
  float* shared_scores = smem.storage.compute.scores;
  __half* shared_probs = smem.storage.compute.probs;
  // The compensated entry reserves one padded probability panel after the
  // legacy layout. Match P's shared-memory stride without aliasing live QK
  // rows. Legacy entries retain their existing shared-memory footprint.
  __half* shared_prob_residual = reinterpret_cast<__half*>(
      grouped_verify_smem_raw + sizeof(GroupedVerifySmem));
  constexpr int kResidualStride = kGroupedVerifyProbStride;
  const int* page_ids =
      block_table + static_cast<int64_t>(group_idx) * max_num_blocks;
  int split_page_offset = 0;
  bool use_staged_page_ids = false;
  if constexpr (STAGE_PARTITION_PAGE_IDS) {
    const int staging_page_size =
        PAGE_BLOCK_SIZE > 0 ? PAGE_BLOCK_SIZE : page_block_size;
    const int split_start_page = split_start / staging_page_size;
    split_page_offset = split_start - split_start_page * staging_page_size;
    const int split_page_count =
        (split_page_offset + split_end - split_start + staging_page_size - 1) /
        staging_page_size;
    use_staged_page_ids =
        stage_page_ids && split_page_count <= kGroupedVerifyPageIdsCapacity;
    if (use_staged_page_ids) {
      for (int idx = tid; idx < split_page_count;
           idx += kGroupedVerifyThreads) {
        smem.page_ids[idx] = __ldg(&page_ids[split_start_page + idx]);
      }
    }
    if (use_staged_page_ids) {
      page_ids = smem.page_ids;
    }
  }

  constexpr int kVecsPerRow = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedQVecsPerRow = kGroupedVerifyQStride / 8;
  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* shared_q_vec = reinterpret_cast<uint4*>(shared_q);
  for (int idx = tid; idx < kGroupedVerifyRows * kVecsPerRow;
       idx += kGroupedVerifyThreads) {
    const int row = idx / kVecsPerRow;
    const int vec_col = idx % kVecsPerRow;
    const int token_idx = row / Traits::kHeadsPerCta;
    const int local_head = row % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const int64_t query_row =
          static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
      shared_q_vec[row * kSharedQVecsPerRow + vec_col] = __ldg(
          q_vec + (query_row * kGroupedVerifyHeads + head_idx) * kVecsPerRow +
          vec_col);
    } else {
      shared_q_vec[row * kSharedQVecsPerRow + vec_col] = make_uint4(0, 0, 0, 0);
    }
  }
  if (tid < kGroupedVerifyRows) {
    smem.row_max[tid] = kXQANegInf;
    smem.row_sum[tid] = 0.0f;
    smem.row_scale[tid] = 1.0f;
  }
  // This barrier publishes both Q and the optional split-local page IDs.
  __syncthreads();

  constexpr int kPanelStrideVec = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;

  volta::fragment<volta::accumulator, 16, 16, 16, float>
      output_fragments[kGroupedVerifyOutputTilesPerWarp];
#pragma unroll
  for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
       ++fragment_idx) {
    volta::fill_fragment(output_fragments[fragment_idx], 0.0f);
  }

  // The conservative baseline computes exact per-split FP32 max/sum in a
  // separate pass. The candidate below fuses this state update with P x V.
  if constexpr (TWO_PASS) {
    for (int tile_start = split_start; tile_start < split_end;
         tile_start += kGroupedVerifyBlockN) {
      const int valid_k_rows =
          min(kGroupedVerifyBlockN, split_end - tile_start);
      if constexpr (SPARSE_PAGE4) {
        const int valid_sparse_pages = (valid_k_rows + 3) / 4;
        if (tid < kGroupedVerifyBlockN / 4) {
          smem.sparse_token_masks[tid] =
              tid < valid_sparse_pages
                  ? __ldg(sparse_token_masks +
                          static_cast<int64_t>(group_idx) * max_num_blocks +
                          (tile_start >> 2) + tid)
                  : 0;
        }
      }
      const int tile_page_offset =
          use_staged_page_ids ? split_page_offset + tile_start - split_start
                              : tile_start;
      load_grouped_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                            PAIRED_BYTE_LOAD ||
                                (KV_DTYPE == int8_kv::KV_CACHE_DTYPE_FP8_E5M2 &&
                                 !ROW_SEQLENS) ||
                                (KV_DTYPE == int8_kv::KV_CACHE_DTYPE_FP8_E4M3 &&
                                 !SPARSE_PAGE4 && !ROW_SEQLENS)>(
          shared_kv, k_cache, page_ids, valid_k_rows, kPanelStrideVec,
          kSharedStrideVec, tile_page_offset, page_block_size, k_block_stride,
          k_token_stride, k_head_stride, aligned_pair,
          SPARSE_PAGE4 ? scales.pages : int8_kv::KVPageMapping{});
      for (int idx = tid + valid_k_rows * kSharedStrideVec;
           idx < kGroupedVerifyBlockN * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();
      // W12: defensively zero K rows no query attends (see the V panel below).
      // K NaN is already masked out of the scores by
      // grouped_verify_key_visible, but zeroing keeps the loaded panel free of
      // the null block's E4M3 NaN. Masks are visible after the __syncthreads()
      // above; the one added inside publishes the zeros before
      // grouped_verify_qk consumes shared_kv.
      if constexpr (SPARSE_PAGE4) {
        for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
             idx += kGroupedVerifyThreads) {
          const int row = idx / kSharedStrideVec;
          const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
          if ((token_mask & (0x11111111u << (row & 3))) == 0) {
            reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
          }
        }
        __syncthreads();
      }

      int active_m_tiles = 0x7;
      if constexpr (SPARSE_PAGE4) {
        uint32_t active_query_nibbles = 0;
#pragma unroll
        for (int page = 0; page < kGroupedVerifyBlockN / 4; ++page) {
          active_query_nibbles |= smem.sparse_token_masks[page];
        }
        active_m_tiles = 0;
#pragma unroll
        for (int token = 0; token < kGroupedVerifyQ8MaxQ; ++token) {
          if ((active_query_nibbles & (0xFu << (token * 4))) != 0) {
            const int first_row = token * kGroupedVerifyHeads;
            const int last_row = first_row + kGroupedVerifyHeads - 1;
            active_m_tiles |= 1 << (first_row / 16);
            active_m_tiles |= 1 << (last_row / 16);
          }
        }
      }
      grouped_verify_qk<COMPENSATE_P>(shared_q, shared_kv, shared_scores,
                                      qk_scale, active_m_tiles);
      __syncthreads();
      if constexpr (kRowScales) {
        int8_kv::scale_attention_tile<KV_DTYPE>(
            shared_scores, kGroupedVerifyRows, valid_k_rows,
            kGroupedVerifyScoreStride, scales.key, page_ids, tile_page_offset,
            page_block_size, 0);
        __syncthreads();
      }

#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, lane_id, prefix_kv_len, row_lengths);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float probability =
            visible ? clamped_softmax_exp<kRowScales>(score - tile_max) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        if (lane_id == 0 &&
            (tile_sum > 0.0f || (kRowScales && isnan(tile_sum)))) {
          const float old_max = smem.row_max[row];
          const float old_sum = smem.row_sum[row];
          const float new_max = fmaxf(old_max, tile_max);
          smem.row_sum[row] =
              old_sum * __expf(fmaxf(old_max - new_max, -80.0f)) +
              tile_sum * __expf(fmaxf(tile_max - new_max, -80.0f));
          smem.row_max[row] = new_max;
        }
      }
      __syncthreads();
    }
  }

  // Recompute QK for the conservative path, or consume it once while updating
  // the online state for the fused path. Both form half P and use identical
  // Volta WMMA P @ V arithmetic.
  for (int tile_start = split_start; tile_start < split_end;
       tile_start += kGroupedVerifyBlockN) {
    const int valid_k_rows = min(kGroupedVerifyBlockN, split_end - tile_start);
    if constexpr (SPARSE_PAGE4) {
      const int valid_sparse_pages = (valid_k_rows + 3) / 4;
      if (tid < kGroupedVerifyBlockN / 4) {
        smem.sparse_token_masks[tid] =
            tid < valid_sparse_pages
                ? __ldg(sparse_token_masks +
                        static_cast<int64_t>(group_idx) * max_num_blocks +
                        (tile_start >> 2) + tid)
                : 0;
      }
    }
    const int tile_page_offset =
        use_staged_page_ids ? split_page_offset + tile_start - split_start
                            : tile_start;
    load_grouped_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                          PAIRED_BYTE_LOAD ||
                              (KV_DTYPE == int8_kv::KV_CACHE_DTYPE_FP8_E5M2 &&
                               !ROW_SEQLENS) ||
                              (KV_DTYPE == int8_kv::KV_CACHE_DTYPE_FP8_E4M3 &&
                               !SPARSE_PAGE4 && !ROW_SEQLENS)>(
        shared_kv, k_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, page_block_size, k_block_stride,
        k_token_stride, k_head_stride, aligned_pair,
        SPARSE_PAGE4 ? scales.pages : int8_kv::KVPageMapping{});
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec;
         idx += kGroupedVerifyThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    // W12: defensively zero K rows no query attends (see the V panel below).
    // K NaN is already masked out of the scores by grouped_verify_key_visible,
    // but zeroing keeps the loaded panel free of the null block's E4M3 NaN.
    // Masks are visible after the __syncthreads() above; the one added inside
    // publishes the zeros before grouped_verify_qk consumes shared_kv.
    if constexpr (SPARSE_PAGE4) {
      for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kSharedStrideVec;
        const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
        if ((token_mask & (0x11111111u << (row & 3))) == 0) {
          reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
        }
      }
      __syncthreads();
    }

    int active_m_tiles = 0x7;
    if constexpr (COMPENSATE_P) {
      // Native MTP4 supplies five rows: only 30 of the 48 packed rows are
      // live. Avoid QK/PV for wholly padded M tiles to pay for the residual
      // product without changing arithmetic in any live row.
      active_m_tiles =
          (1 << ((query_len * Traits::kHeadsPerCta + 15) / 16)) - 1;
    }
    if constexpr (SPARSE_PAGE4) {
      uint32_t active_query_nibbles = 0;
#pragma unroll
      for (int page = 0; page < kGroupedVerifyBlockN / 4; ++page) {
        active_query_nibbles |= smem.sparse_token_masks[page];
      }
      active_m_tiles = 0;
#pragma unroll
      for (int token = 0; token < kGroupedVerifyQ8MaxQ; ++token) {
        if ((active_query_nibbles & (0xFu << (token * 4))) != 0) {
          const int first_row = token * kGroupedVerifyHeads;
          const int last_row = first_row + kGroupedVerifyHeads - 1;
          active_m_tiles |= 1 << (first_row / 16);
          active_m_tiles |= 1 << (last_row / 16);
        }
      }
    }
    grouped_verify_qk<COMPENSATE_P>(shared_q, shared_kv, shared_scores,
                                    qk_scale, active_m_tiles);
    __syncthreads();
    if constexpr (kRowScales) {
      int8_kv::scale_attention_tile<KV_DTYPE>(
          shared_scores, kGroupedVerifyRows, valid_k_rows,
          kGroupedVerifyScoreStride, scales.key, page_ids, tile_page_offset,
          page_block_size, 0);
      __syncthreads();
    }

    if constexpr (TWO_PASS) {
      for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyBlockN;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kGroupedVerifyBlockN;
        const int col = idx % kGroupedVerifyBlockN;
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + col;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, col, prefix_kv_len, row_lengths) &&
            (smem.row_sum[row] > 0.0f ||
             (kRowScales && isnan(smem.row_sum[row])));
        float probability =
            visible ? clamped_softmax_exp<kRowScales>(
                          shared_scores[row * kGroupedVerifyScoreStride + col] -
                          smem.row_max[row])
                    : 0.0f;
        if constexpr (kRowScales) {
          if (visible) {
            const int token = tile_page_offset + col;
            probability = int8_kv::KVCodecTraits<KV_DTYPE>::value_weight(
                probability, scales.value, page_ids[token / page_block_size],
                token % page_block_size, 0);
          }
        }
        shared_probs[row * kGroupedVerifyProbStride + col] =
            __float2half_rn(probability);
        if constexpr (COMPENSATE_P) {
          const float rounded = __half2float(__float2half_rn(probability));
          shared_prob_residual[row * kResidualStride + col] =
              __float2half_rn((probability - rounded) * 2048.0f);
        }
      }
      __syncthreads();
    } else {
#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        if constexpr (COMPENSATE_P) {
          if ((active_m_tiles & (1 << (row / 16))) == 0) {
            continue;
          }
        }
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, lane_id, prefix_kv_len, row_lengths);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float old_max = smem.row_max[row];
        const float new_max = fmaxf(old_max, tile_max);
        const float probability =
            visible ? clamped_softmax_exp<kRowScales>(score - new_max) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        const bool active_sum =
            tile_sum > 0.0f || (kRowScales && isnan(tile_sum));
        const float exp_diff =
            active_sum ? __expf(fmaxf(old_max - new_max, -80.0f)) : 1.0f;
        float pv_weight = probability;
        if constexpr (kRowScales) {
          // Mask before loading/applying a scale: an invalid future token
          // must not contaminate an earlier causal row through 0 * NaN.
          if (visible) {
            const int token = tile_page_offset + lane_id;
            pv_weight = int8_kv::KVCodecTraits<KV_DTYPE>::value_weight(
                probability, scales.value, page_ids[token / page_block_size],
                token % page_block_size, 0);
          }
        }
        shared_probs[row * kGroupedVerifyProbStride + lane_id] =
            __float2half_rn(pv_weight);
        if constexpr (COMPENSATE_P) {
          const float rounded = __half2float(__float2half_rn(pv_weight));
          shared_prob_residual[row * kResidualStride + lane_id] =
              __float2half_rn((pv_weight - rounded) * 2048.0f);
        }
        // Finish every lane's shared-state reads before lane 0 overwrites the
        // online maximum. Shuffle synchronization does not order memory.
        __syncwarp();
        if (lane_id == 0) {
          if (active_sum) {
            smem.row_sum[row] = smem.row_sum[row] * exp_diff + tile_sum;
            smem.row_max[row] = new_max;
          }
          smem.row_scale[row] = exp_diff;
        }
      }
      __syncthreads();
      if constexpr (!COMPENSATE_P) {
#pragma unroll
        for (int fragment_idx = 0;
             fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
          const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
          const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
          grouped_verify_scale_output_fragment(output_fragments[fragment_idx],
                                               smem.row_scale, m_tile * 16);
        }
      }
    }

    load_grouped_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                          PAIRED_BYTE_LOAD ||
                              (KV_DTYPE == int8_kv::KV_CACHE_DTYPE_FP8_E5M2 &&
                               !ROW_SEQLENS) ||
                              (KV_DTYPE == int8_kv::KV_CACHE_DTYPE_FP8_E4M3 &&
                               !SPARSE_PAGE4 && !ROW_SEQLENS)>(
        shared_kv, v_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, page_block_size, v_block_stride,
        v_token_stride, v_head_stride, aligned_pair,
        SPARSE_PAGE4 ? scales.pages : int8_kv::KVPageMapping{});
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec;
         idx += kGroupedVerifyThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    // W12: the grouped planner pads each category to a multiple of 8 with
    // (physical microblock 0 = null block, mask 0) and counts the padding in
    // seq_len; the forward also carries unselected tokens inside partial
    // microblocks. Such rows get P=0, but 0 * NaN survives the P@V MMA when
    // page 0 holds fp16 GDN state that decodes to E4M3 NaN (W8). Zero every
    // row no query attends (mask bit = query*4 + token_in_microblock, so the
    // row is unattended iff (mask & (0x11111111u << (row & 3))) == 0) so the
    // reduction sees 0 * 0 = 0. The leading __syncthreads() makes the V panel
    // load finish before we overwrite its page-0 rows (else the load races past
    // the zero and resurrects the NaN); the trailing one publishes the zeros
    // before P@V. Masks were synced earlier in this iteration.
    if constexpr (SPARSE_PAGE4) {
      __syncthreads();
      for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kSharedStrideVec;
        const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
        if ((token_mask & (0x11111111u << (row & 3))) == 0) {
          reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
         ++fragment_idx) {
      const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
      const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
      const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
      if constexpr (SPARSE_PAGE4 || COMPENSATE_P) {
        if ((active_m_tiles & (1 << m_tile)) == 0) {
          continue;
        }
      }
      volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
          probability_fragment;
      volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major>
          value_fragment;
      volta::fragment<volta::accumulator, 16, 16, 16, float> tile_fragment;
      if constexpr (COMPENSATE_P) {
        // Keep Tensor Core accumulation local to N32. Carrying a large C
        // through every KV tile loses small PV corrections at long contexts.
        // The online state remains FP32 and is updated once per tile below.
        volta::fill_fragment(tile_fragment, 0.0f);
      }
      auto& pv_fragment =
          COMPENSATE_P ? tile_fragment : output_fragments[fragment_idx];
#pragma unroll
      for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
        volta::load_matrix_sync(
            probability_fragment,
            shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
            kGroupedVerifyProbStride);
        volta::load_matrix_sync(
            value_fragment,
            shared_kv + k_offset * kGroupedVerifyKVStride + d_tile * 16,
            kGroupedVerifyKVStride);
        volta::mma_sync(pv_fragment, probability_fragment, value_fragment,
                        pv_fragment);
        if constexpr (COMPENSATE_P) {
          // Scale the residual up before its FP16 conversion so small
          // corrections do not underflow. Every finite E4M3 value divided
          // by 2048 is still exactly representable in FP16, so applying
          // the inverse power of two to V introduces no operand rounding.
          volta::load_matrix_sync(
              probability_fragment,
              shared_prob_residual + m_tile * 16 * kResidualStride + k_offset,
              kResidualStride);
#pragma unroll
          for (int i = 0; i < value_fragment.num_elements / 2; ++i) {
            union {
              uint32_t bits;
              __half2 pair;
            } packed_value;
            packed_value.bits = value_fragment.x[i];
            packed_value.pair =
                __hmul2(packed_value.pair, __float2half2_rn(1.0f / 2048.0f));
            value_fragment.x[i] = packed_value.bits;
          }
          volta::mma_sync(pv_fragment, probability_fragment, value_fragment,
                          pv_fragment);
        }
      }
      if constexpr (COMPENSATE_P) {
        grouped_verify_add_output_tile(output_fragments[fragment_idx],
                                       tile_fragment, smem.row_scale,
                                       m_tile * 16);
      }
    }
    __syncthreads();
  }

  // The compute buffers are dead. Reuse their storage for the dense FP32
  // partial output, then normalize and write only real query/head rows.
  __syncthreads();
  float* shared_output = smem.storage.output;
#pragma unroll
  for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
       ++fragment_idx) {
    const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
    const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
    const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
    volta::store_matrix_sync(
        shared_output + m_tile * 16 * kGroupedVerifyHeadDim + d_tile * 16,
        output_fragments[fragment_idx], kGroupedVerifyHeadDim,
        volta::mem_row_major);
  }
  __syncthreads();

  for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim;
       idx += kGroupedVerifyThreads) {
    const int row = idx / kGroupedVerifyHeadDim;
    const int d = idx % kGroupedVerifyHeadDim;
    const int token_idx = row / Traits::kHeadsPerCta;
    const int local_head = row % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const float sum = smem.row_sum[row];
      // Explicit-row FP32 groups retain the numerator until the final merge.
      // Normalizing each partition and encoding its weight as m+log(sum)
      // introduces avoidable rounding, including at FP16 output midpoints.
      const float scale = sum > 0.0f || (kRowScales && isnan(sum))
                              ? (ROW_SEQLENS ? v_scale : v_scale / sum)
                              : 0.0f;
      int64_t output_idx;
      if constexpr (SPARSE_PAGE4) {
        const int64_t global_token_idx =
            static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
        output_idx = (global_token_idx * kGroupedVerifyHeads + head_idx) *
                         kGroupedVerifyHeadDim +
                     d;
      } else {
        output_idx =
            (((static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
                  kGroupedVerifyHeads +
              head_idx) *
                 kGroupedVerifyHeadDim +
             d);
      }
      if constexpr (std::is_same_v<PARTIAL_T, float>) {
        partial_out[output_idx] = shared_output[idx] * scale;
      } else {
        partial_out[output_idx] = __float2half_rn(shared_output[idx] * scale);
      }
    }
  }
  if (tid < kGroupedVerifyRows) {
    const int token_idx = tid / Traits::kHeadsPerCta;
    const int local_head = tid % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const float sum = smem.row_sum[tid];
      int64_t lse_idx;
      if constexpr (SPARSE_PAGE4) {
        const int64_t global_token_idx =
            static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
        lse_idx = global_token_idx * kGroupedVerifyHeads + head_idx;
      } else {
        lse_idx =
            (static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
                kGroupedVerifyHeads +
            head_idx;
      }
      if constexpr (ROW_SEQLENS) {
        partial_lse[2 * lse_idx] = sum > 0.0f ? smem.row_max[tid] : kXQANegInf;
        if constexpr (kRowScales) {
          // A visible invalid K row keeps a live partition with a NaN sum.
          // The common merge then propagates it instead of dropping it as
          // empty. Zero-length/padded rows still have exactly zero sums.
          if (isnan(sum)) partial_lse[2 * lse_idx] = 0.0f;
        }
        partial_lse[2 * lse_idx + 1] = sum;
      } else {
        partial_lse[lse_idx] = (sum > 0.0f || isnan(sum))
                                   ? smem.row_max[tid] + logf(sum)
                                   : kXQANegInf;
      }
    }
  }
}

}  // namespace int8_page4

namespace int8_page4 {
inline int8_kv::KVScaleView bind(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v, const at::Tensor& sk,
                                 const at::Tensor& sv, int owner_span) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
                  q.size(1) == 6 && q.size(2) == 256 && q.is_contiguous(),
              "INT8 page4 requires contiguous CUDA FP16 [M,6,256] queries");
  TORCH_CHECK(k.dim() == 4 && k.sizes() == v.sizes() &&
                  k.scalar_type() == at::kChar &&
                  v.scalar_type() == at::kChar && k.size(0) > 0 &&
                  k.size(1) == 16 && k.size(2) == 1 && k.size(3) == 256 &&
                  owner_span == 4 && k.stride(3) == 1 && v.stride(3) == 1 &&
                  k.stride(1) == 256 && v.stride(1) == 256,
              "INT8 page4 requires v1 owner pages and explicit mapping (4,4)");
  int8_kv::KVScaleView view;
  view.pages = {owner_span, 4};
  const at::Tensor planes[] = {sk, sv};
  int8_kv::KVScalePlane* targets[] = {&view.key, &view.value};
  for (int i = 0; i < 2; ++i) {
    const auto& s = planes[i];
    TORCH_CHECK(s.sizes() == at::IntArrayRef({k.size(0), 16, 1}) &&
                    s.scalar_type() == at::kFloat && s.device() == q.device(),
                "INT8 page4 requires explicit CUDA FP32 [pages,16,1] scales");
    *targets[i] = {s.data_ptr<float>(), s.stride(0), s.stride(1),
                   s.stride(2),         owner_span,  4};
  }
  TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
              "INT8 page4 cache device mismatch");
  return view;
}
inline void metadata(const at::Tensor& t, const at::Tensor& q,
                     at::ScalarType dtype) {
  TORCH_CHECK(
      t.device() == q.device() && t.scalar_type() == dtype && t.is_contiguous(),
      "INT8 page4 metadata contract mismatch");
}
}  // namespace int8_page4

int64_t flash_attention_int8_page4_abi_version() { return 1; }

at::Tensor flash_attention_decode_int8_page4(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, const at::Tensor& table, const at::Tensor& lengths,
    at::Tensor& partial, at::Tensor& maxima, at::Tensor& sums,
    const at::Tensor& active, const at::Tensor& sk, const at::Tensor& sv,
    int64_t owner_span, double softmax_scale) {
  const auto scales = int8_page4::bind(q, k, v, sk, sv, owner_span);
  for (const auto& t : {table, lengths, active})
    int8_page4::metadata(t, q, at::kInt);
  for (const auto& t : {partial, maxima, sums})
    int8_page4::metadata(t, q, at::kFloat);
  int8_page4::metadata(out, q, at::kHalf);
  const int parts = partial.dim() == 4 ? partial.size(2) : 0;
  TORCH_CHECK(
      q.size(0) > 0 && parts > 0 && parts <= 128 &&
          partial.sizes() == at::IntArrayRef({q.size(0), 6, parts, 256}) &&
          maxima.sizes() == at::IntArrayRef({q.size(0), 6, parts}) &&
          sums.sizes() == maxima.sizes() && out.sizes() == q.sizes() &&
          lengths.sizes() == at::IntArrayRef({q.size(0)}) && table.dim() == 2 &&
          table.size(0) == q.size(0) && table.size(1) > 0 &&
          active.numel() == 1,
      "INT8 page4 XQA workspace/table shape mismatch");
  c10::cuda::CUDAGuard guard(q.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 0, "INT8 page4 requires SM70");
  constexpr int threads = 256;
  constexpr size_t bytes = sizeof(XQATCSmem256WideLayout<true, false>);
  auto kernel =
      int8_page4::flash_attention_decode_xqa_tc_partition_kernel_256_wide<
          256, 6, true, threads, 1, 4, false, false, 3, kXQARouteAllSeqLens,
          false, true, true, false, float>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  kernel<<<dim3(q.size(0), 1, parts), threads, bytes, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()), k.data_ptr(), v.data_ptr(),
      partial.data_ptr<float>(), maxima.data_ptr<float>(),
      sums.data_ptr<float>(), table.data_ptr<int>(), lengths.data_ptr<int>(),
      active.data_ptr<int>(), q.size(0), table.size(1), parts, 6, 1, 4,
      q.stride(0), q.stride(1), partial.stride(0), partial.stride(1),
      partial.stride(2), maxima.stride(0), maxima.stride(1), k.stride(0),
      k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
      softmax_scale, 1.0f, 1.0f, 0, 0, 0, scales);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch_flash_attention_decode_xqa_split_reduce<256, kXQARouteAllSeqLens,
                                                 float>(
      out, lengths, partial, maxima, sums, parts, 8, stream);
  return out;
}

at::Tensor flash_attention_grouped_sparse_int8_page4(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor& out, const at::Tensor& table, const at::Tensor& masks,
    const at::Tensor& lengths, at::Tensor& lse, const at::Tensor& sk,
    const at::Tensor& sv, int64_t owner_span, double softmax_scale) {
  const auto scales = int8_page4::bind(q, k, v, sk, sv, owner_span);
  int8_page4::metadata(table, q, at::kInt);
  int8_page4::metadata(masks, q, at::ScalarType::UInt32);
  int8_page4::metadata(lengths, q, at::kInt);
  int8_page4::metadata(lse, q, at::kFloat);
  TORCH_CHECK(out.scalar_type() == at::kHalf || out.scalar_type() == at::kFloat,
              "INT8 grouped output must be FP16 or FP32");
  int8_page4::metadata(out, q, out.scalar_type());
  const int groups = q.size(0) / 8;
  TORCH_CHECK(groups > 0 && q.size(0) % 8 == 0 && table.dim() == 2 &&
                  table.size(0) == groups && masks.sizes() == table.sizes() &&
                  lengths.sizes() == at::IntArrayRef({groups}) &&
                  lse.sizes() == at::IntArrayRef({q.size(0), 6}) &&
                  out.sizes() == q.sizes(),
              "INT8 grouped page4 metadata mismatch");
  c10::cuda::CUDAGuard guard(q.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 0, "INT8 page4 requires SM70");
  constexpr size_t bytes =
      sizeof(GroupedVerifySmem) +
      kGroupedVerifyRows * kGroupedVerifyProbStride * sizeof(__half);
  static_assert(bytes <= 64 * 1024);
  auto launch = [&](auto tag) {
    using Output = decltype(tag);
    auto kernel =
        int8_page4::flash_attention_grouped_verify_e5m2_partial_kernel<
            8, false, 4, false, false, false, 3, true, Output, false, true>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3(1, 1, groups), kGroupedVerifyThreads, bytes,
             at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr()), k.data_ptr(),
        v.data_ptr(), table.data_ptr<int>(), lengths.data_ptr<int>(),
        reinterpret_cast<Output*>(out.data_ptr()), lse.data_ptr<float>(), 8,
        table.size(1), 4, k.stride(0), k.stride(1), k.stride(2), v.stride(0),
        v.stride(1), v.stride(2), softmax_scale, 1.0f,
        masks.data_ptr<uint32_t>(), groups, nullptr, scales, true, true);
  };
  if (out.scalar_type() == at::kFloat)
    launch(float{});
  else
    launch(__half{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
