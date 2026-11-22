// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "qsa_lexicographic_topk.cuh"
#include "sm70_tp4_push_allgather.cuh"

namespace vllm::qsa {

constexpr int kShardTopK = 512;
constexpr int kShardWorld = 4;
constexpr int kShardCandidates = kShardWorld * kShardTopK;
constexpr int kShardPackWords = 2 * kShardTopK;
static_assert(kShardPackWords * sizeof(int32_t) == kSm70Tp4GatherTileBytes);
static_assert(kShardWorld * kShardPackWords * sizeof(int32_t) ==
              kLexicographicTopKThreads * sizeof(uint4));

__device__ __forceinline__ int shard_lower_bound(const int32_t* groups,
                                                 int size, int group) {
  int first = 0;
  int last = size;
  while (first < last) {
    const int mid = (first + last) / 2;
    if (groups[mid] < group) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  return first;
}

__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_shard_merge_topk(
    const int32_t* gathered, Sm70Tp4GatherMailbox* mailbox, int32_t* output) {
  __shared__ union {
    alignas(16) int32_t packs[kShardWorld][kShardPackWords];
    LexicographicDecodeTopKShared<kShardTopK, kShardCandidates> select;
  } scratch;
  __shared__ int32_t sizes[kShardWorld];
  __shared__ float scores[kShardCandidates];
  __shared__ int32_t groups[kShardCandidates];
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  constexpr int vectors_per_pack =
      kShardPackWords * sizeof(int32_t) / sizeof(uint4);
  const int source = tx / vectors_per_pack;
  const int vec = tx % vectors_per_pack;
  uint4 word;
  if (mailbox != nullptr) {
    const uint32_t sequence = mailbox->sequence[row];
    if (tx < kShardWorld) sm70_tp4_gather_wait(*mailbox, row, tx, sequence);
    __syncthreads();
    sm70_push_load_volatile_16b(
        word, mailbox->payload[sequence & 1][source][row], vec);
  } else {
    const size_t offset =
        (static_cast<size_t>(source) * gridDim.x + row) * kShardPackWords;
    word = reinterpret_cast<const uint4*>(gathered + offset)[vec];
  }
  reinterpret_cast<uint4*>(scratch.packs[source])[vec] = word;
  __syncthreads();

  if (tx < kShardWorld) {
    sizes[tx] = shard_lower_bound(scratch.packs[tx] + kShardTopK, kShardTopK,
                                  INT32_MAX);
  }
  __syncthreads();
  const uint32_t length = sizes[0] + sizes[1] + sizes[2] + sizes[3];
#pragma unroll
  for (int item = 0; item < kShardCandidates / kLexicographicTopKThreads;
       ++item) {
    const int index = tx + item * kLexicographicTopKThreads;
    const int peer = index / kShardTopK;
    const int lane = index % kShardTopK;
    if (lane < sizes[peer]) {
      const int32_t group = scratch.packs[peer][kShardTopK + lane];
      int position = lane;
#pragma unroll
      for (int other = 0; other < kShardWorld; ++other) {
        // Valid IDs are globally unique. Their lower bounds give the original
        // stable merge positions without sorting padding.
        if (other != peer) {
          position += shard_lower_bound(scratch.packs[other] + kShardTopK,
                                        sizes[other], group);
        }
      }
      scores[position] = __int_as_float(scratch.packs[peer][lane]);
      groups[position] = group;
    }
  }
  // All reads from the pack overlay must finish before using selector scratch.
  __syncthreads();
  qsa_lexicographic_decode_topk_body<kShardTopK, kShardCandidates, true>(
      scores, length, output + static_cast<size_t>(row) * kShardTopK,
      scratch.select, groups);
}

}  // namespace vllm::qsa
