// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "custom_all_reduce.cuh"

namespace vllm {

__host__ __device__ inline Sm70Tp4GatherMailbox* sm70_tp4_gather_mailbox(
    RankData peers, int rank) {
  auto* base = const_cast<char*>(static_cast<const char*>(peers.ptrs[rank]));
  return reinterpret_cast<Sm70Tp4GatherMailbox*>(
      base + kSm70Tp4PushAllreduceAndHcBytes);
}

DINLINE void sm70_tp4_gather_copy_tile(uint8_t* dst, const uint8_t* src,
                                       int bytes) {
  int vector_bytes = 0;
  if ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) %
          sizeof(uint4) ==
      0) {
    const int vectors = bytes / sizeof(uint4);
    for (int i = threadIdx.x; i < vectors; i += blockDim.x) {
      uint4 value;
      sm70_push_load_volatile_16b(value, src, i);
      sm70_push_store_volatile_16b(value, dst, i);
    }
    vector_bytes = vectors * sizeof(uint4);
  }
  for (int i = vector_bytes + threadIdx.x; i < bytes; i += blockDim.x) {
    reinterpret_cast<volatile uint8_t*>(dst)[i] =
        reinterpret_cast<const volatile uint8_t*>(src)[i];
  }
}

DINLINE void sm70_tp4_gather_wait(Sm70Tp4GatherMailbox& mailbox, int tile,
                                  int source, uint32_t sequence) {
  while (ld_flag_acquire(&mailbox.arrived[sequence & 1][source][tile]) !=
         sequence) {
  }
  // The sender's IPC mapping can alias the receiver's payload address.
  asm volatile("fence.proxy.alias;" ::: "memory");
}

__global__ void sm70_tp4_gather_send(RankData peers, const uint8_t* input,
                                     int rank, int bytes) {
  const int tile = blockIdx.x;
  const int offset = tile * kSm70Tp4GatherTileBytes;
  const int tile_bytes = min(bytes - offset, kSm70Tp4GatherTileBytes);
  auto& local = *sm70_tp4_gather_mailbox(peers, rank);
  // All lanes must snapshot the old sequence before lane 0 advances it.
  const uint32_t sequence = local.sequence[tile] + 1;
  const int page = sequence & 1;
  for (int peer = 0; peer < 4; ++peer) {
    auto& receiver = *sm70_tp4_gather_mailbox(peers, peer);
    sm70_tp4_gather_copy_tile(receiver.payload[page][rank][tile],
                              input + offset, tile_bytes);
  }
  // The CTA barrier orders every lane's payload stores before lane 0's
  // system-scope release publishes their completion.
  __syncthreads();
  if (threadIdx.x == 0) {
    local.sequence[tile] = sequence;
    for (int peer = 0; peer < 4; ++peer) {
      auto& receiver = *sm70_tp4_gather_mailbox(peers, peer);
      st_flag_release(&receiver.arrived[page][rank][tile], sequence);
    }
  }
}

__global__ void sm70_tp4_gather_receive(RankData peers, uint8_t* output,
                                        int rank, int bytes) {
  const int tile = blockIdx.x;
  const int source = blockIdx.y;
  const int offset = tile * kSm70Tp4GatherTileBytes;
  auto& local = *sm70_tp4_gather_mailbox(peers, rank);
  const uint32_t sequence = local.sequence[tile];
  const int page = sequence & 1;
  if (threadIdx.x == 0) {
    sm70_tp4_gather_wait(local, tile, source, sequence);
  }
  __syncthreads();
  sm70_tp4_gather_copy_tile(output + source * bytes + offset,
                            local.payload[page][source][tile],
                            min(bytes - offset, kSm70Tp4GatherTileBytes));
  // The next send on this stream requires this receive to finish. Reusing
  // page e requires consuming e+1, whose sends prove every peer finished e.
}

}  // namespace vllm
