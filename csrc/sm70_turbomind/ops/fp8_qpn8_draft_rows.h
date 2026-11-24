// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <algorithm>
#include <cstdint>

namespace vllm::sm70 {

inline constexpr int64_t kQpn8DraftRowBlock = 32;
inline constexpr int64_t kQpn8DraftNativeRows = 40;
inline constexpr int64_t kQpn8DraftMaxChunkedRows = 128;

inline bool supports_qpn8_two_phase(int64_t k, int64_t packed_n,
                                    int64_t split_k, bool gated) {
  if (gated) {
    return split_k == 8 && k >= 4096 && packed_n >= 4096;
  }
  return split_k == 12 || split_k == 16;
}

template <typename TryWide, typename RunBlock, typename RunDense>
void dispatch_qpn8_draft_rows(int64_t rows, TryWide try_wide,
                              RunBlock run_block, RunDense run_dense) {
  if (rows <= kQpn8DraftRowBlock) {
    run_block(0, rows);
  } else if (rows > kQpn8DraftMaxChunkedRows) {
    run_dense();
  } else if (rows > kQpn8DraftNativeRows || !try_wide(rows)) {
    for (int64_t begin = 0; begin < rows; begin += kQpn8DraftRowBlock) {
      run_block(begin, std::min(kQpn8DraftRowBlock, rows - begin));
    }
  }
}

}  // namespace vllm::sm70
