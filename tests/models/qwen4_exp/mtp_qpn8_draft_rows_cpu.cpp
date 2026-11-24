// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "fp8_qpn8_draft_rows.h"

#include <cassert>
#include <initializer_list>
#include <vector>

using vllm::sm70::dispatch_qpn8_draft_rows;
using vllm::sm70::supports_qpn8_two_phase;

static void check_rows(int64_t rows, bool gated,
                       std::initializer_list<int64_t> expected_blocks,
                       int expected_wide_launches, int expected_workspaces) {
  // TP4 plain and fused gate-up projection dimensions from the MTP drafter.
  const bool wide_eligible =
      supports_qpn8_two_phase(2560, gated ? 640 : 3584, gated ? 8 : 16, gated);
  std::vector<int64_t> input(rows), output(rows, -1);
  for (int64_t row = 0; row < rows; ++row) input[row] = row;
  auto write_rows = [&](int64_t begin, int64_t count) {
    for (int64_t row = begin; row < begin + count; ++row) {
      assert(output.at(row) == -1);
      output[row] = input[row];
    }
  };
  int64_t offset = 0;
  std::vector<int64_t> blocks;
  int wide_launches = 0, workspaces = 0;
  dispatch_qpn8_draft_rows(
      rows,
      [&](int64_t count) {
        if (!wide_eligible) return false;
        ++wide_launches;
        write_rows(0, count);
        return true;
      },
      [&](int64_t begin, int64_t count) {
        assert(begin == offset);
        blocks.push_back(count);
        write_rows(begin, count);
        offset += count;
      },
      [&] {
        ++workspaces;
        write_rows(0, rows);
      });
  assert(blocks == std::vector<int64_t>(expected_blocks));
  assert(wide_launches == expected_wide_launches);
  assert(workspaces == expected_workspaces);
  assert(output == input);
}

int main() {
  check_rows(33, false, {}, 1, 0);
  check_rows(40, false, {}, 1, 0);
  check_rows(33, true, {32, 1}, 0, 0);
  check_rows(40, true, {32, 8}, 0, 0);
  for (bool gated : {false, true}) {
    check_rows(32, gated, {32}, 0, 0);
    check_rows(41, gated, {32, 9}, 0, 0);
    check_rows(64, gated, {32, 32}, 0, 0);
    check_rows(128, gated, {32, 32, 32, 32}, 0, 0);
    check_rows(129, gated, {}, 0, 1);
  }
}
