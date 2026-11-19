# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.metrics.logits import sum_rows_per_request

pytestmark = pytest.mark.skip_global_cleanup


def test_sum_rows_per_request_with_draft_rows():
    # Five requests owning 5, 4, 1, 5 and 3 logits rows (target plus drafts).
    cu_num_logits = torch.tensor([0, 5, 9, 10, 15, 18], dtype=torch.int32)
    row_counts = torch.zeros(18, dtype=torch.int32)
    row_counts[1], row_counts[4], row_counts[6] = 2, 7, 1
    row_counts[13], row_counts[17] = 3, 5
    per_request = sum_rows_per_request(row_counts, cu_num_logits)
    assert per_request.tolist() == [9, 1, 0, 3, 5]


def test_sum_rows_per_request_one_row_each():
    cu_num_logits = torch.arange(4, dtype=torch.int32)
    row_counts = torch.tensor([0, 4, 2], dtype=torch.int32)
    assert sum_rows_per_request(row_counts, cu_num_logits).tolist() == [0, 4, 2]
