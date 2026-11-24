# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM70 QSA XQA page4 workspace follows the decode kernel's dtype contract:
an FP32 temporary output for E4M3 KV, FP16 otherwise (#648)."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The XQA page4 workspace lives on CUDA"
)


@pytest.fixture(autouse=True)
def _empty_workspace_cache():
    saved = dict(qsa_ops._SM70_QSA_XQA_PAGE4_WORKSPACES)
    qsa_ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    yield
    qsa_ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    qsa_ops._SM70_QSA_XQA_PAGE4_WORKSPACES.update(saved)


def _query(rows: int = 3) -> torch.Tensor:
    return torch.zeros((rows, 6, 256), dtype=torch.float16, device="cuda")


@pytest.mark.parametrize("partial_dtype", [torch.float16, torch.float32])
def test_temporary_output_follows_partial_dtype(partial_dtype):
    temporary_output, max_logits, exp_sums, active = qsa_ops._qsa_xqa_page4_workspace(
        _query(), 8, partial_dtype
    )
    assert temporary_output.dtype == partial_dtype
    assert temporary_output.shape == (3, 6, 8, 256)
    assert max_logits.dtype == exp_sums.dtype == torch.float32
    assert active.dtype == torch.int32 and active.item() == 8


def test_e4m3_and_fp16_workspaces_are_cached_separately():
    q = _query()
    e4m3 = qsa_ops._qsa_xqa_page4_workspace(q, 8, torch.float32)[0]
    fp16 = qsa_ops._qsa_xqa_page4_workspace(q, 8, torch.float16)[0]
    assert (e4m3.dtype, fp16.dtype) == (torch.float32, torch.float16)
    again = qsa_ops._qsa_xqa_page4_workspace(q, 8, torch.float32)[0]
    assert again.data_ptr() == e4m3.data_ptr()


def test_workspace_can_be_created_during_graph_capture():
    q = _query()
    graph = torch.cuda.CUDAGraph()
    # The capture runs on a side stream, so the stream-keyed cache misses and
    # the workspace is allocated inside the capture.
    with torch.cuda.graph(graph):
        active = qsa_ops._qsa_xqa_page4_workspace(q, 8, torch.float32)[3]
    graph.replay()
    torch.accelerator.synchronize()
    assert active.item() == 8
