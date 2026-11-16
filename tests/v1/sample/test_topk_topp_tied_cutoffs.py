# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.sample.ops.topk_topp_triton import (
    sort_topk_with_vocab_ties,
)
from vllm.v1.worker.gpu.spec_decode.dflash2 import sparse_rejection
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _compact_target_requires_reference,
)


def test_compact_guard_only_rejects_ambiguous_cutoffs():
    unique = torch.arange(21, 0, -1, dtype=torch.float32)[None] / 8
    assert not _compact_target_requires_reference(unique, 1.0, 0.95)
    k_tie = unique.clone()
    k_tie[:, -1] = k_tie[:, -2]
    assert _compact_target_requires_reference(k_tie, 1.0, 1.0)
    p_tie = torch.ones(1, 21)
    p_tie[:, :2] = 2.0
    p_tie[:, -1] = -20.0
    assert _compact_target_requires_reference(p_tie, 1.0, 0.95)
    assert not _compact_target_requires_reference(p_tie, 1.0, 1.0)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_compact_guard_checks_each_rows_sampling_parameters(temperature):
    unique = torch.arange(21, 0, -1, dtype=torch.float32) / 8
    tied = torch.ones(21)
    tied[:2] = 2.0
    tied[-1] = -20.0
    probe = torch.stack((unique, tied, tied))
    temperatures = np.array([1.0, temperature, temperature], dtype=np.float32)
    top_p = np.array([1.0, 1.0, 0.95], dtype=np.float32)
    # Using the first request's top_p for the whole batch misses the final
    # row's split tie, changing the retained vocabulary support.
    assert not _compact_target_requires_reference(probe, 1.0, 1.0)
    expected = any(
        _compact_target_requires_reference(probe[i : i + 1], t, p)
        for i, (t, p) in enumerate(zip(temperatures, top_p))
    )
    assert expected
    assert _compact_target_requires_reference(probe, temperatures, top_p) == expected


@pytest.mark.parametrize(
    ("temperatures", "top_ps"),
    [([0.5, 2.0, 1.0], [0.95, 0.9, 1.0]), ([1.0, 2.0, 0.1], [0.8, 0.9, 0.8])],
)
def test_compact_rejection_uses_request_mapping_and_variable_row_counts(
    monkeypatch, temperatures, top_ps
):
    class Speculator:
        def get_sparse_draft_logits(self):
            return None, None

    monkeypatch.setattr(sparse_rejection, "DFlash2Speculator", Speculator)
    monkeypatch.setattr(
        sparse_rejection.envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", True
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    monkeypatch.setattr(
        sparse_rejection, "_supports_sparse_sampling_contract", lambda *args: True
    )
    probe = torch.ones(3, 21)
    probe[:, :2] = 2.0
    probe[:, -1] = -20.0
    states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array(temperatures)),
        top_p=SimpleNamespace(np=np.array(top_ps)),
    )
    batch = SimpleNamespace(
        has_structured_output_reqs=False,
        idx_mapping_np=np.array([2, 0]),
        cu_num_logits_np=np.array([0, 1, 3]),
    )
    # Request 2 is unambiguous; the two rows of request 0 need the reference.
    # No GPU sampling should be attempted after detecting the split tie.
    result = sparse_rejection.try_dflash2_sparse_target_rejection(
        SimpleNamespace(get_topk_tokens_and_logits=lambda *args: (None, probe)),
        Speculator(),
        SimpleNamespace(sampler=SimpleNamespace(sampling_states=states)),
        SimpleNamespace(device=SimpleNamespace(type="cuda")),
        batch,
        None,
    )
    assert result is None


@pytest.mark.parametrize("width", [21, 64, 128])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shortlist_order_preserves_radix_ties_and_value_bits(width, descending):
    torch.manual_seed(1540)
    ids = torch.stack(
        [torch.randperm(248320, device="cuda")[: width + 3] for _ in range(8)]
    )[:, :width]
    storage = torch.randn(8, width + 3, device="cuda")
    values = storage[:, :width]
    output = sort_topk_with_vocab_ties(
        values, ids, vocab_size=248320, descending=descending
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sort_topk_with_vocab_ties(
            values, ids, vocab_size=248320, descending=descending
        )
    for amplitude in (0.1, 2.0):
        values.normal_(0, amplitude)
        values[:, :12] = 1.0
        values[:, 12:18] = torch.tensor(
            [-0.0, 0.0, -float("inf"), float("inf"), float("nan"), -float("nan")],
            device="cuda",
        )
        ordered_ids, permutation = ids.sort(dim=-1, descending=descending)
        ordered_values = values.gather(1, permutation)
        ordered_values, permutation = ordered_values.sort(
            dim=-1, descending=descending, stable=True
        )
        ordered_ids = ordered_ids.gather(1, permutation)
        graph.replay()
        assert torch.equal(output[1], ordered_ids)
        assert torch.equal(
            output[0].view(torch.int32), ordered_values.view(torch.int32)
        )
