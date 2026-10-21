# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE short-conv metadata on a mixed (spec + non-spec) batch must not need a
device->host read.

torch.repeat_interleave with a tensor of repeats and no output_size has to
read the repeats back to size its output; on CUDA that is a stream sync
(PyTorch docs: "If given, [output_size] will avoid stream synchronization").
The CPU container cannot observe the sync itself, so this test records the
call signature instead and checks the metadata is unchanged.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import short_conv_attn


@pytest.fixture(autouse=True)
def _async_indices(monkeypatch):
    # Production: VLLM_PLE_ASYNC_REQUEST_INDICES=1.
    monkeypatch.setenv("VLLM_PLE_ASYNC_REQUEST_INDICES", "1")


def _build(query_lengths, drafts, monkeypatch):
    monkeypatch.setattr(
        short_conv_attn, "mamba_get_block_table_tensor", lambda table, *_: table
    )
    monkeypatch.setattr(
        short_conv_attn, "compute_causal_conv1d_metadata", lambda *_a, **_k: (None,) * 3
    )
    builder = short_conv_attn.PleShortConvAttentionMetadataBuilder.__new__(
        short_conv_attn.PleShortConvAttentionMetadataBuilder
    )
    builder.use_spec_decode = True
    builder.use_full_cuda_graph = False
    builder.num_spec = 4
    builder.kv_cache_spec = None
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none")
    )
    starts = torch.tensor([0, *query_lengths], dtype=torch.int32).cumsum(0).int()
    n = len(query_lengths)
    slots = torch.arange(10, 10 + n, dtype=torch.int32).reshape(n, 1)
    batch = SimpleNamespace(
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        num_reqs=n,
        num_actual_tokens=sum(query_lengths),
        block_table_tensor=slots,
        seq_lens=torch.tensor(query_lengths, dtype=torch.int32) + 16,
        compute_num_computed_tokens=lambda: torch.full((n,), 16),
    )
    return builder.build(
        0,
        batch,
        num_accepted_tokens=(torch.arange(n, dtype=torch.int32) % 5 + 1),
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "query_lengths,drafts",
    [
        # 3 MTP4 verify rows + one 1632-token prefill chunk (mixed step).
        ([5, 5, 5, 1632], [4, 4, 4, -1]),
        # MTP4 verify rows + a fresh decode row (first step after prefill).
        ([5, 1, 5], [4, -1, 4]),
    ],
)
def test_mixed_batch_repeat_interleave_has_output_size(
    query_lengths, drafts, monkeypatch
):
    real = torch.repeat_interleave
    sized: list[bool] = []

    def spy(input, repeats=None, dim=None, *, output_size=None):
        if isinstance(repeats, torch.Tensor):
            sized.append(output_size is not None)
        return real(input, repeats, dim, output_size=output_size)

    monkeypatch.setattr(torch, "repeat_interleave", spy)
    metadata = _build(query_lengths, drafts, monkeypatch)

    def legacy(input, repeats=None, dim=None, *, output_size=None):
        return real(input, repeats, dim)

    monkeypatch.setattr(torch, "repeat_interleave", legacy)
    reference = _build(query_lengths, drafts, monkeypatch)

    assert sized, "mixed batch did not reach the tensor-repeats call"
    assert all(sized), "repeat_interleave without output_size (device sync on CUDA)"
    for name in (
        "spec_token_indx",
        "non_spec_token_indx",
        "spec_state_indices_tensor",
        "state_indices_tensor",
        "num_accepted_tokens",
    ):
        got, want = getattr(metadata, name, None), getattr(reference, name, None)
        if want is None:
            assert got is None
        else:
            torch.testing.assert_close(got, want, rtol=0, atol=0)
