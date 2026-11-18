# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""Metadata and sampling launches must not specialize on batch or token counts.

Each kernel here takes a row/length scalar (requests, query tokens, speculative
decodes, top-k, or a plane stride derived from them) that used to enter its
specialization key, so the first step at a new padded size recompiled it. The
gates run each launch in the Triton CPU interpreter over counts from the three
specialization classes and compare against an independent reference.
"""

import os

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backends.flash_attn_v100 import (
    _sm70_prepare_grouped_smallq_decode_metadata,
    _sm70_prepare_smallq_decode_metadata,
)
from vllm.v1.attention.backends.gdn_attn import (
    _dflash2_gdn_group_metadata_kernel,
)
from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

pytestmark = pytest.mark.skip_global_cleanup

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreter = pytest.mark.skipif(
    not INTERPRETED, reason="Triton kernels need CUDA or TRITON_INTERPRET=1"
)
COUNTS = (1, 2, 3, 15, 16, 17, 33)


def _reference_smallq_metadata(
    block_table, seq_lens, query_start_loc, *, num_query_tokens, real_num_query_tokens
):
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    repeat = query_lens.clone()
    repeat[-1] += num_query_tokens - real_num_query_tokens
    effective = torch.maximum(seq_lens, query_lens)
    table = torch.repeat_interleave(
        block_table.clamp_min(0), repeat, dim=0, output_size=num_query_tokens
    ).contiguous()
    seq = torch.repeat_interleave(effective, repeat, output_size=num_query_tokens)
    ql = torch.repeat_interleave(query_lens, repeat, output_size=num_query_tokens)
    start = torch.repeat_interleave(
        query_start_loc[:-1], repeat, output_size=num_query_tokens
    )
    token = torch.arange(num_query_tokens, dtype=seq_lens.dtype)
    decode = (seq - ql + token - start + 1).contiguous()
    if real_num_query_tokens < num_query_tokens:
        pad = token >= real_num_query_tokens
        decode = decode.masked_fill(pad, 0)
        table = table.masked_fill(pad[:, None], 0)
    return table, decode


@interpreter
@pytest.mark.parametrize("num_reqs", COUNTS)
@pytest.mark.parametrize("padding", (0, 2))
def test_grouped_smallq_metadata_matches_ungrouped_launch(num_reqs, padding):
    query_lens = [(row % 3) + 1 for row in range(num_reqs)]
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    real = int(query_start_loc[-1])
    num_query_tokens = real + padding
    if not num_query_tokens:
        pytest.skip("the launch requires at least one query token")
    seq_lens = torch.tensor([37 * (r + 1) for r in range(num_reqs)], dtype=torch.int32)
    tables = [
        torch.arange(num_reqs * 5, dtype=torch.int32).view(num_reqs, 5) + g * 500
        for g in range(2)
    ]
    for table in tables:
        table[::2, 0] = -1
    grouped = [
        (
            torch.zeros((num_query_tokens, 5), dtype=torch.int32),
            torch.zeros(num_query_tokens, dtype=torch.int32),
            torch.zeros(num_reqs + 1, dtype=torch.int32),
        )
        for _ in range(2)
    ]
    _sm70_prepare_grouped_smallq_decode_metadata(
        [g[0] for g in grouped],
        [g[1] for g in grouped],
        [g[2] for g in grouped],
        tables,
        seq_lens,
        query_start_loc,
        num_reqs=num_reqs,
        num_query_tokens=num_query_tokens,
        real_num_query_tokens=real,
    )
    for group, table in enumerate(tables):
        single_table = torch.zeros_like(grouped[group][0])
        single_seq = torch.zeros_like(grouped[group][1])
        single_query = torch.zeros_like(grouped[group][2])
        _sm70_prepare_smallq_decode_metadata(
            single_table,
            single_seq,
            single_query,
            table,
            seq_lens,
            query_start_loc,
            num_reqs=num_reqs,
            num_query_tokens=num_query_tokens,
            real_num_query_tokens=real,
        )
        expected_table, expected_seq = _reference_smallq_metadata(
            table,
            seq_lens,
            query_start_loc,
            num_query_tokens=num_query_tokens,
            real_num_query_tokens=real,
        )
        assert torch.equal(grouped[group][0], single_table)
        assert torch.equal(grouped[group][1], single_seq)
        assert torch.equal(grouped[group][0], expected_table)
        assert torch.equal(grouped[group][1], expected_seq)
        assert torch.equal(grouped[group][2], query_start_loc)


@interpreter
@pytest.mark.parametrize("num_spec_decodes", COUNTS)
@pytest.mark.parametrize("batch_size", (4, 16, 17))
def test_gdn_group_metadata_matches_torch(num_spec_decodes, batch_size):
    if num_spec_decodes > batch_size:
        pytest.skip("speculative decodes are a subset of the batch")
    width, state_columns, groups = 3, 2, 2
    tables = [
        torch.arange(num_spec_decodes * width + 1, dtype=torch.int32)[
            : num_spec_decodes * width
        ]
        .view(num_spec_decodes, width)
        .add_(g * 400)
        .contiguous()
        for g in range(groups)
    ]
    outputs = [
        torch.zeros(batch_size * width, dtype=torch.int32) for _ in range(groups)
    ]
    num_accepted = torch.tensor(
        [(r % 3) + 1 for r in range(batch_size)], dtype=torch.int32
    )
    query_start_loc = torch.tensor(
        [0, *[width * (r + 1) for r in range(num_spec_decodes)]], dtype=torch.int32
    )
    masks = torch.zeros(batch_size, dtype=torch.bool)
    common_query = torch.zeros(batch_size + 1, dtype=torch.int32)
    common_accepted = torch.zeros(batch_size, dtype=torch.int32)
    selectors = torch.zeros(batch_size, dtype=torch.int32)
    _dflash2_gdn_group_metadata_kernel[(groups,)](
        torch.tensor([t.data_ptr() for t in tables], dtype=torch.uint64),
        torch.tensor([o.data_ptr() for o in outputs], dtype=torch.uint64),
        torch.tensor([t.stride(0) for t in tables], dtype=torch.int64),
        num_accepted,
        num_accepted,
        num_accepted,
        query_start_loc,
        num_accepted,
        num_accepted,
        masks,
        common_query,
        common_accepted,
        selectors,
        num_spec_decodes,
        batch_size,
        WIDTH=width,
        PAD_ID=-1,
        BLOCK=triton.next_power_of_2(max(batch_size * width, batch_size + 1)),
        USE_STATE_START=False,
        USE_SEQ_LEN_START=False,
        MAMBA_BLOCK_SIZE=128,
        STATE_COLUMNS=state_columns,
        num_warps=1,
    )
    for group, (table, output) in enumerate(zip(tables, outputs, strict=True)):
        live = torch.cat(
            [table[:, :state_columns], table.new_full((num_spec_decodes, 1), -1)], dim=1
        ).flatten()  # fmt: skip
        tail = live.new_full(((batch_size - num_spec_decodes) * width,), -1)
        assert torch.equal(output, torch.cat([live, tail])), f"group {group}"
    assert masks.tolist() == [r < num_spec_decodes for r in range(batch_size)]
    # Rows past the speculative prefix load with other=1, not the source value.
    pad = [1] * (batch_size - num_spec_decodes)
    want_row = num_accepted[:num_spec_decodes].tolist() + pad
    assert common_accepted.tolist() == want_row
    assert selectors.tolist() == want_row
    assert torch.equal(common_query[: num_spec_decodes + 1], query_start_loc)
    assert (common_query[num_spec_decodes + 1 :] == int(query_start_loc[-1])).all()


@interpreter
@pytest.mark.parametrize("batch", COUNTS)
@pytest.mark.parametrize("topk", (1, 2, 15, 16, 17))
def test_topk_log_softmax_matches_torch(batch, topk):
    vocab = 400
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(batch, vocab, generator=generator)
    ids = torch.stack(
        [torch.randperm(vocab, generator=generator)[:topk] for _ in range(batch)]
    )
    got = compute_token_logprobs(logits, ids)
    expected = torch.log_softmax(logits.float(), dim=-1).gather(1, ids)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


@interpreter
@pytest.mark.parametrize("num_reqs", COUNTS)
def test_greedy_rejection_matches_torch(num_reqs):
    """The rejection plane strides come from the request/logit counts."""
    steps, vocab = 2, 400
    per_request = steps + 1
    num_logits = num_reqs * per_request
    generator = torch.Generator().manual_seed(17)
    target_logits = torch.randn(num_logits, vocab, generator=generator)
    target_argmax = target_logits.argmax(dim=1)
    draft_sampled = torch.empty(num_logits, dtype=torch.int64)
    draft_sampled[0::per_request] = target_argmax[0::per_request]
    for step in range(1, per_request):
        # Alternate an accepted draft with a mismatch, per request.
        row = torch.arange(num_reqs) * per_request + step
        if step % 2 == 1:
            draft_sampled[row] = target_argmax[row]
        else:
            draft_sampled[row] = (target_argmax[row] + 1) % vocab
    cu = torch.arange(num_reqs + 1, dtype=torch.int32) * per_request
    pos = torch.arange(num_logits, dtype=torch.int32)
    idx = torch.arange(num_reqs, dtype=torch.int32)
    expanded_idx = torch.arange(num_reqs, dtype=torch.int32).repeat_interleave(
        per_request
    )
    expanded_local = torch.arange(per_request, dtype=torch.int32).repeat(num_reqs)
    temperature = torch.zeros(num_reqs, dtype=torch.float32)
    seed = torch.arange(num_reqs, dtype=torch.int64)
    sampled, num_sampled = rejection_sample(
        target_logits,
        None,
        draft_sampled,
        cu,
        pos,
        idx,
        expanded_idx,
        expanded_local,
        temperature,
        seed,
        num_speculative_steps=steps,
    )
    for req in range(num_reqs):
        base = req * per_request
        accepted = 0
        while accepted < steps and int(draft_sampled[base + accepted + 1]) == int(
            target_argmax[base + accepted]
        ):
            accepted += 1
        want = [int(target_argmax[base + i]) for i in range(accepted)]
        got = sampled[req, : int(num_sampled[req])].tolist()
        assert got == want + [int(target_argmax[base + accepted])]
        assert int(num_sampled[req]) == accepted + 1
