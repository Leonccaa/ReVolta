# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""Row-count parameters must stay runtime scalars, not specialization keys.

A shape-specialized launch recompiles the kernel the first time inference
reaches it, so every row/length/count argument listed here is a runtime scalar
marked ``do_not_specialize`` (the ``commit_slots`` fix from the TTFT analysis).
The declaration gate asserts that without a GPU; the numeric gates run the same
kernel in the Triton CPU interpreter over counts that used to select different
specialization classes (equal to one, divisible by 16, and neither).
"""

import math
import os

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_request_segments as segment_ops
from vllm.models.qwen4_exp.nvidia.sm70_ngram_input import prepare_ngram_input
from vllm.triton_utils import triton

pytestmark = pytest.mark.skip_global_cleanup

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
interpreter = pytest.mark.skipif(
    not INTERPRETED, reason="Triton kernels need CUDA or TRITON_INTERPRET=1"
)
compiled_only = pytest.mark.skipif(
    INTERPRETED, reason="reads the compiled JIT parameter metadata"
)
# Counts that used to land in different specialization classes: equal to one,
# divisible by 16 and neither.
COUNTS = (1, 2, 3, 15, 16, 17, 33)


def check_scalars(kernel, names):
    params = {p.name: p for p in kernel.params}
    for name in names:
        assert name in params, f"{kernel.fn.__name__} has no parameter {name}"
        assert not params[name].is_constexpr, (
            f"{kernel.fn.__name__}.{name} is still a constexpr"
        )
        assert params[name].do_not_specialize, (
            f"{kernel.fn.__name__}.{name} is still specialized"
        )


def check_tile_constants(kernel, names):
    params = {p.name: p for p in kernel.params}
    for name in names:
        assert params[name].is_constexpr, (
            f"{kernel.fn.__name__}.{name} is a tile constant, not a runtime count"
        )


@compiled_only
@pytest.mark.parametrize(
    "kernel,names",
    [
        (qsa_ops._qsa_mqa_paged_kernel, ["num_requests", "num_rows", "num_columns"]),
        (qsa_ops._qsa_visible_blocks_kernel, ["rows", "num_requests"]),
        (qsa_ops._qsa_gather_single_request_keys_kernel, ["columns"]),
        (
            qsa_ops._qsa_relu_headsum_visible_kernel,
            ["width", "stride_score_row"],
        ),
        (qsa_ops._expand_qsa_indices_kernel, ["rows", "num_requests"]),
        (qsa_ops._qsa_sparse_paged_gqa_splitk_kernel, ["num_requests", "num_rows"]),
        (qsa_ops._qsa_merge_splitk_kernel, ["num_rows"]),
        (qsa_ops._index_shard_pack_or_map_kernel, ["stride_values_row"]),
        (segment_ops._build_segments, ["num_rows", "num_requests"]),
    ],
)
def test_row_counts_are_unspecialized_runtime_scalars(kernel, names):
    check_scalars(kernel, names)


@compiled_only
@pytest.mark.parametrize(
    "kernel,names",
    [
        (qsa_ops._qsa_mqa_paged_kernel, ["PAGE_SIZE", "NUM_HEADS", "BLOCK_N"]),
        (qsa_ops._qsa_sparse_paged_gqa_splitk_kernel, ["TOPK", "PAGE_SIZE", "BLOCK_N"]),
        (qsa_ops._qsa_merge_splitk_kernel, ["NUM_SPLITS", "BLOCK_SPLITS"]),
        (segment_ops._build_segments, ["MAX_ROWS", "BLOCK_M"]),
    ],
)
def test_tile_constants_stay_constexpr(kernel, names):
    check_tile_constants(kernel, names)


# ---------------------------------------------------------------------------
# Interpreter gates: output must not depend on the count's specialization class


def _mqa_inputs(rows, columns, requests=2, page=16, compress=4):
    generator = torch.Generator().manual_seed(7)
    width = max(1, math.ceil(columns / page))

    def values(shape):
        return torch.randint(-8, 9, shape, generator=generator).float().div_(8).half()

    q = values((rows, 4, 128))
    cache = values((width * requests + 2, page, 1, 128))
    table = torch.arange(width * requests, dtype=torch.int32).reshape(requests, width)
    token_to_req = torch.arange(rows, dtype=torch.int32) % requests
    positions = torch.arange(rows, dtype=torch.int64) % (columns * compress)
    lengths = torch.full((requests,), columns * compress, dtype=torch.int32)
    return q, cache, table, token_to_req, positions, lengths


def _launch_mqa(rows, columns, tensors=None):
    q, cache, table, token_to_req, positions, lengths = (
        _mqa_inputs(rows, columns) if tensors is None else tensors
    )
    visible = torch.zeros(rows, dtype=torch.int32)
    logits = torch.full((rows, columns), float("nan"), dtype=torch.float32)
    qsa_ops._qsa_mqa_paged_kernel[(rows, triton.cdiv(columns, 64))](
        q,
        cache,
        table,
        token_to_req,
        positions,
        lengths,
        visible,
        logits,
        *q.stride(),
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        *table.stride(),
        logits.stride(0),
        rows,
        columns,
        cache.shape[0],
        table.shape[0],
        math.sqrt(128),
        PAGE_SIZE=16,
        PAGE_TABLE_WIDTH=table.shape[1],
        NUM_HEADS=4,
        HEAD_DIM=128,
        BLOCK_N=64,
        BLOCK_D=128,
        TILES_PER_PROG=1,
        STAGES=1,
        MAX_N=16,
        COMPRESS_RATIO=4,
        num_warps=4,
    )
    return visible, logits


@interpreter
@pytest.mark.parametrize("rows", COUNTS)
@pytest.mark.parametrize("columns", (1, 17, 64))
def test_mqa_paged_scores_match_torch(rows, columns):
    tensors = _mqa_inputs(rows, columns)
    visible, logits = _launch_mqa(rows, columns, tensors)
    q, cache, table, token_to_req, positions, lengths = tensors
    expected = torch.full((rows, columns), float("nan"), dtype=torch.float32)
    for row in range(rows):
        request = int(token_to_req[row])
        live = min(
            (int(positions[row]) + 1) // 4, int(lengths[request]) // 4, columns
        )  # fmt: skip
        for column in range(live):
            key = cache[int(table[request, column // 16]), column % 16, 0].float()
            heads = (q[row].float() * key).sum(1).clamp(min=0)
            expected[row, column] = heads.sum() / math.sqrt(128)
        assert int(visible[row]) == min(
            (int(positions[row]) + 1) // 4, int(lengths[request]) // 4
        )
    # Unscored columns stay at the NaN the caller allocated; compare the two
    # parts separately because NaN never compares equal.
    scored = ~torch.isnan(expected)
    assert bool((torch.isnan(logits) == ~scored).all())
    assert torch.equal(logits[scored], expected[scored])


@interpreter
@pytest.mark.parametrize("rows", COUNTS)
@pytest.mark.parametrize("num_requests", (1, 2, 16, 17))
def test_visible_blocks_matches_torch(rows, num_requests):
    columns = 37
    token_to_req = torch.arange(rows, dtype=torch.int32) % num_requests
    positions = torch.arange(rows, dtype=torch.int64) * 3
    lengths = torch.arange(1, num_requests + 1, dtype=torch.int32) * columns + 3
    out = torch.zeros(rows, dtype=torch.int32)
    qsa_ops._qsa_visible_blocks_kernel[(rows,)](
        token_to_req,
        positions,
        lengths,
        out,
        rows,
        num_requests,
        COMPRESS_RATIO=4,
        num_warps=1,
    )
    expected = torch.minimum(
        (positions + 1) // 4, lengths[token_to_req.long()] // 4
    ).int()
    assert torch.equal(out, expected)


@interpreter
@pytest.mark.parametrize("columns", COUNTS)
def test_gather_single_request_keys_matches_torch(columns):
    page, dim, pages = 16, 32, 5
    generator = torch.Generator().manual_seed(11)
    cache = torch.randint(
        -128, 127, (pages + 1, page, 1, dim), generator=generator, dtype=torch.int32
    ).to(torch.int8)
    table = torch.arange(columns // page + 2, dtype=torch.int32) % (pages + 1)
    keys = torch.zeros(columns, dim, dtype=torch.int8)
    valid = torch.zeros(columns, dtype=torch.int8)
    qsa_ops._qsa_gather_single_request_keys_kernel[(columns,)](
        cache,
        table,
        keys,
        valid,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        table.stride(0),
        keys.stride(0),
        columns,
        pages,
        PAGE_SIZE=page,
        PAGE_TABLE_WIDTH=table.shape[0],
        HEAD_DIM=dim,
        BLOCK_D=dim,
        num_warps=1,
    )
    flat = cache[:, :, 0, :]
    expected = torch.stack(
        [flat[int(table[c // page]), c % page] for c in range(columns)]
    )
    assert torch.equal(keys, expected)
    assert bool(valid[:columns].to(torch.bool).all())


@interpreter
@pytest.mark.parametrize("width", COUNTS)
def test_relu_headsum_matches_torch(width):
    rows, heads, column_start, block = 3, 4, 16, 8
    generator = torch.Generator().manual_seed(5)
    # The score buffer is one cuBLAS tile: the kernel indexes it from column 0,
    # while logits and key_valid are addressed by the absolute column.
    scores = torch.randint(-4, 5, (rows * heads, width), generator=generator)
    visible = torch.tensor(
        [width + 5, column_start, width + column_start], dtype=torch.int32
    )
    key_valid = torch.ones(width + column_start, dtype=torch.int8)
    key_valid[-1] = 0
    logits = torch.zeros(rows, width + column_start, dtype=torch.float32)
    qsa_ops._qsa_relu_headsum_visible_kernel[(rows, triton.cdiv(width, block))](
        scores,
        visible,
        key_valid,
        logits,
        scores.stride(0),
        scores.stride(1),
        logits.stride(0),
        width,
        column_start,
        2.0,
        NUM_HEADS=heads,
        BLOCK_N=block,
        num_warps=1,
    )
    summed = scores.view(rows, heads, -1).float().clamp(min=0).sum(1) / 2.0
    absolute = torch.arange(column_start, column_start + width)
    live = (absolute[None, :] < visible[:, None]) & key_valid.bool()[
        None, column_start:
    ]
    window = logits[:, column_start : column_start + width]
    expected = torch.where(live, summed, torch.tensor(-float("inf")))
    assert torch.equal(window, expected)
    # Untouched columns keep their pre-launch value, so the width bound holds.
    assert (logits[:, column_start + width :] == 0).all()
    assert (logits[:, :column_start] == 0).all()


@interpreter
@pytest.mark.parametrize("rows", COUNTS)
def test_expand_indices_matches_torch(rows):
    compress, token_topk, num_requests = 4, 12, 2
    block_topk = token_topk // compress
    block_indices = torch.stack(
        [torch.arange(block_topk, dtype=torch.int32) + r for r in range(rows)]
    )
    positions = torch.arange(rows, dtype=torch.int64) * 3
    lengths = torch.full((num_requests,), 400, dtype=torch.int32)
    token_to_req = torch.arange(rows, dtype=torch.int32) % num_requests
    out = torch.full(
        (rows, token_topk + compress - 1), -7, dtype=torch.int32
    )  # fmt: skip
    qsa_ops._expand_qsa_indices_kernel[(rows, 1)](
        block_indices,
        positions,
        lengths,
        token_to_req,
        out,
        *block_indices.stride(),
        *out.stride(),
        rows,
        num_requests,
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=out.shape[1],
        COLUMN_BLOCK=256,
        num_warps=4,
    )
    for row in range(rows):
        position = int(positions[row])
        sequence_length = int(lengths[int(token_to_req[row])])
        complete = min(
            (position + 1) // compress, sequence_length // compress, block_topk
        )
        expanded_count = complete * compress
        tail_start = ((position + 1) // compress) * compress
        tail_count = (position + 1) - tail_start
        expected = []
        for column in range(out.shape[1]):
            if column < expanded_count:
                token = int(block_indices[row, column // compress]) * compress + (
                    column % compress
                )
            elif (
                column >= expanded_count
                and column - expanded_count < tail_count
                and column - expanded_count < compress - 1
            ):
                token = tail_start + column - expanded_count
            else:
                token = -1
            expected.append(token if 0 <= token < sequence_length else -1)
        assert out[row].tolist() == expected


@interpreter
@pytest.mark.parametrize("rows", COUNTS)
def test_index_shard_pack_matches_torch(rows):
    k, world, rank = 5, 2, 1
    blocks = torch.randint(0, 64, (rows, k), dtype=torch.int32)
    blocks[::2, -1] = -1
    logits = torch.randn(rows, 64, generator=torch.Generator().manual_seed(3))
    packed = qsa_ops._pack_index_shard_candidates(blocks, logits, world, rank)
    pad = torch.iinfo(torch.int32).max
    for row in range(rows):
        for lane in range(k):
            index = int(blocks[row, lane])
            if index >= 0:
                assert packed[row, lane].view(torch.float32) == logits[row, index]
                assert int(packed[row, k + lane]) == index * world + rank
            else:
                assert int(packed[row, k + lane]) == pad


@interpreter
@pytest.mark.parametrize("m", COUNTS)
@pytest.mark.parametrize("max_rows", (1, 2, 4))
def test_build_segments_owners_are_exact(m, max_rows):
    num_requests = 2
    requests = torch.arange(m, dtype=torch.int32) % num_requests
    positions = torch.arange(m, dtype=torch.int64)
    lengths = torch.full((num_requests,), 300, dtype=torch.int32)
    segments = torch.full((m, 3), -7, dtype=torch.int32)
    visible = torch.zeros(m, dtype=torch.int32)
    segment_ops._build_segments[(1,)](
        requests,
        positions,
        lengths,
        segments,
        visible,
        requests.stride(0),
        positions.stride(0),
        lengths.stride(0),
        m,
        num_requests,
        4,
        MAX_ROWS=max_rows,
        BLOCK_M=triton.next_power_of_2(max(m, 1)),
        num_warps=4,
    )
    owners = torch.zeros(m, dtype=torch.int32)
    values = requests.tolist()
    begin = 0
    while begin < m:
        end = begin + 1
        while end < m and values[end] == values[begin] and end - begin < max_rows:
            end += 1
        assert segments[begin].tolist() == [values[begin], begin, end]
        owners[begin:end] += 1
        begin = end
    assert torch.equal(owners, torch.ones(m, dtype=torch.int32))
    assert torch.equal(
        visible, torch.minimum((positions + 1) // 4, torch.tensor(75)).int()
    )


@interpreter
@pytest.mark.parametrize("num_reqs", COUNTS)
@pytest.mark.parametrize("padded", (1, 4, 16, 17))
def test_prepare_ngram_matches_torch(num_reqs, padded):
    length, eos = 5, 999
    generator = torch.Generator().manual_seed(9)
    tokens = torch.randint(0, 50, (4, 64), generator=generator, dtype=torch.int32)
    mapping = torch.randint(0, 4, (padded + 1,), generator=generator, dtype=torch.int32)
    # The runtime keeps computed lengths for every request slot, not only the
    # padded ones, exactly as the real input batch does.
    computed = torch.randint(2, 40, (4,), generator=generator, dtype=torch.int32)
    context = torch.full((padded, length), -7, dtype=torch.int32)
    query = torch.arange(padded + 1, dtype=torch.int32) * 5
    query_out = torch.empty_like(query)
    prepare_ngram_input(
        context, query_out, query, mapping, computed, tokens, num_reqs, eos
    )
    for row in range(padded):
        request = int(mapping[row])
        end = int(computed[request])
        expected = []
        for col in range(length):
            position = end - length + col
            live = row < num_reqs and position >= 0
            expected.append(int(tokens[request, position]) if live else eos)
        assert context[row].tolist() == expected
    assert torch.equal(query_out, query)
