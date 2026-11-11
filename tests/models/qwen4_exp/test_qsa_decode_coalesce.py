# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU interpreter: raw bits, DCP2 order, fixed scratch and whole-step fallback."""

import os
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as ops
from vllm.models.qwen4_exp.nvidia.ops.qsa_coalesce import (
    QSACoalescedExchange,
    QSACoalesceWorkspace,
)

from .test_host_kv_forward import _cpu_kernels
from .test_qsa_index_dcp_shard import _reference_select

__all__ = ["_cpu_kernels"]

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="CPU interpreter gate"
)


class Packed(Exception):
    pass


def _inputs(rows, k, heads, dim, dtype, strided=False):
    generator = torch.Generator().manual_seed(rows + k)
    result = []
    for rank in range(2):
        # Every int16 bit pattern includes NaNs, infinities, zeros and denormals.
        raw = torch.randint(
            -32768, 32768, (rows, heads, dim), dtype=torch.int16, generator=generator
        )
        special = torch.tensor(
            [0, -32768, 31744, -1024, 31745, 32257, 1, -1], dtype=torch.int16
        )
        raw.reshape(-1)[: min(raw.numel(), 8)] = special[: min(raw.numel(), 8)]
        query = raw.view(dtype)
        if strided:
            padded = torch.empty((rows, heads + 1, dim), dtype=dtype)
            padded[:, :heads].copy_(query)
            query = padded[:, :heads]
        # Includes signaling NaNs and nontrivial payloads; never FP copies.
        bits = torch.randint(
            -(2**31), 2**31 - 1, (rows, k + 17), dtype=torch.int32, generator=generator
        )
        bits[:, :8] = torch.tensor(
            [
                0,
                -(2**31),
                2139095040,
                -8388608,
                2139095041,
                2143289345,
                -4194303,
                1065353216,
            ]
        )
        blocks = torch.arange(k, dtype=torch.int32).expand(rows, -1).clone()
        # rank-local empty rows and differing valid tails must not affect routing.
        for row in range(rows):
            blocks[row, (row * 7 + rank * 3) % (k + 1) :] = -1
        result.append((query, blocks, bits.view(torch.float32)))
    return result


def _packed(inputs, k, heads, dim):
    packed = []
    for rank, (query, blocks, logits) in enumerate(inputs):
        ws = QSACoalesceWorkspace(query.shape[0], k, heads, dim, query.dtype)

        def capture(receive, send):
            packed.append(send.clone())
            raise Packed

        with pytest.raises(Packed):
            QSACoalescedExchange(ws, query).exchange(blocks, logits, rank, capture)
    return torch.stack(packed)


@pytest.mark.parametrize(
    "rows,k,heads,dim", [(1, 8, 2, 16), (8, 512, 6, 256), (16, 512, 6, 256)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_exchange_bits_order_strides_and_reuse(rows, k, heads, dim, dtype, strided):
    inputs = _inputs(rows, k, heads, dim, dtype, strided)
    packed = _packed(inputs, k, heads, dim)
    original = torch.stack(
        [
            ops._pack_index_shard_candidates(blocks, logits, 2, rank)
            for rank, (_, blocks, logits) in enumerate(inputs)
        ]
    )
    assert torch.equal(packed[:, :, : 2 * k], original)
    expected_q = torch.cat([item[0] for item in inputs], dim=1)
    ws = QSACoalesceWorkspace(rows + 8, k, heads, dim, dtype)
    pointers = [t.data_ptr() for t in (ws.send, ws.receive, ws.query)]
    for rank in range(2):
        query, blocks, logits = inputs[rank]
        calls: list[Any] = []

        def gather(receive, send, calls=calls, rank=rank):
            calls.append((receive.shape, send.shape))
            assert torch.equal(send, packed[rank])
            receive.copy_(packed)

        transaction = QSACoalescedExchange(ws, query)
        candidates = transaction.exchange(blocks, logits, rank, gather)
        assert len(calls) == 1
        assert torch.equal(candidates, original)
        assert transaction.gathered_query is not None
        assert transaction.gathered_query.shape == expected_q.shape
        assert transaction.gathered_query.stride() == expected_q.stride()
        assert torch.equal(
            transaction.gathered_query.view(torch.int16), expected_q.view(torch.int16)
        )
        # The existing merge must accept the packed receive's wider row stride.
        actual = ops._sort_index_shard_candidates(candidates)
        expected = ops._sort_index_shard_candidates(original)
        for a, e in zip(actual, expected):
            assert torch.equal(a.view(torch.int32), e.view(torch.int32))
        assert pointers == [t.data_ptr() for t in (ws.send, ws.receive, ws.query)]


@pytest.mark.parametrize("rows", [1, 8])
def test_two_rank_merge_matches_original_with_one_collective(monkeypatch, rows):
    k, heads, dim = 8, 2, 16
    inputs = _inputs(rows, k, heads, dim, torch.float16)
    packed = _packed(inputs, k, heads, dim)
    original = packed[:, :, : 2 * k].contiguous()
    for rank, (query, blocks, logits) in enumerate(inputs):
        calls: list[Any] = []

        def gather(receive, send, *, group, calls=calls):
            assert group == "fake-dcp2"
            calls.append(send.clone())
            receive.copy_(packed.view(2 * rows, -1))

        monkeypatch.setattr("torch.distributed.all_gather_into_tensor", gather)
        monkeypatch.setattr(
            "vllm.distributed.get_dcp_group", lambda: NS(device_group="fake-dcp2")
        )
        ws = QSACoalesceWorkspace(rows, k, heads, dim, query.dtype)
        exchange = QSACoalescedExchange(ws, query)
        actual = ops._merge_sharded_index_blocks(
            blocks,
            logits,
            torch.zeros(rows, dtype=torch.int32),
            2,
            rank,
            k,
            use_triton=True,
            select=_reference_select,
            coalesced_exchange=exchange,
        )
        expected = ops._merge_sharded_index_blocks(
            blocks,
            logits,
            torch.zeros(rows, dtype=torch.int32),
            2,
            rank,
            k,
            use_triton=True,
            select=_reference_select,
            all_gather=lambda t: original,
        )
        assert torch.equal(actual, expected)
        assert len(calls) == 1


@pytest.mark.parametrize(
    "world,chunked,debug,fused",
    [
        (2, False, False, True),
        (1, False, False, True),
        (2, True, False, True),
        (2, False, True, True),
        (2, False, False, False),
    ],
)
def test_selection_whole_step_fallback(monkeypatch, world, chunked, debug, fused):
    rows, k = 8, 8
    query = torch.zeros(rows, 2, 16, dtype=torch.float16)
    exchange = QSACoalescedExchange(
        QSACoalesceWorkspace(rows, k, 2, 16, query.dtype), query
    )
    monkeypatch.setenv("VLLM_QSA_INDEX_SHARD_FUSED_MERGE", str(int(fused)))
    monkeypatch.setenv("VLLM_QSA_INDEX_SHARD_DEBUG", str(int(debug)))
    monkeypatch.setattr(
        ops, "_LOGITS_WORKSPACE_BYTES", 32 * 4 * (rows // 2 if chunked else rows)
    )
    monkeypatch.setattr(ops, "_use_sm70_qsa_lexicographic_topk", lambda k: True)
    monkeypatch.setattr(
        ops, "_sm70_qsa_lexicographic_topk_op", lambda: _reference_select
    )
    monkeypatch.setattr(ops, "_use_sm70_qsa_indexer_cublas", lambda *a: False)
    monkeypatch.setattr(ops, "_use_sm70_qsa_request_segments", lambda *a: False)
    monkeypatch.setattr(
        ops,
        "qsa_mqa_paged",
        lambda q, *a, **kw: (
            torch.ones(q.shape[0], 32),
            torch.full((q.shape[0],), 32, dtype=torch.int32),
        ),
    )
    seen = []

    def merge(blocks, *args, **kwargs):
        seen.append(kwargs.get("coalesced_exchange"))
        return blocks

    monkeypatch.setattr(ops, "_merge_sharded_index_blocks", merge)
    monkeypatch.setattr(
        ops, "expand_qsa_block_indices_cuda", lambda *a: a[-1].fill_(-1)
    )
    args = (
        query,
        torch.zeros(1, 32, 1, 16),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows),
        torch.tensor([32]),
        k * 4,
        4,
    )
    with_exchange = ops.qsa_select_paged_tokens(
        *args, index_dcp_world_size=world, coalesced_exchange=exchange
    )
    expected = ops.qsa_select_paged_tokens(*args, index_dcp_world_size=world)
    assert torch.equal(with_exchange, expected)
    if world > 1:
        eligible = not chunked and not debug and fused
        assert all(
            item is (exchange if eligible else None) for item in seen[: len(seen) // 2]
        )
        assert all(item is None for item in seen[len(seen) // 2 :])
    else:
        assert not seen
    assert exchange.gathered_query is None


@pytest.mark.parametrize(
    "mode",
    [
        "off",
        "decode",
        "prefill",
        "mixed",
        "piecewise",
        "world1",
        "padded-mismatch",
        "oversize",
        "skip",
    ],
)
def test_owner_disabled_and_ineligible_call_signatures(monkeypatch, mode):
    from vllm.config import CUDAGraphMode
    from vllm.models.qwen4_exp.nvidia import qsa as owner

    rows, heads, dim, width = 8, 2, 16, 35
    workspace = (
        None
        if mode == "off"
        else QSACoalesceWorkspace(
            4 if mode == "oversize" else rows, 8, heads, dim, torch.float16
        )
    )
    query = torch.zeros(rows, heads, dim, dtype=torch.float16)
    side = NS(
        num_actual_tokens=rows,
        index_dcp_world_size=1 if mode == "world1" else 2,
        token_to_req=torch.zeros(rows, dtype=torch.int32),
        seq_lens=torch.tensor([64]),
        logical_positions=torch.arange(rows),
    )
    context = NS(
        attn_metadata={
            "attn": NS(num_actual_tokens=rows, slot_mapping=torch.arange(rows)),
            "raw": side,
        },
        batch_descriptor=NS(
            uniform=mode not in ("prefill", "mixed"),
            num_tokens=rows + 1 if mode == "padded-mismatch" else rows,
        ),
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
        if mode == "piecewise"
        else CUDAGraphMode.FULL,
    )
    monkeypatch.setattr(owner, "get_forward_context", lambda: context)
    monkeypatch.setattr(owner, "_sm70_dump_qwen_layer_tensor", lambda *a: a[-1])
    seen = []

    class Indexer:
        skip_topk = mode == "skip"
        layer_id, output_width = 3, width
        raw_key_cache = NS(prefix="raw")

        def __call__(self, hidden, positions, out, **kwargs):
            seen.append(kwargs)
            if kwargs:
                kwargs["coalesced_exchange"].gathered_query = torch.cat(
                    (query, query), dim=1
                )
            out.fill_(-1)
            return out

    calls = []
    layer = NS(
        _qsa_kv_scales_finalized=True,
        layer_name="attn",
        host_history=NS(write=lambda *a: None),
        coalesce_workspace=workspace,
        qsa_dcp_sharded=True,
        indexer=Indexer(),
        topk_indices_buffer=torch.empty(rows, width, dtype=torch.int32),
        impl=NS(forward_qsa_host=lambda *a, **kw: calls.append(kw)),
        kv_cache_dtype="auto",
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    owner.Qwen4ExpQSAAttention._run_qsa(
        layer, query, torch.arange(rows), query, query, query, query.clone()
    )
    assert bool(seen[0]) == (mode == "decode")
    assert ("gathered_query" in calls[0]) == (mode == "decode")


def test_registered_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("REVOLTA_RESEARCH_QSA_DECODE_COALESCE_AG", raising=False)
    assert not envs.REVOLTA_RESEARCH_QSA_DECODE_COALESCE_AG


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_every_query_bit_pattern(dtype):
    # All 65536 possible words, including every signaling NaN payload.
    rows, heads, dim, k = 128, 2, 256, 8
    query = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    query = query.view(dtype).view(rows, heads, dim)
    blocks = torch.full((rows, k), -1, dtype=torch.int32)
    logits = torch.empty(rows, 1)
    inputs = [(query, blocks, logits), (query.flip(0), blocks, logits)]
    packed = _packed(inputs, k, heads, dim)
    ws = QSACoalesceWorkspace(rows, k, heads, dim, dtype)
    transaction = QSACoalescedExchange(ws, query)
    transaction.exchange(blocks, logits, 0, lambda receive, send: receive.copy_(packed))
    assert transaction.gathered_query is not None
    assert torch.equal(
        transaction.gathered_query.view(torch.int16),
        torch.cat([item[0] for item in inputs], dim=1).view(torch.int16),
    )


@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_host_attention_uses_preexchanged_query_without_second_gather(
    monkeypatch, _cpu_kernels, e4m3, rank
):
    from vllm.distributed import get_dcp_group

    from .test_host_kv_forward import Arms

    arms = Arms(e4m3=e4m3, topk=8, rows_per_pass=2, dcp=(2, rank))
    original_gather = get_dcp_group().all_gather
    calls = []

    def gather(query, dim):
        calls.append(query.clone())
        return original_gather(query, dim)

    monkeypatch.setattr(get_dcp_group(), "all_gather", gather)
    forward = arms.impl.forward_qsa_host

    def preexchanged(*args, **kwargs):
        query = args[2]
        return forward(*args, **kwargs, gathered_query=original_gather(query, dim=1))

    monkeypatch.setattr(arms.impl, "forward_qsa_host", preexchanged)
    try:
        arms.write(0, list(range(32)))
        arms.write(1, list(range(32)))
        # Arms compares the real original GPU-cache forward to the real host
        # forward: output, partial output and LSE must match byte for byte.
        arms.step([(0, 31), (1, 31)])
        assert len(calls) == 1  # GPU reference only; host skips the second AG.
    finally:
        arms.store.close()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "mode,uniform", [("FULL", 8), ("FULL", None), ("PIECEWISE", None)]
)
def test_v2_real_capture_callback_propagates_uniform_plan(
    monkeypatch, enabled, mode, uniform
):
    from contextlib import nullcontext

    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils as cg

    monkeypatch.setenv("REVOLTA_RESEARCH_QSA_DECODE_COALESCE_AG", str(int(enabled)))
    graph_mode = getattr(CUDAGraphMode, mode)
    desc = cg.BatchExecutionDescriptor(graph_mode, 8, 1, uniform)
    metadata: dict[str, Any] | None = None if mode == "PIECEWISE" else {}
    monkeypatch.setattr(
        cg, "prepare_inputs_to_capture", lambda *a, **kw: (metadata, {})
    )
    contexts = []

    def context(*args, **kwargs):
        contexts.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(cg, "set_forward_context", context)

    def capture(self, create_forward_fn, progress_bar_desc):
        callback, state = create_forward_fn(desc)
        callback(CUDAGraphMode.NONE)  # real manager warmup mode
        callback(graph_mode)
        return {desc: state}

    monkeypatch.setattr(cg.CudaGraphManager, "capture", capture)
    manager = cg.ModelCudaGraphManager.__new__(cg.ModelCudaGraphManager)
    manager.dp_size, manager.max_num_reqs = 1, 8
    manager.is_first_pp_rank = manager.is_last_pp_rank = True
    manager.vllm_config = NS()
    manager.hidden_states, manager.aux_hidden_states = None, []
    manager.capture(
        lambda **kw: torch.zeros(8, 16),
        NS(prepare_dummy_inputs=lambda *a: {}),
        NS(input_ids=torch.zeros(8, dtype=torch.int64), positions=torch.arange(8)),
        None,
        None,
        [],
        None,
    )
    context_desc = contexts[-1]["batch_descriptor"]
    if enabled and mode == "FULL":
        assert contexts[0]["batch_descriptor"] == context_desc
    if enabled and mode == "FULL":
        assert context_desc.num_tokens == 8
        assert context_desc.num_reqs == 1
        assert context_desc.uniform == (uniform is not None)
    elif mode == "FULL":
        assert context_desc is None  # disabled V2 callback exactly as before
    else:
        assert not context_desc.uniform
