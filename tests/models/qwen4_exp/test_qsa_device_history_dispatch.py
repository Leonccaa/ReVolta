# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.distributed
from vllm.config import KernelConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.models.qwen4_exp.nvidia.ops import qsa, qsa_dcp
from vllm.models.qwen4_exp.nvidia.ops.qsa_device_history import (
    load_qsa_device_history,
    qsa_device_history_workspace,
)
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl
from vllm.triton_utils import triton
from vllm.v1.attention.kv_cache_view import KVCacheView, bind_attention_cache
from vllm.v1.attention.kv_codecs import FP8_E4M3
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
)

pytestmark = pytest.mark.skip_global_cleanup
TOPK, RATIO, WIDTH, PAGE = 2048, 4, 2051, 832
K_SCALE, V_SCALE = 0.731, 0.413
BLOCKS = 6


def make_inputs(rows, q_len, rank, device, page=PAGE, interleave=1):
    rng = torch.Generator().manual_seed(603)
    q = torch.randn((rows, 12, 256), generator=rng).half().to(device)
    order = torch.randperm(BLOCKS - 1, generator=rng, dtype=torch.int32) + 1
    # The fourth page is deliberately invalid in both directions. The
    # fifth valid page exercises a real lookup after the invalid entries.
    table = torch.stack([order.roll(req) for req in range(rows // q_len)])
    table = torch.cat(
        (
            table[:, :3],
            torch.full_like(table[:, :1], -1),
            torch.full_like(table[:, :1], BLOCKS),
            table[:, 3:],
        ),
        dim=1,
    )
    requests = torch.arange(rows, dtype=torch.int32) // q_len
    requests[-2:] = torch.tensor([-1, table.shape[0]], dtype=torch.int32)
    selection = torch.full((rows, WIDTH), -1, dtype=torch.int32)
    for row in range(rows - 3):
        groups = torch.randperm(3 * 2 * page // RATIO, generator=rng)[: TOPK // RATIO]
        ids = (groups[:, None] * RATIO + torch.arange(RATIO)).flatten()
        selection[row, : ids.numel()] = ids.to(torch.int32)
    # Page edge, a negative page entry, an invalid physical page, a later
    # valid page and an out-of-table ID.
    selection[0, :12] = torch.tensor(
        [
            2 * page - 2,
            2 * page - 1,
            2 * page,
            2 * page + 1,
            6 * page,
            6 * page + 1,
            8 * page,
            8 * page + 1,
            10 * page,
            10 * page + 1,
            14 * page,
            14 * page + 1,
        ]
    )
    selection[1].fill_(-1)
    selection[1, :interleave] = torch.arange(interleave)
    selection[-2:, 0] = 0  # Check request masking with a non-PAD selected ID.

    # Same padded/packed view constructor as MRv2. Use the second member so
    # both planes have a nonzero storage offset and share the raw allocation.
    raw = torch.empty(BLOCKS * 2 * page * 512, dtype=torch.uint8, device=device)
    cache = bind_attention_cache(
        raw,
        shape=(BLOCKS, 2, page, 1, 256),
        dtype=torch.uint8,
        page_size_bytes=page * 512,
        scheduler_block_size=PAGE,
        kernel_block_size=page,
        member_index=1,
        member_count=2,
    )
    kv_rng = torch.Generator().manual_seed(764 + rank)
    codes = torch.randn(cache.shape, generator=kv_rng).to(torch.float8_e4m3fn)
    cache.copy_(codes.view(torch.uint8).to(device))
    return q, cache, selection.to(device), table.to(device), requests.to(device)


def reference(q, cache, selection, table, requests, *, rank, interleave=1):
    k, v = (plane.view(torch.float8_e4m3fn).float() for plane in cache.unbind(1))
    out = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full_like(q[:, :, 0], -torch.inf, dtype=torch.float32)
    page = cache.shape[2]
    for row, (ids, req) in enumerate(zip(selection.cpu(), requests.tolist())):
        if not 0 <= req < table.shape[0]:
            continue
        ids = ids[(ids >= 0) & ((ids // interleave) % 2 == rank)]
        local = ids // (2 * interleave) * interleave + ids % interleave
        local = local[local // page < table.shape[1]].long().to(q.device)
        physical = table[req, local // page].long()
        valid = (physical >= 0) & (physical < cache.shape[0])
        if not valid.any():
            continue
        physical, off = physical[valid], local[valid] % page
        logits = q[row].float() @ k[physical, off, 0].T * (K_SCALE / 16)
        out[row] = logits.softmax(-1) @ v[physical, off, 0] * V_SCALE
        lse[row] = torch.logsumexp(logits, dim=-1) / math.log(2)
    return out, lse


def assert_close(actual, expected):
    out, lse = actual
    ref_out, ref_lse = expected
    torch.testing.assert_close(out, ref_out, rtol=3e-3, atol=2e-3)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-4, atol=2e-3)
    empty = torch.isneginf(ref_lse)
    assert torch.equal(out[empty], torch.zeros_like(out[empty]))
    assert not torch.signbit(out[empty]).any()


def merge_ranks(results):
    out, lse = (torch.stack(tensors) for tensors in zip(*results))
    merged_lse = torch.logsumexp(lse * math.log(2), dim=0) / math.log(2)
    weights = torch.where(torch.isfinite(lse), torch.exp2(lse - merged_lse), 0.0)
    return (out * weights[..., None]).sum(dim=0), merged_lse


def _rank_forward(inputs, rank, monkeypatch, interleave):
    q, cache, selection, table, requests = inputs
    query = q[:, :6]
    gate = torch.zeros_like(query)
    output = torch.empty_like(query)
    layer = SimpleNamespace(
        qsa_dcp_sharded=True,
        topk_indices_buffer=selection,
        dcp_local_indices_buffer=torch.empty_like(selection),
        dcp_partial_output_buffer=torch.empty_like(q, dtype=torch.float32),
        dcp_partial_lse_buffer=torch.empty_like(q[:, :, 0], dtype=torch.float32),
        cp_kv_cache_interleave_size=interleave,
        indexer=SimpleNamespace(token_topk=TOPK, compress_ratio=RATIO),
        _k_scale_float=K_SCALE,
        _v_scale_float=V_SCALE,
        sm70_qsa_paged_fp16_h12=False,
    )
    impl = Qwen4ExpQSAFlashAttentionImpl.__new__(Qwen4ExpQSAFlashAttentionImpl)
    impl.kv_cache_dtype, impl.head_size = "fp8_e4m3", 256
    impl.alibi_slopes = impl.sinks = None
    impl.sliding_window = (-1, -1)
    impl.dcp_world_size, impl.dcp_rank = 2, rank
    impl.dcp_combine = lambda out, lse, group, is_lse_base_on_e: out[:, :6]
    group = SimpleNamespace(
        all_gather=lambda query, dim: q,
        device_communicator=SimpleNamespace(sm70_dcp2=None),
    )
    monkeypatch.setattr(vllm.distributed, "get_dcp_group", lambda: group)
    metadata = SimpleNamespace(num_actual_tokens=q.shape[0], block_table=table)
    cfg = VllmConfig()

    def run(enabled, uniform=True):
        layer.sm70_qsa_device_history = enabled
        with set_forward_context({}, cfg, is_uniform_decode=uniform):
            impl.forward_qsa(
                layer, query, query, query, cache, metadata, output, requests, gate
            )
        return layer.dcp_partial_output_buffer, layer.dcp_partial_lse_buffer

    return run


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="CPU interpreter")
@pytest.mark.parametrize(
    ("query_len", "reqs", "phase", "enabled", "hits"),
    [
        (8, 1, "decode", True, True),
        (8, 2, "decode", True, True),
        (4, 4, "decode", True, True),
        (8, 3, "decode", True, False),
        (5, 8, "decode", True, False),
        (5, 4, "decode", True, True),
        (5, 1, "decode", True, True),
        (8, 1, "prefill", True, False),
        (8, 2, "mixed", True, False),
        (8, 1, "decode", False, False),
    ],
)
def test_mrv2_phase_and_native_dispatch(
    query_len, reqs, phase, enabled, hits, monkeypatch
):
    assert not KernelConfig().sm70_qsa_device_history
    device = torch.device("cpu")
    rows = query_len * reqs
    runner = GPUModelRunner.__new__(GPUModelRunner)
    req_ids = list(range(reqs))
    computed = np.full(reqs, 1000)
    if phase != "decode":
        computed[0] -= query_len
    runner.req_states = SimpleNamespace(
        req_id_to_index=dict(zip(req_ids, req_ids)),
        num_computed_prefill_tokens=computed,
        prefill_len=SimpleNamespace(np=np.full(reqs, 1000)),
    )
    step = SimpleNamespace(
        num_scheduled_tokens=dict.fromkeys(req_ids, query_len),
        total_num_scheduled_tokens=rows,
    )
    uniform = runner._get_uniform_decode_token_count(step, False) is not None
    inputs = make_inputs(rows, query_len, rank=1, device=device)
    q, cache, selection, table, requests = inputs
    expected_view = KVCacheView.bind(cache, FP8_E4M3, 256)
    dispatches = []

    def localize(indices, out, *, dcp_world_size, dcp_rank, interleave_size, **kwargs):
        width = indices.shape[1]
        qsa_dcp._localize_indices[(rows,)](
            indices,
            out,
            indices.stride(0),
            out.stride(0),
            WIDTH=width,
            WORLD=dcp_world_size,
            RANK=dcp_rank,
            INTERLEAVE=interleave_size,
            BLOCK=triton.next_power_of_2(width),
        )
        return out

    def check_selection(name, query, k, v, local, block_table, token_to_req):
        dispatches.append(name)
        assert query.data_ptr() == q.data_ptr() and block_table is table
        assert token_to_req.data_ptr() == requests.data_ptr()
        for loaded, expected in ((k, expected_view.key), (v, expected_view.value)):
            assert loaded.data_ptr() == expected.data_ptr()
            assert loaded.stride(0) == 2 * PAGE * 512
        assert local.shape == (rows, 1026)
        for src, got in zip(selection, local):
            owned = src[(src >= 0) & (src % 2 == 1)]
            assert torch.equal(got[: owned.numel()], owned // 2)
            assert (got[owned.numel() :] == -1).all()

    def native(query, k, v, local, table, reqs, ks, vs, out, lse, partial, state):
        check_selection("native", query, k, v, local, table, reqs)
        assert (ks, vs) == (K_SCALE, V_SCALE)
        assert partial.shape == (rows * 6 * 17, 256)
        assert state.shape == (rows * 6 * 17, 2)
        out.zero_()
        lse.fill_(-torch.inf)

    def fallback(query, k, v, local, table, reqs, out, **kwargs):
        check_selection("triton", query, k, v, local, table, reqs)
        assert (kwargs["k_scale"], kwargs["v_scale"]) == (K_SCALE, V_SCALE)
        out.zero_()
        kwargs["lse"].fill_(-torch.inf)

    monkeypatch.setattr(qsa_dcp, "qsa_localize_dcp_indices", localize)
    monkeypatch.setattr(qsa, "_qsa_output_gate", lambda *_: None)
    monkeypatch.setattr(qsa, "qsa_sparse_paged_attention", fallback)
    monkeypatch.setattr(
        torch.ops.vllm_sm70_qsa_device, "run_paged_dcp", native, raising=False
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    init_workspace_manager(device)
    try:
        qsa_device_history_workspace(20, 1026)
        lock_workspace()
        _rank_forward(inputs, 1, monkeypatch, 1)(enabled, uniform)
    finally:
        reset_workspace_manager()
    assert dispatches == (["native"] if hits else ["triton"])


def _require_cuda_reader():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("native QSA history requires SM70")
    load_qsa_device_history()


@pytest.mark.parametrize(
    ("rows", "q_len", "page", "interleave"),
    [(8, 8, PAGE, 1), (16, 4, PAGE, 1), (40, 5, PAGE, 1), (20, 5, 64, 4)],
)
def test_sm70_device_history_dcp_ranks(rows, q_len, page, interleave, monkeypatch):
    _require_cuda_reader()
    device = torch.device("cuda")
    init_workspace_manager(device)
    qsa_device_history_workspace(20, WIDTH)
    lock_workspace()
    results, refs = [], []
    try:
        for rank in range(2):
            inputs = make_inputs(rows, q_len, rank, device, page, interleave)
            selection = inputs[2].clone()
            run = _rank_forward(inputs, rank, monkeypatch, interleave)
            expected = reference(*inputs, rank=rank, interleave=interleave)
            baseline = tuple(t.clone() for t in run(False))
            assert_close(baseline, expected)
            candidate = run(True)
            assert_close(candidate, expected)
            if rows > 20:
                assert all(torch.equal(a, b) for a, b in zip(baseline, candidate))
            assert torch.equal(inputs[2], selection)
            results.append(candidate)
            refs.append(expected)
        assert_close(merge_ranks(results), merge_ranks(refs))
    finally:
        reset_workspace_manager()


def test_sm70_device_history_graph_replay(monkeypatch):
    _require_cuda_reader()
    device = torch.device("cuda")
    init_workspace_manager(device)
    qsa_device_history_workspace(20, 1026)
    lock_workspace()
    try:
        inputs = make_inputs(8, 8, 0, device)
        run = _rank_forward(inputs, 0, monkeypatch, 1)
        run(True)  # Compile the surrounding Triton kernels before capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run(True)
        for _ in range(2):
            inputs[0].mul_(0.5)
            inputs[2][0].fill_(-1)
            graph.replay()
            assert_close(result, reference(*inputs, rank=0))
    finally:
        reset_workspace_manager()
