# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA layer attention with the main K/V on the GPU vs in the host store.

Runs the real ``Qwen4ExpQSAFlashAttentionImpl`` methods on the CPU
interpreter: ``forward_qsa`` (paged GPU cache, DCP1 and the DCP2 branch) and
``forward_qsa_host`` (host history store). CUDA-only wrappers are replaced
by the same Triton kernels: the sparse split/merge kernels through
``SparseAttentionPlan``, the DCP localization kernel, and a fixed stand-in
for the DCP collectives that both arms share. The grouped route (native
SM70 build) reports "not applicable" in both arms.

With the same plan (selection width below one 16-column tile, so one split
whatever the row count; or one gather pass) outputs, DCP partials and LSE
must be identical bits. Sequences are QSA-shaped: whole compressed groups
plus the causal tail of the open group.
"""

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv import LayerScales
from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="The QSA kernels run on the CPU interpreter here",
)

KBS, DIM, QHEADS, REQS = 16, 16, 2, 2
SCALES = LayerScales(0.375, 1.75)


@pytest.fixture(autouse=True)
def _cpu_kernels(monkeypatch):
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    from vllm.models.qwen4_exp.nvidia.ops import qsa as ops
    from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp
    from vllm.triton_utils import tl, triton

    def greater(_builder, lhs, rhs):
        return TensorHandle(np.greater(lhs.data, rhs.data), tl.int1)

    monkeypatch.setattr(InterpreterBuilder, "create_fcmpOGT", greater)

    def sparse(
        q, k, v, idx, table, req, out=None, output_gate=None,
        query_positions=None, sequence_lengths=None, kv_cache_dtype="auto",
        k_scale=1.0, v_scale=1.0, lse=None, **_,
    ):  # fmt: skip
        plan = SparseAttentionPlan(
            rows=q.shape[0],
            query_heads=q.shape[1],
            kv_heads=k.shape[2],
            head_dim=q.shape[2],
            selection_width=idx.shape[1],
            device="cpu",
            e4m3=kv_cache_dtype in ("fp8", "fp8_e4m3"),
            scales=LayerScales(k_scale, v_scale),
            output_dtype=out.dtype,
        )
        gate = None if output_gate is None else output_gate.view_as(q)
        out.copy_(plan.run(q, k, v, idx.contiguous(), table, req, output_gate=gate))
        if lse is not None:
            lse.copy_(plan.lse)
        return out

    def localize(indices, out, *, dcp_world_size, dcp_rank, interleave_size, **_):
        qsa_dcp._localize_indices[(indices.shape[0],)](
            indices, out, indices.stride(0), out.stride(0),
            WIDTH=indices.shape[1], WORLD=dcp_world_size, RANK=dcp_rank,
            INTERLEAVE=interleave_size,
            BLOCK=triton.next_power_of_2(indices.shape[1]),
        )  # fmt: skip
        return out

    group = SimpleNamespace(
        # A fixed "remote rank": the other half of the heads, deterministic.
        all_gather=lambda q, dim: torch.cat([q, (q * 0.5).flip(0)], dim=dim),
        device_communicator=SimpleNamespace(sm70_dcp2=None),
    )
    monkeypatch.setattr(ops, "qsa_sparse_paged_attention", sparse)
    monkeypatch.setattr(
        ops, "qsa_dcp_sparse_paged_attention_sm70_grouped_page4", lambda *a, **k: 0
    )
    monkeypatch.setattr(qsa_dcp, "qsa_localize_dcp_indices", localize)
    monkeypatch.setattr("vllm.distributed.get_dcp_group", lambda: group)


class Arms:
    def __init__(
        self, *, e4m3, topk, rows_per_pass, dcp=(1, 0), seed=5, dim=DIM, heads=QHEADS
    ):
        from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl

        self.world, self.rank = dcp
        self.dim, self.heads = dim, heads
        self.topk, self.width = topk, topk + 3
        row_groups = topk // 4 + 1
        dtype = torch.uint8 if e4m3 else torch.float16
        shape = HostHistoryShape(
            num_blocks=17,
            kernel_blocks_per_block=1,
            kernel_block_tokens=KBS,
            num_layers=1,
            kv_heads=1,
            head_dim=dim,
            dtype=dtype,
            hot_slots=rows_per_pass * row_groups + REQS * 9,
            max_requests=REQS,
            selection_width=self.width,
            max_write_rows=128,
            row_groups=row_groups,
            dcp_world_size=self.world,
            dcp_rank=self.rank,
        )
        self.store = HostHistoryStore(shape, ["attn"], device="cpu")
        self.store.bind_pages([(hid, 17 - hid) for hid in range(1, 17)])
        self.host = self.store.layer("attn")
        self.paged = torch.zeros((17, 2, KBS, 1, dim), dtype=dtype)
        self.kv_dtype = "fp8_e4m3" if e4m3 else "auto"
        impl = object.__new__(Qwen4ExpQSAFlashAttentionImpl)
        impl.kv_cache_dtype = self.kv_dtype
        impl.head_size = dim
        impl.dcp_world_size, impl.dcp_rank = self.world, self.rank
        impl.sliding_window, impl.alibi_slopes, impl.sinks = (-1, -1), None, None
        impl.dcp_combine = lambda out, lse, group, is_lse_base_on_e: out[
            :, : out.shape[1] // 2
        ]
        self.impl = impl
        tokens = 64
        gathered_heads = heads * self.world
        self.layer = SimpleNamespace(
            topk_indices_buffer=torch.full((tokens, self.width), -1, dtype=torch.int32),
            _k_scale_float=SCALES.key,
            _v_scale_float=SCALES.value,
            _k_scale=torch.tensor(SCALES.key),
            _v_scale=torch.tensor(SCALES.value),
            qsa_dcp_sharded=self.world > 1,
            cp_kv_cache_interleave_size=1,
            dcp_local_indices_buffer=torch.empty(
                (tokens, self.width), dtype=torch.int32
            ),
            dcp_partial_output_buffer=torch.empty((tokens, gathered_heads, dim)),
            dcp_partial_lse_buffer=torch.empty((tokens, gathered_heads)),
            indexer=SimpleNamespace(token_topk=topk, compress_ratio=4),
            qsa_draft_native_scales=None,
            sm70_qsa_paged_int8_h6=False,
            sm70_qsa_paged_h12=False,
            sm70_qsa_device_history=False,
        )
        self.blocks = {0: [1, 5, 9], 1: [3, 7, 11]}
        self.length = {0: 0, 1: 0}
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = random.Random(seed)

    def write(self, req, positions):
        key = torch.randn((len(positions), 1, self.dim), generator=self.gen).half()
        value = torch.randn(key.shape, generator=self.gen).half()
        span = KBS * self.world
        slots = []
        for p in positions:
            own = p % self.world == self.rank
            local = p // self.world
            block = self.blocks[req][p // span]
            slots.append(block * KBS + local % KBS if own else -1)
        slots = torch.tensor(slots, dtype=torch.int64)
        self.host.write(
            key, value, slots, self.kv_dtype, self.layer._k_scale, self.layer._v_scale
        )
        k_enc, v_enc = self.store.staging[: len(positions), :, 0].unbind(1)
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                self.paged[slot // KBS, 0, slot % KBS] = k_enc[row]
                self.paged[slot // KBS, 1, slot % KBS] = v_enc[row]
        self.length[req] = max(self.length[req], positions[-1] + 1)

    def selection(self, q):
        tail = (q + 1) // 4 * 4
        groups = list(range(tail // 4))
        picked = self.rng.sample(groups, min(len(groups), self.topk // 4))
        row = [g * 4 + i for g in sorted(picked) for i in range(4)]
        row += list(range(tail, q + 1))
        return row + [-1] * (self.width - len(row))

    def step(self, rows):
        n = len(rows)
        reqs = [r for r, _ in rows]
        table = torch.tensor([self.blocks[0], self.blocks[1]], dtype=torch.int32)
        seq_lens = torch.tensor([self.length[0], self.length[1]], dtype=torch.int32)
        token_to_req = torch.tensor(reqs, dtype=torch.int32)
        self.layer.topk_indices_buffer[:n] = torch.tensor(
            [self.selection(q) for _, q in rows], dtype=torch.int32
        )
        query = (
            torch.randn((n, self.heads, self.dim), generator=self.gen) * 0.5
        ).half()
        gate = torch.randn((n, self.heads * self.dim), generator=self.gen).half()
        meta = SimpleNamespace(
            num_actual_tokens=n,
            block_table=table,
            max_query_len=max(reqs.count(r) for r in range(REQS)),
        )
        out_gpu = torch.empty_like(query)
        self.impl.forward_qsa(
            self.layer, query, None, None, self.paged, meta, out_gpu,
            token_to_req, output_gate=gate, sequence_lengths=seq_lens,
            query_positions=torch.tensor([p for _, p in rows], dtype=torch.int64),
        )  # fmt: skip
        partial_gpu = self.layer.dcp_partial_output_buffer[:n].clone()
        lse_gpu = self.layer.dcp_partial_lse_buffer[:n].clone()
        out_host = torch.empty_like(out_gpu)
        self.impl.forward_qsa_host(
            self.layer, self.host, query, meta, out_host, token_to_req,
            seq_lens, output_gate=gate,
        )  # fmt: skip
        assert torch.equal(out_gpu.view(torch.uint8), out_host.view(torch.uint8))
        assert out_host.abs().sum() > 0
        if self.world > 1:
            partial = self.layer.dcp_partial_output_buffer[:n]
            lse = self.layer.dcp_partial_lse_buffer[:n]
            assert torch.equal(partial_gpu.view(torch.uint8), partial.view(torch.uint8))
            assert torch.equal(lse_gpu.view(torch.uint8), lse.view(torch.uint8))
        return out_gpu


def _trace(arms):
    # Prefill request 0 in two chunks, request 1 in one, then decode both.
    arms.write(0, list(range(0, 20)))
    arms.step([(0, q) for q in range(20)])
    arms.write(0, list(range(20, 33)))
    arms.write(1, list(range(0, 25)))
    arms.step([(0, q) for q in range(20, 33)] + [(1, q) for q in range(25)])
    for _ in range(4):
        for req in (0, 1):
            arms.write(req, [arms.length[req]])
        arms.step([(0, arms.length[0] - 1), (1, arms.length[1] - 1)])


@pytest.mark.parametrize("e4m3", [True, False])
@pytest.mark.parametrize("dcp", [(1, 0), (2, 0), (2, 1)])
def test_many_passes_one_split_match_gpu_forward(e4m3, dcp):
    # top-k 8: 11 columns, one tile, one split; two rows per gather pass.
    arms = Arms(e4m3=e4m3, topk=8, rows_per_pass=2, dcp=dcp)
    _trace(arms)


@pytest.mark.parametrize("dcp", [(1, 0), (2, 1)])
def test_one_pass_split_k_matches_gpu_forward(dcp):
    # top-k 32: 35 columns, three tiles, split-K + merge; one pass per step.
    arms = Arms(e4m3=True, topk=32, rows_per_pass=38, dcp=dcp)
    _trace(arms)


@pytest.fixture
def _int1_compares(monkeypatch):
    # Some Triton interpreters type comparison results like their operands,
    # so masks built from them fail in masked loads; compares yield int1.
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    from vllm.triton_utils import tl

    for name in dir(InterpreterBuilder):
        if name.startswith(("create_icmp", "create_fcmp")):
            op = getattr(InterpreterBuilder, name)

            def compare(builder, lhs, rhs, op=op):
                return TensorHandle(op(builder, lhs, rhs).data, tl.int1)

            monkeypatch.setattr(InterpreterBuilder, name, compare)


class PlanarInt8Arms(Arms):
    """Planar INT8 pages in both arms; DCP2 uniform verify (two rows/request)."""

    def __init__(self, *, h12, **kwargs):
        import dataclasses

        from vllm.v1.attention.kv_cache_view import KVCacheView, bind_planar_kv_cache
        from vllm.v1.attention.kv_codecs import INT8

        super().__init__(e4m3=False, **kwargs)
        shape = dataclasses.replace(
            self.store.shape,
            dtype=torch.int8,
            layout_cache_dtype="int8",
            decode_rows=2,
        )
        self.store = HostHistoryStore(shape, ["attn"], device="cpu")
        self.store.bind_pages([(hid, 17 - hid) for hid in range(1, 17)])
        self.host = self.store.layer("attn")
        raw = torch.zeros(17 * shape.page_bytes, dtype=torch.int8)
        self.paged = bind_planar_kv_cache(
            raw,
            num_blocks=17,
            scheduler_block_size=KBS,
            kernel_block_size=KBS,
            num_heads=1,
            head_size=self.dim,
            page_size_bytes=shape.page_bytes,
            dtype=torch.int8,
        )
        self.view = KVCacheView.bind(self.paged, INT8, self.dim)
        self.kv_dtype = self.impl.kv_cache_dtype = "int8"
        self.layer.sm70_qsa_paged_h12 = h12

    def write(self, req, positions):
        key = torch.randn((len(positions), 1, self.dim), generator=self.gen).half()
        value = torch.randn(key.shape, generator=self.gen).half()
        span = KBS * self.world
        slots = torch.tensor(
            [
                self.blocks[req][p // span] * KBS + p // self.world % KBS
                if p % self.world == self.rank
                else -1
                for p in positions
            ],
            dtype=torch.int64,
        )
        self.host.write(
            key, value, slots, "int8", self.layer._k_scale, self.layer._v_scale
        )
        # The GPU arm holds the host writer's encoded bytes and scales.
        stage = self.host.staging_cache
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                for name in ("key", "value", "key_scale", "value_scale"):
                    page = getattr(self.view, name)
                    page[slot // KBS, slot % KBS] = getattr(stage, name)[row, 0]
        self.length[req] = max(self.length[req], positions[-1] + 1)


@pytest.mark.usefixtures("_int1_compares")
def test_planar_int8_h6_reads_the_hot_pool(monkeypatch):
    from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.v1.worker import workspace

    from .test_qsa_paged_fp16_h12_cpu import _forward_config, h12_late_scale

    arms = PlanarInt8Arms(h12=False, topk=8, rows_per_pass=4, dim=256, heads=6)
    arms.layer.sm70_qsa_paged_int8_h6 = True
    arms.write(0, list(range(33)))
    arms.write(1, [0])
    calls = []

    def h6(q, k, v, sk, sv, ids, table, req, pos, lens, out, gate, *scratch):
        hot = pos is None
        assert hot == (lens is None)
        cache = arms.host.hot_cache if hot else arms.view
        for actual, name in zip(
            (k, v, sk, sv), ("key", "value", "key_scale", "value_scale")
        ):
            expected = getattr(cache, name)
            assert actual.data_ptr() == expected.data_ptr()
            assert actual.stride() == expected.stride()
        if hot:
            assert table is arms.store.hot_block_table
        calls.append(hot)
        token = ids.long().clamp_min(0)
        page = table[req.long()[:, None], token // k.size(1)].long()
        offset = token % k.size(1)
        codes_k, codes_v, scales_k, scales_v = (
            plane[page, offset, 0].float() for plane in (k, v, sk, sv)
        )
        valid = ids >= 0
        if not hot:
            valid &= (ids <= pos[:, None]) & (ids < lens[req.long(), None])
        result, _ = h12_late_scale(q, codes_k, codes_v, scales_k, scales_v, valid)
        out.copy_(result.half().float() * gate.float().sigmoid())

    monkeypatch.setattr(
        workspace, "_manager", workspace.WorkspaceManager(torch.device("cpu"))
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.ops.vllm_sm70_qsa_indexer, "paged_int8_h6", h6, raising=False
    )
    with set_forward_context(None, _forward_config()), sm70_decode_graph_compilation():
        get_forward_context().additional_kwargs["sm70_qsa_draft_decode"] = True
        arms.step([(0, 32), (1, 0)])
    assert calls == [False, True]


@pytest.mark.usefixtures("_int1_compares")
@pytest.mark.parametrize("rank", [0, 1])
def test_planar_int8_h12_reads_the_hot_pool(monkeypatch, rank):
    """Uniform decode/verify on host main K/V: the INT8 H12 reader gets the
    hot pool views, its page table and the remapped IDs; the result equals
    the GPU arm's H12 bit for bit and the Triton reader within rounding."""
    from vllm.forward_context import set_forward_context
    from vllm.models.qwen4_exp.nvidia import qsa
    from vllm.models.qwen4_exp.nvidia.ops import qsa as ops
    from vllm.models.qwen4_exp.nvidia.ops import qsa_int8
    from vllm.v1.attention.kv_codecs import INT8

    from .test_qsa_paged_fp16_h12_cpu import _forward_config, h12_late_scale

    h12_reads: list[tuple] = []
    triton_rows: list[int] = []

    def h12(q, k, v, ids, table, req, out, lse, *, k_scale_cache, v_scale_cache):
        # Python model of the CUDA kernel: page-table lookup, then late scales.
        valid = ids >= 0
        pos = ids.long().clamp_min(0)
        page = table[req.long()[:, None], pos // k.shape[1]].long()
        codes_k, codes_v, scales_k, scales_v = (
            plane[page, pos % k.shape[1], 0].float()
            for plane in (k, v, k_scale_cache, v_scale_cache)
        )
        planes = (codes_k, codes_v, scales_k, scales_v)
        h12_reads.append((k, table, ids, valid, planes))
        result, result_lse = h12_late_scale(
            q, codes_k, codes_v, scales_k, scales_v, valid
        )
        out.copy_(result)
        lse.copy_(result_lse)

    def triton(q, k, v, idx, table, req, out, *, k_scale_cache, v_scale_cache,
               lse=None, output_gate=None, kv_cache_dtype, **_):  # fmt: skip
        assert kv_cache_dtype == "int8" and output_gate is None
        triton_rows.append(q.shape[0])
        return qsa_int8.triton_attention(
            q, k, v, idx, table, req, out, k_scale_cache, v_scale_cache,
            lse=lse, codec=INT8,
        )  # fmt: skip

    monkeypatch.setattr(qsa, "qsa_paged_h12", h12)
    monkeypatch.setattr(ops, "qsa_sparse_paged_attention", triton)

    def run(arms, rows):
        with set_forward_context(None, _forward_config(), is_uniform_decode=True):
            out = arms.step(rows)
        layer, n = arms.layer, len(rows)
        partial = layer.dcp_partial_output_buffer[:n].clone()
        return out, partial, layer.dcp_partial_lse_buffer[:n].clone()

    results = {}
    for native in (True, False):
        arms = PlanarInt8Arms(h12=native, topk=8, rows_per_pass=4, dcp=(2, rank))
        h12_reads.clear()
        triton_rows.clear()
        # Request 1's first token lives on rank 0: rank 1 reads an empty shard.
        arms.write(0, list(range(33)))
        arms.write(1, [0])
        steps = [run(arms, [(0, 32), (1, 0)])]
        for _ in range(2):
            for req in (0, 1):
                arms.write(req, [arms.length[req]])
            steps.append(run(arms, [(r, arms.length[r] - 1) for r in (0, 1)]))
        for req in (0, 1):
            arms.write(req, [arms.length[req], arms.length[req] + 1])
        steps.append(
            run(arms, [(r, arms.length[r] - d) for r in (0, 1) for d in (2, 1)])
        )
        results[native] = steps
        if not native:
            assert not h12_reads and triton_rows == [2] * 6 + [4] * 2
            continue
        # GPU arm then host arm per step; the host arm reads the hot pool.
        assert not triton_rows and len(h12_reads) == 8
        width = 6  # three groups of two rank-local positions
        for gpu, host in zip(h12_reads[::2], h12_reads[1::2]):
            assert gpu[0].data_ptr() == arms.view.key.data_ptr()
            assert host[0].data_ptr() == arms.host.key_cache.data_ptr()
            assert host[1] is arms.store.hot_block_table
            assert host[2].data_ptr() == arms.store.indices.data_ptr()
            assert gpu[2].shape[1] == host[2].shape[1] == width
            valid = gpu[3]
            assert torch.equal(valid, host[3])
            for a, b in zip(gpu[4], host[4]):
                assert torch.equal(a[valid], b[valid])
        if rank == 1:
            _, partial, lse = steps[0]
            assert torch.isneginf(lse[1]).all() and not partial[1].any()

    for (out, partial, lse), (t_out, t_partial, t_lse) in zip(
        results[True], results[False]
    ):
        # Triton rounds the PV weights to two FP16 terms and splits by tiles.
        torch.testing.assert_close(partial, t_partial, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(lse, t_lse, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out, t_out, rtol=2e-3, atol=2e-3)
