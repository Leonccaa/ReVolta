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
        all_gather=lambda q, dim: torch.cat([q, (q * 0.5).flip(0)], dim=dim)
    )
    monkeypatch.setattr(ops, "qsa_sparse_paged_attention", sparse)
    monkeypatch.setattr(
        ops, "qsa_dcp_sparse_paged_attention_sm70_grouped_page4", lambda *a, **k: 0
    )
    monkeypatch.setattr(qsa_dcp, "qsa_localize_dcp_indices", localize)
    monkeypatch.setattr("vllm.distributed.get_dcp_group", lambda: group)


class Arms:
    def __init__(self, *, e4m3, topk, rows_per_pass, dcp=(1, 0), seed=5):
        from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl

        self.world, self.rank = dcp
        self.topk, self.width = topk, topk + 3
        row_groups = topk // 4 + 1
        dtype = torch.uint8 if e4m3 else torch.float16
        shape = HostHistoryShape(
            num_blocks=17,
            kernel_blocks_per_block=1,
            kernel_block_tokens=KBS,
            num_layers=1,
            kv_heads=1,
            head_dim=DIM,
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
        self.paged = torch.zeros((17, 2, KBS, 1, DIM), dtype=dtype)
        self.kv_dtype = "fp8_e4m3" if e4m3 else "auto"
        impl = object.__new__(Qwen4ExpQSAFlashAttentionImpl)
        impl.kv_cache_dtype = self.kv_dtype
        impl.head_size = DIM
        impl.dcp_world_size, impl.dcp_rank = self.world, self.rank
        impl.sliding_window, impl.alibi_slopes, impl.sinks = (-1, -1), None, None
        impl.dcp_combine = lambda out, lse, group, is_lse_base_on_e: out[
            :, : out.shape[1] // 2
        ]
        self.impl = impl
        tokens = 64
        heads = QHEADS * self.world
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
            dcp_partial_output_buffer=torch.empty((tokens, heads, DIM)),
            dcp_partial_lse_buffer=torch.empty((tokens, heads)),
            indexer=SimpleNamespace(token_topk=topk, compress_ratio=4),
        )
        self.blocks = {0: [1, 5, 9], 1: [3, 7, 11]}
        self.length = {0: 0, 1: 0}
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = random.Random(seed)

    def write(self, req, positions):
        key = torch.randn((len(positions), 1, DIM), generator=self.gen).half()
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
        query = (torch.randn((n, QHEADS, DIM), generator=self.gen) * 0.5).half()
        gate = torch.randn((n, QHEADS * DIM), generator=self.gen).half()
        meta = SimpleNamespace(
            num_actual_tokens=n,
            block_table=table,
            max_query_len=max(reqs.count(r) for r in range(REQS)),
        )
        out_gpu = torch.empty((n, QHEADS, DIM), dtype=torch.float16)
        self.impl.forward_qsa(
            self.layer, query, None, None, self.paged, meta, out_gpu,
            token_to_req, output_gate=gate, sequence_lengths=seq_lens,
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
