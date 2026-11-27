# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.distributed
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.models.qwen4_exp.nvidia.ops import sm70_dcp2 as dcp2
from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_output_gate
from vllm.v1.attention.ops.dcp_alltoall import (
    _dcp_a2a_pack_send,
    _dcp_a2a_unpack_combine,
    _lse_weighted_combine,
)

from .test_host_kv_forward import Arms
from .test_host_kv_forward import _cpu_kernels as _cpu_kernels  # noqa: F401
from .test_host_kv_union import UnionArms

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("case", ["pex", "mesh"])
def test_dcp2_startup_reuses_tp_decision(monkeypatch, case):
    device = torch.device("cpu")
    hc = None if case == "pex" else SimpleNamespace(rank=3, device=device)
    tp = SimpleNamespace(device_communicator=SimpleNamespace(sm70_hc_ll=hc))
    group = object()
    allocate = Mock(return_value=[128, 256])
    close_peer, free_local = Mock(), Mock()
    monkeypatch.setattr(CustomAllreduce, "create_shared_buffer", allocate)
    monkeypatch.setattr(CustomAllreduce, "free_shared_buffer", free_local)
    monkeypatch.setattr(dcp2.ops, "sm70_ring_close_mem_handle", close_peer)
    monkeypatch.setattr(torch.accelerator, "device_index", lambda _: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    barrier = Mock()
    monkeypatch.setattr(dcp2.dist, "barrier", barrier)
    runtime = dcp2.Sm70Dcp2Runtime.create(tp, group, [10, 11], device, 1)
    if case == "pex":
        assert runtime is None
        allocate.assert_not_called()
        barrier.assert_not_called()
        return
    assert runtime is not None and runtime.rank == 1
    allocate.assert_called_once_with(dcp2.Q_BYTES + dcp2.O_BYTES, group=group)
    barrier.assert_called_once_with(group=group)
    runtime.close()
    close_peer.assert_called_once_with(128)
    free_local.assert_called_once_with([128, 256], rank=1)


@pytest.mark.parametrize(
    "rows,has_runtime,qualifies",
    [
        (0, True, False),
        (8, False, False),
        (5, True, True),
        (8, True, True),
        (16, True, True),
        (24, True, True),
        (40, True, True),
        (64, True, True),
        (65, True, False),
    ],
)
def test_dcp2_dispatch_rows(rows, has_runtime, qualifies):
    query = torch.empty((rows, 6, 256), dtype=torch.float16)
    runtime = SimpleNamespace(device=query.device) if has_runtime else None
    group = SimpleNamespace(device_communicator=SimpleNamespace(sm70_dcp2=runtime))
    partial = torch.empty((rows, 12, 256))
    lse = torch.empty((rows, 12))
    assert dcp2.get_sm70_dcp2_for_batch(group, query, partial, lse, query, query) is (
        runtime if qualifies else None
    )


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="CPU Triton")
@pytest.mark.parametrize("method,rank", [("dflash", 0), ("mtp", 1)])
def test_dcp2_qsa_gpu_and_host_entrypoints(monkeypatch, method, rank):
    width = 8 if method == "dflash" else 5
    arm_type = Arms if method == "dflash" else UnionArms
    arms = arm_type(e4m3=True, topk=8, rows_per_pass=2, dcp=(2, rank), dim=256, heads=6)
    arms.write(0, list(range(40)))
    arms.write(1, list(range(4)))
    rows = [(0, pos) for pos in range(40 - width, 40)]
    if method == "dflash":
        # Force multiple real Host KV passes with the same selected rows.
        arms.store.step_passes = [(start, start + 2) for start in range(0, width, 2)]
    group = vllm.distributed.get_dcp_group()
    group.device_communicator = SimpleNamespace(sm70_dcp2=None)

    def rank_order(query, dim):
        peer = (query * 0.5).flip(0)
        return torch.cat((query, peer) if rank == 0 else (peer, query), dim=dim)

    old_gather = Mock(side_effect=rank_order)
    group.all_gather = old_gather
    events = []
    runtime = dcp2.Sm70Dcp2Runtime(torch.device("cpu"), rank, [128, 256])
    native = torch.ops.vllm_sm70_qsa_device
    heads = slice(rank * 6, (rank + 1) * 6)

    def gather(query, out, positions, lengths, lp, ll, mine, peer, seq, who, stride):
        events.append("Q")
        assert who == rank and (mine, peer) == (
            runtime.pointers[rank],
            runtime.pointers[1 - rank],
        )
        assert positions.dtype == torch.int64 and lengths.tolist() == [0]
        assert seq is runtime.q_seq and stride == 1
        # Emulate the native local and peer writes at DCP head offsets.
        out[:, heads].copy_(query)
        out[:, (1 - rank) * 6 : (2 - rank) * 6].copy_((query * 0.5).flip(0))

    def combine(partial, lse, gate, out, mine, peer, seq, who):
        events.append("O")
        assert who == rank and seq is runtime.o_seq
        assert (mine, peer) == tuple(
            runtime.pointers[i] + dcp2.Q_BYTES for i in (rank, 1 - rank)
        )
        assert partial.shape == (width, 12, 256) and partial.dtype == torch.float32
        assert lse.isfinite().all()
        out.copy_(reference(partial, lse, None, False))
        _qsa_output_gate(out, gate)

    def reference(partial, lse, group, is_lse_base_on_e):
        assert not is_lse_base_on_e
        # Synthetic peer shard with different values and LSE. Rank 1 still
        # owns the second head slice, regardless of which shard is local.
        return _lse_weighted_combine(
            torch.stack([partial[:, heads], partial[:, heads] * 0.5]),
            torch.stack([lse[:, heads], lse[:, heads] + 1]),
            is_lse_base_on_e=False,
        )

    old_combine = Mock(side_effect=reference)
    arms.impl.dcp_combine = old_combine
    extension = "vllm._sm70_qsa_device_C"
    monkeypatch.setitem(sys.modules, extension, ModuleType(extension))
    monkeypatch.setattr(native, "dcp2_q_prep", gather, raising=False)
    monkeypatch.setattr(native, "dcp2_combine_gate", combine, raising=False)
    rng, gen = arms.rng.getstate(), arms.gen.get_state()
    old = arms.step(rows)
    assert old_gather.call_count == old_combine.call_count == 2
    assert events == []  # CPU emission proof for the rejected/legacy owner.
    group.device_communicator.sm70_dcp2 = runtime
    arms.rng.setstate(rng)
    arms.gen.set_state(gen)
    new = arms.step(rows)
    assert old_gather.call_count == old_combine.call_count == 2
    assert events == ["Q", "O", "Q", "O"]  # One pair for GPU, one for all host passes.
    assert torch.equal(old.view(torch.int16), new.view(torch.int16))
    if method == "mtp":
        assert arms.store.union_stats["calls"] == 2


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="CPU Triton")
def test_base2_merge_empty_shards_and_half_rounding():
    # Unequal LSEs (weights 1/3 and 2/3), rank 0 empty, rank 1 empty,
    # and both empty. Include an FP16 rounding boundary before the gate.
    lse = torch.tensor(
        [[0.0, 1.0], [-torch.inf, 0.0], [0.0, -torch.inf], [-torch.inf, -torch.inf]]
    )
    partial = torch.tensor([[0.125, 0.851], [0.0, 1.25], [1.75, 0.0], [0.0, 0.0]])
    sent = torch.empty((2, 2, 4, 1, 2))  # [source, destination, row, head, D+1]
    for rank in range(2):
        rank_partial = torch.stack([partial[:, rank], -partial[:, rank]], dim=1)
        rank_lse = lse[:, rank, None].expand(-1, 2).contiguous()
        _dcp_a2a_pack_send(rank_partial[:, :, None], rank_lse, sent[rank], 2, 1, 1, 1)
    received = sent[:, 0].contiguous()  # A2A routes the first head to rank 0.
    merged = _dcp_a2a_unpack_combine(received, 1, 1, False, False)
    expected = torch.tensor([(0.125 + 2 * 0.851) / 3, 1.25, 1.75, 0.0])
    torch.testing.assert_close(merged.flatten(), expected, rtol=1e-6, atol=1e-7)
    gate = torch.full((4, 1, 1), 0.5, dtype=torch.float16)
    output = torch.empty_like(gate)
    _qsa_output_gate(output, gate, merged=merged)
    baseline = merged.half()
    _qsa_output_gate(baseline, gate)
    assert torch.equal(output, baseline)
    weights = gate.flatten().float().sigmoid()
    expected_gate = (expected.half().float() * weights).half()
    assert torch.equal(output.flatten(), expected_gate)
    assert output.flatten()[0] != (expected * weights).half()[0]
