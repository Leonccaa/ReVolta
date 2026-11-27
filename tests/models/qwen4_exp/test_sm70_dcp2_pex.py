# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.config
import vllm.distributed
from vllm.config import CUDAGraphMode
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.models.qwen4_exp.nvidia.ops import sm70_dcp2
from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_output_gate
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops import dcp_alltoall

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "switch,mode,backend,extra",
    [
        ("0", "pex_p2p", "a2a", 0),
        ("1", "pex_p2p", "a2a", 200),
        ("1", "pex_p2p", "ag_rs", 0),
        ("1", "nvlink", "a2a", 0),
    ],
)
def test_pex_extends_existing_push_allocation(
    monkeypatch, switch, mode, backend, extra
):
    monkeypatch.setenv("VLLM_SM70_DCP2_P2P", switch)
    monkeypatch.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE", "1")
    monkeypatch.setattr(car, "custom_ar", True)
    monkeypatch.setattr(car, "in_the_same_node_as", lambda *a, **kw: [True] * 4)
    monkeypatch.setattr(
        car,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_cuda_alike=lambda: True,
            get_device_capability=lambda: DeviceCapability(7, 0),
            device_count=lambda: 4,
        ),
    )
    monkeypatch.setattr(
        car,
        "_resolve_interconnect",
        lambda *a: car._InterconnectDecision(mode, fully_connected=True),
    )
    monkeypatch.setattr(car.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(car.dist, "get_rank", lambda **kw: 0)
    monkeypatch.setattr(car.dist, "get_world_size", lambda **kw: 4)
    monkeypatch.setattr(car.dist, "all_gather", Mock())
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(dcp_comm_backend=backend)
        ),
    )
    allocate = Mock(return_value=[100, 200, 300, 400])
    monkeypatch.setattr(car.CustomAllreduce, "create_shared_buffer", allocate)
    monkeypatch.setattr(car.CustomAllreduce, "close", Mock())
    monkeypatch.setattr(car.ops, "meta_size", lambda: 0)
    monkeypatch.setattr(car.ops, "init_custom_ar", Mock(return_value=123))
    monkeypatch.setattr(car.ops, "register_buffer", Mock())
    monkeypatch.setattr(car.ops, "sm70_tp4_push_allreduce_buffer_size", lambda: 100)
    monkeypatch.setattr(car.ops, "sm70_dcp2_p2p_buffer_size", lambda: 200)
    register = Mock()
    monkeypatch.setattr(car.ops, "register_sm70_tp4_push_allreduce_buffer", register)

    group = object()
    comm = car.CustomAllreduce(group, torch.device("cpu:0"))
    assert allocate.call_count == 3
    assert allocate.call_args.args[0] == 100 + extra
    register.assert_called_once_with(123, allocate.return_value)
    assert comm.sm70_dcp2_pex_enabled is bool(extra)


@pytest.mark.parametrize(
    "rows,mode,decode,use_pex",
    [
        (0, CUDAGraphMode.NONE, True, False),
        (5, CUDAGraphMode.NONE, False, False),
        (53, CUDAGraphMode.NONE, True, True),
        (54, CUDAGraphMode.NONE, True, False),
        (5, CUDAGraphMode.FULL, True, True),
        (5, CUDAGraphMode.PIECEWISE, True, False),  # First MTP draft forward.
    ],
)
def test_pex_batch_route(monkeypatch, rows, mode, decode, use_pex):
    pex = sm70_dcp2.Sm70Dcp2Pex(handle=123, peer=1, rank=0)
    group = SimpleNamespace(device_communicator=SimpleNamespace(sm70_dcp2_pex=pex))
    monkeypatch.setattr(
        sm70_dcp2,
        "get_forward_context",
        lambda: SimpleNamespace(is_uniform_decode=decode, cudagraph_runtime_mode=mode),
    )
    query = torch.empty((rows, 6, 256), dtype=torch.float16)
    selected = sm70_dcp2.get_sm70_dcp2_for_batch(
        group, query, torch.empty(rows, 12, 256), torch.empty(rows, 12), query, query
    )
    assert selected is (pex if use_pex else None)


@pytest.mark.parametrize("dcp_ranks,rank,peer", [([8, 9], 1, 0), ([10, 11], 0, 3)])
def test_pex_query_rank_mapping_and_layout(monkeypatch, dcp_ranks, rank, peer):
    ca = SimpleNamespace(_ptr=123, sm70_dcp2_pex_enabled=True)
    tp = SimpleNamespace(
        ranks=[8, 9, 10, 11], device_communicator=SimpleNamespace(ca_comm=ca)
    )
    runtime = sm70_dcp2.Sm70Dcp2Pex.create(tp, dcp_ranks, rank)
    assert runtime is not None
    fragments = torch.arange(2 * 5 * 6 * 256, dtype=torch.int16).reshape(2, 5, 6, 256)

    def exchange(handle, peer_rank, local, outbound, local_out, peer_out):
        assert (handle, peer_rank) == (123, peer)
        assert local is outbound
        assert local_out.stride(0) == peer_out.stride(0) == 12 * 256
        assert local_out.data_ptr() - peer_out.data_ptr() == (2 * rank - 1) * 3072
        local_out.copy_(local)
        peer_out.copy_(fragments[1 - rank].view(torch.float16))

    monkeypatch.setattr(sm70_dcp2.ops, "sm70_dcp2_p2p", exchange)
    query = runtime.gather_query(fragments[rank].view(torch.float16))
    assert torch.equal(query.view(torch.int16), torch.cat(tuple(fragments), dim=1))


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="CPU Triton")
@pytest.mark.parametrize("rows,rank", [(5, 0), (20, 1)])
def test_pex_packed_a2a_matches_nccl_layout(monkeypatch, rows, rank):
    peer = 1 - rank
    group = SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setattr(vllm.distributed, "get_dcp_group", lambda: group)
    runtime = sm70_dcp2.Sm70Dcp2Pex(handle=123, peer=peer, rank=rank)
    partial = (
        torch.arange(2 * rows * 12 * 256, dtype=torch.float32).reshape(2, rows, 12, 256)
        / 16384
    )
    lse = torch.arange(2 * rows * 12, dtype=torch.float32).reshape(2, rows, 12) / 32
    lse[peer, 0, rank * 6] = -torch.inf
    # Independent rank-major reference, indexed [source, destination].
    packed = torch.cat([partial, lse.unsqueeze(-1)], dim=-1)
    packed = packed.view(2, rows, 2, 6, 257).transpose(1, 2)
    expected_recv = packed[:, rank].contiguous()

    def nccl(recv, send, *, group, async_op):
        assert async_op
        assert torch.equal(send.view_as(packed[rank]), packed[rank])
        recv.copy_(expected_recv.view(-1))
        return Mock()

    monkeypatch.setattr(dcp_alltoall.dist, "all_to_all_single", nccl)
    baseline = dcp_alltoall.dcp_a2a_lse_reduce(
        partial[rank], lse[rank], group, is_lse_base_on_e=False
    )
    gate = torch.full((rows, 6, 256), 0.5, dtype=torch.float16)
    expected_out = torch.empty_like(gate)
    _qsa_output_gate(expected_out, gate, merged=baseline)

    def exchange(handle, peer_rank, local, outbound, local_out, peer_out):
        assert (handle, peer_rank) == (123, peer)
        delta = (2 * rank - 1) * rows * 6168
        assert local.data_ptr() - outbound.data_ptr() == delta
        assert local_out.data_ptr() - peer_out.data_ptr() == delta
        assert torch.equal(local, packed[rank, rank])
        assert torch.equal(outbound, packed[rank, peer])
        local_out.copy_(local)
        peer_out.copy_(packed[peer, rank])
        for source, received in [(rank, local_out), (peer, peer_out)]:
            assert torch.equal(
                received.view(torch.int32), expected_recv[source].view(torch.int32)
            )

    monkeypatch.setattr(sm70_dcp2.ops, "sm70_dcp2_p2p", Mock(side_effect=exchange))
    out = torch.empty_like(gate)
    runtime.combine_gate(partial[rank], lse[rank], gate, out)
    sm70_dcp2.ops.sm70_dcp2_p2p.assert_called_once()
    assert torch.equal(out.view(torch.int16), expected_out.view(torch.int16))
