# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import distributed
from vllm.config import CUDAGraphMode, KernelConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.models.qwen4_exp.nvidia import qsa
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp, qsa_paged_fp16_h12
from vllm.v1.worker import workspace
from vllm.v1.worker.gpu import cudagraph_utils

pytestmark = pytest.mark.skip_global_cleanup


def _forward_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        ),
    )


@pytest.mark.parametrize(
    "enabled,phase,rows,topk,native",
    [
        (True, True, 48, 2048, True),
        (False, True, 48, 2048, False),
        (True, False, 48, 2048, False),
        (True, True, 49, 2048, False),
        (True, True, 48, 8192, False),
    ],
)
def test_qsa_h12_dcp_reader(monkeypatch, enabled, phase, rows, topk, native):
    layer = SimpleNamespace(
        sm70_qsa_paged_fp16_h12=enabled,
        sm70_qsa_device_history=False,
        dcp_local_indices_buffer=torch.empty(rows, topk + 3, dtype=torch.int32),
        dcp_partial_output_buffer=torch.empty(rows, 12, 256),
        dcp_partial_lse_buffer=torch.empty(rows, 12),
        cp_kv_cache_interleave_size=1,
        indexer=SimpleNamespace(token_topk=topk, compress_ratio=4),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )
    group = SimpleNamespace(
        all_gather=lambda q, dim: torch.cat((q, q), dim=dim),
        device_communicator=SimpleNamespace(sm70_dcp2=None),
    )
    monkeypatch.setattr(distributed, "get_dcp_group", lambda: group)
    monkeypatch.setattr(
        qsa_dcp, "qsa_localize_dcp_indices", lambda src, dst, **kwargs: dst.copy_(src)
    )
    monkeypatch.setattr(
        qsa_ops,
        "qsa_dcp_sparse_paged_attention_sm70_grouped_page4",
        lambda *args, **kwargs: 0,
    )
    logical = torch.zeros_like(layer.dcp_local_indices_buffer)
    query = torch.zeros(rows, 6, 256, dtype=torch.float16)
    output = torch.empty_like(query)
    cache = torch.empty(2, 2, 416, 1, 256, dtype=torch.float16)
    key, value = cache.unbind(1)
    table = torch.zeros(1, 2, dtype=torch.int32)
    requests = torch.zeros(rows, dtype=torch.int32)

    def reader(q, k, v, ids, block_table, token_to_req, out, lse, *, paged):
        assert native is paged
        assert q.shape == (rows, 12, 256)
        assert k is key and v is value
        assert block_table is table and token_to_req.shape == (rows,)
        assert ids.shape == (rows, topk // 2 + 2)
        assert ids.stride(0) == topk + 3
        out.fill_(3)
        lse.fill_(-2)

    def combine(out, lse, dcp_group, *, is_lse_base_on_e):
        assert dcp_group is group and not is_lse_base_on_e
        assert torch.all(out == 3) and torch.all(lse == -2)
        return out[:, :6]

    monkeypatch.setattr(qsa, "qsa_paged_fp16_h12", lambda *a: reader(*a, paged=True))

    def triton_reader(q, k, v, ids, block_table, token_to_req, out, *, lse, **kwargs):
        reader(q, k, v, ids, block_table, token_to_req, out, lse, paged=False)

    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", triton_reader)
    monkeypatch.setattr(qsa_ops, "_qsa_output_gate", lambda out, gate: out.mul_(gate))
    impl = SimpleNamespace(
        dcp_world_size=2, dcp_rank=0, kv_cache_dtype="float16", dcp_combine=combine
    )
    with set_forward_context(None, _forward_config(), is_uniform_decode=phase):
        qsa.Qwen4ExpQSAFlashAttentionImpl._forward_qsa_dcp(
            impl,
            layer,
            query,
            key,
            value,
            logical,
            table,
            requests,
            output,
            torch.ones_like(query),
        )
    assert torch.all(output == 3)


def test_qsa_h12_workspace_survives_capture_lock(monkeypatch):
    assert not KernelConfig().sm70_qsa_paged_fp16_h12
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    layer = SimpleNamespace(
        qsa_dcp_sharded=True,
        dcp_partial_output_buffer=torch.empty(128, 12, 256),
        dcp_local_indices_buffer=torch.empty(128, 2051, dtype=torch.int32),
        sm70_qsa_paged_fp16_h12=True,
        sm70_qsa_device_history=False,
        indexer=SimpleNamespace(token_topk=2048, compress_ratio=4),
        cp_kv_cache_interleave_size=1,
        impl=SimpleNamespace(dcp_world_size=2),
    )
    qsa.Qwen4ExpQSAAttention.reserve_capture_workspace(layer)
    manager.lock()
    partial, state = qsa_paged_fp16_h12.qsa_paged_fp16_h12_workspace(48, 1026)
    assert partial.shape == (48, 12, 17, 256)
    assert state.shape == (48, 12, 17, 2)
    smaller, _ = qsa_paged_fp16_h12.qsa_paged_fp16_h12_workspace(6, 1026)
    assert smaller.data_ptr() == partial.data_ptr()


@pytest.mark.parametrize(
    "mode,uniform_token_count,expected",
    [
        (CUDAGraphMode.FULL, 6, True),
        (CUDAGraphMode.FULL, None, False),
        (CUDAGraphMode.PIECEWISE, None, False),
    ],
)
def test_qsa_h12_capture_phase(monkeypatch, mode, uniform_token_count, expected):
    graphs = object.__new__(cudagraph_utils.ModelCudaGraphManager)
    graphs.dp_size = 1
    graphs.is_first_pp_rank = graphs.is_last_pp_rank = True
    graphs.hidden_states = None
    graphs.vllm_config = _forward_config()
    desc = cudagraph_utils.BatchExecutionDescriptor(mode, 48, 8, uniform_token_count)

    def capture(self, create_forward_fn, progress_bar_desc):
        forward, _ = create_forward_fn(desc)
        forward(CUDAGraphMode.NONE)
        forward(mode)
        return {}

    seen = []

    def model(**kwargs):
        seen.append(get_forward_context().is_uniform_decode)
        return torch.empty(48, 1)

    monkeypatch.setattr(cudagraph_utils.CudaGraphManager, "capture", capture)
    monkeypatch.setattr(
        cudagraph_utils, "prepare_inputs_to_capture", lambda *a, **kw: (None, {})
    )
    graphs.capture(
        model,
        SimpleNamespace(prepare_dummy_inputs=lambda *a: {}),
        SimpleNamespace(input_ids=torch.zeros(48), positions=torch.arange(48)),
        None,
        None,
        [],
        None,
    )
    assert seen == [expected, expected]
