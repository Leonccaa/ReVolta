# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.gdn_spec_replay import (
    replay_gdn_spec_states,
    replay_graph_key,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def _layer(device="cpu"):
    layer = object.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.spec_state_replay = True
    layer._native_gdn_verify = True
    layer._replay_slots = 8
    layer._replay_table = torch.full(
        (8, 9), PAD_SLOT_ID, device=device, dtype=torch.int32
    )
    layer._replay_selectors = torch.full((16,), 9, device=device, dtype=torch.int32)
    layer._replay_qkv = torch.zeros(64, 2560, device=device, dtype=torch.float16)
    layer._replay_a = torch.zeros(64, 12, device=device, dtype=torch.float16)
    layer._replay_b = torch.zeros_like(layer._replay_a)
    layer._replay_out = torch.zeros(1, 64, 12, 128, device=device, dtype=torch.float16)
    layer.A_log = torch.zeros(12, device=device)
    layer.dt_bias = torch.zeros(12, device=device, dtype=torch.float16)
    layer.kv_cache = (None, torch.zeros(24, 12, 128, 128, device=device))
    layer.num_k_heads, layer.num_v_heads, layer.tp_size = 16, 48, 4
    layer.head_k_dim = layer.head_v_dim = 128
    return layer


@pytest.mark.parametrize(
    "requests,tokens,prefills,decodes,enabled,expected",
    [
        (1, 8, 0, 0, True, 8),
        (2, 16, 0, 0, True, 8),
        (3, 24, 0, 0, True, 8),
        (4, 16, 0, 0, True, 8),
        (8, 40, 0, 0, True, 8),
        (9, 45, 0, 0, True, 0),
        (8, 72, 0, 0, True, 0),
        (8, 39, 0, 0, True, 0),
        (4, 16, 1, 0, True, 0),
        (4, 16, 0, 1, True, 0),
        (1, 8, 0, 0, False, 0),
    ],
)
def test_verify_admission(requests, tokens, prefills, decodes, enabled, expected):
    layer = _layer()
    layer._native_gdn_verify = enabled
    metadata = GDNAttentionMetadata(prefills, 0, decodes, 0, requests, tokens, tokens)
    assert layer.native_gdn_verify_width(metadata, requests, tokens) == expected


def test_mrv2_passes_captured_width_to_accepted_prefix_replay(monkeypatch):
    layer = _layer()
    base = torch.tensor([[3], [1], [PAD_SLOT_ID]], dtype=torch.int32)
    cu = torch.tensor([0, 5, 10, 10], dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        0, 0, 0, 0, 2, 10, 15, spec_query_start_loc=cu, spec_state_indices_tensor=base
    )
    state = SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={"gdn": layer})
        ),
    )
    step = MambaHybridModelState._make_spec_replay_step(
        state,
        {"gdn": metadata},
        SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["gdn"])]),
        (torch.tensor([[4, 5], [6, 7]], dtype=torch.int32),),
        torch.tensor([4, 4], dtype=torch.int32),
        full_graph=True,
        num_reqs=3,
        num_tokens=15,
    )
    assert step is not None and step.layers[0].native_width == 8
    qkv = torch.ones(15, 4096, dtype=torch.float16)[:, :2560]
    gates = torch.full((15, 12), 2, dtype=torch.float16)
    verify_table, selectors = layer._spec_state_replay_verify(
        qkv, gates, gates, base, 3
    )
    assert torch.equal(verify_table[:, -1], base[:, 0])
    assert (verify_table[:, :-1] == PAD_SLOT_ID).all()
    assert selectors.tolist() == [9, 9, 9]
    assert torch.equal(layer._replay_qkv[:15], qkv)

    def native(qkv, a, b, a_log, bias, state, out, cu, table, nacc, *params):
        assert out.shape == (15, 12, 128)
        assert cu.tolist() == [0, 3, 5, 7, 10, 10, 10]
        assert nacc.tolist() == [9] * 6
        assert table[:, -1].tolist() == [3, -1, 1, -1, -1, -1]
        assert table[0, :3].tolist() == [4, -1, 3]
        assert table[2, :2].tolist() == [-1, 1]
        assert (table[1::2] == PAD_SLOT_ID).all()
        assert params[-1] == 8
        assert (a == 2).all() and (b == 2).all()
        hits.append(True)

    hits: list[bool] = []
    monkeypatch.setattr(ops, "sm70_gdn_verify_out", native)
    computed = torch.tensor([6, 5], dtype=torch.int32)
    replay_gdn_spec_states(
        step,
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
        num_sampled=torch.tensor([3, 2], dtype=torch.int32),
        num_computed_tokens=computed,
        block_size=3,
    )
    assert hits == [True]
    legacy = replace(step, layers=[replace(step.layers[0], native_width=0)])
    assert replay_graph_key(step, computed, 3) != replay_graph_key(legacy, computed, 3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an SM70 CUDA device",
)
@pytest.mark.parametrize("requests,width", [(1, 8), (2, 5), (4, 4), (8, 5)])
def test_cuda_native_prefix_matches_full_replay(requests, width):
    from vllm.v1.worker.gpu.gdn_spec_replay import split_replay_launch

    assert hasattr(torch.ops._C, "sm70_gdn_verify_out")
    layer = _layer("cuda")
    state = layer.kv_cache[1]
    state.normal_(0, 0.03)
    original = state.clone()
    tokens = requests * width
    for tensor in (layer._replay_qkv, layer._replay_a, layer._replay_b):
        tensor.normal_(0, 0.25)
    cu = torch.arange(0, tokens + 1, width, dtype=torch.int32, device="cuda")
    base = torch.arange(requests, dtype=torch.int32, device="cuda")
    padded = 1 < requests < 8
    if padded:
        base[-1] = PAD_SLOT_ID
    table, selectors = layer._spec_state_replay_verify(
        layer._replay_qkv[:tokens].clone(),
        layer._replay_a[:tokens].clone(),
        layer._replay_b[:tokens].clone(),
        base[:, None],
        requests,
    )
    out = torch.zeros_like(layer._replay_out[0, :tokens])
    layer._launch_native_gdn_verify(out, state, table, cu, selectors, 8)
    assert torch.equal(state.view(torch.int32), original.view(torch.int32))

    num_active = requests - padded
    accepted = (torch.arange(num_active, device="cuda") % (width - 1)) + 2
    commit = table.clone()
    rows = torch.arange(num_active, device="cuda")
    commit[rows, accepted - 1] = base[:num_active]
    commit[rows, 0] = requests + base[:num_active]
    split_table, split_cu = split_replay_launch(commit, cu, accepted)

    full_out = torch.zeros_like(out)
    layer._launch_native_gdn_verify(full_out, state, commit, cu, selectors, 8)
    assert torch.equal(full_out.view(torch.int16), out.view(torch.int16))
    expected = state.clone()
    state.copy_(original)
    layer.replay_spec_states(split_table, split_cu, tokens, native_width=8)
    assert torch.equal(state.view(torch.int32), expected.view(torch.int32))
    written = torch.cat((base[:num_active], requests + base[:num_active]))
    assert not torch.equal(state[written], original[written])
