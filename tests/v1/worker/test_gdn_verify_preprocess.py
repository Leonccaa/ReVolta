# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from the 1Cat #904/#924 bitwise verifier preprocessing tests."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(os.getenv("TRITON_INTERPRET") != "1", reason="CPU interpreter"),
]

CASES = [
    (1, 8, 2560, 10),
    (2, 8, 4096, 10),
    (3, 8, 4096, 10),
    (4, 4, 4096, 10),
    (8, 5, 4096, 10),
    (1, 5, 2560, 7),
    (4, 5, 4096, 7),
]


def _make_layer(batch, tokens, slots, conv, ssm, weight):
    layer = object.__new__(gdn.QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "gdn"
    layer.tp_size = 4
    layer.num_k_heads, layer.num_v_heads = 16, 48
    layer.head_k_dim = layer.head_v_dim = 128
    layer.enable_packed_recurrent_decode = False
    layer.enable_sm70_dflash2_fused_gdn_verify = False
    layer.enable_sm70_fused_sigmoid_mixed_qkv = True
    layer.spec_state_replay = True
    layer._replay_general_geometry = slots == 5  # Native MTP4
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(weight=weight.unsqueeze(1), bias=None)
    layer.kv_cache = (conv, ssm)
    layer.A_log = torch.linspace(-1, 1, 12, device=conv.device)
    layer.dt_bias = torch.linspace(-0.5, 0.5, 12, device=conv.device)
    layer._replay_slots = slots
    layer._replay_qkv = conv.new_empty(tokens, 2560)
    layer._replay_a = conv.new_empty(tokens, 12)
    layer._replay_b = conv.new_empty(tokens, 12)
    layer._replay_out = conv.new_empty(1, tokens, 12, 128)
    layer._replay_table = torch.full(
        (batch, slots + 1), PAD_SLOT_ID, dtype=torch.int32, device=conv.device
    )
    layer._replay_selectors = layer._replay_table.new_full((2 * batch,), slots + 1)
    return layer


def check_verify(batch, q, row_stride, history, device, *, graph=False, fuse=True):
    """Run the target's opaque recurrent op and accepted replay on one TP4 shard."""
    slots = history - 2
    tokens = batch * q
    gen = torch.Generator().manual_seed(17)

    def rand(*shape, dtype=torch.float16):
        return (torch.randn(shape, generator=gen) * 0.1).to(device, dtype)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    raw = rand(tokens, row_stride)
    conv = rand(batch + 1, history, 2560)
    if gdn.is_conv_state_dim_first():
        conv = conv.transpose(1, 2).contiguous()
    ssm = rand(batch + 1, 12, 128, 128, dtype=torch.float32)
    weight, a, b = rand(2560, 4), rand(tokens, 12), rand(tokens, 12)
    layers = [
        _make_layer(batch, tokens, slots, conv.clone(), ssm.clone(), weight)
        for _ in range(2)
    ]
    inputs = [raw.clone(), raw.clone()]
    outputs = [raw.new_empty(tokens + 1, 12, 128) for _ in layers]
    state_ids = i32([[PAD_SLOT_ID] * slots for _ in range(tokens)])
    selectors = i32([1] * tokens)
    cu = i32([0] * (tokens + 1))
    replay_ids = i32([[PAD_SLOT_ID] * (slots + 1) for _ in range(batch)])
    masks = torch.zeros(tokens, dtype=torch.bool, device=device)
    meta = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=batch,
        num_spec_decode_tokens=tokens if fuse else tokens - 1,
        num_actual_tokens=tokens,
        spec_query_start_loc=cu,
        spec_state_indices_tensor=state_ids,
        spec_sequence_masks=masks,
        num_accepted_tokens=selectors,
        spec_state_slot_selectors=selectors,
    )
    meta.max_query_len = q
    update = gdn.causal_conv1d_update
    recurrent = gdn.fused_sigmoid_gating_delta_rule_update_mixed_qkv

    def run(index, *, all_live=False):
        layer = layers[index]

        def conv_update(*args, **kwargs):
            fused_out = kwargs.pop("zero_core_out", None)
            assert (fused_out is not None) == fuse
            if index == 0:
                outputs[index].zero_()
                # Original call uses the physical state table width. For PAD
                # replay the reference shares the existing explicit output
                # buffer, because the allocator-owned recurrent result has
                # undefined values in the skipped rows.
                kwargs["max_query_len"] = state_ids.size(1)
                return update(*args, **kwargs)
            if fused_out is not None:
                kwargs["zero_core_out"] = fused_out
            return update(*args, **kwargs)

        def recurrent_update(**kwargs):
            if index == 0 and all_live:
                kwargs["out"] = None
            return recurrent(**kwargs)

        context = ForwardContext({layer.prefix: layer}, {layer.prefix: meta}, {})
        with (
            override_forward_context(context),
            patch.object(gdn, "causal_conv1d_update", conv_update),
            patch.object(
                gdn,
                "fused_sigmoid_gating_delta_rule_update_mixed_qkv",
                recurrent_update,
            ),
        ):
            args = (
                inputs[index][:, :2560],
                b,
                a,
                outputs[index],
                *layer.kv_cache,
            )
            gdn.qwen_gdn_attention_core_standard_spec(
                *args,
                cu[:0],
                selectors[:0],
                cu,
                state_ids,
                selectors[:0],
                selectors[:0],
                masks,
                selectors,
                selectors,
                layer.prefix,
            )
        layer.replay_spec_states(replay_ids, cu[: batch + 1], tokens)

    def set_metadata(lengths, rows, counts, prefix):
        cu[1 : batch + 1].copy_(i32(lengths).cumsum(0))
        cu[batch + 1 :].fill_(sum(lengths))
        state_ids[:batch, 0].copy_(i32(rows))
        selectors[:batch].copy_(i32(counts))
        masks[:batch].copy_(i32([int(length > 0) for length in lengths]).bool())
        replay_ids.fill_(PAD_SLOT_ID)
        for request, (length, slot) in enumerate(zip(lengths, rows)):
            if length and slot != PAD_SLOT_ID:
                replay_ids[request, -1] = slot
                replay_ids[request, min(length, prefix) - 1] = slot

    live_rows = list(range(batch))
    cases = (
        [
            ([q] * batch, live_rows, [1] * batch, 1),
            ([q - 1] + [q] * (batch - 1), live_rows[::-1], [slots] * batch, q - 1),
            ([q] * (batch - 1) + [0], live_rows, [2] * batch, 2),
            ([q] * batch, [PAD_SLOT_ID] + live_rows[1:], [1] * batch, 2),
        ]
        if fuse
        else [([q - 1] + [q] * (batch - 1), live_rows, [1] * batch, 1)]
    )
    graph_runs = []
    if graph:
        # Capture metadata has PAD states, as in MRv2. Replay updates the same
        # storage, including an initially inactive slot that becomes live.
        set_metadata([q] * batch, [PAD_SLOT_ID] * batch, [1] * batch, 1)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for index in range(2):
                run(index)
        torch.cuda.current_stream().wait_stream(stream)
        for index in range(2):
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                run(index)
            graph_runs.append(captured)

    for lengths, rows, counts, prefix in cases:
        set_metadata(lengths, rows, counts, prefix)
        for index in range(2):
            inputs[index].copy_(raw)
            outputs[index].fill_(float("nan"))
            layers[index].kv_cache[0].copy_(conv)
            layers[index].kv_cache[1].copy_(ssm)
            if graph:
                graph_runs[index].replay()
            else:
                run(
                    index,
                    all_live=all(lengths)
                    and rows == live_rows
                    and sum(lengths) == tokens,
                )
        out_pair = outputs if fuse else [out[: sum(lengths)] for out in outputs]
        pairs = [
            inputs,
            out_pair,
            [layer.kv_cache[0] for layer in layers],
            [layer.kv_cache[1] for layer in layers],
            [layer._replay_qkv for layer in layers],
            [layer._replay_a for layer in layers],
            [layer._replay_b for layer in layers],
        ]
        for left, right in pairs:
            assert torch.equal(left.view(torch.int16), right.view(torch.int16))
        if fuse:
            start = 0
            for length, slot in zip(lengths, rows):
                if slot == PAD_SLOT_ID:
                    assert not outputs[1][start : start + length].count_nonzero()
                start += length
            assert not outputs[1][start:].count_nonzero()
        assert torch.equal(outputs[1][-1], torch.zeros_like(outputs[1][-1]))


@pytest.mark.parametrize("batch,q,row_stride,history", CASES)
def test_gdn_verify_preprocess(monkeypatch, batch, q, row_stride, history):
    monkeypatch.setattr(
        gdn.current_platform, "is_device_capability", lambda cap: cap == 70
    )
    check_verify(batch, q, row_stride, history, torch.device("cpu"))


def test_ragged_verify_uses_original_conv(monkeypatch):
    monkeypatch.setattr(
        gdn.current_platform, "is_device_capability", lambda cap: cap == 70
    )
    check_verify(2, 8, 4096, 10, torch.device("cpu"), fuse=False)
