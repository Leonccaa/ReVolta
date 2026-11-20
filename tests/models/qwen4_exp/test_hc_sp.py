# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import accumulate
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.models.common.ops import sequence_parallel as sp
from vllm.models.qwen4_exp.nvidia import model, ple_layer
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu import model_runner

TP = 4
PLE_CASES = [((5624,), 8192), ((7, 3, 7), 22), ((4000, 4190), 8190)]


def set_tp(monkeypatch, rank, tp_size=TP):
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(sp, "_custom_collective", lambda *_: None)


def make_ple_case(lengths, physical, device, width=8):
    torch.manual_seed(29)
    actual = sum(lengths)
    dtype = torch.float16 if device == "cuda" else torch.float32
    x = torch.randn(physical, width, dtype=dtype, device=device)
    x[actual:].fill_(101)
    state = torch.randn(len(lengths) + 1, width, 9, dtype=dtype, device=device)
    weights = torch.randn(width, 4, dtype=dtype, device=device) / 8
    qsl = torch.tensor(list(accumulate(lengths, initial=0)), device=device)
    context = qsl.new_zeros(len(lengths), 2)
    meta = SimpleNamespace(
        num_prefills=len(lengths),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=actual,
        num_actual_tokens=actual,
        spec_sequence_masks=None,
        non_spec_query_start_loc=qsl,
        state_indices_tensor=torch.arange(1, len(lengths) + 1, device=device),
        has_initial_states_p=torch.arange(len(lengths), device=device) % 2 == 0,
        max_prefill_query_len=max(lengths),
    )
    return x, state, weights, qsl, context, meta


def make_ple_lookup(monkeypatch, x, placement, qsl, context):
    layer = ple_layer.Qwen4ExpPLELayer.__new__(ple_layer.Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.prefix = "test_ple"
    layer.conv_state_len = 9
    layer.short_conv_dilation = 3

    # Exercise the real forward/offload placement dispatch. Fixture outputs at
    # the embedding boundary are complete, with each row tied to its input ID.
    lookup = ple_layer.Qwen4ExpNGramEmbedding.__new__(ple_layer.Qwen4ExpNGramEmbedding)
    nn.Module.__init__(lookup)
    lookup._is_cpu_offloaded = placement in ("process", "disk", "cascade")
    lookup._keeps_local_tables = placement == "cascade"
    lookup.ngram_heads = 1
    lookup.layer_name = "test_lookup"
    lookup.ngram_embedding = lambda ids, **_: x[ids]
    lookup.wait_offloaded_output = lambda hidden, count: x[:count]
    monkeypatch.setattr(ple_layer, "is_offload_process", lambda: False)

    def write_ids(ids, starts, ngram, out, prefix):
        assert starts is qsl and ngram is context
        out[:, 0].copy_(ids)

    monkeypatch.setattr(torch.ops.vllm, "qwen4_exp_compute_ple_ngram_ids", write_ids)
    layer.ple_embedding = lookup
    layer._dequantize_embeddings = lambda value, _: value
    layer.key_proj = lambda value: (value, None)
    layer.value_proj = lambda value: (value, None)
    layer._gate_norm = lambda hidden, key, value: (hidden + value, key)
    return layer


def run_ple_case(monkeypatch, lengths, physical, device, placement, width=8):
    x, state, weights, qsl, context, meta = make_ple_case(
        lengths, physical, device, width
    )
    layer = make_ple_lookup(monkeypatch, x, placement, qsl, context)
    ids = torch.arange(physical, device=device)
    working_state = state.clone()

    def short_conv(inputs, output, prefix):
        assert inputs.shape[0] == physical
        result = layer._short_conv_dilated_dispatch(
            inputs, meta, working_state, weights
        )
        output[: result.shape[0]].copy_(result)

    monkeypatch.setattr(torch.ops.vllm, "qwen4_exp_ple_short_conv", short_conv)
    expected = layer(x, ids, qsl, context)
    expected_state = working_state.clone()
    shards = []
    for rank in range(TP):
        set_tp(monkeypatch, rank)
        working_state.copy_(state)

        def gather(local):
            torch.testing.assert_close(local, sp.sp_shard(x), rtol=0, atol=0)
            return torch.nn.functional.pad(x, (0, 0, 0, (-physical) % TP))

        monkeypatch.setattr(ple_layer, "sp_all_gather", gather)
        shards.append(layer(sp.sp_shard(x), ids, qsl, context, hc_sp=True))
        torch.testing.assert_close(working_state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(shards)[:physical], expected, rtol=0, atol=0)


@pytest.mark.parametrize("lengths,physical", PLE_CASES)
def test_ple_padding_regressions(monkeypatch, lengths, physical):
    run_ple_case(monkeypatch, lengths, physical, "cpu", "resident")


@pytest.mark.parametrize(
    "placement", ["pinned", "hybrid", "process", "disk", "cascade", "nvfp4"]
)
def test_ple_placement_uses_completed_rows(monkeypatch, placement):
    run_ple_case(monkeypatch, (7, 3, 7), 22, "cpu", placement)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths,physical", PLE_CASES)
def test_ple_cuda(monkeypatch, lengths, physical):
    run_ple_case(monkeypatch, lengths, physical, "cuda", "resident")


@pytest.mark.parametrize("draft", ["none", "mtp", "dflash"])
@pytest.mark.parametrize("vision", [False, True])
def test_target_restores_global_outputs(monkeypatch, draft, vision):
    rows, width = 17, 4
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(model, "get_pp_group", lambda: pp)
    backbone = model.Qwen4ExpModel.__new__(model.Qwen4ExpModel)
    nn.Module.__init__(backbone)
    backbone.config = SimpleNamespace(hc_count=2, hidden_size=width)
    backbone.start_layer = 0
    backbone.end_layer = 1
    backbone.aux_hidden_state_layers = (0, 1) if draft == "dflash" else ()
    backbone._aux_hidden_state_final = draft == "dflash"
    backbone.aux_hidden_state_dtype = torch.float16
    backbone._mtp_hidden_buffer = (
        torch.full((rows + 3, width * 2), -1.0) if draft == "mtp" else None
    )
    x = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
    ids = torch.arange(rows)
    positions = ids.expand(3, -1) if vision else ids
    deepstack = (
        IntermediateTensors({"deepstack_input_embeds_0": x / 2}) if vision else None
    )

    class Block(nn.Module):
        def forward(self, hidden_states, aux_hidden_states=None, **kwargs):
            assert kwargs["input_ids"] is ids and kwargs["positions"] is positions
            if aux_hidden_states is not None:
                aux_hidden_states.append(model.hc_stream_mean(hidden_states, 2))
            return hidden_states, None, None

    block = Block()
    block.mlp_hyper_connection = SimpleNamespace(combine=lambda h, *_: h)
    backbone.layers = nn.ModuleList([block])
    backbone.embed_input_ids = lambda _: x
    backbone.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda h, *_, **kw: (h, h[:, :width] * 2, None)
    )
    kwargs = dict(inputs_embeds=x if vision else None, deepstack_input_embeds=deepstack)
    expected = backbone.forward(ids, positions, **kwargs)
    sample, taps = expected if draft == "dflash" else (expected, [])
    base_buffer = backbone._mtp_hidden_buffer
    full_outputs = [sample, *taps]
    if base_buffer is not None:
        full_outputs = [torch.cat((sample, base_buffer[:rows]), dim=-1)]

    for rank in range(TP):
        set_tp(monkeypatch, rank)
        remaining = iter(full_outputs)

        def gather(local, remaining=remaining):
            full = next(remaining)
            torch.testing.assert_close(local, sp.sp_shard(full), atol=0, rtol=0)
            return torch.nn.functional.pad(full, (0, 0, 0, (-rows) % TP))

        monkeypatch.setattr(model, "sp_all_gather", gather)
        if base_buffer is not None:
            backbone._mtp_hidden_buffer = torch.full_like(base_buffer, -1.0)
        actual = backbone.forward(ids, positions, hc_sp=True, **kwargs)
        assert next(remaining, None) is None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if base_buffer is not None:
            torch.testing.assert_close(backbone._mtp_hidden_buffer, base_buffer)
        actual_sample = actual[0] if draft == "dflash" else actual
        torch.testing.assert_close(actual_sample[[0, 6, 9, 16]], sample[[0, 6, 9, 16]])


@pytest.mark.parametrize("attention", ["linear_attention", "full_attention"])
@pytest.mark.parametrize("partial_moe", [False, True])
def test_decoder_reduces_only_partial_outputs(monkeypatch, attention, partial_moe):
    set_tp(monkeypatch, 3)
    rows = 17
    x = torch.arange(rows, dtype=torch.float32).unsqueeze(-1)
    layer = model.Qwen4ExpDecoderLayer.__new__(model.Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.ple = None
    layer.aux_hidden_state_tap = False
    layer.layer_type = attention
    layer._sm70_hc_ll = layer._sm70_hc_ll_attn_partial = True
    layer.attn_hyper_connection = SimpleNamespace(mix=lambda h, **kw: (h, h, h))

    def combine(h, block, injection, *, hc_sp):
        assert hc_sp
        torch.testing.assert_close(block, sp.sp_shard(2 * x))
        return h, h, injection

    layer.mlp_hyper_connection = SimpleNamespace(combine_and_mix=combine)

    def gdn(hidden_states, output, *, defer_tp_reduce):
        assert defer_tp_reduce and torch.equal(hidden_states, x)
        output.copy_(x / 2)

    def qsa(hidden_states, positions, output, *, reduce_results):
        assert not reduce_results and output is None and torch.equal(hidden_states, x)
        return x / 2

    layer.linear_attn = gdn
    layer.self_attn = qsa
    moe = model.Qwen4ExpSparseMoeBlock.__new__(model.Qwen4ExpSparseMoeBlock)
    nn.Module.__init__(moe)
    runner = model.MoERunner.__new__(model.MoERunner)
    runner.can_defer_tp_reduce = lambda: partial_moe
    moe.experts = SimpleNamespace(runner=runner)

    def moe_forward(hidden_states, *, reduce_results=True):
        assert reduce_results != partial_moe and torch.equal(hidden_states, x)
        return x * (3 if reduce_results else 0.75)

    moe.forward = moe_forward
    layer.mlp = moe
    reductions = []

    def reduce(value, dim):
        assert dim == 0 and value.shape[0] == 20
        reductions.append(value)
        return value.chunk(TP)[3] * TP

    monkeypatch.setattr(sp, "tensor_model_parallel_reduce_scatter", reduce)
    monkeypatch.setattr(
        model, "sp_all_gather", lambda local: torch.nn.functional.pad(x, (0, 0, 0, 3))
    )
    _, output, _ = layer(
        sp.sp_shard(x),
        None,
        None,
        torch.arange(rows),
        input_ids=None,
        query_start_loc=None,
        ngram_context=None,
        hc_sp=True,
    )
    torch.testing.assert_close(output, sp.sp_shard(3 * x))
    assert len(reductions) == 1 + partial_moe


@pytest.mark.parametrize(
    "rows,cmax,prefill,drafts,mode,skip,profile,expected",
    [
        (4095, 512, [True], 0, CUDAGraphMode.NONE, False, False, False),
        (4096, 512, [True], 0, CUDAGraphMode.NONE, False, False, True),
        (4096, 8192, [True], 0, CUDAGraphMode.NONE, False, False, False),
        (8191, 8192, [True], 0, CUDAGraphMode.NONE, False, False, False),
        (8192, 8192, [True], 0, CUDAGraphMode.NONE, False, False, False),
        (8193, 8192, [True], 0, CUDAGraphMode.NONE, False, False, True),
        (8193, 512, [True, False], 0, CUDAGraphMode.NONE, False, False, False),
        (8193, 512, [False], 0, CUDAGraphMode.NONE, False, False, False),
        (8193, 512, [True], 1, CUDAGraphMode.NONE, False, False, False),
        (8193, 512, [True], 0, CUDAGraphMode.PIECEWISE, False, False, False),
        (8193, 512, [True], 0, CUDAGraphMode.FULL, False, False, False),
        (8193, 512, [True], 0, CUDAGraphMode.NONE, True, False, False),
        (8193, 512, [False], 0, CUDAGraphMode.NONE, False, True, True),
    ],
)
def test_runner_admission(rows, cmax, prefill, drafts, mode, skip, profile, expected):
    input_batch = SimpleNamespace(
        num_tokens=rows,
        num_tokens_after_padding=12288,
        num_draft_tokens=drafts,
        is_prefilling_np=np.asarray(prefill),
    )
    runner = SimpleNamespace(
        _hc_sp_enabled=True,
        cudagraph_manager=SimpleNamespace(max_captured_tokens=cmax),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=2048),
    )
    min_tokens = model_runner.GPUModelRunner.hc_sp_min_tokens.fget(runner)
    assert (
        bool(
            model_runner._should_use_hc_sp(
                input_batch,
                mode,
                min_tokens,
                skip_compiled=skip,
                profile_hc_sp=profile,
            )
        )
        == expected
    )


@pytest.mark.parametrize("tp_size", [2, 4])
def test_compiled_sharding_keeps_symbolic_tail(monkeypatch, tp_size):
    set_tp(monkeypatch, tp_size - 1, tp_size)
    traced = []

    def backend(graph, _):
        traced.append(graph)
        return graph.forward

    def reduce(x, dim):
        assert dim == 0 and x.shape[0] % tp_size == 0
        return x.chunk(tp_size)[-1] * tp_size

    monkeypatch.setattr(sp, "tensor_model_parallel_reduce_scatter", reduce)

    def run(x):
        return sp.sp_shard(x, dynamic_pad=True), sp.sp_reduce_scatter(
            x, dynamic_pad=True
        )

    torch._dynamo.reset()
    compiled = torch.compile(
        run,
        backend=backend,
        fullgraph=True,
        dynamic=True,
        options={"guard_filter_fn": lambda guards: [False for _ in guards]},
    )
    for rows in (8192, 8190, 8191, 8193, 8194, 8195):
        x = torch.arange(rows * 4).reshape(rows, 4)
        torch.testing.assert_close(compiled(x), run(x), rtol=0, atol=0)
    assert len(traced) == 1


def test_row_parallel_projection_reduction_override(monkeypatch):
    from vllm.model_executor.layers import linear

    projection = linear.RowParallelLinear.__new__(linear.RowParallelLinear)
    nn.Module.__init__(projection)
    projection.input_is_parallel = True
    projection.tp_size = 4
    projection.tp_rank = 0
    projection.bias = None
    projection.skip_bias_add = False
    projection.return_bias = True
    projection.reduce_results = True
    projection.quant_method = SimpleNamespace(apply=lambda _, x, bias: x * 2)
    for name in (
        "_maybe_sm70_awq_mlp_down_tile_gemm_reduce",
        "_maybe_sm70_dense_forward",
        "_maybe_sm70_awq_mlp_down_tile_all_reduce",
    ):
        monkeypatch.setattr(linear, name, lambda *_: None)
    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", lambda x: x * TP)

    x = torch.arange(12).reshape(3, 4)
    torch.testing.assert_close(projection(x, reduce_results=False)[0], x * 2)
    torch.testing.assert_close(projection(x)[0], x * 2 * TP)
    assert projection.reduce_results


def test_gdn_opaque_output_contract(monkeypatch):
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.utils.torch_utils import _encode_layer_name

    layer = gdn.QwenGatedDeltaNetAttention.__new__(gdn.QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.prefix = "test_gdn"
    layer.forward_cuda = (
        lambda x, output, *, defer_tp_reduce: x * 2
        if output is None
        else output.copy_(x * 2)
    )
    layer._output_projection = lambda x, z, output, n, defer_tp_reduce: output.copy_(
        x * (2 if defer_tp_reduce else TP * 2)
    )
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = torch.empty_like(x)
    cache = x.new_empty(0)
    name = _encode_layer_name(layer.prefix)
    with override_forward_context(ForwardContext({layer.prefix: layer}, None, {})):
        direct = torch.ops.vllm.qwen_gdn_full_forward_direct(
            x, cache, cache, name, True
        )
        torch.testing.assert_close(direct, x * 2)
        torch.ops.vllm.qwen_gdn_full_forward(x, out, cache, cache, name, True)
        torch.testing.assert_close(out, x * 2)
        torch.ops.vllm.qwen_gdn_output_projection(x, x, out, 3, name, True)
        torch.testing.assert_close(out, x * 2)
        torch.ops.vllm.qwen_gdn_output_projection(x, x, out, 3, name)
        torch.testing.assert_close(out, x * 2 * TP)


def test_aux_gather_preserves_fp8_bytes(monkeypatch):
    set_tp(monkeypatch, 3)
    full = torch.arange(68, dtype=torch.uint8).reshape(17, 4)
    padded = torch.nn.functional.pad(full, (0, 0, 0, 3))
    local = sp.sp_shard(full)

    def gather(value):
        assert value.dtype == torch.uint8
        torch.testing.assert_close(value, local)
        return padded

    monkeypatch.setattr(model, "sp_all_gather", gather)
    result = model._gather_aux_hidden_state(local.view(torch.float8_e4m3fn), 17)
    assert result.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(result.view(torch.uint8), full)
