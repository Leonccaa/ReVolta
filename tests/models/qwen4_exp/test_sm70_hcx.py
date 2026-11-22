# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.config.kernel import KernelConfig
from vllm.config.offload import OffloadConfig
from vllm.distributed import parallel_state
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
from vllm.models.qwen4_exp.nvidia import hyperconnection as hc
from vllm.models.qwen4_exp.nvidia import model, sm70_hcx
from vllm.v1.attention.ops import sm70_e4m3_long
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _seam_inputs(rows):
    g = torch.Generator().manual_seed(42)
    return tuple(
        torch.randn(rows, width, generator=g, dtype=torch.float16).mul_(0.1)
        for width in (2560, 10240, 4)
    )


@pytest.fixture
def seam(monkeypatch):
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=1, world_size=4)
    )
    module = hc.GatedResidual(
        hc.HyperConnectionConfig(
            hidden_size=2560,
            hc_lowrank=320,
            params_dtype=torch.float16,
            hc_per_branch_norm=True,
        )
    )
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(0, 0.02, generator=g)
    for linear in (
        module.input_mix_weight_down_block_inject,
        module.input_mix_weight_up,
    ):
        dispatch_cpu_unquantized_gemm(linear, remove_weight=False)
    layer = model.Qwen4ExpDecoderLayer.__new__(model.Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer._sm70_hc_ll_attn_partial = True
    layer.layer_idx = 1
    layer.mlp_hyper_connection = module

    config = SimpleNamespace(
        kernel_config=KernelConfig(sm70_hcx=True),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
        offload_config=OffloadConfig(),
    )
    device = torch.device("cpu")
    group = object()
    owner = SimpleNamespace(device=device, rank=1, hcx=None)
    monkeypatch.setattr(model, "get_tp_hc_ll", lambda: owner)
    monkeypatch.setattr(hc, "get_tp_hc_ll", lambda: owner)
    monkeypatch.setattr(model, "get_tp_group", lambda: SimpleNamespace(cpu_group=group))

    monkeypatch.setattr(sm70_hcx.dist, "barrier", lambda *, group: None)
    monkeypatch.setattr(torch.accelerator, "device_index", lambda _: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    pointers = [0x100000 * i for i in range(1, 5)]
    allocate = Mock(return_value=pointers)
    monkeypatch.setattr(CustomAllreduce, "create_shared_buffer", allocate)
    native = Mock()
    monkeypatch.setattr(torch.ops._C, "sm70_hcx_out", native, raising=False)
    return SimpleNamespace(
        layer=layer,
        module=module,
        config=config,
        owner=owner,
        native=native,
        group=group,
        allocate=allocate,
        pointers=pointers,
    )


def test_request_only_and_graph_identity():
    config = KernelConfig()
    base = config.compute_hash()
    assert not config.sm70_hcx and not config.sm70_hcx_active
    config.sm70_hcx = True
    assert config.compute_hash() == base
    config.sm70_hcx_active = True
    assert config.compute_hash() != base


@pytest.mark.parametrize("case", ["no_mesh", "mode"])
def test_no_ipc_without_mesh_or_full_decode_graph(monkeypatch, seam, case):
    if case == "no_mesh":
        monkeypatch.setattr(model, "get_tp_hc_ll", lambda: None)
    else:
        seam.config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    model._prepare_sm70_hcx(seam.config, [seam.layer])
    assert not seam.config.kernel_config.sm70_hcx_active
    assert seam.module._sm70_hcx_name is None
    assert not seam.config.compilation_config.static_forward_context
    seam.allocate.assert_not_called()
    seam.native.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_mrv2_mtp_decoder_load_and_dispatch(monkeypatch, seam, enabled):
    """Use the normal checkpoint hook and the decoder's actual partial producer."""
    monkeypatch.setattr(
        model, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    monkeypatch.setattr(cg, "get_pp_group", Mock())
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(cg, "_worker_device_is_pre_ampere", lambda: False)
    monkeypatch.setattr(sm70_e4m3_long, "long_attention_enabled", lambda: False)
    seam.config.kernel_config.sm70_hcx = enabled
    seam.config.scheduler_config = SimpleNamespace(max_num_seqs=8)
    seam.config.parallel_config = SimpleNamespace(
        tensor_parallel_size=4, data_parallel_size=1
    )
    seam.config.compilation_config.cudagraph_capture_sizes = [6, 12]
    seam.config.compilation_config.max_cudagraph_capture_size = 12
    seam.config.speculative_config = SimpleNamespace(
        method="mtp",
        ngram_assist=False,
        num_speculative_tokens=5,
        num_speculative_tokens_per_batch_size=None,
    )

    backbone = model.Qwen4ExpModel.__new__(model.Qwen4ExpModel)
    nn.Module.__init__(backbone)
    backbone.vllm_config = seam.config
    backbone.config = SimpleNamespace(num_experts=0)
    backbone._qsa_layer_ids = frozenset()
    backbone.layers = nn.ModuleList([seam.layer])
    name = "layers.0.mlp_hyper_connection.hc_norm.weight"
    expected_norm = torch.full_like(seam.module.hc_norm.weight, 0.125)
    assert backbone.load_weights([(name, expected_norm)]) == {name}
    assert torch.equal(seam.module.hc_norm.weight, expected_norm)
    assert seam.config.kernel_config.sm70_hcx_active is enabled

    manager = cg.ModelCudaGraphManager(
        seam.config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 6
    )
    manager._graphs_captured = True
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={str(i): i for i in range(2)},
        num_computed_prefill_tokens=np.full(2, 800),
        prefill_len=SimpleNamespace(np=np.full(2, 800)),
    )
    layer = seam.layer
    layer.ple = None
    layer.aux_hidden_state_tap = False
    layer.layer_type = "full_attention"
    layer.attn_hyper_connection = SimpleNamespace(
        mix=lambda x: (x, x[:, :2560], x[:, :4].contiguous())
    )
    layer.self_attn = lambda hidden_states, **kwargs: hidden_states.contiguous()
    layer.mlp = nn.Identity()
    seam.owner.combine_norm = Mock(
        side_effect=lambda p, r, i, w, e: hc.hc_combine_norm(r, p * 4, i, w, e, 4)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def native(*args):
        args[8].copy_(args[2])
        args[9].copy_(args[0])
        args[10].copy_(args[3])

    seam.native.side_effect = native
    for count in (1, 2):
        schedule = SimpleNamespace(
            num_scheduled_tokens=dict.fromkeys(map(str, range(count)), 6),
            total_num_scheduled_tokens=count * 6,
        )
        uniform = runner._get_uniform_decode_token_count(schedule, False)
        desc = manager.dispatch(count, count * 6, uniform)
        assert (desc.cg_mode, desc.uniform_token_count) == (CUDAGraphMode.FULL, 6)
        ctx = ForwardContext(
            seam.config.compilation_config.static_forward_context,
            {},
            {},
            is_uniform_decode=desc.uniform_token_count is not None,
        )
        _, residual, _ = _seam_inputs(desc.num_tokens)
        with torch.no_grad(), override_forward_context(ctx):
            hidden, mlp, injection = layer(
                residual,
                None,
                None,
                torch.arange(desc.num_tokens),
                input_ids=None,
                query_start_loc=None,
                ngram_context=None,
            )
        if enabled and count == 1:
            assert torch.equal(hidden, residual)
            assert torch.equal(mlp, residual[:, :2560])
            assert torch.equal(injection, residual[:, :4])
        assert torch.isfinite(mlp).all()
    assert seam.native.call_count == int(enabled)
    assert seam.owner.combine_norm.call_count == 2 - int(enabled)


def test_native_pack_indices_and_padding():
    g = torch.Generator().manual_seed(17)
    down = torch.randn((336, 10240), dtype=torch.float16, generator=g)
    up = torch.randn((10240, 320), dtype=torch.float16, generator=g)
    for rank in range(4):
        wd = sm70_hcx.pack_down(down, rank)
        wu = sm70_hcx.pack_up(up, rank)
        assert wd.shape == (80, 6, 4, 2, 32, 8) and wd.is_contiguous()
        assert wu.shape == (80, 5, 4, 2, 32, 8) and wu.is_contiguous()
        # Last CTA, stream 3, down row 83 (true injection only on rank 3).
        expected = down[323, 10232:] if rank == 3 else torch.zeros(8, dtype=wd.dtype)
        torch.testing.assert_close(wd[79, 5, 3, 1, 11], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            wd[0, 0, 0, 0, 0], down[80 * rank, :8], rtol=0, atol=0
        )
        torch.testing.assert_close(
            wu[79, 4, 3, 1, 31], up[7680 + rank * 640 + 639, 312:], rtol=0, atol=0
        )


def test_attention_partial_dispatch_and_dynamic_compile(monkeypatch, seam):
    model._prepare_sm70_hcx(seam.config, [seam.layer])
    assert seam.config.kernel_config.sm70_hcx_active
    assert seam.allocate.call_count == 1
    rt = seam.owner.hcx
    ctx = ForwardContext(seam.config.compilation_config.static_forward_context, {}, {})
    capture = Mock(return_value=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capture)

    def base(partial, residual, injection, weight, eps):
        return hc.hc_combine_norm(residual, partial * 4, injection, weight, eps, 4)

    seam.owner.combine_norm = Mock(side_effect=base)
    ar = Mock(side_effect=lambda p: p * 4)
    monkeypatch.setattr(hc, "tensor_model_parallel_all_reduce", ar)

    def native(*args):
        (
            p,
            p1,
            res,
            inj,
            norm,
            eps,
            down,
            up,
            h_out,
            b_out,
            i_out,
            xn,
            sq,
            dpart,
            bar,
            seq,
            ar_ptrs,
            lora,
            hb,
            rank,
            dbg,
            full,
            schedule,
        ) = args
        assert p1 is dbg is None and (rank, full, schedule) == (1, 1, True)
        assert (ar_ptrs, lora, hb) == (rt.ar, rt.lora, rt.hb)
        assert ar_ptrs == seam.pointers
        assert lora == [p + 655360 for p in seam.pointers]
        assert hb == [p + 655360 + 21504 for p in seam.pointers]
        assert (xn.shape, sq.numel(), dpart.numel()) == ((8, 10240), 2560, 245760)
        assert bar.numel() == 4 and seq.numel() == 1
        assert down is seam.module._hcx_down and up is seam.module._hcx_up
        # This is a dispatcher stub, not a simulation of CUDA HCX arithmetic.
        h_out.copy_(res + 1)
        b_out.copy_(p + 2)
        i_out.copy_(inj + 3)

    seam.native.side_effect = native

    def call(p, residual, inj):
        return seam.module.combine_and_mix(residual, hc.HCTensorParallelPartial(p), inj)

    def baseline(p, residual, inj):
        h, xn = base(p, residual, inj, seam.module.hc_norm.weight, 1e-6)
        b, i = seam.module._project(xn)
        return h, b, i

    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(call, backend=backend, fullgraph=True, dynamic=True)
    # Compile at M65 (explicit all-reduce), use the same graph at M6 and M12,
    # then return to M6. Mode differences are decided at runtime in the op.
    with torch.no_grad(), override_forward_context(ctx):
        for m, uniform, capturing, hcx in [
            (65, False, False, False),
            (6, False, True, False),
            (6, True, False, False),
            (6, True, True, True),
            (12, True, True, False),
            (8, True, True, True),
            (6, True, True, True),
        ]:
            p, r, i = _seam_inputs(m)
            ctx.is_uniform_decode = uniform
            capture.return_value = capturing
            expected = (r + 1, p + 2, i + 3) if hcx else baseline(p, r, i)
            for out, ref in zip(compiled(p, r, i), expected, strict=True):
                assert out.is_contiguous()
                torch.testing.assert_close(out, ref, rtol=0, atol=0)
        ctx.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        for out, ref in zip(compiled(p, r, i), baseline(p, r, i), strict=True):
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert len(graphs) == 1
    assert seam.native.call_count == 3
    assert seam.owner.combine_norm.call_count == 4
    assert ar.call_count == 1


def test_moe_partial_seam_on_next_attention(monkeypatch, seam):
    producer = model.Qwen4ExpDecoderLayer.__new__(model.Qwen4ExpDecoderLayer)
    nn.Module.__init__(producer)
    producer.layer_idx = 1
    producer._sm70_hc_ll = True
    producer._sm70_hc_ll_attn_partial = False
    moe = model.Qwen4ExpSparseMoeBlock.__new__(model.Qwen4ExpSparseMoeBlock)
    nn.Module.__init__(moe)
    moe.experts = SimpleNamespace(
        runner=SimpleNamespace(can_defer_tp_reduce=lambda: True)
    )
    producer.mlp = moe
    consumer = seam.layer
    consumer.layer_idx = 2
    consumer._sm70_hc_ll_attn_partial = False
    consumer.ple = None
    consumer.attn_hyper_connection = seam.module

    model._prepare_sm70_hcx(seam.config, [producer, consumer])

    assert seam.module._sm70_hcx_name == "qwen4exp.hcx.moe.1"
    assert seam.owner.hcx is not None and not seam.owner.hcx.top1_enabled
    assert seam.config.kernel_config.sm70_hcx_active


def test_top1x_uses_the_hcx_peer_mailbox(monkeypatch, seam):
    from vllm.model_executor.layers import logits_processor as lp

    seam.config.kernel_config = KernelConfig(sm70_top1x=True)
    model._prepare_sm70_hcx(seam.config, [seam.layer])
    runtime = seam.owner.hcx
    assert runtime.top1_enabled and not seam.config.kernel_config.sm70_hcx_active

    calls = []

    def top1x(out, pairs, buffers, seq, rank):
        calls.append((buffers, rank))
        out.fill_(7)

    monkeypatch.setattr(torch.ops._C, "sm70_top1x_out", top1x, raising=False)
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.sm70_hc_ll.get_tp_hc_ll",
        lambda: seam.owner,
    )
    processor = lp.LogitsProcessor(248320)
    tokens = processor._maybe_sm70_top1x(torch.zeros(3, 2))
    assert tokens.tolist() == [7, 7, 7]
    assert calls[0][1] == 1 and len(calls[0][0]) == 4
    assert processor._maybe_sm70_top1x(torch.zeros(65, 2)) is None
    runtime.top1_enabled = False
    assert processor._maybe_sm70_top1x(torch.zeros(3, 2)) is None


def test_top1x_switch_does_not_change_compile_hash():
    assert KernelConfig().compute_hash() == KernelConfig(sm70_top1x=True).compute_hash()
