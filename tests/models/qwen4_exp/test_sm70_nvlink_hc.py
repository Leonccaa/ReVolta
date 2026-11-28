# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import combinations
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.config as config_module
import vllm.config.vllm as vllm_config_module
from vllm.config import SchedulerConfig, VllmConfig
from vllm.config.kernel import KernelConfig
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators import sm70_hc_ll as hc_ll
from vllm.lora.layers import ReplicatedLinearWithLoRA
from vllm.model_executor import parameter
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.qwen4_exp.nvidia import hyperconnection as hc
from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc
from vllm.platforms import sm70_interconnect
from vllm.platforms.interface import DeviceCapability

_UUIDS = ["GPU-0", "GPU-1", "GPU-2", "GPU-3"]
_MESH = set(combinations(range(4), 2))


class FakeNvml:
    NVML_P2P_CAPS_INDEX_NVLINK = 0
    NVML_P2P_STATUS_OK = 0

    def __init__(self, edges):
        self.edges = edges

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByUUID(self, uuid):
        return _UUIDS.index(uuid)

    def nvmlDeviceGetP2PStatus(self, source, peer, capability):
        return 0 if tuple(sorted((source, peer))) in self.edges else 1


@pytest.fixture
def settings(monkeypatch):
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        kernel_config=KernelConfig(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(
            architectures=["Qwen4ExpForCausalLM"], dtype=torch.float16
        ),
        speculative_config=SimpleNamespace(method="dflash"),
    )
    monkeypatch.setattr(config_module, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setenv("VLLM_SM70_NVLINK_HC", "1")
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", "1")
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(car.current_platform, "is_fully_connected", lambda _: False)
    monkeypatch.setattr(car, "gpu_p2p_access_check_fresh", lambda _: True)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda *_: True)
    return cfg


@pytest.mark.parametrize(
    "hc_ll_request,async_scheduling,is_draft,platform_disables_async,"
    "async_hc_ll,expect_warning",
    [
        (None, True, False, False, False, False),
        ("0", True, False, False, False, False),
        ("1", False, False, False, False, False),
        ("1", True, False, False, False, True),
        ("1", None, False, False, False, True),
        ("1", True, True, False, False, False),
        ("1", True, False, True, False, False),
        ("1", True, False, False, True, False),
    ],
)
def test_async_scheduling_warns_when_hc_ll_requested(
    monkeypatch,
    hc_ll_request,
    async_scheduling,
    is_draft,
    platform_disables_async,
    async_hc_ll,
    expect_warning,
):
    monkeypatch.delenv("VLLM_SM70_NVLINK_HC", raising=False)
    if hc_ll_request is not None:
        monkeypatch.setenv("VLLM_SM70_NVLINK_HC", hc_ll_request)
    if not platform_disables_async:
        monkeypatch.setattr(
            car.current_platform, "check_and_update_config", lambda _: None
        )
    warning_once = Mock()
    monkeypatch.setattr(vllm_config_module.logger, "warning_once", warning_once)

    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_model_len=1024,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
        ),
        kernel_config=KernelConfig(sm70_hc_ll_async=async_hc_ll),
        is_speculative_draft=is_draft,
    )

    assert config.scheduler_config.async_scheduling is (
        async_scheduling is not False and not platform_disables_async
    )
    messages = [
        call
        for call in warning_once.call_args_list
        if "VLLM_SM70_NVLINK_HC=1" in call.args[0]
    ]
    assert len(messages) == int(expect_warning)
    if messages:
        (warning,) = messages
        assert "asynchronous scheduling is enabled" in warning.args[0]
        assert "--no-async-scheduling" in warning.args[0]
        assert warning.kwargs["scope"] == "global"


@pytest.mark.parametrize(
    "case",
    [
        "pex",
        "forced_nvlink",
        "partial_mesh",
        "atomic",
        "default_off",
        "async",
        "full_mesh",
    ],
)
def test_physical_gate_and_policy(monkeypatch, settings, case):
    request = "auto" if case == "pex" else "nvlink"
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", request)
    if case == "pex":
        monkeypatch.setattr(
            sm70_interconnect, "_sm70_pcie_switch_tp4_mode", lambda *_: True
        )
    if case == "default_off":
        monkeypatch.delenv("VLLM_SM70_NVLINK_HC")
    if case == "async":
        settings.scheduler_config.async_scheduling = True
    edges = {
        "forced_nvlink": set(),
        "partial_mesh": {(0, 1), (1, 2), (2, 3), (0, 3)},
    }.get(case, _MESH)
    monkeypatch.setattr(hc_ll, "import_pynvml", lambda: FakeNvml(edges))
    monkeypatch.setattr(
        car.dist,
        "all_gather_object",
        lambda out, value, group: out.__setitem__(
            slice(None), [True if value is None else value] * 4
        ),
    )
    decision = car._resolve_interconnect(
        "tp", 0, 4, DeviceCapability(7, 0), list(range(4)), list(range(4))
    )
    assert decision.fully_connected
    assert decision.mode == ("pex_p2p" if case == "pex" else "nvlink")
    if case in ("pex", "default_off", "async"):
        assert not decision.hc_ll_requested
        return

    assert decision.hc_ll_requested
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 4)
    monkeypatch.setattr(hc_ll.current_platform, "get_device_uuid", _UUIDS.__getitem__)
    # Ranks gather UUIDs, then each rank's (here identical) admission.
    monkeypatch.setattr(
        hc_ll.dist,
        "all_gather_object",
        lambda out, value, group: out.__setitem__(
            slice(None), _UUIDS if isinstance(value, str) else [value] * 4
        ),
    )
    monkeypatch.setattr(
        hc_ll.ops,
        "sm70_ring_native_peer_atomics",
        lambda src, dst: case != "atomic",
    )
    allocate = Mock(return_value=None)
    monkeypatch.setattr(hc_ll.Sm70HcLLCommunicator, "__init__", allocate)
    owners = [
        hc_ll.Sm70HcLLCommunicator.create("tp", torch.device("cuda", rank), rank)
        for rank in range(4)
    ]
    assert [owner is not None for owner in owners] == [case == "full_mesh"] * 4


@pytest.mark.parametrize("method", ["dflash", "mtp"])
def test_request_scope_and_compilation_identity(settings, method):
    settings.speculative_config.method = method
    assert car._sm70_hc_ll_requested()
    settings.scheduler_config.async_scheduling = True
    assert not car._sm70_hc_ll_requested()

    config = settings.kernel_config
    disabled = config.compute_hash()
    config.sm70_hc_ll_async = True
    assert car._sm70_hc_ll_requested()
    assert config.compute_hash() == disabled
    config.sm70_nvlink_hc_active = True
    assert config.compute_hash() != disabled
    config.sm70_nvlink_hc_active = False
    assert config.compute_hash() == disabled


def test_lossless_tp4_packs_keep_checkpoint_weights():
    down = torch.arange(336, dtype=torch.float16)[:, None].expand(-1, 10240)
    up = (torch.arange(10240) // 640).to(torch.float16)[:, None].expand(-1, 320)
    for rank in range(4):
        d = sm70_fp16_hc._pack_hc_batch_weight(down, "down", rank)
        u = sm70_fp16_hc._pack_hc_batch_weight(up, "up", rank)
        unpacked = d.permute(0, 3, 1, 2, 4).reshape(96, 10240)
        assert torch.equal(unpacked[:88], down[80 * rank : 80 * rank + 88])
        assert not unpacked[88:].count_nonzero()
        assert u[0, 0, 0, :, 0, 0].tolist() == [rank + 4 * i for i in range(4)]


@pytest.mark.parametrize("rows,use_ll", [(5, True), (8, True), (40, True), (65, False)])
def test_partial_is_reduced_once_with_legacy_fallback(monkeypatch, rows, use_ll):
    residual = torch.randn((rows, 10240), dtype=torch.float16)
    partial = torch.randn((rows, 2560), dtype=torch.float16)
    injection = torch.zeros((rows, 4), dtype=torch.float16)
    weight = torch.zeros(10240, dtype=torch.float16)
    eps = 1e-6

    def baseline(block, res, inject, w, eps, secondary=None):
        return hc.hc_combine_norm(res, (block * 4).half(), inject, w, eps, 4)

    reduce = Mock(side_effect=lambda block: block * 4)
    native = Mock(side_effect=baseline)
    monkeypatch.setattr(hc, "tensor_model_parallel_all_reduce", reduce)
    monkeypatch.setattr(
        hc,
        "get_tp_hc_ll",
        lambda: SimpleNamespace(device=partial.device, combine_norm=native),
    )
    result = hc._hc_ll_combine_norm(partial, residual, injection, weight, eps, 4)
    ref = baseline(partial, residual, injection, weight, eps)
    for output, expected in zip(result, ref):
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert native.call_count == int(use_ll)
    assert reduce.call_count == int(not use_ll)


@pytest.mark.parametrize(
    "method,widths,expected",
    [
        ("dflash", [8, 8, 8, 4, 5], [8, 16, 24, 16, 40]),
        ("mtp", [5] * 5, [5, 10, 15, 20, 40]),
    ],
)
def test_mrv2_capture_rows_reach_partial_hc(monkeypatch, method, widths, expected):
    import numpy as np

    from vllm.config import CompilationConfig, CUDAGraphMode
    from vllm.models.qwen4_exp.nvidia import model
    from vllm.v1.attention.ops import sm70_e4m3_long
    from vllm.v1.worker.gpu import cudagraph_utils as cg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    monkeypatch.setattr(cg, "get_pp_group", Mock())
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(cg, "_worker_device_is_pre_ampere", lambda: False)
    monkeypatch.setattr(sm70_e4m3_long, "long_attention_enabled", lambda: False)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(tensor_parallel_size=4, data_parallel_size=1),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=sorted(set(expected)),
            max_cudagraph_capture_size=max(expected),
        ),
        speculative_config=SimpleNamespace(
            method=method,
            ngram_assist=False,
            num_speculative_tokens=widths[0] - 1,
            num_speculative_tokens_per_batch_size=(
                [(1, 3, 7), (4, 4, 3), (5, 8, 4)] if method == "dflash" else None
            ),
        ),
    )
    manager = cg.ModelCudaGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, widths[0]
    )
    manager._graphs_captured = True
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={str(i): i for i in range(8)},
        num_computed_prefill_tokens=np.full(8, 800),
        prefill_len=SimpleNamespace(np=np.full(8, 800)),
    )
    use_ll = Mock(side_effect=lambda p, r, i, w, e, s=None: (r, r))
    monkeypatch.setattr(
        hc,
        "get_tp_hc_ll",
        lambda: SimpleNamespace(device=torch.device("cpu"), combine_norm=use_ll),
    )
    mlp_hc = hc.GatedResidual.__new__(hc.GatedResidual)
    torch.nn.Module.__init__(mlp_hc)
    mlp_hc.hc_count = 4
    mlp_hc.config = SimpleNamespace(rms_norm_eps=1e-6)
    mlp_hc.hc_norm = SimpleNamespace(weight=torch.zeros(10240, dtype=torch.float16))
    monkeypatch.setattr(
        mlp_hc, "_project", lambda x: (x[:, :2560], x[:, :4].contiguous())
    )
    layer = model.Qwen4ExpDecoderLayer.__new__(model.Qwen4ExpDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.ple = None
    layer.aux_hidden_state_tap = False
    layer._sm70_hc_ll_attn_partial = True
    layer.attn_hyper_connection = SimpleNamespace(
        mix=lambda x: (x, x[:, :2560], x[:, :4].contiguous())
    )
    layer.mlp_hyper_connection = mlp_hc
    layer.mlp = torch.nn.Identity()
    for count, width, rows in zip((1, 2, 3, 4, 8), widths, expected, strict=True):
        schedule = SimpleNamespace(
            num_scheduled_tokens=dict.fromkeys(map(str, range(count)), width),
            total_num_scheduled_tokens=count * width,
        )
        uniform = runner._get_uniform_decode_token_count(schedule, False)
        descriptor = manager.dispatch(count, count * width, uniform)
        assert descriptor.cg_mode == CUDAGraphMode.FULL
        assert descriptor.num_tokens == rows
        # DFlash2 verifier and MTP4 verifier use the same decoder boundary.
        # Cover both an in-place GDN and a QSA output projection.
        layer.layer_type = "linear_attention" if count % 2 else "full_attention"
        layer.linear_attn = lambda hidden_states, output: output.copy_(hidden_states)
        layer.self_attn = lambda hidden_states, **kwargs: hidden_states
        use_ll.reset_mock()
        layer(
            torch.zeros((descriptor.num_tokens, 10240), dtype=torch.float16),
            None,
            None,
            torch.arange(rows),
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        use_ll.assert_called_once()
        assert use_ll.call_args.args[0].shape == (rows, 2560)


def test_ll_projection_uses_tp_pack_with_legacy_route_intact(monkeypatch):
    x = torch.zeros((8, 10240), dtype=torch.float16)
    down = torch.zeros((336, 10240), dtype=torch.float16)
    up = torch.zeros((10240, 320), dtype=torch.float16)
    down_pack = sm70_fp16_hc._pack_hc_batch_weight(down, "down", 0)
    up_pack = sm70_fp16_hc._pack_hc_batch_weight(up, "up", 0)
    output = (torch.ones((8, 2560)), torch.ones((8, 4)))
    project = Mock(return_value=output)
    get_ll = Mock(return_value=SimpleNamespace(device=x.device, project=project))
    monkeypatch.setattr(sm70_fp16_hc, "get_tp_hc_ll", get_ll)
    run = torch.ops.vllm.qwen38_sm70_fp16_fused_hc
    block, injection = run(x, down, up, down_pack, up_pack)
    assert not block.count_nonzero() and not injection.count_nonzero()
    get_ll.assert_not_called()
    block, injection = run(x, down, up, down_pack, up_pack, False, False, True)
    project.assert_called_once_with(x, down_pack, up_pack)
    assert torch.equal(block, output[0]) and torch.equal(injection, output[1])


def test_moe_tail_passes_one_rank_local_half_sum(monkeypatch):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    shape = (8, 2560)
    shared, routed = (
        torch.full(shape, value, dtype=torch.float16) for value in (1, 2**-12)
    )
    runner = Mock(spec=moe_runner.MoERunner)
    runner.layer_name = "cpu_moe"
    runner.apply_routed_input_transform.return_value = (routed, routed)
    runner._maybe_pad_hidden_states.return_value = (routed, shape[-1])
    runner._forward_entry = Mock(return_value=(shared, routed))
    runner._maybe_reduce_shared_expert_output.return_value = shared
    runner._maybe_apply_routed_scale_to_output.return_value = (shared, routed)
    runner.apply_routed_output_transform.return_value = routed
    runner._maybe_sm70_moe_sum2_allreduce.return_value = shared * 4
    runner._maybe_add_zero_expert_output.side_effect = lambda x: x
    runner._compute_expert_outputs.side_effect = (
        lambda *args: moe_runner.MoERunner._compute_expert_outputs(runner, *args)
    )
    monkeypatch.setattr(moe_runner, "_sm70_dump_moe_runner_tensor", lambda x, *_: x)

    partial = moe_runner.MoERunner.forward(runner, routed, routed, reduce_results=False)
    assert torch.equal(partial, shared)
    runner._maybe_sm70_moe_sum2_allreduce.assert_not_called()
    runner._maybe_reduce_final_output.assert_not_called()

    old_output = moe_runner.MoERunner.forward(runner, routed, routed)
    assert torch.equal(old_output, shared * 4)
    runner._maybe_sm70_moe_sum2_allreduce.assert_called_once()


def _hc_linear(monkeypatch, in_features, out_features, role):
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    layer = ReplicatedLinear(
        in_features,
        out_features,
        bias=False,
        params_dtype=torch.float16,
        return_bias=False,
        disable_tp=True,
    )
    layer.weight.data.copy_(torch.randn((out_features, in_features)) * 0.01)
    layer.cpu_linear = torch.nn.functional.linear
    layer._sm70_qwen38_hc_batch_role = role
    layer.register_buffer(
        "_sm70_qwen38_hc_batch_packed",
        sm70_fp16_hc._pack_hc_batch_weight(layer.weight, role, 0),
        persistent=False,
    )
    return layer


def _final_mixer(monkeypatch):
    return SimpleNamespace(
        use_combine=False,
        hc_count=4,
        input_mix_weight_down=_hc_linear(monkeypatch, 10240, 320, "down"),
        input_mix_weight_up=_hc_linear(monkeypatch, 320, 10240, "up"),
        _sm70_qwen38_fp16_fused_hc_no_combine=True,
    )


def _mock_ll(monkeypatch, rows):
    project = Mock(
        return_value=(
            torch.ones((rows, 2560), dtype=torch.float16),
            torch.ones((rows, 4)),
        )
    )
    monkeypatch.setattr(
        sm70_fp16_hc, "get_tp_hc_ll", lambda: SimpleNamespace(project=project)
    )
    monkeypatch.setattr(sm70_fp16_hc, "use_sm70_decode_graph_semantics", lambda: True)
    return project


def _generic_final_mix(x, down_weight, up_weight):
    lora = hc.hc_silu(torch.nn.functional.linear(x, down_weight), 4)
    return hc.hc_gate_mix(x, torch.nn.functional.linear(lora, up_weight), 4)


@pytest.mark.parametrize(
    "rows,hc_sp,use_ll", [(64, False, True), (65, False, False), (4, True, False)]
)
def test_final_mixer_uses_ll_projection(monkeypatch, rows, hc_sp, use_ll):
    x = torch.randn((rows, 10240)).half()
    module = _final_mixer(monkeypatch)
    down, up = module.input_mix_weight_down, module.input_mix_weight_up
    project = _mock_ll(monkeypatch, rows)

    block, injection = hc.GatedResidual._project(module, x, hc_sp=hc_sp)
    assert injection is None
    if use_ll:
        project.assert_called_once()
        args = project.call_args.args
        assert torch.equal(args[0], x)
        assert args[1] is down._sm70_qwen38_hc_batch_packed
        assert args[2] is up._sm70_qwen38_hc_batch_packed
        assert torch.equal(block, project.return_value[0])
        return
    project.assert_not_called()
    torch.testing.assert_close(
        block, _generic_final_mix(x, down.weight, up.weight), rtol=0, atol=0
    )


def test_final_mixer_keeps_lora_wrapped_projection(monkeypatch):
    x = torch.randn((4, 10240)).half()
    module = _final_mixer(monkeypatch)
    base = module.input_mix_weight_up
    module.input_mix_weight_up = ReplicatedLinearWithLoRA(base)
    # Stand-in for the adapter update; the wrapper's own forward must run.
    module.input_mix_weight_up.apply = (
        lambda x, bias=None: torch.nn.functional.linear(x, base.weight) + 1
    )
    project = _mock_ll(monkeypatch, 4)

    block, injection = hc.GatedResidual._project(module, x)
    project.assert_not_called()
    assert injection is None
    down_weight = module.input_mix_weight_down.weight
    lora = hc.hc_silu(torch.nn.functional.linear(x, down_weight), 4)
    gate = torch.nn.functional.linear(lora, base.weight) + 1
    torch.testing.assert_close(block, hc.hc_gate_mix(x, gate, 4), rtol=0, atol=0)


def test_direct_reload_refreshes_final_mixer_ll_packs(monkeypatch):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    mixer = torch.nn.Module()
    mixer.input_mix_weight_down = _hc_linear(monkeypatch, 10240, 320, "down")
    mixer.input_mix_weight_up = _hc_linear(monkeypatch, 320, 10240, "up")
    model = torch.nn.Module()
    model.mixer = mixer
    # CPU stand-in for the CUDA-only pack preparation (rank 0 shard).
    monkeypatch.setattr(
        sm70_fp16_hc,
        "_prepare_hc_batch_weight",
        lambda layer: layer.register_buffer(
            "_sm70_qwen38_hc_batch_packed",
            sm70_fp16_hc._pack_hc_batch_weight(
                layer.weight, layer._sm70_qwen38_hc_batch_role, 0
            ),
            persistent=False,
        ),
    )
    layers = (mixer.input_mix_weight_down, mixer.input_mix_weight_up)
    packed = [layer._sm70_qwen38_hc_batch_packed for layer in layers]
    new = [torch.randn_like(layer.weight) * 0.01 for layer in layers]
    runner = SimpleNamespace(
        get_model=lambda: model, model_config=SimpleNamespace(quantization=None)
    )

    GPUModelRunner.reload_weights(
        runner,
        weights_iterator=[
            ("mixer.input_mix_weight_down.weight", new[0]),
            ("mixer.input_mix_weight_up.weight", new[1]),
        ],
        is_checkpoint_format=False,
    )
    for layer, old, weight, role in zip(layers, packed, new, ("down", "up")):
        assert layer._sm70_qwen38_hc_batch_packed is old
        assert torch.equal(old, sm70_fp16_hc._pack_hc_batch_weight(weight, role, 0))
