# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DFlash2 next-round native GDN deferred commit (q=8)."""

from __future__ import annotations

from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import _resolve_gdn_spec_state_replay
from vllm.model_executor.layers.mamba.gdn.spec_replay import TMAX


def _make_vllm_config(
    *,
    method: str = "dflash",
    num_spec_tokens: int = 7,
    deferred: bool | None = True,
    async_scheduling: bool = True,
    fused_gdn_verify: bool | None = True,
    explicit_fused: bool = False,
    spec_state_replay: bool | None = None,
    tp_size: int = 4,
    dtype: torch.dtype = torch.float16,
    ssm_dtype: str = "float32",
    quantization: str = "awq",
    long_context: bool = False,
    kv_connector: str | None = None,
):
    policy = Sm70DFlash2Config(
        fused_gdn_verify=fused_gdn_verify,
        spec_state_replay=spec_state_replay,
    )
    policy.resolved = True
    policy.explicit_fields = ("fused_gdn_verify",) if explicit_fused else ()
    spec = NS(
        method=method,
        gdn_deferred_commit_metadata=deferred,
        gdn_state_replay=None,
        sm70_dflash2=policy,
        num_speculative_tokens=num_spec_tokens,
        use_qwen4_exp_mtp=lambda: method == "mtp",
    )
    spec.use_dflash = MethodType(SpeculativeConfig.use_dflash, spec)
    spec.use_dflash_ddtree = MethodType(SpeculativeConfig.use_dflash_ddtree, spec)
    spec.num_speculative_state_tokens = MethodType(
        SpeculativeConfig.num_speculative_state_tokens, spec
    )
    return NS(
        speculative_config=spec,
        model_config=NS(
            is_hybrid=True,
            dtype=dtype,
            quantization=quantization,
            architecture="Qwen4ExpForConditionalGeneration",
            hf_text_config=NS(linear_key_head_dim=128, linear_value_head_dim=128),
        ),
        cache_config=NS(
            mamba_cache_mode="align",
            mamba_ssm_cache_dtype=ssm_dtype,
        ),
        scheduler_config=NS(async_scheduling=async_scheduling),
        kv_transfer_config=NS(kv_connector=kv_connector) if kv_connector else None,
        parallel_config=NS(pipeline_parallel_size=1, tensor_parallel_size=tp_size),
        kernel_config=NS(long_context_kv=NS(enabled=long_context)),
        use_v2_model_runner=True,
    )


def _patch_sm70(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr(
        "vllm.config.vllm._any_participating_device_is_capability",
        lambda _cfg, cap: cap == (7, 0),
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, True, id="async-awq"),
        pytest.param({"quantization": "modelopt_fp4"}, False, id="nvfp4"),
        pytest.param({"method": "mtp"}, False, id="mtp"),
        pytest.param({"method": "dflash_ddtree"}, False, id="ddtree"),
        pytest.param({"async_scheduling": False}, False, id="sync"),
        pytest.param({"long_context": True}, False, id="host-kv"),
        pytest.param({"kv_connector": "NixlConnector"}, False, id="kv-connector"),
        pytest.param({"explicit_fused": True}, False, id="explicit-fused-verify"),
        pytest.param({"spec_state_replay": False}, False, id="replay-disabled"),
        pytest.param({"num_spec_tokens": 15}, False, id="unsupported-width"),
    ],
)
def test_gdn_deferred_commit_automatic_policy(monkeypatch, overrides, expected):
    _patch_sm70(monkeypatch)
    cfg = _make_vllm_config(deferred=None, **overrides)
    spec = cfg.speculative_config
    assert _resolve_gdn_spec_state_replay(cfg) is expected
    assert spec.gdn_deferred_commit_metadata is expected
    if expected:
        assert spec.sm70_dflash2.fused_gdn_verify is False


def test_gdn_deferred_commit_explicit_false_overrides_auto(monkeypatch):
    _patch_sm70(monkeypatch)
    cfg = _make_vllm_config(deferred=False)
    spec = cfg.speculative_config
    assert not _resolve_gdn_spec_state_replay(cfg)
    assert spec.gdn_deferred_commit_metadata is False
    assert spec.sm70_dflash2.fused_gdn_verify is True


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_dflash2_q8_deferred_commit_resolves_and_overrides_auto_fused(
    monkeypatch, async_scheduling
):
    _patch_sm70(monkeypatch)
    cfg = _make_vllm_config(
        method="dflash",
        num_spec_tokens=7,
        deferred=True,
        async_scheduling=async_scheduling,
        fused_gdn_verify=True,
        explicit_fused=False,
    )
    spec = cfg.speculative_config
    assert spec.num_speculative_state_tokens() == 7
    spec.gdn_spec_state_replay = _resolve_gdn_spec_state_replay(cfg)
    assert spec.gdn_spec_state_replay is True
    assert spec.sm70_dflash2.fused_gdn_verify is False
    assert SpeculativeConfig.num_speculative_state_blocks(spec) == 0


def test_dflash2_deferred_commit_rejects_explicit_fused_conflict(monkeypatch):
    _patch_sm70(monkeypatch)
    cfg = _make_vllm_config(
        method="dflash",
        num_spec_tokens=7,
        deferred=True,
        fused_gdn_verify=True,
        explicit_fused=True,
    )
    with pytest.raises(ValueError, match="gdn_deferred_commit_metadata"):
        _resolve_gdn_spec_state_replay(cfg)


def test_dflash2_deferred_commit_rejects_explicit_replay_false(monkeypatch):
    _patch_sm70(monkeypatch)
    cfg = _make_vllm_config(
        method="dflash",
        num_spec_tokens=7,
        deferred=True,
        fused_gdn_verify=False,
        spec_state_replay=False,
    )
    with pytest.raises(ValueError, match="spec_state_replay=False"):
        _resolve_gdn_spec_state_replay(cfg)


def test_dflash2_q16_deferred_commit_rejected_above_tmax(monkeypatch):
    _patch_sm70(monkeypatch)
    cfg = _make_vllm_config(
        method="dflash",
        num_spec_tokens=15,
        deferred=True,
        fused_gdn_verify=False,
    )
    with pytest.raises(ValueError, match="gdn_deferred_commit_metadata requires"):
        _resolve_gdn_spec_state_replay(cfg)


def test_dflash_ddtree_deferred_commit_rejects_explicit_opt_in(monkeypatch):
    _patch_sm70(monkeypatch)
    cfg = _make_vllm_config(method="dflash_ddtree")
    with pytest.raises(ValueError, match="requires linear verification"):
        _resolve_gdn_spec_state_replay(cfg)


@pytest.mark.parametrize(
    "architecture,key_dim,value_dim",
    [
        ("FalconH1ForCausalLM", 128, 128),
        ("Qwen4ExpForConditionalGeneration", 64, 128),
        ("Qwen4ExpForConditionalGeneration", 128, 64),
    ],
)
def test_dflash_deferred_commit_requires_native_gdn_layout(
    monkeypatch, architecture, key_dim, value_dim
):
    _patch_sm70(monkeypatch)
    cfg = _make_vllm_config(fused_gdn_verify=False)
    cfg.model_config.architecture = architecture
    cfg.model_config.hf_text_config = NS(
        linear_key_head_dim=key_dim, linear_value_head_dim=value_dim
    )
    with pytest.raises(ValueError, match="requires Qwen GDN with 128 by 128 state"):
        _resolve_gdn_spec_state_replay(cfg)


def _ref_gdn_step(
    state: torch.Tensor,
    qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h: int,
    hv: int,
    scale: float,
):
    """FP32 reference for one sequence of T tokens (matches gdn_verify_sm70.cu)."""
    kd = 128
    t_len = qkv.shape[0]
    q = qkv[:, : h * kd].view(t_len, h, kd).float()
    k = qkv[:, h * kd : 2 * h * kd].view(t_len, h, kd).float()
    v = qkv[:, 2 * h * kd :].view(t_len, hv, kd).float()
    q = q / (q.pow(2).sum(dim=-1, keepdim=True) + 1e-6).sqrt() * scale
    k = k / (k.pow(2).sum(dim=-1, keepdim=True) + 1e-6).sqrt()
    repeat = hv // h
    q = q.repeat_interleave(repeat, dim=1)
    k = k.repeat_interleave(repeat, dim=1)
    g = torch.exp(-torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float()))
    beta = torch.sigmoid(b.float())

    s = state.clone().float()  # [HV, V, K]
    outs = []
    states = []
    for t in range(t_len):
        s = s * g[t, :, None, None]
        kv_mem = (s * k[t, :, None, :]).sum(dim=-1)  # [HV, V]
        delta = (v[t] - kv_mem) * beta[t, :, None]
        s = s + delta[:, :, None] * k[t, :, None, :]
        o_t = (s * q[t, :, None, :]).sum(dim=-1)  # [HV, V]
        outs.append(o_t.to(torch.float16))
        states.append(s.clone())
    return torch.stack(outs, dim=0), states


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)
@pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
@torch.inference_mode()
def test_sm70_gdn_verify_deferred_q8_matches_reference_and_graph(num_reqs):
    if not hasattr(torch.ops._C, "sm70_gdn_verify_out"):
        pytest.skip("requires sm70_gdn_verify_out")

    torch.manual_seed(2026 + num_reqs)
    h, hv, kd = 4, 12, 128
    width = 8  # DFlash2 q=8 (7 drafts + 1 anchor)
    scale = kd**-0.5
    device = torch.device("cuda")

    a_log = torch.randn(hv, dtype=torch.float32, device=device) * 0.2 - 1.0
    dt_bias = torch.randn(hv, dtype=torch.float32, device=device) * 0.2
    num_slots = num_reqs * 2 + 4
    state = (
        torch.randn(num_slots, hv, kd, kd, dtype=torch.float32, device=device) * 0.05
    )
    ref_state = state.clone()

    seq_floats = TMAX * (h * kd + hv * kd + hv * 2)
    rbuf = torch.empty(2 * num_reqs * seq_floats, dtype=torch.float32, device=device)

    slots = torch.arange(num_reqs, dtype=torch.int32, device=device)
    ckpt_slots = torch.arange(num_reqs, 2 * num_reqs, dtype=torch.int32, device=device)
    cu = torch.arange(
        0, (num_reqs + 1) * width, width, dtype=torch.int32, device=device
    )

    # Round 1: fresh verify (R0=0), with mid-round checkpoint at token 3 for req 0
    qkv1 = (
        torch.randn(
            num_reqs * width, (2 * h + hv) * kd, dtype=torch.float16, device=device
        )
        * 0.2
    )
    a1 = torch.randn(num_reqs * width, hv, dtype=torch.float16, device=device) * 0.2
    b1 = torch.randn(num_reqs * width, hv, dtype=torch.float16, device=device) * 0.2
    out1 = torch.empty(num_reqs * width, hv, kd, dtype=torch.float16, device=device)
    rn1 = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    rmap1 = 2 * torch.arange(num_reqs, dtype=torch.int32, device=device)  # side=0
    ckt1 = torch.full((num_reqs,), -1, dtype=torch.int32, device=device)
    ckt1[0] = 3

    ops.sm70_gdn_verify_out(
        qkv1,
        a1,
        b1,
        a_log,
        dt_bias,
        state,
        out1,
        cu,
        slots.unsqueeze(1),
        None,
        h,
        hv,
        scale,
        1,
        rbuf,
        rn1,
        None,
        width,
        rmap1,
        ckt1,
        ckpt_slots,
    )

    # Reference round 1
    accepted1 = [(i % width) + 1 for i in range(num_reqs)]
    ref_post1 = []
    for i in range(num_reqs):
        s0 = ref_state[slots[i]]
        o_ref, st_list = _ref_gdn_step(
            s0,
            qkv1[i * width : (i + 1) * width],
            a1[i * width : (i + 1) * width],
            b1[i * width : (i + 1) * width],
            a_log,
            dt_bias,
            h,
            hv,
            scale,
        )
        max_o = (
            (out1[i * width : (i + 1) * width].float() - o_ref.float())
            .abs()
            .max()
            .item()
        )
        assert max_o < 5e-3, f"round1 out max_err={max_o}"
        if i == 0:
            max_ck = (state[ckpt_slots[0]] - st_list[3]).abs().max().item()
            assert max_ck < 5e-3, f"round1 ckpt max_err={max_ck}"
        ref_post1.append(st_list[accepted1[i] - 1])

    # Round 2: commit the accepted prefix from side 0, then verify qkv2 in graph.
    qkv2 = (
        torch.randn(
            num_reqs * width, (2 * h + hv) * kd, dtype=torch.float16, device=device
        )
        * 0.2
    )
    a2 = torch.randn(num_reqs * width, hv, dtype=torch.float16, device=device) * 0.2
    b2 = torch.randn(num_reqs * width, hv, dtype=torch.float16, device=device) * 0.2
    out2 = torch.empty(num_reqs * width, hv, kd, dtype=torch.float16, device=device)
    rn2 = torch.tensor(accepted1, dtype=torch.int32, device=device)
    rmap2 = (
        2 * torch.arange(num_reqs, dtype=torch.int32, device=device) + 1
    )  # side=1, replays side 0
    ckt2 = torch.full((num_reqs,), -1, dtype=torch.int32, device=device)

    state_before_r2 = state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.sm70_gdn_verify_out(
            qkv2,
            a2,
            b2,
            a_log,
            dt_bias,
            state,
            out2,
            cu,
            slots.unsqueeze(1),
            None,
            h,
            hv,
            scale,
            1,
            rbuf,
            rn2,
            None,
            width,
            rmap2,
            ckt2,
            ckpt_slots,
        )
    state.copy_(state_before_r2)
    graph.replay()

    for i in range(num_reqs):
        max_s = (state[slots[i]] - ref_post1[i]).abs().max().item()
        assert max_s < 5e-3, f"round2 committed state max_err={max_s}"
        o_ref2, _ = _ref_gdn_step(
            ref_post1[i],
            qkv2[i * width : (i + 1) * width],
            a2[i * width : (i + 1) * width],
            b2[i * width : (i + 1) * width],
            a_log,
            dt_bias,
            h,
            hv,
            scale,
        )
        max_o2 = (
            (out2[i * width : (i + 1) * width].float() - o_ref2.float())
            .abs()
            .max()
            .item()
        )
        assert max_o2 < 5e-3, f"round2 out max_err={max_o2}"


def test_replay_row_kernel_does_not_specialize_on_row_counts():
    # A new decode/prefill mix must not compile a new variant during serving.
    from vllm.model_executor.layers.mamba.gdn import spec_replay

    kernel = spec_replay._prepare_replay_rows_kernel
    if not hasattr(kernel, "do_not_specialize"):
        pytest.skip("Triton interpreter does not specialize")
    names = {param.name for param in kernel.params if param.do_not_specialize}
    assert {"num_spec", "num_plain"} <= names
