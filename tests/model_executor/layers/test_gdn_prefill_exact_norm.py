# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers import layernorm
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


@pytest.fixture
def projection(monkeypatch, default_vllm_config):
    # Exercise the SM70 selection on CPU, with the native CUDA launch replaced.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(layernorm, "_sm70_gated_norm_device_supported", lambda _: True)
    monkeypatch.setattr(layernorm.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(layernorm.envs, "VLLM_SM70_QWEN38_WIDE_M_GATED_NORM", False)
    monkeypatch.setattr(
        torch.ops._C,
        "sm70_rmsnorm_gated_exact_out",
        lambda out, *_: out.fill_(2),
        raising=False,
    )
    norm = layernorm.RMSNormGated(
        128, device=torch.device("cpu"), dtype=torch.float16, norm_before_gate=True
    )
    norm._sm70_rmsnorm_gated_exact = True
    # Preserve the callable used by the disabled RMSNormGated custom-op policy.
    norm._forward_method = norm.forward_native
    monkeypatch.setattr(norm, "forward_cuda", lambda x, z: torch.full_like(x, 3))
    layer = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        norm=norm,
        out_proj=lambda x: (x, None),
        enable_sm70_dflash2_fused_gdn_norm=False,
        enable_sm70_gdn_rmsnorm_onepass=False,
        _sm70_mtp_prefill_norm=True,
    )
    layer_type = gdn.QwenGatedDeltaNetAttention
    layer._full_forward = MethodType(layer_type._full_forward, layer)
    project = MethodType(layer_type._compute_output_projection, layer)

    def forward(x, out):
        out.copy_(project(x, x, x.size(0)))

    layer._forward_method = forward
    return layer, project


def forward_context(layer, tokens, prefills):
    metadata = GDNAttentionMetadata(
        num_prefills=prefills,
        num_prefill_tokens=tokens if prefills else 0,
        num_decodes=0 if prefills else tokens,
        num_decode_tokens=0 if prefills else tokens,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=tokens,
    )
    return override_forward_context(
        ForwardContext({layer.prefix: layer}, {layer.prefix: metadata}, {})
    )


@pytest.mark.parametrize(
    "tokens,wide,prefills,method,exact,batch_invariant,expected",
    [
        pytest.param(128, False, 1, "mtp", True, False, 2, id="prefill"),
        pytest.param(16, False, 1, "mtp", True, False, 2, id="old-small-path"),
        pytest.param(48, True, 1, "mtp", True, False, 2, id="old-wide-path"),
        pytest.param(49, True, 1, "mtp", True, False, 2, id="prefill-wide"),
        pytest.param(128, False, 0, "mtp", True, False, 0, id="decode"),
        pytest.param(128, False, 1, "dflash", True, False, 3, id="dflash"),
        pytest.param(128, False, 1, "none", True, False, 0, id="no-mtp"),
        pytest.param(128, False, 1, "mtp", False, False, 0, id="exact-disabled"),
        pytest.param(128, False, 1, "mtp", True, True, 0, id="batch-invariant"),
    ],
)
def test_opaque_gdn_prefill_norm(
    projection,
    monkeypatch,
    tokens,
    wide,
    prefills,
    method,
    exact,
    batch_invariant,
    expected,
):
    layer, _ = projection
    layer._sm70_mtp_prefill_norm = method == "mtp"
    layer.enable_sm70_dflash2_fused_gdn_norm = method == "dflash"
    layer.norm._sm70_rmsnorm_gated_exact = exact
    monkeypatch.setattr(layernorm.envs, "VLLM_SM70_QWEN38_WIDE_M_GATED_NORM", wide)
    monkeypatch.setattr(layernorm.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    x = torch.zeros(tokens, 12, 128, dtype=torch.float16)
    out = torch.empty(tokens, 12 * 128, dtype=x.dtype)
    with forward_context(layer, tokens, prefills):
        torch.ops.vllm.qwen_gdn_full_forward(
            x, out, x.new_empty(0), x.new_empty(0), _encode_layer_name(layer.prefix)
        )
    torch.testing.assert_close(out, torch.full_like(out, expected))


def test_traced_gdn_projection_keeps_native_norm(projection):
    layer, project = projection
    x = torch.zeros(128, 12, 128, dtype=torch.float16)
    with forward_context(layer, 128, 1):
        compiled = torch.compile(project, backend="eager", fullgraph=True)
        out = compiled(x, x, 128)
    torch.testing.assert_close(out, torch.zeros_like(out))
