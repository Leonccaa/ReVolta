# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.hy_v4.nvidia.hc import (
    HYV4HCHeadLayer,
    HYV4HCPostLayer,
    HYV4HCPreLayer,
)
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig

pytestmark = [pytest.mark.skip_global_cleanup, pytest.mark.usefixtures("tp1")]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_eager_ihc_against_transformers(dtype):
    ref = pytest.importorskip("transformers.models.hy_v4_flash.modeling_hy_v4_flash")
    torch.manual_seed(14)
    config = HYV4FlashConfig(hidden_size=32)
    pre = HYV4HCPreLayer(config, config.hidden_size)
    post = HYV4HCPostLayer(config)
    head = HYV4HCHeadLayer(config, config.hidden_size)
    for module in (pre, head):
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
    ref_pre = ref.HYV4FlashHyperConnection(config)
    ref_pre.fn.copy_(pre.hc_fn.weight)
    ref_pre.base.copy_(pre.hc_base)
    ref_pre.scale.copy_(pre.hc_scale)
    ref_head = ref.HYV4FlashHyperHead(config)
    ref_head.hc_fn.copy_(head.hc_head_fn.weight)
    ref_head.hc_base.copy_(head.hc_head_base)
    ref_head.hc_scale.copy_(head.hc_head_scale)

    streams = torch.randn(7, 4, config.hidden_size, dtype=dtype)
    sublayer = torch.randn(7, config.hidden_size, dtype=dtype)
    expected_gates, expected_pre = ref_pre(streams.unsqueeze(0))
    mixed, gates = pre(streams)
    assert gates.dtype == torch.float32
    torch.testing.assert_close(mixed, expected_pre.squeeze(0), atol=0, rtol=0)
    torch.testing.assert_close(gates, expected_gates.squeeze(0), atol=0, rtol=0)
    expected_post = (
        expected_gates.squeeze(0).unsqueeze(-1) * sublayer.float().unsqueeze(-2)
        + streams.float()
    ).to(dtype)
    scattered = post(sublayer, streams, gates)
    torch.testing.assert_close(scattered, expected_post, atol=0, rtol=0)
    torch.testing.assert_close(
        head(scattered), ref_head(expected_post.unsqueeze(0)).squeeze(0), atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [0, 1, 7, 128])
@torch.inference_mode()
def test_triton_ihc_matches_eager(tokens):
    from vllm.models.hy_v4.nvidia.hc import triton_ihc_supported

    torch.manual_seed(31)
    config = HYV4FlashConfig()
    with torch.device("cuda"):
        pre = HYV4HCPreLayer(config, config.hidden_size)
        post = HYV4HCPostLayer(config)
        pre.hc_fn.weight.normal_(std=0.006)
        x = torch.randn(tokens, 4, config.hidden_size, dtype=torch.float16)
    assert triton_ihc_supported(x)
    mixed, gates = pre(x)
    expected, expected_gates = pre.forward_native(x)
    torch.testing.assert_close(mixed, expected, atol=2e-4, rtol=1e-3)
    torch.testing.assert_close(gates, expected_gates, atol=2e-6, rtol=2e-6)
    scattered = post(mixed, x, gates)
    torch.testing.assert_close(
        scattered, post.forward_native(mixed, x, gates), atol=2e-4, rtol=1e-3
    )
