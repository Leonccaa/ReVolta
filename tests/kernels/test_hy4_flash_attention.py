# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def expanded_attention(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    latent: torch.Tensor,
    k_rope: torch.Tensor,
    kv_b: torch.Tensor,
    visible: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """FP32 oracle: materialize per-head K/V instead of absorbing projections."""
    k_nope, value = torch.einsum("sl,hdl->hsd", latent.float(), kv_b.float()).split(
        [q_nope.shape[-1], kv_b.shape[1] - q_nope.shape[-1]], dim=-1
    )
    q = torch.cat((q_nope, q_rope), dim=-1).float().transpose(0, 1)
    k = torch.cat((k_nope, k_rope.float().expand(k_nope.shape[0], -1, -1)), dim=-1)
    logits = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(~visible, -torch.inf)
    sink = sinks.float()[:, None, None].expand(-1, q.shape[1], 1)
    probs = torch.cat((logits, sink), dim=-1).softmax(dim=-1)[..., :-1]
    return (probs @ value).transpose(0, 1)


def test_expanded_fp32_oracle():
    hf = pytest.importorskip("transformers.models.hy_v4_flash.modeling_hy_v4_flash")
    rng = torch.Generator().manual_seed(17)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=rng)

    q_nope, q_rope = rand(7, 4, 12), rand(7, 4, 8)
    latent, k_rope = rand(7, 16), rand(7, 8)
    kv_b, sinks = rand(4, 22, 16) / 4, rand(4)
    visible = torch.ones(7, 7, dtype=torch.bool).tril()
    visible[4:, 1] = False
    k, v = torch.einsum("sl,hdl->hsd", latent, kv_b).split([12, 10], dim=-1)
    k = torch.cat((k, k_rope.expand(4, -1, -1)), dim=-1)
    q = torch.cat((q_nope, q_rope), dim=-1).transpose(0, 1)
    mask = torch.zeros_like(visible, dtype=torch.float32).masked_fill(
        ~visible, -torch.inf
    )
    actual, _ = hf.eager_attention_forward(
        SimpleNamespace(num_key_value_groups=1, sinks=sinks, training=False),
        q[None],
        k[None],
        v[None],
        mask,
        scaling=20**-0.5,
    )
    torch.testing.assert_close(
        actual[0],
        expanded_attention(q_nope, q_rope, latent, k_rope, kv_b, visible, sinks),
    )
