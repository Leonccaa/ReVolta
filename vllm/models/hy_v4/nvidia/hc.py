# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""iHC (independent Hyper-Connections) layers for HY V4 (NVIDIA).

iHC replaces the single residual stream of a standard transformer with
`hc_mult` parallel residual channels. Each decoder sub-block reduces the
channels to one hidden state (`HYV4HCPreLayer`), runs the sub-block, then
scatters the result back over the channels (`HYV4HCPostLayer`). The final
`HYV4HCHeadLayer` merges the channels before the model's output norm.

"""

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm import envs
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

# Triton iHC kernels from vLLM, adapted from SGLang's HY V4 implementation:
# https://github.com/sgl-project/sglang/pull/36805
_BLOCK_K = 1024
_BLOCK_D = 1024


def triton_ihc_supported(x: torch.Tensor) -> bool:
    """Return whether the in-tree Triton path can run for this input."""
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and not envs.VLLM_BATCH_INVARIANT
    )


@triton.jit
def _ihc_pre_stage1(
    x_ptr,
    weight_ptr,
    partial_ptr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    k_offsets = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_TOTAL
    x = tl.load(
        x_ptr + token_idx * K_TOTAL + k_offsets,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    sum_squares = tl.sum(x * x, axis=0)

    weight_offsets = hc_idx[:, None] * K_TOTAL + k_offsets[None, :]
    weight_mask = hc_mask[:, None] & k_mask[None, :]
    pre_mix = tl.sum(
        tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0) * x[None, :],
        axis=1,
    )
    post_mix = tl.sum(
        tl.load(
            weight_ptr + HC_MULT * K_TOTAL + weight_offsets,
            mask=weight_mask,
            other=0.0,
        )
        * x[None, :],
        axis=1,
    )

    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, sum_squares)
    tl.store(partial + 1 + hc_idx, pre_mix, mask=hc_mask)
    tl.store(partial + 1 + HC_POW2 + hc_idx, post_mix, mask=hc_mask)


@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    sum_squares = tl.zeros((), dtype=tl.float32)
    pre_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    post_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        partial = partial_row + split_idx * PARTIAL_STRIDE
        sum_squares += tl.load(partial)
        pre_mix += tl.load(partial + 1 + hc_idx, mask=hc_mask, other=0.0)
        post_mix += tl.load(
            partial + 1 + HC_POW2 + hc_idx,
            mask=hc_mask,
            other=0.0,
        )

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(
            post_ptr + token_idx * HC_MULT + hc_idx,
            post,
            mask=hc_mask,
        )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel

    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    post = tl.load(
        post_ptr + token_idx * HC_MULT + hc_idx,
        mask=hc_mask,
        other=0.0,
    )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x = tl.load(
        x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    residual_row = residual_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    output_row = output_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    for channel_idx in tl.static_range(HC_MULT):
        residual = tl.load(
            residual_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, post, 0.0), axis=0)
        output = channel_gate * x + residual
        tl.store(
            output_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=hidden_mask,
        )


def triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce iHC channels and produce the post gates."""
    assert x.dim() == 3
    assert x.is_cuda and weight.is_cuda and scale.is_cuda and base.is_cuda
    assert weight.dtype == torch.float32
    assert scale.dtype == torch.float32 and base.dtype == torch.float32

    x = x.contiguous()
    assert weight.is_contiguous()
    scale = scale.contiguous()
    base = base.contiguous()
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    assert weight.shape == (2 * hc_mult, k_total)
    assert scale.shape == (2,)
    assert base.shape == (2 * hc_mult,)

    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=x.device),
        )

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    hc_pow2 = triton.next_power_of_2(hc_mult)
    num_splits = triton.cdiv(k_total, _BLOCK_K)
    partial_stride = 1 + 2 * hc_pow2
    partial = torch.empty(
        (num_tokens, num_splits, partial_stride),
        dtype=torch.float32,
        device=x.device,
    )

    _ihc_pre_stage1[(num_tokens, num_splits)](
        x,
        weight,
        partial,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        BLOCK_K=_BLOCK_K,
        PARTIAL_STRIDE=partial_stride,
        num_warps=8,
        enable_fp_fusion=False,
    )
    _ihc_pre_stage2[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        x,
        partial,
        scale,
        base,
        output,
        post,
        HIDDEN_SIZE=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        PARTIAL_STRIDE=partial_stride,
        BLOCK_D=_BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, post


def triton_ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Scatter a sub-block output back over the iHC residual channels."""
    assert x.dim() == 2
    assert x.is_cuda and residual.is_cuda and post.is_cuda
    assert post.dtype == torch.float32

    x = x.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    num_tokens, hidden_size = x.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert post.shape == (num_tokens, hc_mult)

    if num_tokens == 0:
        return torch.empty((0, hc_mult, hidden_size), dtype=x.dtype, device=x.device)

    output = torch.empty(
        (num_tokens, hc_mult, hidden_size), dtype=x.dtype, device=x.device
    )
    _ihc_post_kernel[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        x,
        residual,
        post,
        output,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        BLOCK_D=_BLOCK_D,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


class HYV4HCPreLayer(nn.Module):
    """iHC pre-processing layer (2D-activation adaptation).

    Steps:
        1. RMS-normalize the flattened `[num_tokens, hc * d]` input.
        2. Project to the pre/post gating logits.
        3. Turn the logits into sigmoid gates.
        4. Reduce over the channel dim with the pre gates.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        hidden_dim: int,
        hc_mult: int = 4,
        magnitude: float = 2.0,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        hc_eps: float = 1e-6,
        layernorm_epsilon: float = 1e-5,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.hc_mult = hc_mult
        self.magnitude = magnitude
        self.hc_eps = hc_eps
        self.layernorm_epsilon = layernorm_epsilon
        hc_in_dim = hc_mult * hidden_dim
        mix_hc = 2 * hc_mult  # pre + post only, no comb

        self.hc_fn = ReplicatedLinear(
            input_size=hc_in_dim,
            output_size=mix_hc,
            params_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.hc_fn",
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))

        self.reset_parameters(init_std, base_noise_std)

    def reset_parameters(self, init_std: float, base_noise_std: float = 0.0) -> None:
        """Initialize the gate scale and per-channel gate bias."""
        del init_std  # hc_fn is initialized by ReplicatedLinear
        nn.init.constant_(self.hc_scale, 0.01)
        with torch.no_grad():
            self.hc_base[: self.hc_mult].fill_(
                -torch.log(torch.tensor(self.hc_mult - 1.0, dtype=self.hc_base.dtype))
            )
            self.hc_base[self.hc_mult : 2 * self.hc_mult].fill_(0.0)
            if base_noise_std > 0.0:
                self.hc_base.add_(torch.randn_like(self.hc_base) * base_noise_std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce the iHC channels and emit the post gates.

        Args:
            x: Input of shape `[num_tokens, hc, d]`.

        Returns:
            A tuple of the pre-gated reduction `[num_tokens, d]` and the post
            gates `[num_tokens, hc]` consumed by `HYV4HCPostLayer`.

        """
        if triton_ihc_supported(x):
            return triton_ihc_pre(
                x,
                self.hc_fn.weight,
                self.hc_scale,
                self.hc_base,
                self.magnitude,
                self.hc_eps,
                self.layernorm_epsilon,
            )
        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.size()  # [num_tokens, hc, d]
        hc = self.hc_mult
        hc_eps = self.hc_eps

        x_flat = x.flatten(1).float()  # [num_tokens, hc*d]

        rsqrt = torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.layernorm_epsilon
        )
        mixes = self.hc_fn(x_flat)[0] * rsqrt  # [num_tokens, 2*hc]

        pre_raw = mixes[..., :hc]
        post_raw = mixes[..., hc : 2 * hc]

        pre = (
            torch.sigmoid(
                pre_raw * self.hc_scale[0].float() + self.hc_base[:hc].float()
            )
            + hc_eps
        )
        post = (
            self.magnitude
            * torch.sigmoid(
                post_raw * self.hc_scale[1].float() + self.hc_base[hc : 2 * hc].float()
            )
            + hc_eps
        )

        y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)  # [num_tokens, d]
        return y.to(x.dtype), post


class HYV4HCPostLayer(nn.Module):
    """iHC post-processing layer (2D-activation adaptation).

    Applies post-gating to the sub-block output and adds the multi-channel
    residual (no comb mixing)::

        y[n, i, d] = post[n, i] * x[n, d] + residual[n, i, d]
    """

    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.config = config

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
    ) -> torch.Tensor:
        """Scatter the sub-block output back onto the iHC channels.

        Args:
            x: Attention/MLP output of shape `[num_tokens, d]`.
            residual: Multi-channel residual `[num_tokens, hc, d]`.
            post: Post gates `[num_tokens, hc]` from `HYV4HCPreLayer`.

        Returns:
            The updated residual channels `[num_tokens, hc, d]`.

        """
        if triton_ihc_supported(x):
            return triton_ihc_post(x, residual, post)
        return self.forward_native(x, residual, post)

    def forward_native(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        residual = residual.float()
        post = post.float()

        post_gated = post.unsqueeze(-1) * x.unsqueeze(-2)  # [num_tokens, hc, d]
        y = post_gated + residual
        return y.to(dtype)


class HYV4HCHeadLayer(nn.Module):
    """iHC head layer (2D-activation adaptation).

    Merges the iHC channels back into a single hidden state before the final
    layer norm, using an RMS-normed projection plus sigmoid-gated reduction.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        hidden_size: int,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_head_fn = ReplicatedLinear(
            input_size=hc_mult * hidden_size,
            output_size=hc_mult,
            params_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.hc_head_fn",
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.reset_parameters(init_std, base_noise_std)

    def reset_parameters(
        self, init_std: float = 6e-3, base_noise_std: float = 0.0
    ) -> None:
        """Initialize the head gate scale and per-channel gate bias."""
        del init_std  # hc_head_fn is initialized by ReplicatedLinear
        nn.init.constant_(self.hc_head_scale, 0.01)
        with torch.no_grad():
            self.hc_head_base.fill_(
                -torch.log(
                    torch.tensor(self.hc_mult - 1.0, dtype=self.hc_head_base.dtype)
                )
            )
            if base_noise_std > 0.0:
                self.hc_head_base.add_(
                    torch.randn_like(self.hc_head_base) * base_noise_std
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge the iHC channels into a single hidden state.

        Args:
            x: Input of shape `[num_tokens, hc, d]`.

        Returns:
            The merged hidden state `[num_tokens, d]`.

        """
        shape, x_dtype = x.size(), x.dtype

        x = x.flatten(1).float()  # [num_tokens, hc*d]
        rsqrt = torch.rsqrt(
            x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        mixes = self.hc_head_fn(x)[0] * rsqrt  # [num_tokens, hc]
        pre = (
            torch.sigmoid(
                mixes * self.hc_head_scale.float() + self.hc_head_base.float()
            )
            + self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)  # [num_tokens, d]
        return y.to(x_dtype)


class HYV4HCLayer(nn.Module):
    """Wrapper owning one iHC boundary (pre + post) of a decoder sub-block."""

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_ihc = getattr(config, "enable_ihc", False)
        if self.enable_ihc:
            self.hc_pre = HYV4HCPreLayer(
                config,
                config.hidden_size,
                config.hc_mult,
                config.hc_magnitude,
                init_std,
                base_noise_std,
                config.hc_eps,
                config.rms_norm_eps,
                prefix=f"{prefix}.hc_pre",
            )
            self.hc_post = HYV4HCPostLayer(config)

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the sub-block input to 3D when iHC is enabled."""
        if not self.enable_ihc:
            return hidden_states
        return self._prepare_input_to_3d(hidden_states)

    def _prepare_input_to_3d(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reshape the iHC input to `[num_tokens, hc, h]`.

        Accepted inputs are `[num_tokens, hc, h]` (no-op), `[num_tokens, h]`
        (broadcast over the channels) and `[num_tokens, hc * h]` (reshape).
        """
        if hidden_states.dim() == 3:
            return hidden_states
        if hidden_states.dim() != 2:
            raise RuntimeError(
                f"HC expects a 2D/3D tensor, got shape={tuple(hidden_states.shape)}"
            )

        d0, d1 = hidden_states.shape
        h = self.config.hidden_size
        hc = self.config.hc_mult

        if d1 == h:
            return hidden_states.unsqueeze(1).repeat(1, hc, 1)

        expected = hc * h
        if d1 == expected:
            return hidden_states.reshape(d0, hc, h)

        raise RuntimeError(
            f"HC expects last dim to be hidden_size ({h})"
            f"or hc_mult*hidden_size ({expected}), got {d1}. "
            f"hc_mult={hc}, hidden_size={h}."
        )

    def pre(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Reduce the iHC channels and produce the post gates.

        Returns:
            A tuple of the reduced hidden states `[num_tokens, d]`, the post
            gates `[num_tokens, hc]` (`None` when iHC is disabled) and the
            residual (the untouched input).

        """
        if not self.enable_ihc:
            return hidden_states, None, hidden_states
        reduced, post_gates = self.hc_pre(hidden_states)
        return reduced, post_gates, hidden_states

    def post(
        self,
        output_with_bias: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply post-gating and add the residual."""
        if not self.enable_ihc:
            return output_with_bias + residual
        assert post_gates is not None
        return self.hc_post(output_with_bias, residual, post_gates)
