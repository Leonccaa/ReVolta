# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

from typing import NamedTuple

import torch
from torch import nn

from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.sm70_hc_ll import MAX_ROWS, get_tp_hc_ll
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
    maybe_apply_fused_hc,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.utils.torch_utils import direct_register_custom_op

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from .sm70_fp16_hc import maybe_apply_qwen38_sm70_fp16_fused_hc
from .sm70_hcx import HD, INJ
from .sm70_hcx import MAX_ROWS as HCX_MAX_ROWS


class HCTensorParallelPartial(NamedTuple):
    """A block output whose one remaining TP sum is owned by HC."""

    value: torch.Tensor


def _hc_ll_combine_norm(
    partial: torch.Tensor,
    residual: torch.Tensor,
    injection: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Layers hand over partials only after the HC LL geometry admission.
    ll = get_tp_hc_ll()
    if ll is not None and 1 <= residual.shape[0] <= MAX_ROWS:
        return ll.combine_norm(partial, residual, injection, weight, eps)
    return hc_combine_norm(
        residual,
        tensor_model_parallel_all_reduce(partial),
        injection,
        weight,
        eps,
        hc_count,
    )


def _hc_ll_combine_norm_fake(partial, residual, injection, weight, eps, hc_count):
    return (
        torch.empty_like(residual, memory_format=torch.contiguous_format),
        torch.empty_like(residual, memory_format=torch.contiguous_format),
    )


direct_register_custom_op(
    op_name="qwen4exp_sm70_hc_ll_combine_norm",
    op_func=_hc_ll_combine_norm,
    fake_impl=_hc_ll_combine_norm_fake,
)


def _hc_hcx_combine_and_mix(
    partial: torch.Tensor,
    residual: torch.Tensor,
    injection: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx = get_forward_context()
    module = ctx.no_compile_layers[name]
    norm = module.hc_norm.weight
    eps = module.config.rms_norm_eps
    ll = get_tp_hc_ll()
    runtime = ll.hcx if ll is not None else None
    if (
        runtime is not None
        and ctx.is_uniform_decode
        and ctx.cudagraph_runtime_mode == CUDAGraphMode.NONE
        and 1 <= residual.shape[0] <= HCX_MAX_ROWS
        # MRV2 uses CUDAGraphMode.NONE inside both eager warmup and the actual
        # capture. Leave the eager runs on HC LL: the peer kernel is first
        # executed only at coordinated graph replay, not during rank-local JIT.
        and torch.cuda.is_current_stream_capturing()
    ):
        return runtime.run(
            partial, residual, injection, norm, eps, module._hcx_down, module._hcx_up
        )
    # The partial is still unreduced at all M. Preserve both baseline stages,
    # including explicit TP AR when HC LL declines (e.g. M > 64).
    hidden, xn = _hc_ll_combine_norm(partial, residual, injection, norm, eps, 4)
    block, inject = module._project(xn)
    assert inject is not None
    return hidden.contiguous(), block.contiguous(), inject.contiguous()


def _hc_hcx_combine_and_mix_fake(partial, residual, injection, name):
    return (
        torch.empty_like(residual, memory_format=torch.contiguous_format),
        partial.new_empty((residual.shape[0], HD)),
        injection.new_empty((residual.shape[0], INJ)),
    )


direct_register_custom_op(
    op_name="qwen4exp_sm70_hcx_combine_and_mix",
    op_func=_hc_hcx_combine_and_mix,
    fake_impl=_hc_hcx_combine_and_mix_fake,
)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the RMSNorm. Final mixers use
    ``use_combine=False`` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    _sm70_hcx_name: str | None = None

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def _project(
        self, xn: torch.Tensor, *, hc_sp: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_combine:
            fused_fp16 = maybe_apply_qwen38_sm70_fp16_fused_hc(
                self.input_mix_weight_down_block_inject,
                self.input_mix_weight_up,
                xn,
                getattr(self, "_sm70_qwen38_fp16_fused_hc", False),
                allow_tp_ll=not hc_sp,
            )
            if fused_fp16 is not None:
                return fused_fp16

            fused = maybe_apply_fused_hc(
                self.input_mix_weight_down_block_inject,
                self.input_mix_weight_up,
                xn,
            )
            if fused is not None:
                return fused

            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)
        return hc_gate_mix(xn, gate, self.hc_count), injection

    def mix(
        self, hidden_states: torch.Tensor, *, hc_sp: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        block_input, injection = (
            self._project(xn, hc_sp=True) if hc_sp else self._project(xn)
        )

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | HCTensorParallelPartial,
        prev_injection: torch.Tensor,
        *,
        hc_sp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        if isinstance(prev_block_output, HCTensorParallelPartial):
            hcx_name = self._sm70_hcx_name
            if (
                hcx_name is not None
                and use_sm70_decode_graph_semantics()
                and is_forward_context_available()
            ):
                return torch.ops.vllm.qwen4exp_sm70_hcx_combine_and_mix(
                    prev_block_output.value, hidden_states, prev_injection, hcx_name
                )
            hidden_states, xn = torch.ops.vllm.qwen4exp_sm70_hc_ll_combine_norm(
                prev_block_output.value,
                hidden_states,
                prev_injection,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )
        else:
            hidden_states, xn = hc_combine_norm(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )

        block_input, injection = (
            self._project(xn, hc_sp=True) if hc_sp else self._project(xn)
        )

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor | HCTensorParallelPartial,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(block_output, HCTensorParallelPartial):
            block_output = tensor_model_parallel_all_reduce(block_output.value)
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
