# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense FFN and MoE blocks for HY V4 (NVIDIA)."""

import torch
from torch import nn

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig


class HYV4FeedForward(nn.Module):
    """Dense SwiGLU feed-forward block.

    Used both for the dense decoder layers and for the MoE shared experts.
    The routed-expert SwiGLU clamp does not apply here, matching the reference
    implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return out


class HYV4MoEFused(nn.Module):
    """HY V4 sigmoid routing with shared and clamped routed experts."""

    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        routed_hidden_size: int | None = None,
        routed_input_transform: nn.Module | None = None,
        routed_output_transform: nn.Module | None = None,
    ):
        super().__init__()
        self.n_routed_experts = config.num_experts
        top_k = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: HYV4FeedForward | None
        if config.num_shared_experts > 0:
            self.shared_experts = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_hidden_dim * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
            )
        else:
            self.shared_experts = None

        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )

        raw_swiglu_limit = getattr(config, "swiglu_limit", 0)
        use_swiglu_clamp = bool(raw_swiglu_limit) and float(raw_swiglu_limit) > 0
        moe_swiglu_limit = float(raw_swiglu_limit) if use_swiglu_clamp else None
        if routed_input_transform is not None:
            self.routed_expert_down_proj = routed_input_transform
        if routed_output_transform is not None:
            self.routed_expert_up_proj = routed_output_transform

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=(
                config.hidden_size if routed_hidden_size is None else routed_hidden_size
            ),
            intermediate_size=intermediate_size,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            renormalize=config.route_norm,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=config.router_scaling_factor,
            router_logits_dtype=torch.float32,
            e_score_correction_bias=self.expert_bias,
            shared_experts=self.shared_experts,
            swiglu_limit=moe_swiglu_limit,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts); the gate stores fp32 weights
        # and emits fp32 logits, so no explicit upcast is needed here.
        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(orig_shape)
