# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 decoder and text backbone shared by Flash (vllm-project/vllm#68296)."""

from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig

from .hc import HYV4HCHeadLayer, HYV4HCLayer


class HYV4DecoderLayer(nn.Module):
    enable_ihc: bool
    self_attn: nn.Module
    mlp: nn.Module
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    hc_attn_layer: HYV4HCLayer
    hc_mlp_layer: HYV4HCLayer

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.enable_ihc:
            return self._forward_ihc(positions, hidden_states)
        return self._forward_normal(positions, hidden_states, residual)

    def _forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return self.mlp(hidden_states), residual

    def _forward_ihc(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)

        hidden_states, post_gates, residual = self.hc_mlp_layer.pre(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)
        return hidden_states, None


class HYV4Model(nn.Module):
    decoder_layer_cls: type[HYV4DecoderLayer]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.model_config is not None
        config = cast(HYV4FlashConfig, vllm_config.model_config.hf_text_config)
        self.config = config
        self.enable_ihc = config.enable_ihc
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=vllm_config.device_config.device,
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(
                config=config,
                vllm_config=vllm_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=maybe_prefix(prefix, "layers"),
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.enable_ihc:
            self.hc_head = HYV4HCHeadLayer(
                config,
                hidden_size=config.hidden_size,
                hc_mult=config.hc_mult,
                hc_eps=config.hc_eps,
                prefix=maybe_prefix(prefix, "hc_head"),
            )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class HYV4ForCausalLM(nn.Module):
    model_cls: type[HYV4Model]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.model_config is not None
        config = cast(HYV4FlashConfig, vllm_config.model_config.hf_text_config)
        self.config = config
        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            params_dtype=torch.float32,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states.float())
