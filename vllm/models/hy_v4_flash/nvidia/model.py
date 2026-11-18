# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash text backbone (vllm-project/vllm#68296)."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import regex as re
import torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, cached_file

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.models.hy_v4.nvidia.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4Model,
)
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.sequence import IntermediateTensors

from .moe import HYV4FlashFeedForward, HYV4FlashMoE

_AWQ_EXPERT = re.compile(
    r"(layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)"
    r"\.(qweight|qzeros|scales)"
)
_PACKED_EXPERT = re.compile(r"(layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)")
_EXPERT_SHARDS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
_ENGRAM = re.compile(r"layers\.(\d+)\.engram_layer\.(.+)")
_SKIP_PREFIXES = ("vit.", "dfly_layers.", "model.dfly_layers.")
_ENGRAM_WEIGHT = ".engram_layer.multi_head_embedding.embedding.weight"


class HYV4FlashDecoderLayer(HYV4DecoderLayer):
    def __init__(
        self,
        config: HYV4FlashConfig,
        vllm_config: VllmConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        ihc_sequence_parallel: bool = False,
    ) -> None:
        from vllm.models.hy_v4_flash.sm70.attention import HYV4FlashAttention

        super().__init__()
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.enable_ihc = config.enable_ihc
        self.self_attn = HYV4FlashAttention(
            vllm_config,
            config,
            self.layer_idx,
            topk_indices_buffer,
            f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        assert config.mlp_layer_types is not None
        dense = config.mlp_layer_types[self.layer_idx] == "dense"
        mlp_cls = HYV4FlashFeedForward if dense else HYV4FlashMoE
        self.mlp = mlp_cls(config, quant_config, f"{prefix}.mlp")
        self.hc_attn_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_attn_layer"
        )
        self.hc_mlp_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_mlp_layer"
        )


class HYV4FlashModel(HYV4Model):
    decoder_layer_cls = HYV4FlashDecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.engram_hash = None
        self.engram_indices = {
            layer_id - 1: index
            for index, layer_id in enumerate(self.config.engram_layer_ids or ())
        }
        if self.engram_indices:
            from .engram import FlashEngram, FlashNgramHashState

            self.engram_hash = FlashNgramHashState(
                self.config, dummy=vllm_config.load_config.load_format == "dummy"
            )
            for layer_idx in self.engram_indices:
                self.layers[layer_idx].engram_layer = FlashEngram(
                    vllm_config,
                    self.engram_hash.layout,
                    layer_idx + 1,
                    f"{prefix}.layers.{layer_idx}.engram_layer",
                )
        self.token_lookback_depth = (
            self.config.engram_max_ngram_size - 1 if self.engram_indices else 0
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        if self.engram_hash is not None:
            hashes = input_ids.new_empty(
                (
                    len(self.engram_indices),
                    input_ids.numel(),
                    self.engram_hash.offsets.shape[1],
                ),
                dtype=torch.int32,
            )
            self.engram_hash.prepare_hashes_with_output(
                input_ids,
                lookback_token_ids,
                self.layers[0].self_attn.attn.prefix,
                hashes,
            )
        residual = None
        for index, layer in enumerate(self.layers):
            if index in self.engram_indices:
                hidden = layer.hc_attn_layer.prepare_input(hidden)
                hidden = layer.engram_layer(hidden, hashes[self.engram_indices[index]])
            hidden, residual = layer(positions, hidden, residual)
        if self.enable_ihc:
            hidden = self.hc_head(hidden)
        elif residual is not None:
            hidden = hidden + residual
        return self.norm(hidden)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, value in weights:
            if match := _AWQ_EXPERT.fullmatch(name):
                prefix, expert_id, projection, suffix = match.groups()
                shard = _EXPERT_SHARDS[projection]
                weight = "w2" if shard == "w2" else "w13"
                name = f"{prefix}.{weight}_{suffix}"
                param = params[name]
                param.weight_loader(
                    param, value, name, shard_id=shard, expert_id=int(expert_id)
                )
            elif match := _PACKED_EXPERT.fullmatch(name):
                prefix, projection = match.groups()
                gate_up = projection == "gate_up_proj"
                name = f"{prefix}.{'w13' if gate_up else 'w2'}_weight"
                param = params[name]
                shards = ("w1", "w3") if gate_up else ("w2",)
                for expert_id, expert in enumerate(value.unbind(0)):
                    for shard, tensor in zip(shards, expert.chunk(len(shards), dim=0)):
                        param.weight_loader(
                            param, tensor, name, shard_id=shard, expert_id=expert_id
                        )
            else:
                if match := _ENGRAM.fullmatch(name):
                    layer, leaf = match.groups()
                    if leaf == "lookup_table":
                        assert self.engram_hash is not None
                        self.engram_hash.load_lookup(value, layer_id=int(layer) + 1)
                        continue
                    engram_layer = self.layers[int(layer)].engram_layer
                    embedding = engram_layer.multi_head_embedding
                    if leaf.startswith("lookup_per_head_norms."):
                        _, head, suffix = leaf.split(".", 2)
                        if not embedding.head_start <= int(head) < embedding.head_end:
                            continue
                        local_head = int(head) - embedding.head_start
                        name = (
                            f"layers.{layer}.engram_layer.lookup_per_head_norms."
                            f"{local_head}.{suffix}"
                        )
                param = params[name]
                loader: Any = getattr(param, "weight_loader", default_weight_loader)
                if (shard_id := getattr(value, "shard_id", None)) is None:
                    loader(param, value)
                else:
                    loader(param, value, shard_id)
            loaded.add(name)
        return loaded


class HYV4FlashForCausalLM(HYV4ForCausalLM):
    model_cls = HYV4FlashModel
    requires_raw_input_tokens = True
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
        },
        orig_to_new_substr={
            ".compressed_indexer.": ".indexer.",
            ".gate.e_score_correction_bias": ".expert_bias",
        },
        orig_to_new_suffix={
            ".hc_fn": ".hc_fn.weight",
            ".hc_head_fn": ".hc_head_fn.weight",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.quant_config is not None:
            vllm_config.quant_config.packed_modules_mapping = (
                self.packed_modules_mapping
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.token_lookback_depth = self.model.token_lookback_depth
        self.checkpoint_model_config = vllm_config.model_config
        self.checkpoint_cache_dir = vllm_config.load_config.download_dir

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
        )

    @staticmethod
    def skip_checkpoint_weight(name: str) -> bool:
        return name.startswith(_SKIP_PREFIXES) or name.endswith(
            (_ENGRAM_WEIGHT, _ENGRAM_WEIGHT + "_scale_inv")
        )

    def _load_engram_tables(self) -> None:
        if not self.model.engram_indices:
            return
        config = self.checkpoint_model_config
        assert config is not None
        index_path = cached_file(
            config.model,
            SAFE_WEIGHTS_INDEX_NAME,
            cache_dir=self.checkpoint_cache_dir,
            revision=config.revision,
        )
        assert index_path is not None
        index = Path(index_path)
        weight_map = json.loads(index.read_text())["weight_map"]
        for layer_idx in self.model.engram_indices:
            name = f"model.layers.{layer_idx}{_ENGRAM_WEIGHT}"
            if name + "_scale_inv" not in weight_map:
                raise ValueError(
                    "Engram tables require FP8 weights and BF16 row scales"
                )
            path = index.parent / weight_map[name]
            self.model.layers[
                layer_idx
            ].engram_layer.multi_head_embedding.load_from_safetensors(str(path), name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=list(_SKIP_PREFIXES))
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self._load_engram_tables()
        return loaded
