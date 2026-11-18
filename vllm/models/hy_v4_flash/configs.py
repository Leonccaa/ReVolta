# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The text configuration of the HYV4 Flash checkpoint."""

from copy import deepcopy
from typing import Any

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import ALLOWED_LAYER_TYPES, PretrainedConfig

SPARSE_ATTENTION = (
    "indexed_attention"
    if "indexed_attention" in ALLOWED_LAYER_TYPES
    else "deepseek_sparse_attention"
)
SLIDING_ATTENTION = "sliding_attention"
_SPARSE_ALIASES = {
    "sparse",
    "sparse_attention",
    "indexed_attention",
    "deepseek_sparse_attention",
}


@strict
class HYV4FlashConfig(PretrainedConfig):
    model_type = "hy_v4_flash"
    architectures: list[str] | None = None
    attribute_map = {
        "num_experts": "n_routed_experts",
        "num_local_experts": "n_routed_experts",
        "num_shared_experts": "n_shared_experts",
        "expert_hidden_dim": "moe_intermediate_size",
        "route_norm": "norm_topk_prob",
        "router_scaling_factor": "routed_scaling_factor",
    }
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    vocab_size: int = 120832
    hidden_size: int = 3072
    intermediate_size: int = 13824
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    n_routed_experts: int = 512
    n_shared_experts: int = 1
    num_experts_per_tok: int = 16
    routed_scaling_factor: float = 3.998
    norm_topk_prob: bool = True
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    indexer_types: list[str] | None = None
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    enable_ihc: bool = True
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    gated_mla: bool = True
    gating_type: str = "elementwise"
    learnable_sink: bool = True
    swiglu_limit: float = 10.0
    rope_parameters: dict | None = None
    routed_expert_hidden_size: int = 1536
    sandwich_norm: bool = True
    dsa_skip_attn_post_norm: bool = False
    swa_num_attention_heads: int = 64
    swa_qk_nope_head_dim: int = 192
    swa_qk_rope_head_dim: int = 64
    swa_v_head_dim: int = 256
    sliding_window: int = 512
    use_compressed_indexer: bool = True
    compressed_indexer_compress_ratio: int = 4
    compressed_indexer_num_attention_heads: int = 16
    compressed_indexer_head_dim: int = 128
    compressed_indexer_group_topk: int = 512
    compressed_indexer_candidate_topk: int = 2048
    engram_layer_ids: list[int] | None = None
    engram_max_ngram_size: int = 3
    engram_n_embed_per_ngram: int = 768
    engram_n_head_per_ngram: int = 8
    engram_pad_id: int = 120002
    engram_seed: int = 0
    engram_fp8_embedding: bool = True
    engram_layer_head_vocab_sizes: dict | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        config_dict = deepcopy(config_dict)
        text = config_dict.pop("text_config", {})
        config_dict.pop("vision_config", None)
        config_dict.update(text)
        config_dict["model_type"] = cls.model_type
        config_dict["architectures"] = ["HYV4FlashForCausalLM"]
        return super().from_dict(config_dict, **kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        if self.architectures is None:
            self.architectures = ["HYV4FlashForCausalLM"]
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.swa_qk_head_dim = self.swa_qk_nope_head_dim + self.swa_qk_rope_head_dim
        if self.layer_types is None:
            self.layer_types = [
                SPARSE_ATTENTION if i < 2 or (i - 1) % 4 == 0 else SLIDING_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            SPARSE_ATTENTION if kind in _SPARSE_ALIASES else kind
            for kind in self.layer_types
        ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.indexer_types is None:
            self.indexer_types = ["full"] * self.num_hidden_layers
        rope = deepcopy(self.rope_parameters) or {
            SPARSE_ATTENTION: {"rope_type": "default", "rope_theta": 1e7},
            SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 1e4},
        }
        for alias in _SPARSE_ALIASES - {SPARSE_ATTENTION}:
            if alias in rope:
                rope[SPARSE_ATTENTION] = rope.pop(alias)
        self.rope_parameters = rope
        super().__post_init__(**kwargs)

    def rope_for_layer(self, layer_idx: int) -> dict[str, Any]:
        assert self.layer_types is not None and self.rope_parameters is not None
        rope = self.rope_parameters
        return deepcopy(rope.get(self.layer_types[layer_idx], rope))
