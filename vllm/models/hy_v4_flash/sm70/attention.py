# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hy4 Flash text MLA with a shared paged reader for both layer types."""

from math import lcm
from typing import cast

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import current_workspace_manager

from .cache import POOL_PAGE, FlashCache, FlashMetadata, FlashSWABackend
from .dcp import apply_attn_sink
from .decode import decode_slots
from .indexpool import FlashCompressedIndexer
from .kernels import (
    cache_fp8_ds_mla,
    pack_fp8_ds_mla_torch,
    paged_attention,
    paged_attention_torch,
)
from .rotary import apply_rotary, rotary_phases
from .splitk import decode_workspace_specs, paged_attention_splitk
from .swa import window_indices


class HYV4FlashAttention(nn.Module):
    def __init__(self, vllm_config, config, layer_idx, topk_indices_buffer, prefix):
        super().__init__()
        # Projection names and TP axes follow vLLM PR #68296. The KV-A and
        # IndexPool projections are replicated, so every rank sees the same KV.
        self.is_window_layer = config.layer_types[layer_idx] == "sliding_attention"
        self.layer_idx = self.layer_id = layer_idx
        self.prefix = prefix
        attn_prefix = "swa_" if self.is_window_layer else ""
        self.num_heads = getattr(config, f"{attn_prefix}num_attention_heads")
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.qk_nope_head_dim = getattr(config, f"{attn_prefix}qk_nope_head_dim")
        self.qk_rope_head_dim = getattr(config, f"{attn_prefix}qk_rope_head_dim")
        self.v_head_dim = getattr(config, f"{attn_prefix}v_head_dim")
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scaling = self.qk_head_dim**-0.5
        self.window = config.sliding_window
        hidden_size, q_rank = config.hidden_size, config.q_lora_rank
        eps = config.rms_norm_eps
        quant = vllm_config.quant_config
        self.q_a_proj = ReplicatedLinear(
            hidden_size,
            q_rank,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_a_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.q_a_layernorm = RMSNorm(q_rank, eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps)
        self.q_b_proj = ColumnParallelLinear(
            q_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.linear_gate = ColumnParallelLinear(
            hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.linear_gate",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.o_proj",
        )
        self.learnable_sink_param = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            self.learnable_sink_param, {"weight_loader": sharded_weight_loader(0)}
        )
        self.post_proj_norm = (
            RMSNorm(hidden_size, eps)
            if config.sandwich_norm
            and (self.is_window_layer or not config.dsa_skip_attn_post_norm)
            else None
        )
        theta = config.rope_for_layer(layer_idx)["rope_theta"]
        self.register_buffer(
            "inv_freq",
            1.0
            / (
                theta
                ** (
                    torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32)
                    / self.qk_rope_head_dim
                )
            ),
            persistent=False,
        )

        world = vllm_config.parallel_config.decode_context_parallel_size
        global_page = POOL_PAGE * config.compressed_indexer_compress_ratio
        block_size = (
            global_page // world
            if world > 1
            else lcm(vllm_config.cache_config.block_size, global_page)
        )
        sharded = world > 1 and not self.is_window_layer
        if sharded:
            layers, _, attention = prefix.rsplit(".", 2)
            window_idx = config.layer_types.index("sliding_attention")
            self._swa_prefix = f"{layers}.{window_idx}.{attention}.attn"
        fp8_kv = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        main_cache_dtype = "fp8_ds_mla" if fp8_kv else "auto"
        cache_args = dict(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=torch.uint8 if fp8_kv else torch.float16,
            cache_dtype_str=main_cache_dtype,
            kv_quant_mode=get_kv_quant_mode(main_cache_dtype),
            dcp_sharded=sharded,
        )
        if self.is_window_layer:
            self.attn = FlashCache(
                vllm_config,
                f"{prefix}.attn",
                SlidingWindowMLASpec(
                    **cache_args,
                    sliding_window=self.window,
                ),
                FlashSWABackend,
            )
            self.indexer = None
            self.width = self.window
        else:
            self.attn = FlashCache(
                vllm_config, f"{prefix}.attn", MLAAttentionSpec(**cache_args)
            )
            self.indexer = FlashCompressedIndexer(
                vllm_config,
                config,
                f"{prefix}.indexer",
                block_size,
            )
            self.width = self.indexer.width
        self.topk_indices_buffer = topk_indices_buffer
        self.dcp_group = self.indexer.dcp_group if self.indexer is not None else None
        self.dcp_world = world if sharded else 1
        self.dcp_rank = self.dcp_group.rank_in_group if self.dcp_group else 0
        self._decode_workspace_specs = decode_workspace_specs(
            vllm_config.scheduler_config.max_num_seqs,
            self.num_local_heads * self.dcp_world,
            self.width // self.dcp_world,
            self.kv_lora_rank,
        )
        self.register_buffer(
            "_k_scale", torch.ones((), dtype=torch.float32), persistent=False
        )

    def _aggregate(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: dict[str, FlashMetadata],
    ) -> torch.Tensor:
        main = metadata[self.attn.prefix]
        tokens = main.num_actual_tokens
        cache = self.attn.kv_cache
        page = cache.shape[1]
        indices = self.topk_indices_buffer[:tokens, : self.width]
        slots = main.slot_mapping[:tokens]
        local_width = self.width // self.dcp_world
        if q.is_cuda and main.decode:
            active_slots = (
                slots
                if self.dcp_group is None
                else metadata[self._swa_prefix].slot_mapping[:tokens]
            )
            counts = main.counts
            assert counts is not None
            groups = None
            ratio = 1
            if self.indexer is not None:
                ratio = self.indexer.ratio
                groups = self.indexer.decode_groups(
                    hidden_states[:tokens],
                    q_c[:tokens],
                    positions[:tokens],
                    self.inv_freq,
                    cos[:tokens],
                    sin[:tokens],
                    metadata[self.indexer.k_cache.prefix],
                    metadata[self.indexer.tail_cache.prefix],
                    active_slots,
                )
            counts = counts[:tokens]
            decode_slots(
                main.seq_lens,
                active_slots,
                main.block_table,
                groups,
                indices[:, :local_width],
                counts,
                page,
                ratio,
                self.dcp_world,
                self.dcp_rank,
            )
        else:
            counts = self._eager_slots(
                hidden_states, q_c, positions, cos, sin, metadata, indices, page
            )
        if self.dcp_group is not None:
            indices = indices[:, :local_width]
            q = self.dcp_group.all_gather(q[:tokens].contiguous(), dim=1)
            lse = q.new_empty(q.shape[:2], dtype=torch.float32)
            sinks = None
        else:
            lse = None
            sinks = self.learnable_sink_param
        if q.is_cuda:
            if cache.dtype == torch.uint8:
                cache_fp8_ds_mla(kv_c[:tokens], k_pe[:tokens], cache, slots)
            else:
                ops.concat_and_cache_mla(
                    kv_c[:tokens], k_pe[:tokens], cache, slots, "auto", self._k_scale
                )
            if main.decode:
                state, acc = current_workspace_manager().get_simultaneous(
                    *self._decode_workspace_specs
                )
                latent = paged_attention_splitk(
                    q[:tokens],
                    cache,
                    indices,
                    counts,
                    sinks,
                    self.scaling,
                    state,
                    acc,
                    lse=lse,
                )
            else:
                latent = paged_attention(
                    q[:tokens],
                    cache,
                    indices,
                    counts,
                    sinks,
                    self.scaling,
                    self.kv_lora_rank,
                    lse=lse,
                )
            return self._merge_dcp(latent, lse) if lse is not None else latent
        if cache.dtype == torch.uint8:
            kv = pack_fp8_ds_mla_torch(kv_c[:tokens], k_pe[:tokens])
        else:
            kv = torch.cat((kv_c[:tokens], k_pe[:tokens]), dim=-1)
        cache[slots // page, slots % page] = kv
        return paged_attention_torch(
            q[:tokens],
            cache,
            indices,
            counts,
            self.learnable_sink_param,
            self.scaling,
            self.kv_lora_rank,
        )

    def _merge_dcp(self, latent: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        assert self.dcp_group is not None
        out, global_lse = cast(
            tuple[torch.Tensor, torch.Tensor],
            dcp_a2a_lse_reduce(
                latent, lse, self.dcp_group, return_lse=True, is_lse_base_on_e=True
            ),
        )
        return apply_attn_sink(out, global_lse, self.learnable_sink_param)

    def _profile_dcp(self, q: torch.Tensor) -> None:
        assert self.dcp_group is not None and self.indexer is not None
        # The profile has no cache. Exercise the real exchanges with empty
        # partials and keep gathered Q live until A2A scratch has been used.
        gathered = self.dcp_group.all_gather(q.contiguous(), dim=1)
        rows, heads, _ = gathered.shape
        self._merge_dcp(
            q.new_zeros((rows, heads, self.kv_lora_rank)),
            q.new_full((rows, heads), -torch.inf, dtype=torch.float32),
        )
        del gathered

    def _eager_slots(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: dict[str, FlashMetadata],
        indices: torch.Tensor,
        page: int,
    ) -> torch.Tensor:
        main = metadata[self.attn.prefix]
        tokens = main.num_actual_tokens
        if self.indexer is not None:
            self.indexer(
                hidden_states[:tokens],
                q_c[:tokens],
                positions[:tokens],
                self.inv_freq,
                cos[:tokens],
                sin[:tokens],
                metadata[self.indexer.k_cache.prefix],
                metadata[self.indexer.tail_cache.prefix],
                indices,
            )
        elif self.is_window_layer:
            for begin, end, prefix, length in main.spans:
                indices[begin:end] = window_indices(
                    torch.arange(prefix, length, device=indices.device),
                    self.window,
                )
        # Both selectors return compact, increasing logical slots with -1 padding.
        # Translate in place using the main cache table (not the IndexPool table).
        counts = torch.empty(tokens, dtype=torch.int32, device=indices.device)
        for req, (begin, end, _, _) in enumerate(main.spans):
            rows = indices[begin:end]
            if self.dcp_group is not None:
                # All complete groups and the open group start on rank zero.
                logical = rows[:, self.dcp_rank :: self.dcp_world] // self.dcp_world
                rows = rows[:, : self.width // self.dcp_world]
            else:
                logical = rows
            live = logical >= 0
            counts[begin:end] = live.sum(dim=-1, dtype=torch.int32)
            blocks = main.block_table[req][logical.clamp_min(0) // page]
            rows.copy_((blocks * page + logical % page).masked_fill(~live, -1))

        return counts

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_c, k_pe = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
        kv_c = self.kv_a_layernorm(kv_c.contiguous())
        cos, sin = rotary_phases(positions, self.inv_freq)
        q_nope, q_pe = q.split((self.qk_nope_head_dim, self.qk_rope_head_dim), -1)
        q_pe = apply_rotary(q_pe, cos[:, None], sin[:, None])
        k_pe = apply_rotary(k_pe, cos, sin)
        kv_b = self.kv_b_proj.weight.view(self.num_local_heads, -1, self.kv_lora_rank)
        w_uk, w_uv = kv_b.split((self.qk_nope_head_dim, self.v_head_dim), dim=1)
        latent_q = torch.bmm(q_nope.transpose(0, 1), w_uk).transpose(0, 1)
        q = torch.cat((latent_q, q_pe), dim=-1)

        metadata = cast(
            dict[str, FlashMetadata] | None, get_forward_context().attn_metadata
        )
        tokens = 0 if metadata is None else metadata[self.attn.prefix].num_actual_tokens
        if tokens:
            assert metadata is not None
            latent = self._aggregate(
                q, kv_c, k_pe, hidden_states, q_c, positions, cos, sin, metadata
            )
        else:
            # Profile runs have no bound cache or scheduled slots.
            if metadata is None and q.is_cuda:
                # Reserve decode scratch before sizing the KV cache.
                current_workspace_manager().get_simultaneous(
                    *self._decode_workspace_specs
                )
                if self.dcp_group is not None:
                    self._profile_dcp(q)
                if self.indexer is not None:
                    self.indexer.profile(q)
            latent = torch.zeros_like(latent_q)
        if 0 < tokens < q.shape[0]:
            latent = torch.nn.functional.pad(
                latent, (0, 0, 0, 0, 0, q.shape[0] - tokens)
            )
        out = torch.bmm(latent.transpose(0, 1), w_uv.transpose(1, 2))
        out = out.transpose(0, 1).reshape(-1, self.num_local_heads * self.v_head_dim)
        out = out * torch.sigmoid(self.linear_gate(hidden_states)[0])
        out = self.o_proj(out)[0]
        return self.post_proj_norm(out) if self.post_proj_norm is not None else out
