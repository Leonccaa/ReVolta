# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec

from .cache import FlashCache, FlashMetadata, QuerySpan
from .dcp import select_history
from .decode import gather_decode_group, store_decode_group
from .kernels import (
    index_prefill_gemm_elements,
    index_prefill_key_rows,
    index_prefill_scores,
    index_scores,
)
from .rotary import apply_rotary, rotary_phases


def pool_page_ids(
    groups: torch.Tensor,
    table: torch.Tensor,
    page: int,
    pages_per_block: int,
) -> torch.Tensor:
    logical_page = groups // page
    return (
        table[logical_page // pages_per_block] * pages_per_block
        + logical_page % pages_per_block
    )


def select_tokens(
    logits: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
    history: int,
) -> torch.Tensor:
    """Select complete groups, then append the current group's visible prefix."""
    selected = logits.topk(history, dim=-1, sorted=False).indices.sort(dim=-1).values
    return expand_groups(selected, positions, ratio)


def expand_groups(
    selected: torch.Tensor, positions: torch.Tensor, ratio: int
) -> torch.Tensor:
    current = positions // ratio
    offsets = torch.arange(ratio, device=positions.device)
    recalled = (selected[..., None] * ratio + offsets).flatten(1)
    tail = current[:, None] * ratio + offsets
    return torch.cat(
        (recalled, tail.masked_fill(tail > positions[:, None], -1)), dim=1
    ).int()


class FlashCompressedIndexer(nn.Module):
    """Coarse-only Hy4 IndexPool with FP16 pages and an open-group raw ring."""

    def __init__(self, vllm_config: VllmConfig, config, prefix: str, block_size: int):
        super().__init__()
        self.dcp_world = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_group = get_dcp_group() if self.dcp_world > 1 else None
        self.dcp_rank = self.dcp_group.rank_in_group if self.dcp_group else 0
        self.ratio = config.compressed_indexer_compress_ratio
        self.history = config.compressed_indexer_group_topk - 1
        self.width = (self.history + 1) * self.ratio
        self.num_heads = config.compressed_indexer_num_attention_heads
        self.dim = config.compressed_indexer_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.eps = config.rms_norm_eps
        model_config = vllm_config.model_config
        if self.dcp_group is not None:
            assert model_config is not None
            self.max_decode_rows = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = (
            model_config.max_model_len if model_config is not None else 0
        )
        self.logits_budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
        for name, in_dim, out_dim in (
            ("query_projection", config.q_lora_rank, self.num_heads * self.dim),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, self.dim),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    in_dim,
                    out_dim,
                    bias=False,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.token_key_norm = RMSNorm(self.dim, self.eps)
        self.key_value_projection = nn.Linear(
            self.dim, self.dim, bias=False, dtype=torch.float32
        )
        self.key_gate_projection = nn.Linear(
            self.dim, self.dim, bias=False, dtype=torch.float32
        )
        self.key_norm = nn.RMSNorm(self.dim, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.ratio, self.dim, dtype=torch.float32)
        )
        self.k_cache = FlashCache(
            vllm_config,
            f"{prefix}.k_cache",
            MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=self.dim,
                dtype=torch.float16,
                compress_ratio=self.ratio,
                dcp_sharded=self.dcp_world > 1,
            ),
        )
        self.tail_cache = FlashCache(
            vllm_config,
            f"{prefix}.tail_cache",
            CircularBufferSpec(
                block_size=self.ratio,
                num_kv_heads=1,
                head_size=self.dim + 1,
                dtype=torch.float32,
                dcp_sharded=False,
            ),
        )

    def _rotate(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (
                x[..., : -self.rope_dim],
                apply_rotary(x[..., -self.rope_dim :], cos, sin),
            ),
            -1,
        )

    def decode_groups(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: FlashMetadata,
        tail_metadata: FlashMetadata,
        slots: torch.Tensor,
    ) -> torch.Tensor | None:
        """Update the decode caches and return the sorted history group IDs."""
        workspace = metadata.index_workspace
        assert workspace is not None
        keys = self.token_key_norm(self.token_key_projection(hidden)[0])
        rows = keys.shape[0]
        pool, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        page = pool.shape[1]
        pages_per_block = self.k_cache.spec.storage_block_size // page
        table, ring_table = metadata.block_table, tail_metadata.block_table
        raw = workspace.raw[:rows]
        group_positions = workspace.positions[:rows]
        gather_decode_group[(rows,)](
            keys,
            metadata.seq_lens,
            slots,
            ring_table,
            tail,
            raw,
            group_positions,
            ring_table.stride(0),
            tail.stride(0),
            tail.stride(1),
            self.ratio,
            self.dim,
        )
        values = self.key_value_projection(raw)
        gates = self.key_gate_projection(raw) + self.intra_group_ape
        pooled = self.key_norm((values * gates.softmax(dim=1)).sum(dim=1)).to(
            keys.dtype
        )
        group_cos, group_sin = rotary_phases(group_positions, inv_freq)
        store_decode_group[(rows,)](
            keys,
            positions,
            self._rotate(pooled, group_cos, group_sin),
            metadata.seq_lens,
            slots,
            table,
            ring_table,
            pool,
            tail,
            table.stride(0),
            ring_table.stride(0),
            pool.stride(0),
            pool.stride(1),
            tail.stride(0),
            tail.stride(1),
            self.ratio,
            self.dim,
            page,
            pages_per_block,
            self.dcp_world,
            self.dcp_rank,
        )
        groups = workspace.logits.shape[1]
        logits: torch.Tensor | None = workspace.logits[:rows]
        if self.dcp_group is not None:
            if metadata.max_seq_len <= self.width:
                return None
            groups = cdiv(metadata.max_seq_len, self.ratio * self.dcp_world)
            logits = None
        elif groups == 0:
            return None
        query = self.query_projection(q_c)[0].view(rows, self.num_heads, self.dim)
        query = self._rotate(query, cos[:, None], sin[:, None])
        weights = self.head_weight_projection(hidden)[0].float()
        weights *= self.num_heads**-0.5 * self.dim**-0.5
        logits = index_scores(
            query,
            weights,
            pool,
            table,
            0,
            groups,
            self.ratio,
            pages_per_block,
            logits=logits,
            lengths=metadata.seq_lens,
            slots=slots,
            group_topk=self.history + 1,
            world=self.dcp_world,
            rank=self.dcp_rank,
        )
        if self.dcp_group is not None:
            lengths = metadata.seq_lens[:rows]
            upper = ((lengths - 1) // self.ratio).masked_fill(
                (slots < 0) | (lengths <= self.width), 0
            )
            return select_history(logits, self.local_lengths(upper), self.dcp_group)
        topk_groups = workspace.topk_groups[:rows]
        torch.topk(
            logits,
            self.history,
            sorted=False,
            out=(workspace.topk_values[:rows], topk_groups),
        )
        sorted_groups = workspace.sorted_groups[:rows]
        torch.sort(topk_groups, out=(sorted_groups, workspace.sort_order[:rows]))
        return sorted_groups

    def profile(self, q: torch.Tensor) -> None:
        if self.max_model_len <= self.width:
            return
        rows = q.shape[0]
        groups = cdiv(self.max_model_len, self.ratio * self.dcp_world)
        # Count logits, gathered FP16 keys and the FP32 GEMM output together.
        logits = max(groups, min(rows * groups, self.logits_budget))
        key_rows = index_prefill_key_rows(groups)
        keys = key_rows * self.dim // 2
        gemm = index_prefill_gemm_elements(rows * self.num_heads, key_rows)
        elements = logits + keys + gemm
        if self.dcp_group is not None:
            # Decode scores all requests together.
            decode = min(rows, self.max_decode_rows) * groups
            elements = max(elements, decode)
        score_scratch = q.new_empty((elements,), dtype=torch.float32)
        if self.dcp_group is not None:
            select_history(
                score_scratch[:rows, None],
                q.new_zeros(rows, dtype=torch.int32),
                self.dcp_group,
            )

    def local_lengths(self, groups: torch.Tensor) -> torch.Tensor:
        return ((groups + self.dcp_world - 1 - self.dcp_rank) // self.dcp_world).int()

    def prepare_groups(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        span: QuerySpan,
        table: torch.Tensor,
        ring_id: torch.Tensor,
    ) -> None:
        """Gather any previous open group before writing the next raw tail."""
        pool, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        page = pool.shape[1]
        pages_per_block = self.k_cache.spec.storage_block_size // page
        first, end = span.prefix // self.ratio, span.length // self.ratio
        prev = span.prefix % self.ratio
        if first < end:
            stop = end * self.ratio - span.prefix
            raw = keys[:stop].float()
            group_pos = positions[:stop]
            if prev:
                saved = tail[ring_id, :prev][0]
                raw = torch.cat((saved[:, : self.dim], raw), dim=0)
                group_pos = torch.cat(
                    (saved[:, self.dim].to(positions.dtype), group_pos)
                )
            raw = raw.reshape(-1, self.ratio, self.dim)
            group_pos = group_pos[:: self.ratio]
            groups = torch.arange(first, end, device=keys.device)
            if self.dcp_group is not None:
                owner = slice(
                    (self.dcp_rank - first) % self.dcp_world, None, self.dcp_world
                )
                raw, group_pos = raw[owner], group_pos[owner]
                groups = groups[owner] // self.dcp_world
            values = self.key_value_projection(raw)
            gates = self.key_gate_projection(raw) + self.intra_group_ape
            pooled = self.key_norm((values * gates.softmax(dim=1)).sum(dim=1)).to(
                keys.dtype
            )
            cos, sin = rotary_phases(group_pos, inv_freq)
            pages = pool_page_ids(groups, table, page, pages_per_block)
            pool[pages, groups % page] = self._rotate(pooled, cos, sin)
        # Earlier members of the same still-open group are left in the ring.
        tail_start = max(span.prefix, end * self.ratio)
        if tail_start < span.length:
            start = tail_start - span.prefix
            tail[ring_id, tail_start % self.ratio : span.length % self.ratio] = (
                torch.cat(
                    (keys[start:].float(), positions[start:, None].float()), dim=-1
                )
            )

    def forward(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: FlashMetadata,
        tail_metadata: FlashMetadata,
        output: torch.Tensor,
    ) -> None:
        keys = self.token_key_norm(self.token_key_projection(hidden)[0])
        pool = self.k_cache.kv_cache
        page = pool.shape[1]
        pages_per_block = self.k_cache.spec.storage_block_size // page
        columns = torch.arange(self.width, device=positions.device)
        for req, span in enumerate(metadata.spans):
            begin, end, prefix, length = span
            table = metadata.block_table[req]
            self.prepare_groups(
                keys[begin:end],
                positions[begin:end],
                inv_freq,
                span,
                table,
                tail_metadata.block_table[req, :1],
            )
            full_end = min(end, begin + max(0, self.width - prefix))
            output[begin:full_end] = columns.masked_fill(
                columns
                > torch.arange(prefix, prefix + full_end - begin, device=keys.device)[
                    :, None
                ],
                -1,
            )
            if full_end == end:
                continue
            query = self.query_projection(q_c[full_end:end])[0].view(
                -1, self.num_heads, self.dim
            )
            query = self._rotate(
                query, cos[full_end:end, None], sin[full_end:end, None]
            )
            weights = self.head_weight_projection(hidden[full_end:end])[0].float()
            weights *= self.num_heads**-0.5 * self.dim**-0.5
            first = prefix + full_end - begin
            groups = cdiv(length, self.ratio * self.dcp_world)
            chunk = max(1, self.logits_budget // groups)
            for start in range(0, end - full_end, chunk):
                stop = start + chunk
                q, w = query[start:stop], weights[start:stop]
                logical = torch.arange(
                    first + start, first + start + q.shape[0], device=q.device
                )
                if q.is_cuda:
                    logits = index_prefill_scores(
                        q,
                        w,
                        pool,
                        table,
                        first + start,
                        groups,
                        self.ratio,
                        pages_per_block,
                        world=self.dcp_world,
                        rank=self.dcp_rank,
                    )
                else:
                    g = torch.arange(groups, device=q.device)
                    k = pool[
                        pool_page_ids(g, table, page, pages_per_block), g % page
                    ].float()
                    logits = ((q.float() @ k.T).relu() * w[..., None]).sum(1)
                    logits.masked_fill_(g >= logical[:, None] // self.ratio, -torch.inf)
                if self.dcp_group is not None:
                    local_lengths = self.local_lengths(logical // self.ratio)
                    selected = select_history(logits, local_lengths, self.dcp_group)
                    tokens = expand_groups(selected, logical, self.ratio)
                else:
                    tokens = select_tokens(logits, logical, self.ratio, self.history)
                del logits
                output[full_end + start : full_end + start + q.shape[0]] = tokens
