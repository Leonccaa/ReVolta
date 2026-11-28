# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, cast

import torch
from torch import nn

from vllm import envs
from vllm.compilation.sm70_decode_graph import is_sm70_decode_graph_compiling
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    _sm70_dump_qwen_layer_tensor,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import (
    QSAForwardMetadata,
    build_qsa_metadata,
    qsa_dcp_block_geometry,
    qsa_draft_dcp_sharded,
    qsa_index_dcp_sharded,
    qsa_index_shard_world,
)
from .indexer_qsa import QSAIndexer
from .ops.qsa import _SM70_INDEXER_CUBLAS, expand_qsa_block_indices_cuda
from .ops.qsa_device_history import (
    H6_MAX_ROWS,
    load_qsa_device_history,
    load_qsa_paged_fp16_h6,
    load_qsa_paged_int8_h6,
    qsa_device_history,
    qsa_device_history_workspace,
    try_qsa_paged_fp16_h6,
    try_qsa_paged_int8_h6,
)
from .ops.qsa_device_history import (
    MAX_ROWS as DEVICE_HISTORY_MAX_ROWS,
)
from .ops.qsa_device_history import (
    MAX_SELECTION_WIDTH as DEVICE_HISTORY_MAX_WIDTH,
)
from .ops.qsa_paged_fp16_h12 import (
    H12_MAX_ROWS,
    H12_MAX_WIDTH,
    load_qsa_paged_h12,
    qsa_paged_h12,
    qsa_paged_h12_workspace,
)
from .ops.qsa_request_segments import warmup_build_segments
from .ops.sm70_dcp2 import get_sm70_dcp2_for_batch

if TYPE_CHECKING:
    from .host_kv.history import HostHistoryLayer


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # The replicated draft's slot map depends on logical positions as well
    # as the block table; FlashAttention's update hook only receives a table.
    supports_update_block_table: bool = False
    # QSA's DCP attention localizes its own selections and never reads the
    # per-rank context lengths, which cost a dozen small kernels per build.
    builds_dcp_context_lens: ClassVar[bool] = False

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.replicated_draft = (
            vllm_config.parallel_config.decode_context_parallel_size == 2
            and all("mtp" in name.split(".") for name in layer_names)
            and not qsa_draft_dcp_sharded(vllm_config)
        )
        if self.replicated_draft:
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.draft_block_table_buffer: torch.Tensor | None = None
            self.draft_token_to_req = torch.empty(
                max_tokens, dtype=torch.int32, device=device
            )
            self.draft_logical_positions = torch.empty(
                max_tokens, dtype=torch.int64, device=device
            )
            self.draft_slot_mapping = torch.empty(
                max_tokens, dtype=torch.int64, device=device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if not self.replicated_draft:
            return metadata
        # Draft K/V owns the full global span on every DCP rank. The mixed
        # target/draft group has already expanded each scheduler page into
        # kernel blocks using the target's rank-local page size, so only
        # expand the remaining draft-to-target ratio.
        draft_page_size, _, _ = qsa_dcp_block_geometry(
            self.vllm_config, self.layer_names[0]
        )
        if draft_page_size % self.block_size:
            raise RuntimeError("QSA draft kernel block must divide its physical page")
        group_page_size = (
            self.vllm_config.cache_config.block_size
            // self.vllm_config.parallel_config.decode_context_parallel_size
        )
        if group_page_size >= self.block_size and group_page_size % self.block_size:
            raise RuntimeError("QSA group page must divide into kernel blocks")
        group_blocks_per_page = max(1, group_page_size // self.block_size)
        draft_blocks_per_page = draft_page_size // self.block_size
        if draft_blocks_per_page % group_blocks_per_page:
            raise RuntimeError("QSA draft blocks must divide the group page")
        expansion = draft_blocks_per_page // group_blocks_per_page
        if expansion > 1:
            table = common_attn_metadata.block_table_tensor
            rows, columns = table.shape
            if self.draft_block_table_buffer is None:
                self.draft_block_table_buffer = torch.empty(
                    (
                        self.vllm_config.scheduler_config.max_num_seqs,
                        columns * expansion,
                    ),
                    dtype=table.dtype,
                    device=table.device,
                )
            if (
                rows > self.draft_block_table_buffer.shape[0]
                or columns * expansion > self.draft_block_table_buffer.shape[1]
            ):
                raise RuntimeError("QSA draft block-table buffer is too small")
            expanded = self.draft_block_table_buffer[:rows, : columns * expansion]
            expanded_view = expanded.view(rows, columns, expansion)
            for sub_block in range(expansion):
                torch.mul(table, expansion, out=expanded_view[:, :, sub_block])
                expanded_view[:, :, sub_block].add_(sub_block)
            common_attn_metadata = common_attn_metadata.replace(
                block_table_tensor=expanded
            )
            metadata.block_table = expanded
        _, _, slot_mapping = build_qsa_metadata(
            common_attn_metadata,
            self.draft_token_to_req,
            self.draft_logical_positions,
            self.draft_slot_mapping,
            storage_block_size=self.block_size,
            compress_ratio=1,
            map_plain_slot=True,
        )
        if common_attn_metadata.is_dummy_batch:
            slot_mapping.fill_(PAD_SLOT_ID)
        metadata.slot_mapping = slot_mapping
        return metadata


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "int8",
        cast(CacheDType, FP8_E4M3_PER_TOKEN_HEAD.name),
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # QSA chooses its split-K reduction depth from the runtime batch
        # shape, so it cannot inherit FlashAttention's stronger guarantee.
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = True
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size not in (1, 2):
            raise NotImplementedError("Qwen4Exp QSA supports DCP1 or DCP2")
        from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD

        if self.kv_cache_dtype not in (
            "auto",
            "float16",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
            "int8",
            FP8_E4M3_PER_TOKEN_HEAD.name,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires FP16/BF16 or E4M3 main KV cache"
            )
        if self.kv_cache_dtype in ("int8", FP8_E4M3_PER_TOKEN_HEAD.name):
            from .ops.qsa_int8 import require_row_scaled_page4_abi

            require_row_scaled_page4_abi()
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_gate: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        # This tree's FlashAttention cache ABI keeps K/V on dimension 1:
        # [num_blocks, 2, block_size, num_kv_heads, head_size].
        from vllm.v1.attention.kv_cache_view import KVCacheView
        from vllm.v1.attention.kv_codecs import resolve_kv_codec

        codec = resolve_kv_codec(self.kv_cache_dtype)
        if self.kv_cache_dtype == "auto" and kv_cache.dtype == torch.bfloat16:
            codec = resolve_kv_codec("bfloat16")
        assert codec is not None
        view = KVCacheView.bind(kv_cache, codec, self.head_size)
        key_cache, value_cache = view.key, view.value
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("Qwen4Exp QSA requires FP16/BF16 queries")
        if codec.element_dtype == torch.float8_e4m3fn:
            if key_cache.dtype != torch.uint8 or value_cache.dtype != torch.uint8:
                raise RuntimeError("Qwen4Exp QSA E4M3 cache must use uint8 storage")
        elif self.kv_cache_dtype == "int8":
            if key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8:
                raise RuntimeError("Qwen4Exp QSA INT8 cache requires signed payloads")
        elif key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
            raise RuntimeError("Qwen4Exp QSA FP16/BF16 cache must match query dtype")

        from .ops.qsa import qsa_sparse_paged_attention

        if getattr(layer, "qsa_dcp_sharded", False):
            if output_gate is None:
                raise RuntimeError("QSA DCP requires its output gate")
            self._forward_qsa_dcp(
                layer,
                query[:num_tokens],
                key_cache,
                value_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                output[:num_tokens],
                output_gate[:num_tokens],
                **view.scale_kwargs,
            )
            return output

        draft_scales = layer.qsa_draft_native_scales
        if (
            draft_scales is not None or layer.sm70_qsa_paged_int8_h6
        ) and _sm70_qsa_draft_decode():
            assert query_positions is not None and sequence_lengths is not None
            gate = output_gate[:num_tokens] if output_gate is not None else None
            if layer.sm70_qsa_paged_int8_h6:
                if try_qsa_paged_int8_h6(
                    query[:num_tokens],
                    key_cache,
                    value_cache,
                    logical_indices,
                    attn_metadata.block_table,
                    token_to_req,
                    query_positions[:num_tokens],
                    sequence_lengths,
                    output[:num_tokens],
                    gate,
                    **view.scale_kwargs,
                ):
                    return output
            elif try_qsa_paged_fp16_h6(
                query[:num_tokens],
                kv_cache,
                draft_scales,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                query_positions[:num_tokens],
                sequence_lengths,
                output[:num_tokens],
                gate,
            ):
                return output

        qsa_metadata: dict[str, torch.Tensor] = {}
        if query_positions is not None:
            qsa_metadata["query_positions"] = query_positions[:num_tokens]
        if sequence_lengths is not None:
            qsa_metadata["sequence_lengths"] = sequence_lengths
        if output_gate is not None:
            qsa_metadata["output_gate"] = output_gate[:num_tokens]
        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            **qsa_metadata,
            **view.scale_kwargs,
        )
        return output

    def forward_qsa_host(
        self,
        layer: torch.nn.Module,
        host: HostHistoryLayer,
        query: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        sequence_lengths: torch.Tensor,
        output_gate: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        gathered_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sparse attention over the main K/V in the host history store.

        Same selection, DCP localization, routes, merge and gate as the GPU
        path; the K/V operands are the layer's hot pool and the selection is
        remapped to hot token IDs (identity page table). Query rows run in
        passes that fit the hot pool. Positions and sequence lengths are not
        passed to the sparse kernel: the remapped IDs are not positions, and
        the selection is already causal (the remap also drops columns at or
        past each request's length). Under DCP2, small uniform decode steps
        read the planar INT8 hot pool with the native H12 reader when the
        layer admits it, like the GPU arm.

        With ``prefill_gather="union"`` a mixed step (more query rows than a
        decode step) instead fills the union of its selections into the hot
        slots per window of whole eight-row groups and runs the GPU arm's
        attention call unchanged: original selection, positions and lengths,
        a page-size-4 table of hot slots, every row (other windows' rows
        masked to request -1). Decode steps keep the single remap pass."""
        if self.sliding_window != (-1, -1) or self.alibi_slopes is not None:
            raise NotImplementedError("QSA host history needs plain attention")
        if self.sinks is not None:
            raise NotImplementedError("QSA host history needs plain attention")
        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output
        if query.dtype != torch.float16:
            raise NotImplementedError("QSA host history requires FP16 queries")
        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        from .host_kv.history import for_each_pass, run_union_windows
        from .ops.qsa import qsa_sparse_paged_attention

        selection = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        block_table = attn_metadata.block_table
        hot_table = host.store.hot_block_table
        key_cache, value_cache = host.key_cache, host.value_cache
        k_scale, v_scale = layer._k_scale_float, layer._v_scale_float
        scale_kwargs = (
            host.hot_cache.scale_kwargs if host.store.shape.planar_scales else {}
        )
        decode_rows = host.store.shape.decode_rows
        union = (
            attn_metadata.max_query_len > decode_rows
            and host.store.shape.prefill_gather == "union"
            # Union windows read counts on the host; inside a graph capture
            # (never expected: QSA is a splitting op) keep the remap passes.
            and not (query.is_cuda and torch.cuda.is_current_stream_capturing())
        )
        if not getattr(layer, "qsa_dcp_sharded", False):
            query = query[:num_tokens]
            out = output[:num_tokens]
            gate = None if output_gate is None else output_gate[:num_tokens]
            single_pass = attn_metadata.max_query_len <= decode_rows
            native_draft = (
                single_pass
                and layer.sm70_qsa_paged_int8_h6
                and _sm70_qsa_draft_decode()
            )

            def attend(start: int, end: int, hot_ids: torch.Tensor) -> None:
                rows_gate = None if gate is None else gate[start:end]
                if native_draft and try_qsa_paged_int8_h6(
                    query[start:end],
                    key_cache,
                    value_cache,
                    hot_ids,
                    hot_table,
                    token_to_req[start:end],
                    None,
                    None,
                    out[start:end],
                    rows_gate,
                    **scale_kwargs,
                ):
                    return
                qsa_sparse_paged_attention(
                    query[start:end],
                    key_cache,
                    value_cache,
                    hot_ids,
                    hot_table,
                    token_to_req[start:end],
                    out[start:end],
                    output_gate=rows_gate,
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    **scale_kwargs,
                )

            if union:
                metadata: dict[str, torch.Tensor] = {
                    "sequence_lengths": sequence_lengths
                }
                if query_positions is not None:
                    metadata["query_positions"] = query_positions[:num_tokens]
                if gate is not None:
                    metadata["output_gate"] = gate

                def attend_union(
                    table: torch.Tensor, requests: torch.Tensor, outs
                ) -> None:
                    # The GPU arm's call (forward_qsa), hot K/V and table.
                    qsa_sparse_paged_attention(
                        query,
                        key_cache,
                        value_cache,
                        selection,
                        table,
                        requests,
                        outs[0],
                        kv_cache_dtype=self.kv_cache_dtype,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        **scale_kwargs,
                        **metadata,
                    )

                run_union_windows(
                    host,
                    selection,
                    token_to_req,
                    block_table,
                    sequence_lengths,
                    [out],
                    attend_union,
                    attend,
                )
                return output
            for_each_pass(
                host,
                selection,
                token_to_req,
                block_table,
                sequence_lengths,
                attend,
                single_pass=single_pass,
            )
            return output

        if output_gate is None:
            raise RuntimeError("QSA DCP requires its output gate")
        from vllm.distributed import get_dcp_group

        from .ops.qsa import (
            _qsa_output_gate,
            qsa_dcp_sparse_paged_attention_sm70_grouped_page4,
        )
        from .ops.qsa_dcp import (
            qsa_dcp_local_selection_width,
            qsa_localize_dcp_indices,
        )

        buffers = (
            layer.dcp_local_indices_buffer,
            layer.dcp_partial_output_buffer,
            layer.dcp_partial_lse_buffer,
        )
        if any(buffer is None for buffer in buffers):
            raise RuntimeError("QSA DCP target workspaces are not initialized")
        local_indices = qsa_localize_dcp_indices(
            selection,
            layer.dcp_local_indices_buffer[:num_tokens],
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            interleave_size=layer.cp_kv_cache_interleave_size,
            local_block_size=host.kernel_block_tokens,
        )
        group = get_dcp_group()
        partial_output = layer.dcp_partial_output_buffer[:num_tokens]
        partial_lse = layer.dcp_partial_lse_buffer[:num_tokens]
        out = output[:num_tokens]
        gate = output_gate[:num_tokens].view_as(out)
        dcp2 = None
        if gathered_query is None:
            query = query[:num_tokens]
            dcp2 = get_sm70_dcp2_for_batch(
                group, query, partial_output, partial_lse, gate, out
            )
            gathered_query = (
                dcp2.gather_query(query)
                if dcp2 is not None
                else group.all_gather(query.contiguous(), dim=1)
            )
        if partial_output.shape != gathered_query.shape:
            raise RuntimeError("QSA DCP partial output has wrong head geometry")
        local_width = qsa_dcp_local_selection_width(
            layer.indexer.token_topk,
            layer.indexer.compress_ratio,
            self.dcp_world_size,
            layer.cp_kv_cache_interleave_size,
            local_indices.shape[1],
        )

        def attend_dcp(start: int, end: int, hot_ids: torch.Tensor) -> None:
            rows_q = gathered_query[start:end]
            rows_req = token_to_req[start:end]
            rows_out = partial_output[start:end]
            rows_lse = partial_lse[start:end]
            grouped = qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
                rows_q,
                key_cache,
                value_cache,
                hot_ids,
                hot_table,
                rows_req,
                rows_out,
                rows_lse,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
                local_width=local_width,
                **scale_kwargs,
            )
            if (
                grouped == 0
                and layer.sm70_qsa_paged_h12
                and end - start <= H12_MAX_ROWS
                and 0 < local_width <= H12_MAX_WIDTH
                and get_forward_context().is_uniform_decode
            ):
                # The gather filled this pass's groups on the current stream.
                qsa_paged_h12(
                    rows_q,
                    key_cache,
                    value_cache,
                    hot_ids[:, :local_width],
                    hot_table,
                    rows_req,
                    rows_out,
                    rows_lse,
                    **scale_kwargs,
                )
                grouped = end - start
            if grouped < end - start:
                qsa_sparse_paged_attention(
                    rows_q[grouped:],
                    key_cache,
                    value_cache,
                    hot_ids[grouped:, :local_width],
                    hot_table,
                    rows_req[grouped:],
                    rows_out[grouped:],
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    **scale_kwargs,
                    lse=rows_lse[grouped:],
                )

        def attend_dcp_union(table: torch.Tensor, requests: torch.Tensor, outs) -> None:
            # Match the GPU arm's prefill routing on the whole batch shape.
            rows_out, rows_lse = outs
            grouped = qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
                gathered_query,
                key_cache,
                value_cache,
                local_indices,
                table,
                requests,
                rows_out,
                rows_lse,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
                local_width=local_width,
                **scale_kwargs,
            )
            if grouped < num_tokens:
                qsa_sparse_paged_attention(
                    gathered_query[grouped:],
                    key_cache,
                    value_cache,
                    local_indices[grouped:, :local_width],
                    table,
                    requests[grouped:],
                    rows_out[grouped:],
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    **scale_kwargs,
                    lse=rows_lse[grouped:],
                )

        if union:
            run_union_windows(
                host,
                local_indices,
                token_to_req,
                block_table,
                sequence_lengths,
                [partial_output, partial_lse],
                attend_dcp_union,
                attend_dcp,
            )
        else:
            for_each_pass(
                host,
                local_indices,
                token_to_req,
                block_table,
                sequence_lengths,
                attend_dcp,
                single_pass=attn_metadata.max_query_len <= decode_rows,
            )
        if dcp2 is not None:
            dcp2.combine_gate(partial_output, partial_lse, gate, out)
            return output
        merged = cast(
            torch.Tensor,
            self.dcp_combine(
                partial_output, partial_lse, group, is_lse_base_on_e=False
            ),
        )
        _qsa_output_gate(out, gate, merged=merged)
        return output

    def _forward_qsa_dcp(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        logical_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
    ) -> None:
        from vllm.distributed import get_dcp_group

        from .ops.qsa import (
            _qsa_output_gate,
            qsa_dcp_sparse_paged_attention_sm70_grouped_page4,
            qsa_sparse_paged_attention,
        )
        from .ops.qsa_dcp import (
            qsa_dcp_local_selection_width,
            qsa_localize_dcp_indices,
        )

        local_indices_buffer = layer.dcp_local_indices_buffer
        partial_output_buffer = layer.dcp_partial_output_buffer
        partial_lse_buffer = layer.dcp_partial_lse_buffer
        if any(
            buffer is None
            for buffer in (
                local_indices_buffer,
                partial_output_buffer,
                partial_lse_buffer,
            )
        ):
            raise RuntimeError("QSA DCP target workspaces are not initialized")
        local_indices = qsa_localize_dcp_indices(
            logical_indices,
            local_indices_buffer[: query.shape[0]],
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            interleave_size=layer.cp_kv_cache_interleave_size,
            local_block_size=key_cache.shape[1],
        )
        group = get_dcp_group()
        partial_output = partial_output_buffer[: query.shape[0]]
        partial_lse = partial_lse_buffer[: query.shape[0]]
        gate = output_gate.view_as(output)
        dcp2 = get_sm70_dcp2_for_batch(
            group, query, partial_output, partial_lse, gate, output
        )
        gathered_query = (
            dcp2.gather_query(query)
            if dcp2 is not None
            else group.all_gather(query.contiguous(), dim=1)
        )
        if partial_output.shape != gathered_query.shape:
            raise RuntimeError("QSA DCP partial output has wrong head geometry")
        # Only this prefix of the compacted columns can hold an owned token.
        local_width = qsa_dcp_local_selection_width(
            layer.indexer.token_topk,
            layer.indexer.compress_ratio,
            self.dcp_world_size,
            layer.cp_kv_cache_interleave_size,
            local_indices.shape[1],
        )
        scale_kwargs = {}
        from vllm.v1.attention.kv_codecs import resolve_kv_codec

        codec = resolve_kv_codec(self.kv_cache_dtype)
        if codec is not None and codec.per_token_head:
            scale_kwargs = {
                "k_scale_cache": k_scale_cache,
                "v_scale_cache": v_scale_cache,
            }
        # Small uniform decode batches may use the native H12 reader below.
        grouped_rows = qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
            gathered_query,
            key_cache,
            value_cache,
            local_indices,
            block_table,
            token_to_req,
            partial_output,
            partial_lse,
            self.kv_cache_dtype,
            layer._k_scale_float,
            layer._v_scale_float,
            local_width=local_width,
            **scale_kwargs,
        )
        if (
            grouped_rows == 0
            and layer.sm70_qsa_paged_h12
            and query.shape[0] <= H12_MAX_ROWS
            and local_width <= H12_MAX_WIDTH
            and get_forward_context().is_uniform_decode
        ):
            qsa_paged_h12(
                gathered_query,
                key_cache,
                value_cache,
                local_indices[:, :local_width],
                block_table,
                token_to_req,
                partial_output,
                partial_lse,
                **scale_kwargs,
            )
            grouped_rows = query.shape[0]
        elif (
            grouped_rows == 0
            and layer.sm70_qsa_device_history
            and query.shape[0] <= DEVICE_HISTORY_MAX_ROWS
            and local_width <= DEVICE_HISTORY_MAX_WIDTH
            and get_forward_context().is_uniform_decode
        ):
            qsa_device_history(
                gathered_query,
                key_cache,
                value_cache,
                local_indices[:, :local_width],
                block_table,
                token_to_req,
                partial_output,
                partial_lse,
                layer._k_scale_float,
                layer._v_scale_float,
            )
            grouped_rows = query.shape[0]
        if grouped_rows < query.shape[0]:
            qsa_sparse_paged_attention(
                gathered_query[grouped_rows:],
                key_cache,
                value_cache,
                local_indices[grouped_rows:, :local_width],
                block_table,
                token_to_req[grouped_rows:],
                partial_output[grouped_rows:],
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                lse=partial_lse[grouped_rows:],
                compacted_indices=True,
                **scale_kwargs,
            )
        if dcp2 is not None:
            dcp2.combine_gate(partial_output, partial_lse, gate, output)
            return
        merged = cast(
            torch.Tensor,
            self.dcp_combine(
                partial_output,
                partial_lse,
                group,
                is_lse_base_on_e=False,
            ),
        )
        _qsa_output_gate(output, gate, merged=merged)


def _verify_e4m3_kv_requirements(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> None:
    """Admit calibrated E4M3 from kernel capabilities, independent of TP/MTP."""
    if cache_config.cache_dtype == "int8":
        if model_config.dtype != torch.float16:
            raise NotImplementedError("QSA INT8 requires FP16 activations on SM70")
        return
    from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD

    if cache_config.cache_dtype not in (
        "fp8",
        "fp8_e4m3",
        FP8_E4M3_PER_TOKEN_HEAD.name,
    ):
        return
    from .ops.qsa import qsa_e4m3_capability_reason

    if reason := qsa_e4m3_capability_reason(model_config.dtype):
        raise NotImplementedError(f"QSA E4M3 cache unavailable: {reason}")


def _warmup_sm70_qsa_folded8(
    forward_context: dict[str, object], vllm_config: VllmConfig
) -> bool:
    for layer in forward_context.values():
        if isinstance(layer, Qwen4ExpQSAAttention) and layer._warmup_sm70_qsa_folded8(
            vllm_config
        ):
            return True
    return False


def _warmup_sm70_qsa_segments(
    forward_context: dict[str, object], vllm_config: VllmConfig
) -> bool:
    if not envs.VLLM_SM70_QSA_REQUEST_SEGMENTS:
        return False
    for layer in forward_context.values():
        if isinstance(layer, Qwen4ExpQSAAttention):
            indexer = layer.indexer
            cache = indexer.compressed_key_cache
            if cache.dtype != torch.float16 or (
                indexer.index_n_heads,
                indexer.index_head_dim,
            ) != (4, 128):
                continue
            sharded = qsa_index_dcp_sharded(
                vllm_config, cache.prefix, indexer.compress_ratio
            )
            warmup_build_segments(
                layer.topk_indices_buffer.device,
                indexer.compress_ratio,
                qsa_index_shard_world(vllm_config) if sharded else 1,
                get_tensor_model_parallel_rank() if sharded else 0,
            )
            return True
    return False


def _sm70_qsa_draft_decode() -> bool:
    return is_sm70_decode_graph_compiling() and bool(
        get_forward_context().additional_kwargs.get("sm70_qsa_draft_decode")
    )


def _sm70_qsa_paged_int8_h6_admitted(
    vllm_config: VllmConfig,
    kv_cache_dtype: str,
    dcp_sharded: bool,
    geometry: tuple[int, int, int],
) -> bool:
    return (
        vllm_config.kernel_config.sm70_qsa_paged_int8_h12 is True
        and kv_cache_dtype == "int8"
        and vllm_config.is_speculative_draft
        and not dcp_sharded
        and geometry == (6, 1, 256)
        and vllm_config.model_config.dtype == torch.float16
        and current_platform.is_device_capability(70)
    )


def _sm70_qsa_paged_h12_admitted(
    vllm_config: VllmConfig,
    kv_cache_dtype: str,
    dcp_sharded: bool,
    tp_size: int,
    geometry: tuple[int, int, int],
) -> bool:
    """Admit the native H12 reader for a TP4/DCP2 SM70 target layer.

    Host main K/V uses the INT8 instance on the planar hot pool; FP16 host
    history keeps the Triton reader."""
    kernel_config = vllm_config.kernel_config
    host_kv = kernel_config.long_context_kv
    if kv_cache_dtype == "int8":
        enabled = kernel_config.sm70_qsa_paged_int8_h12 is True
    else:
        enabled = kernel_config.sm70_qsa_paged_fp16_h12 and kv_cache_dtype in (
            "auto",
            "float16",
        )
    return (
        enabled
        and vllm_config.use_v2_model_runner
        and not vllm_config.is_speculative_draft
        and dcp_sharded
        and vllm_config.parallel_config.decode_context_parallel_size == 2
        and tp_size == 4
        and geometry == (6, 1, 256)
        and vllm_config.model_config.dtype == torch.float16
        and (
            not host_kv.enabled
            or host_kv.main_kv_placement == "gpu"
            or kv_cache_dtype == "int8"
        )
        and current_platform.is_device_capability(70)
    )


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = True
    # The paged indexer and sparse attention switch launch profiles after 32
    # query rows. Advertise the first row count in the wider profile so the
    # generic MRV2 warmup can compile it before serving traffic.
    kernel_warmup_prefill_token_counts = (33,)

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        dcp_local_indices_buffer: torch.Tensor | None = None,
        dcp_partial_output_buffer: torch.Tensor | None = None,
        dcp_partial_lse_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("Qwen4Exp QSA requires FP16 or BF16")
        from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD

        if cache_config.cache_dtype not in (
            "auto",
            "float16",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
            "int8",
            FP8_E4M3_PER_TOKEN_HEAD.name,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires FP16/BF16 or E4M3 main KV cache"
            )
        _verify_e4m3_kv_requirements(vllm_config, model_config, cache_config)
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support prefill context parallelism"
            )
        if parallel_config.decode_context_parallel_size not in (1, 2):
            raise NotImplementedError("Qwen4Exp QSA supports DCP1 or DCP2")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True
        qkv_quant_config = quant_config
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            qkv_quant_config = None

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=qkv_quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        mm_config = model_config.multimodal_config
        # Qwen4Exp M-RoPE positions never exceed the token count unless video
        # pruning re-spaces them, so a 1x cos/sin cache suffices otherwise.
        video_pruning = mm_config is not None and (
            mm_config.is_multimodal_pruning_enabled()
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            mrope_positions_factor=4 if video_pruning else 1,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        _, _, self.qsa_dcp_sharded = qsa_dcp_block_geometry(
            vllm_config, self.layer_name
        )
        self.qsa_dcp_sharded &= parallel_config.decode_context_parallel_size > 1
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_local_indices_buffer = dcp_local_indices_buffer
        self.dcp_partial_output_buffer = dcp_partial_output_buffer
        self.dcp_partial_lse_buffer = dcp_partial_lse_buffer
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        kernel_config = vllm_config.kernel_config
        host_kv = kernel_config.long_context_kv
        native_draft = (
            kernel_config.sm70_qsa_paged_fp16_h6
            and vllm_config.is_speculative_draft
            and not self.qsa_dcp_sharded
            and not host_kv.enabled
            and self.kv_cache_dtype in ("auto", "float16")
            and model_config.dtype == torch.float16
            and (self.num_heads, self.num_kv_heads, self.head_dim) == (6, 1, 256)
            and current_platform.is_device_capability(70)
        )
        if native_draft:
            load_qsa_paged_fp16_h6()
        self.register_buffer(
            "qsa_draft_native_scales",
            torch.ones(2, dtype=torch.float32) if native_draft else None,
            persistent=False,
        )
        self.sm70_qsa_paged_int8_h6 = _sm70_qsa_paged_int8_h6_admitted(
            vllm_config,
            self.kv_cache_dtype,
            self.qsa_dcp_sharded,
            (self.num_heads, self.num_kv_heads, self.head_dim),
        )
        if self.sm70_qsa_paged_int8_h6:
            load_qsa_paged_int8_h6()
        self.sm70_qsa_device_history = (
            kernel_config.sm70_qsa_device_history
            and vllm_config.use_v2_model_runner
            and not vllm_config.is_speculative_draft
            and not host_kv.enabled
            and self.qsa_dcp_sharded
            and parallel_config.decode_context_parallel_size == 2
            and tp_size == 4
            and (self.num_heads, self.num_kv_heads, self.head_dim) == (6, 1, 256)
            and model_config.dtype == torch.float16
            and self.kv_cache_dtype == "fp8_e4m3"
            and current_platform.is_device_capability(70)
        )
        if self.sm70_qsa_device_history:
            load_qsa_device_history()
        self.sm70_qsa_paged_h12 = _sm70_qsa_paged_h12_admitted(
            vllm_config,
            self.kv_cache_dtype,
            self.qsa_dcp_sharded,
            tp_size,
            (self.num_heads, self.num_kv_heads, self.head_dim),
        )
        if self.sm70_qsa_paged_h12:
            load_qsa_paged_h12(self.kv_cache_dtype)
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3") and (
            cache_config.calculate_kv_scales
        ):
            raise ValueError(
                "QSA calibrated E4M3 forbids calculate_kv_scales; "
                "load an offline scale overlay instead"
            )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD

        if self.kv_cache_dtype not in (
            "fp8",
            "fp8_e4m3",
            "int8",
            FP8_E4M3_PER_TOKEN_HEAD.name,
        ) and (self.kv_cache_torch_dtype != model_config.dtype):
            raise NotImplementedError(
                "Qwen4Exp QSA main cache dtype must match the model dtype"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        # Bound by the model runner in long-context mode when the main K/V
        # lives in the host history store (no GPU tensor then).
        self.host_history: HostHistoryLayer | None = None
        set_default_quant_scales(self, register_buffer=True)
        self._qsa_kv_scales_finalized = self.kv_cache_dtype not in (
            "fp8",
            "fp8_e4m3",
        )
        if not self._qsa_kv_scales_finalized:
            # Keep checkpoint loading state separate from runtime scales.
            # Negative is deliberately invalid and the slots are deleted once
            # validation copies them into the runtime buffers.
            self.k_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
            self.v_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._set_topk_indices_buffer(
            max_tokens=max_tokens,
            topk_indices_buffer=topk_indices_buffer,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def _set_topk_indices_buffer(
        self,
        *,
        max_tokens: int,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        if topk_indices_buffer is None:
            self.register_buffer(
                "topk_indices_buffer",
                torch.empty(
                    max_tokens,
                    self.indexer.output_width,
                    dtype=torch.int32,
                ),
                persistent=False,
            )
            return

        expected_width = self.indexer.output_width
        if (
            topk_indices_buffer.dtype != torch.int32
            or topk_indices_buffer.ndim != 2
            or topk_indices_buffer.shape[0] < max_tokens
            or topk_indices_buffer.shape[1] != expected_width
        ):
            raise ValueError(
                "QSA shared top-k buffer must have dtype int32 and shape "
                f"[{max_tokens} or more, {expected_width}], got "
                f"dtype={topk_indices_buffer.dtype}, "
                f"shape={tuple(topk_indices_buffer.shape)}"
            )
        self.topk_indices_buffer = topk_indices_buffer

    def _warmup_sm70_qsa_folded8(self, vllm_config: VllmConfig) -> bool:
        indexer = self.indexer
        cache = indexer.compressed_key_cache
        if (
            not _SM70_INDEXER_CUBLAS
            or indexer.prefill_tp_group is None
            or cache.dtype != torch.float16
            or (indexer.index_n_heads, indexer.index_head_dim) != (4, 128)
            or (indexer.token_topk, indexer.compress_ratio) != (2048, 4)
            or qsa_index_dcp_sharded(vllm_config, cache.prefix, indexer.compress_ratio)
            or min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.model_config.max_model_len,
            )
            < 2048
        ):
            return False

        # Eight rows exercise the folded layout. The expansion kernel's key
        # is independent of row count, so no long model prefill is needed.
        rows = 8
        out = self.topk_indices_buffer[:rows]
        blocks = out.new_zeros((rows, indexer.token_topk // indexer.compress_ratio))
        positions = torch.arange(rows, dtype=torch.int64, device=out.device)
        expand_qsa_block_indices_cuda(
            blocks,
            positions,
            out.new_full((1,), rows),
            out.new_zeros(rows),
            indexer.compress_ratio,
            indexer.token_topk,
            out,
            folded8=True,
        )
        return True

    def adopt_default_kv_scales(self) -> None:
        """Use the module's own unit scales when the checkpoint has none.

        The calibrated overlay exists to keep FP8 E4M3 K/V inside range; a
        checkpoint that was never calibrated has no such overlay, so the layer
        keeps the 1.0 defaults set at construction instead of the -1.0 loading
        sentinel and is marked finalized so nothing re-validates it.
        """
        set_default_quant_scales(self, register_buffer=False)
        if hasattr(self, "k_scale"):
            del self.k_scale
        if hasattr(self, "v_scale"):
            del self.v_scale
        self._qsa_kv_scales_finalized = True

    def validate_loaded_kv_scales(self) -> None:
        if self.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            return
        if self._qsa_kv_scales_finalized:
            raise RuntimeError(
                f"QSA E4M3 scales already finalized for {self.layer_name}"
            )
        if not hasattr(self, "k_scale") or not hasattr(self, "v_scale"):
            raise RuntimeError(
                f"QSA E4M3 loading slots are unavailable for {self.layer_name}"
            )

        scales = {
            "K": float(self.k_scale.item()),
            "V": float(self.v_scale.item()),
        }
        invalid = [
            name
            for name, value in scales.items()
            if not math.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(
                f"QSA E4M3 calibrated scales are required for {self.layer_name} "
                f"(invalid: {', '.join(invalid)}). Refusing to start."
            )
        self._k_scale.copy_(scales["K"])
        self._v_scale.copy_(scales["V"])
        self._k_scale_float = scales["K"]
        self._v_scale_float = scales["V"]
        del self.k_scale
        del self.v_scale
        self._qsa_kv_scales_finalized = True

    def reserve_capture_workspace(self) -> None:
        """Reserve reader and DCP combine workspaces before capture."""
        if not self.qsa_dcp_sharded:
            width = self.indexer.output_width
            if 0 < width <= DEVICE_HISTORY_MAX_WIDTH and (
                self.sm70_qsa_paged_int8_h6
                or (
                    self.qsa_draft_native_scales is not None
                    and self.kv_cache.is_contiguous()
                )
            ):
                qsa_device_history_workspace(H6_MAX_ROWS, width)
            return

        from vllm.v1.attention.ops.dcp_alltoall import (
            dcp_a2a_lse_reduce,
            reserve_dcp_a2a_workspace,
        )

        partial = self.dcp_partial_output_buffer
        if partial is None:
            return
        if self.sm70_qsa_paged_h12 or self.sm70_qsa_device_history:
            from .ops.qsa_dcp import qsa_dcp_local_selection_width

            assert self.dcp_local_indices_buffer is not None
            width = qsa_dcp_local_selection_width(
                self.indexer.token_topk,
                self.indexer.compress_ratio,
                self.impl.dcp_world_size,
                self.cp_kv_cache_interleave_size,
                self.dcp_local_indices_buffer.shape[1],
            )
            if self.sm70_qsa_paged_h12 and width <= H12_MAX_WIDTH:
                qsa_paged_h12_workspace(min(partial.shape[0], H12_MAX_ROWS), width)
            if self.sm70_qsa_device_history and width <= DEVICE_HISTORY_MAX_WIDTH:
                qsa_device_history_workspace(
                    min(partial.shape[0], DEVICE_HISTORY_MAX_ROWS), width
                )
        if getattr(self.impl, "dcp_combine", None) is not dcp_a2a_lse_reduce:
            return
        max_tokens, num_heads, head_dim = partial.shape
        reserve_dcp_a2a_workspace(
            max_tokens,
            num_heads,
            head_dim,
            self.impl.dcp_world_size,
            partial.dtype,
            partial.device,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size, _, dcp_sharded = qsa_dcp_block_geometry(
            vllm_config, self.layer_name
        )
        if (
            not dcp_sharded
            and vllm_config.parallel_config.decode_context_parallel_size == 1
        ):
            # On one rank the MTP page has the same DCP geometry as the target.
            # Mark it consistently so matching pages can share the host store.
            host = vllm_config.kernel_config.long_context_kv
            dcp_sharded = (
                host.enabled
                and host.main_kv_placement == "host"
                and host.draft_main_placement == "host"
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            dcp_sharded=dcp_sharded,
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split, normalize, and rotate Q/K using this tree's Qwen3 API."""
        q_gate, key, value = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
        token_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*token_shape, self.num_heads, 2 * self.head_dim)
        query, gate = torch.chunk(q_gate, 2, dim=-1)
        query = self.q_norm(query).reshape(*token_shape, self.q_size)
        key = self.k_norm(
            key.view(*token_shape, self.num_kv_heads, self.head_dim)
        ).reshape(*token_shape, self.kv_size)
        query, key = self.rotary_emb(positions, query, key)
        return query, key, value, gate.reshape(*token_shape, self.q_size)

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor | None = None,
        *,
        row_indices: torch.Tensor | None = None,
    ) -> None:
        if not self._qsa_kv_scales_finalized:
            raise RuntimeError(
                f"QSA E4M3 scales were not finalized for {self.layer_name}"
            )
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        # Long-context mode: main K/V in the host history store, by history
        # block ID (no GPU tensor is bound then).
        host = getattr(self, "host_history", None)
        if host is None and self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        if os.getenv("VLLM_QSA_KV_CALIBRATION_DIR"):
            from .ops.qsa_kv_calibration import observe_qsa_kv

            observe_qsa_kv(
                self.indexer.layer_id,
                key[:num_tokens],
                value[:num_tokens],
            )
        num_queries = num_tokens if row_indices is None else row_indices.numel()
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_queries],
            row_indices=row_indices,
        )
        if selected.shape != (
            num_queries,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        selected = _sm70_dump_qwen_layer_tensor(
            "qsa_selected_indices",
            self.indexer.layer_id,
            "qsa",
            selected,
        )
        self._write_main_kv(key, value, main_metadata.slot_mapping)
        token_to_req = side_metadata.token_to_req
        logical_positions = side_metadata.logical_positions
        if row_indices is not None:
            # Readers use one query per request against the original history.
            # Keep the full block table (including replicated DCP draft pages).
            main_metadata = replace(
                main_metadata, num_actual_tokens=num_queries, max_query_len=1
            )
            query = query.index_select(0, row_indices)
            token_to_req = token_to_req.index_select(0, row_indices)
            logical_positions = logical_positions.index_select(0, row_indices)
            if output_gate is not None:
                output_gate = output_gate.index_select(0, row_indices)
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if host is not None:
            impl.forward_qsa_host(
                self,
                host,
                query,
                main_metadata,
                output,
                token_to_req=token_to_req,
                sequence_lengths=side_metadata.seq_lens,
                output_gate=output_gate,
                query_positions=logical_positions,
            )
        else:
            impl.forward_qsa(
                self,
                query,
                key,
                value,
                self.kv_cache,
                main_metadata,
                output,
                token_to_req=token_to_req,
                output_gate=output_gate,
                query_positions=logical_positions,
                sequence_lengths=side_metadata.seq_lens,
            )
        _sm70_dump_qwen_layer_tensor(
            "qsa_core_out",
            self.indexer.layer_id,
            "qsa",
            output[:num_queries],
        )

    def _write_main_kv(
        self, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        host = self.host_history
        if host is not None:
            host.write(
                key,
                value,
                slot_mapping,
                self.kv_cache_dtype,
                self._k_scale,
                self._v_scale,
            )
        else:
            impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
            impl.do_kv_cache_update(self, key, value, self.kv_cache, slot_mapping)

    def forward_kv_only(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Write main and indexer caches without sparse attention."""
        metadata = cast(
            dict[str, AttentionMetadata], get_forward_context().attn_metadata
        )
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        num_tokens = main_metadata.num_actual_tokens
        qkv, _ = self.qkv_proj(hidden_states)
        self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
            select=False,
        )
        _, k, v, _ = self._project_qkv_gate(qkv, positions)
        key = k.view(hidden_states.shape[0], self.num_kv_heads, self.head_dim)
        value = v.view(hidden_states.shape[0], self.num_kv_heads, self.head_dim)
        self._write_main_kv(key, value, main_metadata.slot_mapping)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor | None,
        hidden_states: torch.Tensor,
        *,
        reduce_results: bool | None = None,
        row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        indexer_input = hidden_states
        index_proj = self.indexer.index_qk_proj
        packed_qkv = getattr(self.qkv_proj, "_sm70_qsa_input_packed", None)
        packed_index = getattr(index_proj, "_sm70_qsa_input_packed", None)
        if (
            packed_qkv is not None
            and packed_index is not None
            and is_sm70_decode_graph_compiling()
        ):
            qkv, indexer_input = torch.ops.vllm.qwen38_sm70_fp16_qsa_input(
                hidden_states,
                self.qkv_proj.weight,
                index_proj.weight,
                packed_qkv,
                packed_index,
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if row_indices is not None:
            attn_output = query.new_empty(
                (row_indices.numel(), self.num_heads, self.head_dim)
            )
            self._run_qsa(
                indexer_input,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                row_indices=row_indices,
            )
        elif current_platform.opaque_attention_op():
            attn_output = torch.empty_like(query)
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                indexer_input,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                encoded_layer_name,
            )
        else:
            attn_output = torch.empty_like(query)
            qwen4_exp_qsa_with_output(
                indexer_input,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                encoded_layer_name,
            )
        flat_output = attn_output.flatten(1)
        if reduce_results is None:
            projected_output, _ = self.o_proj(flat_output)
        else:
            projected_output, _ = self.o_proj(
                flat_output, reduce_results=reduce_results
            )
        if output is not None:
            output.copy_(projected_output)
        return projected_output


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
        output_gate,
    )


def qwen4_exp_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, output_gate, layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_with_output",
]
