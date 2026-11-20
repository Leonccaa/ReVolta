# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    FP8_DS_MLA_BYTES,
    AttentionSpec,
    CircularBufferSpec,
    MLAAttentionSpec,
)

POOL_PAGE = 64


class QuerySpan(NamedTuple):
    begin: int
    end: int
    prefix: int
    length: int


class IndexPoolDecodeWorkspace(NamedTuple):
    raw: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    topk_values: torch.Tensor
    topk_groups: torch.Tensor
    sorted_groups: torch.Tensor
    sort_order: torch.Tensor


@dataclass
class FlashMetadata(AttentionMetadata):
    spans: tuple[QuerySpan, ...]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    decode: bool
    counts: torch.Tensor | None
    index_workspace: IndexPoolDecodeWorkspace | None


class FlashMetadataBuilder(AttentionMetadataBuilder[FlashMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    uses_physical_block_table = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.counts: torch.Tensor | None = None
        self.index_workspace: IndexPoolDecodeWorkspace | None = None
        if device.type == "cuda" and not isinstance(kv_cache_spec, CircularBufferSpec):
            rows = vllm_config.scheduler_config.max_num_seqs
            if (
                isinstance(kv_cache_spec, MLAAttentionSpec)
                and (ratio := kv_cache_spec.compress_ratio) > 1
            ):
                assert vllm_config.model_config is not None
                model = vllm_config.model_config
                history = model.hf_text_config.compressed_indexer_group_topk - 1
                groups = cdiv(model.max_model_len, ratio)
                # Every sparse layer in this attention group uses the same scratch.
                dcp = vllm_config.parallel_config.decode_context_parallel_size > 1
                topk_shape = (
                    (rows, history) if not dcp and groups > history + 1 else (rows, 0)
                )
                self.index_workspace = IndexPoolDecodeWorkspace(
                    torch.empty(
                        (rows, ratio, kv_cache_spec.head_size),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(rows, dtype=torch.int64, device=device),
                    torch.empty(
                        (rows, groups if topk_shape[1] else 0),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(topk_shape, dtype=torch.float32, device=device),
                    torch.empty(topk_shape, dtype=torch.int64, device=device),
                    torch.empty(topk_shape, dtype=torch.int64, device=device),
                    torch.empty(topk_shape, dtype=torch.int64, device=device),
                )
            else:
                self.counts = torch.empty(rows, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMetadata:
        cm = common_attn_metadata
        decode = cm.max_query_len == 1 and (
            cm.is_dummy_batch or cm.is_prefilling is None or not cm.is_prefilling.any()
        )
        spans: tuple[QuerySpan, ...] = ()
        if not cm.is_dummy_batch and not (decode and cm.seq_lens.is_cuda):
            starts = cm.query_start_loc_cpu[: cm.num_reqs + 1].tolist()
            lens = cm.seq_lens_cpu_upper_bound
            assert lens is not None
            spans = tuple(
                QuerySpan(b, e, n - (e - b), n)
                for b, e, n in zip(starts, starts[1:], lens[: cm.num_reqs].tolist())
            )
        # The runner owns fixed device buffers for all four inputs. Its padded
        # rows have zero sequence length and PAD slots, including dummy writes.
        return FlashMetadata(
            spans,
            cm.block_table_tensor,
            cm.slot_mapping,
            cm.num_actual_tokens if decode or not cm.is_dummy_batch else 0,
            cm.query_start_loc,
            cm.seq_lens,
            decode,
            self.counts,
            self.index_workspace,
        )


class FlashSparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "HYV4_FLASH_SM70_SPARSE"

    @staticmethod
    def get_impl_cls() -> type:
        # Explicit model-owned dispatch: indexer and window metadata feed
        # the same paged kernel; no generic AttentionImpl is constructed.
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[FlashMetadataBuilder]:
        return FlashMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        row_size = FP8_DS_MLA_BYTES if cache_dtype_str == "fp8_ds_mla" else head_size
        return (num_blocks, block_size, row_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_dcp(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False


class FlashSWABackend(FlashSparseBackend):
    @staticmethod
    def get_name() -> str:
        return "HYV4_FLASH_SM70_SWA"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True


class FlashCache(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        spec: AttentionSpec,
        backend: type[AttentionBackend] = FlashSparseBackend,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.backend = backend
        self.kv_cache: torch.Tensor = torch.empty(0)
        vllm_config.compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> AttentionSpec:
        return self.spec
