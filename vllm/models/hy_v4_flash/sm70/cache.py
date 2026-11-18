# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

POOL_PAGE = 64


class QuerySpan(NamedTuple):
    begin: int
    end: int
    prefix: int
    length: int


@dataclass
class FlashMetadata(AttentionMetadata):
    spans: tuple[QuerySpan, ...]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    query_start_loc: torch.Tensor


class FlashMetadataBuilder(AttentionMetadataBuilder[FlashMetadata]):
    # The indexer walks requests in eager mode. Use the scheduler's exact host
    # lengths in M1 (no speculative decoding), without syncing device lengths.
    uses_physical_block_table = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMetadata:
        cm = common_attn_metadata
        spans: tuple[QuerySpan, ...] = ()
        if not cm.is_dummy_batch:
            starts = cm.query_start_loc_cpu[: cm.num_reqs + 1].tolist()
            lens = cm.seq_lens_cpu_upper_bound
            assert lens is not None
            spans = tuple(
                QuerySpan(b, e, n - (e - b), n)
                for b, e, n in zip(starts, starts[1:], lens[: cm.num_reqs].tolist())
            )
        return FlashMetadata(
            spans,
            cm.block_table_tensor,
            cm.slot_mapping,
            0 if cm.is_dummy_batch else cm.num_actual_tokens,
            cm.query_start_loc,
        )


class FlashSparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "float16"]

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
        return (num_blocks, block_size, head_size)

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
        return False

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
