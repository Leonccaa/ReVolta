# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy draft-only channel-QPN8 projection adapted from 1Cat #821."""

import torch

from vllm import _sm70_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
    prepare_channel_qpn8_weight,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = init_logger(__name__)


class MTPQPN8LogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, weight: torch.Tensor) -> None:
        super().__init__(vocab_size)
        codes, scales = prepare_channel_qpn8_weight(weight)
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)

    @torch.inference_mode()
    def reload_pack(self, weight: torch.Tensor) -> None:
        # Captured graphs retain the original buffer addresses.
        codes, scales = prepare_channel_qpn8_weight(weight)
        self.codes.copy_(codes)
        self.scales.copy_(scales)

    def _get_local_top1_pair(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x = hidden_states
        if (
            embedding_bias is not None
            or x.ndim != 2
            or x.shape[1] != 2560
            or not 1 <= x.shape[0] <= 8
            or x.dtype != torch.float16
            or x.device != self.codes.device
        ):
            return super()._get_local_top1_pair(lm_head, x, embedding_bias)

        rows = x.shape[0]
        logits = x.new_empty((rows, 62080))
        _sm70_ops.fp8_qpn8_gemm_sm70_out(
            logits, x.contiguous(), self.codes, self.scales, 8, 2, True
        )
        shard = lm_head.shard_indices
        valid, start = shard.num_org_elements, shard.org_vocab_start_index
        pairs = logits.new_empty((rows, 2), dtype=torch.float32)
        partial = logits.new_empty((rows, (valid + 511) // 512, 2), dtype=torch.float32)
        torch.ops._C.qwen38_mtp_local_top1_sm70_out(
            pairs, partial, logits, valid, start
        )
        logger.info_once("SM70 greedy MTP draft QPN8 head enabled.")
        return pairs
