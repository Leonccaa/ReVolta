# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy draft-only channel-QPN8 projection adapted from 1Cat #821."""

import torch

from vllm import _sm70_ops
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
    prepare_channel_qpn8_weight,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class MTPQPN8LogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, weight: torch.Tensor) -> None:
        super().__init__(vocab_size)
        codes, scales = prepare_channel_qpn8_weight(weight)
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)

    def _get_local_top1_pair(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x = hidden_states
        rows = x.shape[0]
        if not 1 <= rows <= 8:
            return super()._get_local_top1_pair(lm_head, x, embedding_bias)

        logits = x.new_empty((rows, lm_head.weight.shape[0]))
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
        return pairs
