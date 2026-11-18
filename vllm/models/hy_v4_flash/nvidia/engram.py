# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Engram hash state."""

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.models.hy_v4_flash.common.ngram_hash import (
    hash_ngrams_torch,
    hash_ngrams_triton,
)


class FlashEngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids or ())
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // self.n_heads
        self.pad_token_id = config.engram_pad_id
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        groups = config.engram_layer_head_vocab_sizes
        self.primes = tuple(
            groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids
        )
        flat = [[p for order in group for p in order] for group in self.primes]
        self.offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64
        )


class FlashNgramHashState(nn.Module):
    def __init__(self, config, *, dummy=False):
        super().__init__()
        self.layout = FlashEngramLayout(config)
        self.lookback_depth = self.layout.max_ngram_size - 1
        layers = len(self.layout.layer_ids)
        self.register_buffer(
            "token_maps",
            torch.zeros((layers, config.vocab_size), dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "map_sizes", torch.zeros(layers, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "multipliers",
            torch.zeros((layers, self.layout.max_ngram_size), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "primes",
            torch.tensor(self.layout.primes, dtype=torch.int64).flatten(1),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.tensor(self.layout.offsets, dtype=torch.int64),
            persistent=False,
        )
        if dummy:
            identity = torch.arange(config.vocab_size, dtype=torch.int32)
            for layer_id in self.layout.layer_ids:
                self.load_lookup(identity, layer_id)

    def load_lookup(self, value, layer_id):
        index = self.layout.layer_ids.index(layer_id)
        compressed_vocab_size = int(value.max().item()) + 1
        bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
        values = (
            np.random.default_rng(self.layout.seed + 10007 * layer_id).integers(
                0, bound, size=self.layout.max_ngram_size, dtype=np.int64
            )
            * 2
            + 1
        )
        self.token_maps[index, : value.numel()].copy_(value)
        self.map_sizes[index] = value.numel()
        self.multipliers[index].copy_(torch.from_numpy(values))

    @eager_break_during_capture
    def prepare_hashes_with_output(
        self, input_ids, lookback_token_ids, layer_name, out
    ):
        metadata = get_forward_context().attn_metadata
        layer_metadata = metadata[layer_name] if isinstance(metadata, dict) else None
        out.copy_(self.prepare_hashes(input_ids, lookback_token_ids, layer_metadata))

    def prepare_hashes(self, input_ids, lookback_token_ids, metadata):
        if metadata is None or metadata.num_actual_tokens == 0:
            return input_ids.new_full(
                (len(self.layout.layer_ids), input_ids.numel(), self.offsets.shape[1]),
                -1,
                dtype=torch.int32,
            )
        fn = hash_ngrams_triton if input_ids.is_cuda else hash_ngrams_torch
        return fn(
            input_ids,
            metadata.query_start_loc,
            lookback_token_ids,
            self.token_maps,
            self.map_sizes,
            self.multipliers,
            self.primes,
            self.offsets,
            self.layout.pad_token_id,
            metadata.num_actual_tokens,
        )
