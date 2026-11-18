# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Engram n-gram hashing and TP-sharded host embeddings."""

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.models.hy_v4_flash.common.ngram_hash import (
    hash_ngrams_torch,
    hash_ngrams_triton,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    _e4m3fn_byte_to_float,
    _pin_host_tensor_exact,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_accelerator_view_from_cpu_tensor,
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


def head_range(sizes, tp, rank):
    """Partition complete hash heads, balancing their row counts."""
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


@triton.jit
def _flash_engram_gather_kernel(
    storage_ptr,
    ids_ptr,
    output_ptr,
    vocab_start,
    local_rows,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.load(ids_ptr + row).to(tl.int64) - vocab_start
    owned = (index >= 0) & (index < local_rows)
    offsets = tl.arange(0, BLOCK_D)
    storage = storage_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    raw = tl.load(storage + index * dim + offsets, owned & (offsets < dim), other=0)
    scales = (storage + local_rows.to(tl.int64) * dim).to(tl.pointer_type(tl.bfloat16))
    scale = tl.load(scales + index, owned, other=0).to(tl.float32)
    decoded = _e4m3fn_byte_to_float(raw) * scale
    tl.store(output_ptr + row * dim + offsets, decoded, offsets < dim)


def flash_engram_gather(
    indices: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    table = get_forward_context().no_compile_layers[layer_name]
    _flash_engram_gather_kernel[(indices.numel(),)](
        table._uva_ptr,
        indices,
        output,
        table.vocab_start,
        table.local_rows,
        dim=table.dim,
        BLOCK_D=triton.next_power_of_2(table.dim),
        num_warps=4,
    )


def flash_engram_gather_fake(
    indices: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None:
    return


direct_register_custom_op(
    op_name="hy4_flash_engram_gather",
    op_func=flash_engram_gather,
    mutates_args=["output"],
    fake_impl=flash_engram_gather_fake,
)


class FlashEmbedding(nn.Module):
    """Rank-local complete heads in one precisely registered host allocation."""

    def __init__(self, sizes, dim, vllm_config, prefix):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_start, self.head_end = head_range(
            sizes, self.tp_size, get_tensor_model_parallel_rank()
        )
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.local_rows = self.vocab_end - self.vocab_start
        self.n_hash_cols = len(sizes)
        self.dim = dim
        self.layer_name = prefix
        self.use_cuda = vllm_config.device_config.device_type == "cuda"
        self._uva_ptr = 0
        self._uva_view = None
        if self.use_cuda:
            vllm_config.compilation_config.static_forward_context[prefix] = self
        self.storage = torch.empty(0, dtype=torch.uint8, device="cpu")
        if vllm_config.load_config.load_format == "dummy":
            self._allocate()
            self.storage.zero_()

    def _allocate(self):
        self.storage = torch.empty(
            self.local_rows * (self.dim + 2), dtype=torch.uint8, device="cpu"
        )
        if self.use_cuda:
            _pin_host_tensor_exact(self.storage)
            self._uva_view = get_accelerator_view_from_cpu_tensor(self.storage)
            self._uva_ptr = self._uva_view.data_ptr()

    def load_from_safetensors(self, path, weight_name):
        """Read only this rank's row range, in bounded blocks, from one shard."""
        self._allocate()
        weight = self.storage[: self.local_rows * self.dim].view(
            self.local_rows, self.dim
        )
        scale = self.storage[self.local_rows * self.dim :].view(torch.bfloat16)
        with safe_open(path, framework="pt", device="cpu") as reader:
            weight_slice = reader.get_slice(weight_name)
            scale_slice = reader.get_slice(weight_name + "_scale_inv")
            for start in range(self.vocab_start, self.vocab_end, 131072):
                end = min(start + 131072, self.vocab_end)
                local_start = start - self.vocab_start
                local_end = end - self.vocab_start
                weight[local_start:local_end].copy_(
                    weight_slice[start:end, :].view(torch.uint8)
                )
                scale[local_start:local_end].copy_(scale_slice[start:end])

    def forward(self, hashes):
        indices = hashes[:, self.head_start : self.head_end].long().contiguous()
        output = torch.empty(
            (*indices.shape, self.dim), dtype=torch.float32, device=hashes.device
        )
        if indices.numel() == 0:
            return output
        if hashes.is_cuda:
            torch.ops.vllm.hy4_flash_engram_gather(
                indices.flatten(), output.view(-1, self.dim), self.layer_name
            )
        else:
            local = indices - self.vocab_start
            valid = (local >= 0) & (local < self.local_rows)
            local = torch.where(valid, local, 0)
            weights = self.storage[: self.local_rows * self.dim].view(
                self.local_rows, self.dim
            )
            scales = self.storage[self.local_rows * self.dim :].view(torch.bfloat16)
            rows = weights[local].view(torch.float8_e4m3fn).float()
            output = rows * scales[local].float().unsqueeze(-1)
            output.masked_fill_(~valid.unsqueeze(-1), 0)
        return output
