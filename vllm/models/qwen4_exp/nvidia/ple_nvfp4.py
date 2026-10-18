# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 PLE: TP-sharded pinned packed rows, GPU gather then dequantization."""

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_accelerator_view_from_cpu_tensor,
)

logger = init_logger(__name__)


@triton.jit
def _gather_rows(table, ids, output, ROW_BYTES: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, B)
    idx = tl.load(ids + row)
    ptr = table.to(tl.int64).to(tl.pointer_type(tl.uint8))
    data = tl.load(ptr + idx * ROW_BYTES + offsets, offsets < ROW_BYTES, 0)
    tl.store(output + row * ROW_BYTES + offsets, data, offsets < ROW_BYTES)


@triton.jit
def _dequant_rows(rows, global_scale, output, D: tl.constexpr, B: tl.constexpr):
    # Both values and scales are already on the GPU. Explicit integer decode
    # works on SM70 too; no FP8 instructions or activation quantization.
    row = tl.program_id(0)
    col = tl.arange(0, B)
    stride: tl.constexpr = D // 2 + D // 16
    packed = tl.load(rows + row * stride + col // 2, col < D, 0)
    code = (packed >> ((col % 2) * 4)) & 15
    mag = code & 7
    value = tl.where(
        mag < 2,
        mag * 0.5,
        tl.where(
            mag < 4,
            1.0 + (mag - 2) * 0.5,
            tl.where(mag < 6, 2.0 + (mag - 4), 4.0 + (mag - 6) * 2.0),
        ),
    )
    value = tl.where(code & 8 != 0, -value, value)
    sf = tl.load(rows + row * stride + D // 2 + col // 16, col < D, 0).to(tl.int32)
    exp = (sf >> 3) & 15
    mant = sf & 7
    scale = tl.where(
        exp == 0,
        mant.to(tl.float32) * (2.0**-9),
        (1.0 + mant.to(tl.float32) / 8.0) * tl.exp2(exp - 7.0),
    )
    value = (value * scale) * tl.load(global_scale).to(tl.float32)
    tl.store(output + row * D + col, value, col < D)


def nvfp4_ple_gather(
    ids: torch.Tensor, output: torch.Tensor, global_scale: torch.Tensor, layer_name: str
) -> None:
    if not ids.numel():
        return
    layer = get_forward_context().no_compile_layers[layer_name]
    rows = torch.empty(
        (ids.numel(), layer.row_bytes), dtype=torch.uint8, device=ids.device
    )
    _gather_rows[(ids.numel(),)](
        layer._host_ptr,
        ids,
        rows,
        layer.row_bytes,
        triton.next_power_of_2(layer.row_bytes),
    )
    _dequant_rows[(ids.numel(),)](
        rows,
        global_scale,
        output,
        layer.embedding_dim,
        triton.next_power_of_2(layer.embedding_dim),
        enable_fp_fusion=False,
    )


def _fake_gather(ids, output, global_scale, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="nvfp4_ple_gather",
    op_func=nvfp4_ple_gather,
    mutates_args=["output"],
    fake_impl=_fake_gather,
)


class NVFP4PLEMethod(QuantizeMethodBase):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **attrs,
    ):
        d = input_size_per_partition
        if d % 16:
            raise ValueError("NVFP4 PLE requires embedding_dim divisible by 16")
        loader = attrs.get("weight_loader")
        # Zero-row parameter keeps model loader bookkeeping without a second
        # table. Packed rows are allocated only after leaving the meta context.
        weight = ModelWeightParameter(
            data=torch.empty((0, d // 2), dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=loader,
        )
        weight._vllm_keep_on_cpu = True
        layer.register_parameter("weight", weight)
        layer.register_parameter(
            "weight_scale_2",
            PerTensorScaleParameter(
                data=torch.ones(1, dtype=torch.float32),
                weight_loader=loader,
            ),
        )

    def apply(self, layer, x, bias=None):
        raise NotImplementedError("NVFP4 PLE supports embedding lookup only")

    def embedding(self, layer, input_):
        out = torch.empty(
            (*input_.shape, layer.embedding_dim),
            device=input_.device,
            dtype=layer.output_dtype,
        )
        torch.ops.vllm.nvfp4_ple_gather(
            input_.reshape(-1),
            out,
            layer.weight_scale_2,
            layer.layer_name,
        )
        return out

    def process_weights_after_loading(self, layer):
        layer.prepare()


class NVFP4PinnedPLEEmbedding(VocabParallelEmbedding):
    """Host-only resident PLE. Existing FP8 placement/offload stays independent."""

    def __init__(
        self, num_embeddings, embedding_dim, params_dtype, padding_size, prefix
    ):
        with torch.device("meta"):
            super().__init__(
                num_embeddings,
                embedding_dim,
                params_dtype=params_dtype,
                padding_size=padding_size,
                prefix=prefix,
                quant_method=NVFP4PLEMethod(),
            )
        self.weight = ModelWeightParameter(
            data=torch.empty((0, embedding_dim // 2), dtype=torch.uint8, device="cpu"),
            input_dim=1,
            output_dim=0,
            weight_loader=self.weight_loader,
        )
        self.weight._vllm_keep_on_cpu = True
        self.weight_scale_2 = PerTensorScaleParameter(
            data=torch.ones(1, dtype=torch.float32),
            weight_loader=self.weight_loader,
        )
        self.output_dtype = params_dtype or torch.float16
        self.row_bytes = embedding_dim // 2 + embedding_dim // 16
        self.layer_name = prefix
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self
        self.storage: torch.Tensor | None = None
        self._host_view: torch.Tensor | None = None
        self._host_ptr = 0
        self._loaded_parts = set()

    def materialize(self):
        if self.storage is None:
            self.storage = torch.zeros(
                (self.num_embeddings_per_partition, self.row_bytes),
                dtype=torch.uint8,
                device="cpu",
            )
            # Share the established registration/lifetime handling of FP8 PLE.
            from .ple_layer import _pin_host_tensor_exact

            _pin_host_tensor_exact(self.storage)
            logger.info(
                "NVFP4 PLE pinned table: TP rank %d, %d rows, %d bytes/row, "
                "%.3f GiB; gather packed rows then dequantize on GPU",
                self.weight.tp_rank,
                self.num_embeddings_per_partition,
                self.row_bytes,
                self.storage.numel() / 2**30,
            )

    def load_checkpoint_tensor(self, name, tensor, split_parts):
        from ..common.ple import copy_ple_embedding_shard_

        prefix = "ngram_embedding.shard_"
        if name == "ngram_embedding.weight_scale_2":
            if tensor.numel() != 1 or not bool(
                torch.isfinite(tensor).all() and (tensor > 0).all()
            ):
                raise ValueError("NVFP4 PLE global scale must be finite and positive")
            self.weight_scale_2.data.copy_(tensor.reshape(1))
            self._loaded_parts.add("global")
            return {"ngram_embedding.weight_scale_2"}
        if not name.startswith(prefix):
            return None
        index_text, field = name[len(prefix) :].split(".", 1)
        if not index_text.isdigit() or field not in ("weight", "weight_scale"):
            raise ValueError(f"Unknown NVFP4 PLE tensor {name}")
        index = int(index_text)
        if not 0 <= index < split_parts:
            raise ValueError(f"Invalid NVFP4 PLE shard {index}")
        shard_size = (self.org_vocab_size + split_parts - 1) // split_parts
        start = index * shard_size
        count = max(0, min(shard_size, self.org_vocab_size - start))
        width = self.embedding_dim // (2 if field == "weight" else 16)
        dtype = torch.uint8 if field == "weight" else torch.float8_e4m3fn
        if tensor.dtype != dtype or tuple(tensor.shape) != (count, width):
            raise ValueError(f"Invalid NVFP4 PLE {name}: {tensor.dtype} {tensor.shape}")
        if field == "weight_scale" and bool((tensor.view(torch.uint8) > 126).any()):
            raise ValueError("NVFP4 PLE scales must be finite nonnegative E4M3")
        part = (index, field)
        if part in self._loaded_parts:
            raise ValueError(f"Duplicate NVFP4 PLE shard component: {name}")
        self.materialize()
        assert self.storage is not None
        dest = (
            self.storage[:, :width]
            if field == "weight"
            else self.storage[:, self.embedding_dim // 2 :]
        )
        copy_ple_embedding_shard_(
            dest,
            tensor.view(torch.uint8),
            checkpoint_start=start,
            tp_start=self.shard_indices.org_vocab_start_index,
            tp_end=self.shard_indices.org_vocab_end_index,
        )
        self._loaded_parts.add((index, field))
        return {"ngram_embedding.weight"}

    def check_loaded(self, split_parts):
        expected = {"global"} | {
            (i, field)
            for i in range(split_parts)
            for field in ("weight", "weight_scale")
        }
        if self._loaded_parts != expected:
            raise ValueError(
                f"Incomplete NVFP4 PLE checkpoint: {expected - self._loaded_parts}"
            )

    def prepare(self):
        self.materialize()
        assert self.storage is not None
        if self._host_view is None:
            self._host_view = get_accelerator_view_from_cpu_tensor(self.storage)
            self._host_ptr = self._host_view.data_ptr()
