# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time expansion for a vision row shard that crosses NVFP4 groups."""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.utils.fp4_local import (
    dequantize_nvfp4_cpu,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter, set_weight_attrs


class ModelOptNvFp4VisionFC2Method(LinearMethodBase):
    """Keep checkpoint groups intact until expansion, then slice dense columns.

    Only the mixed-precision config's SM70/TP4 vision-fc2 gate selects this
    method. fc1 keeps its original column partition, including its bias.
    RowParallelLinear still owns the all-reduce and rank-zero bias addition.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if (
            layer.tp_size != 4
            or input_size % 16
            or input_size_per_partition * layer.tp_size != input_size
            or input_size_per_partition % 16 == 0
            or output_partition_sizes != [output_size]
            or params_dtype != torch.float16
        ):
            raise ValueError("Invalid SM70 NVFP4 vision fc2 expansion geometry/dtype")

        # Full packed staging is small (2.66 MiB/layer for K=4304,N=1152).
        # Slicing packed scales at K/TP would truncate a boundary group.
        # Use unsplit loaders for all three tensors, independent of load order.
        for name, shape, dtype in (
            ("weight", (output_size, input_size // 2), torch.uint8),
            ("weight_scale", (output_size, input_size // 16), torch.float8_e4m3fn),
            ("weight_scale_2", (1,), torch.float32),
        ):
            param = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(param, {"weight_loader": default_weight_loader})
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight.device
        # Expand on CPU, round once to FP16, then slice. In particular, do not
        # round K/TP down to a group count or reinterpret a partial group.
        weight = dequantize_nvfp4_cpu(
            layer.weight, layer.weight_scale, layer.weight_scale_2
        )
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("NVFP4 vision fc2 weights are not finite in FP16")
        width = layer.input_size_per_partition
        shard = weight.narrow(1, layer.tp_rank * width, width).contiguous()
        replace_parameter(layer, "weight", shard.to(device))
        del layer.weight_scale
        del layer.weight_scale_2

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)
