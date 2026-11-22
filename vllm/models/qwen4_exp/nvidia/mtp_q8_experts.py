# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed Q8_0 routed experts for the standalone SM70 MTP draft."""

# Adapted from 1Cat-vLLM v1.5.4:
# https://github.com/1CatAI/1Cat-vLLM/blob/1fa9c438bc/vllm/models/qwen4_exp/nvidia/mtp_q8_experts.py

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

GGML_Q8_0 = 8
_BLOCK = 32
_BLOCK_BYTES = 34
_Q8_1_BYTES = 36
# The SM70 grouped ABI and GGUF vector route both accept up to 64 rows.
_SMALL_M_MAX_ROWS = 64


def quantize_q8_0(weight: torch.Tensor) -> torch.Tensor:
    """Pack one FP16/BF16 expert shard into Q8_0 blocks."""
    blocks = weight.float().reshape(-1, _BLOCK)
    scale = blocks.abs().amax(dim=1, keepdim=True) / 127
    inverse = torch.where(scale > 0, 1 / scale, 0)
    codes = (blocks * inverse).round().clamp_(-127, 127).to(torch.int8)
    packed = torch.cat((scale.half().view(torch.uint8), codes.view(torch.uint8)), dim=1)
    return packed.reshape(*weight.shape[:-1], weight.shape[-1] // _BLOCK * _BLOCK_BYTES)


class MTPQ80MoEMethod(FusedMoEMethodBase):
    """Quantize each checkpoint projection after slicing the local TP rank."""

    @property
    def topk_indices_dtype(self):
        return torch.int32

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        local_width = intermediate_size_per_partition
        top_k = self.moe.experts_per_token
        self.workspace_bytes = (
            _SMALL_M_MAX_ROWS
            * (hidden_size + top_k * local_width)
            // _BLOCK
            * _Q8_1_BYTES
        )
        if (
            hidden_size % _BLOCK
            or local_width % _BLOCK
            or not 1 <= top_k <= 16
            or max(hidden_size, top_k * local_width) // _BLOCK * _Q8_1_BYTES > 44 * 1024
        ):
            raise ValueError("Unsupported Q8_0 MTP expert shape or top-k")
        self.hidden = hidden_size
        self.local_width = local_width
        self.num_experts = num_experts
        self.loaded_experts: dict[str, set[int]] = {
            s: set() for s in ("w1", "w3", "w2")
        }
        for name, n, k in (
            ("w13_weight", 2 * local_width, hidden_size),
            ("w2_weight", hidden_size, local_width),
        ):
            param = nn.Parameter(
                torch.empty(
                    (num_experts, n, k // _BLOCK * _BLOCK_BYTES), dtype=torch.uint8
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, param)
            set_weight_attrs(
                param, {**extra_weight_attrs, "packed_expert_loader": self.load_expert}
            )

    def load_expert(self, layer, param, weight, shard_id, expert_id):
        local_width = self.local_width
        full_width = local_width * layer.tp_size
        is_down = shard_id == "w2"
        expected = (self.hidden, full_width) if is_down else (full_width, self.hidden)
        if weight.shape != expected or weight.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError(
                "Q8_0 MTP requires BF16 or FP16 checkpoint expert projections"
            )
        start = layer.tp_rank * local_width
        local = weight.narrow(1 if is_down else 0, start, local_width)
        target = param[expert_id]
        if not is_down:
            offset = local_width if shard_id == "w3" else 0
            target = target.narrow(0, offset, local_width)
        target.copy_(quantize_q8_0(local.to(device=target.device)))
        self.loaded_experts[shard_id].add(expert_id)

    def process_weights_after_loading(self, layer):
        if any(len(ids) != self.num_experts for ids in self.loaded_experts.values()):
            raise ValueError("Incomplete Q8_0 MTP expert checkpoint")
        if layer.w13_weight.is_cuda:
            # Reserve the small-M workspace before the runner captures graphs.
            self._workspace()
        logger.info(
            "Q8_0 MTP routed experts use %.2f MiB per rank.",
            (layer.w13_weight.numel() + layer.w2_weight.numel()) / 2**20,
        )

    def get_fused_moe_quant_config(self, layer):
        return None

    def _workspace(self) -> torch.Tensor:
        (workspace,) = current_workspace_manager().get_simultaneous(
            ((self.workspace_bytes,), torch.uint8)
        )
        return workspace

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        rows = x.shape[0]
        if rows == 0:
            return torch.empty_like(x)
        x = x.contiguous()
        topk_weights = topk_weights.float().contiguous()
        topk_ids = topk_ids.contiguous()
        if rows <= _SMALL_M_MAX_ROWS:
            out = torch.empty_like(x)
            ops.sm70_mtp_q8_0_moe_out(
                out,
                self._workspace(),
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
            )
            return out

        from vllm.model_executor.layers.quantization.gguf import fused_moe_gguf

        # The legacy GGUF MMQ skips expert IDs >255. Its vector kernel accepts
        # the full expert bank, and fused_moe_gguf selects it for M <= 64.
        out = torch.empty_like(x)
        for start in range(0, rows, _SMALL_M_MAX_ROWS):
            chunk = slice(start, start + _SMALL_M_MAX_ROWS)
            out[chunk] = fused_moe_gguf(
                x[chunk],
                layer.w13_weight,
                layer.w2_weight,
                topk_weights[chunk],
                topk_ids[chunk],
                GGML_Q8_0,
                GGML_Q8_0,
                "silu",
            )
        return out
