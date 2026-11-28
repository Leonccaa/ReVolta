# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small startup probes for the SM70 Qwen4Exp kernel cache."""

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

from .ple_layer import _warmup_sm70_ple_ngram_ids
from .qsa import _warmup_sm70_qsa_folded8, _warmup_sm70_qsa_segments


def warmup_sm70_qwen4_exp_kernels(
    vllm_config: VllmConfig, device: torch.device
) -> list[str]:
    forward_context = vllm_config.compilation_config.static_forward_context
    warmed = []
    if _warmup_sm70_qsa_folded8(forward_context, vllm_config):
        warmed.append("qsa_folded8")
    if _warmup_sm70_qsa_segments(forward_context, vllm_config):
        warmed.append("qsa_segments")
    if _warmup_sm70_ple_ngram_ids(forward_context):
        warmed.append("ple_ngram_ids")
    if _warmup_sm70_qwen38_router(vllm_config, device):
        warmed.append("qwen38_router")
    return warmed


def _warmup_sm70_qwen38_router(vllm_config: VllmConfig, device: torch.device) -> bool:
    model_config = vllm_config.model_config
    config = model_config.hf_text_config
    if (
        not envs.VLLM_SM70_QWEN38_ROUTER_TOPK
        or model_config.dtype != torch.float16
        or model_config.get_num_experts() != 512
        or config.num_experts_per_tok != 10
        or not getattr(config, "norm_topk_prob", True)
    ):
        return False

    logits = torch.zeros((10, 512), dtype=torch.float16, device=device)
    for rows in (5, 9, 10):
        probe = logits[:rows]
        fused_topk(probe, probe, topk=10, renormalize=True)
    return True
