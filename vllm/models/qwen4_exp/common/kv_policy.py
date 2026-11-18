# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA cache format selection without loading model weights."""

from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD


def resolve_qsa_auto_e4m3(cfg) -> bool:
    policy = cfg.kernel_config
    policy.qsa_auto_e4m3_active = False
    model = cfg.model_config
    reason = None
    if not policy.qsa_auto_e4m3:
        reason = "disabled by KernelConfig"
    elif getattr(cfg, "is_speculative_draft", False):
        # A drafter's config copies the target model config; the drafter keeps
        # the KV dtype its loader chose (e.g. FP16 for SM70 DFlash).
        reason = "speculative draft keeps its own KV dtype"
    elif model is None or not getattr(model.hf_text_config, "indexer_n_heads", None):
        reason = "no QSA selector metadata"
    elif cfg.cache_config.cache_dtype != "auto":
        reason = "explicit KV dtype takes precedence"
    elif cfg.cache_config.calculate_kv_scales:
        reason = "runtime scale calculation takes precedence"
    elif cfg.parallel_config.prefill_context_parallel_size != 1:
        reason = "QSA does not support prefill context parallelism"
    else:
        from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_e4m3_capability_reason

        reason = qsa_e4m3_capability_reason(model.dtype)
    policy.qsa_auto_e4m3_reason = reason
    policy.qsa_auto_e4m3_active = reason is None
    if reason is None:
        cfg.cache_config.cache_dtype = FP8_E4M3_PER_TOKEN_HEAD.name
        cfg.cache_config.cache_dtype_from_checkpoint = False
    return policy.qsa_auto_e4m3_active
