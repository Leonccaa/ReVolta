# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA cache format selection without loading model weights."""

from vllm.revolta_profile import has_qsa_capability
from vllm.v1.attention.kv_codecs import (
    FP8_E4M3_PER_TOKEN_HEAD,
    INT8,
    INT8_PER_TOKEN_HEAD,
)


def resolve_qsa_int8_aliases(cfg) -> None:
    """QSA uses the same planar RNE codec for int8_per_token_head and int8.

    Normalize before cache planning so both spellings share the existing INT8
    page layout, kernels and cache identity. Other backends resolve their own
    per-token-head layout.
    """
    if not has_qsa_capability(cfg.model_config):
        return
    if cfg.cache_config.cache_dtype == INT8_PER_TOKEN_HEAD.name:
        cfg.cache_config.cache_dtype = INT8.name
    spec = cfg.speculative_config
    if (
        spec is not None
        and spec.method == "mtp"
        and spec.kv_cache_dtype == INT8_PER_TOKEN_HEAD.name
    ):
        spec.kv_cache_dtype = INT8.name


def resolve_qsa_auto_cache_dtype(cfg) -> bool:
    # A drafter keeps the dtype its loader chose and shares the target policy.
    if getattr(cfg, "is_speculative_draft", False):
        return False
    policy = cfg.kernel_config
    policy.qsa_auto_e4m3_active = False
    model = cfg.model_config
    reason = None
    if not policy.qsa_auto_e4m3:
        reason = "disabled by KernelConfig"
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
    if reason is not None:
        return False

    from vllm.platforms import current_platform

    if current_platform.is_device_capability(70):
        cfg.cache_config.cache_dtype = INT8.name
        policy.qsa_auto_e4m3_reason = "SM70 defaults to INT8"
    else:
        cfg.cache_config.cache_dtype = FP8_E4M3_PER_TOKEN_HEAD.name
        policy.qsa_auto_e4m3_active = True
    cfg.cache_config.cache_dtype_from_checkpoint = False
    return True
