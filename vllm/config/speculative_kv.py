# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission for independent EAGLE/MTP KV storage, before loading weights."""

from typing import TYPE_CHECKING

from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def validate_speculative_kv_cache_dtype(config: "VllmConfig") -> None:
    spec = config.speculative_config
    if (
        config.is_speculative_draft
        or spec is None
        or spec.method not in ("eagle", "eagle3", "mtp")
        or spec.kv_cache_dtype is None
    ):
        return
    target = config.model_config
    draft = spec.draft_model_config
    if target is None or draft is None:
        return
    target_dtype = kv_cache_dtype_str_to_dtype(config.cache_config.cache_dtype, target)
    draft_dtype = kv_cache_dtype_str_to_dtype(spec.kv_cache_dtype, draft)
    target_storage = config.cache_config.cache_dtype
    draft_storage = spec.kv_cache_dtype
    same_format = target_storage == draft_storage or {
        target_storage,
        draft_storage,
    } <= {"fp8", "fp8_e4m3"}
    if same_format or (
        target_storage in ("auto", "float16", "bfloat16")
        and draft_storage in ("auto", "float16", "bfloat16")
        and target_dtype == draft_dtype
    ):
        return

    reason = None
    parallel = config.parallel_config
    qsa = bool(
        getattr(target.hf_text_config, "indexer_n_heads", 0)
        and getattr(draft.hf_text_config, "indexer_n_heads", 0)
    )
    # QSA's FP16/BF16 reader requires storage to match the query dtype.
    if config.cache_config.cache_dtype not in (
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ):
        reason = "the target KV layout does not support an independent draft dtype"
    elif spec.kv_cache_dtype not in ("auto", "float16", "bfloat16"):
        reason = "a quantized draft with a different target dtype is not supported"
    elif draft_dtype != draft.dtype:
        reason = "the unquantized draft cache must match the draft activation dtype"
    elif not config.use_v2_model_runner:
        reason = "independent draft KV storage requires Model Runner V2"
    elif target.use_mla or draft.use_mla:
        reason = "MLA cache layouts do not support this independent draft dtype"
    elif not qsa:
        # Other backends can use the target cache dtype when scheduling the
        # draft (e.g. FlashAttention AOT metadata). Do not admit them merely
        # because their tensors can be allocated with heterogeneous dtypes.
        reason = "independent draft KV storage is currently supported only for QSA"
    elif parallel.pipeline_parallel_size != 1:
        reason = "independent draft KV storage does not support pipeline parallelism"
    elif parallel.prefill_context_parallel_size != 1:
        reason = "independent draft KV storage does not support prefill CP"
    elif parallel.decode_context_parallel_size != 1 and not (
        spec.method == "mtp" and parallel.decode_context_parallel_size == 2
    ):
        reason = "mixed-dtype DCP is supported only for native QSA MTP with DCP2"
    elif config.scheduler_config.disable_hybrid_kv_cache_manager:
        reason = "independent draft pages require the hybrid KV cache manager"
    elif (
        config.kv_transfer_config is not None
        and config.kv_transfer_config.kv_connector is not None
        and (
            config.kv_transfer_config.kv_connector != "OffloadingConnector"
            or config.kv_transfer_config.kv_connector_module_path is not None
            or config.kv_transfer_config.kv_role != "kv_both"
        )
    ):
        reason = (
            "mixed-dtype KV supports only the built-in OffloadingConnector (kv_both)"
        )
    if reason is not None:
        raise ValueError(
            f"speculative_config.kv_cache_dtype={spec.kv_cache_dtype!r} with "
            f"target cache_dtype={config.cache_config.cache_dtype!r}: {reason}. "
            "Use matching target/draft KV dtypes or remove the incompatible option."
        )
