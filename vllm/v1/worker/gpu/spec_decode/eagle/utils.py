# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model

logger = init_logger(__name__)


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """Share when the draft has no own copy, or its copy matches the target."""

    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    # torch.equal on GPU allocates a bool mask the size of the input.
    # Use the faster GPU path when there is plenty of headroom;
    # otherwise compare on CPU.
    w = draft.weight
    if w.is_cuda and torch.cuda.mem_get_info(w.device)[0] < w.numel() * 2:
        return torch.equal(w.cpu(), target.weight.cpu())
    return torch.equal(w, target.weight)


def get_target_lm_head(target_model: nn.Module, target_language_model: nn.Module):
    """Return the language-model head for plain and conditional targets."""
    return getattr(target_language_model, "lm_head", None) or getattr(
        target_model, "lm_head", None
    )


def _share_mtp_topk_buffer(
    target_inner: nn.Module, draft_inner: nn.Module, vllm_config: VllmConfig
) -> None:
    """Reuse the target workspace for one sequential MTP QSA layer.

    This loader is used by Model Runner V2's EagleSpeculator. Its MTP
    propose runs after the target forward, on the same stream, without
    calling the target between draft selection, compaction and decode.
    Overlapping/async paths must keep independent workspaces.
    """
    iter_qsa = getattr(draft_inner, "_iter_qsa_attentions", None)
    if iter_qsa is None:
        return

    speculative_config = vllm_config.speculative_config
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    reason = None
    attentions = list(iter_qsa())
    if getattr(speculative_config, "method", None) != "mtp":
        reason = "only the sequential MTP propose path is eligible"
    elif getattr(scheduler_config, "async_scheduling", None) is not False:
        reason = "synchronous scheduling must be explicitly resolved"
    elif len(attentions) != 1:
        # Each layer's step-0 indices survive until the last draft step.
        # Sharing across multiple draft layers would overwrite those indices.
        reason = "exactly one draft QSA layer is required"
    else:
        shared = getattr(target_inner, "topk_indices_buffer", None)
        own = attentions[0].topk_indices_buffer
        if (
            not isinstance(shared, torch.Tensor)
            or not isinstance(own, torch.Tensor)
            or shared.dtype != torch.int32
            or own.dtype != torch.int32
            or shared.ndim != 2
            or own.ndim != 2
            or shared.shape[0] < own.shape[0]
            or shared.shape[1] != own.shape[1]
            or shared.device != own.device
        ):
            reason = "target and draft workspace shape, dtype or device differ"
        else:
            # get_model has finished device/dtype transforms; bind before
            # warmup and graph capture. Assignment preserves the draft's
            # nonpersistent buffer registration (and named_buffers alias).
            # Any later allocating Module._apply needs rebinding before capture.
            attentions[0].topk_indices_buffer = shared
            logger.info("Sharing target top-k workspace with the MTP QSA layer.")
            return

    logger.info("Keeping private draft QSA top-k workspace: %s.", reason)


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    draft_cache_dtype = getattr(speculative_config, "kv_cache_dtype", None)
    draft_vllm_config = vllm_config
    if (
        draft_cache_dtype is not None
        and draft_cache_dtype != vllm_config.cache_config.cache_dtype
    ):
        # Keep the inherited/same-dtype path identical, including its config
        # identity. Scope the override before rerunning target-only admission.
        draft_vllm_config = replace(
            vllm_config,
            is_speculative_draft=True,
            cache_config=replace(
                vllm_config.cache_config,
                cache_dtype=draft_cache_dtype,
            ),
        )
        logger.info_once(
            "Using %s draft KV cache while the target uses %s.",
            draft_cache_dtype,
            vllm_config.cache_config.cache_dtype,
        )
    with set_model_tag("eagle_head"):
        eagle_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = eagle_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            eagle_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(eagle_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del eagle_model.lm_head
        eagle_model.lm_head = target_lm_head

        # MTP layers route logits through layer.shared_head.head, not
        # eagle_model.lm_head, so the per-layer copies need fixing up too.
        layers = getattr(draft_inner, "layers", None)
        if layers is not None:
            items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
            for layer in items:
                sh = getattr(layer, "shared_head", None)
                if sh is not None and hasattr(sh, "head"):
                    del sh.head
                    sh.head = target_lm_head

    # Preserve the predictor-level alias; QSA reads its layer-local attribute.
    if hasattr(target_inner, "topk_indices_buffer"):
        if hasattr(draft_inner, "topk_indices_buffer"):
            del draft_inner.topk_indices_buffer
        draft_inner.topk_indices_buffer = target_inner.topk_indices_buffer

    _share_mtp_topk_buffer(target_inner, draft_inner, vllm_config)

    return eagle_model
