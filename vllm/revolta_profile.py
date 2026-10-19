# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide defaults for the FP16 QSA SM70 TP4 serving profile.

Imported before env_override imports torch, solely to remember environment
provenance. Allocator policy stays in the launcher: model/TP information is not
available here before CUDA initialization. Operator admission remains upstream.
"""

import json
import os
from typing import Any, Literal

InterconnectMode = Literal["pex_p2p", "no_p2p", "nvlink"]

COMMON_DEFAULTS = {
    "VLLM_SM70_QWEN38_WIDE_M": "1",
    "VLLM_SM70_PLE_PREFILL_CONV_CHUNK": "1",
    "VLLM_SM70_QWEN38_GDN_NO_PACK": "1",
    "VLLM_SM70_GREEDY_VERIFY": "1",
    "VLLM_SM70_SKIP_REJECTED_BATCH_PACKS": "1",
    "VLLM_SM70_QSA_SHARED_KEY": "1",
    "VLLM_SM70_QSA_COMPACT_TOPK": "1",
    "VLLM_SM70_PLE_NGRAM_BATCH": "1",
    "VLLM_SM70_PLE_INPUT_PREPARE": "1",
    "VLLM_PLE_ASYNC_REQUEST_INDICES": "1",
    "VLLM_EARLY_MODEL_INPUTS": "1",
    "VLLM_SM70_QWEN38_HYBRID_PLE": "0",
    "VLLM_USE_AOT_COMPILE": "0",
    "VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS": "1",
    "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP": "0",
}
AWQ_DEFAULTS = {
    # Writing QPN_M1 into environ preserves its explicit, fail-closed admission.
    "VLLM_SM70_AWQ_QWEN38_QPN_M1": "1",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN": "1",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS": "40",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT": "8",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_PACK_ROWS": "24",
    "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": "1",
    # Measured for the AWQ TP4 graph pool. NVFP4 retains automatic measurement.
    "VLLM_V2_CUDAGRAPH_MEM_MIB": "700",
}
CUSTOM_AR_DEFAULTS = {"VLLM_SM70_CAR_VMM_SAFE": "1"}

# Include the entire deployment inventory in the audit, even entries whose
# defaults remain upstream. In particular, never write the three FP16 flags.
UPSTREAM_AND_DEPLOYMENT = (
    "VLLM_SM70_AWQ_TURBOMIND",
    "VLLM_SM70_AWQ_WARMUP",
    "VLLM_SM70_AUX_KERNEL_WARMUP",
    "VLLM_SM70_AWQ_MOE_COMPACT_METADATA",
    "VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE",
    "VLLM_SM70_QSA_INDEXER_CUBLAS",
    "VLLM_SM70_QWEN38_FP16_GEMV",
    "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16",
    "VLLM_SM70_QWEN38_FUSED_HC_FP16",
    "VLLM_LOGGING_LEVEL",
    "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
    "VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS",
    "VLLM_SM70_DISABLE_QWEN3NEXT_SHARED_MOE_OVERLAP",
    "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8",
    "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH",
    "NCCL_GRAPH_REGISTER",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "VLLM_SM70_AWQ_MOE_INT4_ZERO",
    "VLLM_QWEN4EXP_PLE_HOST_GIB",
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT",
    "VLLM_CACHE_ROOT",
)
AUDITED_NAMES = tuple(
    sorted(
        set(COMMON_DEFAULTS)
        | set(AWQ_DEFAULTS)
        | set(CUSTOM_AR_DEFAULTS)
        | set(UPSTREAM_AND_DEPLOYMENT)
    )
)
# Diagnostic transport only: never use it to resolve an operator setting.
# The VLLM_ prefix also carries it through worker environment propagation.
_PROVENANCE_ENV = "VLLM_REVOLTA_PROFILE_PROVENANCE"


def _inherited_provenance() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(os.environ.get(_PROVENANCE_ENV, "{}"))
        if payload.get("version") != 1:
            return {}
        rows = payload["values"]
        return {
            name: row
            for name, row in rows.items()
            if name in AUDITED_NAMES
            and row["source"] in ("profile", "environment", "upstream_default")
            and isinstance(row["value"], str)
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        # A stale or malformed diagnostic must not affect serving defaults.
        return {}


_INHERITED_PROVENANCE = _inherited_provenance()
_INITIAL_ENV = {name: os.environ[name] for name in AUDITED_NAMES if name in os.environ}
_PROFILE_VALUES: dict[str, str] = {}


def _source(name: str) -> str:
    current = os.environ.get(name)
    if name in _PROFILE_VALUES and current == _PROFILE_VALUES[name]:
        return "profile"
    inherited = _INHERITED_PROVENANCE.get(name)
    if inherited is not None and current == inherited["value"]:
        return inherited["source"]
    if name in _INITIAL_ENV or (
        current is not None
        and (
            inherited is not None
            or name in _PROFILE_VALUES
            or (not name.startswith("VLLM_") and name != "NCCL_GRAPH_REGISTER")
        )
    ):
        return "environment"
    return "upstream_default"


def _export_provenance() -> None:
    # Refresh at the final audit too, after upstream materializes its defaults.
    os.environ[_PROVENANCE_ENV] = json.dumps(
        {
            "version": 1,
            "values": {
                name: {"value": os.environ.get(name), "source": _source(name)}
                for name in AUDITED_NAMES
                if name in os.environ
            },
        },
        sort_keys=True,
    )


def has_qsa_capability(model_config: Any) -> bool:
    """Use selector metadata and attention owners, never model names."""
    text = getattr(model_config, "hf_text_config", None)
    fields = (
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
    )
    values = [getattr(text, field, None) for field in fields]
    return all(isinstance(v, int) and v > 0 for v in values) and any(
        kind in ("full_attention", "qwen_sparse_attention")
        for kind in (getattr(text, "layer_types", None) or ())
    )


def all_visible_devices_sm70() -> bool:
    """Conservative early check, before EngineArgs applies its MTP defaults.

    Unknown/mixed visible devices retain upstream defaults; worker topology
    discovery belongs to the interconnect resolver. Allocator configuration must
    already be present before querying platform capabilities.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return False
    count = current_platform.device_count()
    return count > 0 and all(
        current_platform.is_device_capability((7, 0), device_id=i) for i in range(count)
    )


def apply_revolta_profile(
    model_config: Any,
    tensor_parallel_size: int,
    *,
    is_sm70: bool,
    interconnect_mode: InterconnectMode = "pex_p2p",
) -> bool:
    """Apply before config defaults and before model/native module imports.

    W10.3 can supply its resolved interconnect mode here. Until integrated,
    pex_p2p preserves the existing deployment; this does not select an AR
    backend. Explicit environment overrides always win, including in no_p2p.
    """
    import torch

    if interconnect_mode not in ("pex_p2p", "no_p2p", "nvlink"):
        raise ValueError(f"Unknown ReVolta interconnect mode: {interconnect_mode}")
    enabled = (
        is_sm70
        and tensor_parallel_size == 4
        and has_qsa_capability(model_config)
        and getattr(model_config, "dtype", None) == torch.float16
    )
    if not enabled:
        return False
    defaults = dict(COMMON_DEFAULTS)
    if getattr(model_config, "quantization", None) == "awq":
        defaults.update(AWQ_DEFAULTS)
    if interconnect_mode != "no_p2p":
        defaults.update(CUSTOM_AR_DEFAULTS)
    for name, value in defaults.items():
        if name not in os.environ:
            os.environ.setdefault(name, value)
            _PROFILE_VALUES[name] = value
        elif name not in _PROFILE_VALUES and name not in _INHERITED_PROVENANCE:
            # Also recognize programmatic overrides supplied after import vllm
            # but before EngineArgs/VllmConfig construction.
            _INITIAL_ENV.setdefault(name, os.environ[name])
    _export_provenance()
    return True


def profile_audit(
    *,
    enabled: bool,
    interconnect_mode: InterconnectMode = "pex_p2p",
    quantization: str | None = None,
) -> str:
    """One JSON line, after upstream config policy has resolved its defaults."""
    from vllm import envs

    values = {}
    for name in AUDITED_NAMES:
        value: str | None
        if name in os.environ:
            value = os.environ[name]
        elif name == "VLLM_SM70_QSA_INDEXER_CUBLAS":
            # The consumer's import-time default differs from envs' raw getter.
            value = "1"
        elif name in envs.environment_variables:
            raw = envs.environment_variables[name]()
            if isinstance(raw, bool):
                value = str(int(raw))
            else:
                value = str(raw) if raw is not None else None
        else:
            value = None
        values[name] = {"value": value, "source": _source(name)}
    _export_provenance()
    return json.dumps(
        {
            "enabled": enabled,
            "interconnect_mode": interconnect_mode,
            "quantization": quantization,
            "values": values,
        },
        sort_keys=True,
    )
