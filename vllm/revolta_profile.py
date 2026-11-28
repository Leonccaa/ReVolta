# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide defaults for the FP16 QSA SM70 TP4 serving profile.

Imported before env_override imports torch, solely to remember environment
provenance. Allocator policy stays in the launcher: model/TP information is not
available here before CUDA initialization. Operator admission remains upstream.
"""

import json
import os
from typing import TYPE_CHECKING, Any, Literal

from vllm import envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig
    from vllm.platforms.sm70_interconnect import _InterconnectDecision

logger = init_logger(__name__)

InterconnectMode = Literal["pex_p2p", "no_p2p", "nvlink"]

COMMON_DEFAULTS = {
    "VLLM_SM70_QWEN38_WIDE_M": "1",
    "VLLM_SM70_PLE_PREFILL_CONV_CHUNK": "1",
    "VLLM_SM70_QWEN38_GDN_NO_PACK": "1",
    "VLLM_SM70_GREEDY_VERIFY": "1",
    "VLLM_SM70_SKIP_REJECTED_BATCH_PACKS": "1",
    "VLLM_SM70_QSA_SHARED_KEY": "1",
    "VLLM_SM70_QSA_REQUEST_SEGMENTS": "1",
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
    # Measured for the AWQ TP4 graph pool.
    "VLLM_V2_CUDAGRAPH_MEM_MIB": "700",
}
AWQ_MTP_WIDE_VERIFY_DEFAULTS = {
    "VLLM_SM70_MTP_GDN_NATIVE_VERIFY": "1",
    "VLLM_SM70_QWEN38_NATIVE_ROUTER_TOPK": "1",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS": "48",
}
NVFP4_DEFAULTS = {
    "VLLM_SM70_MTP_Q8_PREFILL_MMQ": "1",
    "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE": "1",
    "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE": "1",
    "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE": "1",
    "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "0",
    "VLLM_SM70_NVFP4_TURBOMIND": "1",
    "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL": "1",
    "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES": "1",
    "VLLM_V2_CUDAGRAPH_MEM_MIB": "700",
}
# DFlash2 also captures the eight-row verify graphs with their width shapes and
# the drafter's graphs. The single-request verifier tail graphs add about
# 0.3 GiB without a measured speed gain on the AWQ TP4 profile (greedy tokens
# identical, 256K decode unchanged), so DFlash2 leaves them off; the remaining
# pool is about 1.1 GiB. With the MTP figure the excess goes to the KV cache
# and a large mixed step can OOM. The graphs depend on the speculative method,
# not on the weight format, so AWQ and NVFP4 both use these values.
DFLASH_DEFAULTS = {
    "VLLM_V2_CUDAGRAPH_MEM_MIB": "1200",
    "VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS": "0",
}
HYBRID_SERVING_DEFAULTS = {
    "VLLM_SM70_QSA_GROUPED_LPT": "1",
    "VLLM_SM70_GDN_SKINNY_MPLAN": "1",
}
DFLASH_SERVING_DEFAULTS = {"VLLM_SM70_DFLASH2_FUSED_QKV_PACK": "1"}
AWQ_DFLASH_DEFAULTS = {"VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP": "1"}
# With a planar KV cache (fp8_per_token_head, the QSA auto choice, or int8) the
# graphs are smaller than with the scalar FP8 cache. Measured per GPU on TP4:
# DFlash2 0.92 GiB (AWQ; 1.13 GiB with fp8_e4m3) and 0.95 GiB (NVFP4); MTP4
# 0.53 GiB (AWQ) and 0.51 GiB (NVFP4). The reserves keep about 50 MiB of
# headroom; GPU memory is the limit, so they stay tight. Other KV dtypes keep
# the values above.
AWQ_DFLASH_PLANAR_KV_DEFAULTS = {"VLLM_V2_CUDAGRAPH_MEM_MIB": "1000"}
# The same AWQ profile with the mixed-prefill latency target enabled measured
# 1094/1094/1074/1094 MiB per GPU, above that reserve (800 MiB when the target
# is 0). The serving profile resolves the target to 0, so this tier applies
# only when a positive target is set explicitly.
AWQ_DFLASH_PLANAR_KV_MIXED_PREFILL_DEFAULTS = {"VLLM_V2_CUDAGRAPH_MEM_MIB": "1150"}
NVFP4_DFLASH_PLANAR_KV_DEFAULTS = {"VLLM_V2_CUDAGRAPH_MEM_MIB": "1024"}
MTP_PLANAR_KV_DEFAULTS = {"VLLM_V2_CUDAGRAPH_MEM_MIB": "600"}
# Dynamic MTP captured 0.64 GiB per rank on AWQ TP4 with planar KV.
MTP_DSD_PLANAR_KV_DEFAULTS = {"VLLM_V2_CUDAGRAPH_MEM_MIB": "705"}
# NVFP4 + MTP5 planar DSD captured 0.73 GiB per rank on TP4, above that
# reserve; the width-5 verify graphs need 800 MiB. Widths below 5 measured
# within 705, so only a schedule whose widest captured graph reaches 5 moves.
MTP_DSD_PLANAR_KV_WIDE_DEFAULTS = {"VLLM_V2_CUDAGRAPH_MEM_MIB": "800"}
MTP_DSD_WIDE_MIN_K = 5
PLANAR_KV_DTYPES = ("fp8_per_token_head", "int8")
CUSTOM_AR_DEFAULTS = {"VLLM_SM70_CAR_VMM_SAFE": "1"}

# Config defaults have their own audit namespace; never add them to the W9
# environment inventory or use environment provenance to resolve them.
KERNEL_DEFAULTS = {"ple_disk_cascade": False}

# Include the entire deployment inventory in the audit, even entries whose
# defaults remain upstream. In particular, never write the three FP16 flags.
UPSTREAM_AND_DEPLOYMENT = (
    "VLLM_SM70_NVLINK_HC",
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
        | set(HYBRID_SERVING_DEFAULTS)
        | set(DFLASH_SERVING_DEFAULTS)
        | set(UPSTREAM_AND_DEPLOYMENT)
    )
)
# Include NVFP4-only entries in NVFP4 audits and provenance transport.
# VLLM_SM70_FP8_TURBOMIND is no longer written (its unset default is already 1;
# kernel_config.sm70_fp8.enabled replaces it for FP8 linear layers) but stays
# audited for NVFP4 mixed checkpoints, whose MoE consumers still read it.
_ALL_AUDITED_NAMES = tuple(
    sorted(set(AUDITED_NAMES) | set(NVFP4_DEFAULTS) | {"VLLM_SM70_FP8_TURBOMIND"})
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
            if name in _ALL_AUDITED_NAMES
            and row["source"] in ("profile", "environment", "upstream_default")
            and isinstance(row["value"], str)
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        # A stale or malformed diagnostic must not affect serving defaults.
        return {}


_INHERITED_PROVENANCE = _inherited_provenance()
_INITIAL_ENV = {
    name: os.environ[name] for name in _ALL_AUDITED_NAMES if name in os.environ
}
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


def value_source(name: str) -> str:
    """Audit provenance of an audited variable: profile, environment or
    upstream_default. Diagnostics only; never resolve a setting from it."""
    return _source(name)


def _export_provenance() -> None:
    # Refresh at the final audit too, after upstream materializes its defaults.
    os.environ[_PROVENANCE_ENV] = json.dumps(
        {
            "version": 1,
            "values": {
                name: {"value": os.environ.get(name), "source": _source(name)}
                for name in _ALL_AUDITED_NAMES
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
        kind in ("full_attention", "qwen_sparse_attention", "indexed_attention")
        for kind in (getattr(text, "layer_types", None) or ())
    )


def _has_awq_qpn_shape(model_config: Any, tensor_parallel_size: int) -> bool:
    """The measured native-g32 QPN banks are E512, K=2560, TP shard N=160.

    Missing metadata keeps AWQ's generic admission. Do not turn its implicit
    fallback into an explicit fail-closed QPN request for an unknown shape.
    """
    text = getattr(model_config, "hf_text_config", None)
    if tensor_parallel_size != 4 or any(
        getattr(text, name, None) != value
        for name, value in (
            ("hidden_size", 2560),
            ("moe_intermediate_size", 640),
            ("num_experts", 512),
            ("num_experts_per_tok", 10),
        )
    ):
        return False
    for owner in ("hf_config", "hf_text_config", "model_arch_config"):
        quant = getattr(getattr(model_config, owner, None), "quantization_config", None)
        if quant is not None:
            return (
                isinstance(quant, dict)
                and quant.get("q_group_size", quant.get("group_size")) == 32
                and quant.get("w_bit", quant.get("bits")) == 4
            )
    return False


def all_visible_devices_sm70() -> bool:
    """Conservative early check, before EngineArgs applies its MTP defaults.

    Unknown/mixed visible devices retain upstream defaults; worker topology
    discovery belongs to the interconnect resolver. The supported NVIDIA
    platform uses NVML for capability queries before CUDA initialization.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return False
    count = current_platform.device_count()
    return count > 0 and all(
        current_platform.is_device_capability((7, 0), device_id=i) for i in range(count)
    )


def frontend_interconnect_mode(
    *, disable_custom_all_reduce: bool = False, device_ids: tuple[int, ...] = ()
) -> InterconnectMode:
    """Select frontend defaults from topology before worker P2P admission."""
    if disable_custom_all_reduce:
        return "no_p2p"
    requested = os.environ.get("VLLM_SM70_INTERCONNECT", "auto").strip().lower()
    if requested == "nccl":
        return "no_p2p"
    if requested == "nvlink":
        return "nvlink"
    if requested == "auto" and device_ids:
        try:
            if not _cuda_ids_name_nvml_devices(device_ids):
                return "pex_p2p"
            decision = _select_sm70_interconnect(device_ids, disable_custom_all_reduce)
        except Exception as error:
            # Workers repeat the query and resolve the path; keep the old default.
            logger.warning_once(
                "SM70 frontend topology query failed (%s); keeping PCIe defaults.",
                type(error).__name__,
            )
            return "pex_p2p"
        if decision.mode == "nvlink":
            return "nvlink"
        logger.warning_once(
            "Selected SM70 TP4 CUDA devices %s do not form a 6/6 direct NVLink "
            "mesh. Check CUDA_VISIBLE_DEVICES and nvidia-smi topo -m if NVLink "
            "was intended.",
            device_ids,
        )
    return "pex_p2p"


def local_tp4_device_ids(parallel: "ParallelConfig") -> tuple[int, ...]:
    """Only local, single-replica TP4 has known placement in the frontend."""
    from vllm.platforms import current_platform

    if (
        parallel.tensor_parallel_size == 4
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.nnodes == 1
        and parallel.distributed_executor_backend in (None, "mp")
        and current_platform.is_cuda()
        and current_platform.device_count() >= 4
        and all(
            current_platform.is_device_capability((7, 0), device_id=i) for i in range(4)
        )
    ):
        return (0, 1, 2, 3)
    return ()


def _cuda_ids_name_nvml_devices(device_ids: tuple[int, ...]) -> bool:
    """Whether numeric CUDA ordinals map to the same NVML devices.

    CUDA enumerates FASTEST_FIRST by default, so a numeric ordinal names an
    NVML index only in PCI bus order. UUID masks name devices exactly, and a
    selection of every GPU is the same set in any order.
    """
    from vllm.platforms import current_platform

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" or (
        visible and all(v.strip().startswith("GPU-") for v in visible.split(","))
    ):
        return True
    physical_ids = sorted(
        current_platform.device_id_to_physical_device_id(i) for i in device_ids
    )
    return physical_ids == sorted(current_platform.get_all_gpu_pci_bus_ids())


def _select_sm70_interconnect(
    device_ids: tuple[int, ...], disable_custom_all_reduce: bool
) -> "_InterconnectDecision":
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability
    from vllm.platforms.sm70_interconnect import _select_interconnect

    physical_ids = [
        current_platform.device_id_to_physical_device_id(i) for i in device_ids
    ]
    return _select_interconnect(
        len(physical_ids),
        DeviceCapability(7, 0),
        physical_ids,
        disable_custom_all_reduce,
    )


def is_nvlink_auto_profile(model_config: Any, mode: InterconnectMode) -> bool:
    return (
        mode == "nvlink"
        and envs.VLLM_SM70_INTERCONNECT == "auto"
        and (
            getattr(model_config, "quantization", None) == "awq"
            or has_nvfp4_weights(model_config)
        )
    )


def _logged_interconnect_mode(mode: InterconnectMode, policy: str) -> str:
    """Distinguish the frontend NVLink candidate from worker admission."""
    if mode == "nvlink" and policy.strip().lower() == "auto":
        return "nvlink (pending worker admission)"
    if mode != "pex_p2p":
        return mode
    requested = policy.strip().lower()
    if requested == "pex":
        return mode
    if requested == "auto":
        return "auto (resolved by workers)"
    return "pending (resolved by workers)"


def has_nvfp4_weights(model_config: Any) -> bool:
    """Resolve ModelOpt formats from metadata, independent of checkpoint names."""
    method = getattr(model_config, "quantization", None)
    if method == "modelopt_fp4":
        return True
    if method != "modelopt_mixed":
        return False
    for owner in ("hf_config", "hf_text_config", "model_arch_config"):
        config = getattr(
            getattr(model_config, owner, None), "quantization_config", None
        )
        if not isinstance(config, dict):
            continue
        quant = config.get("quantization", config)
        if not isinstance(quant, dict):
            continue
        if str(quant.get("quant_algo", "")).upper() in ("NVFP4", "W4A16_NVFP4"):
            return True
        layers = quant.get("quantized_layers", {})
        if isinstance(layers, dict) and any(
            isinstance(info, dict)
            and str(info.get("quant_algo", "")).upper() in ("NVFP4", "W4A16_NVFP4")
            for info in layers.values()
        ):
            return True
    return False


def apply_revolta_profile(
    model_config: Any,
    tensor_parallel_size: int,
    *,
    is_sm70: bool,
    interconnect_mode: InterconnectMode = "pex_p2p",
    speculative_method: str | None = None,
    hybrid_serving: bool = False,
    allow_shared_expert_overlap: bool = True,
    kv_cache_dtype: str | None = None,
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None,
    num_speculative_tokens: int | None = None,
    mixed_prefill_step_latency_ms: float = 0,
) -> bool:
    """Apply before config defaults and before model/native module imports.

    Pass frontend_interconnect_mode(); this does not select an AR backend.
    Explicit environment overrides always win, including in no_p2p. A later
    call may refine a value this profile wrote (EngineArgs may not know the
    speculative method or the KV dtype that VllmConfig resolves).
    Pass the resolved scheduler mixed-prefill step latency target only when it
    is actually resolved; the default 0 selects the target-disabled graph tier
    until the refining call.
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
    mtp_planar_kv_defaults = MTP_PLANAR_KV_DEFAULTS
    if num_speculative_tokens_per_batch_size is not None:
        # The verify graphs are captured up to the global draft count; schedule
        # entries only add narrower widths, so the global count sets the tier.
        mtp_planar_kv_defaults = (
            MTP_DSD_PLANAR_KV_WIDE_DEFAULTS
            if (num_speculative_tokens or 0) >= MTP_DSD_WIDE_MIN_K
            else MTP_DSD_PLANAR_KV_DEFAULTS
        )
    if getattr(model_config, "quantization", None) == "awq":
        if _has_awq_qpn_shape(model_config, tensor_parallel_size):
            defaults.update(AWQ_DEFAULTS)
            if hybrid_serving and envs.VLLM_SM70_AWQ_MOE_COMPACT_METADATA:
                defaults["VLLM_SM70_AWQ_MOE_INT4_ZERO"] = "1"
            if speculative_method == "dflash":
                defaults.update(DFLASH_DEFAULTS)
                if allow_shared_expert_overlap:
                    defaults.update(AWQ_DFLASH_DEFAULTS)
                if kv_cache_dtype in PLANAR_KV_DTYPES:
                    defaults.update(AWQ_DFLASH_PLANAR_KV_DEFAULTS)
                    if mixed_prefill_step_latency_ms > 0:
                        defaults.update(AWQ_DFLASH_PLANAR_KV_MIXED_PREFILL_DEFAULTS)
            elif speculative_method == "mtp" and kv_cache_dtype in PLANAR_KV_DTYPES:
                defaults.update(mtp_planar_kv_defaults)
    elif has_nvfp4_weights(model_config):
        defaults.update(NVFP4_DEFAULTS)
        if speculative_method == "dflash":
            defaults.update(DFLASH_DEFAULTS)
            if kv_cache_dtype in PLANAR_KV_DTYPES:
                defaults.update(NVFP4_DFLASH_PLANAR_KV_DEFAULTS)
        elif speculative_method == "mtp" and kv_cache_dtype in PLANAR_KV_DTYPES:
            defaults.update(mtp_planar_kv_defaults)
    if interconnect_mode != "no_p2p":
        defaults.update(CUSTOM_AR_DEFAULTS)
    if speculative_method in ("mtp", "dflash") and is_nvlink_auto_profile(
        model_config, interconnect_mode
    ):
        defaults["VLLM_SM70_NVLINK_HC"] = "1"
    if hybrid_serving:
        defaults.update(HYBRID_SERVING_DEFAULTS)
        if speculative_method == "dflash":
            defaults.update(DFLASH_SERVING_DEFAULTS)
        for alias, name in envs.PROMOTED_RESEARCH_ENVS.items():
            if alias in os.environ:
                defaults.pop(name, None)

    _apply_profile_defaults(defaults)
    return True


def apply_awq_mtp_wide_verify_defaults(
    config: "VllmConfig", device_ids: tuple[int, ...]
) -> bool:
    """Apply the wide verifier settings within the qualified ReVolta profile."""
    model = config.model_config
    spec = config.speculative_config
    parallel = config.parallel_config
    if (
        spec is None
        or not spec.use_qwen4_exp_mtp()
        or spec.num_speculative_tokens != 4
        or not spec.gdn_spec_state_replay
        or model.quantization != "awq"
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 2
        or config.scheduler_config.async_scheduling
        or config.kernel_config.long_context_kv.enabled
        or not _has_awq_qpn_shape(model, parallel.tensor_parallel_size)
    ):
        return False

    decision = _select_sm70_interconnect(device_ids, parallel.disable_custom_all_reduce)
    if decision.mode != "pex_p2p":
        return False

    _apply_profile_defaults(AWQ_MTP_WIDE_VERIFY_DEFAULTS)
    return True


def _apply_profile_defaults(defaults: dict[str, str]) -> None:
    for name, value in defaults.items():
        if name not in os.environ or _PROFILE_VALUES.get(name) == os.environ[name]:
            os.environ[name] = value
            _PROFILE_VALUES[name] = value
        elif name not in _PROFILE_VALUES and name not in _INHERITED_PROVENANCE:
            # Also recognize programmatic overrides supplied after import vllm
            # but before EngineArgs/VllmConfig construction.
            _INITIAL_ENV.setdefault(name, os.environ[name])
    _export_provenance()


def apply_revolta_allocator_default() -> None:
    """Set the allocator after connector selection and before worker startup."""
    from vllm.env_override import _default_nccl_graph_register

    names = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    if any(name in os.environ for name in names):
        return
    for name in names:
        os.environ[name] = _PROFILE_VALUES[name] = "expandable_segments:True"
    _default_nccl_graph_register(os.environ)
    _export_provenance()


def apply_revolta_kernel_profile(kernel_config: Any, *, enabled: bool) -> None:
    """Apply only to unset fields; the config carries provenance to workers."""
    value = kernel_config.ple_disk_cascade
    source = kernel_config.ple_disk_cascade_source
    default = KERNEL_DEFAULTS["ple_disk_cascade"]
    if (source == "profile" and value != default) or (
        source == "upstream_default" and value is not True
    ):
        # Also recognize a programmatic override of a resolved config.
        kernel_config.ple_disk_cascade_source = "explicit"
    elif value is None or (enabled and source == "upstream_default"):
        kernel_config.ple_disk_cascade = default if enabled else True
        kernel_config.ple_disk_cascade_source = (
            "profile" if enabled else "upstream_default"
        )


def apply_revolta_nvlink_kernel_profile(
    config: "VllmConfig",
) -> dict[str, dict[str, Any]]:
    spec = config.speculative_config
    assert spec is not None
    kernel = config.kernel_config
    kv_dtype = config.cache_config.cache_dtype
    dcp2 = config.parallel_config.decode_context_parallel_size == 2
    host = kernel.long_context_kv
    fp16_target = kv_dtype in ("auto", "float16")
    defaults: dict[str, Any] = {
        "sm70_hc_ll_async": True,
        "sm70_qsa_paged_fp16_h12": (
            dcp2
            and fp16_target
            and (not host.enabled or host.main_kv_placement == "gpu")
        ),
    }
    if spec.method == "mtp":
        defaults["sm70_sparse_target_sampling"] = "device_top64"

    values = {}
    for name, default in defaults.items():
        value = getattr(kernel, name)
        explicit = name in kernel._nvlink_profile_fields
        if not explicit:
            setattr(kernel, name, default)
        values[name] = {
            "value": value if explicit else default,
            "source": (
                "explicit" if explicit else "profile" if default else "upstream_default"
            ),
        }
    return values


def profile_audit(
    *,
    enabled: bool,
    interconnect_mode: InterconnectMode = "pex_p2p",
    quantization: str | None = None,
    kernel_config: Any = None,
    config_values: dict[str, dict[str, Any]] | None = None,
) -> str:
    """One JSON line, after upstream config policy has resolved its defaults."""
    from vllm import envs

    values = {}
    names = (
        _ALL_AUDITED_NAMES
        if quantization in ("modelopt_fp4", "modelopt_mixed")
        else AUDITED_NAMES
    )
    for name in names:
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
    policy = os.environ.get("VLLM_SM70_INTERCONNECT", "auto")
    return json.dumps(
        {
            "enabled": enabled,
            "interconnect_mode": _logged_interconnect_mode(interconnect_mode, policy),
            "interconnect_policy": policy,
            "quantization": quantization,
            "values": values,
            "kernel_config": (
                {
                    "ple_disk_cascade": {
                        "value": kernel_config.ple_disk_cascade,
                        "source": kernel_config.ple_disk_cascade_source,
                    }
                }
                if kernel_config is not None
                else {}
            ),
            **({"config_values": config_values} if config_values else {}),
        },
        sort_keys=True,
    )
