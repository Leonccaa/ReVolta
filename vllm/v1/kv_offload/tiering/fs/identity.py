# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One fail-closed eligibility decision for the tier and prefix-hash seed.

Checkpoint generations must stay immutable from manifest creation through model
loading and serving. Stat verification is a fast check of that contract; full
verification also detects rewrites that preserve stat fields.
"""

import hashlib
import json
import os
from dataclasses import dataclass

from vllm.config.utils import normalize_value
from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.fs.content_manifest import (
    ManifestError,
    load_manifest,
    rehash_and_verify,
    verify_manifest,
)
from vllm.version import __version__

logger = init_logger(__name__)
PAGE_FORMAT = "revolta_fs/page-v2+crc32c-full-identity-xattr"


def canonical_digest(value) -> str:
    data = json.dumps(
        normalize_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(data.encode()).hexdigest()


def _fields(config, names: tuple[str, ...]) -> dict:
    return {name: getattr(config, name, None) for name in names}


def _local_model(config, role: str) -> str:
    path = getattr(config, "model", None)
    if not isinstance(path, str) or not path or not os.path.exists(path):
        raise ManifestError(f"{role}: model must resolve to a local checkpoint")
    for name in ("model_weights", "secondary_weights"):
        if getattr(config, name, None):
            raise ManifestError(f"{role}: {name} is not supported")
    if getattr(config, "is_multimodal_model", False):
        raise ManifestError(f"{role}: multimodal weight sources are not supported")
    if getattr(config, "runner_type", "generate") != "generate":
        raise ManifestError(f"{role}: only generation models are supported")
    hf = getattr(config, "hf_config", None)
    if getattr(hf, "secondary_weights", None):
        raise ManifestError(f"{role}: secondary weights are not supported")
    return os.path.realpath(path)


def _loader_identity(config) -> dict:
    from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if config is None:
        raise ManifestError("effective load configuration is missing")
    fmt = config.load_format
    loader = _LOAD_FORMAT_TO_MODEL_LOADER.get(fmt)
    if fmt not in ("auto", "safetensors") or loader is not DefaultModelLoader:
        raise ManifestError("only the default auto/safetensors loader is supported")
    if getattr(config, "model_loader_extra_config", None) or getattr(
        config, "secondary_weights", None
    ):
        raise ManifestError("extra loader configuration/weights are not supported")
    return {
        "format": fmt,
        "loader": f"{loader.__module__}.{loader.__qualname__}",
        **_fields(
            config,
            ("ignore_patterns", "safetensors_load_strategy", "pt_load_map_location"),
        ),
    }


def _model_identity(config) -> dict:
    result = _fields(
        config,
        (
            "model",
            "revision",
            "code_revision",
            "dtype",
            "quantization",
            "hf_overrides",
            "hf_config",
            "hf_text_config",
            "max_model_len",
        ),
    )
    overrides = result["hf_overrides"]
    if callable(overrides):
        # SpeculativeConfig installs this built-in transformation itself. Its
        # resolved output is recorded below along with the producer version.
        from vllm.config.speculative import SpeculativeConfig

        if overrides is not SpeculativeConfig.hf_config_override:
            raise ManifestError("custom callable hf_overrides cannot be identified")
        result["hf_overrides"] = "SpeculativeConfig.hf_config_override"
    return result


@dataclass(frozen=True)
class RevoltaIdentity:
    content_digest: str
    persistent_digest: str


def disable_revolta_root(vllm_config, root: str, reason: str) -> None:
    """Remember a construction failure before EngineCore initializes NONE_HASH.

    Scheduler tier construction precedes init_none_hash. In particular a root
    rejected for ownership, xattr or namespace conflicts must stay excluded at
    that later call, even if its checkpoint identity would otherwise be valid.
    This runtime attribute is not a configuration/hash input.
    """
    disabled = getattr(vllm_config, "_revolta_fs_disabled_roots", None)
    if disabled is None:
        disabled = {}
        vllm_config._revolta_fs_disabled_roots = disabled
    disabled[os.path.realpath(root)] = reason


def get_revolta_identity(
    vllm_config, root: str, manifest_path: str | None, content_verify: str = "stat"
) -> RevoltaIdentity | None:
    """Validate supported sources and return full persistent configuration ID.

    Does not create or write the cache root. Both seed initialization and tier
    construction call this function, including when the tier is disabled.
    """
    try:
        if vllm_config is None:
            raise ManifestError("engine configuration is required")
        disabled = getattr(vllm_config, "_revolta_fs_disabled_roots", {})
        if os.path.realpath(root) in disabled:
            raise ManifestError(disabled[os.path.realpath(root)])
        if not manifest_path:
            raise ManifestError("content_manifest_path is required")
        if content_verify not in ("stat", "full"):
            raise ManifestError("content_verify must be stat or full")
        if getattr(vllm_config, "lora_config", None) is not None:
            raise ManifestError("LoRA content identity is not supported")
        if getattr(vllm_config.cache_config, "calculate_kv_scales", False):
            raise ManifestError("runtime KV scale calibration is not supported")
        target = vllm_config.model_config
        roots = {"target": _local_model(target, "target")}
        loader = _loader_identity(vllm_config.load_config)
        speculative = vllm_config.speculative_config
        draft_identity = None
        if speculative is not None:
            draft = getattr(speculative, "draft_model_config", None)
            if draft is None:
                raise ManifestError("effective draft model configuration is missing")
            draft_root = _local_model(draft, "draft")
            if roots["target"] != draft_root:
                roots["draft"] = draft_root
            draft_identity = {
                **_fields(
                    speculative,
                    (
                        "method",
                        "model",
                        "num_speculative_tokens",
                        "quantization",
                        "hf_overrides",
                        "kv_cache_dtype",
                        "mtp_expert_quantization",
                        "attention_backend",
                        "moe_backend",
                        "draft_tensor_parallel_size",
                    ),
                ),
                "effective_model": _model_identity(draft),
                "loader": _loader_identity(
                    getattr(speculative, "draft_load_config", None)
                    or vllm_config.load_config
                ),
            }
        manifest = load_manifest(manifest_path)
        content = verify_manifest(manifest, roots)
        if content_verify == "full":
            rehash_and_verify(manifest)
        # Check the optional checksum dependency before creating the root.
        import google_crc32c  # noqa: F401

        identity = {
            "format": PAGE_FORMAT,
            "producer_version": __version__,
            "content_manifest_digest": content,
            "content_algorithm": manifest["algorithm"],
            "target": _model_identity(target),
            "loader": loader,
            "draft": draft_identity,
            "cache": _fields(
                vllm_config.cache_config,
                (
                    "cache_dtype",
                    "block_size",
                    "prefix_caching_hash_algo",
                    "mamba_cache_dtype",
                    "mamba_ssm_cache_dtype",
                    "mamba_block_size",
                ),
            ),
            "parallel": _fields(
                vllm_config.parallel_config,
                (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "rank",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                ),
            ),
            "attention": getattr(vllm_config, "attention_config", None),
            "kernel": getattr(vllm_config, "kernel_config", None),
            "additional": getattr(vllm_config, "additional_config", None),
            "compile_hash": vllm_config.compute_hash(),
        }
        return RevoltaIdentity(content, canonical_digest(identity))
    except (ManifestError, OSError, ValueError, TypeError, ImportError) as exc:
        logger.error("revolta_fs disabled: %s", exc)
        return None
