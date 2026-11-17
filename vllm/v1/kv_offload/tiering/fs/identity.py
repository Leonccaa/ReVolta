# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One fail-closed eligibility decision for the tier and prefix-hash seed.

Checkpoint generations must stay immutable from manifest creation through model
loading and serving. Stat verification is a fast check of that contract; full
verification also detects rewrites that preserve stat fields.
"""

import dataclasses
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

logger = init_logger(__name__)
PAGE_FORMAT = "revolta_fs/page-v2+crc32c-full-identity-xattr"
# Stored pages stay valid across engine versions. Add 1 only when a change
# makes already stored KV/GDN state wrong (for example a fix to how it is
# computed) or changes the page format. Differences that only reorder
# floating-point accumulation do not count.
KV_COMPAT_EPOCH = 1

# The persistent identity is a whitelist: only fields that certainly change the
# bytes or the meaning of a stored page belong in it (see get_revolta_identity
# and the persistent_layout of the tiering spec). Tuning options and whole
# configuration objects stay out, so changing them never discards the archive.
# A new feature that changes what is stored must add its field here, or add 1
# to KV_COMPAT_EPOCH.
#
# Weights, config files and checkpoint structure are covered by the content
# manifest; the page keys already carry token and multimodal hashes. What is
# left is what changes values without changing either.
_MODEL_FIELDS = ("dtype", "quantization", "hf_overrides")
# Server-level processor options are merged inside the processor and are not
# part of the multimodal input hash (which only sees per-request kwargs).
_MULTIMODAL_FIELDS = ("mm_processor_kwargs", "mm_encoder_fp8_scale_path")
_DRAFT_FIELDS = (
    "method",
    "quantization",
    "hf_overrides",
    "kv_cache_dtype",
    "mtp_expert_quantization",
    "draft_tensor_parallel_size",
)


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
    runner = "generate" if role == "target" else "draft"
    if getattr(config, "runner_type", runner) != runner:
        raise ManifestError(f"{role}: expected the {runner} runner")
    hf = getattr(config, "hf_config", None)
    if getattr(hf, "secondary_weights", None):
        raise ManifestError(f"{role}: secondary weights are not supported")
    return os.path.realpath(path)


def _check_loader(config) -> None:
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


def _check_model(config) -> None:
    overrides = getattr(config, "hf_overrides", None)
    if callable(overrides):
        # SpeculativeConfig installs this built-in transformation itself.
        from vllm.config.speculative import SpeculativeConfig

        if overrides is not SpeculativeConfig.hf_config_override:
            raise ManifestError("custom callable hf_overrides cannot be identified")
    mm = getattr(config, "multimodal_config", None)
    if (
        getattr(mm, "mm_encoder_attn_dtype", None) == "fp8"
        and getattr(mm, "mm_encoder_fp8_scale_path", None) is None
    ):
        raise ManifestError(
            "runtime multimodal encoder scale calibration is not supported"
        )


def _model_identity(config) -> dict:
    _check_model(config)
    result = _fields(config, _MODEL_FIELDS)
    if callable(result["hf_overrides"]):
        result["hf_overrides"] = "SpeculativeConfig.hf_config_override"
    mm = getattr(config, "multimodal_config", None)
    result["multimodal_config"] = (
        None if mm is None else _plain(_fields(mm, _MULTIMODAL_FIELDS))
    )
    return result


def _plain(value, depth: int = 0):
    """A JSON-safe, deterministic rendering of a config object for the
    identity digest. Unknown leaf objects are rendered by type and str(), so
    an unexpected field still changes the digest when its value changes."""
    if depth > 8:
        return f"<{type(value).__qualname__}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name, None), depth + 1)
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            str(k): _plain(v, depth + 1)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(v, depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(_plain(v, depth + 1)) for v in value)
    return f"{type(value).__module__}.{type(value).__qualname__}:{value}"


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
    for tier in getattr(vllm_config, "_revolta_fs_managers", ()):
        if tier._root_dir == os.path.realpath(root) and not tier.disabled:
            tier.shutdown()
            tier.disabled_reason = reason


def _verify_config_sources(config, manifest: dict) -> None:
    """Reject file references without manifest coverage, including nested HF
    overrides, online quantization or multimodal encoder scale inputs.

    Paths are resolved as the loader resolves them, relative to the process
    working directory. A directory reference must itself be a manifest root.
    """
    files = set()
    roots = set()
    for entry in manifest["roles"].values():
        root = entry["root"]
        roots.add(os.path.realpath(root))
        for rel in entry["files"]:
            files.add(
                os.path.realpath(
                    root if os.path.isfile(root) else os.path.join(root, rel)
                )
            )

    def visit(value, name=""):
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, str(key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, name)
        elif isinstance(value, (str, os.PathLike)) and value:
            path = os.fspath(value)
            # get_quant_config passes this serialized configuration to
            # TorchAOConfig. Inspect its nested sources as well.
            if name == "quantization_config_dict_json":
                visit(json.loads(path))
                return
            path_field = name.endswith(
                (
                    "_file",
                    "_files",
                    "_path",
                    "_paths",
                    "_dir",
                    "_dirs",
                    "_name_or_path",
                    "_source",
                    "_sources",
                )
            ) or name in {
                "file",
                "files",
                "path",
                "paths",
                "dir",
                "source",
                "sources",
                "vision_tower",
                "mm_vision_tower",
                "audio_tower",
                "model_id",
                "processor_name",
                "tokenizer_name",
            }
            # Loader/processor source fields can be relative or remote even
            # before they exist. A slash alone also occurs in AWQ metadata
            # (formulae and provenance notes), which remains in the digest.
            if (
                path_field or os.path.isabs(path) or os.path.exists(path)
            ) and os.path.realpath(path) not in files | roots:
                raise ManifestError(f"{name}: source is outside the content manifest")

    for name in ("tokenizer", "generation_config"):
        source = getattr(config, name, None)
        if source and source not in ("auto", "vllm"):
            visit(source, name + "_path")
    visit(
        _fields(
            config,
            (
                "hf_config_path",
                "quantization_config",
                "hf_overrides",
                "hf_config",
                "hf_text_config",
                "multimodal_config",
            ),
        )
    )
    # DefaultModelLoader can follow shard names outside the checkpoint through
    # a safetensors index. The index's own hash does not bind those shards.
    for filename in (
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ):
        index = os.path.join(config.model, filename)
        if os.path.isfile(index):
            with open(index) as source:
                weights = json.load(source)["weight_map"]
            if not isinstance(weights, dict):
                raise ManifestError("invalid checkpoint weight map")
            for shard in weights.values():
                if os.path.realpath(os.path.join(config.model, shard)) not in files:
                    raise ManifestError("indexed shard is outside the content manifest")


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
        _check_loader(vllm_config.load_config)
        speculative = vllm_config.speculative_config
        draft_identity = None
        if speculative is not None:
            draft = getattr(speculative, "draft_model_config", None)
            if draft is None:
                raise ManifestError("effective draft model configuration is missing")
            draft_root = (
                roots["target"] if draft is target else _local_model(draft, "draft")
            )
            if roots["target"] != draft_root:
                roots["draft"] = draft_root
            draft_identity = {
                **_fields(speculative, _DRAFT_FIELDS),
            }
            _check_model(draft)
            _check_loader(
                getattr(speculative, "draft_load_config", None)
                or vllm_config.load_config
            )
        manifest = load_manifest(manifest_path)
        # Auxiliary directory inputs also require an exact file set. A newly
        # added file can change loader selection even if older files match.
        required_roots = {
            role: entry["root"] for role, entry in manifest["roles"].items()
        }
        required_roots.update(roots)
        content = verify_manifest(manifest, required_roots)
        _verify_config_sources(target, manifest)
        if speculative is not None:
            _verify_config_sources(draft, manifest)
        if content_verify == "full":
            rehash_and_verify(manifest)
        # Check the optional checksum dependency before creating the root.
        import google_crc32c  # noqa: F401

        identity = {
            "format": PAGE_FORMAT,
            "kv_compat_epoch": KV_COMPAT_EPOCH,
            "content_manifest_digest": content,
            "content_algorithm": manifest["algorithm"],
            "target": _model_identity(target),
            "draft": draft_identity,
            "cache": _fields(
                vllm_config.cache_config,
                (
                    "cache_dtype",
                    "block_size",
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
                    "cp_kv_cache_interleave_size",
                ),
            ),
            # The backend fixes the order of K/V inside a page.
            "attention_backend": getattr(
                getattr(vllm_config, "attention_config", None), "backend", None
            ),
        }
        kernel = getattr(vllm_config, "kernel_config", None)
        if (
            getattr(kernel, "qsa_dcp_index_shard", False)
            and vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            # The indexer is split over the TP group; pages written when it
            # was split over DCP ranks only have another layout.
            identity["index_shard_world"] = (
                vllm_config.parallel_config.tensor_parallel_size
            )
        return RevoltaIdentity(content, canonical_digest(identity))
    except (
        ManifestError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
    ) as exc:
        logger.error("revolta_fs disabled: %s", exc)
        return None
