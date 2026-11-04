# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FS budget validation and process-independent prefix-hash identity."""

import fcntl
import json
import os
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig


def validate_fs_budget(max_size_bytes: Any, min_free_bytes: Any = None) -> None:
    if type(max_size_bytes) is not int or max_size_bytes <= 0:
        raise ValueError(
            "FS tier requires an explicit positive integer max_size_bytes "
            "(total page budget shared by all groups and layouts in root_dir)"
        )
    if min_free_bytes is not None and (
        type(min_free_bytes) is not int or min_free_bytes < 0
    ):
        raise ValueError("FS tier min_free_bytes must be a non-negative integer")


# Secondary tier types backed by FileSystemTierManager (same root_dir layout,
# hash seed and budget rules).
FS_TIER_TYPES = ("fs", "revolta_fs")


def fs_tier_configs(config: "KVTransferConfig | None") -> list[dict]:
    if config is None or config.kv_connector != "OffloadingConnector":
        return []
    tiers = config.kv_connector_extra_config.get("secondary_tiers", [])
    if not isinstance(tiers, list) or any(not isinstance(t, dict) for t in tiers):
        raise ValueError("secondary_tiers must be a list of tier configurations")
    result = [tier for tier in tiers if tier.get("type") in FS_TIER_TYPES]
    for tier in result:
        validate_fs_budget(tier.get("max_size_bytes"), tier.get("min_free_bytes"))
        if not isinstance(tier.get("root_dir"), str) or not tier["root_dir"]:
            raise ValueError("FS tier requires a non-empty root_dir")
    return result


def get_fs_hash_seed(
    config: "KVTransferConfig | None",
    *,
    vllm_config: "VllmConfig | None" = None,
    hash_fn: Callable[[Any], bytes] | None = None,
) -> str | None:
    """Lock configured roots and persist one seed before hashing requests.

    This short identity lock is separate from FileSystemCache's lifetime
    writer lock. Group pools and engines can read the same identity without
    weakening single-engine ownership. Existing roots must agree because a
    process has one NONE_HASH; changing an identity would lose cache hits.
    """
    tiers = fs_tier_configs(config)
    enabled = []
    for tier in tiers:
        if tier["type"] == "revolta_fs":
            from vllm.v1.kv_offload.tiering.fs.identity import get_revolta_identity

            if (
                get_revolta_identity(
                    vllm_config,
                    tier["root_dir"],
                    tier.get("content_manifest_path"),
                    tier.get("content_verify", "stat"),
                )
                is None
            ):
                continue
        enabled.append(tier)
    roots = sorted({os.path.realpath(t["root_dir"]) for t in enabled})
    try:
        seed_hash_fn = None
        if any(tier["type"] == "revolta_fs" for tier in enabled):
            from vllm.utils.hashing import get_hash_fn_by_name

            seed_hash_fn = hash_fn or get_hash_fn_by_name(
                getattr(
                    getattr(vllm_config, "cache_config", None),
                    "prefix_caching_hash_algo",
                    "sha256",
                )
            )
        return _locked_fs_hash_seed(roots, seed_hash_fn)
    except (OSError, ValueError, TypeError) as exc:
        if not any(tier["type"] == "revolta_fs" for tier in enabled):
            raise
        from vllm.v1.kv_offload.tiering.fs.identity import disable_revolta_root

        for tier in enabled:
            if tier["type"] == "revolta_fs":
                disable_revolta_root(vllm_config, tier["root_dir"], str(exc))
        # Preserve plain fs startup failures and seed selection. Never repair
        # or overwrite an unreadable identity belonging to a disabled tier.
        return _locked_fs_hash_seed(
            sorted(
                {os.path.realpath(t["root_dir"]) for t in enabled if t["type"] == "fs"}
            )
        )


def _locked_fs_hash_seed(
    roots: list[str], hash_fn: Callable[[Any], bytes] | None = None
) -> str | None:
    if not roots:
        return None
    with ExitStack() as locks:
        seeds = set()
        missing = []
        for root in roots:
            os.makedirs(root, exist_ok=True)
            lock = locks.enter_context(
                open(os.path.join(root, ".hash_seed.lock"), "a+")
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = Path(root) / ".hash_seed.json"
            if not path.exists():
                missing.append(path)
                continue
            with path.open() as source:
                data = json.load(source)
            if (
                not isinstance(data, dict)
                or data.get("version") != 1
                or not isinstance(data.get("seed"), str)
                or not data["seed"]
            ):
                raise ValueError(
                    f"Invalid FS hash identity in {path}; refusing to reset it"
                )
            seeds.add(data["seed"])
        if len(seeds) > 1:
            raise ValueError(
                "FS tier roots have different hash seeds; use matching roots"
            )
        seed = next(iter(seeds)) if seeds else os.getenv("PYTHONHASHSEED") or None
        if seed is None:
            seed = os.urandom(32).hex()
        if hash_fn is not None:
            # Validate before propagating a seed to other roots, and while
            # failures can still disable every already-created manager.
            seed.encode("utf-8")
            hash_fn(seed)
        for path in missing:
            temporary = path.with_suffix(".json.new")
            fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as target:
                json.dump({"version": 1, "seed": seed}, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return seed
