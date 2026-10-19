# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FS budget validation and process-independent prefix-hash identity."""

import fcntl
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig


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


def fs_tier_configs(config: "KVTransferConfig | None") -> list[dict]:
    if config is None or config.kv_connector != "OffloadingConnector":
        return []
    tiers = config.kv_connector_extra_config.get("secondary_tiers", [])
    if not isinstance(tiers, list) or any(not isinstance(t, dict) for t in tiers):
        raise ValueError("secondary_tiers must be a list of tier configurations")
    result = [tier for tier in tiers if tier.get("type") == "fs"]
    for tier in result:
        validate_fs_budget(tier.get("max_size_bytes"), tier.get("min_free_bytes"))
        if not isinstance(tier.get("root_dir"), str) or not tier["root_dir"]:
            raise ValueError("FS tier requires a non-empty root_dir")
    return result


def get_fs_hash_seed(config: "KVTransferConfig | None") -> str | None:
    """Lock configured roots and persist one seed before hashing requests.

    This short identity lock is separate from FileSystemCache's lifetime
    writer lock. Group pools and engines can read the same identity without
    weakening single-engine ownership. Existing roots must agree because a
    process has one NONE_HASH; changing an identity would lose cache hits.
    """
    roots = sorted({os.path.realpath(t["root_dir"]) for t in fs_tier_configs(config)})
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
