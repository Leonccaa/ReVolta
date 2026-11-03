# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReVolta file-system secondary tier (``type: "revolta_fs"``).

A stricter variant of ``fs`` for using the disk tier as a persistent cold
store across engine restarts. It shares the ``fs`` directory layout, hash
seed, budget and thread pool, and only takes effect when configured: plain
``fs`` tiers are unchanged.
"""

from typing import TYPE_CHECKING

from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


class RevoltaFileSystemTierManager(FileSystemTierManager):
    """File-system secondary tier for persistent, verified KV reuse."""

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        **kwargs,
    ):
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, root_dir, **kwargs
        )
