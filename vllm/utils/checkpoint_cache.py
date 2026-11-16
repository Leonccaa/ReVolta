# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, post-load checkpoint page-cache advice (never a global cache drop)."""

import os
import stat
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)
FileId = tuple[int, int]
_loading_files: ContextVar[set[str] | None] = ContextVar(
    "checkpoint_files", default=None
)


@contextmanager
def collect_checkpoint_files() -> Iterator[set[str]]:
    """Collect only this startup's loader inputs, including the draft model."""
    files: set[str] = set()
    token = _loading_files.set(
        files if envs.REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP else None
    )
    try:
        yield files
    finally:
        _loading_files.reset(token)


def record_checkpoint_files(files: Iterable[str]) -> None:
    active = _loading_files.get()
    if active is not None:
        active.update(os.path.realpath(path) for path in files)


def _mapped_file_ids(maps: Path = Path("/proc/self/maps")) -> set[FileId] | None:
    """Protect live file-backed tensors, using inode identity for aliases."""
    try:
        result = set()
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            major, minor = (int(part, 16) for part in fields[3].split(":"))
            inode = int(fields[4])
            if inode:
                result.add((os.makedev(major, minor), inode))
        return result
    except (OSError, ValueError, IndexError):
        return None  # Do not guess which checkpoint mappings remain live.


def _cgroup_memory(
    root: Path = Path("/sys/fs/cgroup"),
    membership: Path = Path("/proc/self/cgroup"),
) -> tuple[int | None, int | None]:
    """Current leaf cgroup usage and file charge, in bytes; unknown is None."""
    root = root.resolve()
    node = root
    try:
        entry = next(
            line[3:]
            for line in membership.read_text().splitlines()
            if line.startswith("0::")
        )
        candidate = (root / entry.lstrip("/")).resolve()
        if candidate.is_relative_to(root) and (candidate / "memory.current").is_file():
            node = candidate
    except (OSError, StopIteration):
        pass
    try:
        current = int((node / "memory.current").read_text())
    except (OSError, ValueError):
        current = None
    try:
        values = dict(
            line.split() for line in (node / "memory.stat").read_text().splitlines()
        )
        file_bytes = int(values["file"])
    except (OSError, ValueError, KeyError):
        file_bytes = None
    return current, file_bytes


def _synchronize_loaders() -> "GroupCoordinator":
    import torch

    from vllm.distributed.parallel_state import get_world_group

    # Finish asynchronous weight copies before any rank can advise shared files.
    torch.accelerator.synchronize()
    group = get_world_group()
    group.barrier()  # CPU/Gloo, covers TP, PP and draft loading on every worker.
    return group


def _gather_mapped_files(group: "GroupCoordinator") -> set[FileId] | None:
    import torch.distributed as dist

    local = _mapped_file_ids()
    if group.world_size == 1:
        return local
    mappings: list[set[FileId] | None] = [None] * group.world_size
    dist.all_gather_object(mappings, local, group=group.cpu_group)
    if any(item is None for item in mappings):
        return None
    return set().union(*(item for item in mappings if item is not None))


def release_checkpoint_cache(files: set[str], *, protect_all: bool = False) -> None:
    """Advise only completed loader inputs after all ranks finish loading.

    The advice is best effort. File sizes are advised bytes, not bytes actually
    evicted. Live mmap inputs are excluded across all ranks. An independent PLE
    disk reader is outside this group, so its modes conservatively skip advice.
    All ranks must use the same switch and call this even with no local files.
    """
    if not envs.REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP:
        logger.info("Checkpoint cache drop disabled (research switch is off).")
        return
    started = time.monotonic()
    group = _synchronize_loaders()
    before = _cgroup_memory()
    mapped = _gather_mapped_files(group)
    advised_files = advised_bytes = skipped = failed = 0
    seen: set[FileId] = set()
    supported = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
    try:
        for path in sorted(files):
            if protect_all or mapped is None or not supported:
                skipped += 1
                continue
            try:
                # Never write, truncate, fsync or fault in checkpoint contents.
                with open(path, "rb") as source:
                    info = os.fstat(source.fileno())
                    identity = (info.st_dev, info.st_ino)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    if identity in mapped or not stat.S_ISREG(info.st_mode):
                        skipped += 1
                        continue
                    os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    advised_files += 1
                    advised_bytes += info.st_size
            except OSError as error:
                failed += 1
                logger.warning("Checkpoint cache advice failed: errno=%s", error.errno)
    finally:
        # No rank may start the large host allocation while siblings advise.
        group.barrier()
    after = _cgroup_memory()
    logger.info(
        "Checkpoint cache drop: advised_files=%d advised_bytes=%d skipped=%d "
        "failed=%d protect_all=%s mappings_known=%s supported=%s; "
        "cgroup memory.current=%s -> %s bytes, memory.stat file=%s -> %s bytes; "
        "elapsed=%.3f s (advice is not an eviction guarantee).",
        advised_files,
        advised_bytes,
        skipped,
        failed,
        protect_all,
        mapped is not None,
        supported,
        before[0],
        after[0],
        before[1],
        after[1],
        time.monotonic() - started,
    )
