# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A disk budget shared by all group pools using a dedicated FS cache root.

The root belongs to one scheduler process. A file lock excludes other engines;
group managers in the same process share accounting, reservations and read pins.
Only completed .bin pages count toward the budget; in-flight writes reserve their
full size before I/O. This is a disposable cache, not durable model storage.
"""

import errno
import fcntl
import heapq
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import ClassVar


class FileSystemCache:
    _registry: ClassVar[dict[str, "FileSystemCache"]] = {}
    _registry_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def acquire(
        cls,
        root: str,
        max_size_bytes: int | None,
        min_free_bytes: int | None = None,
        max_files: int = 1_000_000,
        external_eviction: bool = False,
    ) -> "FileSystemCache":
        root = os.path.realpath(root)
        from vllm.v1.kv_offload.tiering.fs.config import validate_fs_budget

        validate_fs_budget(max_size_bytes, min_free_bytes)
        if max_files <= 0:
            raise ValueError("max_files must be positive")
        if min_free_bytes is None:
            os.makedirs(root, exist_ok=True)
            stat = os.statvfs(root)
            min_free_bytes = max(1, stat.f_blocks * stat.f_frsize // 20)
        with cls._registry_lock:
            cache = cls._registry.get(root)
            if cache is None:
                cache = cls(
                    root,
                    max_size_bytes,
                    min_free_bytes,
                    max_files,
                    external_eviction=external_eviction,
                )
                cls._registry[root] = cache
            elif (cache.max_size_bytes, cache.min_free_bytes, cache.max_files) != (
                max_size_bytes,
                min_free_bytes,
                max_files,
            ):
                raise ValueError("FS tiers sharing a root must use the same budget")
            cache._users += 1
            cache.external_eviction |= external_eviction
            return cache

    def __init__(
        self,
        root: str,
        max_size_bytes: int | None,
        min_free_bytes: int,
        max_files: int,
        *,
        external_eviction: bool = False,
    ):
        self.root = root
        self.max_size_bytes = max_size_bytes
        self.min_free_bytes = min_free_bytes
        self.max_files = max_files
        self.external_eviction = external_eviction
        self._claims: dict[str, tuple[int, int]] = {}
        self._users = 0
        self._lock = threading.Lock()
        self._pins: dict[str, int] = {}
        self._writing: set[str] = set()
        self.loss_listeners: list[Callable[[str], None]] = []
        self._reserved = 0
        self._bytes = 0
        self._pages: OrderedDict[str, int] = OrderedDict()
        os.makedirs(root, exist_ok=True)
        self._lock_fd = os.open(
            os.path.join(root, ".lock"), os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # The lock makes abandoned atomic-write temporaries safe to remove.
            if not external_eviction:
                for path in Path(root).rglob("*.tmp"):
                    path.unlink()
            pages: list[tuple[int, str, int]] = []
            for path in Path(root).rglob("*"):
                if not path.is_file() or not (
                    path.suffix == ".bin"
                    or external_eviction
                    and (path.name.endswith(".unit.json") or path.suffix == ".tmp")
                ):
                    continue
                stat = path.stat()
                heapq.heappush(pages, (stat.st_mtime_ns, str(path), stat.st_size))
                if not external_eviction and len(pages) > max_files:
                    _, victim, _ = heapq.heappop(pages)
                    os.unlink(victim)
            for _, name, size in sorted(pages):
                self._pages[name] = size
                self._bytes += size
            try:
                if not external_eviction:
                    self._make_room(0)
            except OSError as exc:
                # Other filesystem users may consume the reserve. Keep reads
                # and CPU serving available; subsequent stores fail as jobs.
                if exc.errno != errno.ENOSPC:
                    raise
        except BaseException:
            os.close(self._lock_fd)
            raise

    def release(self) -> None:
        with self._registry_lock:
            self._users -= 1
            if self._users == 0:
                del self._registry[self.root]
                os.close(self._lock_fd)

    def _forget(self, path: str) -> None:
        known = path in self._pages
        self._bytes -= self._pages.pop(path, 0)
        if known and self.loss_listeners and not os.path.exists(path):
            for listener in self.loss_listeners:
                listener(path)

    def _make_room(self, size: int) -> None:
        def fits() -> bool:
            pending_files = sum(p not in self._pages for p in self._writing)
            if len(self._pages) + pending_files + int(size > 0) > self.max_files:
                return False
            if self.max_size_bytes is not None and (
                self._bytes + self._reserved + size > self.max_size_bytes
            ):
                return False
            stat = os.statvfs(self.root)
            free = stat.f_bavail * stat.f_frsize
            return free >= self.min_free_bytes + self._reserved + size

        # Never evict a page being read or written, even across group pools.
        while not fits():
            if self.external_eviction:
                raise OSError(errno.ENOSPC, "Unified SSD reservation required")
            victim = next((p for p in self._pages if p not in self._pins), None)
            if victim is None:
                raise OSError(
                    errno.ENOSPC, "FS cache budget or free-space reserve exhausted"
                )
            with suppress(FileNotFoundError):
                os.unlink(victim)
            self._forget(victim)

    def reserve_paths(self, paths: dict[str, int]) -> bool:
        """Atomically reserve a complete transaction, sharing pending paths."""
        with self._lock:
            fresh = {
                p: n
                for p, n in paths.items()
                if p not in self._pages and p not in self._claims
            }
            size = sum(fresh.values())
            stat = os.statvfs(self.root)
            pending_files = sum(path not in self._pages for path in self._claims)
            if (
                len(self._pages) + pending_files + len(fresh) > self.max_files
                or self.max_size_bytes is not None
                and self._bytes + self._reserved + size > self.max_size_bytes
                or stat.f_bavail * stat.f_frsize
                < self.min_free_bytes + self._reserved + size
            ):
                return False
            for path in paths:
                amount, refs = self._claims.get(path, (fresh.get(path, 0), 0))
                self._claims[path] = amount, refs + 1
            self._reserved += size
            return True

    def release_paths(self, paths: Iterable[str]) -> None:
        with self._lock:
            for path in paths:
                amount, refs = self._claims[path]
                if refs == 1:
                    self._reserved -= amount
                    del self._claims[path]
                else:
                    self._claims[path] = amount, refs - 1

    def discard(self, path: str) -> bool:
        """Explicit authority deletion. Pins include every transaction source."""
        with self._lock:
            if path in self._pins or path in self._claims or path in self._writing:
                return False
            with suppress(FileNotFoundError):
                os.unlink(path)
            self._forget(path)
            return True

    def lookup(self, path: str, on_found: Callable[[], None] | None = None) -> bool:
        """Optionally install a proof under the deletion lock.

        The callback must not re-enter the cache. This makes publication and
        loss notifications ordered even when a path is deleted and rewritten.
        """
        with self._lock:
            if not os.path.isfile(path):
                self._forget(path)
                return False
            if path in self._pages:
                self._pages.move_to_end(path)
            if on_found is not None:
                on_found()
            return True

    def touch(self, paths: Iterable[str]) -> None:
        """Mark resident pages as recently used, in the given order (the last
        one becomes the most recent). In memory only: no filesystem calls."""
        with self._lock:
            for path in paths:
                if path in self._pages:
                    self._pages.move_to_end(path)

    @contextmanager
    def read(self, path: str) -> Iterator[None]:
        with self._lock:
            self._pins[path] = self._pins.get(path, 0) + 1
            if path in self._pages:
                self._pages.move_to_end(path)
        try:
            yield
        finally:
            with self._lock:
                self._pins[path] -= 1
                if self._pins[path] == 0:
                    del self._pins[path]
                if not os.path.exists(path):
                    self._forget(path)

    @contextmanager
    def write(self, path: str, size: int) -> Iterator[None]:
        reserved = 0
        with self._lock:
            if path in self._writing:
                raise OSError(errno.EBUSY, "FS cache page already being written")
            if not os.path.exists(path):
                if self.max_size_bytes is not None and size > self.max_size_bytes:
                    raise OSError(errno.ENOSPC, "FS cache page exceeds budget")
                if path not in self._claims:
                    if self.external_eviction:
                        raise OSError(errno.ENOSPC, "Unified SSD reservation required")
                    self._make_room(size)
                    reserved = size
                    self._reserved += size
            self._writing.add(path)
            self._pins[path] = self._pins.get(path, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._reserved -= reserved
                if path in self._claims:
                    amount, refs = self._claims[path]
                    if os.path.exists(path):
                        self._reserved -= amount
                        self._claims[path] = 0, refs
                self._writing.remove(path)
                self._pins[path] -= 1
                if self._pins[path] == 0:
                    del self._pins[path]
                self._forget(path)
                if os.path.exists(path):
                    actual_size = os.path.getsize(path)
                    self._pages[path] = actual_size
                    self._bytes += actual_size
