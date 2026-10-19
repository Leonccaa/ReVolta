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
from collections.abc import Iterator
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
        min_free_bytes: int,
        max_files: int = 1_000_000,
    ) -> "FileSystemCache":
        root = os.path.realpath(root)
        if max_size_bytes is not None and max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes must be non-negative")
        if max_files <= 0:
            raise ValueError("max_files must be positive")
        with cls._registry_lock:
            cache = cls._registry.get(root)
            if cache is None:
                cache = cls(root, max_size_bytes, min_free_bytes, max_files)
                cls._registry[root] = cache
            elif (cache.max_size_bytes, cache.min_free_bytes, cache.max_files) != (
                max_size_bytes,
                min_free_bytes,
                max_files,
            ):
                raise ValueError("FS tiers sharing a root must use the same budget")
            cache._users += 1
            return cache

    def __init__(
        self, root: str, max_size_bytes: int | None, min_free_bytes: int, max_files: int
    ):
        self.root = root
        self.max_size_bytes = max_size_bytes
        self.min_free_bytes = min_free_bytes
        self.max_files = max_files
        self._users = 0
        self._lock = threading.Lock()
        self._pins: dict[str, int] = {}
        self._writing: set[str] = set()
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
            for path in Path(root).rglob("*.tmp"):
                path.unlink()
            pages: list[tuple[int, str, int]] = []
            for path in Path(root).rglob("*.bin"):
                stat = path.stat()
                heapq.heappush(pages, (stat.st_mtime_ns, str(path), stat.st_size))
                if len(pages) > max_files:
                    _, victim, _ = heapq.heappop(pages)
                    os.unlink(victim)
            for _, name, size in sorted(pages):
                self._pages[name] = size
                self._bytes += size
            try:
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
        self._bytes -= self._pages.pop(path, 0)

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
            victim = next((p for p in self._pages if p not in self._pins), None)
            if victim is None:
                raise OSError(
                    errno.ENOSPC, "FS cache budget or free-space reserve exhausted"
                )
            with suppress(FileNotFoundError):
                os.unlink(victim)
            self._forget(victim)

    def lookup(self, path: str) -> bool:
        with self._lock:
            if not os.path.isfile(path):
                self._forget(path)
                return False
            if path in self._pages:
                self._pages.move_to_end(path)
            return True

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
                self._writing.remove(path)
                self._pins[path] -= 1
                if self._pins[path] == 0:
                    del self._pins[path]
                self._forget(path)
                if os.path.exists(path):
                    actual_size = os.path.getsize(path)
                    self._pages[path] = actual_size
                    self._bytes += actual_size
