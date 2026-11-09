# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded direct host-pool SSD write-through and verified cold promotion.

No GPU or second CPU KV allocation is involved. All controller methods run on
one thread, like the existing offload manager. Jobs pin the authoritative row
until physical I/O finishes, including cancellation and shutdown. Only poll()
can publish a verified read, and that publication is local to this KV group.
"""

import copy
import math
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from vllm.v1.kv_offload.base import OffloadingSpec, OffloadKey, ReqContext
from vllm.v1.kv_offload.host.pool import HostKVPage, HostKVPool, HostKVPrefix
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.fs.config import host_kv_ssd_config
from vllm.v1.kv_offload.tiering.fs.revolta import (
    ChecksumError,
    RevoltaFileSystemTierManager,
    StoreCancelled,
    load_verified_block,
    store_verified_block,
)


@dataclass(frozen=True)
class HostKVLayout:
    """Byte ABI supplied by host KV pool/host-history integration, bound into the
    persistent namespace.

    storage_layout must describe head geometry, packing, layer/member order
    and padding. Bump version whenever that contract changes. Scales are the
    actual calibrated (K, V) values in layer_names order, not their file path.
    Parallel geometry and model content come from the validated engine spec.
    """

    version: str
    storage_layout: str
    encoding: str
    layer_names: tuple[str, ...]
    kv_scales: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not all(
            (self.version, self.storage_layout, self.encoding, self.layer_names)
        ):
            raise ValueError("Host KV requires an explicit storage ABI")
        if len(self.kv_scales) != len(self.layer_names) or len(
            set(self.layer_names)
        ) != len(self.layer_names):
            raise ValueError(
                "Host KV requires unique layers and their actual KV scales"
            )
        if any(
            len(pair) != 2 or any(not math.isfinite(x) or x <= 0 for x in pair)
            for pair in self.kv_scales
        ):
            raise ValueError("Host KV scales must be finite and positive")


class _WriteRateLimiter:
    """Shared by this adapter's writers, at most one page of initial burst."""

    def __init__(self, bytes_per_second: int):
        self._rate = bytes_per_second
        self._next = 0.0
        self._lock = threading.Lock()
        self.stopped = threading.Event()
        self.unthrottled = threading.Event()

    def unthrottle(self) -> None:
        """Final flush: lift the rate limit. A writer already waiting finishes
        at most one page interval (page / rate) first."""
        self.unthrottled.set()

    def acquire(self, size: int) -> None:
        with self._lock:
            while not self.unthrottled.is_set():
                now = time.monotonic()
                delay = max(0.0, self._next - now)
                if self.stopped.wait(delay):
                    raise StoreCancelled("Host KV SSD writer stopped")
                if time.monotonic() >= self._next:
                    break
            if not self.unthrottled.is_set():
                self._next = max(time.monotonic(), self._next) + size / self._rate


class _HostTier(RevoltaFileSystemTierManager):
    """Reuse existing identity, CRC32C, atomic publication and disk budget.

    Host rows need not be O_DIRECT aligned. Buffered readv/write still access
    the pool buffer directly. The shared checksum helper may make a temporary
    page copy when its native CRC library is unavailable.
    The kernel page cache is outside our KV pool budget, as with any buffered
    file I/O. This format has its own namespace, separate from GPU layouts.
    """

    def __init__(self, limiter: _WriteRateLimiter, *args, **kwargs):
        self.limiter = limiter
        super().__init__(*args, **kwargs)

    def _write_page(self, key: OffloadKey, path: str, block_id: int) -> None:
        self.limiter.acquire(self._block_size)
        with self._cache.write(path, self._block_size):
            written = store_verified_block(
                path,
                self._primary_kv_view,
                block_id * self._block_size,
                self._block_size,
                key,
                self._identity_digest,
                direct_io=False,
            )
        self._count("pages_written" if written else "pages_existing")

    def _read_page(self, key: OffloadKey, path: str, block_id: int) -> None:
        with self._cache.read(path):
            try:
                load_verified_block(
                    path,
                    self._primary_kv_view,
                    block_id * self._block_size,
                    self._block_size,
                    key,
                    self._identity_digest,
                    direct_io=False,
                )
            except ChecksumError:
                self._count("checksum_failures")
                raise


class HostKVPrefixCache:
    """host prefix adapter controller; construction is explicit, SSD is disabled by
    default.

    Configure kv_connector_extra_config["host_kv.ssd"]. host-history integration must
    remove main
    KV from ordinary CPU offload and call this controller over the actual host
    backing. The existing connector continues to own index/state/draft groups.
    This class never increments global external-hit counters or claims that
    other groups/ranks can restore a prefix.
    """

    def __init__(self, pool: HostKVPool, spec: OffloadingSpec, layout: HostKVLayout):
        self.pool = pool
        self._tier: _HostTier | None = None
        self._pending: dict[int, tuple[HostKVPage, bool]] = {}
        self._next_job = 0
        self._closed = False
        self._stats: Counter[str] = Counter()
        options = host_kv_ssd_config(spec.vllm_config.kv_transfer_config)
        self._max_pending = 0 if options is None else options["max_pending_blocks"]
        self._drain_timeout = 0.0 if options is None else options["drain_timeout_s"]
        self._limiter = _WriteRateLimiter(
            (64 << 20) if options is None else options["write_bytes_per_second"]
        )
        if options is None:
            return
        # Shallow-copy only the spec wrapper. Never mutate the caller's config
        # or copy its KV storage. Host identity includes physical row geometry.
        host_spec = copy.copy(spec)
        host_spec.__dict__["persistent_layout"] = {
            **(getattr(spec, "persistent_layout", None) or {}),
            "host_kv": {
                "abi": asdict(layout),
                "group_idx": pool.group_idx,
                "block_tokens": pool.block_tokens,
                "page_bytes": pool.page_bytes,
                "io": "buffered-host-direct-v1",
            },
        }
        self._tier = _HostTier(
            self._limiter,
            offloading_spec=host_spec,
            primary_kv_view=pool.view,
            tier_type="revolta_fs",
            root_dir=options["root_dir"],
            max_size_bytes=options["max_size_bytes"],
            min_free_bytes=options.get("min_free_bytes"),
            max_files=options["max_files"],
            content_manifest_path=options.get("content_manifest_path"),
            content_verify=options.get("content_verify", "stat"),
            n_read_threads=1,
            n_write_threads=1,
            # Never expire a job while its source/destination can still be in
            # use. Bounded backpressure limits retained rows even on slow I/O.
            store_timeout_s=0,
            drain_timeout_s=0,
            max_inflight_store_blocks=self._max_pending,
            write_delay_s=0,
        )

    @property
    def enabled(self) -> bool:
        return not self._closed and self._tier is not None and not self._tier.disabled

    @property
    def pending_blocks(self) -> int:
        """Queued, active or awaiting poll; includes reads and writes."""
        return len(self._pending)

    def stats(self) -> dict[str, int]:
        """Group-local RAM prefix queries and verified SSD pages/bytes only."""
        return {**self._stats, "pending_blocks": self.pending_blocks}

    def reader_ready(self, key: OffloadKey) -> bool:
        """Only a verified, published RAM row can serve a reader-ready probe.

        SSD existence and pending reservations never qualify. Call poll on
        the controller thread before a lookup to publish completed reads.
        """
        return not self._closed and self.pool.is_ready(self.pool.lookup(key))

    def _has_capacity(self) -> bool:
        if self.pending_blocks >= self._max_pending:
            self._stats["queue_rejections"] += 1
            return False
        return True

    def _submit(self, page: HostKVPage, loading: bool) -> bool:
        tier = self._tier
        if tier is None:
            self.pool.release(page)
            return False
        self._next_job += 1
        job_id = self._next_job
        self._pending[job_id] = (page, loading)
        job = JobMetadata(
            job_id=job_id,
            keys=[page.key],
            block_ids=np.array([page.block_id], dtype=np.int64),
            is_promotion=loading,
            req_context=ReqContext(req_id="host-kv-prefix"),
        )
        try:
            if loading:
                tier.submit_load(job)
            else:
                tier.submit_store(job)
        except Exception:
            self._pending.pop(job_id)
            self.pool.release(page)
            self._stats["submit_failures"] += 1
            return False
        return True

    def submit_store(self, key: OffloadKey) -> bool:
        """Enqueue an immutable host page; False skips persistence, RAM stays."""
        self.poll()
        if not self.enabled or not self._has_capacity():
            return False
        # At most one store per key may be queued, including retries by host-history
        # integration.
        if any(p.key == key and not load for p, load in self._pending.values()):
            return False
        page = self.pool.pin(self.pool.lookup(key))
        return page is not None and self._submit(page, loading=False)

    def prefetch(self, key: OffloadKey) -> bool:
        """Reserve and enqueue a direct SSD read; no hit before verification."""
        self.poll()
        if not self.enabled or not self._has_capacity():
            return False
        if self.pool.lookup(key) is not None:
            return False
        tier = self._tier
        if tier is None:
            return False
        try:
            if tier.lookup(key) is not True:
                return False
        except OSError:
            self._stats["lookup_failures"] += 1
            return False
        page = self.pool.reserve(key)
        return page is not None and self._submit(page, loading=True)

    def poll(self) -> None:
        if self._tier is None or self._closed:
            return
        for result in self._tier.get_finished_jobs():
            page, loading = self._pending.pop(result.job_id)
            if loading and result.success:
                self.pool.publish(page)
                self._stats["ssd_verified_pages"] += 1
                self._stats["ssd_verified_bytes"] += self.pool.page_bytes
            elif not result.success:
                self._stats["load_failures" if loading else "store_failures"] += 1
            self.pool.release(page)

    def acquire_prefix(self, keys: Sequence[OffloadKey]) -> HostKVPrefix:
        """Pin only this group's ready contiguous prefix, starting at token 0.

        A pending or corrupt tail never contributes tokens. The caller may
        prefetch the first missing key and retry, holding the prefix leases.
        complete-prefix contract owns cross-group/rank intersection, cancellation and
        final commit.
        """
        if self._closed:
            return HostKVPrefix(self.pool.group_idx, 0, ())
        self.poll()
        prefix = self.pool.acquire_prefix(keys)
        self._stats["group_prefix_queries"] += 1
        self._stats["group_prefix_tokens"] += prefix.available_tokens
        if self.enabled and self._tier is not None and prefix.pages:
            self._tier.touch([page.key for page in prefix.pages])
        return prefix

    def flush(self, timeout: float) -> bool:
        """CPU control-plane drain; does not turn failures into durable stores."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Host KV flush timeout must be finite and >= 0")
        deadline = time.monotonic() + timeout
        while True:
            self.poll()
            if not self._pending:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def close(self) -> None:
        if self._closed:
            return
        self.flush(self._drain_timeout)
        self._limiter.stopped.set()
        if self._tier is not None:
            # Join physical tasks before releasing any row, including aborted
            # reads. No timed-out writer can read a recycled source buffer.
            self._tier.shutdown()
        for page, loading in self._pending.values():
            self._stats["cancelled_loads" if loading else "cancelled_stores"] += 1
            self.pool.release(page)
        self._pending.clear()
        self._closed = True
