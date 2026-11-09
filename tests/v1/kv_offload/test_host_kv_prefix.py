# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only direct host KV persistence, using real files and fresh processes."""

import errno
import json
import mmap
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tests.v1.kv_offload.test_revolta_fs import checkpoint_spec
from vllm.config import KVTransferConfig
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host.pool import HostKVPool
from vllm.v1.kv_offload.host.ssd import (
    HostKVLayout,
    HostKVPrefixCache,
    _WriteRateLimiter,
)
from vllm.v1.kv_offload.tiering.fs.config import fs_tier_configs

# Deliberately unaligned page lengths and ordinary bytearray storage: these
# cannot accidentally rely on the old O_DIRECT CPU-offload row geometry.
PAGE_BYTES = 257
BLOCK_TOKENS = 16
LAYOUT = HostKVLayout(
    version="cpu-fixture-v1",
    storage_layout="layer,K/V,token,head,dim; uint8; no padding",
    encoding="uint8-fixture",
    layer_names=("qsa.0", "qsa.1"),
    kv_scales=((0.5, 2.0), (0.25, 1.0)),
)


def key(n, group=3):
    return make_offload_key(sha256(n), group)


def payload(n=0):
    return bytes((i + n) % 256 for i in range(PAGE_BYTES))


def pool(slots=4, group=3):
    backing = bytearray(slots * PAGE_BYTES)
    view = memoryview(backing).cast("B", shape=[slots, PAGE_BYTES])
    return HostKVPool(view, group_idx=group, block_tokens=BLOCK_TOKENS)


def commit(pages, k, data=None):
    page = pages.reserve(k)
    assert page is not None
    page.view[:] = payload() if data is None else data
    pages.publish(page)
    bid = page.block_id
    pages.release(page)
    return bid


def cache(root, pages=None, layout=LAYOUT, **overrides):
    spec, manifest = checkpoint_spec(root)
    options = {
        "enabled": True,
        "root_dir": str(root),
        "max_size_bytes": 1 << 20,
        "min_free_bytes": 0,
        "content_manifest_path": manifest,
        "max_pending_blocks": 4,
        "write_bytes_per_second": 1 << 30,
        "drain_timeout_s": 0,
        **overrides,
    }
    spec.vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"host_kv.ssd": options},
    )
    kv_cache_utils.init_none_hash(
        sha256,
        kv_transfer_config=spec.vllm_config.kv_transfer_config,
        vllm_config=spec.vllm_config,
    )
    return HostKVPrefixCache(pool() if pages is None else pages, spec, layout)


@pytest.fixture
def caches():
    opened = []

    def make(*args, **kwargs):
        result = cache(*args, **kwargs)
        opened.append(result)
        return result

    yield make
    for result in reversed(opened):
        result.close()


def stored(c, keys):
    for i, k in enumerate(keys):
        commit(c.pool, k, payload(i))
        assert c.submit_store(k)
    assert c.flush(5)
    assert c.stats().get("store_failures", 0) == 0


def test_disabled_default_keeps_ram_prefix_and_does_no_ssd_io(tmp_path, caches):
    root = tmp_path / "ssd"
    c = caches(root, enabled=False)
    assert not c.enabled and not root.exists()
    assert fs_tier_configs(None) == []
    commit(c.pool, key(1))
    assert not c.submit_store(key(1))
    assert not c.prefetch(key(2))
    prefix = c.acquire_prefix([key(1), key(2)])
    assert prefix.group_idx == 3 and prefix.available_tokens == BLOCK_TOKENS
    prefix.release()
    assert not root.exists()


@pytest.mark.parametrize("options", [{}, {"host_kv.ssd": {}}])
def test_absent_switch_has_no_side_effects(tmp_path, options):
    spec, _ = checkpoint_spec(tmp_path / "unused")
    config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=options,
    )
    spec.vllm_config.kv_transfer_config = config
    before = {thread.ident for thread in threading.enumerate()}
    c = HostKVPrefixCache(pool(), spec, LAYOUT)
    try:
        assert not c.enabled and c._tier is None
        assert fs_tier_configs(config) == []
        assert {thread.ident for thread in threading.enumerate()} == before
    finally:
        c.close()


def test_pool_is_authority_without_a_second_payload_and_lru_respects_leases():
    pages = pool(slots=2)
    bid = commit(pages, key(1), payload(7))
    assert pages.view.obj[bid * PAGE_BYTES : (bid + 1) * PAGE_BYTES] == payload(7)
    active = pages.pin(pages.lookup(key(1)))
    assert active is not None and active.view.readonly
    commit(pages, key(2))
    # Touch the other page, making the active page oldest. It still cannot go.
    other = pages.pin(pages.lookup(key(2)))
    assert other is not None
    pages.release(other)
    commit(pages, key(3))
    assert pages.is_ready(pages.lookup(key(1))) and pages.lookup(key(2)) is None
    pages.release(active)
    with pytest.raises(ValueError, match="released"):
        _ = active.view
    pages.release(active)  # Idempotent release cannot drop a later lease.
    commit(pages, key(4))
    assert pages.lookup(key(1)) is None


def test_unpublished_aborted_and_mixed_group_pages_cannot_hit():
    pages = pool(slots=1)
    page = pages.reserve(key(1))
    assert page is not None
    page.view[:] = payload()
    assert pages.lookup(key(1)) is not None
    assert not pages.is_ready(pages.lookup(key(1)))
    assert pages.pin(pages.lookup(key(1))) is None
    assert pages.reserve(key(2)) is None
    pages.release(page)
    assert pages.lookup(key(1)) is None
    commit(pages, key(2))
    with pytest.raises(ValueError, match="group"):
        pages.acquire_prefix([key(2), key(3, group=4)])
    commit(pages, key(3))  # The mixed-group failure leaked no read pin.


def test_an_in_flight_row_cannot_be_reserved_twice():
    pages = pool(slots=2)
    first = pages.reserve(key(1))
    assert first is not None
    # An unpublished row may be located, but it is never ready to hit.
    handle = pages.lookup(key(1))
    assert handle is None or not pages.is_ready(handle)
    # One writer per key: a duplicate hash must not get a second destination.
    assert pages.reserve(key(1)) is None
    first.view[:] = payload(1)
    pages.publish(first)
    assert pages.reserve(key(1)) is None
    pages.release(first)
    prefix = pages.acquire_prefix([key(1)])
    assert [bytes(p.view) for p in prefix.pages] == [payload(1)]
    prefix.release()
    commit(pages, key(2))  # The second slot was never taken by a duplicate.


def test_only_a_contiguous_prefix_contributes_group_availability():
    pages = pool()
    commit(pages, key(3))
    assert pages.acquire_prefix([key(1), key(2), key(3)]).available_tokens == 0
    commit(pages, key(1))
    prefix = pages.acquire_prefix([key(1), key(2), key(3)])
    assert prefix.available_tokens == BLOCK_TOKENS
    assert len(prefix.pages) == 1
    prefix.release()


def test_real_files_restart_read_directly_into_new_host_pool(tmp_path, caches):
    root = tmp_path / "ssd"
    first = caches(root)
    keys = [key(1), key(2), key(3)]
    stored(first, keys)
    first.close()
    second = caches(root)
    assert second.pool.view.obj is not first.pool.view.obj
    assert second.acquire_prefix(keys).available_tokens == 0
    for k in keys:
        assert second.prefetch(k)
    # A reserved destination is not yet readable, regardless of worker timing.
    assert second.pool.lookup(keys[0]) is not None
    assert not second.pool.is_ready(second.pool.lookup(keys[0]))
    assert second.flush(5)
    prefix = second.acquire_prefix(keys)
    assert prefix.available_tokens == 3 * BLOCK_TOKENS
    assert [bytes(p.view) for p in prefix.pages] == [payload(i) for i in range(3)]
    assert second.stats()["ssd_verified_pages"] == 3
    assert second.stats()["ssd_verified_bytes"] == 3 * PAGE_BYTES
    prefix.release()


def test_full_length_page_key_is_written_and_hits_after_restart(tmp_path, caches):
    """A concatenated page key (98 x 32-byte hashes + group) would need a
    6272-character file name. The SSD file is keyed by a fixed-length digest,
    stable across restarts, and the stored identity still binds the key."""
    root = tmp_path / "ssd"
    long_key = make_offload_key(b"".join(sha256(i) for i in range(98)), 3)
    assert len(long_key) == 98 * 32 + 4
    first = caches(root)
    stored(first, [long_key])
    path = Path(first._tier.file_mapper.get_file_name(first.tier_key(long_key)))
    assert path.is_file() and len(path.name) < 255
    first.close()
    second = caches(root)
    assert second.prefetch(long_key)
    assert second.flush(5)
    assert second.stats().get("load_failures", 0) == 0
    prefix = second.acquire_prefix([long_key])
    assert prefix.available_tokens == BLOCK_TOKENS
    assert bytes(prefix.pages[0].view) == payload(0)
    prefix.release()


def test_aggregate_worker_rows_share_authoritative_mmap_and_restore(tmp_path, caches):
    root = tmp_path / "ssd"
    backing = mmap.mmap(-1, 2 * 2 * PAGE_BYTES)
    view = memoryview(backing).cast("B", shape=[2, 2, PAGE_BYTES])
    pages = HostKVPool(view, group_idx=3, block_tokens=BLOCK_TOKENS)
    c = caches(root, pages=pages)
    expected = payload(1) + payload(2)
    try:
        bid = commit(pages, key(1), expected)
        assert pages.page_bytes == 2 * PAGE_BYTES
        assert (
            backing[bid * pages.page_bytes : (bid + 1) * pages.page_bytes] == expected
        )
        assert c.submit_store(key(1)) and c.flush(5)
        c.close()
        backing[:] = bytes(len(backing))
        # A new controller uses the same caller-owned storage, with no retained
        # hash metadata. The SSD promotion restores both worker slices in place.
        restored = HostKVPool(view, group_idx=3, block_tokens=BLOCK_TOKENS)
        second = caches(root, pages=restored)
        try:
            assert second.prefetch(key(1)) and second.flush(5)
            prefix = second.acquire_prefix([key(1)])
            assert prefix.available_tokens == BLOCK_TOKENS
            assert bytes(prefix.pages[0].view) == expected
            assert backing[: restored.page_bytes] == expected
            prefix.release()
        finally:
            second.close()
    finally:
        c.close()
        view.release()
        backing.close()


@pytest.mark.parametrize("damage", ["hash", "truncate", "payload", "no_xattr"])
def test_corruption_is_a_miss_and_never_publishes_partial_bytes(
    tmp_path,
    caches,
    damage,
):
    root = tmp_path / "ssd"
    first = caches(root)
    keys = [key(1), key(2)]
    stored(first, keys)
    path = Path(first._tier.file_mapper.get_file_name(first.tier_key(keys[1])))
    head_path = Path(first._tier.file_mapper.get_file_name(first.tier_key(keys[0])))
    first.close()
    if damage == "hash":
        shutil.copy2(head_path, path)  # Includes the wrong full key in its xattr.
    elif damage == "truncate":
        with path.open("r+b") as f:
            f.truncate(PAGE_BYTES - 1)
    elif damage == "payload":
        with path.open("r+b") as f:
            f.write(b"corrupt")
    else:
        os.removexattr(path, "user.revolta.kv_crc32c")
    second = caches(root)
    assert second.prefetch(keys[0]) and second.flush(5)
    head = second.acquire_prefix(keys)
    assert head.available_tokens == BLOCK_TOKENS
    assert second.prefetch(keys[1]) and second.flush(5)
    assert second.pool.lookup(keys[1]) is None
    assert not path.exists()
    after = second.acquire_prefix(keys)
    assert after.available_tokens == BLOCK_TOKENS
    assert second.stats()["load_failures"] == 1
    assert second.stats()["ssd_verified_pages"] == 1
    head.release()
    after.release()


@pytest.mark.parametrize(
    "changed", ["scales", "encoding", "layout", "tp", "dcp", "rank"]
)
def test_incompatible_identity_cannot_reuse_host_pages(tmp_path, caches, changed):
    root = tmp_path / "ssd"
    first = caches(root)
    stored(first, [key(1)])
    original = first._tier.file_mapper.get_file_name(first.tier_key(key(1)))
    first.close()
    layout = LAYOUT
    if changed == "scales":
        layout = replace(layout, kv_scales=((1.0, 2.0), (0.25, 1.0)))
    elif changed == "encoding":
        layout = replace(layout, encoding="other-fixture")
    elif changed == "layout":
        layout = replace(layout, version="cpu-fixture-v2")
    if changed in ("tp", "dcp", "rank"):
        # Build normally then change the input spec before constructing the tier.
        spec, manifest = checkpoint_spec(root)
        config = first._tier._offloading_spec.vllm_config.kv_transfer_config
        spec.vllm_config.kv_transfer_config = config
        setattr(
            spec.vllm_config.parallel_config,
            {
                "tp": "tensor_parallel_size",
                "dcp": "decode_context_parallel_size",
                "rank": "rank",
            }[changed],
            2 if changed != "rank" else 1,
        )
        second = HostKVPrefixCache(pool(), spec, layout)
    else:
        second = caches(root, layout=layout)
    try:
        assert second.enabled
        assert (
            second._tier.file_mapper.get_file_name(second.tier_key(key(1))) != original
        )
        assert not second.prefetch(key(1))
        assert second.acquire_prefix([key(1)]).available_tokens == 0
    finally:
        second.close()


@pytest.mark.parametrize("failure", [errno.ENOSPC, errno.EIO])
def test_store_failure_keeps_ram_and_leaves_no_persisted_hit(
    tmp_path,
    caches,
    monkeypatch,
    failure,
):
    c = caches(tmp_path / "ssd")
    commit(c.pool, key(1))
    path = Path(c._tier.file_mapper.get_file_name(c.tier_key(key(1))))

    def fail(*args, **kwargs):
        raise OSError(failure, "injected write failure")

    monkeypatch.setattr("vllm.v1.kv_offload.tiering.fs.revolta.os.write", fail)
    assert c.submit_store(key(1)) and c.flush(5)
    prefix = c.acquire_prefix([key(1)])
    assert prefix.available_tokens == BLOCK_TOKENS
    assert bytes(prefix.pages[0].view) == payload()
    assert not path.exists() and not list(path.parent.glob("*.tmp"))
    assert c.stats()["store_failures"] == 1
    prefix.release()


def test_queue_cap_counts_active_queued_and_completed_until_polled(
    tmp_path,
    caches,
    monkeypatch,
):
    c = caches(tmp_path / "ssd", pages=pool(slots=5), max_pending_blocks=3)
    entered = threading.Event()
    proceed = threading.Event()
    write = c._tier._write_page

    def blocked(*args):
        entered.set()
        assert proceed.wait(5)
        write(*args)

    monkeypatch.setattr(c._tier, "_write_page", blocked)
    try:
        for n in range(1, 6):
            commit(c.pool, key(n))
        assert c.submit_store(key(1)) and entered.wait(5)
        assert c.submit_store(key(2))
        assert c.submit_store(key(3))  # At least one task must remain queued.
        assert not c.submit_store(key(4))
        assert c.pending_blocks == 3
        assert c.stats()["queue_rejections"] == 1
        # Source pages remain pinned even though the caller ended its request.
        commit(c.pool, key(6))
        commit(c.pool, key(7))
        assert c.pool.is_ready(c.pool.lookup(key(1)))
        assert c.pool.is_ready(c.pool.lookup(key(2)))
        assert c.pool.is_ready(c.pool.lookup(key(3)))
    finally:
        proceed.set()
    deadline = time.monotonic() + 5
    while c._tier._pool.pending_store_blocks and time.monotonic() < deadline:
        time.sleep(0.005)
    # Finished physical jobs still retain their rows until the controller polls.
    assert c._tier._pool.pending_store_blocks == 0 and c.pending_blocks == 3
    assert c.flush(5) and c.pending_blocks == 0


def test_pending_load_holds_destination_and_queue_capacity(
    tmp_path, caches, monkeypatch
):
    root = tmp_path / "ssd"
    first = caches(root)
    stored(first, [key(1), key(2)])
    first.close()
    second = caches(root, pages=pool(slots=1), max_pending_blocks=1)
    entered = threading.Event()
    proceed = threading.Event()
    read = second._tier._read_page

    def blocked(*args):
        entered.set()
        assert proceed.wait(5)
        read(*args)

    monkeypatch.setattr(second._tier, "_read_page", blocked)
    try:
        assert second.prefetch(key(1)) and entered.wait(5)
        assert second.pool.lookup(key(1)) is not None
        assert not second.pool.is_ready(second.pool.lookup(key(1)))
        assert not second.prefetch(key(2))
        assert second.pool.reserve(key(3)) is None
        assert second.acquire_prefix([key(1)]).available_tokens == 0
    finally:
        proceed.set()
    assert second.flush(5)
    prefix = second.acquire_prefix([key(1), key(2)])
    assert prefix.available_tokens == BLOCK_TOKENS
    assert not second.prefetch(key(2))  # Active prefix pins the only host row.
    prefix.release()
    assert second.prefetch(key(2)) and second.flush(5)


def test_close_waits_for_physical_read_before_aborting_destination(
    tmp_path,
    caches,
    monkeypatch,
):
    root = tmp_path / "ssd"
    first = caches(root)
    stored(first, [key(1)])
    first.close()
    second = caches(root, pages=pool(slots=1))
    entered = threading.Event()
    proceed = threading.Event()
    read_done = threading.Event()
    read = second._tier._read_page

    def blocked(*args):
        entered.set()
        assert proceed.wait(5)
        read(*args)
        read_done.set()

    monkeypatch.setattr(second._tier, "_read_page", blocked)
    assert second.prefetch(key(1)) and entered.wait(5)
    assert second.pool.reserve(key(2)) is None
    timer = threading.Timer(0.05, proceed.set)
    timer.start()
    try:
        second.close()
    finally:
        proceed.set()
        timer.join()
    assert read_done.is_set()
    # Discarded reads never publish a hit, even if bytes arrived before join.
    assert second.pool.lookup(key(1)) is None
    assert second.stats()["cancelled_loads"] == 1
    assert second.stats().get("ssd_verified_pages", 0) == 0
    commit(second.pool, key(2))


def test_rate_limiter_one_page_burst_and_idle_does_not_accumulate_credit(monkeypatch):
    clock = [0.0]
    waits = []

    class Timer:
        def wait(self, delay):
            waits.append(delay)
            clock[0] += delay
            return False

    monkeypatch.setattr("vllm.v1.kv_offload.host.ssd.time.monotonic", lambda: clock[0])
    limiter = _WriteRateLimiter(1024)
    limiter.stopped = Timer()
    limiter.acquire(256)
    limiter.acquire(256)
    limiter.acquire(256)
    assert waits == [0.0, 0.25, 0.25]
    clock[0] = 20
    limiter.acquire(256)
    limiter.acquire(256)
    assert waits[-2:] == [0.0, 0.25]


def test_real_async_writers_share_the_configured_rate(tmp_path, caches, monkeypatch):
    c = caches(tmp_path / "ssd", write_bytes_per_second=PAGE_BYTES * 10)
    # Record actual file publication; both worker threads can write stores.
    writes = []
    import vllm.v1.kv_offload.host.ssd as module

    write = module.store_verified_block

    def record(*args, **kwargs):
        writes.append(time.monotonic())
        return write(*args, **kwargs)

    monkeypatch.setattr(module, "store_verified_block", record)
    stored(c, [key(1), key(2), key(3)])
    assert len(writes) == 3
    # Scheduling and fsync may slow writes, but the limiter cannot accelerate them.
    assert writes[-1] - writes[0] >= 0.19


def test_close_cancels_rate_wait_and_releases_rows_after_workers_join(tmp_path, caches):
    c = caches(tmp_path / "ssd", pages=pool(slots=1), write_bytes_per_second=1)
    stored(c, [key(1)])  # Arms a 257-second interval; no actual long sleep.
    assert c.submit_store(key(1))
    assert c.pool.reserve(key(2)) is None
    started = time.monotonic()
    c.close()
    assert time.monotonic() - started < 2
    assert c.pending_blocks == 0
    commit(c.pool, key(2))
    assert not c.submit_store(key(2))
    assert c.acquire_prefix([key(2)]).available_tokens == 0
    assert not c.reader_ready(key(2))


@pytest.mark.parametrize("missing_manifest", [True, False])
def test_unverifiable_content_disables_ssd_without_touching_root(
    tmp_path,
    caches,
    missing_manifest,
):
    root = tmp_path / "ssd"
    spec, manifest = checkpoint_spec(root)
    if not missing_manifest:
        Path(manifest).write_text("broken")
    c = caches(root, content_manifest_path=None if missing_manifest else manifest)
    assert not c.enabled and not root.exists()
    commit(c.pool, key(1))
    assert not c.submit_store(key(1))
    prefix = c.acquire_prefix([key(1)])
    assert prefix.available_tokens == BLOCK_TOKENS
    prefix.release()


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_pending_blocks", 0),
        ("max_pending_blocks", True),
        ("write_bytes_per_second", -1),
        ("write_bytes_per_second", 1.5),
        ("drain_timeout_s", float("nan")),
        ("drain_timeout_s", -1),
        ("max_size_bytes", None),
        ("max_files", 0),
        ("unexpected", 1),
    ],
)
def test_invalid_resource_bounds_are_rejected(tmp_path, option, value):
    with pytest.raises(ValueError):
        cache(tmp_path / "ssd", **{option: value})


def test_fresh_processes_reuse_real_prefix_hashes_and_payloads(tmp_path):
    root = tmp_path / "ssd"
    # No inherited NONE_HASH, Python object, memoryview or pending I/O survives.
    snippet = r"""
import json, sys
from pathlib import Path
from tests.v1.kv_offload.test_host_kv_prefix import cache, commit, payload
from vllm.v1.core import kv_cache_utils as u
from vllm.utils.hashing import sha256
from vllm.v1.kv_offload.base import make_offload_key
c = cache(Path(sys.argv[1]))
hashes = []
parent = None
for start in (0, 16, 32):
    parent = u.hash_block_tokens(sha256, parent, list(range(start, start + 16)))
    hashes.append(parent)
keys = [make_offload_key(h, 3) for h in hashes]
try:
    if sys.argv[2] == "store":
        for i, k in enumerate(keys):
            commit(c.pool, k, payload(i))
            assert c.submit_store(k)
        assert c.flush(5)
        assert not c.stats().get("store_failures", 0)
    else:
        for k in keys:
            assert c.prefetch(k)
        assert c.flush(5)
        prefix = c.acquire_prefix(keys)
        assert prefix.available_tokens == 48
        assert [bytes(p.view) for p in prefix.pages] == [payload(i) for i in range(3)]
        assert c.stats()["ssd_verified_pages"] == 3
        prefix.release()
finally:
    c.close()
print(json.dumps([h.hex() for h in hashes]))
"""
    results = []
    for action in ("store", "load"):
        env = dict(os.environ)
        env.pop("PYTHONHASHSEED", None)
        result = subprocess.run(
            [sys.executable, "-c", snippet, str(root), action],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        results.append(json.loads(result.stdout.splitlines()[-1]))
    assert results[0] == results[1]
