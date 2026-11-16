# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ReVolta file-system secondary tier (``revolta_fs``)."""

import json
import mmap
import os
import shutil
import time
from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_offload.test_fs_tier import (
    _BLOCK_ELEMENTS,
    _DTYPE,
    drain,
    key,
    make_job,
)
from vllm.v1.kv_cache_interface import KVBlockSegments, KVCacheTensor
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.config import fs_tier_configs
from vllm.v1.kv_offload.tiering.fs.content_manifest import build_manifest
from vllm.v1.kv_offload.tiering.fs.revolta import (
    CHECKSUM_XATTR,
    RevoltaFileSystemTierManager,
    read_page_identity,
)


def _view(num_blocks: int = 4):
    backing = mmap.mmap(-1, num_blocks * _BLOCK_ELEMENTS * 4)
    tensor = torch.frombuffer(backing, dtype=_DTYPE).reshape(
        num_blocks, _BLOCK_ELEMENTS
    )
    return tensor, memoryview(tensor.numpy())


def make_spec(model_dir, config_hash="cfg-1"):
    model = SimpleNamespace(
        model=str(model_dir),
        runner_type="generate",
        dtype=torch.float16,
        quantization=None,
        hf_overrides={},
        hf_config={},
        hf_text_config={},
        model_weights=None,
    )
    config = SimpleNamespace(
        model_config=model,
        speculative_config=None,
        lora_config=None,
        load_config=SimpleNamespace(
            load_format="auto",
            model_loader_extra_config={},
            ignore_patterns=["original/**/*"],
        ),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="torch.float32"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            rank=0,
        ),
        compute_hash=lambda: config_hash,
    )
    return SimpleNamespace(
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[], kv_cache_tensors=[]),
        block_size_factor=1,
        persistent_layout=None,
    )


def checkpoint_spec(root):
    # Keep immutable fixtures outside the disk budget/namespace being tested.
    model = root.parent / (root.name + "-checkpoint")
    if not model.exists():
        model.mkdir()
        (model / "config.json").write_text('{"a": 1}')
        (model / "model.safetensors").write_bytes(b"w" * 4096)
    manifest = root.parent / (root.name + "-manifest.json")
    if not manifest.exists():
        manifest.write_text(json.dumps(build_manifest({"target": str(model)})))
    return make_spec(model), str(manifest)


def make_tier(root, view, spec=None, **kwargs):
    if spec is None:
        spec, manifest = checkpoint_spec(root)
        kwargs.setdefault("content_manifest_path", manifest)
    kwargs.setdefault("max_size_bytes", 64 << 20)
    return RevoltaFileSystemTierManager(
        offloading_spec=spec,
        primary_kv_view=view,
        tier_type="revolta_fs",
        root_dir=str(root),
        **kwargs,
    )


@pytest.fixture
def revolta_tier(tmp_path):
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    yield tier, tensor
    tier.shutdown()


def test_registered_and_shares_fs_seed_rules(tmp_path):
    tensor, view = _view()
    spec, manifest = checkpoint_spec(tmp_path)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "revolta_fs",
            "root_dir": str(tmp_path),
            "max_size_bytes": 64 << 20,
            "n_read_threads": 1,
            "n_write_threads": 1,
            "content_manifest_path": manifest,
        },
        view,
        spec,
    )
    try:
        assert isinstance(tier, RevoltaFileSystemTierManager)
        assert tier.tier_type == "revolta_fs"
    finally:
        tier.shutdown()
    config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_extra_config={
            "secondary_tiers": [
                {"type": "revolta_fs", "root_dir": str(tmp_path), "max_size_bytes": 1}
            ]
        },
    )
    assert [t["type"] for t in fs_tier_configs(config)] == ["revolta_fs"]


def test_store_then_load_roundtrip(revolta_tier):
    tier, tensor = revolta_tier
    tensor[0].fill_(1.5)
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert [r.success for r in drain(tier)] == [True]
    assert tier.lookup(key(1)) is True
    tensor[1].zero_()
    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    assert [r.success for r in drain(tier)] == [True]
    assert torch.equal(tensor[1], tensor[0])


# ---------------------------------------------------------------------------
# Integrity (CRC32C attribute, atomic publication)
# ---------------------------------------------------------------------------


def _store(tier, tensor, k, slot=0, value=1.0, job_id=1):
    tensor[slot].fill_(value)
    tier.submit_store(make_job(job_id, [k], [slot]))
    assert [r.success for r in drain(tier)] == [True]
    return tier.file_mapper.get_file_name(k)


def _load(tier, k, slot=1, job_id=2):
    tier.submit_load(make_job(job_id, [k], [slot], is_promotion=True))
    return [r.success for r in drain(tier)]


def test_unified_tensor_schema_reuses_pre_split_ssd_page(tmp_path):
    # The old dataclass had precisely these three fields and this FQN. Use it
    # to write an actual checksummed page, then reopen with today's dataclass.
    legacy = make_dataclass(
        "KVCacheTensor",
        [("size", int), ("shared_by", list), ("packed_members", object)],
    )
    legacy.__module__ = KVCacheTensor.__module__
    tensor, view = _view()
    spec, manifest = checkpoint_spec(tmp_path)
    spec.kv_cache_config.block_segments = None
    spec.kv_cache_config.kv_cache_tensors = [legacy(4096, ["layer"], None)]
    writer = make_tier(tmp_path, view, spec=spec, content_manifest_path=manifest)
    try:
        assert not writer.disabled
        old_layout = writer._extra_layout.copy()
        old_path = _store(writer, tensor, key(1), value=7.0)
    finally:
        writer.shutdown()
    spec.kv_cache_config.kv_cache_tensors = [KVCacheTensor(4096, ["layer"])]
    reader = make_tier(tmp_path, view, spec=spec, content_manifest_path=manifest)
    try:
        assert not reader.disabled
        assert reader._extra_layout == old_layout
        assert reader.file_mapper.get_file_name(key(1)) == old_path
        tensor[1].zero_()
        assert reader.lookup(key(1))
        assert _load(reader, key(1)) == [True]
        assert torch.equal(tensor[1], torch.full_like(tensor[1], 7.0))
    finally:
        reader.shutdown()


def test_split_persistence_binds_tensor_counts_and_segments(tmp_path):
    tensor, view = _view()
    spec, manifest = checkpoint_spec(tmp_path)
    layouts = []
    for blocks, primary in ((2, 1), (3, 1), (2, 0)):
        spec.kv_cache_config.block_segments = KVBlockSegments(
            ("state", "history"), (4, 4), (), primary
        )
        spec.kv_cache_config.kv_cache_tensors = [
            KVCacheTensor(4096, ["layer"], num_blocks=blocks)
        ]
        tier = make_tier(tmp_path, view, spec=spec, content_manifest_path=manifest)
        try:
            assert not tier.disabled
            layouts.append(tier._extra_layout["page_layout"])
        finally:
            tier.shutdown()
    assert len(set(layouts)) == 3


def test_unified_explicit_tensor_count_changes_identity(tmp_path):
    tensor, view = _view()
    spec, manifest = checkpoint_spec(tmp_path)
    layouts = []
    for blocks in (None, 2, 3):
        spec.kv_cache_config.kv_cache_tensors = [
            KVCacheTensor(4096, ["layer"], num_blocks=blocks)
        ]
        tier = make_tier(tmp_path, view, spec=spec, content_manifest_path=manifest)
        try:
            assert not tier.disabled
            layouts.append(tier._extra_layout["page_layout"])
        finally:
            tier.shutdown()
    assert len(set(layouts)) == 3


def test_page_has_checksum_and_no_temp_files(revolta_tier):
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1))
    assert read_page_identity(path) is not None
    leftovers = [
        f
        for _, _, files in os.walk(os.path.dirname(path))
        for f in files
        if f.endswith(".tmp")
    ]
    assert leftovers == []
    assert tier.get_revolta_stats()["pages_written"] == 1


def test_equal_length_corruption_is_a_miss(revolta_tier):
    """Counterexample 3: an equal-length page with flipped bytes must not load."""
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1), value=2.0)
    with open(path, "r+b") as f:
        f.seek(4096)
        byte = f.read(1)
        f.seek(4096)
        f.write(bytes([byte[0] ^ 0xFF]))
    assert os.path.getsize(path) == _BLOCK_ELEMENTS * 4
    assert _load(tier, key(1)) == [False]
    assert not os.path.exists(path)
    assert tier.lookup(key(1)) is False
    assert tier.get_revolta_stats()["checksum_failures"] == 1


def test_page_without_checksum_is_rejected_and_replaced(revolta_tier):
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1), value=3.0)
    os.removexattr(path, CHECKSUM_XATTR)
    assert _load(tier, key(1)) == [False]
    assert not os.path.exists(path)
    # A bare file (e.g. written by plain fs) is replaced, not trusted.
    with open(path, "wb") as f:
        f.write(b"\0" * (_BLOCK_ELEMENTS * 4))
    _store(tier, tensor, key(1), value=3.0, job_id=3)
    assert read_page_identity(path) is not None
    assert _load(tier, key(1), job_id=4) == [True]
    assert torch.equal(tensor[1], tensor[0])


def test_intact_page_of_another_key_is_rejected(revolta_tier):
    """A valid page copied to another key's path fails the tag check."""
    tier, tensor = revolta_tier
    src = _store(tier, tensor, key(1), value=4.0)
    dst = tier.file_mapper.get_file_name(key(2))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    os.setxattr(dst, CHECKSUM_XATTR, os.getxattr(src, CHECKSUM_XATTR))
    tier._cache.lookup(dst)
    assert _load(tier, key(2)) == [False]
    assert _load(tier, key(1), job_id=3) == [True]


def test_wrong_size_page_is_rejected(revolta_tier):
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1))
    with open(path, "ab") as f:
        f.write(b"\0" * 4096)
    assert _load(tier, key(1)) == [False]
    assert not os.path.exists(path)


def test_missing_xattr_support_disables_tier(tmp_path, monkeypatch):
    tensor, view = _view()

    def no_xattr(*args, **kwargs):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(os, "setxattr", no_xattr)
    tier = make_tier(tmp_path, view)
    assert tier.disabled
    assert "extended attributes" in tier.disabled_reason
    tier.shutdown()


# ---------------------------------------------------------------------------
# Identity (content manifest, fail closed)
# ---------------------------------------------------------------------------


def _checkpoint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"a": 1}')
    (model / "model-00001.safetensors").write_bytes(b"w" * 4096)
    return model


def _write_manifest(tmp_path, roles):
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(build_manifest(roles)))
    return str(path)


def _identity_tier(tmp_path, model, manifest, **kwargs):
    tensor, view = _view()
    tier = make_tier(
        tmp_path / "kv",
        view,
        spec=make_spec(model),
        content_manifest_path=manifest,
        **kwargs,
    )
    return tier, tensor


def test_manifest_enables_and_binds_namespace(tmp_path):
    model = _checkpoint(tmp_path)
    manifest = _write_manifest(tmp_path, {"target": str(model)})
    tier, tensor = _identity_tier(tmp_path, model, manifest)
    try:
        assert not tier.disabled
        layout = tier.file_mapper.get_run_config()["persistent_layout"]
        assert len(layout["content_digest"]) == 64
        _store(tier, tensor, key(1))
        assert _load(tier, key(1)) == [True]
    finally:
        tier.shutdown()


def test_same_path_weight_swap_disables_tier(tmp_path):
    """Counterexample 1: same path, same index and config, new shard bytes."""
    model = _checkpoint(tmp_path)
    manifest = _write_manifest(tmp_path, {"target": str(model)})
    tier, tensor = _identity_tier(tmp_path, model, manifest)
    _store(tier, tensor, key(1))
    tier.shutdown()
    shard = model / "model-00001.safetensors"
    st = shard.stat()
    shard.write_bytes(b"x" * 4096)  # same name and size, new contents
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    tier, _ = _identity_tier(tmp_path, model, manifest)
    assert tier.disabled
    assert tier.lookup(key(1)) is False
    tier.submit_store(make_job(5, [key(2)], [0]))
    assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(5, True)]
    tier.shutdown()


def test_full_verification_catches_restored_mtime(tmp_path):
    """stat checks miss a rewrite that restores size, mtime and inode;
    content_verify=full recomputes the hashes."""
    model = _checkpoint(tmp_path)
    manifest = _write_manifest(tmp_path, {"target": str(model)})
    shard = model / "model-00001.safetensors"
    st = shard.stat()
    with open(shard, "r+b") as f:
        f.write(b"y")
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns))
    tier, _ = _identity_tier(tmp_path, model, manifest)
    assert not tier.disabled
    tier.shutdown()
    tier, _ = _identity_tier(tmp_path, model, manifest, content_verify="full")
    assert tier.disabled
    tier.shutdown()


def test_unlisted_file_or_missing_manifest_disables(tmp_path):
    model = _checkpoint(tmp_path)
    manifest = _write_manifest(tmp_path, {"target": str(model)})
    (model / "model-00002.safetensors").write_bytes(b"z")
    tier, _ = _identity_tier(tmp_path, model, manifest)
    assert tier.disabled
    tier.shutdown()
    tier, _ = _identity_tier(tmp_path, model, None)
    assert tier.disabled
    tier.shutdown()
    assert not (tmp_path / "kv").exists()


def test_namespace_config_conflict_disables(tmp_path):
    model = _checkpoint(tmp_path)
    manifest = _write_manifest(tmp_path, {"target": str(model)})
    tier, _ = _identity_tier(tmp_path, model, manifest)
    config_path = tier.file_mapper.get_config_file_path()
    tier.shutdown()
    with open(config_path, "w") as f:
        f.write('{"someone": "else"}')
    tier, _ = _identity_tier(tmp_path, model, manifest)
    assert tier.disabled
    tier.shutdown()


def test_manifest_tool_writes_verifiable_manifest(tmp_path):
    import importlib.util
    import json

    from vllm.v1.kv_offload.tiering.fs.content_manifest import (
        load_manifest,
        verify_manifest,
    )

    tool = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "tools",
        "kv_offload",
        "make_content_manifest.py",
    )
    spec = importlib.util.spec_from_file_location("make_content_manifest", tool)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = _checkpoint(tmp_path)
    out = tmp_path / "out" / "m.json"
    out.parent.mkdir()
    assert module.main(["--role", f"target={model}", "--out", str(out)]) == 0
    manifest = load_manifest(str(out))
    assert verify_manifest(manifest, {"target": str(model)}) == manifest["digest"]
    assert json.loads(out.read_text())["roles"]["target"]["files"].keys() == {
        "config.json",
        "model-00001.safetensors",
    }
    with pytest.raises(SystemExit):
        module.main(["--role", f"target={model}", "--out", str(model / "m.json")])


# ---------------------------------------------------------------------------
# Recency and chain integrity
# ---------------------------------------------------------------------------

_PAGE = _BLOCK_ELEMENTS * 4


def test_touch_keeps_chain_head_most_recent(tmp_path):
    """D1: a hot head used only through the CPU tier must not be evicted
    before the chain tail."""
    tensor, view = _view()
    tier = make_tier(
        tmp_path, view, max_size_bytes=3 * _PAGE, mtime_update_interval_s=0
    )
    try:
        chain = [key(1), key(2), key(3)]
        tensor[:3].fill_(1.0)
        tier.submit_store(make_job(1, chain, [0, 1, 2]))
        assert all(r.success for r in drain(tier))
        tier.touch(chain)
        tier.flush_mtime_updates()
        mtimes = [os.stat(tier.file_mapper.get_file_name(k)).st_mtime_ns for k in chain]
        assert mtimes[0] >= mtimes[1] >= mtimes[2]
        tier.submit_store(make_job(2, [key(4)], [3]))
        assert all(r.success for r in drain(tier))
        assert tier.lookup(key(1)) is True  # head kept
        assert tier.lookup(key(3)) is False  # tail evicted
    finally:
        tier.shutdown()


def test_restart_order_follows_touch(tmp_path):
    tensor, view = _view()
    tier = make_tier(
        tmp_path, view, max_size_bytes=3 * _PAGE, mtime_update_interval_s=0
    )
    chain = [key(1), key(2), key(3)]
    tensor[:3].fill_(1.0)
    tier.submit_store(make_job(1, chain, [0, 1, 2]))
    assert all(r.success for r in drain(tier))
    time.sleep(0.02)
    tier.touch(chain)
    tier.flush_mtime_updates()
    tier.shutdown()
    tier = make_tier(tmp_path, view, max_size_bytes=3 * _PAGE)
    try:
        tier.submit_store(make_job(2, [key(4)], [3]))
        assert all(r.success for r in drain(tier))
        assert tier.lookup(key(1)) is True
        assert tier.lookup(key(3)) is False
    finally:
        tier.shutdown()


def test_mtime_updates_are_throttled(tmp_path):
    tensor, view = _view()
    tier = make_tier(tmp_path, view, mtime_update_interval_s=3600)
    try:
        path = _store(tier, tensor, key(1))
        tier.touch([key(1)])
        tier.flush_mtime_updates()
        first = os.stat(path).st_mtime_ns
        time.sleep(0.02)
        tier.touch([key(1)])
        tier.flush_mtime_updates()
        assert os.stat(path).st_mtime_ns == first
        assert tier.get_revolta_stats()["mtime_updates"] == 1
    finally:
        tier.shutdown()


def test_chain_stops_at_first_failed_page_per_group(tmp_path, monkeypatch):
    from vllm.v1.kv_offload.base import make_offload_key
    from vllm.v1.kv_offload.tiering.fs import revolta

    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    real = revolta.store_verified_block
    bad = tier.file_mapper.get_file_name(make_offload_key(b"\x00" * 7 + b"\x02", 0))

    def flaky(path, *args, **kwargs):
        if path == bad:
            raise OSError(28, "injected ENOSPC")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(revolta, "store_verified_block", flaky)
    try:
        g0 = [make_offload_key(i.to_bytes(8, "big"), 0) for i in (1, 2, 3)]
        g1 = [make_offload_key(i.to_bytes(8, "big"), 1) for i in (1, 2, 3)]
        tier.submit_store(make_job(1, g0 + g1, [0, 1, 2, 0, 1, 2]))
        assert [r.success for r in drain(tier)] == [False]
        assert [tier.lookup(k) for k in g0] == [True, False, False]
        assert [tier.lookup(k) for k in g1] == [True, True, True]
        assert tier.get_revolta_stats()["skipped_after_chain_failure"] == 1
    finally:
        tier.shutdown()


# ---------------------------------------------------------------------------
# I/O robustness: deadlines, retries, backpressure, shutdown drain
# ---------------------------------------------------------------------------

import errno  # noqa: E402
import threading  # noqa: E402

from vllm.v1.kv_offload.tiering.fs import revolta as revolta_mod  # noqa: E402


def test_hung_write_after_snapshot_expires_once(tmp_path, monkeypatch):
    release = threading.Event()
    real_fsync_dir = revolta_mod._fsync_dir

    def hang(path):
        release.wait(5)
        real_fsync_dir(path)

    monkeypatch.setattr(revolta_mod, "_fsync_dir", hang)
    tensor, view = _view()
    tier = make_tier(tmp_path, view, store_timeout_s=0.1, n_write_threads=1)
    try:
        tensor[0].fill_(1.0)
        tier.submit_store(make_job(1, [key(1)], [0]))
        time.sleep(0.3)
        assert [(r.job_id, r.success) for r in drain(tier)] == [(1, False)]
        release.set()
        assert drain(tier) == []  # the late completion is not reported again
        assert tier._pool.num_timed_out == 1
        # The late write completed a page whose checksum was taken before
        # the deadline: it is still a valid page of key 1.
        assert _load(tier, key(1), job_id=2) == [True]
    finally:
        release.set()
        tier.shutdown()


def test_job_does_not_expire_while_taking_its_snapshot(tmp_path, monkeypatch):
    real_crc = revolta_mod.crc32c

    def slow_crc(view, crc=0):
        if len(view) > 4096:  # the page payload, not the key tag
            time.sleep(0.3)
        return real_crc(view, crc)

    tensor, view = _view()
    tier = make_tier(tmp_path, view, store_timeout_s=0.05, n_write_threads=1)
    monkeypatch.setattr(revolta_mod, "crc32c", slow_crc)
    try:
        tensor[0].fill_(1.0)
        tier.submit_store(make_job(1, [key(1)], [0]))
        time.sleep(0.5)
        assert [r.success for r in drain(tier, max_rounds=100)] == [True]
    finally:
        tier.shutdown()


def test_queued_tasks_of_expired_job_never_run(tmp_path, monkeypatch):
    release = threading.Event()
    real_fsync_dir = revolta_mod._fsync_dir

    def hang(path):
        release.wait(5)
        real_fsync_dir(path)

    monkeypatch.setattr(revolta_mod, "_fsync_dir", hang)
    tensor, view = _view()
    tier = make_tier(
        tmp_path, view, store_timeout_s=0.1, n_write_threads=1, n_read_threads=0
    )
    try:
        tensor[:2].fill_(1.0)
        tier.submit_store(make_job(1, [key(1)], [0]))
        tier.submit_store(make_job(2, [key(2)], [1]))  # queued behind job 1
        time.sleep(0.3)
        results = drain(tier)
        assert sorted((r.job_id, r.success) for r in results) == [
            (1, False),
            (2, False),
        ]
        release.set()
        time.sleep(0.2)
        assert not os.path.exists(tier.file_mapper.get_file_name(key(2)))
    finally:
        release.set()
        tier.shutdown()


def test_transient_errors_are_retried(tmp_path, monkeypatch):
    real = revolta_mod.store_verified_block
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EAGAIN, "try again")
        return real(*args, **kwargs)

    monkeypatch.setattr(revolta_mod, "store_verified_block", flaky)
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    try:
        _store(tier, tensor, key(1))
        assert tier.get_revolta_stats()["io_retries"] == 1
    finally:
        tier.shutdown()


def test_backpressure_skips_store_jobs(tmp_path, monkeypatch):
    release = threading.Event()
    real_fsync_dir = revolta_mod._fsync_dir

    def hang(path):
        release.wait(5)
        real_fsync_dir(path)

    monkeypatch.setattr(revolta_mod, "_fsync_dir", hang)
    tensor, view = _view()
    tier = make_tier(tmp_path, view, max_inflight_store_blocks=1, store_timeout_s=0)
    try:
        tensor[:2].fill_(1.0)
        tier.submit_store(make_job(1, [key(1)], [0]))
        tier.submit_store(make_job(2, [key(2)], [1]))
        first = list(tier.get_finished_jobs())
        assert [(r.job_id, r.success) for r in first] == [(2, True)]
        stats = tier.get_revolta_stats()
        assert stats["store_jobs_skipped_backpressure"] == 1
        release.set()
        assert [(r.job_id, r.success) for r in drain(tier)] == [(1, True)]
        assert tier._inflight_stores == 0
        assert tier.lookup(key(2)) is False
    finally:
        release.set()
        tier.shutdown()


def test_shutdown_drains_queued_stores(tmp_path, monkeypatch):
    real = revolta_mod.store_verified_block

    def slow(*args, **kwargs):
        time.sleep(0.05)
        return real(*args, **kwargs)

    monkeypatch.setattr(revolta_mod, "store_verified_block", slow)
    tensor, view = _view()
    tier = make_tier(
        tmp_path, view, n_write_threads=1, n_read_threads=0, drain_timeout_s=5
    )
    keys = [key(i) for i in range(1, 5)]
    tensor[:4].fill_(1.0)
    tier.submit_store(make_job(1, keys, [0, 1, 2, 3]))
    tier.shutdown()
    assert all(os.path.exists(tier.file_mapper.get_file_name(k)) for k in keys)
