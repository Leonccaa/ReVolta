# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for persistent identity, startup and concurrent chain I/O."""

import copy
import dataclasses
import errno
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_fs_tier import drain, key, make_job
from tests.v1.kv_offload.test_revolta_fs import (
    _PAGE,
    _checkpoint,
    _load,
    _store,
    _view,
    _write_manifest,
    checkpoint_spec,
    make_spec,
    make_tier,
)
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.tiering.fs import content_manifest, revolta
from vllm.v1.kv_offload.tiering.fs.content_manifest import (
    ManifestError,
    build_manifest,
    load_manifest,
    manifest_digest,
)
from vllm.v1.kv_offload.tiering.fs.identity import get_revolta_identity


@pytest.fixture
def revolta_tier(tmp_path):
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    try:
        yield tier, tensor
    finally:
        tier.shutdown()


def _seed_config(root, manifest):
    return SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_extra_config={
            "secondary_tiers": [
                {
                    "type": "revolta_fs",
                    "root_dir": str(root),
                    "max_size_bytes": 64 << 20,
                    "content_manifest_path": manifest,
                }
            ]
        },
    )


def _draft(model):
    config = make_spec(model).vllm_config.model_config
    config.runner_type = "draft"
    return SimpleNamespace(
        method="dflash",
        model=str(model),
        num_speculative_tokens=4,
        quantization=None,
        hf_overrides={},
        draft_model_config=config,
        draft_load_config=None,
    )


def test_crc_collision_still_rejects_full_key(revolta_tier):
    """These full keys collide under the former identity+key CRC32C tag."""
    tier, tensor = revolta_tier
    first = OffloadKey(
        bytes.fromhex(
            "8aaca1de2926b7646606f8b42f4c12d0ceb57fc81992dd7a80f16dcafb20cf8d00000000"
        )
    )
    second = OffloadKey(
        bytes.fromhex(
            "ed6d9182a2d8b4de405f359a00c59928d6e55903495d730576790eef8982283d00000000"
        )
    )
    assert first != second
    assert revolta.crc32c(tier._identity_digest + first) == revolta.crc32c(
        tier._identity_digest + second
    )
    src = _store(tier, tensor, first, value=7)
    dst = tier.file_mapper.get_file_name(second)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    assert _load(tier, second) == [False]
    assert _load(tier, first, job_id=3) == [True]


@pytest.mark.parametrize("changed", ["version", "namespace", "key", "key_length"])
def test_full_xattr_fields_are_checked(revolta_tier, changed):
    tier, tensor = revolta_tier
    path = _store(tier, tensor, key(1))
    raw = bytearray(os.getxattr(path, revolta.CHECKSUM_XATTR))
    offsets = {
        "version": 0,
        "namespace": 8,
        "key_length": 40,
        "key": revolta._XATTR_FORMAT.size,
    }
    raw[offsets[changed]] ^= 1
    os.setxattr(path, revolta.CHECKSUM_XATTR, raw)
    assert _load(tier, key(1)) == [False]


def test_another_namespace_page_is_rejected(tmp_path):
    tensor, view = _view()
    spec, manifest = checkpoint_spec(tmp_path / "kv")
    first = make_tier(tmp_path / "kv", view, spec=spec, content_manifest_path=manifest)
    src = _store(first, tensor, key(1))
    first.shutdown()
    spec.vllm_config.model_config.quantization = "awq"
    second = make_tier(tmp_path / "kv", view, spec=spec, content_manifest_path=manifest)
    try:
        dst = second.file_mapper.get_file_name(key(1))
        assert src != dst
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        assert _load(second, key(1)) == [False]
    finally:
        second.shutdown()


@pytest.mark.parametrize(
    "change",
    [
        "target_quantization",
        "target_dtype",
        "target_overrides",
        "target_rope",
        "draft_method",
        "draft_model",
        "draft_tokens",
        "draft_quantization",
        "effective_draft_quantization",
        "effective_draft_dtype",
        "draft_overrides",
        "draft_rope",
        "load_format",
        "draft_load_format",
        "kv_dtype",
        "hash_algo",
    ],
)
def test_dedicated_identity_covers_effective_configuration(tmp_path, change):
    model = _checkpoint(tmp_path)
    draft_dir = tmp_path / "draft"
    shutil.copytree(model, draft_dir)
    manifest = _write_manifest(
        tmp_path, {"target": str(model), "draft": str(draft_dir)}
    )
    config = make_spec(model).vllm_config
    config.speculative_config = _draft(draft_dir)
    spec = config.speculative_config
    before = get_revolta_identity(config, str(tmp_path / "kv"), manifest)
    assert before is not None and len(before.persistent_digest) == 64
    if change == "target_quantization":
        config.model_config.quantization = "awq"
    elif change == "target_dtype":
        config.model_config.dtype = "float32"
    elif change == "target_overrides":
        config.model_config.hf_overrides = {"rope_theta": 20000}
    elif change == "target_rope":
        config.model_config.hf_config = {"rope_scaling": {"factor": 2}}
    elif change == "draft_method":
        spec.method = "eagle3"
    elif change == "draft_model":
        spec.model = str(model)
    elif change == "draft_tokens":
        spec.num_speculative_tokens = 8
    elif change == "draft_quantization":
        spec.quantization = "fp8"
    elif change == "effective_draft_quantization":
        spec.draft_model_config.quantization = "fp8"
    elif change == "effective_draft_dtype":
        spec.draft_model_config.dtype = "float32"
    elif change == "draft_overrides":
        spec.draft_model_config.hf_overrides = {"rope_theta": 30000}
    elif change == "draft_rope":
        spec.draft_model_config.hf_text_config = {
            "rope_parameters": {"rope_theta": 12345}
        }
    elif change == "load_format":
        config.load_config.load_format = "safetensors"
    elif change == "draft_load_format":
        spec.draft_load_config = copy.copy(config.load_config)
        spec.draft_load_config.load_format = "safetensors"
    elif change == "kv_dtype":
        config.cache_config.cache_dtype = "float16"
    elif change == "hash_algo":
        config.cache_config.prefix_caching_hash_algo = "sha256_cbor"
    after = get_revolta_identity(config, str(tmp_path / "kv"), manifest)
    assert after is not None
    assert before.persistent_digest != after.persistent_digest
    assert config.compute_hash() == "cfg-1"  # compile identity deliberately unchanged


def test_physical_layout_changes_namespace(tmp_path):
    tensor, view = _view()
    spec, manifest = checkpoint_spec(tmp_path / "kv")
    first = make_tier(tmp_path / "kv", view, spec=spec, content_manifest_path=manifest)
    first_digest = first._identity_digest
    first.shutdown()
    spec.persistent_layout = {"storage": {"strides": [64, 1], "role": "recurrent"}}
    second = make_tier(tmp_path / "kv", view, spec=spec, content_manifest_path=manifest)
    try:
        assert second._identity_digest != first_digest
    finally:
        second.shutdown()


@pytest.mark.parametrize(
    "reason",
    [
        "no_manifest",
        "remote_target",
        "remote_draft",
        "unlisted_draft",
        "model_weights",
        "secondary_weights",
        "loader_extras",
        "dummy",
        "pt",
        "sharded_state",
        "draft_loader",
        "lora",
        "kv_calibration",
    ],
)
def test_disabled_seed_path_never_touches_root(tmp_path, reason, monkeypatch):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    config = spec.vllm_config
    if reason == "no_manifest":
        manifest = None
    elif reason == "remote_target":
        config.model_config.model = "example/remote-target"
    elif reason == "remote_draft":
        config.speculative_config = _draft("example/remote-draft")
    elif reason == "unlisted_draft":
        draft = tmp_path / "unlisted-draft"
        shutil.copytree(config.model_config.model, draft)
        config.speculative_config = _draft(draft)
    elif reason in ("model_weights", "secondary_weights"):
        setattr(config.model_config, reason, "another/source")
    elif reason == "loader_extras":
        config.load_config.model_loader_extra_config = {
            "secondary_weights": ["another/source"]
        }
    elif reason in ("dummy", "pt", "sharded_state"):
        config.load_config.load_format = reason
    elif reason == "draft_loader":
        config.speculative_config = _draft(config.model_config.model)
        config.speculative_config.draft_load_config = SimpleNamespace(
            load_format="dummy"
        )
    elif reason == "lora":
        config.lora_config = {}
    elif reason == "kv_calibration":
        config.cache_config.calculate_kv_scales = True
    tensor, view = _view()
    seed_config = _seed_config(root, manifest)
    # Exercise EngineCore's entry before AND after tier construction.
    real_makedirs = os.makedirs

    def forbid_root(path, *args, **kwargs):
        assert not str(path).startswith(str(root)), "disabled seed attempted root I/O"
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(os, "makedirs", forbid_root)
    kv_cache_utils.init_none_hash(
        sha256, kv_transfer_config=seed_config, vllm_config=config
    )
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    try:
        assert tier.disabled
        kv_cache_utils.init_none_hash(
            sha256, kv_transfer_config=seed_config, vllm_config=config
        )
        assert not root.exists()
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(1, True)]
        assert list(tier.get_finished_jobs()) == []
        assert tier.lookup(key(1)) is False
    finally:
        tier.shutdown()


def test_constructor_disabled_root_stays_excluded_from_seed(tmp_path, monkeypatch):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    tensor, view = _view()

    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "no user xattrs")

    monkeypatch.setattr(os, "setxattr", unsupported)
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    assert tier.disabled
    kv_cache_utils.init_none_hash(
        sha256,
        kv_transfer_config=_seed_config(root, manifest),
        vllm_config=spec.vllm_config,
    )
    assert not (root / ".hash_seed.lock").exists()
    assert not (root / ".hash_seed.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "role",
        "root",
        "files",
        "record",
        "size",
        "ino",
        "mtime_ns",
        "hash",
        "algorithm",
        "version",
        "digest",
        "relative_path",
        "bool_size",
        "bool_version",
    ],
)
def test_manifest_schema_damage_disables_without_root_io(tmp_path, damage):
    model = _checkpoint(tmp_path)
    manifest = build_manifest({"target": str(model)})
    entry = manifest["roles"]["target"]
    record = entry["files"]["config.json"]
    if damage == "role":
        manifest["roles"]["target"] = []
    elif damage == "root":
        del entry["root"]
    elif damage == "files":
        entry["files"] = []
    elif damage == "record":
        entry["files"]["config.json"] = "bad"
    elif damage in ("size", "ino", "mtime_ns", "hash"):
        del record[damage]
    elif damage == "relative_path":
        entry["files"]["../escape"] = entry["files"].pop("config.json")
        manifest["digest"] = manifest_digest(manifest["roles"])
    elif damage == "bool_size":
        record["size"] = True
        manifest["digest"] = manifest_digest(manifest["roles"])
    elif damage == "bool_version":
        manifest["version"] = True
    else:
        manifest[damage] = ["bad"]
    path = tmp_path / "bad-manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ManifestError):
        load_manifest(str(path))
    tensor, view = _view()
    tier = make_tier(
        tmp_path / "kv", view, spec=make_spec(model), content_manifest_path=str(path)
    )
    assert tier.disabled
    assert not (tmp_path / "kv").exists()


def test_full_rehash_io_failure_disables_tier_and_seed(tmp_path, monkeypatch):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    tensor, view = _view()

    def io_failure(*args, **kwargs):
        raise OSError(errno.EIO, "rehash read failure")

    monkeypatch.setattr(content_manifest, "_hash_file", io_failure)
    seed = _seed_config(root, manifest)
    seed.kv_connector_extra_config["secondary_tiers"][0]["content_verify"] = "full"
    kv_cache_utils.init_none_hash(
        sha256, kv_transfer_config=seed, vllm_config=spec.vllm_config
    )
    tier = make_tier(
        root, view, spec=spec, content_manifest_path=manifest, content_verify="full"
    )
    assert tier.disabled
    assert not root.exists()


@pytest.mark.parametrize("threads", [None, (0, 2)])
@pytest.mark.parametrize("failure_errno", [errno.EIO, errno.ENOSPC])
def test_group_failure_cannot_publish_later_page(
    tmp_path, monkeypatch, threads, failure_errno
):
    tensor, view = _view()
    options = (
        {}
        if threads is None
        else dict(zip(("n_read_threads", "n_write_threads"), threads))
    )
    tier = make_tier(tmp_path, view, **options)
    entered = threading.Event()
    release = threading.Event()
    tail_started = threading.Event()
    other_group_started = threading.Event()
    other_group_release = threading.Event()
    failed_group_done = threading.Event()
    original = tier._write_page
    original_group = tier._store_group
    other = make_offload_key(b"other-group", 1)

    def controlled(k, path, bid):
        if k == key(1):
            entered.set()
            assert release.wait(5)
            raise OSError(failure_errno, "head rejected")
        if k == key(2):
            tail_started.set()
        if k == other:
            other_group_started.set()
            assert other_group_release.wait(5)
        original(k, path, bid)

    def group(pages):
        try:
            original_group(pages)
        finally:
            if pages[0][0] == key(1):
                failed_group_done.set()

    monkeypatch.setattr(tier, "_write_page", controlled)
    monkeypatch.setattr(tier, "_store_group", group)
    try:
        tensor.fill_(1)
        tier.submit_store(make_job(1, [key(1), key(2), other], [0, 1, 2]))
        assert entered.wait(5) and other_group_started.wait(5)
        assert not tail_started.is_set()
        release.set()
        # A failed group's task must not complete the job while another is active.
        assert list(tier.get_finished_jobs()) == []
        other_group_release.set()
        # M2 retries transient EIO before ending this group. Do not interpret
        # drain's short idle interval as a completion while retries are active.
        assert failed_group_done.wait(5)
        assert [r.success for r in drain(tier)] == [False]
        assert not tail_started.is_set()
        assert [tier.lookup(k) for k in (key(1), key(2), other)] == [False, False, True]
    finally:
        release.set()
        other_group_release.set()
        tier.shutdown()


def test_scheduler_touch_and_pool_load_serialize_mtime_pruning(tmp_path, monkeypatch):
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    pruning = threading.Event()
    release = threading.Event()
    load_scheduling = threading.Event()
    errors = []
    real_schedule = tier._schedule_mtime

    class GatedDict(dict):
        def items(self):
            iterator = iter(super().items())
            yield next(iterator)
            pruning.set()
            assert release.wait(5)
            yield from iterator

    def schedule(paths):
        if threading.current_thread().name.startswith("vllm_kv_py_fs"):
            load_scheduling.set()
        real_schedule(paths)

    def touch():
        try:
            tier.touch([key(1)])
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=touch, name="scheduler-touch")
    try:
        _store(tier, tensor, key(1))
        _store(tier, tensor, key(2), job_id=3)
        # Force pruning without allocating millions of disk pages.
        tier._mtime_at = GatedDict({f"old-{i}": 0.0 for i in range(5)})
        tier._cache.max_files = 1
        monkeypatch.setattr(tier, "_schedule_mtime", schedule)
        thread.start()
        assert pruning.wait(5)
        tier.submit_load(make_job(4, [key(2)], [1], is_promotion=True))
        assert load_scheduling.wait(5)
        assert thread.is_alive()
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert errors == []
        assert [r.success for r in drain(tier)] == [True]
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(5)
        tier.shutdown()


def test_just_read_head_then_extended_chain_preserves_restart_order(tmp_path):
    tensor, view = _view()
    options = dict(max_size_bytes=2 * _PAGE, mtime_update_interval_s=3600)
    tier = make_tier(tmp_path, view, **options)
    try:
        tensor.fill_(1)
        tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
        assert [r.success for r in drain(tier)] == [True]
        assert _load(tier, key(1)) == [True]
        tier.flush_mtime_updates()
        tier.touch([key(1), key(2)])
        tier.flush_mtime_updates()
        assert (
            os.stat(tier.file_mapper.get_file_name(key(1))).st_mtime_ns
            > os.stat(tier.file_mapper.get_file_name(key(2))).st_mtime_ns
        )
    finally:
        tier.shutdown()
    tier = make_tier(tmp_path, view, **options)
    try:
        tier.submit_store(make_job(3, [key(3)], [2]))
        assert [r.success for r in drain(tier)] == [True]
        assert [tier.lookup(k) for k in (key(1), key(2))] == [True, False]
    finally:
        tier.shutdown()


def test_flush_waits_for_inflight_mtime(tmp_path, monkeypatch):
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    entered, release, flushed = (threading.Event() for _ in range(3))
    real_utime = os.utime

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        real_utime(*args, **kwargs)

    def flush():
        tier.flush_mtime_updates()
        flushed.set()

    thread = threading.Thread(target=flush)
    try:
        _store(tier, tensor, key(1))
        monkeypatch.setattr(os, "utime", blocked)
        tier.touch([key(1)])
        assert entered.wait(5)
        thread.start()
        assert not flushed.wait(0.05)
        release.set()
        assert flushed.wait(5)
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(5)
        tier.shutdown()


def test_disabled_completion_releases_cpu_pins_once(tmp_path):
    from tests.v1.kv_offload.test_tiering_offloading import _mock_mmap_region
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor, view = _view()
    spec, _ = checkpoint_spec(tmp_path / "kv")
    tier = make_tier(tmp_path / "kv", view, spec=spec)
    primary = CPUPrimaryTierOffloadingManager(2, _mock_mmap_region(2))
    manager = TieringOffloadingManager(primary, [tier])
    ctx = ReqContext(req_id="disabled-test")
    try:
        manager.prepare_store([key(1)], ctx)
        manager.complete_store([key(1)], ctx, success=True)
        assert primary._policy.get(key(1)).ref_cnt == 1
        manager._process_finished_jobs()
        manager._process_finished_jobs()
        assert primary._policy.get(key(1)).ref_cnt == 0
        assert primary.lookup(key(1), ctx) is True
        assert not manager._transfer_jobs
    finally:
        manager.shutdown()


def test_revolta_seed_and_page_survive_fresh_processes(tmp_path):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    snippet = """
import json, sys
from pathlib import Path
from tests.v1.kv_offload.test_revolta_fs import (
    make_spec, make_tier, _view, _store, _load,
)
from tests.v1.kv_offload.test_revolta_fs_safety import _seed_config
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils as u
from vllm.v1.kv_offload.base import make_offload_key
root, model, manifest, action = sys.argv[1:]
spec = make_spec(model)
tensor, view = _view()
tier = make_tier(Path(root), view, spec=spec, content_manifest_path=manifest)
assert not tier.disabled
u.init_none_hash(
    sha256, kv_transfer_config=_seed_config(root, manifest),
    vllm_config=spec.vllm_config,
)
block = u.hash_block_tokens(sha256, None, list(range(16)), None)
k = make_offload_key(block, 0)
try:
    if action == 'store':
        _store(tier, tensor, k, value=7)
    else:
        assert tier.lookup(k)
        assert _load(tier, k) == [True]
        assert tensor[1, 0].item() == 7
finally:
    tier.shutdown()
print(json.dumps([u.NONE_HASH.hex(), k.hex()]))
"""
    results = []
    for seed, action in (("17", "store"), ("93", "load")):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                snippet,
                str(root),
                spec.vllm_config.model_config.model,
                manifest,
                action,
            ],
            env=dict(os.environ, PYTHONHASHSEED=seed),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(json.loads(result.stdout.splitlines()[-1]))
    assert results[0] == results[1]


def test_multimodal_model_is_supported_and_identified(tmp_path):
    """A multimodal checkpoint (vision encoder in the same checkpoint) keeps
    the tier enabled; its effective processing config is part of the ID."""
    import dataclasses as dc

    @dc.dataclass
    class MM:
        limit_per_prompt: dict
        mm_processor_kwargs: dict | None = None
        mm_encoder_tp_mode: str = "weights"

    model = _checkpoint(tmp_path)
    manifest = _write_manifest(tmp_path, {"target": str(model)})
    config = make_spec(model).vllm_config
    config.model_config.is_multimodal_model = True
    config.model_config.multimodal_config = MM(limit_per_prompt={"image": 4})
    first = get_revolta_identity(config, str(tmp_path / "kv"), manifest)
    assert first is not None
    config.model_config.multimodal_config = MM(
        limit_per_prompt={"image": 4}, mm_encoder_tp_mode="data"
    )
    second = get_revolta_identity(config, str(tmp_path / "kv"), manifest)
    assert second is not None
    assert second.persistent_digest != first.persistent_digest
    config.model_config.multimodal_config = MM(
        limit_per_prompt={"image": 4}, mm_processor_kwargs={"max_pixels": 1024}
    )
    third = get_revolta_identity(config, str(tmp_path / "kv"), manifest)
    assert third is not None
    assert third.persistent_digest not in (
        first.persistent_digest,
        second.persistent_digest,
    )


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"\xff",
        b"",
        b'{"version":1,"seed":""}',
        b'{"version":1,"seed":"\\ud800"}',
        b'{"version":1,"seed":"valid-prefix\\udfff"}',
    ],
)
def test_bad_seed_disables_existing_manager_at_init_none_hash(tmp_path, raw, hash_fn):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    spec.vllm_config.cache_config.prefix_caching_hash_algo = hash_fn.__name__
    tensor, view = _view()
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    path = root / ".hash_seed.json"
    path.write_bytes(raw)
    try:
        kv_cache_utils.init_none_hash(
            hash_fn,
            kv_transfer_config=_seed_config(root, manifest),
            vllm_config=spec.vllm_config,
        )
        assert tier.disabled and tier._closed
        assert path.read_bytes() == raw
        assert not tier.lookup(key(1))
        assert all(not thread.is_alive() for thread in tier._pool._threads)
        from vllm.v1.kv_offload.tiering.fs.config import get_fs_hash_seed

        assert (
            get_fs_hash_seed(_seed_config(root, manifest), vllm_config=spec.vllm_config)
            is None
        )
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert [r.success for r in tier.get_finished_jobs()] == [True]
    finally:
        tier.shutdown()


def test_cbor_seed_validation_precedes_propagation_to_missing_roots(tmp_path):
    from vllm.v1.kv_offload.tiering.fs.config import get_fs_hash_seed

    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    spec.vllm_config.cache_config.prefix_caching_hash_algo = "sha256_cbor"
    tensor, view = _view()
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    missing = tmp_path / "other"
    other = make_tier(missing, view, spec=spec, content_manifest_path=manifest)
    config = _seed_config(root, manifest)
    config.kv_connector_extra_config["secondary_tiers"] += _seed_config(
        missing, manifest
    ).kv_connector_extra_config["secondary_tiers"]
    path = root / ".hash_seed.json"
    raw = b'{"version":1,"seed":"\\ud800"}'
    path.write_bytes(raw)
    try:
        assert get_fs_hash_seed(config, vllm_config=spec.vllm_config) is None
        assert tier.disabled and other.disabled
        assert tier._closed and other._closed
        assert path.read_bytes() == raw
        assert not (missing / ".hash_seed.json").exists()
    finally:
        tier.shutdown()
        other.shutdown()


def test_cbor_unicode_seed_survives_real_init_none_hash(tmp_path):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    spec.vllm_config.cache_config.prefix_caching_hash_algo = "sha256_cbor"
    tensor, view = _view()
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    seed = "valid-\u00e9-\U0001f600"
    raw = json.dumps({"version": 1, "seed": seed}).encode()
    path = root / ".hash_seed.json"
    path.write_bytes(raw)
    try:
        kv_cache_utils.init_none_hash(
            sha256_cbor,
            kv_transfer_config=_seed_config(root, manifest),
            vllm_config=spec.vllm_config,
        )
        assert not tier.disabled
        assert sha256_cbor(seed) == kv_cache_utils.NONE_HASH
        assert path.read_bytes() == raw
    finally:
        tier.shutdown()


def test_non_utf8_namespace_config_disables_startup(tmp_path):
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    path = Path(tier.file_mapper.get_config_file_path())
    tier.shutdown()
    path.write_bytes(b"\xff\xfe")
    tier = make_tier(tmp_path, view)
    assert tier.disabled
    assert path.read_bytes() == b"\xff\xfe"


@pytest.mark.parametrize("error_at", ["read", "lock", "write", "conflict"])
def test_seed_failures_disable_all_revolta_roots(tmp_path, monkeypatch, error_at):
    from vllm.v1.kv_offload.tiering.fs.config import get_fs_hash_seed

    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    tensor, view = _view()
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    second_root = tmp_path / "second"
    second = make_tier(second_root, view, spec=spec, content_manifest_path=manifest)
    config = _seed_config(root, manifest)
    config.kv_connector_extra_config["secondary_tiers"] += _seed_config(
        second_root, manifest
    ).kv_connector_extra_config["secondary_tiers"]
    path = root / ".hash_seed.json"
    raw = b'{"version":1,"seed":"first"}'
    if error_at in ("read", "conflict"):
        path.write_bytes(raw)
    if error_at == "conflict":
        (second_root / ".hash_seed.json").write_text('{"version":1,"seed":"second"}')
    elif error_at == "read":
        real_open = Path.open

        def denied(path, *args, **kwargs):
            if path.name == ".hash_seed.json":
                raise PermissionError(errno.EACCES, "seed read denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
    elif error_at == "lock":
        import builtins

        builtin_open = builtins.open

        def denied(path, *args, **kwargs):
            if str(path).endswith(".hash_seed.lock"):
                raise PermissionError(errno.EACCES, "seed lock denied")
            return builtin_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", denied)
    else:
        os_open = os.open

        def denied(path, *args, **kwargs):
            if str(path).endswith(".hash_seed.json.new"):
                raise PermissionError(errno.EACCES, "seed write denied")
            return os_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", denied)
    try:
        assert get_fs_hash_seed(config, vllm_config=spec.vllm_config) is None
        assert tier.disabled and second.disabled
        assert tier._closed and second._closed
        if error_at in ("read", "conflict"):
            with open(path, "rb") as saved:
                assert saved.read() == raw
    finally:
        tier.shutdown()
        second.shutdown()


@pytest.mark.parametrize("raw", [b"{", b"\xff", b'{"version":1,"seed":""}'])
def test_plain_fs_bad_seed_still_raises(tmp_path, raw):
    from vllm.v1.kv_offload.tiering.fs.config import get_fs_hash_seed

    config = _seed_config(tmp_path, "unused")
    config.kv_connector_extra_config["secondary_tiers"][0]["type"] = "fs"
    (tmp_path / ".hash_seed.json").write_bytes(raw)
    with pytest.raises(ValueError):
        get_fs_hash_seed(config)
    assert (tmp_path / ".hash_seed.json").read_bytes() == raw


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_plain_fs_surrogate_seed_keeps_hash_algorithm_behavior(tmp_path, hash_fn):
    config = _seed_config(tmp_path, "unused")
    config.kv_connector_extra_config["secondary_tiers"][0]["type"] = "fs"
    path = tmp_path / ".hash_seed.json"
    raw = b'{"version":1,"seed":"\\ud800"}'
    path.write_bytes(raw)
    if hash_fn is sha256_cbor:
        with pytest.raises(UnicodeEncodeError):
            kv_cache_utils.init_none_hash(hash_fn, kv_transfer_config=config)
    else:
        kv_cache_utils.init_none_hash(hash_fn, kv_transfer_config=config)
        assert hash_fn("\ud800") == kv_cache_utils.NONE_HASH
    assert path.read_bytes() == raw


def test_real_runner_resolution_is_role_specific(tmp_path):
    from vllm.config import ModelConfig

    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    cfg = spec.vllm_config
    cfg.model_config.runner_type = ModelConfig._get_runner_type(
        None, [], "generate", "none"
    )
    cfg.speculative_config = _draft(cfg.model_config.model)
    draft = cfg.speculative_config.draft_model_config
    draft.runner_type = ModelConfig._get_runner_type(None, [], "draft", "none")
    assert draft.runner_type == "draft"
    assert get_revolta_identity(cfg, str(root), manifest) is not None
    draft.runner_type = "generate"
    assert get_revolta_identity(cfg, str(root), manifest) is None
    draft.runner_type = "draft"
    cfg.model_config.runner_type = "pooling"
    assert get_revolta_identity(cfg, str(root), manifest) is None


@pytest.mark.parametrize("role", ["target", "draft"])
@pytest.mark.parametrize(
    "source",
    [
        "hf_overrides",
        "hf_config",
        "hf_text_config",
        "quantization_config",
        "hf_config_path",
        "multimodal_config",
        "index",
    ],
)
def test_external_config_sources_disable_tier(tmp_path, role, source):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    cfg = spec.vllm_config
    cfg.speculative_config = _draft(cfg.model_config.model)
    model = (
        cfg.model_config
        if role == "target"
        else cfg.speculative_config.draft_model_config
    )
    external = tmp_path / "external.json"
    external.write_text('{"config":"A"}')
    if source == "index":
        (Path(model.model) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"w": str(external)}})
        )
        manifest = _write_manifest(tmp_path, {"target": model.model})
    elif source == "hf_config_path":
        model.hf_config_path = str(external)
    else:
        setattr(model, source, {"nested": {"quantization_config_file": str(external)}})
    assert get_revolta_identity(cfg, str(root), manifest) is None
    tensor, view = _view()
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    assert tier.disabled and not root.exists()
    external.write_text('{"config":"B"}')
    assert get_revolta_identity(cfg, str(root), manifest) is None


def test_transient_read_error_preserves_page(tmp_path, monkeypatch):
    tensor, view = _view()
    tier = make_tier(tmp_path, view)
    path = _store(tier, tensor, key(1), value=7)
    real_readv = os.readv
    calls = []

    def transient(fd, buffers):
        calls.append(True)
        if len(calls) == 1:
            raise OSError(errno.EAGAIN, "transient read")
        return real_readv(fd, buffers)

    monkeypatch.setattr(os, "readv", transient)
    try:
        # Direct verification keeps the valid file available for the next read.
        with pytest.raises(OSError):
            revolta.load_verified_block(
                path,
                view,
                tier._block_size,
                tier._block_size,
                key(1),
                tier._identity_digest,
            )
        assert os.path.exists(path)
        assert _load(tier, key(1)) == [True]
        assert len(calls) == 2
        assert tensor[1].eq(7).all()
    finally:
        tier.shutdown()


def raw_config(cls, **overrides):
    # Real hash code, without model or device initialization.
    from pydantic.fields import FieldInfo
    from pydantic_core import PydanticUndefined

    obj = object.__new__(cls)
    for field in dataclasses.fields(cls):
        if field.name in overrides:
            value = overrides[field.name]
        elif isinstance(field.default, FieldInfo):
            value = field.default.get_default(call_default_factory=True)
            if value is PydanticUndefined:
                value = None
        elif field.default is not dataclasses.MISSING:
            value = field.default
        elif field.default_factory is not dataclasses.MISSING:
            value = field.default_factory()
        else:
            value = None
        setattr(obj, field.name, value)
    for name, value in overrides.items():
        setattr(obj, name, value)
    return obj


def test_external_quant_file_real_config_hash_and_loader_path(tmp_path, monkeypatch):
    from transformers import LlamaConfig

    from vllm.config import ModelConfig, VllmConfig
    from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
    from vllm.model_executor.model_loader import weight_utils

    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    cfgfile = tmp_path / "quant.json"
    cfgfile.write_text('{"_type": "Float8WeightOnlyConfig", "_data": {"version": 1}}')
    hf = LlamaConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2)
    model = raw_config(
        ModelConfig,
        model=spec.vllm_config.model_config.model,
        hf_config=hf,
        hf_text_config=hf,
        runner_type="generate",
        quantization="torchao",
        hf_overrides={"quantization_config_file": str(cfgfile)},
    )
    cfg = raw_config(VllmConfig, model_config=model, device_config=None)
    # Real file resolution/JSON parsing; avoid initializing optional TorchAO kernels.
    monkeypatch.setattr(
        TorchAOConfig, "from_config", classmethod(lambda cls, config: config)
    )
    selected_a = weight_utils.get_quant_config(model, cfg.load_config)
    before_hash = cfg.compute_hash()
    before = get_revolta_identity(cfg, str(root), manifest)
    assert before is None
    cfgfile.write_text('{"_type": "Float8WeightOnlyConfig", "_data": {"version": 2}}')
    selected_b = weight_utils.get_quant_config(model, cfg.load_config)
    assert selected_a != selected_b
    assert cfg.compute_hash() == before_hash
    assert get_revolta_identity(cfg, str(root), manifest) == before


@pytest.mark.parametrize("role", ["target", "draft"])
def test_manifest_bound_quantization_file_changes_identity(tmp_path, role):
    root = tmp_path / "kv"
    spec, _ = checkpoint_spec(root)
    cfg = spec.vllm_config
    cfg.speculative_config = _draft(cfg.model_config.model)
    model = (
        cfg.model_config
        if role == "target"
        else cfg.speculative_config.draft_model_config
    )
    path = Path(model.model) / "quant.json"
    path.write_text('{"version":1}')
    model.hf_overrides = {"quantization_config_file": str(path)}
    manifest = _write_manifest(tmp_path, {"target": model.model})
    before = get_revolta_identity(cfg, str(root), manifest)
    assert before is not None
    path.write_text('{"version":2}')
    assert get_revolta_identity(cfg, str(root), manifest) is None
    manifest = _write_manifest(tmp_path, {"target": model.model})
    after = get_revolta_identity(cfg, str(root), manifest)
    assert after is not None and before != after


@pytest.mark.parametrize("role", ["target", "draft"])
@pytest.mark.parametrize("source", ["vision_tower", "quantization_config_dict_json"])
def test_unresolved_nested_sources_disable_identity(tmp_path, role, source):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    cfg = spec.vllm_config
    cfg.speculative_config = _draft(cfg.model_config.model)
    model = (
        cfg.model_config
        if role == "target"
        else cfg.speculative_config.draft_model_config
    )
    value = "example/external-encoder"
    if source.endswith("_json"):
        value = json.dumps({"nested": {"scale_path": "missing-scales.json"}})
    model.hf_overrides = {source: value}
    assert get_revolta_identity(cfg, str(root), manifest) is None


def test_auxiliary_config_directory_requires_exact_manifest_files(tmp_path):
    root = tmp_path / "kv"
    spec, _ = checkpoint_spec(root)
    auxiliary = tmp_path / "auxiliary"
    auxiliary.mkdir()
    (auxiliary / "config.json").write_text("{}")
    model = spec.vllm_config.model_config
    model.hf_config_path = str(auxiliary)
    manifest = _write_manifest(
        tmp_path, {"target": model.model, "config": str(auxiliary)}
    )
    assert get_revolta_identity(spec.vllm_config, str(root), manifest) is not None
    (auxiliary / "extra.json").write_text("{}")
    assert get_revolta_identity(spec.vllm_config, str(root), manifest) is None


def test_awq_multimodal_metadata_through_real_model_config(tmp_path):
    from vllm.config import ModelConfig, MultiModalConfig
    from vllm.models.qwen4_exp.config import Qwen4ExpConfig

    model = tmp_path / "checkpoint"
    # Synthetic dimensions and metadata exercise the composite config loader
    # without loading weights or depending on a deployed checkpoint.
    hf = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 524288,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
        },
        quantization_config={
            "quant_method": "awq",
            "bits": 4,
            "group_size": 32,
            "zero_point": True,
            "meta": {
                "selection": "s1/s2 re-searched synthetic candidates",
                "code_rule": "round(w/FP16(scale)+integer_zero), synthetic rule",
                "old_template_provenance": "template-provenance/",
            },
        },
    )
    hf.save_pretrained(model)
    cfg = ModelConfig(
        model=str(model),
        # Pin the fixture's codec: a CPU platform can automatically select
        # awq_marlin, while this test exercises AWQ metadata identity.
        quantization="awq",
        dtype="half",
        max_model_len=524288,
        language_model_only=False,
        limit_mm_per_prompt={"image": 16, "video": 1},
        mm_processor_kwargs={},
    )
    assert cfg.is_multimodal_model and cfg.runner_type == "generate"
    assert cfg.quantization == "awq"
    assert isinstance(cfg.multimodal_config, MultiModalConfig)
    assert cfg.multimodal_config.language_model_only is False
    assert cfg.multimodal_config.get_limit_per_prompt("image") == 16
    assert cfg.multimodal_config.get_limit_per_prompt("video") == 1
    assert cfg.multimodal_config.mm_processor_kwargs == {}
    spec = make_spec(model)
    spec.vllm_config.model_config = cfg
    manifest = _write_manifest(tmp_path, {"target": str(model)})
    root = tmp_path / "kv"
    before = get_revolta_identity(spec.vllm_config, str(root), manifest)
    assert before is not None
    tensor, view = _view()
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    try:
        assert not tier.disabled
    finally:
        tier.shutdown()
    cfg.hf_config.quantization_config["meta"]["selection"] += " updated"
    after = get_revolta_identity(spec.vllm_config, str(root), manifest)
    assert after is not None and before != after


@pytest.mark.parametrize(
    "field",
    [
        "quantization_config_file",
        "vocab_file",
        "merges_file",
        "tokenizer_file",
        "scale_path",
        "processor_dir",
        "vision_tower",
        "source",
        "encoder_source",
    ],
)
def test_unresolved_loader_processor_sources_still_disable(tmp_path, field):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    spec.vllm_config.model_config.hf_overrides = {field: "unresolved-source"}
    assert get_revolta_identity(spec.vllm_config, str(root), manifest) is None
    assert not root.exists()


@pytest.mark.parametrize("exists", [False, True])
def test_absolute_or_existing_paths_in_unknown_fields_still_disable(
    tmp_path, monkeypatch, exists
):
    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    external = tmp_path / "external.json"
    value = str(external)
    if exists:
        external.write_text("{}")
        monkeypatch.chdir(tmp_path)
        value = external.name
    spec.vllm_config.model_config.hf_config = {"unknown": value}
    assert get_revolta_identity(spec.vllm_config, str(root), manifest) is None


@pytest.mark.parametrize("role", ["target", "draft"])
@pytest.mark.parametrize("scales", ["dynamic", "external", "manifest"])
def test_multimodal_encoder_fp8_requires_manifest_bound_fixed_scales(
    tmp_path, role, scales
):
    from vllm.config import MultiModalConfig

    root = tmp_path / "kv"
    spec, manifest = checkpoint_spec(root)
    cfg = spec.vllm_config
    cfg.speculative_config = _draft(cfg.model_config.model)
    model = (
        cfg.model_config
        if role == "target"
        else cfg.speculative_config.draft_model_config
    )
    scale_path = None
    if scales != "dynamic":
        scale_path = (
            Path(model.model) if scales == "manifest" else tmp_path
        ) / "encoder-scales.json"
        scale_path.write_text('{"layer": {"q": 1, "k": 1, "v": 1}}')
        manifest = _write_manifest(tmp_path, {"target": cfg.model_config.model})
    model.multimodal_config = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=None if scale_path is None else str(scale_path),
    )
    identity = get_revolta_identity(cfg, str(root), manifest)
    tensor, view = _view()
    tier = make_tier(root, view, spec=spec, content_manifest_path=manifest)
    try:
        assert tier.disabled == (scales != "manifest")
        if scales != "manifest":
            assert identity is None and not root.exists()
            kv_cache_utils.init_none_hash(
                sha256, kv_transfer_config=_seed_config(root, manifest), vllm_config=cfg
            )
            assert not root.exists()
        else:
            assert identity is not None
            assert scale_path is not None
            scale_path.write_text('{"layer": {"q": 2, "k": 1, "v": 1}}')
            assert get_revolta_identity(cfg, str(root), manifest) is None
            manifest = _write_manifest(tmp_path, {"target": cfg.model_config.model})
            updated = get_revolta_identity(cfg, str(root), manifest)
            assert updated is not None and updated != identity
    finally:
        tier.shutdown()


@pytest.mark.parametrize("method", ["draft_model", "ngram"])
def test_real_speculative_config_runner_identity(tmp_path, method):
    from transformers import LlamaConfig

    from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    hf = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    hf.architectures = ["LlamaForCausalLM"]
    hf.save_pretrained(target_path)
    hf.save_pretrained(draft_path)
    target = ModelConfig(model=str(target_path), dtype="half", max_model_len=128)
    speculative = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        model=str(draft_path) if method == "draft_model" else None,
        method=method,
        num_speculative_tokens=2,
    )
    cfg = make_spec(target_path).vllm_config
    cfg.model_config = target
    cfg.speculative_config = speculative
    manifest = _write_manifest(
        tmp_path, {"target": str(target_path), "draft": str(draft_path)}
    )
    assert target.runner_type == "generate"
    if method == "draft_model":
        assert speculative.draft_model_config.runner_type == "draft"
    else:
        assert speculative.draft_model_config is target
    assert get_revolta_identity(cfg, str(tmp_path / "kv"), manifest) is not None
