# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process restarts must reuse FS hashes and enforce bounded disk usage."""

import concurrent.futures
import inspect
import json
import os
import subprocess
import sys

import pytest

from vllm.config import KVTransferConfig
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_offload.tiering.fs.cache import FileSystemCache

SNIPPET = r"""
import inspect, json, sys
from vllm.config import KVTransferConfig
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils as u
root = sys.argv[1]
config = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
    kv_connector_extra_config={"secondary_tiers": [
        {"type": "fs", "root_dir": root, "max_size_bytes": 1048576}
    ]} if root else {})
# RC1c has no FS-aware initialization. Exercise its actual startup behavior
# too, so the baseline demonstrates changed hashes rather than a missing API.
kwargs = {}
if "kv_transfer_config" in inspect.signature(u.init_none_hash).parameters:
    kwargs["kv_transfer_config"] = config
u.init_none_hash(sha256, **kwargs)
parent = None
hashes = []
for start in range(0, 64, 16):
    parent = u.hash_block_tokens(sha256, parent, list(range(start, start + 16)), None)
    hashes.append(parent.hex())
if len(sys.argv) > 2 and sys.argv[2]:
    import mmap
    from types import SimpleNamespace as NS
    from vllm.v1.kv_offload.base import make_offload_key
    from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
    backing = mmap.mmap(-1, 8192)
    view = memoryview(backing).cast("B", shape=[2, 4096])
    spec = NS(block_size_factor=1, persistent_layout=None,
        vllm_config=NS(model_config=NS(model="test/fs-restart"),
            cache_config=NS(block_size=16, cache_dtype="float32"),
            parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                prefill_context_parallel_size=1, decode_context_parallel_size=1,
                rank=0)),
        kv_cache_config=NS(kv_cache_groups=[]))
    tier = FileSystemTierManager(spec, view, "fs", root,
        n_read_threads=1, n_write_threads=1, max_size_bytes=1048576)
    key = make_offload_key(parent, 0)
    path = tier.file_mapper.get_file_name(key)
    try:
        if sys.argv[2] == "store":
            backing[:4096] = bytes(range(256)) * 16
            tier._store(path, 0)
        else:
            assert tier.lookup(key), "persisted page missed after process restart"
            tier._load(path, 1)
            assert backing[4096:] == bytes(range(256)) * 16
    finally:
        tier.shutdown()
        view.release()
        backing.close()
print(json.dumps(hashes))
"""


def hashes(root="", seed=None, action=""):
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    if seed is not None:
        env["PYTHONHASHSEED"] = seed
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET, str(root), action],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_fs_hashes_survive_restart_without_environment_seed(tmp_path):
    first = hashes(tmp_path, action="store")
    # A page indexed by the first process must be discoverable after restart.
    path = tmp_path / (first[-1] + ".bin")
    cache = FileSystemCache.acquire(str(tmp_path), 1048576, 0)
    try:
        with cache.write(str(path), 4096):
            path.write_bytes(b"a" * 4096)
    finally:
        cache.release()
    second = hashes(tmp_path, action="load")
    assert first == second
    cache = FileSystemCache.acquire(str(tmp_path), 1048576, 0)
    try:
        assert cache.lookup(str(tmp_path / (second[-1] + ".bin")))
    finally:
        cache.release()


def test_shared_root_serializes_first_seed_creation(tmp_path):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: hashes(tmp_path), range(3)))
    assert all(result == results[0] for result in results)
    # A later launch cannot silently replace a populated cache's identity.
    assert hashes(tmp_path, "123") == results[0]


def test_cpu_only_hashes_keep_legacy_pythonhashseed_zero():
    first = hashes(seed="0")
    assert hashes(seed="0") == first
    # The fixture is computed on RC1c, without an FS tier.
    assert first == [
        "6358e31578a64b75ebe28cdb99d3beae65de82a2087321dc6ec84ae048faaaa6",
        "3c3c50f324aefbdfbb124dfdc541efac3e94e59c40f4be08afe70601eeca92c8",
        "0728d91353ef7b6761329eaaf146e88046d84f6384628314a0a6c35dd9b88e0a",
        "71c151a876275d6b365dfa670e47acb855b683a8880eaf3e31a806c3164561b4",
    ]


@pytest.mark.parametrize("budget", [None, 0, -1, True, "4096"])
def test_fs_config_requires_positive_byte_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="max_size_bytes"):
        KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "secondary_tiers": [
                    {"type": "fs", "root_dir": str(tmp_path), "max_size_bytes": budget}
                ]
            },
        )


def test_fs_cache_rejects_unbounded_budget(tmp_path):
    with pytest.raises(ValueError, match="max_size_bytes"):
        cache = FileSystemCache.acquire(str(tmp_path), None, 0)
        cache.release()


def test_default_free_space_reserve_is_nonzero(tmp_path):
    cache = FileSystemCache.acquire(str(tmp_path), 1048576)
    try:
        stat = os.statvfs(tmp_path)
        assert cache.min_free_bytes == max(1, stat.f_blocks * stat.f_frsize // 20)
    finally:
        cache.release()


def initialize_fs_roots(roots):
    config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "secondary_tiers": [
                {"type": "fs", "root_dir": str(root), "max_size_bytes": 1048576}
                for root in roots
            ]
        },
    )
    kwargs = {}
    if (
        "kv_transfer_config"
        in inspect.signature(kv_cache_utils.init_none_hash).parameters
    ):
        kwargs["kv_transfer_config"] = config
    kv_cache_utils.init_none_hash(sha256, **kwargs)


def test_multiple_roots_share_identity_and_reject_conflicts(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    initialize_fs_roots([first, second])
    assert (first / ".hash_seed.json").read_bytes() == (
        second / ".hash_seed.json"
    ).read_bytes()
    (second / ".hash_seed.json").write_text(json.dumps({"version": 1, "seed": "other"}))
    with pytest.raises(ValueError, match="different hash seeds"):
        initialize_fs_roots([first, second])


@pytest.mark.parametrize("contents", ["{broken", '{"version": 1, "seed": ""}'])
def test_corrupt_seed_is_not_replaced(tmp_path, contents):
    path = tmp_path / ".hash_seed.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        initialize_fs_roots([tmp_path])
    assert path.read_text() == contents
