# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import errno
import mmap
import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.utils import checkpoint_cache as cache


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setitem(
        vars(cache.envs), "REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP", True
    )
    events = []
    group = SimpleNamespace(barrier=lambda: events.append("host-allocation-fence"))

    def synchronize():
        events.append("all-loaders-and-copies-complete")
        return group

    monkeypatch.setattr(cache, "_synchronize_loaders", synchronize)
    monkeypatch.setattr(cache, "_gather_mapped_files", lambda group: set())
    monkeypatch.setattr(cache, "_cgroup_memory", lambda: (100, 60))
    return events


def test_main_and_draft_collection_and_scope_reset(enabled, tmp_path):
    main, draft = tmp_path / "main.safetensors", tmp_path / "draft.safetensors"
    with cache.collect_checkpoint_files() as files:
        cache.record_checkpoint_files([str(main)])
        cache.record_checkpoint_files([str(draft), str(main)])
    cache.record_checkpoint_files([str(tmp_path / "later")])
    assert files == {str(main), str(draft)}


def test_disabled_does_not_advise_or_synchronize(monkeypatch, tmp_path):
    monkeypatch.setitem(
        vars(cache.envs), "REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP", False
    )
    advise, synchronize = Mock(), Mock()
    monkeypatch.setattr(os, "posix_fadvise", advise)
    monkeypatch.setattr(cache, "_synchronize_loaders", synchronize)
    with cache.collect_checkpoint_files() as files:
        cache.record_checkpoint_files([str(tmp_path / "checkpoint")])
    cache.release_checkpoint_cache(files)
    assert not files
    advise.assert_not_called()
    synchronize.assert_not_called()


def test_advice_after_load_barrier_deduplicates_aliases_and_logs_bytes(
    enabled, monkeypatch, tmp_path
):
    path = tmp_path / "checkpoint"
    payload = b"checkpoint data"
    path.write_bytes(payload)
    alias = tmp_path / "alias"
    os.link(path, alias)
    log = Mock()
    monkeypatch.setattr(cache.logger, "info", log)
    snapshots = iter([(100, 60), (70, 30)])
    monkeypatch.setattr(cache, "_cgroup_memory", lambda: next(snapshots))

    def advise(fd, offset, length, advice):
        assert enabled == ["all-loaders-and-copies-complete"]
        assert (offset, length, advice) == (0, 0, os.POSIX_FADV_DONTNEED)
        assert os.fstat(fd).st_ino == path.stat().st_ino
        enabled.append("advise")

    monkeypatch.setattr(os, "posix_fadvise", advise)
    cache.release_checkpoint_cache({str(path), str(alias)})
    assert enabled[-1] == "host-allocation-fence"
    message = log.call_args.args[0] % log.call_args.args[1:]
    assert f"advised_files=1 advised_bytes={len(payload)}" in message
    assert "memory.current=100 -> 70" in message
    assert "memory.stat file=60 -> 30" in message
    assert path.read_bytes() == payload


@pytest.mark.parametrize("protection", ["ple-disk", "unknown-maps", "remote-mmap"])
def test_protected_files_are_not_advised(enabled, monkeypatch, tmp_path, protection):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"data")
    info = path.stat()
    mapped = None if protection == "unknown-maps" else {(info.st_dev, info.st_ino)}
    monkeypatch.setattr(cache, "_gather_mapped_files", lambda group: mapped)
    advise = Mock()
    monkeypatch.setattr(os, "posix_fadvise", advise)
    cache.release_checkpoint_cache({str(path)}, protect_all=protection == "ple-disk")
    advise.assert_not_called()
    assert enabled[-1] == "host-allocation-fence"


def test_errors_are_best_effort_and_still_fence(enabled, monkeypatch, tmp_path):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"data")
    monkeypatch.setattr(
        os, "posix_fadvise", Mock(side_effect=OSError(errno.EINVAL, "unsupported"))
    )
    cache.release_checkpoint_cache({str(path), str(tmp_path / "missing")})
    assert enabled[-1] == "host-allocation-fence"


def test_unsupported_platform_and_empty_rank_still_fence(
    enabled, monkeypatch, tmp_path
):
    monkeypatch.delattr(os, "posix_fadvise")
    cache.release_checkpoint_cache({str(tmp_path / "checkpoint")})
    cache.release_checkpoint_cache(set())
    assert enabled.count("host-allocation-fence") == 2


def test_live_mmap_is_identified_by_inode(tmp_path):
    path = tmp_path / "checkpoint with spaces"
    path.write_bytes(bytes(mmap.PAGESIZE))
    with (
        path.open("rb") as source,
        mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ),
    ):
        info = path.stat()
        assert (info.st_dev, info.st_ino) in cache._mapped_file_ids()
    assert cache._mapped_file_ids(tmp_path / "missing-maps") is None


@pytest.mark.parametrize("container_root", [False, True])
def test_cgroup_reads_leaf_or_namespaced_root(tmp_path, container_root):
    root = tmp_path / "cgroup"
    node = root if container_root else root / "worker"
    node.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/worker\n")
    (node / "memory.current").write_text("100")
    (node / "memory.stat").write_text("anon 30\nfile 60\nshmem 10\n")
    assert cache._cgroup_memory(root, membership) == (100, 60)
    (node / "memory.stat").unlink()
    assert cache._cgroup_memory(root, membership) == (100, None)


@pytest.mark.parametrize("remote", [set(), None, {(1, 2)}])
def test_remote_mappings_are_gathered_on_cpu_group(monkeypatch, remote):
    import torch.distributed as dist

    monkeypatch.setattr(cache, "_mapped_file_ids", lambda: {(3, 4)})
    group = SimpleNamespace(world_size=2, cpu_group=object())

    def gather(output, local, *, group):
        assert group is not None
        output[:] = [local, remote]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    expected = None if remote is None else {(3, 4)} | remote
    assert cache._gather_mapped_files(group) == expected


def test_device_copies_finish_before_world_barrier(monkeypatch):
    import torch

    from vllm.distributed import parallel_state

    events = []
    group = SimpleNamespace(barrier=lambda: events.append("world-barrier"))
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda: events.append("copies")
    )
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: group)
    assert cache._synchronize_loaders() is group
    assert events == ["copies", "world-barrier"]


def test_default_loader_tracks_real_main_and_draft_inputs(enabled, tmp_path):
    import torch
    from safetensors.torch import save_file

    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    loader = DefaultModelLoader(
        LoadConfig(load_format="safetensors", safetensors_load_strategy="lazy")
    )
    with cache.collect_checkpoint_files() as files:
        for kind in ("main", "draft"):
            folder = tmp_path / kind
            folder.mkdir()
            path = folder / "model.safetensors"
            save_file({"weight": torch.ones(2)}, str(path))
            source = DefaultModelLoader.Source(str(folder), revision=None)
            iterator = loader._get_weights_iterator(source)
            assert str(path) not in files
            weights = dict(iterator)
            assert torch.equal(weights["weight"], torch.ones(2))
    assert files == {
        str(tmp_path / kind / "model.safetensors") for kind in ("main", "draft")
    }
    with cache.collect_checkpoint_files() as incomplete:
        iterator = loader._get_weights_iterator(source)
        next(iterator)
        iterator.close()
    assert not incomplete


def _gloo_drop_worker(rank, rendezvous, files, waiting, loaded, release, results):
    import torch
    import torch.distributed as dist

    from vllm.distributed import parallel_state

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    group = SimpleNamespace(
        world_size=2, cpu_group=dist.group.WORLD, barrier=dist.barrier
    )
    parallel_state.get_world_group = lambda: group
    torch.accelerator.synchronize = lambda: waiting.set() if rank == 0 else None
    vars(cache.envs)["REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP"] = True
    advised = []

    def advise(fd, offset, length, advice):
        assert loaded.is_set(), "A rank advised while its sibling was still loading"
        advised.append(os.fstat(fd).st_ino)

    os.posix_fadvise = advise
    try:
        if rank == 1:
            # This sibling retains one checkpoint as a file-backed tensor.
            with (
                open(files[1], "rb") as source,
                mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ),
            ):
                assert release.wait(15)
                loaded.set()
                cache.release_checkpoint_cache(set(files))
        else:
            cache.release_checkpoint_cache(set(files))
        results.put(advised)
    finally:
        dist.destroy_process_group()


def test_two_cpu_ranks_wait_for_reader_and_protect_its_mapping(tmp_path):
    import multiprocessing as mp

    context = mp.get_context("spawn")
    waiting, loaded, release = (context.Event() for _ in range(3))
    results = context.Queue()
    files = [str(tmp_path / name) for name in ("copied", "retained")]
    for filename in files:
        with open(filename, "wb") as target:
            target.write(bytes(mmap.PAGESIZE))
    rendezvous = (tmp_path / "gloo").as_uri()
    processes = [
        context.Process(
            target=_gloo_drop_worker,
            args=(rank, rendezvous, files, waiting, loaded, release, results),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert waiting.wait(15)
        assert not loaded.is_set()
        release.set()
        for process in processes:
            process.join(25)
            assert process.exitcode == 0
        for _ in processes:
            assert results.get(timeout=5) == [os.stat(files[0]).st_ino]
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
