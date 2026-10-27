# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU composition proof: existing tier managers and FS I/O behind group routing.

These tests drive grouped managers directly and never register memory with CUDA
or run GPU transfers. Grouped TieringOffloadingSpec itself is production code
whenever the CPU pool is partitioned: `get_manager` routes to
`_get_grouped_manager`, one primary tier plus one filesystem tier per group.
"""

import json
import multiprocessing
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.base import (
    OffloadingManager,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import GroupedCPUOffloadingManager

GROUP_PAGES = {0: 8192, 2: 4096}
WORLD_SIZE = 2
CTX = ReqContext("grouped-fs")


def _key(group, number=1):
    return make_offload_key(number.to_bytes(8, "big"), group)


def _spec():
    # The existing FileMapper retains the original group IDs in OffloadKey.
    return SimpleNamespace(
        block_size_factor=1,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="test/grouped-hybrid"),
            cache_config=SimpleNamespace(block_size=16, cache_dtype="float16"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=WORLD_SIZE,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                rank=0,
            ),
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(block_size=16),
                    layer_names=[f"layer-{g}"],
                )
                for g in range(3)
            ]
        ),
    )


def _build(root, pages=None, **fs_options):
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    pages = GROUP_PAGES if pages is None else pages
    spec = object.__new__(TieringOffloadingSpec)
    config = _spec()
    spec.vllm_config = config.vllm_config
    spec.vllm_config.instance_id = f"grouped-fs-test-{uuid.uuid4().hex}"
    spec.vllm_config.parallel_config.world_size = WORLD_SIZE
    spec.vllm_config.kv_events_config = SimpleNamespace(enable_kv_cache_events=True)
    spec.kv_cache_config = config.kv_cache_config
    spec.block_size_factor = 1
    spec.cpu_group_page_sizes = pages
    spec.num_blocks = 2
    spec.cpu_group_num_blocks = {group: 2 for group in pages}
    spec.cpu_group_offloaded_tokens = {group: 16 for group in pages}
    spec.partition_by_group = True
    spec.eviction_policy = "lru"
    spec.extra_config = {}
    spec.persistent_layout = {"version": "test-grouped-row-v1"}
    spec.secondary_tier_configs = [
        {
            "type": "fs",
            "root_dir": str(root),
            "max_size_bytes": 1 << 24,
            "n_read_threads": 1,
            "n_write_threads": 1,
            **fs_options,
        }
    ]
    spec._manager = None
    return spec.get_manager()


def _settle(manager):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        list(manager.take_events())
        if all(
            not m._transfer_jobs and not m._pending_load_submissions
            for m in manager.managers.values()
        ):
            return
        time.sleep(0.005)
    raise AssertionError("FS jobs did not finish")


def _payload(group, pages=None):
    # Distinguish both groups and both TP rank slices within each stored row.
    pages = GROUP_PAGES if pages is None else pages
    return b"".join(
        bytes([31 + group + rank * 70]) * pages[group] for rank in range(WORLD_SIZE)
    )


def _write_rows(manager, output, payloads):
    for key, block_id in zip(output.keys_to_store, output.store_spec.block_ids):
        group = int.from_bytes(key[-4:], "big")
        primary = manager.managers[group].primary_tier
        with primary.get_kv_memoryview().cast("B") as view:
            size = len(payloads[group])
            offset = int(block_id) * size
            view[offset : offset + size] = payloads[group]
    manager.complete_store(output.keys_to_store, CTX)
    _settle(manager)


def _phase(root, phase):
    """Runs in a fresh spawn process, so no cache index or mapping is inherited."""
    manager = _build(root)
    paths = []
    try:
        manager.on_new_request(CTX)
        keys = [_key(g) for g in GROUP_PAGES]
        if phase == "write":
            output = manager.prepare_store(keys, CTX)
            assert list(output.store_spec.block_ids) == [0, 0]
            _write_rows(manager, output, {g: _payload(g) for g in GROUP_PAGES})
            for g, child in manager.managers.items():
                path = Path(child.secondary_tiers[0].file_mapper.get_file_name(_key(g)))
                assert path.read_bytes() == _payload(g)
                paths.append(str(path))
            assert paths[0] != paths[1]
        else:
            assert all(
                child.primary_tier.lookup(_key(g), CTX) is False
                for g, child in manager.managers.items()
            )
            # Occupy slot 0: persisted identity must not depend on the old slot.
            filler = manager.prepare_store([_key(g, 2) for g in GROUP_PAGES], CTX)
            _write_rows(
                manager,
                filler,
                {g: bytes([99]) * len(_payload(g)) for g in GROUP_PAGES},
            )
            assert all(manager.lookup(key, CTX) is None for key in keys)
            _settle(manager)
            if phase == "corrupt":
                assert manager.lookup(_key(0), CTX) is False
                assert manager.managers[0].primary_tier.lookup(_key(0), CTX) is False
                keys = [_key(2)]
            assert all(manager.lookup(key, CTX) is True for key in keys)
            loaded = manager.prepare_load(keys, CTX)
            assert list(loaded.block_ids) == [1] * len(keys)
            for key, bid in zip(keys, loaded.block_ids):
                g = int.from_bytes(key[-4:], "big")
                with (
                    manager.managers[g]
                    .primary_tier.get_kv_memoryview()
                    .cast("B") as view
                ):
                    size = len(_payload(g))
                    assert bytes(
                        view[int(bid) * size : (int(bid) + 1) * size]
                    ) == _payload(g)
            manager.complete_load(keys, CTX)
            # Fill beyond capacity; old filler rows are evicted through standard LRU.
            manager.touch(keys, CTX)
            pressure = manager.prepare_store([_key(g, 3) for g in GROUP_PAGES], CTX)
            assert set(pressure.evicted_keys) == (
                {_key(2, 2)}
                if phase == "corrupt"
                else {_key(g, 2) for g in GROUP_PAGES}
            )
            assert list(pressure.store_spec.block_ids) == (
                [1, 0] if phase == "corrupt" else [0, 0]
            )
            _write_rows(
                manager,
                pressure,
                {g: bytes([100]) * len(_payload(g)) for g in GROUP_PAGES},
            )
        manager.on_request_finished(CTX)
        paths += [
            m.primary_tier._mmap_region.mmap_path for m in manager.managers.values()
        ]
    finally:
        manager.shutdown()
    assert all(not Path(p).exists() for p in paths if p.endswith(".mmap"))
    (Path(root) / f"{phase}.json").write_text(
        json.dumps({"phase": phase, "pass": True})
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_fs_groups_survive_process_restart(tmp_path, corrupt):
    for phase in ("write", "corrupt" if corrupt else "read"):
        if phase == "corrupt":
            files = list(tmp_path.glob("**/*_g0/*.bin"))
            assert len(files) == 1
            files[0].write_bytes(b"truncated")
        process = multiprocessing.get_context("spawn").Process(
            target=_phase, args=(str(tmp_path), phase)
        )
        process.start()
        process.join(60)
        if process.is_alive():
            process.terminate()
            process.join(10)
            pytest.fail(f"{phase} process did not finish")
        assert process.exitcode == 0
        assert json.loads((tmp_path / f"{phase}.json").read_text())["pass"]


def test_group_routing_forwards_all_lifecycle_hooks():
    children = {g: MagicMock(spec=OffloadingManager) for g in (0, 2, 5)}
    for g, child in children.items():
        child.on_new_request.return_value = RequestOffloadingContext(
            policy=OffloadPolicy.REQUEST_LEVEL if g == 0 else OffloadPolicy.BLOCK_LEVEL
        )
    manager = GroupedCPUOffloadingManager(children)
    assert manager.on_new_request(CTX).policy == OffloadPolicy.REQUEST_LEVEL
    manager.on_request_finished(CTX)
    manager.shutdown()
    for child in children.values():
        child.on_new_request.assert_called_once_with(CTX)
        child.on_request_finished.assert_called_once_with(CTX)
        child.shutdown.assert_called_once_with()


def test_tiering_shared_rows_accept_mixed_gpu_block_sizes():
    """Distinct group keys may share the tier's fixed-size worker rows."""
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    spec = object.__new__(TieringOffloadingSpec)
    spec.partition_by_group = False
    spec.gpu_block_size = (3200, 1600)
    spec._manager = None
    spec.vllm_config = SimpleNamespace(
        instance_id=f"mixed-tier-test-{uuid.uuid4().hex}",
        parallel_config=SimpleNamespace(world_size=2),
        kv_events_config=None,
    )
    spec.cpu_page_size_per_worker = 64
    spec.num_blocks = 4
    spec.eviction_policy = "lru"
    spec.secondary_tier_configs = []
    spec.extra_config = {}
    manager = spec.get_manager()
    try:
        keys = [_key(0, 1), _key(1, 1)]
        prepared = manager.prepare_store(keys, CTX)
        assert prepared is not None
        assert set(prepared.keys_to_store) == set(keys)
        manager.complete_store(keys, CTX)
        assert all(manager.lookup(key, CTX) is True for key in keys)
    finally:
        manager.shutdown()
        spec._manager = None
        assert spec._scheduler_mmap is not None
        spec._scheduler_mmap.cleanup()


def test_spec_group_regions_share_geometry_and_cleanup_partial_failure(monkeypatch):
    import vllm.v1.kv_offload.tiering.spec as spec_module

    spec = object.__new__(spec_module.TieringOffloadingSpec)
    spec.vllm_config = SimpleNamespace(
        instance_id=f"grouped-spec-test-{uuid.uuid4().hex}",
        parallel_config=SimpleNamespace(world_size=2),
    )
    spec.cpu_group_page_sizes = GROUP_PAGES
    spec.num_blocks = 2
    spec.cpu_group_num_blocks = {group: 2 for group in GROUP_PAGES}
    scheduler = spec._create_group_regions(None)
    worker = {}
    try:
        worker = spec._create_group_regions(1)
        for group, page in GROUP_PAGES.items():
            assert scheduler[group].mmap_path == worker[group].mmap_path
            assert scheduler[group].total_size_bytes == page * 2 * 2
            tensor = worker[group].create_next_view(page)
            tensor[1].fill_(group + 17)
            with scheduler[group].create_kv_memoryview().cast("B") as view:
                assert bytes(view[3 * page : 4 * page]) == bytes([group + 17]) * page
        del tensor
    finally:
        for region in worker.values():
            region.cleanup()
        for region in scheduler.values():
            region.cleanup()

    created: list = []
    real_region = spec_module.SharedOffloadRegion

    def fail_second(**kwargs):
        if created:
            raise RuntimeError("injected group creation failure")
        region = real_region(**kwargs)
        created.append((region, region.mmap_obj))
        return region

    monkeypatch.setattr(spec_module, "SharedOffloadRegion", fail_second)
    with pytest.raises(RuntimeError, match="injected group creation failure"):
        spec._create_group_regions(None)
    region, mapping = created[0]
    assert mapping.closed
    assert region.fd is None
    assert not Path(region.mmap_path).exists()


def test_spec_tier_creation_failure_closes_all_regions(tmp_path, monkeypatch):
    import vllm.v1.kv_offload.tiering.spec as spec_module

    created: list = []
    real_region = spec_module.SharedOffloadRegion
    real_factory = spec_module.SecondaryTierFactory.create_secondary_tier
    calls = 0

    def track_region(**kwargs):
        region = real_region(**kwargs)
        created.append((region, region.mmap_obj))
        return region

    def fail_second_tier(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected secondary creation failure")
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(spec_module, "SharedOffloadRegion", track_region)
    monkeypatch.setattr(
        spec_module.SecondaryTierFactory, "create_secondary_tier", fail_second_tier
    )
    with pytest.raises(RuntimeError, match="injected secondary creation failure"):
        _build(tmp_path)
    assert len(created) == 2
    for region, mapping in created:
        assert mapping.closed
        assert region.fd is None
        assert not Path(region.mmap_path).exists()


@pytest.mark.parametrize(
    "topology", [None, "decode_cp", "backend", "pipeline", "prefill_cp", "nodes"]
)
def test_spec_declares_layout_and_rejects_unsupported_topology(monkeypatch, topology):
    import vllm.v1.kv_offload.tiering.spec as spec_module

    def parent_init(self, config, cache):
        self.extra_config = {}
        self.partition_by_group = True
        self.cpu_group_page_sizes = GROUP_PAGES

    monkeypatch.setattr(spec_module.CPUOffloadingSpec, "__init__", parent_init)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2 if topology == "pipeline" else 1,
            prefill_context_parallel_size=2 if topology == "prefill_cp" else 1,
            # DCP ranks are ordinary single-node workers with their own rows.
            decode_context_parallel_size=2 if topology == "decode_cp" else 1,
            nnodes=2 if topology == "nodes" else 1,
        ),
        attention_config=SimpleNamespace(
            backend=None if topology == "backend" else "FLASH_ATTN_V100"
        ),
        model_config=SimpleNamespace(revision="fixed-model-revision"),
        compute_hash=lambda: "configuration-hash",
    )
    cache = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[],
        kv_cache_tensors=[SimpleNamespace(size=8192, shared_by=["a", "b"])],
    )
    if topology not in (None, "decode_cp"):
        with pytest.raises(ValueError, match="Grouped tiering"):
            spec_module.TieringOffloadingSpec(config, cache)
        return
    spec = spec_module.TieringOffloadingSpec(config, cache)
    layout = spec.persistent_layout
    assert layout["config_hash"] == "configuration-hash"
    assert layout["model_revision"] == "fixed-model-revision"
    assert layout["tensors"] == [{"page_size": 4096, "shared_by": ["a", "b"]}]
    # Changing runtime capacity does not change physical bytes per cache row.
    cache.num_blocks *= 2
    cache.kv_cache_tensors[0].size *= 2
    assert spec_module.TieringOffloadingSpec(config, cache).persistent_layout == layout


def test_scheduler_unlinks_region_created_by_worker():
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    args = dict(
        instance_id=f"scheduler-unlink-test-{uuid.uuid4().hex}",
        total_size_bytes=4096,
        num_blocks=1,
        num_workers=1,
        cpu_page_size=4096,
    )
    worker = SharedOffloadRegion(rank=0, **args)
    scheduler = SharedOffloadRegion(rank=None, **args)
    try:
        assert worker._creator and not scheduler._creator
        scheduler.cleanup()
        assert not Path(worker.mmap_path).exists()
        # Unlink does not invalidate an already open worker mapping.
        assert worker.mmap_obj is not None and not worker.mmap_obj.closed
        worker.cleanup()
        scheduler.cleanup()
    finally:
        worker.cleanup()
        scheduler.cleanup()


def test_tp4_dcp2_demoted_state_survives_cpu_eviction(tmp_path, monkeypatch):
    """All four owner slices, including replicated state bytes, reload exactly."""
    import sys

    module = sys.modules[__name__]
    original_spec = _spec
    monkeypatch.setattr(module, "WORLD_SIZE", 4)

    def dcp_spec():
        spec = original_spec()
        spec.vllm_config.parallel_config.decode_context_parallel_size = 2
        return spec

    monkeypatch.setattr(module, "_spec", dcp_spec)
    manager = _build(tmp_path)
    try:
        keys = [_key(g) for g in GROUP_PAGES]
        output = manager.prepare_store(keys, CTX)
        _write_rows(manager, output, {g: _payload(g) for g in GROUP_PAGES})
        filler = manager.prepare_store([_key(g, 2) for g in GROUP_PAGES], CTX)
        _write_rows(manager, filler, {g: _payload(g) for g in GROUP_PAGES})
        # demote_superseded_states routes here: it changes CPU priority only.
        manager.demote(keys, CTX)
        pressure = manager.prepare_store([_key(g, 3) for g in GROUP_PAGES], CTX)
        assert set(pressure.evicted_keys) == set(keys)
        _write_rows(manager, pressure, {g: _payload(g) for g in GROUP_PAGES})
        assert all(manager.lookup(k, CTX) is None for k in keys)
        _settle(manager)
        assert all(manager.lookup(k, CTX) is True for k in keys)
        loaded = manager.prepare_load(keys, CTX)
        for k, bid in zip(keys, loaded.block_ids):
            group = int.from_bytes(k[-4:], "big")
            primary = manager.managers[group].primary_tier
            with primary.get_kv_memoryview().cast("B") as view:
                size = len(_payload(group))
                assert bytes(view[int(bid) * size : (int(bid) + 1) * size]) == _payload(
                    group
                )
            fields = manager.managers[group].secondary_tiers[0].file_mapper.fields
            assert fields["tp_size"] == 4 and fields["dcp_size"] == 2
        manager.complete_load(keys, CTX)
    finally:
        manager.shutdown()


def test_secondary_enospc_keeps_cpu_copy_and_releases_pins(tmp_path, monkeypatch):
    import errno

    import vllm.v1.kv_offload.tiering.fs.io as fs_io

    def full(*args):
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(fs_io.os, "write", full)
    manager = _build(tmp_path)
    try:
        keys = [_key(g) for g in GROUP_PAGES]
        output = manager.prepare_store(keys, CTX)
        _write_rows(manager, output, {g: _payload(g) for g in GROUP_PAGES})
        assert all(manager.lookup(k, CTX) is True for k in keys)
        # Two rounds must remain evictable despite every secondary write failing.
        for number in (2, 3, 4):
            output = manager.prepare_store([_key(g, number) for g in GROUP_PAGES], CTX)
            assert output is not None
            _write_rows(manager, output, {g: _payload(g) for g in GROUP_PAGES})
        assert all(manager.lookup(k, CTX) is False for k in keys)
    finally:
        manager.shutdown()


def test_fs_budget_applies_across_group_managers(tmp_path):
    pytest.importorskip("vllm.v1.kv_offload.tiering.fs.cache")
    budget = sum(GROUP_PAGES.values()) * WORLD_SIZE
    manager = _build(tmp_path, max_size_bytes=budget)
    try:
        for number in (1, 2, 3):
            keys = [_key(g, number) for g in GROUP_PAGES]
            output = manager.prepare_store(keys, CTX)
            _write_rows(manager, output, {g: _payload(g) for g in GROUP_PAGES})
            assert all(manager.lookup(k, CTX) is True for k in keys)
            files = list(tmp_path.rglob("*.bin"))
            assert 0 < sum(p.stat().st_size for p in files) <= budget
        # Group count must not multiply the configured disk capacity.
        children = list(manager.managers.values())
        assert (
            children[0].secondary_tiers[0]._cache
            is children[1].secondary_tiers[0]._cache
        )
    finally:
        manager.shutdown()


def test_grouped_fs_pages_use_group_page_size(tmp_path):
    """Each group's disk page is that group's own row, not the widest one."""
    pages = {0: 8192, 2: 4096, 9: 2048}
    manager = _build(tmp_path, pages)
    try:
        children = list(manager.managers.values())
        assert children[0].secondary_tiers[0]._block_size == pages[0] * WORLD_SIZE
        assert children[-1].secondary_tiers[0]._block_size == pages[9] * WORLD_SIZE
        keys = [_key(g) for g in pages]
        output = manager.prepare_store(keys, CTX)
        assert output is not None
        payloads = {g: _payload(g, pages) for g in pages}
        _write_rows(manager, output, payloads)
        paths = []
        for group in pages:
            path = Path(
                manager.managers[group]
                .secondary_tiers[0]
                .file_mapper.get_file_name(_key(group))
            )
            # <base>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin keeps its shape.
            assert path.parent.name.endswith(f"_g{group}")
            assert path.parts[-4].endswith("_r0")
            assert path.stat().st_size == pages[group] * WORLD_SIZE
            assert path.read_bytes() == payloads[group]
            paths.append(path)
        assert len({p.parent for p in paths}) == len(pages)
        assert (
            sum(p.stat().st_size for p in tmp_path.rglob("*.bin"))
            == sum(pages.values()) * WORLD_SIZE
        )
    finally:
        manager.shutdown()


def test_heterogeneous_group_regions_have_distinct_geometries():
    """Per-group slot counts size the shared mappings, not one global count."""
    import vllm.v1.kv_offload.tiering.spec as spec_module

    pages = {0: 8192, 2: 4096, 9: 2048}
    # A 832-token drafter group needs twice the slots of the 1,664-token groups
    # to cover the same tokens, at a quarter of the target page.
    slots = {0: 4, 2: 4, 9: 8}
    spans = {0: 1664, 2: 1664, 9: 832}
    spec = object.__new__(spec_module.TieringOffloadingSpec)
    spec.vllm_config = SimpleNamespace(
        instance_id=f"grouped-geometry-test-{uuid.uuid4().hex}",
        parallel_config=SimpleNamespace(world_size=WORLD_SIZE),
    )
    spec.cpu_group_page_sizes = pages
    spec.cpu_group_num_blocks = slots
    spec.cpu_group_offloaded_tokens = spans
    spec.num_blocks = 4
    regions = spec._create_group_regions(None)
    try:
        for group, page in pages.items():
            assert regions[group].num_blocks == slots[group]
            assert regions[group].total_size_bytes == page * WORLD_SIZE * slots[group]
        assert {g: r.total_size_bytes for g, r in regions.items()} == {
            0: 65536,
            2: 32768,
            9: 32768,
        }
        assert regions[9].total_size_bytes < regions[0].total_size_bytes
    finally:
        for region in regions.values():
            region.cleanup()


def _tiering_spec(monkeypatch, group_tokens, pages):
    """A TieringOffloadingSpec whose parent resolved `group_tokens` per group."""
    import vllm.v1.kv_offload.tiering.spec as spec_module

    def parent_init(self, config, cache):
        self.extra_config = {}
        self.partition_by_group = True
        self.cpu_group_page_sizes = pages
        self.cpu_group_offloaded_tokens = group_tokens

    monkeypatch.setattr(spec_module.CPUOffloadingSpec, "__init__", parent_init)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            nnodes=1,
        ),
        attention_config=SimpleNamespace(backend="FLASH_ATTN_V100"),
        model_config=SimpleNamespace(revision="fixed-model-revision"),
        compute_hash=lambda: "configuration-hash",
    )
    cache = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[],
        kv_cache_tensors=[SimpleNamespace(size=8192, shared_by=["a", "b"])],
    )
    return spec_module.TieringOffloadingSpec(config, cache)


MAPPER_ARGS = dict(
    root_dir="/tmp/cache",
    model_name="test/model",
    hash_block_size=16,
    gpu_blocks_per_file=1,
    tp_size=2,
    pp_size=1,
    pcp_size=1,
    dcp_size=1,
    rank=1,
    dtype="float16",
)

# The dict TieringOffloadingSpec wrote for equal-span grouped pools so far.
LEGACY_V1_LAYOUT = {
    "version": "grouped-worker-interleaved-v1",
    "config_hash": "configuration-hash",
    "attention_backend": "FLASH_ATTN_V100",
    "model_revision": "fixed-model-revision",
    "group_pages": {0: 8192, 2: 4096, 5: 512},
    "tensors": [{"page_size": 4096, "shared_by": ["a", "b"]}],
}


def test_mtp4_persistent_layout_unchanged(monkeypatch):
    from vllm.v1.kv_offload.file_mapper import FileMapper

    pages = LEGACY_V1_LAYOUT["group_pages"]
    spec = _tiering_spec(monkeypatch, {0: 784, 2: 784, 5: 784}, pages)
    layout = spec.persistent_layout
    assert layout == LEGACY_V1_LAYOUT
    assert set(layout) == {
        "version",
        "config_hash",
        "attention_backend",
        "model_revision",
        "group_pages",
        "tensors",
    }
    assert layout["version"] == "grouped-worker-interleaved-v1"
    # Same base_path and the same group suffixes, so existing MTP4 disk roots
    # keep being reused after the upgrade.
    key = make_offload_key(bytes(range(8)), 2)
    assert FileMapper(**MAPPER_ARGS, persistent_layout=layout).get_file_name(
        key
    ) == FileMapper(
        **MAPPER_ARGS, persistent_layout=dict(LEGACY_V1_LAYOUT)
    ).get_file_name(key)


def test_heterogeneous_persistent_layout_isolates_namespace(monkeypatch):
    from copy import deepcopy

    from vllm.v1.kv_offload.file_mapper import FileMapper

    pages = {0: 8192, 2: 4096, 9: 2048}
    tokens = {0: 1664, 2: 1664, 9: 832}
    layout = _tiering_spec(monkeypatch, tokens, pages).persistent_layout
    assert layout["version"] == "grouped-worker-interleaved-v2"
    assert layout["group_tokens"] == tokens
    equal = _tiering_spec(monkeypatch, dict.fromkeys(tokens, 1664), pages)
    assert "group_tokens" not in equal.persistent_layout
    legacy = deepcopy(LEGACY_V1_LAYOUT)
    legacy["group_pages"] = pages
    mapper = FileMapper(**MAPPER_ARGS, persistent_layout=layout)
    assert mapper.base_path != (
        FileMapper(**MAPPER_ARGS, persistent_layout=equal.persistent_layout).base_path
    )
    assert mapper.base_path != (
        FileMapper(**MAPPER_ARGS, persistent_layout=legacy).base_path
    )
    # Path shape is unchanged: only the namespace directory and page bytes move.
    path = mapper.get_file_name(make_offload_key(bytes(range(8)), 9))
    assert path.startswith(mapper.base_path + "_r1/")
    assert path.endswith("/000/10_g9/0001020304050607.bin")


def test_non_grouped_tier_keeps_row_and_legacy_namespace(monkeypatch):
    """Without a partition: one row per key, no persistent_layout, one manager."""
    import vllm.v1.kv_offload.tiering.spec as spec_module

    def parent_init(self, config, cache):
        self.vllm_config = config
        self.kv_cache_config = cache
        self.extra_config = {}
        self.partition_by_group = False
        self.cpu_group_page_sizes = {}
        self.cpu_group_num_blocks = {}
        self.cpu_group_offloaded_tokens = {}
        self.cpu_page_size_per_worker = 64
        self.num_blocks = 4
        self.eviction_policy = "lru"

    monkeypatch.setattr(spec_module.CPUOffloadingSpec, "__init__", parent_init)
    config = SimpleNamespace(
        instance_id=f"legacy-tier-test-{uuid.uuid4().hex}",
        parallel_config=SimpleNamespace(
            world_size=WORLD_SIZE,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            nnodes=1,
        ),
        attention_config=SimpleNamespace(backend="FLASH_ATTN_V100"),
        model_config=SimpleNamespace(revision="fixed-model-revision"),
        compute_hash=lambda: "configuration-hash",
        kv_events_config=None,
    )
    cache = SimpleNamespace(num_blocks=2, kv_cache_groups=[], kv_cache_tensors=[])
    spec = spec_module.TieringOffloadingSpec(config, cache)
    assert spec.persistent_layout is None
    manager = spec.get_manager()
    try:
        assert isinstance(manager, spec_module.TieringOffloadingManager)
        view = manager.primary_tier.get_kv_memoryview()
        # Every group, however wide its own block, addresses the same row.
        assert view.strides[0] == 64 * WORLD_SIZE
        assert view.shape[0] == 4
    finally:
        manager.shutdown()
        spec._manager = None
        assert spec._scheduler_mmap is not None
        spec._scheduler_mmap.cleanup()
