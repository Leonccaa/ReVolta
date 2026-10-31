# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    OffloadingSpec,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.gpu_worker import partition_kv_caches
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)
from vllm.v1.kv_offload.cpu.spec import (
    CPUOffloadingSpec,
    group_slot_counts,
    mamba_state_slots,
)

CTX = ReqContext("test")
GROUPS = (0, 2, 3, 4, 5)


@pytest.mark.parametrize("error_code", [1, 2])
def test_mmap_registration_failure_is_fatal(monkeypatch, tmp_path, error_code):
    from vllm.v1.kv_offload.cpu import gpu_worker as worker

    backing = (tmp_path / "offload.mmap").open("w+b")
    region = SimpleNamespace(
        fd=backing.fileno(),
        rank=3,
        mmap_path="/dev/shm/test-offload.mmap",
        total_size_bytes=4096,
        _base=torch.zeros(4096, dtype=torch.int8),
        is_pinned=False,
    )
    calls = []

    def register(ptr, size, flags):
        calls.append((ptr, size, flags))
        return SimpleNamespace(value=error_code)

    monkeypatch.setattr(
        torch.cuda, "cudart", lambda: SimpleNamespace(cudaHostRegister=register)
    )
    with pytest.raises(RuntimeError, match="KV batch transfers require pinned memory"):
        worker.pin_mmap_region(region)
    assert calls == [(region._base.data_ptr(), 4096, 0)]
    assert not region.is_pinned
    backing.close()


@pytest.mark.parametrize("raises", [False, True])
def test_mmap_registration_holds_and_releases_shared_file_lock(
    monkeypatch, tmp_path, raises
):
    from vllm.v1.kv_offload.cpu import gpu_worker as worker

    path = tmp_path / "offload.mmap"
    with path.open("w+b") as backing, path.open("r+b") as contender:
        region = SimpleNamespace(
            fd=backing.fileno(),
            rank=0,
            mmap_path=str(path),
            total_size_bytes=4096,
            _base=torch.zeros(4096, dtype=torch.int8),
            is_pinned=False,
        )

        def register(ptr, size, flags):
            # Independently opened descriptors contend even in one process.
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if raises:
                raise RuntimeError("registration raised")
            return SimpleNamespace(value=0)

        monkeypatch.setattr(
            torch.cuda, "cudart", lambda: SimpleNamespace(cudaHostRegister=register)
        )
        if raises:
            with pytest.raises(RuntimeError, match="registration raised"):
                worker.pin_mmap_region(region)
        else:
            worker.pin_mmap_region(region)
        assert region.is_pinned is not raises
        # Both successful and exceptional registration must release the lock.
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender.fileno(), fcntl.LOCK_UN)


def key(group, index):
    return make_offload_key(index.to_bytes(8, "big"), group)


def manager(capacity=2, policy="lru"):
    return GroupedCPUOffloadingManager(
        {
            g: CPUOffloadingManager(capacity, cache_policy=policy, enable_events=True)
            for g in GROUPS
        }
    )


def store(m, keys):
    out = m.prepare_store(keys, CTX)
    assert out is not None
    m.complete_store(out.keys_to_store, CTX)
    return out


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_large_context_capacity(policy):
    # Measured Flash-Next geometry: 16 GiB buys 419 all-tensor slots, or
    # 107 slots per group when each key pays for only its own tensors.
    old = CPUOffloadingManager(419, cache_policy=policy)
    new = manager(107, policy)
    a = [key(g, i) for g in GROUPS for i in range(48)]
    b = [key(g, i + 100) for g in GROUPS for i in range(48)]
    for m in (old, new):
        store(m, a)
        store(m, b)
    assert not all(old.lookup(k, CTX) for k in a)
    assert all(new.lookup(k, CTX) for k in a + b)
    assert list(new.prepare_load(a, CTX).block_ids) == list(range(48)) * 5
    new.complete_load(a, CTX)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_group_pool_lifecycle_and_order(policy):
    m = manager(policy=policy)
    keys = [key(5, 1), key(0, 2), key(5, 3)]
    out = m.prepare_store(keys, CTX)
    assert out.keys_to_store == keys
    assert list(out.store_spec.block_ids) == [0, 0, 1]
    assert m.lookup(keys[0], CTX) is None
    m.complete_store(keys, CTX)
    assert list(m.prepare_load(keys, CTX).block_ids) == [0, 0, 1]
    # A blocked group defers the entire operation, releasing other reservations.
    partial = m.prepare_store([key(5, 4), key(0, 4)], CTX)
    assert partial is None
    assert m.lookup(key(0, 4), CTX) is False
    m.complete_load(keys, CTX)
    m.touch([keys[0]], CTX)
    out = store(m, [key(5, 4)])
    assert out.evicted_keys == [keys[2]]
    # Eviction recycles the old slot within the bounded group pool.
    assert list(out.store_spec.block_ids) == [1]
    assert m.lookup(keys[0], CTX) is True
    assert m.lookup(keys[2], CTX) is False
    assert m.lookup(keys[1], CTX) is True
    assert list(m.take_events())
    assert not list(m.take_events())
    m.reset_cache()
    assert all(m.lookup(k, CTX) is False for k in keys)
    assert list(store(m, [keys[0]]).store_spec.block_ids) == [0]


def test_group_views_preserve_gpu_aliases_but_separate_cpu_indices():
    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor],
        [
            [CanonicalKVCacheRef(0, 16)],
            [CanonicalKVCacheRef(0, 16)],
            [CanonicalKVCacheRef(0, 8)],
        ],
    )
    split = partition_kv_caches(caches, {0: 32, 2: 32}, 2)
    assert split.tensors[0].tensor is split.tensors[1].tensor
    assert split.group_data_refs[0][0].tensor_idx == 0
    assert split.group_data_refs[1] == []
    assert split.group_data_refs[2][0].tensor_idx == 1
    assert split.group_data_refs[2][0].page_size_bytes == 8
    with pytest.raises(AssertionError, match="byte budget"):
        partition_kv_caches(caches, {0: 31, 2: 32}, 2)


FLASH_NEXT_PAGES = {
    0: 10235904,
    2: 9633792,
    3: 9633792,
    4: 9633792,
    5: 802816,
}


def _flash_next_spec(
    monkeypatch, layout="equal", retention_interval=None, **extra_config
):
    def init(self, config, caches):
        self.vllm_config = config
        self.kv_cache_config = caches
        self.extra_config = {"cpu_bytes_to_use": 16 * 1024**3, **extra_config}
        self.block_size_factor = 1
        self.gpu_block_size = tuple(
            getattr(g.kv_cache_spec, "block_size", 0) for g in caches.kv_cache_groups
        )

    monkeypatch.setattr(OffloadingSpec, "__init__", init)
    fa = FullAttentionSpec(
        block_size=784, num_kv_heads=1, head_size=256, dtype=torch.float16
    )
    small = FullAttentionSpec(
        block_size=784, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    uniform = UniformTypeKVCacheSpecs(
        block_size=784,
        kv_cache_specs={
            **{f"fa{i}": fa for i in range(12)},
            **{f"index{i}": small for i in range(12)},
        },
    )
    mamba = MambaSpec(
        block_size=784,
        shapes=((1,),),
        dtypes=(torch.float16,),
        page_size_padded=802816,
        mamba_cache_mode="align",
    )
    group = lambda spec, count: SimpleNamespace(
        kv_cache_spec=spec, layer_names=list(range(count))
    )
    groups = [group(uniform, 24), group(SimpleNamespace(prefix_cacheable=False), 12)]
    groups += [group(mamba, n) for n in (12, 12, 12, 1)]
    groups[0].layer_names = list(uniform.kv_cache_specs)
    for g in (2, 3, 4):
        groups[g].layer_names = [f"m{g}_{i}" for i in range(12)]
    groups[5].layer_names = ["ple"]
    tensors = [
        SimpleNamespace(
            size=104 * 802816,
            shared_by=[f"fa{i}", f"m2_{i}", f"m3_{i}", f"m4_{i}"]
            + (["ple"] if i == 0 else []),
        )
        for i in range(12)
    ] + [SimpleNamespace(size=104 * 50176, shared_by=[f"index{i}"]) for i in range(12)]
    if layout == "scheduler":
        groups[0].kv_cache_spec = fa
    if layout == "mixed":
        groups[2] = group(replace(mamba, block_size=392), 12)
    elif layout == "attention_only":
        groups = [group(uniform, 24), group(uniform, 24)]
    return CPUOffloadingSpec(
        SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=4),
            model_config=SimpleNamespace(max_model_len=65536),
            cache_config=SimpleNamespace(
                prefix_cache_retention_interval=retention_interval
            ),
        ),
        SimpleNamespace(
            kv_cache_groups=groups,
            num_blocks=104,
            kv_cache_tensors=tensors,
        ),
    )


@pytest.mark.parametrize("layout", ["equal", "scheduler", "mixed", "attention_only"])
def test_actual_flash_next_group_budget(monkeypatch, layout):
    if layout == "mixed":
        # Unequal recurrent spans cannot share one state slot count, and the
        # spec no longer sizes every state pool from the last spec it saw.
        with pytest.raises(ValueError, match="state slot geometry"):
            _flash_next_spec(monkeypatch, layout)
        return
    spec = _flash_next_spec(monkeypatch, layout)
    if layout not in ("equal", "scheduler"):
        assert not spec.partition_by_group
        assert spec.num_blocks == 419
        assert spec.cpu_group_num_blocks == {}
        return
    assert spec.cpu_group_page_sizes == FLASH_NEXT_PAGES
    # Dense retention (None): one state slot per token slot, 107 each.
    assert spec.num_blocks == 107
    assert spec.retention_interval is None
    assert spec.cpu_group_num_blocks == dict.fromkeys(FLASH_NEXT_PAGES, 107)
    assert spec.cpu_page_size_per_worker * 4 * spec.num_blocks <= 16 * 1024**3
    assert spec.num_blocks >= 2 * 48


def _grouped_bytes(spec):
    return 4 * sum(
        spec.cpu_group_page_sizes[g] * n for g, n in spec.cpu_group_num_blocks.items()
    )


# DFlash2 geometry, family-A decomposition of the measured 25,559,040 B row:
# 10 main-KV owner pages (425,984 B) plus 10 compressed pages (212,992 B) per
# worker per 1,664-token target block = 6,389,760 B. The 832-token drafter
# shares 5 owner pages, each of six GDN groups owns 9 pages, the PLE group 1.
DFLASH2_OWNER_PAGE = 425_984
DFLASH2_COMPRESSED_PAGE = 212_992
DFLASH2_ROW = 10 * (DFLASH2_OWNER_PAGE + DFLASH2_COMPRESSED_PAGE)
DFLASH2_TARGET_SPAN = 1664
DFLASH2_DRAFTER_SPAN = 832
DFLASH2_GDN_LAYERS = 9
DFLASH2_RECURRENT = tuple(range(2, 9))
DFLASH2_PAGES = {
    0: DFLASH2_ROW,
    **{g: DFLASH2_GDN_LAYERS * DFLASH2_OWNER_PAGE for g in range(2, 8)},
    8: DFLASH2_OWNER_PAGE,
    9: 5 * DFLASH2_OWNER_PAGE,
}
DFLASH2_BYTES = 16 * 1024**3
WORLD_SIZE = 4


def _dflash2_spec(
    monkeypatch,
    *,
    drafter_span: int = DFLASH2_DRAFTER_SPAN,
    mamba_span: int = DFLASH2_TARGET_SPAN,
    ple_span: int | None = None,
    keep: tuple[int, ...] | None = None,
    num_gpu_blocks: int = 104,
    retention_interval: int | None = None,
    cpu_bytes: int = DFLASH2_BYTES,
    max_model_len: int = 65536,
    **extra_config,
):
    """A CSA+linear-like cache config whose drafter block is a target divisor."""

    def init(self, config, caches):
        self.vllm_config = config
        self.kv_cache_config = caches
        self.extra_config = {"cpu_bytes_to_use": cpu_bytes, **extra_config}
        self.block_size_factor = 1
        self.gpu_block_size = tuple(
            getattr(g.kv_cache_spec, "block_size", 0) for g in caches.kv_cache_groups
        )

    monkeypatch.setattr(OffloadingSpec, "__init__", init)
    ple_span = mamba_span if ple_span is None else ple_span
    main = FullAttentionSpec(
        block_size=DFLASH2_TARGET_SPAN, num_kv_heads=1, head_size=256, dtype=torch.half
    )
    compressed = FullAttentionSpec(
        block_size=DFLASH2_TARGET_SPAN, num_kv_heads=1, head_size=16, dtype=torch.half
    )
    target = UniformTypeKVCacheSpecs(
        block_size=DFLASH2_TARGET_SPAN,
        kv_cache_specs={
            **{f"fa{i}": main for i in range(10)},
            **{f"c{i}": compressed for i in range(10)},
        },
    )
    state = dict(shapes=((1, 8),), dtypes=(torch.float32,), mamba_cache_mode="align")
    gdn = MambaSpec(
        block_size=mamba_span,
        page_size_padded=DFLASH2_OWNER_PAGE,
        **state,
    )
    ple = MambaSpec(
        block_size=ple_span,
        page_size_padded=DFLASH2_OWNER_PAGE,
        tp_replicated=True,
        **state,
    )
    drafter = SlidingWindowSpec(
        block_size=drafter_span,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.half,
        sliding_window=2048,
        page_size_padded=DFLASH2_OWNER_PAGE,
        dcp_sharded=False,
    )
    group = lambda spec, names: SimpleNamespace(  # noqa: E731
        kv_cache_spec=spec, layer_names=list(names)
    )
    all_groups = {
        0: group(
            target,
            [f"fa{i}" for i in range(10)] + [f"c{i}" for i in range(10)],
        ),
        # The compressor-state ring is not prefix-cacheable: no offload keys.
        1: group(
            SimpleNamespace(prefix_cacheable=False), [f"ring{i}" for i in range(10)]
        ),
        **{
            g: group(gdn, [f"m{g}_{i}" for i in range(DFLASH2_GDN_LAYERS)])
            for g in range(2, 8)
        },
        8: group(ple, ["ple"]),
        9: group(drafter, [f"d{i}" for i in range(5)]),
    }
    indices = sorted(all_groups) if keep is None else list(keep)
    groups = [all_groups[i] for i in indices]
    tensors = [
        SimpleNamespace(
            size=num_gpu_blocks * DFLASH2_OWNER_PAGE,
            shared_by=[f"fa{i}"]
            + [f"m{g}_{i}" for g in range(2, 8) if i < DFLASH2_GDN_LAYERS]
            + (["ple"] if i == 0 else [])
            + ([f"d{i}"] if i < 5 else []),
        )
        for i in range(10)
    ] + [
        SimpleNamespace(
            size=num_gpu_blocks * DFLASH2_COMPRESSED_PAGE,
            shared_by=[f"c{i}", f"ring{i}"],
        )
        for i in range(10)
    ]
    return CPUOffloadingSpec(
        SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=WORLD_SIZE),
            model_config=SimpleNamespace(max_model_len=max_model_len),
            cache_config=SimpleNamespace(
                prefix_cache_retention_interval=retention_interval
            ),
        ),
        SimpleNamespace(
            kv_cache_groups=groups,
            num_blocks=num_gpu_blocks,
            kv_cache_tensors=tensors,
        ),
    )


def _legacy_grouped_counts(
    pages: dict[int, int],
    recurrent: set[int],
    last_mamba_spec: MambaSpec,
    budget_per_worker: int,
    retention_interval: int | None,
    reference_tokens: int,
) -> tuple[int, dict[int, int]]:
    """Sizing of `CPUOffloadingSpec` before group spans could differ.

    One shared ``num_blocks``, one state slot count for every recurrent group
    taken from the last ``MambaSpec`` seen.
    """
    token_page = sum(p for g, p in pages.items() if g not in recurrent)
    state_page = sum(pages[g] for g in recurrent)
    block_size = last_mamba_spec.block_size

    def state_slots(num_token_blocks: int) -> int:
        return mamba_state_slots(
            num_token_blocks,
            block_size,
            block_size,
            last_mamba_spec,
            retention_interval,
            reference_tokens,
        )

    def bytes_per_worker(num_token_blocks: int) -> int:
        return (
            num_token_blocks * token_page + state_slots(num_token_blocks) * state_page
        )

    num_blocks = budget_per_worker // (token_page + state_page)
    while bytes_per_worker(num_blocks + 1) <= budget_per_worker:
        num_blocks += 1
    while num_blocks > 0 and bytes_per_worker(num_blocks) > budget_per_worker:
        num_blocks -= 1
    return num_blocks, {
        g: state_slots(num_blocks) if g in recurrent else num_blocks for g in pages
    }


def test_partition_gate_allows_divisor_block_sizes(monkeypatch):
    # A drafter block that is a divisor of the target block now partitions.
    spec = _dflash2_spec(monkeypatch)
    assert spec.partition_by_group
    assert spec.cpu_group_offloaded_tokens == {
        **dict.fromkeys([0, *DFLASH2_RECURRENT], DFLASH2_TARGET_SPAN),
        9: DFLASH2_DRAFTER_SPAN,
    }
    # Single cacheable group: keep the historical single pool.
    assert not _dflash2_spec(monkeypatch, keep=(0, 1)).partition_by_group
    # No recurrent group: "auto" keeps the row layout, "force" opts in.
    assert not _dflash2_spec(monkeypatch, keep=(0, 1, 9)).partition_by_group
    assert _dflash2_spec(
        monkeypatch, keep=(0, 1, 9), cpu_group_partition="force"
    ).partition_by_group
    # No GPU blocks at all: nothing to size.
    assert not _dflash2_spec(monkeypatch, num_gpu_blocks=0).partition_by_group
    # Explicit rollback reproduces the row-per-key layout.
    assert not _dflash2_spec(monkeypatch, cpu_group_partition="off").partition_by_group
    with pytest.raises(ValueError, match="cpu_group_partition"):
        _dflash2_spec(monkeypatch, cpu_group_partition="yes")


def test_dflash2_group_pages_are_group_bytes(monkeypatch):
    spec = _dflash2_spec(monkeypatch)
    assert spec.partition_by_group
    assert spec.cpu_group_page_sizes == DFLASH2_PAGES
    # The target group references every tensor of the pool, so its page stays
    # the whole GPU row; the drafter pays 5 owner pages for 832 tokens and each
    # state group only its own pages. The ring's layers charge nothing.
    assert DFLASH2_PAGES[0] == DFLASH2_ROW
    assert DFLASH2_PAGES[9] == 5 * DFLASH2_OWNER_PAGE < DFLASH2_ROW
    assert spec.cpu_page_size_per_worker == sum(DFLASH2_PAGES.values())


def test_dflash2_slots_balance_token_capacity(monkeypatch):
    spec = _dflash2_spec(monkeypatch)
    slots = spec.cpu_group_num_blocks
    spans = spec.cpu_group_offloaded_tokens
    assert spec.num_blocks == slots[0] == slots[8]
    assert slots[9] == 2 * slots[0]
    # Every group covers exactly the tokens of the token pool.
    coverage = {g: n * spans[g] for g, n in slots.items()}
    assert set(coverage.values()) == {spec.num_blocks * DFLASH2_TARGET_SPAN}
    assert _grouped_bytes(spec) <= DFLASH2_BYTES
    # The solve is tight: one more token slot per group, or one more block in
    # the search, exceeds the budget.
    grown = dict(slots)
    grown[9] += 1
    assert 4 * sum(DFLASH2_PAGES[g] * n for g, n in grown.items()) > DFLASH2_BYTES
    assert (
        4
        * sum(
            DFLASH2_PAGES[g] * n
            for g, n in group_slot_counts(
                spans, set(DFLASH2_RECURRENT), spec.num_blocks + 1, DFLASH2_TARGET_SPAN
            ).items()
        )
        > DFLASH2_BYTES
    )
    # Token-proportional cost of the family-A decomposition (design section 3).
    per_token = 4 * (
        DFLASH2_PAGES[0] / DFLASH2_TARGET_SPAN + DFLASH2_PAGES[9] / DFLASH2_DRAFTER_SPAN
    )
    assert abs(per_token - 25_600) / 25_600 < 0.01


@pytest.mark.parametrize("retention_interval", [None, 0])
def test_equal_span_sizing_matches_legacy_grouped_layout(
    monkeypatch, retention_interval
):
    # With one shared span the per-group solve degenerates to the single
    # num_blocks the grouped layout used before, per group and for MTP4.
    spec = _dflash2_spec(
        monkeypatch,
        drafter_span=DFLASH2_TARGET_SPAN,
        retention_interval=retention_interval,
    )
    assert spec.partition_by_group
    assert spec.cpu_group_offloaded_tokens == dict.fromkeys(
        DFLASH2_PAGES, DFLASH2_TARGET_SPAN
    )
    legacy_blocks, legacy_counts = _legacy_grouped_counts(
        spec.cpu_group_page_sizes,
        set(DFLASH2_RECURRENT),
        spec.kv_cache_config.kv_cache_groups[8].kv_cache_spec,
        DFLASH2_BYTES // WORLD_SIZE,
        retention_interval,
        65536,
    )
    assert spec.num_blocks == legacy_blocks
    assert spec.cpu_group_num_blocks == legacy_counts
    if retention_interval is None:
        assert set(spec.cpu_group_num_blocks.values()) == {legacy_blocks}
    assert _grouped_bytes(spec) <= DFLASH2_BYTES
    # The measured Flash-Next (MTP4) fixture keeps its 107 dense slots too.
    flash = _flash_next_spec(monkeypatch, retention_interval=retention_interval)
    flash_legacy_blocks, flash_legacy_counts = _legacy_grouped_counts(
        flash.cpu_group_page_sizes,
        {2, 3, 4, 5},
        flash.kv_cache_config.kv_cache_groups[5].kv_cache_spec,
        DFLASH2_BYTES // WORLD_SIZE,
        retention_interval,
        65536,
    )
    assert flash.num_blocks == flash_legacy_blocks
    assert flash.cpu_group_num_blocks == flash_legacy_counts
    if retention_interval is None:
        assert flash.cpu_group_num_blocks == dict.fromkeys(FLASH_NEXT_PAGES, 107)


def test_group_slot_counts_covers_equal_tokens_and_keeps_equal_spans():
    spans = {0: 1664, 2: 1664, 9: 832}
    counts = group_slot_counts(spans, {2}, 4, 1664)
    assert counts == {0: 4, 2: 4, 9: 8}
    # Equal spans reproduce the legacy single count exactly.
    assert group_slot_counts(dict.fromkeys(spans, 1664), {2}, 4, 1664) == {
        0: 4,
        2: 4,
        9: 4,
    }
    # A recurrent group hands its own count to the state-slot function.
    assert group_slot_counts(
        spans, {2}, 4, 1664, lambda group, blocks: min(blocks, 3)
    ) == {0: 4, 2: 3, 9: 8}
    # Never zero slots, even when the budget cannot afford one block.
    assert group_slot_counts({0: 3200, 1: 1664}, set(), 1, 1664) == {0: 1, 1: 1}
    with pytest.raises(ValueError, match="reference_span"):
        group_slot_counts(spans, set(), 4, 0)
    with pytest.raises(ValueError, match="non-positive token span"):
        group_slot_counts({0: 0}, set(), 4, 1664)


def test_recurrent_span_mismatch_rejected(monkeypatch):
    # A PLE group at another span cannot share the GDN state slot count.
    with pytest.raises(ValueError, match="state slot geometry"):
        _dflash2_spec(monkeypatch, ple_span=DFLASH2_DRAFTER_SPAN)


def test_partition_off_knob_keeps_row_layout(monkeypatch):
    spec = _dflash2_spec(monkeypatch, cpu_group_partition="off")
    assert not spec.partition_by_group
    assert spec.cpu_group_page_sizes == {}
    assert spec.cpu_group_num_blocks == {}
    assert spec.cpu_group_offloaded_tokens == {}
    # One row per key for every group, exactly as before the partition.
    assert spec.cpu_page_size_per_worker == DFLASH2_ROW
    assert spec.num_blocks == DFLASH2_BYTES // (DFLASH2_ROW * WORLD_SIZE) == 672


def test_heterogeneous_span_with_explicit_block_size_rejected():
    # `block_size` is a single global factor, so it still needs one cacheable
    # block size; the pre-existing OffloadingSpec guard must surface here.
    main = FullAttentionSpec(
        block_size=DFLASH2_TARGET_SPAN, num_kv_heads=1, head_size=256, dtype=torch.half
    )
    state = MambaSpec(
        block_size=DFLASH2_TARGET_SPAN,
        shapes=((1, 8),),
        dtypes=(torch.float32,),
        page_size_padded=DFLASH2_OWNER_PAGE,
        mamba_cache_mode="align",
    )
    drafter = SlidingWindowSpec(
        block_size=DFLASH2_DRAFTER_SPAN,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.half,
        sliding_window=2048,
        page_size_padded=DFLASH2_OWNER_PAGE,
        dcp_sharded=False,
    )
    group = lambda spec, names: SimpleNamespace(  # noqa: E731
        kv_cache_spec=spec, layer_names=list(names)
    )
    caches = SimpleNamespace(
        kv_cache_groups=[
            group(main, ["fa0"]),
            group(state, ["m0"]),
            group(drafter, ["d0"]),
        ],
        num_blocks=16,
        kv_cache_tensors=[
            SimpleNamespace(size=16 * DFLASH2_OWNER_PAGE, shared_by=["fa0", "m0", "d0"])
        ],
    )

    def config(extra):
        return SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
            parallel_config=SimpleNamespace(
                world_size=WORLD_SIZE,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            cache_config=SimpleNamespace(
                block_size=16,
                hash_block_size=None,
                enable_prefix_caching=True,
                prefix_cache_retention_interval=None,
            ),
            model_config=SimpleNamespace(max_model_len=65536),
        )

    base = {"cpu_bytes_to_use": DFLASH2_BYTES}
    spec = CPUOffloadingSpec(config(dict(base)), deepcopy(caches))
    assert spec.partition_by_group
    assert spec.cpu_group_offloaded_tokens == {
        0: DFLASH2_TARGET_SPAN,
        1: DFLASH2_TARGET_SPAN,
        2: DFLASH2_DRAFTER_SPAN,
    }
    with pytest.raises(AssertionError, match="same block size"):
        CPUOffloadingSpec(config({**base, "block_size": 3328}), deepcopy(caches))


def test_group_slots_scaled_per_group_span(monkeypatch):
    """Per-group slot counts reach the private CPU tensors unchanged."""
    import vllm.v1.kv_offload.cpu.gpu_worker as worker

    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        worker, "SingleDirectionOffloadingHandler", lambda **kw: SimpleNamespace(**kw)
    )
    caches = CanonicalKVCaches(
        [
            CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16),
            CanonicalKVCacheTensor(torch.zeros((4, 8), dtype=torch.int8), 8),
        ],
        [
            [CanonicalKVCacheRef(0, 16)],
            [],
            [CanonicalKVCacheRef(0, 16)],
            [],
            [],
            [],
            [],
            [],
            [],
            [CanonicalKVCacheRef(1, 8)],
        ],
    )
    handlers = worker.CpuGpuOffloadingHandlers(
        caches,
        2,
        4,
        group_page_sizes={0: 32, 2: 32, 9: 16},
        group_num_blocks={0: 4, 2: 1, 9: 8},
    )
    cpu = handlers.gpu_to_cpu_handler.cpu_tensors
    # The narrow-span group holds twice the slots of the wide one at half the
    # page, i.e. the same number of tokens per byte-free layout change.
    assert [tuple(t.shape) for t in cpu] == [(4, 32), (1, 32), (8, 16)]


def test_flash_next_semantic_retention_budget(monkeypatch):
    # retention 0 keeps only the replay boundary (+ one junction allowance)
    # per request. Sized for 64K requests (84 blocks): 2 states per request.
    spec = _flash_next_spec(monkeypatch, retention_interval=0)
    assert spec.partition_by_group
    assert spec.retention_interval == 0
    assert spec.state_slots_reference_tokens == 65536
    token_slots = spec.num_blocks
    requests = -(-token_slots // 84)
    state_slots = requests * 2
    assert spec.cpu_group_num_blocks == {
        0: token_slots,
        2: state_slots,
        3: state_slots,
        4: state_slots,
        5: state_slots,
    }
    # Nearly the whole 16 GiB now buys token pages: > 3.5x the dense layout.
    assert token_slots >= 380
    assert _grouped_bytes(spec) <= 16 * 1024**3
    spec.cpu_group_num_blocks[0] += 1
    assert _grouped_bytes(spec) > 16 * 1024**3


def test_flash_next_periodic_retention_budget(monkeypatch):
    # retention 8 blocks: periodic states every 8 blocks plus replay/junction.
    spec = _flash_next_spec(monkeypatch, retention_interval=8 * 784)
    token_slots = spec.num_blocks
    # 84-block reference request: boundaries at blocks 8,16,...,80 (10),
    # the replay boundary at block 81 (index 80 already counted? no: replay
    # boundary 65535 -> block 83 -> index 82) plus the junction allowance.
    per_request = mamba_state_slots(
        84,
        784,
        784,
        spec.kv_cache_config.kv_cache_groups[2].kv_cache_spec,
        8 * 784,
        65536,
    )
    assert per_request == 10 + 1 + 1
    assert spec.cpu_group_num_blocks[2] == min(
        token_slots, -(-token_slots // 84) * per_request
    )
    assert token_slots > 107
    assert _grouped_bytes(spec) <= 16 * 1024**3


def test_state_slots_follow_reference_length(monkeypatch):
    # A shorter sizing reference reserves more state slots per token slot.
    long = _flash_next_spec(monkeypatch, retention_interval=0)
    short = _flash_next_spec(
        monkeypatch,
        retention_interval=0,
        mamba_state_slots_reference_tokens=8 * 784,
    )
    assert short.cpu_group_num_blocks[2] > long.cpu_group_num_blocks[2]
    assert short.num_blocks < long.num_blocks
    assert _grouped_bytes(short) <= 16 * 1024**3
    with pytest.raises(ValueError, match="reference_tokens"):
        _flash_next_spec(
            monkeypatch, retention_interval=0, mamba_state_slots_reference_tokens=0
        )


def test_private_cpu_allocations_obey_group_budget(monkeypatch):
    import vllm.v1.kv_offload.cpu.gpu_worker as worker

    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        worker, "SingleDirectionOffloadingHandler", lambda **kw: SimpleNamespace(**kw)
    )
    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor],
        [
            [CanonicalKVCacheRef(0, 16)],
            [CanonicalKVCacheRef(0, 16)],
            [CanonicalKVCacheRef(0, 8)],
        ],
    )
    handlers = worker.CpuGpuOffloadingHandlers(
        caches, 2, 2, group_page_sizes={0: 32, 2: 32}
    )
    cpu = handlers.gpu_to_cpu_handler.cpu_tensors
    assert sum(t.numel() for t in cpu) == 2 * (32 + 32)
    cpu[0][0].fill_(7)
    cpu[1][0].fill_(3)
    assert torch.all(cpu[0][0] == 7)
    assert torch.all(cpu[1][0] == 3)
    assert torch.all(cpu[0][1] == 0)
    assert handlers.cpu_to_gpu_handler.cpu_tensors is cpu


def test_private_cpu_allocations_use_group_slot_counts(monkeypatch):
    import vllm.v1.kv_offload.cpu.gpu_worker as worker

    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        worker, "SingleDirectionOffloadingHandler", lambda **kw: SimpleNamespace(**kw)
    )
    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor],
        [[CanonicalKVCacheRef(0, 16)], [], [CanonicalKVCacheRef(0, 8)]],
    )
    handlers = worker.CpuGpuOffloadingHandlers(
        caches,
        2,
        4,
        group_page_sizes={0: 32, 2: 32},
        group_num_blocks={0: 4, 2: 1},
    )
    cpu = handlers.gpu_to_cpu_handler.cpu_tensors
    assert [t.shape for t in cpu] == [(4, 32), (1, 32)]
    with pytest.raises(AssertionError):
        worker.CpuGpuOffloadingHandlers(caches, 2, 4, group_num_blocks={0: 4})


def test_grouped_worker_views_share_scheduler_rows(monkeypatch):
    import uuid

    import vllm.v1.kv_offload.cpu.gpu_worker as worker
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        worker, "SingleDirectionOffloadingHandler", lambda **kw: SimpleNamespace(**kw)
    )
    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor],
        [[CanonicalKVCacheRef(0, 16)], [], [CanonicalKVCacheRef(0, 8)]],
    )
    scheduler = {}
    regions = []
    handlers = []
    try:
        for group in (0, 2):
            instance = f"grouped-worker-test-{uuid.uuid4().hex}"
            args = dict(
                instance_id=instance,
                total_size_bytes=8192,
                num_blocks=2,
                num_workers=2,
                cpu_page_size=2048,
            )
            scheduler[group] = SharedOffloadRegion(rank=None, **args)
            regions.append(
                {rank: SharedOffloadRegion(rank=rank, **args) for rank in (0, 1)}
            )
        for rank in (0, 1):
            handler = worker.CpuGpuOffloadingHandlers(
                caches,
                2,
                2,
                group_page_sizes={0: 2048, 2: 2048},
                group_mmap_regions={g: regions[i][rank] for i, g in enumerate((0, 2))},
            )
            handlers.append(handler)
            for i, cpu in enumerate(handler.gpu_to_cpu_handler.cpu_tensors):
                assert cpu.stride() == (4096, 1)
                cpu[1].fill_(10 + rank + i * 20)
        for i, group in enumerate((0, 2)):
            with scheduler[group].create_kv_memoryview().cast("B") as view:
                for rank in (0, 1):
                    offset = 4096 + rank * 2048
                    assert (
                        bytes(view[offset : offset + 32])
                        == bytes([10 + rank + i * 20]) * 32
                    )
                    view[offset : offset + 32] = bytes([90 + rank]) * 32
            for rank in (0, 1):
                assert torch.all(
                    handlers[rank].gpu_to_cpu_handler.cpu_tensors[i][1] == 90 + rank
                )
    finally:
        for handler in handlers:
            handler.gpu_to_cpu_handler.cpu_tensors.clear()
        for group_regions in regions:
            for region in group_regions.values():
                region.cleanup()
        for region in scheduler.values():
            region.cleanup()


@pytest.mark.parametrize("failure", ["pin", "view", "handler"])
def test_grouped_worker_initialization_failure_releases_regions(monkeypatch, failure):
    import uuid
    from pathlib import Path

    import vllm.v1.kv_offload.cpu.gpu_worker as worker
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    def fail(*args, **kwargs):
        raise RuntimeError("injected initialization failure")

    tensor = CanonicalKVCacheTensor(torch.zeros((4, 16), dtype=torch.int8), 16)
    caches = CanonicalKVCaches(
        [tensor], [[CanonicalKVCacheRef(0, 16)], [], [CanonicalKVCacheRef(0, 16)]]
    )
    regions = {
        group: SharedOffloadRegion(
            instance_id=f"grouped-failure-test-{uuid.uuid4().hex}",
            total_size_bytes=4096,
            num_blocks=2,
            rank=0,
            num_workers=1,
            cpu_page_size=2048,
        )
        for group in (0, 2)
    }
    mappings = [region.mmap_obj for region in regions.values()]
    monkeypatch.setattr(worker, "is_pin_memory_available", lambda: failure == "pin")
    monkeypatch.setattr(worker, "pin_mmap_region", fail)
    monkeypatch.setattr(worker, "SingleDirectionOffloadingHandler", fail)
    if failure == "view":
        monkeypatch.setattr(regions[2], "create_next_view", fail)
    try:
        with pytest.raises(RuntimeError, match="injected initialization failure"):
            worker.CpuGpuOffloadingHandlers(
                caches,
                2,
                2,
                group_page_sizes={0: 2048, 2: 2048},
                group_mmap_regions=regions,
            )
        assert all(mapping.closed for mapping in mappings)
        for region in regions.values():
            assert region.fd is None
            assert region.mmap_obj is None
            assert not Path(region.mmap_path).exists()
    finally:
        for region in regions.values():
            region.cleanup()


@pytest.mark.parametrize("policy", ["lru", "arc"])
@pytest.mark.parametrize("blocked_first", [True, False])
def test_deferred_group_releases_reservations_and_retries(policy, blocked_first):
    m = manager(capacity=1, policy=policy)
    pinned = key(5, 1)
    store(m, [pinned])
    m.prepare_load([pinned], CTX)
    keys = [key(5, 2), key(0, 2)]
    if not blocked_first:
        keys.reverse()
    for _ in range(3):
        assert m.prepare_store(keys, CTX) is None
        assert all(m.lookup(k, CTX) is False for k in keys)
        assert m.lookup(pinned, CTX) is True
    m.complete_load([pinned], CTX)
    out = store(m, keys)
    assert out.keys_to_store == keys
    assert all(m.lookup(k, CTX) is True for k in keys)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_deferred_store_preserves_existing_and_inflight_keys(policy):
    m = manager(capacity=3, policy=policy)
    ready, inflight, fresh = [key(0, i) for i in range(3)]
    pinned = [key(5, i) for i in range(3)]
    store(m, [ready, *pinned])
    m.prepare_store([inflight], CTX)
    m.prepare_load(pinned, CTX)
    assert m.prepare_store([ready, inflight, fresh, key(5, 3)], CTX) is None
    assert m.lookup(ready, CTX) is True
    assert m.lookup(inflight, CTX) is None
    assert m.lookup(fresh, CTX) is False
    m.complete_store([inflight], CTX)
    assert m.lookup(inflight, CTX) is True
    m.complete_load(pinned, CTX)
    assert store(m, [fresh, key(5, 3)]).keys_to_store == [fresh, key(5, 3)]
