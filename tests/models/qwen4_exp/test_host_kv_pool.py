# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/interpreter state-machine, physical-layout and transfer tests."""

import os
import random
from dataclasses import replace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv import (
    ActiveHostKV,
    HostKVConfig,
    HostKVResourceError,
    LayerScales,
    UnsupportedHostKVConfig,
    kernels,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _probe_large_offsets(Out):
    x = tl.arange(0, 4)
    offsets = kernels.host_offsets(
        x,
        x % 2,
        60 + x,
        x % 2,
        x,
        2**38,
        2**36,
        2**29,
        2**34,
        64,
    )
    tl.store(Out + x, offsets)


def _config(**kwargs) -> HostKVConfig:
    return replace(HostKVConfig(1, 128, 1, 1, 128, hot_groups=10), **kwargs)


def _pool(config=None, *, e4m3=True, heads=2, padded=True):
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU transfer fixtures need TRITON_INTERPRET=1")
    config = config or _config()
    pages = config.requests * ((config.max_tokens + 31) // 32)
    dtype = torch.uint8 if e4m3 else torch.float16
    shape = (pages, 2, 32, heads, 16)
    if padded:
        # Real HND strides, two packed members plus interior page padding.
        layout = torch.empty_strided(
            shape,
            (32 * heads * 16 * 4 + 64, 32 * heads * 16, 16, 32 * 16, 1),
            dtype=dtype,
            device="meta",
        )
    else:
        layout = torch.empty(shape, dtype=dtype, device="meta")
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32)
    table = table.reshape(config.requests, -1)
    pool = ActiveHostKV(
        config, layout, device="cpu", scales=LayerScales(0.25, 1.75), block_table=table
    )
    return pool


def _fill(pool, length=None):
    length = pool.config.max_tokens if length is None else length
    generator = torch.Generator().manual_seed(3201)
    rows = []
    for request in range(pool.config.requests):
        key = torch.randn((length, pool.heads, pool.dim), generator=generator).half()
        value = torch.randn(key.shape, generator=generator).half()
        pool.write(
            key,
            value,
            torch.full((length,), request, dtype=torch.int32),
            torch.arange(length, dtype=torch.int32),
        )
        rows.append((key, value))
    return rows


@pytest.mark.parametrize("policy", ["bucket", "lru"])
@pytest.mark.parametrize("e4m3", [False, True])
def test_device_memory_account_matches_allocated_tensors(policy, e4m3):
    pool = _pool(_config(eviction_policy=policy), e4m3=e4m3)
    actual = sum(
        t.numel() * t.element_size()
        for name, t in vars(pool).items()
        if isinstance(t, torch.Tensor) and name not in ("host", "host_device")
    )
    parts = pool.config.device_memory_bytes(
        page_tokens=pool.page_tokens,
        token_bytes=2 * pool.heads * pool.dim * pool.hot.element_size(),
    )
    assert parts["payload"] == pool.hot.numel() * pool.hot.element_size()
    assert sum(parts.values()) == actual


def _step(pool, tokens, length=None, requests=None, qpos=None):
    c = pool.config
    length = c.max_tokens if length is None else length
    lengths = torch.full((c.requests,), length, dtype=torch.int32)
    requests = torch.zeros(c.rows, dtype=torch.int32) if requests is None else requests
    qpos = (
        torch.full((c.rows,), length - 1, dtype=torch.int64) if qpos is None else qpos
    )
    return pool.gather(
        torch.tensor(tokens, dtype=torch.int32).reshape(c.rows, -1),
        requests,
        lengths,
        qpos,
    )


def test_capacity_admission_and_fraction():
    assert _config().required_groups == 10
    assert _config().capacities() == (10, 10)
    assert _config(hot_groups=None, hot_ratio=0.3125).capacities() == (10, 10)
    assert _config(max_tokens=17, hot_groups=5).capacities() == (5, 5)
    for config in (
        _config(hot_groups=9),
        _config(max_misses=9),
        _config(max_misses=11),
        _config(hot_ratio=0.5),
        _config(hot_groups=None, hot_ratio=float("nan")),
        _config(hot_groups=None, hot_ratio=0),
        _config(hot_groups=33),
        _config(rows=0),
        _config(max_tokens=2**31),
    ):
        with pytest.raises(ValueError):
            config.capacities()
    # Four DFlash query rows may select disjoint groups; no overlap assumption.
    assert _config(rows=4, selection_width=3, hot_groups=21).required_groups == 21


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_layer_scale(scale):
    with pytest.raises(ValueError, match="positive"):
        LayerScales(scale, 1)


@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_physical_pages_scales_and_transfer_are_exact(e4m3, padded):
    pool = _pool(e4m3=e4m3, padded=padded)
    rows = _fill(pool)
    assert not pool.host.is_pinned()  # Explicit interpreter fixture.
    assert pool.scales == LayerScales(0.25, 1.75)
    remapped = _step(pool, [5]).clone()
    logical_page, offset = 5 // 32, 5 % 32
    physical = int(pool.block_table[0, logical_page])
    slot, within = divmod(int(remapped[0, 0]), 4)
    assert torch.equal(pool.hot[slot, :, within], pool.host[physical, :, offset])
    key, value = rows[0]
    for plane, data, scale in ((0, key, 0.25), (1, value, 1.75)):
        expected = (
            (
                (data[5].float() / scale)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            if e4m3
            else data[5]
        )
        assert torch.equal(pool.host[physical, plane, offset], expected)
    assert int(pool.miss_count) == 9  # One historical group + eight tail groups.
    _step(pool, [5])
    assert int(pool.miss_count) == 0


def test_step_lru_and_unselected_tail_are_protected():
    pool = _pool(_config(eviction_policy="lru"))
    _fill(pool)
    _step(pool, [0])
    _step(pool, [4])
    _step(pool, [0])  # Refresh group zero; group one is now oldest.
    _step(pool, [8])
    assert int(pool.group_to_slot[0, 0]) >= 0
    assert int(pool.group_to_slot[0, 1]) == -1
    assert int(pool.group_to_slot[0, 2]) >= 0
    assert (pool.group_to_slot[0, 24:32] >= 0).all()
    resident = pool.slot_owner[pool.slot_owner >= 0]
    assert resident.unique().numel() == resident.numel() == 10


def test_unaligned_32_token_tail_occupies_nine_groups():
    pool = _pool()
    _fill(pool, 127)
    _step(pool, [0], 127)
    assert (pool.group_to_slot[0, 23:32] >= 0).all()
    assert int(pool.miss_count) == 10


def test_recent_tail_becomes_evictable_when_suffix_advances():
    pool = _pool()
    _fill(pool)
    _step(pool, [0], 64)
    assert (pool.group_to_slot[0, 8:16] >= 0).all()
    _step(pool, [0], 96)
    assert (pool.group_to_slot[0, 16:24] >= 0).all()
    assert int((pool.group_to_slot[0, 8:16] >= 0).sum()) <= 1


def test_warm_eliminates_next_step_misses():
    pool = _pool()
    _fill(pool)
    args = (
        torch.tensor([[4]], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([128], dtype=torch.int32),
        torch.tensor([127], dtype=torch.int64),
    )
    pool.warm(*args)
    assert int(pool.miss_count) == 9
    pool.prefetch(*args)
    assert int(pool.miss_count) == 0
    ptr = pool.indices.data_ptr()
    assert pool.gather(*args).data_ptr() == ptr
    assert int(pool.miss_count) == 0


def test_request_identity_is_part_of_group_key():
    pool = _pool(_config(requests=2, rows=2, hot_groups=20))
    _fill(pool)
    mapped = _step(pool, [0, 0], requests=torch.tensor([0, 1], dtype=torch.int32))
    assert mapped[0, 0] != mapped[1, 0]
    for req in range(2):
        slot = int(mapped[req, 0]) // 4
        page = int(pool.block_table[req, 0])
        assert torch.equal(pool.hot[slot, :, 0], pool.host[page, :, 0])


@pytest.mark.parametrize("e4m3", [False, True])
def test_append_prefill_overwrite_and_partial_group_write_through(e4m3):
    pool = _pool(e4m3=e4m3)
    _fill(pool, 61)
    mapped = _step(pool, [60], 61)
    slot = int(mapped[0, 0]) // 4
    generator = torch.Generator().manual_seed(102)
    key = torch.randn((7, pool.heads, pool.dim), generator=generator).half()
    value = -key
    positions = torch.arange(61, 68, dtype=torch.int32)
    pool.write(key, value, torch.zeros(7, dtype=torch.int32), positions)
    for pos in range(61, 64):
        page = int(pool.block_table[0, pos // 32])
        assert torch.equal(pool.hot[slot, :, pos % 4], pool.host[page, :, pos % 32])
    # Overwriting a resident old row also updates the GPU replica immediately.
    pool.write(
        key[:1],
        value[:1],
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([60], dtype=torch.int32),
    )
    _step(pool, [60], 68)
    slot = int(pool.group_to_slot[0, 15])
    page = int(pool.block_table[0, 1])
    assert torch.equal(pool.hot[slot, :, 0], pool.host[page, :, 28])


@pytest.mark.parametrize("e4m3", [False, True])
def test_ready_group_survives_visible_length_growth_without_rewrite(e4m3):
    pool = _pool(e4m3=e4m3)
    _fill(pool)
    # Host rows 61..63 are already committed when row 60 first becomes visible.
    for length in (61, 62, 63, 64, 65, 61, 64):
        mapped = _step(pool, [length - 1], length)
        slot, within = divmod(int(mapped[0, 0]), 4)
        page = int(pool.block_table[0, (length - 1) // 32])
        assert torch.equal(
            pool.hot[slot, :, within], pool.host[page, :, (length - 1) % 32]
        )


def test_flat_metadata_products_are_checked_without_allocating():
    with pytest.raises(ValueError, match="requests.*max_tokens"):
        _config(
            requests=8, max_tokens=2**29, max_write_rows=1, hot_groups=73
        ).capacities()


def test_host_offset_products_are_int64_without_large_allocation():
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU offset fixture needs TRITON_INTERPRET=1")
    offsets = torch.empty(4, dtype=torch.int64)
    _probe_large_offsets[(1,)](offsets)
    expected = [
        x * 2**38 + (x % 2) * 2**36 + (60 + x) * 2**29 + (x % 2) * 2**34 + x
        for x in range(4)
    ]
    assert offsets.tolist() == expected


def test_sort_resource_failure_keeps_mapping_consistent_and_allows_retry(monkeypatch):
    pool = _pool(_config(eviction_policy="lru"))
    _fill(pool)
    _step(pool, [0])
    owners, mapping = pool.slot_owner.clone(), pool.group_to_slot.clone()
    with monkeypatch.context() as patch:

        def fail_sort(*args, **kwargs):
            raise torch.OutOfMemoryError("injected sort scratch failure")

        patch.setattr(torch, "sort", fail_sort)
        with pytest.raises(HostKVResourceError, match="sort workspace"):
            _step(pool, [4])
    assert torch.equal(pool.slot_owner, owners)
    assert torch.equal(pool.group_to_slot, mapping)
    _step(pool, [4])
    assert int(pool.miss_count) == 1
    assert int(pool.group_to_slot[0, 1]) >= 0


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"rows": 2**16, "selection_width": 2**15}, "rows.*selection_width"),
        ({"selection_width": 128 * 65535 + 1}, "grid Y"),
        ({"max_write_rows": 2**31}, "max_write_rows"),
        ({"requests": 65536, "hot_groups": 600000}, "requests.*hot_groups"),
        ({"max_tokens": 2**31 - 1, "hot_groups": 2**29}, "hot token"),
    ],
)
def test_other_flat_products_and_launch_bounds(changes, match):
    with pytest.raises(UnsupportedHostKVConfig, match=match):
        _config(**changes).capacities()


@pytest.mark.parametrize("ratio,slots", [(0.1, 52429), (0.2, 104858)])
def test_eight_request_long_context_profiles_are_admitted(ratio, slots):
    config = HostKVConfig(8, 262144, 16, 2051, 4096, hot_ratio=ratio)
    assert config.required_groups == 32888
    assert config.capacities() == (slots, 32888)
    grouped = replace(config, qsa_block_topk=512)
    assert grouped.required_groups == 8280
    assert grouped.capacities() == (slots, 8280)


@pytest.mark.parametrize("rows", [1, 8])
def test_qsa_513_group_boundary_and_disjoint_verify_rows(rows):
    config = HostKVConfig(
        1, 20000, rows, 2051, 1, hot_groups=rows * 513 + 9, qsa_block_topk=512
    )
    pool = _pool(config, heads=1, padded=False)
    pool.host.fill_(56)
    pool.ready.fill_(1)
    selection = []
    for row in range(rows):
        start = row * 513 * 4
        selection.append(torch.arange(start, start + 2051, dtype=torch.int32))
    args = (
        torch.stack(selection),
        torch.zeros(rows, dtype=torch.int32),
        torch.tensor([19999], dtype=torch.int32),
        torch.arange(rows, dtype=torch.int64) * 513 * 4 + 2050,
    )
    if rows == 1:
        pool.gather(*args)
        assert (pool.hot == 56).all()
    else:
        pool._cpu_validate_gather(*args)
        pool._gather_metadata(*args)
        pool._remap(*args)
    assert config.required_groups == rows * 513 + 9
    assert int(pool.wanted_count) == int(pool.miss_count) == pool.slots
    assert (pool.indices >= 0).all()
    assert pool.slot_owner.unique().numel() == pool.slots
    with pytest.raises(UnsupportedHostKVConfig, match="hot capacity"):
        replace(config, hot_groups=pool.slots - 1).capacities()
    # A changed verify row count must be admitted against its new bound.
    with pytest.raises(UnsupportedHostKVConfig, match="hot capacity"):
        replace(config, rows=rows + 1).capacities()


@pytest.mark.parametrize(
    "changes",
    [
        {"qsa_block_topk": 513},
        {"qsa_block_topk": 0},
        {"qsa_block_topk": 512.0},
        {"selection_width": 2048},
    ],
)
def test_unsupported_qsa_geometry_is_rejected_at_construction(changes):
    config = HostKVConfig(1, 20000, 8, 2051, 1, hot_groups=4113, qsa_block_topk=512)
    with pytest.raises(UnsupportedHostKVConfig, match="QSA requires"):
        replace(config, **changes).capacities()


def test_qsa_arbitrary_columns_cannot_bypass_group_bound():
    pool = _pool(_config(selection_width=7, qsa_block_topk=1, hot_groups=11))
    _fill(pool)
    for tokens in ([0, 4, 8, 12, -1, -1, -1], [-1, 0, 1, 2, -1, -1, -1]):
        owners = pool.slot_owner.clone()
        with pytest.raises(ValueError, match="QSA group layout"):
            _step(pool, tokens)
        assert torch.equal(pool.slot_owner, owners)


@pytest.mark.parametrize("width", [1026, 2051])
@pytest.mark.parametrize("interleave", [1, 2])
def test_qsa_dcp2_local_pairs_preserve_the_513_group_bound(width, interleave):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import _localize_indices

    pool = _pool(
        HostKVConfig(
            1,
            8192,
            1,
            width,
            1,
            hot_groups=522,
            qsa_block_topk=512,
            qsa_group_columns=2,
        ),
        heads=1,
        padded=False,
    )
    pool.ready.fill_(1)
    # Alternate blocks that land in different local hot groups, plus the
    # three-token open group. Exercise the existing localization kernel.
    blocks = torch.arange(512, dtype=torch.int32) * 3
    global_selection = torch.cat(
        (
            (blocks[:, None] * 4 + torch.arange(4)).flatten().int(),
            torch.arange(12000, 12003, dtype=torch.int32),
        )
    )[None]
    for rank in (0, 1):
        local = torch.empty_like(global_selection)
        _localize_indices[(1,)](
            global_selection,
            local,
            2051,
            2051,
            WIDTH=2051,
            WORLD=2,
            RANK=rank,
            INTERLEAVE=interleave,
            BLOCK=4096,
        )
        selection = local[:, :width].contiguous()
        args = (
            selection,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([8191], dtype=torch.int32),
            torch.tensor([8190], dtype=torch.int64),
        )
        pool._cpu_validate_gather(*args)
        pool._gather_metadata(*args)
        pool._remap(*args)
        selected = selection[selection >= 0]
        wanted = set((selected // 4).tolist())
        wanted.update(range((8191 - 32) // 4, (8191 - 1) // 4 + 1))
        assert (selected // 4).unique().numel() == 513
        assert int(pool.wanted_count) == len(wanted) == 522
        assert (pool.indices[selection >= 0] >= 0).all()
    if width == 2051:
        selection[0, 1026] = 7000
        with pytest.raises(ValueError, match="QSA group layout"):
            pool._cpu_validate_gather(*args)


def test_bucket_warm_path_skips_sort_and_victim_scans(monkeypatch):
    pool = _pool()
    _fill(pool)
    _step(pool, [0])
    pool.victim_counts.fill_(-123)
    victims = pool.victim_slots.clone()

    def fail_sort(*args, **kwargs):
        raise AssertionError("bucket path called sort")

    monkeypatch.setattr(torch, "sort", fail_sort)
    _step(pool, [0])
    assert int(pool.miss_count) == 0
    assert (pool.victim_counts == -123).all()
    assert torch.equal(pool.victim_slots, victims)
    assert (pool.age[pool.slot_owner >= 0] == 2).all()
    assert pool.priority.numel() == pool.sorted_priority.numel() == 0


@pytest.mark.parametrize("hand", [0, 131000])
def test_bucket_prefix_ranks_across_tiles_and_multiple_chunks(hand):
    slots = 131073  # 257 tiles require two chunks per worker.
    pool = _pool(
        _config(max_tokens=4 * slots, hot_groups=slots, max_misses=1024),
        heads=1,
        padded=False,
    )
    pool.slot_owner.copy_(torch.arange(slots, dtype=torch.int32))
    pool.slot_owner[::11] = -1
    pool.wanted[0, ::7] = 1
    pool.age.copy_(torch.arange(slots, dtype=torch.int64) % 7)
    pool.clock.fill_(10)
    pool.eviction_hand.fill_(hand)
    pool.miss_count.fill_(1024)
    pool._select_victims()
    owners, ages, wanted = (
        pool.slot_owner.tolist(),
        pool.age.tolist(),
        pool.wanted[0].tolist(),
    )
    eligible = [i for i, owner in enumerate(owners) if owner < 0 or not wanted[owner]]

    def priority(i):
        if owners[i] < 0:
            return 0, (i - hand) % slots
        return (1 if ages[i] < 2 else 2), (i - hand) % slots

    assert pool.victim_slots[:1024].tolist() == sorted(eligible, key=priority)[:1024]


def test_bucket_can_evict_a_newer_slot_within_one_bucket():
    misses = {}
    for policy in ("bucket", "lru"):
        pool = _pool(_config(eviction_policy=policy))
        _fill(pool)
        owners = torch.tensor([0, 1, *range(24, 32)], dtype=torch.int32)
        pool.slot_owner.copy_(owners)
        pool.group_to_slot[0, owners.long()] = torch.arange(10, dtype=torch.int32)
        pool.age.fill_(1)
        pool.age[1] = 0
        pool.clock.fill_(3)
        _step(pool, [8])
        _step(pool, [0])
        misses[policy] = int(pool.miss_count)
    assert misses == {"bucket": 1, "lru": 0}


@pytest.mark.parametrize("policy", ["bucket", "lru"])
def test_churn_preserves_stable_groups_across_query_batches(policy):
    pool = _pool(
        _config(
            max_tokens=4096,
            max_write_rows=1,
            selection_width=16,
            hot_groups=128,
            eviction_policy=policy,
        ),
        heads=1,
        padded=False,
    )
    pool.ready.fill_(1)
    requests = torch.zeros(1, dtype=torch.int32)
    lengths = torch.tensor([4096], dtype=torch.int32)
    qpos = torch.tensor([4095], dtype=torch.int64)
    for step in range(15):
        for batch in range(4):
            stable = torch.arange(batch * 12, (batch + 1) * 12)
            churn_start = 64 + step * 16 + batch * 4
            churn = torch.arange(churn_start, churn_start + 4)
            selection = torch.cat((stable, churn)).int()[None] * 4
            pool._gather_metadata(selection, requests, lengths, qpos)
            if step >= 2:
                assert int(pool.miss_count) == 4
            assert (pool.group_to_slot[0, stable] >= 0).all()


def test_error_categories_and_payload_limit(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.host_kv.pool import _check_product

    with pytest.raises(ValueError) as invalid:
        _config(rows=1.5).capacities()
    assert type(invalid.value) is ValueError
    with pytest.raises(UnsupportedHostKVConfig, match="group transfer"):
        _pool(heads=129)
    _check_product("byte span", 2**62, limit=2**63 - 1)
    with pytest.raises(UnsupportedHostKVConfig, match="byte span"):
        _check_product("byte span", 2**62, 2, limit=2**63 - 1)

    def fail_allocation(*args, **kwargs):
        raise torch.OutOfMemoryError("injected allocation failure")

    monkeypatch.setattr(torch, "empty_strided", fail_allocation)
    layout = torch.empty((4, 2, 32, 2, 16), dtype=torch.uint8, device="meta")
    with pytest.raises(HostKVResourceError) as resource:
        ActiveHostKV(_config(), layout, device="cpu")
    assert isinstance(resource.value.__cause__, torch.OutOfMemoryError)


@pytest.mark.parametrize("e4m3", [False, True])
def test_truncate_reenter_prewritten_suffix_with_short_then_growing_length(e4m3):
    pool = _pool(e4m3=e4m3)
    _fill(pool)
    _step(pool, [60], 64)
    pool.truncate(torch.tensor([61], dtype=torch.int32))
    assert not pool.ready[0, 61:].any()
    # Re-enter using a whole already-computed chunk, then open it one row at a time.
    payload = torch.full((67, pool.heads, pool.dim), 7, dtype=pool.host.dtype)
    pool.write_encoded(
        payload,
        payload,
        torch.zeros(67, dtype=torch.int32),
        torch.arange(61, 128, dtype=torch.int32),
    )
    for length in (61, 62, 63, 64, 65, 128):
        mapped = _step(pool, [length - 1], length)
        slot, within = divmod(int(mapped[0, 0]), 4)
        page = int(pool.block_table[0, (length - 1) // 32])
        assert torch.equal(
            pool.hot[slot, :, within], pool.host[page, :, (length - 1) % 32]
        )


@pytest.mark.parametrize("query_requests", [[0, 1], [0, 0]])
def test_full_cold_union_at_exact_capacity_and_across_worker_grid(query_requests):
    pool = _pool(
        _config(
            requests=2,
            max_tokens=2048,
            max_write_rows=2048,
            rows=2,
            selection_width=128,
            hot_groups=274,
        ),
        heads=1,
    )
    _fill(pool)
    selection = torch.arange(128, dtype=torch.int32)[None].repeat(2, 1) * 4
    if query_requests[1] == 0:
        selection[1] += 512
    args = (
        selection,
        torch.tensor(query_requests, dtype=torch.int32),
        torch.tensor([2047, 2047], dtype=torch.int32),
        torch.tensor([2046, 2046], dtype=torch.int64),
    )
    pool.gather(*args)
    assert int(pool.wanted_count) == int(pool.miss_count) == pool.slots == 274
    assert pool.slot_owner.unique().numel() == pool.slots
    assert pool.gather_workers == 256
    for row, req in enumerate(query_requests):
        for column, token in enumerate(selection[row].tolist()):
            slot, within = divmod(int(pool.indices[row, column]), 4)
            page = int(pool.block_table[req, token // 32])
            assert torch.equal(
                pool.hot[slot, :, within], pool.host[page, :, token % 32]
            )
    pool.gather(*args)
    assert int(pool.miss_count) == 0
    assert not pool.claims.any()


def test_duplicates_padding_and_causal_columns_preserve_column_positions():
    pool = _pool(_config(selection_width=7, hot_groups=16))
    _fill(pool, 64)
    mapped = _step(
        pool, [5, -1, 5, 9, 64, 63, 0], 64, qpos=torch.tensor([9], dtype=torch.int64)
    )
    assert mapped[0, 0] == mapped[0, 2]
    assert torch.equal(mapped[0, [1, 4, 5]], torch.full((3,), -1, dtype=torch.int32))
    for column, token in ((0, 5), (2, 5), (3, 9), (6, 0)):
        assert int(mapped[0, column]) % 4 == token % 4


def test_truncate_and_request_release_do_not_expose_retired_data():
    pool = _pool()
    _fill(pool)
    _step(pool, [60])
    pool.truncate(torch.tensor([61], dtype=torch.int32))
    assert not pool.ready[0, 61:].any()
    assert (pool.group_to_slot[0, 15:] == -1).all()
    _step(pool, [60], 61)
    pool.truncate(torch.tensor([0], dtype=torch.int32))
    assert not pool.ready.any()
    assert (pool.group_to_slot == -1).all()
    assert (pool.slot_owner == -1).all()
    _fill(pool, 32)
    _step(pool, [0], 32)
    assert int(pool.miss_count) == 8


def test_metadata_rejection_precedes_payload_mutation():
    pool = _pool()
    _fill(pool)
    saved = pool.host.clone()
    key = torch.ones((2, pool.heads, pool.dim), dtype=pool.host.dtype)
    with pytest.raises(Exception, match="duplicate host KV write coordinate"):
        pool.write_encoded(
            key,
            key,
            torch.zeros(2, dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
        )
    assert torch.equal(pool.host, saved)
    with pytest.raises(Exception, match="invalid host KV write coordinate"):
        pool.write_encoded(
            key[:1],
            key[:1],
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([128], dtype=torch.int32),
        )
    assert torch.equal(pool.host, saved)
    # Padding writes do not publish a token or modify any payload.
    pool.write_encoded(
        key[:1],
        key[:1],
        torch.tensor([999], dtype=torch.int32),
        torch.tensor([-1], dtype=torch.int32),
    )
    assert torch.equal(pool.host, saved)


def test_uncommitted_selection_and_tail_fail_before_residency_changes():
    pool = _pool()
    _fill(pool, 64)
    with pytest.raises(Exception, match="tail KV is not committed"):
        _step(pool, [0], 96)
    assert (pool.group_to_slot == -1).all()
    pool.ready[0, 0] = 0
    with pytest.raises(Exception, match="selected KV is not committed"):
        _step(pool, [0], 64)
    assert (pool.slot_owner == -1).all()


def test_rejects_shared_host_pages_and_overlapping_layout():
    pool = _pool()
    config = pool.config
    with pytest.raises(ValueError, match="distinct"):
        ActiveHostKV(
            config,
            pool.host,
            device="cpu",
            block_table=torch.zeros((1, 4), dtype=torch.int32),
        )
    overlapping = torch.empty_strided(
        (4, 2, 32, 2, 16), (1, 1, 1, 1, 1), dtype=torch.uint8, device="meta"
    )
    with pytest.raises(ValueError, match="overlapping"):
        ActiveHostKV(config, overlapping, device="cpu")


@pytest.mark.parametrize("policy", ["bucket", "lru"])
@pytest.mark.parametrize("e4m3", [False, True])
def test_lru_matches_independent_reference_across_requests_and_tail_changes(
    e4m3, policy
):
    pool = _pool(
        _config(
            requests=2,
            max_tokens=520,
            max_write_rows=520,
            rows=2,
            selection_width=2,
            hot_groups=22,
            eviction_policy=policy,
        ),
        heads=1,
        e4m3=e4m3,
    )
    _fill(pool)
    rng = random.Random(20261107)
    owners = [-1] * pool.slots
    ages = [0] * pool.slots
    hand = 0
    for step in range(1, 31):
        lengths = [rng.randrange(33, 521), rng.randrange(33, 521)]
        selection = [[rng.randrange(lengths[r]) for _ in range(2)] for r in range(2)]
        wanted: set[int] = set()
        for req in range(2):
            g = pool.config.groups_per_request
            wanted.update(req * g + t // 4 for t in selection[req])
            wanted.update(
                req * g + t // 4 for t in range(lengths[req] - 32, lengths[req])
            )
        expected_misses = wanted - set(owners)
        pool.gather(
            torch.tensor(selection, dtype=torch.int32),
            torch.arange(2, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int64) - 1,
        )
        # Atomic compaction order is intentionally unconstrained, including
        # between different CTAs. Independently check the miss set and victim
        # policy, then assign that verified set in the observed arrival order.
        misses = pool.miss_groups[: int(pool.miss_count)].tolist()
        assert set(misses) == expected_misses
        assert len(misses) == len(expected_misses)
        for gid in misses:
            candidates = [i for i, owner in enumerate(owners) if owner not in wanted]
            if policy == "lru":
                slot = min(candidates, key=lambda i: (ages[i], i))
            else:

                def bucket(i, step=step):
                    if owners[i] < 0:
                        return 0
                    return 1 if ages[i] < step - 8 else 2

                slot = min(
                    candidates, key=lambda i: (bucket(i), (i - hand) % pool.slots)
                )
            owners[slot], ages[slot] = gid, step
        if policy == "bucket" and misses:
            hand = (slot + 1) % pool.slots
        if policy == "bucket":
            assert int(pool.eviction_hand) == hand
        for slot, gid in enumerate(owners):
            if gid in wanted:
                ages[slot] = step
        assert int(pool.miss_count) == len(misses)
        assert pool.slot_owner.tolist() == owners
        assert pool.age.tolist() == ages
        for slot, gid in enumerate(owners):
            if gid >= 0:
                req, group = divmod(gid, pool.config.groups_per_request)
                assert int(pool.group_to_slot[req, group]) == slot
                for within in range(4):
                    token = group * 4 + within
                    page = int(pool.block_table[req, token // 32])
                    assert torch.equal(
                        pool.hot[slot, :, within], pool.host[page, :, token % 32]
                    )


@pytest.mark.parametrize("e4m3", [False, True])
def test_random_writes_truncation_and_visibility_match_independent_payload(e4m3):
    pool = _pool(
        _config(
            requests=2,
            max_tokens=65,
            max_write_rows=65,
            rows=2,
            selection_width=3,
            hot_groups=24,
        ),
        e4m3=e4m3,
        heads=1,
    )
    rng = random.Random(7823)
    expected = torch.zeros((2, 65, 2, pool.heads, pool.dim), dtype=pool.host.dtype)
    ready = torch.zeros((2, 65), dtype=torch.bool)
    for step in range(40):
        req = rng.randrange(2)
        if step % 5 == 0:
            length = rng.randrange(66)
            pool.truncate(
                torch.tensor(
                    [length if r == req else 65 for r in range(2)], dtype=torch.int32
                )
            )
            ready[req, length:] = False
        else:
            # Filling the prefix closes holes after truncation. It also
            # overwrites resident rows and commits future-visible rows early.
            length = rng.randrange(1, 66)
            key = torch.full(
                (length, pool.heads, pool.dim), step + 1, dtype=pool.host.dtype
            )
            value = key + 1
            pool.write_encoded(
                key,
                value,
                torch.full((length,), req, dtype=torch.int32),
                torch.arange(length, dtype=torch.int32),
            )
            expected[req, :length, 0] = key
            expected[req, :length, 1] = value
            ready[req, :length] = True
        lengths = []
        for r in range(2):
            holes = (~ready[r]).nonzero().flatten()
            frontier = int(holes[0]) if holes.numel() else 65
            lengths.append(rng.randrange(frontier + 1))
        selection = torch.tensor(
            [[rng.randrange(-1, 65) for _ in range(3)] for _ in range(2)],
            dtype=torch.int32,
        )
        qpos = torch.tensor(
            [max(0, length - 1) for length in lengths], dtype=torch.int64
        )
        pool.gather(
            selection,
            torch.arange(2, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
            qpos,
        )
        assert torch.equal(pool.ready.bool(), ready)
        assert not pool.claims.any()
        for r in range(2):
            for token in ready[r].nonzero().flatten().tolist():
                page = int(pool.block_table[r, token // 32])
                assert torch.equal(pool.host[page, :, token % 32], expected[r, token])
            for column, token in enumerate(selection[r].tolist()):
                mapped = int(pool.indices[r, column])
                if 0 <= token < lengths[r] and token <= int(qpos[r]):
                    assert mapped >= 0
                    slot, within = divmod(mapped, 4)
                    assert torch.equal(pool.hot[slot, :, within], expected[r, token])
                else:
                    assert mapped == -1
        residents = pool.slot_owner[pool.slot_owner >= 0]
        assert residents.unique().numel() == residents.numel()
        for slot, gid in enumerate(pool.slot_owner.tolist()):
            if gid < 0:
                continue
            r, group = divmod(gid, pool.config.groups_per_request)
            assert int(pool.group_to_slot[r, group]) == slot
            for token in range(group * 4, min(group * 4 + 4, 65)):
                if ready[r, token]:
                    assert torch.equal(pool.hot[slot, :, token % 4], expected[r, token])
