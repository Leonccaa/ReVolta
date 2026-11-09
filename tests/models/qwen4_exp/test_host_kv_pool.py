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
    LayerScales,
)


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
    pool = _pool()
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


def test_lru_matches_independent_reference_across_requests_and_tail_changes():
    pool = _pool(
        _config(
            requests=2,
            max_tokens=520,
            max_write_rows=520,
            rows=2,
            selection_width=2,
            hot_groups=22,
        ),
        heads=1,
    )
    _fill(pool)
    rng = random.Random(20261107)
    owners = [-1] * pool.slots
    ages = [0] * pool.slots
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
        misses = sorted(wanted - set(owners))
        for gid in misses:
            candidates = [i for i, owner in enumerate(owners) if owner not in wanted]
            slot = min(candidates, key=lambda i: (ages[i], i))
            owners[slot], ages[slot] = gid, step
        for slot, gid in enumerate(owners):
            if gid in wanted:
                ages[slot] = step
        pool.gather(
            torch.tensor(selection, dtype=torch.int32),
            torch.arange(2, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int64) - 1,
        )
        assert int(pool.miss_count) == len(misses)
        assert pool.slot_owner.tolist() == owners
        assert pool.age.tolist() == ages
        for slot, gid in enumerate(owners):
            if gid >= 0:
                req, group = divmod(gid, pool.config.groups_per_request)
                assert int(pool.group_to_slot[req, group]) == slot
