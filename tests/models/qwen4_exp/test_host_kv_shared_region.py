# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""shared hot region shared hot region on the worker, CPU interpreter.

The hot slots alias the index pages of DONATED history IDs. The engine trace
of test_host_kv_history drives a GPU arm and the host arm; here the host
arm's hot slots live in per-layer "index" tensors whose INUSE pages the
indexer keeps overwriting. Outputs must still match the GPU arm bit for bit,
which fails if a hot group ever lives in a reserved (null, ring or in-use)
page or survives a claim.
"""

import os

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryStore,
    hot_alias_view,
)

from .test_host_kv_history import (
    DIM,
    HEADS,
    KBS,
    WIDTH,
    Engine,
    _interpreter_boolean_comparison,  # noqa: F401  (autouse fixture)
    _prefill,
    _shape,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="The host history kernels run on the CPU interpreter here",
)

K = 4  # hot slots per index page


def _shared_shape(e4m3=True, blocks=33, **kwargs):
    return _shape(
        e4m3,
        blocks=blocks,
        hot_slots=blocks * K,
        slots_per_block=K,
        **kwargs,
    )


def _slot_bytes(shape):
    return 8 * HEADS * DIM * shape.dtype.itemsize


class SharedEngine(Engine):
    """Engine whose hot slots alias index pages, with scheduler-like roles."""

    def __init__(self, shape, *, dcp=(1, 0), seed=7):
        import random

        self.shape = shape
        self.index = [
            torch.zeros((shape.num_blocks, K * _slot_bytes(shape)), dtype=torch.uint8)
            for _ in range(shape.num_layers)
        ]
        views = [hot_alias_view(index, shape) for index in self.index]
        self.store = HostHistoryStore(
            shape, ["l0", "l1"], device="cpu", hot_views=views
        )
        self.store.bind_pages(
            [(hid, shape.num_blocks - hid) for hid in range(1, shape.num_blocks)]
        )
        self.layer = self.store.layer("l1")
        self.world, self.rank = dcp
        self.paged = torch.zeros(
            (shape.num_kernel_blocks, 2, KBS, HEADS, DIM), dtype=shape.dtype
        )
        self.free = list(range(1, shape.num_blocks))
        self.blocks: dict[str, list[int]] = {}
        self.length: dict[str, int] = {}
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.e4m3 = shape.dtype == torch.uint8
        self.calls = 0
        self.misses = 0
        self.inuse: set[int] = set()
        self.donations = 0
        self.store.set_step_plan(self.hot_slots(), None)

    def hot_slots(self) -> int:
        return (self.shape.num_blocks - 1 - len(self.inuse)) * K

    def ensure_blocks(self, req, tokens):
        span = KBS * self.world
        need = -(-tokens // span)
        new = []
        while len(self.blocks.setdefault(req, [])) < need:
            block = self.free.pop(0)
            self.blocks[req].append(block)
            new.append(block)
        if new:
            # Step head: claim, then zero (index and GPU arm) and begin.
            self.store.apply_roles([], new)
            self.inuse.update(new)
            for block in new:
                self.paged[block].zero_()
                for index in self.index:
                    index[block].zero_()
            self.store.begin_blocks(new)
            self.store.set_step_plan(self.hot_slots(), None)

    def finish(self, req):
        blocks = super().finish(req)
        self.store.apply_roles(blocks, [])
        self.inuse.difference_update(blocks)
        self.donations += len(blocks)
        self.store.set_step_plan(self.hot_slots(), None)
        return blocks

    def scribble(self):
        """The indexer writes every in-use (and null) page each step."""
        for index in self.index:
            for block in [0, *self.inuse]:
                index[block] = torch.randint(
                    0, 255, index[block].shape, generator=self.gen, dtype=torch.uint8
                )

    def step(self, rows):
        self.scribble()
        super().step(rows)
        self.check_roles()

    def check_roles(self):
        owner = self.store.slot_owner
        for block in [0, *self.inuse]:
            assert (owner[:, block * K : (block + 1) * K] == -2).all()
        free = [b for b in range(1, self.shape.num_blocks) if b not in self.inuse]
        for block in free:
            assert (owner[:, block * K : (block + 1) * K] >= -1).all()
        # owner[s] = g  <=>  map[g] = s
        for layer in range(self.shape.num_layers):
            for slot, gid in enumerate(owner[layer].tolist()):
                if gid >= 0:
                    assert int(self.store.group_to_slot[layer, gid]) == slot
            mapped = (self.store.group_to_slot[layer] >= 0).nonzero().flatten()
            for gid in mapped.tolist():
                slot = int(self.store.group_to_slot[layer, gid])
                assert int(owner[layer, slot]) == gid


@pytest.mark.parametrize("e4m3", [True, False])
def test_shared_region_matches_gpu_arm_under_indexer_writes(e4m3):
    engine = SharedEngine(_shared_shape(e4m3))
    _prefill(engine, "a", 0, 24)
    _prefill(engine, "a", 24, 31)
    _prefill(engine, "b", 0, 40)
    for _ in range(12):
        for req in ("a", "b"):
            engine.write(req, [engine.length[req]])
        engine.step([("a", engine.length["a"] - 1), ("b", engine.length["b"] - 1)])


def test_claim_of_a_page_holding_other_groups_unmaps_them_and_still_matches():
    engine = SharedEngine(_shared_shape(blocks=16), seed=5)
    _prefill(engine, "a", 0, 60)  # 4 IDs in use; hot groups fill free pages
    owner = engine.store.slot_owner
    victims = [
        b
        for b in range(1, 16)
        if b not in engine.inuse and (owner[:, b * K : (b + 1) * K] >= 0).any()
    ]
    assert victims, "no hot group lives in a donated page"
    # B takes exactly a page that holds A's hot groups (claim, then zero).
    engine.free = victims[:1] + [b for b in engine.free if b != victims[0]]
    _prefill(engine, "b", 0, 10)
    assert victims[0] in engine.blocks["b"]
    engine.step([("a", 59), ("a", 30), ("b", 9)])


def test_donated_ids_rejoin_the_region_and_pass_plan_follows_u_step():
    engine = SharedEngine(_shared_shape(blocks=16), seed=9)
    _prefill(engine, "a", 0, 40)
    _prefill(engine, "b", 0, 40)
    before = engine.store.available_slots
    engine.finish("b")
    assert engine.store.available_slots == before + 3 * K
    _prefill(engine, "a", 40, 70)
    assert engine.donations == 3


def test_apply_roles_rejects_static_pool_and_null_or_overlap():
    shared = SharedEngine(_shared_shape(blocks=8)).store
    with pytest.raises(ValueError):
        shared.apply_roles([0], [])
    with pytest.raises(ValueError):
        shared.apply_roles([2], [2])
    static = HostHistoryStore(_shape(blocks=6), ["l0", "l1"], device="cpu")
    with pytest.raises(ValueError):
        static.apply_roles([], [1])
    static.apply_roles([], [])


def test_reset_hot_state_restores_initial_roles():
    engine = SharedEngine(_shared_shape(blocks=16), seed=13)
    _prefill(engine, "a", 0, 40)
    engine.store.reset_hot_state()
    owner = engine.store.slot_owner
    assert (owner[:, :K] == -2).all() and (owner[:, K:] == -1).all()
    assert (engine.store.group_to_slot == -1).all()
    assert engine.store.available_slots == 15 * K


def test_fixed_bytes_exclude_aliased_payload_and_per_slot_metadata():
    shape = _shared_shape(blocks=8)
    store = SharedEngine(shape).store
    assert sum(store.fixed_device_memory_bytes().values()) == sum(
        shape.fixed_device_memory_bytes().values()
    )
    assert "hot" not in shape.fixed_device_memory_bytes()
    assert "slot_owner" not in shape.fixed_device_memory_bytes()


def test_non_divisible_standin_alias_and_sentinels():
    """1.9B stand-in geometry: logical page 100352 B, padded to 106496 B =
    13 slots of 8192 B. The alias covers the padding; a non-multiple stride
    is rejected; neighbouring pages are untouched by a slot write."""
    from dataclasses import replace

    from vllm.v1.attention.kv_cache_view import padded_page_stride

    slot = 8192
    stride = padded_page_stride(100352, slot, element_bytes=2)
    assert stride == 106496 and stride // slot == 13
    shape = replace(
        _shape(False, blocks=3),
        kv_heads=8,
        head_dim=64,
        hot_slots=3 * 13,
        slots_per_block=13,
    )
    assert slot == 8 * 8 * 64 * 2
    index = torch.zeros((3, stride), dtype=torch.uint8)
    view = hot_alias_view(index, shape)
    assert view.data_ptr() == index.data_ptr()
    assert view.untyped_storage().data_ptr() == index.untyped_storage().data_ptr()
    # The last slot of page 1 ends exactly at the page boundary, inside the
    # GPU padding [100352, 106496) of that page.
    view[2 * 13 - 1].view(torch.uint8).fill_(0xAB)
    page = index[1]
    assert (page[12 * slot :] == 0xAB).all() and (page[: 12 * slot] == 0).all()
    assert (index[0] == 0).all() and (index[2] == 0).all()
    with pytest.raises(ValueError, match="hot slots"):
        hot_alias_view(torch.zeros((3, 100352), dtype=torch.uint8), shape)


def test_alias_must_cover_whole_storage():
    shape = _shared_shape(blocks=8)
    index = torch.zeros((9, K * _slot_bytes(shape)), dtype=torch.uint8)
    with pytest.raises(ValueError):
        hot_alias_view(index, shape)
    good = torch.zeros((8, K * _slot_bytes(shape)), dtype=torch.uint8)
    view = hot_alias_view(good, shape)
    with pytest.raises(ValueError, match="alias whole index pages"):
        HostHistoryStore(
            shape, ["l0", "l1"], device="cpu", hot_views=[view, view.clone()[1:]]
        )
    with pytest.raises(ValueError, match="aliased view per layer"):
        HostHistoryStore(shape, ["l0", "l1"], device="cpu")


def test_widths_constant_is_the_selection_width():
    assert _shared_shape().selection_width == WIDTH
