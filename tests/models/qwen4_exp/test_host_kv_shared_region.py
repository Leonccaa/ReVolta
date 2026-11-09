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


def _production_layer_store():
    """FP8 per-layer production page geometry, k=26, DCP2, eight requests."""
    from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryShape

    shape = HostHistoryShape(
        num_blocks=25,
        kernel_blocks_per_block=1,
        kernel_block_tokens=832,
        num_layers=1,
        kv_heads=1,
        head_dim=256,
        dtype=torch.uint8,
        hot_slots=25 * 26,
        slots_per_block=26,
        max_requests=8,
        selection_width=2051,
        max_write_rows=8192,
        dcp_world_size=2,
        max_context_tokens=262144,
        decode_rows=8,
    )
    index = torch.full((25, 26 * 2048), 99, dtype=torch.uint8)
    store = HostHistoryStore(
        shape,
        ["l0"],
        device="cpu",
        debug=False,
        hot_views=[hot_alias_view(index, shape)],
    )
    store.bind_pages([(hid, hid) for hid in range(1, 25)])
    store.host.fill_(77)
    store.ready.fill_(1)
    store.apply_roles([], list(range(1, 24)))
    store.set_step_plan(26, None)
    store.victim_slots.fill_(26)  # a formerly valid slot in a now claimed page
    return store, index


def test_insufficient_victims_keep_reserved_index_pages():
    store, index = _production_layer_store()
    before = index.clone()
    owners = store.slot_owner.clone()
    # Eight 32-local-token tails need 64 slots, but only 26 remain DONATED.
    # The interpreter suppresses device assertions. Exercise the data guard
    # itself, which also protects builds with assertions disabled.
    store.layer("l0").gather(
        torch.full((8, 1), -1, dtype=torch.int32),
        torch.arange(8, dtype=torch.int32),
        torch.arange(1, 9, dtype=torch.int32).view(8, 1),
        torch.full((8,), 64, dtype=torch.int32),
    )
    assert store.miss_count.item() == 0
    assert torch.equal(index, before)
    assert torch.equal(store.slot_owner, owners)
    assert (store.group_to_slot == -1).all()


def test_insufficient_victims_clear_count_even_without_device_asserts():
    from vllm.models.qwen4_exp.nvidia.host_kv import kernels
    from vllm.triton_utils import triton

    store, _ = _production_layer_store()
    layer = store.layer("l0")
    store.miss_count.fill_(64)
    kernels.count_victim_buckets[(store.victim_tiles,)](
        store.wanted,
        layer.slot_owner,
        layer.age,
        layer.clock,
        store.miss_count,
        store.victim_counts,
        layer.eviction_hand,
        store.shape.hot_slots,
        store.victim_tiles,
        512,
    )
    before = store.victim_slots.clone()
    kernels.select_bucket_victims[(store.victim_tiles,)](
        store.wanted,
        layer.slot_owner,
        layer.age,
        layer.clock,
        store.miss_count,
        store.victim_counts,
        store.victim_slots,
        layer.eviction_hand,
        store.shape.hot_slots,
        store.victim_tiles,
        512,
        triton.next_power_of_2(store.victim_tiles),
        debug=False,
    )
    assert store.miss_count.item() == 0
    assert torch.equal(store.victim_slots, before)


@pytest.mark.parametrize("full_graph", [False, True])
def test_live_dummy_rejected_before_model_or_fullgraph_dispatch(full_graph):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    store, index = _production_layer_store()
    before = index.clone()

    def unexpected_dispatch(*args):
        raise AssertionError("live dummy reached model dispatch")

    runner = SimpleNamespace(
        host_history=store,
        _host_history_live=True,
        _get_uniform_decode_token_count=unexpected_dispatch,
        cudagraph_manager=SimpleNamespace(run_fullgraph=unexpected_dispatch),
    )
    output = SimpleNamespace(
        num_scheduled_tokens={"dummy": 1}, total_num_scheduled_tokens=1
    )
    with pytest.raises(RuntimeError, match="dummy.*before serving"):
        GPUModelRunner.execute_model(
            runner, output, dummy_run=True, skip_attn_for_dummy_run=not full_graph
        )
    assert torch.equal(index, before)


def test_production_layer_gather_fits_current_hot_capacity():
    store, index = _production_layer_store()
    before = index[:24].clone()
    indices = store.layer("l0").gather(
        torch.zeros((1, 1), dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.ones((1, 1), dtype=torch.int32),
        torch.full((1,), 64, dtype=torch.int32),
    )
    assert store.miss_count.item() == 8
    assert indices.min().item() >= 24 * 26 * 4
    assert torch.equal(index[:24], before)
    slots = store.group_to_slot[0]
    resident = slots[slots >= 0].long()
    assert resident.numel() == 8
    assert (store.hot[0][resident] == 77).all()


def test_dummy_wrapper_restores_connector_after_rejection():
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    disabled: list[bool] = []

    def reject_dummy(*args, **kwargs):
        raise RuntimeError("Host KV dummy batches are only supported before serving")

    runner = SimpleNamespace(
        max_num_reqs=8,
        is_first_pp_rank=True,
        execute_model=reject_dummy,
        kv_connector=SimpleNamespace(set_disabled=disabled.append),
    )
    with pytest.raises(RuntimeError, match="dummy.*before serving"):
        GPUModelRunner._dummy_run(runner, 8)
    assert disabled == [True, False]
