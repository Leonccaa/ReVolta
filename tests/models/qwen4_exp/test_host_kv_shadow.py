# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""value arbitration layer-0 hot shadow table against its CPU reference
(interpreter)."""

import os

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryStore
from vllm.models.qwen4_exp.nvidia.host_kv.shadow import EPOCH_MOD, HotShadowTable
from vllm.v1.core.value_arbiter import (
    HOT_CLAIM_EVICTED,
    HOT_FORCED_RECENT,
    HOT_GATHERS,
    HOT_MISSES,
    HOT_REFETCH,
    HotShadowRef,
)

from .test_host_kv_history import Engine, _interpreter_boolean_comparison, _shape

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="The host history kernels run on the CPU interpreter here",
)
__all__ = ["_interpreter_boolean_comparison"]


class ShadowEngine(Engine):
    """Gathers on layer 0 with the shadow table and mirrors every call in
    the CPU reference."""

    def __init__(self, shape, **kwargs):
        super().__init__(shape, **kwargs)
        self.store = HostHistoryStore(shape, ["l0", "l1"], device="cpu",
                                      value_shadow=True)  # fmt: skip
        self.store.bind_pages(
            [(hid, shape.num_blocks - hid) for hid in range(1, shape.num_blocks)]
        )
        self.layer = self.store.layer("l0")
        self.ref = HotShadowRef(shape.num_groups)
        self.expected = {HOT_REFETCH: 0, HOT_FORCED_RECENT: 0, HOT_MISSES: 0,
                         HOT_GATHERS: 0}  # fmt: skip
        self.window = 0
        layer, original = self.layer, self.layer.gather

        def gather(*args):
            owner = layer.slot_owner.clone()
            age = layer.age.clone()
            clock = int(layer.clock) + 1
            out = original(*args)
            misses = self.store.miss_groups[: int(self.store.miss_count)].tolist()
            self.expected[HOT_REFETCH] += self.ref.check(misses)
            self.expected[HOT_MISSES] += len(misses)
            self.expected[HOT_GATHERS] += 1
            after = layer.slot_owner
            recent = [
                int(owner[s])
                for s in range(owner.numel())
                if owner[s] >= 0 and after[s] != owner[s] and age[s] >= clock - 8
            ]
            self.expected[HOT_FORCED_RECENT] += len(recent)
            self.ref.record(recent)
            return out

        layer.gather = gather

    def advance(self, window):
        self.window = window
        self.store.shadow.set_window(window)
        self.ref.set_window(window)

    def ensure_blocks(self, req, tokens):
        before = set(self.blocks.get(req, []))
        super().ensure_blocks(req, tokens)
        new = [b for b in self.blocks[req] if b not in before]
        gpb = self.shape.groups_per_block
        self.ref.clear_groups(g for b in new for g in range(b * gpb, (b + 1) * gpb))

    def check(self):
        dev = self.store.shadow.epoch.tolist()
        want = [-1 if e < 0 else e % EPOCH_MOD for e in self.ref.epoch]
        assert dev == want
        counters = self.store.shadow.counters.tolist()
        for index, value in self.expected.items():
            assert counters[index] == value, index


def _drive(engine, windows, jumps=()):
    for w in range(1, windows + 1):
        target = engine.window + (3 if w in jumps else 1)
        engine.advance(target)
        engine.write("a", [engine.length["a"]])
        engine.step([("a", engine.length["a"] - 1)])
        engine.check()


def _engine(seed=3):
    engine = ShadowEngine(_shape(rows_per_pass=1, requests=1, blocks=12), seed=seed)
    engine.write("a", list(range(100)))
    engine.step([("a", q) for q in range(100)])
    engine.check()
    return engine


def test_shadow_table_matches_reference_and_counts_refetches():
    engine = _engine()
    _drive(engine, 12, jumps={5, 9})
    c = engine.store.shadow.counters
    assert int(c[HOT_FORCED_RECENT]) > 0 and int(c[HOT_REFETCH]) > 0


def test_shadow_epochs_cross_the_modulus_without_aliasing():
    engine = _engine(seed=5)
    # Leave records, then pass more than EPOCH_MOD windows one at a time:
    # every stale record is expired actively; nothing aliases back.
    assert any(e >= 0 for e in engine.ref.epoch)
    for _ in range(EPOCH_MOD + 3):
        engine.advance(engine.window + 1)
    engine.check()
    assert all(e == -1 for e in engine.store.shadow.epoch.tolist())
    _drive(engine, 4)


def test_shadow_cleared_when_block_ids_are_begun():
    engine = _engine(seed=11)
    engine.advance(1)
    engine.write("a", [engine.length["a"]])
    engine.step([("a", engine.length["a"] - 1)])
    blocks = engine.finish("a")
    gpb = engine.shape.groups_per_block
    # Seed records on the freed blocks, then reuse them.
    groups = [b * gpb for b in blocks]
    engine.store.shadow.epoch[groups] = engine.window % EPOCH_MOD
    engine.ref.record(groups)
    engine.ensure_blocks("c", 16 * len(blocks))
    engine.check()
    assert all(engine.store.shadow.epoch[g] == -1 for g in groups)


def test_windows_only_move_forward():
    table = HotShadowTable(8, 4, torch.device("cpu"))
    table.set_window(3)
    with pytest.raises(ValueError):
        table.set_window(2)


def test_claim_evictions_counted_separately():
    table = HotShadowTable(16, 4, torch.device("cpu"))
    owner = torch.tensor([5, -1, 7, -1, -1, -1, 9, 9], dtype=torch.int32)
    table.count_claims(torch.tensor([0, 1], dtype=torch.int64), owner, 4)
    assert int(table.counters[HOT_CLAIM_EVICTED]) == 4
    assert int(table.counters[HOT_FORCED_RECENT]) == 0


def test_harvest_is_one_step_late():
    table = HotShadowTable(8, 4, torch.device("cpu"))
    table.counters[HOT_MISSES] = 3
    assert table.harvest(10) is None
    table.counters[HOT_MISSES] = 5
    step, values = table.harvest(11)
    assert step == 10 and values[HOT_MISSES] == 3
    step, values = table.harvest(12)
    assert step == 11 and values[HOT_MISSES] == 5


def test_store_without_shadow_launches_nothing_extra():
    store = HostHistoryStore(_shape(), ["l0", "l1"], device="cpu")
    assert store.shadow is None
