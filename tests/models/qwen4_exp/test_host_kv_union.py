# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""selection-union gather (selection-union gather): mixed steps read the union of their
selections from hot slots.

CPU interpreter checks of the union path: overflow-safe probe and reset,
exact first-reference window planning, page-size-4 hot tables, eight-row
windows with masking, the per-row fallback (L3), insertion ages, and the
model forward against the GPU-resident arm, bit for bit, for every legal
window split.
"""

import dataclasses
import functools
import os
import random

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv import history as history_mod
from vllm.models.qwen4_exp.nvidia.host_kv import history_kernels as hk
from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryStore,
    UnionStepInfo,
    for_each_union_window,
)
from vllm.models.qwen4_exp.nvidia.host_kv.planning import (
    history_groups,
    plan_union_windows_upper,
    union_bound,
    union_fast_path,
    union_window_groups,
)

from .test_host_kv_forward import Arms, _trace
from .test_host_kv_forward import _cpu_kernels as _forward_cpu_kernels  # noqa: F401
from .test_host_kv_history import (
    DIM,
    HEADS,
    KBS,
    QHEADS,
    SCALES,
    WIDTH,
    Engine,
    _bits,
    _shape,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="The host KV kernels run on the CPU interpreter here",
)


# --------------------------------------------------------------------------
# Model forward: union arm vs GPU-resident arm.


class UnionArms(Arms):
    def __init__(self, *, union=True, insert_age="old", **kwargs):
        super().__init__(**kwargs)
        store: HostHistoryStore = vars(self)["store"]
        shape = dataclasses.replace(
            store.shape,
            prefill_gather="union" if union else "passes",
            prefill_insert_age=insert_age,
        )
        self.store = HostHistoryStore(shape, ["attn"], device="cpu")
        self.store.bind_pages([(hid, 17 - hid) for hid in range(1, 17)])
        self.host = self.store.layer("attn")


def _sweep(monkeypatch, groups):
    monkeypatch.setattr(
        history_mod,
        "for_each_union_window",
        functools.partial(for_each_union_window, max_window_groups=groups),
    )


@pytest.mark.parametrize("e4m3", [True, False])
@pytest.mark.parametrize("dcp", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("groups", [None, 1, 2, 3])
def test_forward_union_matches_gpu_arm_for_every_window_split(
    monkeypatch, e4m3, dcp, groups
):
    # top-k 8 (one split for any row count), a hot pool far below the
    # mixed step's union: windows (and possibly L3 rows) are forced.
    if groups is not None:
        _sweep(monkeypatch, groups)
    arms = UnionArms(e4m3=e4m3, topk=8, rows_per_pass=2, dcp=dcp)
    _trace(arms)
    stats = arms.store.union_stats
    assert stats["calls"] == 2
    assert stats["windows"] + stats["l3_groups"] >= 2


@pytest.mark.parametrize("dcp", [(1, 0), (2, 1)])
@pytest.mark.parametrize("groups", [None, 1, 2])
def test_forward_union_split_k_with_windows(monkeypatch, dcp, groups):
    # top-k 32: split-K + merge depend on the row count. The passes arm
    # needs one pass to match; the union arm keeps the full shape, so any
    # window split matches as long as no eight-row group falls back (L3).
    if groups is not None:
        _sweep(monkeypatch, groups)
    arms = UnionArms(e4m3=True, topk=32, rows_per_pass=9, dcp=dcp)
    _trace(arms)
    assert arms.store.union_stats["l3_groups"] == 0


@pytest.mark.parametrize("dcp", [(1, 0), (2, 1)])
def test_forward_passes_arm_never_enters_the_union_path(monkeypatch, dcp):
    def fail(*args, **kwargs):
        raise AssertionError("union path used with prefill_gather=passes")

    monkeypatch.setattr(history_mod, "run_union_windows", fail)
    arms = UnionArms(union=False, e4m3=True, topk=8, rows_per_pass=2, dcp=dcp)
    _trace(arms)
    assert arms.store.union_stats["calls"] == 0
    assert arms.store.union_scratch == {}


# --------------------------------------------------------------------------
# Store level: one layer, both arms, explicit capacities.


class UnionEngine(Engine):
    """Engine whose mixed steps run ``for_each_union_window``."""

    def __init__(self, shape, *, grouped_mock=False, **kwargs):
        super().__init__(shape, **kwargs)
        self.grouped_mock = grouped_mock
        self.results = []

    def inputs(self, rows):
        reqs = sorted({r for r, _ in rows}, key=lambda r: [x for x, _ in rows].index(r))
        table = self.table(reqs)
        seq_lens = torch.tensor([self.length[r] for r in reqs], dtype=torch.int32)
        token_to_req = torch.tensor([reqs.index(r) for r, _ in rows], dtype=torch.int32)
        selection = torch.tensor(
            [self.selection(r, q) for r, q in rows], dtype=torch.int32
        )
        query = (torch.randn((len(rows), QHEADS, DIM), generator=self.gen) * 0.5).half()
        return reqs, table, seq_lens, token_to_req, selection, query

    def plan(self, rows):
        return SparseAttentionPlan(
            rows=rows,
            query_heads=QHEADS,
            kv_heads=HEADS,
            head_dim=DIM,
            selection_width=WIDTH,
            device="cpu",
            e4m3=self.e4m3,
            scales=SCALES,
        )

    def group_signature(self, key, table, selection, requests, page):
        """A stand-in for the grouped planner: per aligned eight-row group,
        the dedup set of physical microblocks its live rows reference."""
        out = torch.zeros(selection.shape[0], dtype=torch.float64)
        for g in range(0, selection.shape[0], 8):
            pages = set()
            for row in range(g, min(g + 8, selection.shape[0])):
                req = int(requests[row])
                if req < 0:
                    continue
                for token in selection[row].tolist():
                    if token < 0:
                        continue
                    phys = int(table[req, token // page])
                    assert phys >= 0, "live row reads a non-resident microblock"
                    pages.add((phys, token % page // 4))
            total = sum(
                float(_bits(key[p, 4 * m : 4 * m + 4]).double().sum()) for p, m in pages
            )
            out[g : g + 8] = total
        return out

    def union_step(
        self,
        rows,
        *,
        capacity=None,
        max_window_groups=None,
        req_class=None,
        fast_path=False,
        capacity_delta=None,
    ):
        """Attend ``rows`` in both arms and compare bits; return the result.

        ``capacity_delta`` sets U to the step's exact union plus the delta."""
        self.calls += 1
        reqs, table, seq_lens, token_to_req, selection, query = self.inputs(rows)
        if capacity_delta is not None:
            tail, sel = _gids(self, table, seq_lens, token_to_req, selection)
            capacity = len(tail | set(sel)) + capacity_delta
        n = len(rows)
        gpu_k, gpu_v = self.paged.unbind(1)
        plan = self.plan(n)
        expected = plan.run(query, gpu_k, gpu_v, selection, table, token_to_req).clone()
        expected_lse = plan.lse.clone()
        if self.grouped_mock:
            expected_sig = self.group_signature(
                gpu_k, table, selection, token_to_req, KBS
            )
        outputs = [
            torch.full((n, QHEADS, DIM), 7.0, dtype=torch.float16),
            torch.full((n, QHEADS), 7.0),
            torch.full((n,), 7.0, dtype=torch.float64),
        ]
        layer, store = self.layer, self.store
        row_spans = []

        def attend_union(mb_table, requests, outs):
            # Every live (row, column) reads a slot that holds exactly the
            # host page bytes of its logical token.
            for row in range(n):
                req = int(requests[row])
                if req < 0:
                    continue
                assert req == int(token_to_req[row])
                for token in selection[row].tolist():
                    if token < 0:
                        continue
                    slot = int(mb_table[req, token // 4])
                    assert slot >= 0
                    kb = int(table[req, token // KBS])
                    gid = kb * (KBS // 4) + token % KBS // 4
                    assert int(layer.slot_owner[slot]) == gid
                    page = int(store.page_of[kb])
                    host = layer.host_view[page, 0, :, token % KBS]
                    assert torch.equal(
                        _bits(layer.hot[slot, :, token % 4]), _bits(host)
                    )
            # One owner per slot and a consistent map.
            owners = layer.slot_owner
            live = owners >= 0
            assert owners[live].unique().numel() == int(live.sum())
            slots = torch.nonzero(live).flatten()
            assert torch.equal(layer.group_to_slot[owners[live].long()].long(), slots)
            p = self.plan(n)
            outs[0].copy_(
                p.run(
                    query,
                    layer.key_cache,
                    layer.value_cache,
                    selection,
                    mb_table,
                    requests.contiguous(),
                )
            )
            outs[1].copy_(p.lse)
            if self.grouped_mock:
                outs[2].copy_(
                    self.group_signature(
                        layer.key_cache, mb_table, selection, requests, 4
                    )
                )

        def attend_rows(start, end, hot_ids):
            row_spans.append((start, end))
            p = self.plan(end - start)
            outputs[0][start:end] = p.run(
                query[start:end],
                layer.key_cache,
                layer.value_cache,
                hot_ids,
                store.hot_block_table,
                token_to_req[start:end].contiguous(),
            )
            outputs[1][start:end] = p.lse

        result = for_each_union_window(
            layer,
            selection,
            token_to_req,
            table,
            seq_lens,
            outputs,
            attend_union,
            attend_rows,
            capacity=self.shape.hot_slots if capacity is None else capacity,
            fast_path=fast_path,
            req_class=req_class,
            max_window_groups=max_window_groups,
        )
        # Every row is executed exactly once, in order.
        cursor = 0
        for start, end, kind in result.spans:
            assert start == cursor and end > start
            assert start % 8 == 0
            assert kind == "union" or end - start <= 8
            cursor = end
        assert cursor == n or result.level in ("l0", "l1")
        fallback = torch.zeros(n, dtype=torch.bool)
        for start, end in row_spans:
            fallback[start:end] = True
            # L3 rows: one row per call, compared with the same shape.
            for row in range(start, end):
                p = self.plan(1)
                ref = p.run(
                    query[row : row + 1],
                    gpu_k,
                    gpu_v,
                    selection[row : row + 1],
                    table,
                    token_to_req[row : row + 1].contiguous(),
                )
                assert torch.equal(_bits(ref), _bits(outputs[0][row : row + 1]))
        keep = ~fallback
        assert torch.equal(_bits(expected[keep]), _bits(outputs[0][keep]))
        assert torch.equal(_bits(expected_lse[keep]), _bits(outputs[1][keep]))
        if self.grouped_mock:
            assert torch.equal(expected_sig[keep], outputs[2][keep])
        self.check_bytes()
        self.results.append(result)
        return result, (reqs, table, seq_lens, token_to_req, selection)


def _history(engine, reqs=("a", "b"), lengths=(60, 45)):
    for req, length in zip(reqs, lengths):
        start = engine.length.get(req, 0)
        for chunk in range(start, length, 100):
            engine.write(req, list(range(chunk, min(chunk + 100, length))))


def _mixed_rows(engine, decode=("a",), prefill="b", chunk=19):
    rows = []
    for req in decode:
        engine.write(req, [engine.length[req]])
        rows.append((req, engine.length[req] - 1))
    start = engine.length[prefill]
    engine.write(prefill, list(range(start, start + chunk)))
    rows += [(prefill, q) for q in range(start, start + chunk)]
    return rows


@pytest.mark.parametrize("e4m3", [True, False])
def test_window_sweep_is_bitwise_and_rows_not_multiple_of_eight(e4m3):
    # 1 decode row + 19 prefill rows = 20 rows (not a multiple of eight).
    engine = UnionEngine(_shape(e4m3, rows_per_pass=8, blocks=40), seed=11)
    _history(engine)
    for groups in (None, 1, 2, 3):
        result, _ = engine.union_step(_mixed_rows(engine), max_window_groups=groups)
        assert result.l3_groups == 0
        expect = {None: 1, 1: 3, 2: 2, 3: 1}[groups]
        assert result.windows == expect


def test_dcp2_rank_local_union_matches_gpu_arm():
    for rank in (0, 1):
        shape = _shape(rows_per_pass=8, blocks=40, dcp_world_size=2, dcp_rank=rank)
        engine = UnionEngine(shape, dcp=(2, rank))
        _history(engine)
        for groups in (None, 1, 2):
            engine.union_step(_mixed_rows(engine), max_window_groups=groups)


def test_l0_l1_l2_levels_and_sync_counts():
    engine = UnionEngine(_shape(rows_per_pass=8, blocks=40), seed=5)
    _history(engine)
    r0, _ = engine.union_step(_mixed_rows(engine), fast_path=True)
    assert (r0.level, r0.syncs, r0.windows) == ("l0", 0, 1)
    r1, _ = engine.union_step(_mixed_rows(engine), capacity_delta=0)
    assert (r1.level, r1.syncs, r1.windows) == ("l1", 1, 1)
    # Capacity one below the union: the probe overflows, then exact windows.
    _history(engine, lengths=(200, 160))
    r2, _ = engine.union_step(_mixed_rows(engine, chunk=40), capacity_delta=-1)
    assert r2.level == "l2" and r2.windows + r2.l3_groups >= 2
    assert r2.syncs == 1 + r2.windows + r2.l3_groups
    assert engine.store.union_stats["probe_overflows"] == 1


def test_fast_path_bound_is_checked_on_cpu():
    engine = UnionEngine(_shape(rows_per_pass=8, blocks=40), seed=5)
    _history(engine)
    with pytest.raises(RuntimeError, match="fast path"):
        engine.union_step(_mixed_rows(engine), fast_path=True, capacity=20)


def _gids(engine, table, seq_lens, token_to_req, selection, rows=None, tails=True):
    """Brute-force union of the referenced gids (tails of every request)."""
    world, rank = engine.world, engine.rank

    def local(length):
        return length // world + (1 if length % world > rank else 0)

    def gid(req, token):
        return int(table[req, token // KBS]) * (KBS // 4) + token % KBS // 4

    tail: set[int] = set()
    sel: dict[int, int] = {}
    if tails:
        for req, length in enumerate(seq_lens.tolist()):
            n = local(length)
            tail |= {gid(req, t) for t in range(max(0, n - 32), n)}
    for row in range(selection.shape[0]) if rows is None else rows:
        req = int(token_to_req[row])
        for token in selection[row].tolist():
            if token >= 0:
                sel.setdefault(gid(req, token), row)
    return tail, sel


def _state(layer):
    return [
        x.clone()
        for x in (layer.group_to_slot, layer.slot_owner, layer.age, layer.clock)
    ]


def test_overflowing_probe_restores_every_state_bit():
    # A pool far below one step's union: the probe count exceeds M (the
    # Touched capacity). RESET must clear every claim; nothing else moves.
    engine = UnionEngine(_shape(rows_per_pass=1, requests=2, blocks=40), seed=2)
    _history(engine, lengths=(200, 160))
    engine.union_step(_mixed_rows(engine, chunk=4))  # warm some slots
    rows = _mixed_rows(engine, chunk=30)
    _, table, seq_lens, token_to_req, selection, _ = engine.inputs(rows)
    layer, store = engine.layer, engine.store
    req_class = torch.ones(seq_lens.shape[0], dtype=torch.int32)
    scan = torch.zeros(64, dtype=torch.int32)
    tail, sel = _gids(engine, table, seq_lens, token_to_req, selection)
    assert len(tail | set(sel)) > store.max_misses
    layer._union_clear()
    before = _state(layer)
    assert int((store.wanted != 0).sum()) == 0
    args = (selection, token_to_req, table, seq_lens, req_class, scan)
    layer._union_scan(hk.UNION_MARK, *args)
    # The count is exact even past M; Touched holds only its first M.
    assert int(store.wanted_count) == len(tail | set(sel))
    assert int((store.wanted != 0).sum()) == len(tail | set(sel))
    layer._union_scan(hk.UNION_RESET, *args)
    store.wanted_count.zero_()
    assert int((store.wanted != 0).sum()) == 0
    for x, y in zip(before, _state(layer)):
        assert torch.equal(x, y)
    # The next union step still matches the GPU arm (no leftover marks).
    engine.union_step(_mixed_rows(engine, chunk=6))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_first_reference_counts_equal_brute_force(seed):
    engine = UnionEngine(_shape(rows_per_pass=4, blocks=40), seed=seed)
    _history(engine, lengths=(150, 120))
    rows = _mixed_rows(engine, decode=("a",), chunk=37)
    _, table, seq_lens, token_to_req, selection, _ = engine.inputs(rows)
    layer, store = engine.layer, engine.store
    req_class = torch.ones(seq_lens.shape[0], dtype=torch.int32)
    scan = torch.zeros(64, dtype=torch.int32)
    n = selection.shape[0]
    groups8 = -(-n // 8)
    layer._union_clear()
    for g in range(groups8):
        start, remaining = 8 * g, groups8 - g
        args = (
            selection[start:],
            token_to_req[start:],
            table,
            seq_lens,
            req_class,
            scan,
            remaining,
        )
        scan.zero_()
        for mode in (hk.UNION_FIRST, hk.UNION_COUNT, hk.UNION_RESET):
            layer._union_scan(mode, *args)
        assert int((store.wanted != 0).sum()) == 0
        tail, sel = _gids(
            engine, table, seq_lens, token_to_req, selection, range(start, n)
        )
        first = [0] * remaining
        for gid, row in sel.items():
            if gid not in tail:
                first[(row - start) // 8] += 1
        assert scan[: remaining + 1].tolist() == [len(tail)] + first
        # The greedy window's union is exactly tail + prefix sum.
        for k in range(1, remaining + 1):
            exact = tail | {
                gid for gid, row in sel.items() if row < min(start + 8 * k, n)
            }
            assert len(exact) == len(tail) + sum(first[:k])


@pytest.mark.parametrize("seed", range(6))
def test_exact_windows_never_exceed_the_cpu_bound_windows(seed):
    rng = random.Random(seed)
    engine = UnionEngine(_shape(rows_per_pass=6, blocks=60), seed=seed)
    _history(engine, lengths=(rng.randint(80, 300), rng.randint(80, 300)))
    rows = _mixed_rows(engine, decode=("a",), chunk=rng.randint(9, 40))
    capacity = rng.randint(3 * WIDTH + 18, 6 * WIDTH + 18)
    result, (reqs, *_rest) = engine.union_step(rows, capacity=capacity)
    rows_by_req = [sum(1 for r, _ in rows if r == req) for req in reqs]
    groups = [history_groups(engine.length[req]) for req in reqs]
    upper = plan_union_windows_upper(
        rows_by_req, groups, capacity, row_groups=WIDTH, tail_groups=9
    )
    if upper is not None and result.l3_groups == 0:
        assert result.windows <= len(upper)
    if result.level == "l1":
        assert len(upper or []) >= 1


@pytest.mark.parametrize("rows_dec", [1, 8])
def test_l3_only_when_one_group_exceeds_the_floor(rows_dec):
    # Hot pool at the frozen floor F = sum_q min(G, rows_dec*rg + 9).
    shape = _shape(rows_per_pass=rows_dec, blocks=60, decode_rows=rows_dec)
    engine = UnionEngine(shape, seed=9)
    _history(engine, lengths=(300, 280))
    total_l3 = 0
    for _ in range(3):
        result, _ = engine.union_step(_mixed_rows(engine, chunk=21))
        total_l3 += result.l3_groups
        assert result.spans[-1][1] == 22  # progress over every row
    if rows_dec == 8:
        assert total_l3 == 0
    else:
        assert total_l3 > 0


def test_l3_between_normal_windows_and_short_tail_group():
    # Normal window -> L3 -> normal window, ending with a short group: a
    # capacity that fits most eight-row groups but not a dense one.
    engine = UnionEngine(_shape(rows_per_pass=8, blocks=60), seed=4)
    _history(engine, lengths=(300, 280))
    seen = set()
    for capacity in range(2 * WIDTH + 18, 8 * WIDTH + 18, 6):
        result, _ = engine.union_step(
            _mixed_rows(engine, chunk=27), capacity=capacity, max_window_groups=1
        )
        kinds = [kind for _, _, kind in result.spans]
        for i in range(1, len(kinds) - 1):
            if kinds[i] == "rows" and kinds[i - 1] == kinds[i + 1] == "union":
                seen.add("middle")
        if result.spans[-1][1] - result.spans[-1][0] < 8:
            seen.add("short-tail")
    assert seen == {"middle", "short-tail"}


@pytest.mark.parametrize("policy", ["old", "recent"])
def test_class_bits_and_insertion_age(policy):
    shape = _shape(rows_per_pass=8, blocks=40, prefill_insert_age=policy)
    engine = UnionEngine(shape, seed=6)
    _history(engine)
    engine.union_step(_mixed_rows(engine))
    layer = engine.layer
    age_before = layer.age.clone()
    owner_before = layer.slot_owner.clone()
    rows = _mixed_rows(engine)
    # Request 0 ("a") is decode class (bits 2), request 1 prefill (bits 1).
    result, (reqs, table, seq_lens, token_to_req, selection) = engine.union_step(
        rows, req_class=torch.tensor([2, 1], dtype=torch.int32)
    )
    assert reqs == ["a", "b"] and result.level == "l1"
    clock = int(layer.clock)
    decode_rows = [i for i, r in enumerate(token_to_req.tolist()) if r == 0]
    prefill_rows = [i for i, r in enumerate(token_to_req.tolist()) if r == 1]
    tail_a, sel_a = _gids(
        engine, table[:1], seq_lens[:1], token_to_req, selection, decode_rows
    )
    tail_b, sel_b = _gids(
        engine, table, seq_lens, token_to_req, selection, prefill_rows, tails=False
    )
    # Request b's tail is prefill class as well.
    tail_bb, _ = _gids(engine, table[1:], seq_lens[1:], token_to_req, selection, [])
    decode = tail_a | set(sel_a)
    prefill_only = (tail_bb | set(sel_b)) - decode
    assert prefill_only and decode
    wanted = engine.store.wanted
    for gid in decode:
        assert int(wanted[gid]) in (2, 3) if policy == "old" else 1
        assert int(layer.age[layer.group_to_slot[gid]]) == clock
    old_age = {int(o): int(a) for o, a in zip(owner_before, age_before) if o >= 0}
    for gid in prefill_only:
        slot = int(layer.group_to_slot[gid])
        if policy == "recent":
            assert int(layer.age[slot]) == clock
        elif gid in old_age:  # hit: never younger than before
            assert int(layer.age[slot]) == max(old_age[gid], clock - 9)
        else:  # miss: inserted old
            assert int(layer.age[slot]) == clock - 9
    counters = engine.store.union_scratch["counters"]
    assert int(counters[0]) > 0 and int(counters[1]) > 0
    assert (int(counters[2]) > 0) == (policy == "old")


@pytest.mark.parametrize("mutation", ["alignment", "masking"])
def test_mutations_are_caught(monkeypatch, mutation):
    # The grouped stand-in folds each aligned eight-row group's dedup set
    # into the result and rejects live rows reading non-resident slots.
    engine = UnionEngine(_shape(rows_per_pass=8, blocks=60), grouped_mock=True)
    _history(engine, lengths=(300, 280))
    capacity = 3 * WIDTH + 18
    # Unmutated: every split passes.
    for groups in (1, 2):
        engine.union_step(
            _mixed_rows(engine, chunk=27), capacity=8 * WIDTH + 18,
            max_window_groups=groups,
        )  # fmt: skip
    if mutation == "alignment":
        monkeypatch.setattr(history_mod, "_UNION_ROW_GROUP", 5)
    else:
        monkeypatch.setattr(history_mod, "_MASK_OTHER_WINDOWS", False)
    with pytest.raises(AssertionError):
        for _ in range(3):
            engine.union_step(
                _mixed_rows(engine, chunk=27), capacity=capacity,
                max_window_groups=1,
            )  # fmt: skip


def test_union_bill_items():
    from vllm.models.qwen4_exp.nvidia.host_kv.budget import (
        union_gather_memory_bytes,
    )

    bill = union_gather_memory_bytes(
        table_rows=8,
        table_tokens=1 << 19,
        max_tokens=8192,
        max_requests=8,
        output_heads=12,
        head_dim=256,
        dcp_world_size=2,
    )
    assert bill == {
        "microblock_table": 4 << 20,
        "union_scan": 4 * 8193,
        "union_counters": 32,
        "masked_requests": 4 * 8192,
        "request_classes": 32,
        "window_output": 8192 * 12 * 256 * 4,
        "window_lse": 8192 * 12 * 4,
        "grouped_plan_workspace": 1024 * 4160 * 8,
        "grouped_slice_output": 8192 * 6 * 256 * 2,
    }
    # Production geometry: about 157 MiB, 251 history IDs of 655,832 B.
    assert -(-sum(bill.values()) // 655_832) == 251
    plain = union_gather_memory_bytes(
        table_rows=2, table_tokens=64, max_tokens=16, max_requests=2,
        output_heads=2, head_dim=16,
    )  # fmt: skip
    assert plain["window_lse"] == plain["grouped_slice_output"] == 0
    assert plain["window_output"] == 16 * 2 * 16 * 2


def test_store_bill_is_unchanged_by_the_union_scratch():
    engine = UnionEngine(_shape(rows_per_pass=8, blocks=40))
    before = engine.store.fixed_device_memory_bytes()
    _history(engine)
    engine.union_step(_mixed_rows(engine))
    assert engine.store.union_scratch
    assert engine.store.fixed_device_memory_bytes() == before
    assert before == engine.shape.fixed_device_memory_bytes()


def test_union_rejects_graph_capture_and_bad_capacity():
    engine = UnionEngine(_shape(rows_per_pass=8, blocks=40))
    _history(engine)
    with pytest.raises(ValueError, match="union tensors"):
        engine.union_step(_mixed_rows(engine), capacity=engine.shape.hot_slots + 1)


def test_cpu_planning_helpers():
    assert union_bound([0, 3], [5, 10_000]) == 5 + 3 * 513 + 9
    assert union_fast_path([1, 0], [600, 4], 522 + 4)
    assert not union_fast_path([1, 0], [600, 4], 522 + 3)
    # Groups of eight rows, every tail charged even without rows.
    windows = plan_union_windows_upper([2, 20], [10**6, 10**6], 8 * 513 + 18)
    assert plan_union_windows_upper([2, 20], [1000, 1000], 2000) == [(0, 22)]
    assert windows == [(0, 8), (8, 16), (16, 22)]
    assert plan_union_windows_upper([9], [10**6], 513 + 9) is None
    assert union_window_groups(5, [3, 4, 1, 9], 13) == 3
    assert union_window_groups(5, [3, 4, 1, 9], 13, max_groups=2) == 2
    assert union_window_groups(5, [9], 13) == 0


def test_step_info_drives_capacity_fast_path_and_classes(monkeypatch):
    engine = UnionEngine(_shape(rows_per_pass=8, blocks=40), seed=3)
    _history(engine)
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return history_mod.UnionWindowResult("l1", 1, 1, 0, [])

    monkeypatch.setattr(history_mod, "for_each_union_window", spy)
    rows = _mixed_rows(engine)
    _, table, seq_lens, token_to_req, selection, _ = engine.inputs(rows)
    store = engine.store
    store.union_step = UnionStepInfo(
        capacity=50, rows_by_req=(1, 19), groups=(600, 600), decode=(True, False)
    )
    history_mod.run_union_windows(
        engine.layer, selection, token_to_req, table, seq_lens, [], None, None
    )
    assert seen["capacity"] == 50 and seen["fast_path"] is False
    assert seen["req_class"].tolist() == [2, 1]
    store.union_step = UnionStepInfo(capacity=50, rows_by_req=(1, 19), groups=(9, 30))
    history_mod.run_union_windows(
        engine.layer, selection, token_to_req, table, seq_lens, [], None, None
    )
    assert seen["fast_path"] is True
    store.union_step = UnionStepInfo(capacity=10_000)
    history_mod.run_union_windows(
        engine.layer, selection, token_to_req, table, seq_lens, [], None, None
    )
    assert seen["capacity"] == store.shape.hot_slots
    # Without step info: static capacity, classes from row counts.
    store.union_step = None
    history_mod.run_union_windows(
        engine.layer, selection, token_to_req, table, seq_lens, [], None, None
    )
    assert seen["req_class"].tolist() == [2, 1]


def test_unbound_page_is_rejected_and_null_block_reads_zeros():
    engine = UnionEngine(_shape(rows_per_pass=8, blocks=40), seed=8)
    _history(engine)
    rows = _mixed_rows(engine)
    _, table, seq_lens, token_to_req, selection, _ = engine.inputs(rows)
    layer, store = engine.layer, engine.store
    # Null block: point request a's first block at ID 0 (page 0, all ready,
    # never written). Its groups fill with zeros.
    null_table = table.clone()
    null_table[0, 0] = 0
    req_class = torch.ones(2, dtype=torch.int32)
    scan = torch.zeros(64, dtype=torch.int32)
    layer._union_clear()
    layer._union_scan(
        hk.UNION_MARK, selection, token_to_req, null_table, seq_lens, req_class, scan
    )
    hk.begin_fill[(1,)](store.miss_count, layer.clock)
    layer._fill_from_marks()
    for g in range(KBS // 4):
        slot = int(layer.group_to_slot[g])
        if slot >= 0:
            assert not layer.hot[slot].any()
    # An unbound page (page_of = -1) of a selected block is rejected.
    victim = int(table[1, 0])
    store.page_of[victim] = -1
    with pytest.raises(ValueError, match="unbound host page"):
        engine.union_step(rows)


# --------------------------------------------------------------------------
# shared hot region shared region: hot slots alias DONATED index pages, U = available
# slots.


def _shared_union_engine(seed=7, blocks=60, **kwargs):
    from .test_host_kv_shared_region import SharedEngine, _shared_shape

    class SharedUnionEngine(SharedEngine, UnionEngine):
        def __init__(self, shape, **kw):
            SharedEngine.__init__(self, shape, **kw)
            self.grouped_mock = False
            self.results = []

        def union_step(self, rows, **kw):
            self.scribble()
            kw.setdefault("capacity", self.store.available_slots)
            out = UnionEngine.union_step(self, rows, **kw)
            self.check_roles()
            return out

    shape = _shared_shape(blocks=blocks, prefill_gather="union", **kwargs)
    return SharedUnionEngine(shape, seed=seed)


@pytest.mark.parametrize("groups", [None, 1, 2])
def test_shared_region_union_windows_match_gpu_arm(groups):
    engine = _shared_union_engine(seed=3)
    _history(engine, lengths=(120, 90))
    for _ in range(3):
        result, _ = engine.union_step(
            _mixed_rows(engine, chunk=19), max_window_groups=groups
        )
        assert result.spans[-1][1] == 20
    # A finished request donates its IDs; U grows; claims unmap residents.
    engine.finish("b")
    _history(engine, reqs=("c",), lengths=(70,))
    engine.union_step(_mixed_rows(engine, decode=("a",), prefill="c", chunk=17))


def test_shared_region_union_under_u_step_floor():
    # U at the floor of one row plus tails (rows_dec = 1): L3 keeps progress
    # and never takes a reserved slot.
    engine = _shared_union_engine(seed=4, blocks=40)
    _history(engine, lengths=(150, 130))
    floor = WIDTH + 2 * 9
    result, _ = engine.union_step(_mixed_rows(engine, chunk=21), capacity=floor)
    assert result.l3_groups > 0 and result.spans[-1][1] == 22


def test_plan_bills_union_transients_only_for_the_union_arm():
    from types import SimpleNamespace

    from vllm.v1.core import long_context_kv as lc

    geometry = SimpleNamespace(
        dcp_world_size=2,
        dcp_interleave=1,
        block_tokens=256,
        max_model_len=1 << 20,
        hot_query_heads=6,
        hot_head_dim=256,
        hot_max_write_rows=8192,
    )
    bill = lc.union_gather_bytes(geometry, 8)  # type: ignore[arg-type]
    # 4 MiB table (+1 block), 96 MiB window output, LSE, grouped workspace.
    assert 156 << 20 < bill < 158 << 20


def test_runner_hands_union_metadata_to_the_store_for_mixed_steps():
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    for gather, mixed in (("union", True), ("union", False), ("passes", True)):
        shape = _shape(rows_per_pass=8, blocks=40, prefill_gather=gather)
        store = HostHistoryStore(shape, ["l0", "l1"], device="cpu")
        runner = SimpleNamespace(host_history=store)
        rows = {"x": (1, 30, 0), "y": (19 if mixed else 1, 40, 1 if mixed else 0)}
        output = SimpleNamespace(host_kv=SimpleNamespace(requests=rows, hot_slots=0))
        batch = SimpleNamespace(
            req_ids=["y", "x"], num_tokens=sum(r[0] for r in rows.values())
        )
        GPUModelRunner._set_host_kv_plan(runner, output, batch)  # type: ignore[arg-type]
        if gather == "union" and mixed:
            assert store.union_step == UnionStepInfo(
                capacity=store.available_slots,
                rows_by_req=(19, 1),
                groups=(40, 30),
                decode=(False, True),
            )
            assert store.step_passes is None
        else:
            assert store.union_step is None
        # A dummy batch clears it.
        store.set_step_plan(None, None)
        assert store.union_step is None


def test_union_windows_feed_the_p4_shadow_once_per_fill():
    from vllm.v1.core.value_arbiter import HOT_GATHERS, HOT_MISSES

    shape = _shape(rows_per_pass=8, blocks=60)
    engine = UnionEngine(shape, seed=12)
    engine.store = HostHistoryStore(
        shape, ["l0", "l1"], device="cpu", value_shadow=True
    )
    engine.store.bind_pages(
        [(hid, shape.num_blocks - hid) for hid in range(1, shape.num_blocks)]
    )
    # The traced layer is layer 0, the one the shadow observes.
    engine.layer = engine.store.layer("l0")
    _history(engine, lengths=(200, 160))
    counters = engine.store.shadow.counters
    before = counters.clone()
    result, _ = engine.union_step(_mixed_rows(engine, chunk=27), max_window_groups=1)
    fills = result.windows + sum(e - s for s, e, k in result.spans if k == "rows")
    assert int(counters[HOT_GATHERS] - before[HOT_GATHERS]) == fills
    assert int(counters[HOT_MISSES] - before[HOT_MISSES]) > 0
