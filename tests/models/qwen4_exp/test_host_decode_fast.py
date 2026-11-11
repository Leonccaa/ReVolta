# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interpreter gates for the opt-in decode kernels; CUDA gates are separate."""

import os
from functools import partial

import pytest
import torch

# A CPU harness must import vllm from this worktree (the run_cpu.py script
# next to the venv inserts its own directory first on sys.path). Fail fast
# with the imported path instead of per-test AttributeErrors when it does not.
from vllm.models.qwen4_exp.nvidia.host_kv import history as _history_module

from .test_host_kv_history import (
    Engine,
    _bits,
    _interpreter_boolean_comparison,
    _shape,
)

if not hasattr(_history_module, "ORDERED_SUB_SWITCH_ENV"):
    raise ImportError(
        "the ordered fused-mark sub-switch is missing from "
        f"{_history_module.__file__}; run this gate with a harness whose "
        "sys.path[0] is this worktree"
    )

__all__ = ["_interpreter_boolean_comparison"]
pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="CPU interpreter gate"
)


@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1)])
def test_decode_growth_eviction_reuse_matches_paged_bytes_and_attention(
    monkeypatch, e4m3, world, rank
):
    monkeypatch.setenv("REVOLTA_RESEARCH_HOST_DECODE_FAST", "1")
    engine = Engine(
        _shape(
            e4m3,
            blocks=65,
            rows_per_pass=2,
            requests=2,
            dcp_world_size=world,
            dcp_rank=rank,
        ),
        dcp=(world, rank),
        seed=17,
    )
    for req in ("a", "b"):
        engine.write(req, list(range(120)))
    # Model the real entry: only decode writes and single-pass gathers opt in.
    # The real parent QSA forward keeps its original call signature.
    # Small writes must therefore select fusion without an explicit hint.
    engine.layer.gather = partial(engine.layer.gather, decode=True)
    misses = []
    for _ in range(6):
        for req in ("a", "b"):
            engine.write(req, [engine.length[req]])
        engine.step([(r, engine.length[r] - 1) for r in ("a", "b")])
        misses.append(int(engine.store.miss_count))
    assert sum(misses) > 0  # Cold and replacement reads, not just hit-only.
    old = engine.finish("a")
    engine.write("c", list(range(20)))
    assert engine.blocks["c"][0] == old[0]
    engine.step([("b", engine.length["b"] - 1), ("c", 19)])
    engine.store.close()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("c", [1, 3, 4, 8])
def test_fused_shard_metadata_matches_torch_for_padding_and_boundaries(
    monkeypatch, dtype, c
):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _shared_key_local_inputs

    positions = torch.tensor(
        [-100, -c - 1, -c, -2, -1, 0, c - 1, c, 32767, 32768, 2**30],
        dtype=dtype,
    )
    lengths = torch.tensor([-99, 0, c - 1, c, 32768, 2**30], dtype=dtype)
    for world in (2, 3, 4):
        for rank in range(world):
            monkeypatch.setenv("REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED", "0")
            expected = _shared_key_local_inputs(positions, lengths, c, world, rank)
            monkeypatch.setenv("REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED", "1")
            actual = _shared_key_local_inputs(positions, lengths, c, world, rank)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual))
            assert actual[0].dtype == dtype and actual[1].dtype == dtype


def test_fused_shard_metadata_empty_and_noncontiguous_fallback(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _shared_key_local_inputs

    monkeypatch.setenv("REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED", "1")
    for positions, lengths in (
        (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int32)),
        (torch.arange(20, dtype=torch.int64)[::2], torch.tensor([32])),
    ):
        actual = _shared_key_local_inputs(positions, lengths, 4, 2, 1)
        monkeypatch.setenv("REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED", "0")
        expected = _shared_key_local_inputs(positions, lengths, 4, 2, 1)
        monkeypatch.setenv("REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED", "1")
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))


class _CommitKernelSpy:
    """Count ``commit_slots`` launches (the unfused publish path)."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = 0

    def __getitem__(self, grid):
        self.calls += 1
        return self.kernel[grid]


def _four_same_group_engine(monkeypatch, spy, fast):
    """Write one hot group's four tokens in a single small chunk.

    ``max_requests=4`` makes the parent-free heuristic (a write of at most
    ``max_requests * decode_rows`` rows) select the fused publish, exactly
    like a production prefill chunk of that size hits the fused write path
    while ``REVOLTA_RESEARCH_HOST_DECODE_FAST=1`` (review N2).
    """
    monkeypatch.setenv("REVOLTA_RESEARCH_HOST_DECODE_FAST", "1" if fast else "0")
    spy.calls = 0
    engine = Engine(_shape(True, blocks=40, rows_per_pass=2, requests=4), seed=29)
    assert engine.store.decode_fast is fast
    engine.write("a", list(range(4)))  # slots share one four-token group
    assert engine.store.ready[[engine.slot("a", p) for p in range(4)]].tolist() == [
        1,
        1,
        1,
        1,
    ]
    assert spy.calls == (0 if fast else 1)  # the fused publish replaced commit
    engine.step([("a", 3)])  # gather the group hot (unfused read path)
    engine.write("a", list(range(4)))  # rewrite: the hot-group refresh path
    engine.step([("a", 3)])
    return engine


def test_fused_write_of_four_same_group_tokens_publishes_like_baseline(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.host_kv import history_kernels as hk

    spy = _CommitKernelSpy(hk.commit_slots)
    monkeypatch.setattr(hk, "commit_slots", spy)
    baseline = _four_same_group_engine(monkeypatch, spy, fast=False)
    fused = _four_same_group_engine(monkeypatch, spy, fast=True)
    # Same trace, both publish paths: readiness, host pages and the hot pool
    # (including the in-kernel hot refresh) are bit-identical. Each step also
    # asserted bit-identical attention against the paged cache.
    assert torch.equal(fused.store.ready, baseline.store.ready)
    assert torch.equal(_bits(fused.layer.host_view), _bits(baseline.layer.host_view))
    assert torch.equal(_bits(fused.store.hot), _bits(baseline.store.hot))
    assert fused.misses == baseline.misses > 0
    fused.store.close()
    baseline.store.close()


def _switch_path_trace(monkeypatch, orig_run, outs, fast):
    """Rebuild the store under ``REVOLTA_RESEARCH_HOST_DECODE_FAST`` and run
    the decode/eviction/reuse trace, recording every attention output."""
    monkeypatch.setenv("REVOLTA_RESEARCH_HOST_DECODE_FAST", "1" if fast else "0")
    engine = Engine(_shape(True, blocks=65, rows_per_pass=2, requests=2), seed=17)

    def record(self, *args, **kwargs):
        out = orig_run(self, *args, **kwargs)
        outs.append((out.clone(), self.lse.clone()))
        return out

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.host_kv.attention.SparseAttentionPlan.run",
        record,
    )
    engine.layer.gather = partial(engine.layer.gather, decode=True)
    for req in ("a", "b"):
        engine.write(req, list(range(120)))
    misses = []
    for _ in range(6):
        for req in ("a", "b"):
            engine.write(req, [engine.length[req]])
        engine.step([(r, engine.length[r] - 1) for r in ("a", "b")])
        misses.append(int(engine.store.miss_count))
    old = engine.finish("a")
    engine.write("c", list(range(20)))
    assert engine.blocks["c"][0] == old[0]  # reused IDs retire and rewrite
    engine.step([("b", engine.length["b"] - 1), ("c", 19)])
    assert sum(misses) > 0
    return engine


def test_switch_off_then_on_same_inputs_matches_bits_after_store_rebuild(
    monkeypatch,
):
    """Review N3: the fused and unfused paths over one rebuilt store.

    The same trace runs with the switch off, then with it on after
    ``monkeypatch.setenv`` and a fresh store (the switch is read at store
    construction); ``ready``, the host pages and every attention output and
    LSE must match bit for bit. Hot-slot identities may differ (fused marks
    interleave claims), which is why outputs -- not slot numbers -- compare.
    """
    from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan

    orig_run = SparseAttentionPlan.run
    slow_outs: list = []
    fast_outs: list = []
    slow = _switch_path_trace(monkeypatch, orig_run, slow_outs, fast=False)
    fast = _switch_path_trace(monkeypatch, orig_run, fast_outs, fast=True)
    assert fast.store.decode_fast and not slow.store.decode_fast
    assert len(slow_outs) == len(fast_outs) > 0
    for (want, want_lse), (got, got_lse) in zip(slow_outs, fast_outs):
        assert torch.equal(_bits(want), _bits(got))
        assert torch.equal(_bits(want_lse), _bits(got_lse))
    assert torch.equal(fast.store.ready, slow.store.ready)
    assert torch.equal(_bits(fast.layer.host_view), _bits(slow.layer.host_view))
    fast.store.close()
    slow.store.close()


def _ordered_marks_trace(monkeypatch, *, fast, ordered, world=1, rank=0):
    """Run the decode trace under the fused mark kernel and snapshot, after
    every step, the effective Touched stream, the Misses list and the whole
    eviction state (mapping, owners, ages, hand, ready)."""
    monkeypatch.setenv("REVOLTA_RESEARCH_HOST_DECODE_FAST", "1" if fast else "0")
    monkeypatch.setenv(
        "REVOLTA_RESEARCH_HOST_DECODE_FAST_ORDERED", "1" if ordered else "0"
    )
    engine = Engine(
        _shape(
            True,
            blocks=65,
            rows_per_pass=2,
            requests=2,
            dcp_world_size=world,
            dcp_rank=rank,
        ),
        dcp=(world, rank),
        seed=17,
    )
    assert engine.store.decode_fast_ordered is (fast and ordered)
    engine.layer.gather = partial(engine.layer.gather, decode=True)
    for req in ("a", "b"):
        engine.write(req, list(range(120)))
    steps = []
    for _ in range(6):
        for req in ("a", "b"):
            engine.write(req, [engine.length[req]])
        engine.step([(r, engine.length[r] - 1) for r in ("a", "b")])
        store = engine.store
        tail = store.wanted_groups[: int(store.wanted_count)].tolist()
        if store.wanted_sel is None:
            sel: list[int] = []
        else:
            sel = store.wanted_sel[: int(store.sel_count)].tolist()
        # The ordered collect treats a selection claim of a tail-wanted group
        # as the tail entry's duplicate; mirror that to get the stream the
        # refill actually consumed (the plain layout keeps everything in one).
        owned = [(int(store.wanted[g]) & 1) != 0 for g in sel]
        stream = tail + [g for g, o in zip(sel, owned) if not o]
        steps.append(
            (
                stream,
                store.miss_groups[: int(store.miss_count)].tolist(),
                engine.layer.group_to_slot.clone(),
                engine.layer.slot_owner.clone(),
                engine.layer.age.clone(),
                engine.layer.eviction_hand.clone(),
                store.ready.clone(),
            )
        )
    engine.store.close()
    return steps


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 1)])
def test_ordered_fused_marks_keep_baseline_touched_and_miss_order(
    monkeypatch, world, rank
):
    """Review N1: with the ordered sub-switch the fused mark kernel keeps the
    baseline's Touched order (tail claims first), so the Misses/victims
    pairing -- and the whole cache evolution -- matches the unfused path
    item for item under the interpreter."""
    baseline = _ordered_marks_trace(
        monkeypatch, fast=False, ordered=False, world=world, rank=rank
    )
    ordered = _ordered_marks_trace(
        monkeypatch, fast=True, ordered=True, world=world, rank=rank
    )
    assert len(baseline) == len(ordered) > 0
    assert any(misses for _, misses, *_ in baseline)
    for want, got in zip(baseline, ordered):
        assert want[0] == got[0]  # Touched stream, item for item
        assert want[1] == got[1]  # Misses order, item for item
        for w, g in zip(want[2:], got[2:]):
            assert torch.equal(w, g)  # mapping, owners, ages, hand, ready


def test_ordered_sub_switch_requires_parent_switch(monkeypatch):
    monkeypatch.setenv("REVOLTA_RESEARCH_HOST_DECODE_FAST", "0")
    monkeypatch.setenv("REVOLTA_RESEARCH_HOST_DECODE_FAST_ORDERED", "1")
    engine = Engine(_shape(True, blocks=33), seed=5)
    assert not engine.store.decode_fast
    assert not engine.store.decode_fast_ordered
    engine.store.close()
