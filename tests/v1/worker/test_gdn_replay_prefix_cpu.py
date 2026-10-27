# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN speculative state replay over the accepted prefix only (CPU).

Each verifier row is replayed as two sequences over the same recorded rows: the
accepted prefix with the row's state table, then the rejected tail with an
all-padding row. The index tests check, with a model of the recurrent kernel's
index logic, that every stored state comes from the same start block and the
same token rows as in the full replay. The interpreter tests run the Triton table
kernel and the real recurrent kernel (TRITON_INTERPRET=1) and compare the state
pools of the full and the accepted-prefix replay bitwise.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.gdn_spec_replay import (
    GDNSpecReplayLayer,
    GDNSpecReplayStep,
    build_replay_state_indices,
    replay_graph_key,
    split_replay_launch,
)

pytestmark = pytest.mark.cpu_test

PAD = PAD_SLOT_ID


def _kernel_writes(table, cu_seqlens, num_columns):
    """What the recurrent kernel stores, following its index logic.

    fused_sigmoid_gating_delta_rule_update_kernel: a sequence with no tokens
    returns; the start state is the table entry at the selector column (the
    replay selector is num_columns, i.e. the last column) and a padded start
    returns; token t is stored at its column when that entry is not padded.
    Returns {block: (start block, first token row, tokens processed)}: the
    stored state is a function of exactly these.
    """
    writes = {}
    for seq in range(table.shape[0]):
        bos, eos = int(cu_seqlens[seq]), int(cu_seqlens[seq + 1])
        if eos - bos == 0:
            continue
        start = int(table[seq, num_columns - 1])
        if start < 0:
            continue
        for t in range(eos - bos):
            dst = int(table[seq, t])
            if dst >= 0:
                assert dst not in writes, "one block stored twice in a launch"
                writes[dst] = (start, bos, t + 1)
    return writes


def _tokens_processed(table, cu_seqlens, num_columns):
    total = 0
    for seq in range(table.shape[0]):
        n = int(cu_seqlens[seq + 1]) - int(cu_seqlens[seq])
        if n > 0 and int(table[seq, num_columns - 1]) >= 0:
            total += n
    return total


def test_split_launch_hand_example():
    # Three verifier rows of 8, 8 and 5 tokens accepting 3, 8 and 1.
    table = torch.full((4, 9), PAD, dtype=torch.int32)
    table[:, -1] = torch.tensor([7, 9, 11, PAD])
    table[0, 2] = 7
    table[0, 0] = 100  # a boundary inside the prefix
    table[1, 7] = 9
    table[2, 0] = 11
    cu = torch.tensor([0, 8, 16, 21, 21], dtype=torch.int32)
    split_table, split_cu = split_replay_launch(
        table, cu, torch.tensor([3, 8, 1], dtype=torch.int32)
    )
    assert split_cu.dtype == torch.int32
    # Row 3 is graph padding: no prefix, its (empty) tail after it.
    assert split_cu.tolist() == [0, 3, 8, 16, 16, 17, 21, 21, 21]
    assert torch.equal(split_table[0::2], table)
    assert (split_table[1::2] == PAD).all()
    assert _kernel_writes(split_table, split_cu, 9) == _kernel_writes(table, cu, 9)
    assert _kernel_writes(table, cu, 9) == {
        100: (7, 0, 1),
        7: (7, 0, 3),
        9: (9, 8, 8),
        11: (11, 16, 1),
    }
    assert _tokens_processed(split_table, split_cu, 9) == 12
    assert _tokens_processed(table, cu, 9) == 21


def _random_case(gen, *, num_columns, block_size, full_graph):
    """A verifier launch, its replay table and accepted counts."""
    slots = num_columns - 1
    if full_graph:
        num_reqs = int(torch.randint(1, 9, (1,), generator=gen))
        num_spec = int(torch.randint(1, num_reqs + 1, (1,), generator=gen))
        batch = num_spec
        spec_rows = torch.arange(num_spec)
        width = int(torch.randint(1, slots + 1, (1,), generator=gen))
        lengths = torch.full((num_reqs,), width)
        # Padding rows carry the padded tokens or none.
        if bool(torch.rand(1, generator=gen) < 0.5):
            lengths[num_spec:] = 0
    else:
        num_reqs = int(torch.randint(1, 9, (1,), generator=gen))
        num_spec = num_reqs
        batch = num_reqs + int(torch.randint(0, 3, (1,), generator=gen))
        spec_rows = torch.randperm(batch, generator=gen)[:num_spec].sort().values
        lengths = torch.randint(1, slots + 1, (num_reqs,), generator=gen)
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(lengths, 0)
    lens = lengths[:num_spec]
    # Accepted counts from 1 to the full row, with both ends forced often.
    accepted = (torch.rand(num_spec, generator=gen) * lens).long() + 1
    pick = torch.rand(num_spec, generator=gen)
    accepted = torch.where(pick < 0.2, torch.ones_like(accepted), accepted)
    accepted = torch.where(pick > 0.8, lens, accepted).to(torch.int32)
    computed = accepted + torch.randint(0, 6 * block_size, (num_spec,), generator=gen)
    perm = torch.randperm(1000, generator=gen).to(torch.int32)
    base = torch.full((num_reqs,), PAD, dtype=torch.int32)
    base[:num_spec] = perm[:num_spec]
    width_bt = (6 * block_size + 8) // block_size + 2
    block_table = perm[num_spec : num_spec + batch * width_bt].view(batch, width_bt)
    table = build_replay_state_indices(
        base_state_ids=base,
        num_reqs=num_reqs,
        num_columns=num_columns,
        spec_rows=spec_rows,
        num_accepted=accepted,
        num_computed_tokens=computed,
        block_table=block_table,
        block_size=block_size,
    )
    return table, cu, accepted, lens


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("block_size", [2, 3, 4, 16, 1664])
@pytest.mark.parametrize("num_columns", [5, 9])
def test_split_launch_stores_the_same_states(full_graph, block_size, num_columns):
    gen = torch.Generator().manual_seed(block_size * 31 + num_columns)
    seen = dict(boundary_in_prefix=0, truncated=0, accepted_all=0, accepted_one=0)
    for _ in range(150):
        table, cu, accepted, lens = _random_case(
            gen, num_columns=num_columns, block_size=block_size, full_graph=full_graph
        )
        split_table, split_cu = split_replay_launch(table, cu, accepted)
        num_reqs = table.shape[0]
        assert split_table.shape == (2 * num_reqs, num_columns)
        assert split_cu.shape == (2 * num_reqs + 1,)
        # The split covers the verifier's rows in order, nothing more.
        assert torch.equal(split_cu[0::2], cu)
        assert bool((split_cu[1:] >= split_cu[:-1]).all())
        prefix = (split_cu[1::2] - split_cu[:-1:2]).long()
        assert torch.equal(prefix[: lens.numel()], accepted.long())
        assert (prefix[lens.numel() :] == 0).all()
        # The tails never start: their whole table row is padding.
        assert (split_table[1::2] == PAD).all()
        want = _kernel_writes(table, cu, num_columns)
        assert _kernel_writes(split_table, split_cu, num_columns) == want
        # Every stored column lies in the accepted prefix.
        for r in range(lens.numel()):
            cols = [c for c in range(num_columns - 1) if table[r, c] >= 0]
            assert cols and max(cols) == int(accepted[r]) - 1
            seen["boundary_in_prefix"] += len(cols) > 1
        seen["truncated"] += int((accepted < lens).sum())
        seen["accepted_all"] += int((accepted == lens).sum())
        seen["accepted_one"] += int((accepted == 1).sum())
        assert _tokens_processed(split_table, split_cu, num_columns) == int(
            accepted.sum()
        )
    assert seen["truncated"] and seen["accepted_all"] and seen["accepted_one"]
    if block_size <= 4:
        assert seen["boundary_in_prefix"]


def test_split_launch_clamps_counts_to_the_row():
    # num_sampled 0 (no sample) commits position 0 like the table does; a count
    # past the row cannot extend the sequence into the next row.
    table = torch.full((2, 9), PAD, dtype=torch.int32)
    table[:, -1] = torch.tensor([3, 4])
    table[0, 0] = 3
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    _, split_cu = split_replay_launch(table, cu, torch.tensor([0, 6]))
    assert split_cu.tolist() == [0, 1, 4, 8, 8]


def _step(accepted_prefix):
    layer = SimpleNamespace(_replay_slots=8)
    base = torch.zeros(4, 9, dtype=torch.int32)[:, 0]
    return GDNSpecReplayStep(
        num_reqs=4,
        num_tokens=32,
        spec_rows=torch.arange(3),
        layers=[GDNSpecReplayLayer(layer, 0, base, torch.zeros(5, dtype=torch.int32))],
        block_tables=(torch.zeros(4, 8, dtype=torch.int32),),
        full_graph=True,
        accepted_prefix=accepted_prefix,
    )


def test_accepted_prefix_is_the_default_and_part_of_the_graph_key():
    step = GDNSpecReplayStep(
        num_reqs=1, num_tokens=8, spec_rows=torch.arange(1), layers=[], block_tables=()
    )
    assert step.accepted_prefix
    computed = torch.zeros(8, dtype=torch.int32)
    on = _step(True)
    off = _step(False)
    off.layers = on.layers
    off.block_tables = on.block_tables
    assert replay_graph_key(on, computed, 16) != replay_graph_key(off, computed, 16)
    # Per-step inputs (accepted counts, spec rows) are not part of the key.
    again = _step(True)
    again.layers, again.block_tables = on.layers, on.block_tables
    again.spec_rows = torch.arange(2)
    assert replay_graph_key(again, computed, 16) == replay_graph_key(on, computed, 16)


def test_layer_selectors_cover_the_split_launch(monkeypatch):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(gdn.current_platform, "device_type", "cpu")
    layer = SimpleNamespace(
        num_v_heads=48, tp_size=4, key_dim=2048, value_dim=6144, head_v_dim=128
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_state_tokens=lambda: 7),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    gdn.QwenGatedDeltaNetAttention._init_spec_state_replay(layer, config)
    assert layer._replay_slots == 8
    # The verifier reads the first max_num_seqs; the replay up to twice that.
    assert layer._replay_selectors.shape == (16,)
    assert (layer._replay_selectors == 9).all()
    assert layer._replay_table.shape == (8, 9)
    assert layer._replay_qkv.shape == (64, 2560)


_PREFIX_TABLE_CODE = r"""
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.gdn_spec_replay import (
    build_replay_prefix_launch,
    build_replay_state_indices,
    build_replay_state_table,
    fill_replay_graph_inputs,
    split_replay_launch,
)

gen = torch.Generator().manual_seed(2)
cases = 0
for block_size in (3, 4, 16, 1664):
    for num_columns in (5, 9):
        for _ in range(30):
            slots = num_columns - 1
            batch = int(torch.randint(1, 9, (1,), generator=gen))
            num_reqs = batch + int(torch.randint(0, 3, (1,), generator=gen))
            spec_rows = torch.nonzero(torch.rand(batch, generator=gen) < 0.8)
            spec_rows = spec_rows.reshape(-1)
            num_spec = spec_rows.numel()
            num_states = 16
            idx_mapping = torch.randperm(num_states, generator=gen)[:batch]
            idx_mapping = idx_mapping.to(torch.int32)
            lengths = torch.randint(0, slots + 1, (num_reqs,), generator=gen)
            lengths[:num_spec] = lengths[:num_spec].clamp(min=1)
            cu = torch.zeros(num_reqs + 1, dtype=torch.int32)
            cu[1:] = torch.cumsum(lengths, 0)
            # Counts from 0 (no sample) past the row (clamped).
            num_sampled = torch.randint(0, slots + 2, (batch,), generator=gen)
            num_sampled = num_sampled.to(torch.int32)
            computed = torch.randint(1, 6 * block_size, (num_states,), generator=gen)
            computed = computed.to(torch.int32)
            width = 8
            block_table = torch.randint(0, 10_000, (batch, width), generator=gen)
            block_table = block_table.to(torch.int32)
            state_table = torch.randint(0, 10_000, (num_reqs, 3), generator=gen)
            state_table = state_table.to(torch.int32)
            state_table[num_spec:, 0] = PAD_SLOT_ID
            base = state_table[:, 0]
            req = idx_mapping[spec_rows].long()
            kw = dict(
                base_state_ids=base, num_reqs=num_reqs, num_columns=num_columns,
                spec_rows=spec_rows, num_computed_tokens=computed[req],
                block_table=block_table, block_size=block_size)
            want_table = build_replay_state_indices(
                num_accepted=num_sampled[spec_rows], **kw)
            want = split_replay_launch(want_table, cu, num_sampled[spec_rows])
            kw.update(idx_mapping=idx_mapping, num_sampled=num_sampled,
                      num_computed_tokens=computed)
            got = build_replay_prefix_launch(cu_seqlens=cu, **kw)
            assert got[1].dtype == cu.dtype
            assert torch.equal(got[0], want[0]), (got[0], want[0])
            assert torch.equal(got[1], want[1]), (got[1], want[1], cu, num_sampled)
            # The full-replay table is unchanged by the SPLIT variant.
            assert torch.equal(build_replay_state_table(**kw), want_table)
            # A FULL graph's fixed-size buffers give the same launch.
            if num_spec == batch:
                rows_buf = torch.full((num_reqs,), 5, dtype=torch.int64)
                idx_buf = torch.zeros(num_reqs, dtype=torch.int32)
                sampled_buf = torch.full((num_reqs,), 3, dtype=torch.int32)
                fill_replay_graph_inputs(
                    rows_buf, idx_buf, sampled_buf, spec_rows=spec_rows,
                    idx_mapping=idx_mapping, num_sampled=num_sampled)
                kw.update(spec_rows=rows_buf, idx_mapping=idx_buf,
                          num_sampled=sampled_buf)
                graph = build_replay_prefix_launch(cu_seqlens=cu, **kw)
                assert torch.equal(graph[0], want[0]) and torch.equal(graph[1], want[1])
            cases += 1
print("PREFIX_TABLE_EQUAL", cases)
"""


def _interpret(code, cwd):
    # An empty working directory: `python -c` puts it first on sys.path, so the
    # child imports the same vllm as this process, not a checkout in the cwd.
    env = dict(os.environ, TRITON_INTERPRET="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-3000:], result.stderr[-4000:])
    return result


def test_fused_prefix_launch_equals_the_torch_reference(tmp_path):
    result = _interpret(_PREFIX_TABLE_CODE, tmp_path)
    assert result.returncode == 0 and "PREFIX_TABLE_EQUAL 240" in result.stdout


_E2E_CODE = r"""
import types
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.gdn_spec_replay import (
    GDNSpecReplayLayer,
    GDNSpecReplayStep,
    _replay_launches,
    fill_replay_graph_inputs,
    replay_gdn_spec_states,
)

H, HV, K, V = 1, 2, 16, 16
QKV = 2 * H * K + HV * V
SLOTS = 8
MAX_REQS = 6
MAX_TOKENS = MAX_REQS * SLOTS
NUM_BLOCKS = 120
gen = torch.Generator().manual_seed(11)


def make_layer():
    pool = torch.randn(NUM_BLOCKS, HV, V, K, generator=gen) * 0.3
    layer = SimpleNamespace(
        A_log=torch.rand(HV, generator=gen) * 2 - 1,
        dt_bias=(torch.rand(HV, generator=gen) - 0.5).half(),
        _replay_qkv=torch.randn(MAX_TOKENS, QKV, generator=gen).half(),
        _replay_a=(torch.randn(MAX_TOKENS, HV, generator=gen) * 0.5).half(),
        _replay_b=torch.randn(MAX_TOKENS, HV, generator=gen).half(),
        _replay_out=torch.empty(1, MAX_TOKENS, HV, V).half(),
        _replay_selectors=torch.full((2 * MAX_REQS,), SLOTS + 1, dtype=torch.int32),
        _replay_slots=SLOTS,
        num_k_heads=H,
        num_v_heads=HV,
        tp_size=1,
        head_k_dim=K,
        head_v_dim=V,
        kv_cache=(None, pool),
    )
    layer.replay_spec_states = types.MethodType(
        QwenGatedDeltaNetAttention.replay_spec_states, layer
    )
    return layer


layers = [make_layer() for _ in range(3)]
groups = [0, 1, 0]  # layers 0 and 2 share a cache group (one launch table)
originals = [l.kv_cache[1].clone() for l in layers]
stats = dict(cases=0, changed=0, truncated=0, saved_tokens=0, tokens=0)
for case in range(36):
    full = case % 2 == 1
    block_size = (2, 3, 4, 16)[case % 4]
    num_reqs = int(torch.randint(1, MAX_REQS + 1, (1,), generator=gen))
    if full:
        num_spec = int(torch.randint(1, num_reqs + 1, (1,), generator=gen))
        batch = num_spec
        spec_rows = torch.arange(num_spec)
        width = int(torch.randint(2, SLOTS + 1, (1,), generator=gen))
        lengths = torch.full((num_reqs,), width)
    else:
        num_spec = num_reqs
        batch = num_reqs + int(torch.randint(0, 3, (1,), generator=gen))
        spec_rows = torch.randperm(batch, generator=gen)[:num_spec].sort().values
        lengths = torch.randint(1, SLOTS + 1, (num_reqs,), generator=gen)
    num_tokens = int(lengths.sum())
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(lengths, 0)
    lens = lengths[:num_spec]
    accepted = (torch.rand(num_spec, generator=gen) * lens).long() + 1
    pick = torch.rand(num_spec, generator=gen)
    accepted = torch.where(pick < 0.2, torch.ones_like(accepted), accepted)
    accepted = torch.where(pick > 0.8, lens, accepted)
    num_sampled = torch.ones(batch, dtype=torch.int32)
    num_sampled[spec_rows] = accepted.to(torch.int32)
    idx_mapping = torch.randperm(16, generator=gen)[:batch].to(torch.int32)
    computed = torch.randint(1, 5 * block_size, (16,), generator=gen).to(torch.int32)
    computed[idx_mapping[spec_rows].long()] += accepted.to(torch.int32)
    entries, block_tables = [], []
    for g in range(2):
        perm = torch.randperm(NUM_BLOCKS, generator=gen).to(torch.int32)
        base = torch.full((num_reqs, 3), PAD_SLOT_ID, dtype=torch.int32)
        base[:num_spec, 0] = perm[:num_spec]
        width_bt = 10  # boundary columns up to (5 * bs + 8) // bs - 1
        bt = perm[num_spec : num_spec + batch * width_bt].view(batch, width_bt)
        block_tables.append(bt.contiguous())
        entries.append(base[:, 0])
    replay_layers = [
        GDNSpecReplayLayer(layer, g, entries[g], cu) for layer, g in zip(layers, groups)
    ]
    pools = {}
    for prefix in (False, True):
        for layer, original in zip(layers, originals):
            layer.kv_cache[1].copy_(original)
        step = GDNSpecReplayStep(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            spec_rows=spec_rows,
            layers=replay_layers,
            block_tables=tuple(block_tables),
            full_graph=full,
            accepted_prefix=prefix,
        )
        kw = dict(num_computed_tokens=computed, block_size=block_size)
        if full:
            # The captured graph's fixed-size inputs.
            rows_buf = torch.full((num_reqs,), -1, dtype=torch.int64)
            idx_buf = torch.zeros(num_reqs, dtype=torch.int32)
            sampled_buf = torch.ones(num_reqs, dtype=torch.int32)
            fill_replay_graph_inputs(
                rows_buf, idx_buf, sampled_buf, spec_rows=spec_rows,
                idx_mapping=idx_mapping, num_sampled=num_sampled)
            _replay_launches(step, spec_rows=rows_buf, idx_mapping=idx_buf,
                             num_sampled=sampled_buf, **kw)
        else:
            replay_gdn_spec_states(step, idx_mapping=idx_mapping,
                                   num_sampled=num_sampled, **kw)
        pools[prefix] = [layer.kv_cache[1].clone() for layer in layers]
    for want, got, original in zip(pools[False], pools[True], originals):
        assert torch.equal(want, got), (case, (want - got).abs().max())
        changed = (want != original).flatten(1).any(1)
        stats["changed"] += int(changed.sum())
        # Only base and boundary blocks change: at most two per request.
        assert int(changed.sum()) <= 2 * num_spec
    stats["cases"] += 1
    stats["truncated"] += int((accepted < lens).sum())
    stats["saved_tokens"] += int((lens - accepted).sum())
    stats["tokens"] += int(lens.sum())
assert stats["changed"] > 0 and stats["truncated"] > 0, stats
print("E2E_PREFIX_BITWISE", stats["cases"], stats, _replay_launches.__module__,
      __import__("vllm").__file__)
"""


def test_prefix_replay_writes_the_full_replay_states_bitwise(tmp_path):
    result = _interpret(_E2E_CODE, tmp_path)
    assert result.returncode == 0 and "E2E_PREFIX_BITWISE 36" in result.stdout
