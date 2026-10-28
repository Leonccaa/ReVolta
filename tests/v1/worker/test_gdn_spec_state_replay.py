# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN speculative state replay: allocation, state tables and replay indices (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backends.gdn_attn import select_gdn_state_block_ids
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.gdn_spec_replay import build_replay_state_indices

pytestmark = pytest.mark.cpu_test

SLOTS = 8  # 1 + 7 speculative tokens: the verifier table has SLOTS + 1 columns
BS = 1664


def _patch_env_flag(monkeypatch, name, value):
    """Override one vllm.envs boolean switch for this test only.

    Every declared vllm.envs name resolves lazily through the module
    ``__getattr__`` (vllm/envs.py:19565), so ``monkeypatch.setattr(envs, name,
    value)`` records the *resolved* value as the old value and puts it back as a
    concrete module attribute on undo. That shadow then hides the getter from
    every later test in the same pytest process, which makes unrelated tests that
    only change the environment order-dependent. Deleting any pre-existing
    shadow and writing the flag into ``os.environ`` (what the getter reads) is
    restored exactly by monkeypatch, so nothing leaks.
    """

    monkeypatch.delitem(vars(envs), name, raising=False)
    monkeypatch.setenv(name, "1" if value else "0")


def _table(**kwargs):
    defaults = dict(num_columns=SLOTS + 1, block_size=BS)
    defaults.update(kwargs)
    return build_replay_state_indices(**defaults).tolist()


def _block_table(rows):
    # Distinct block ids: row r, column c -> 100 * r + c.
    return torch.tensor([[100 * r + c for c in range(8)] for r in range(rows)])


def test_last_accepted_position_maps_to_the_running_block():
    # Two requests in one block, 3 and 8 tokens accepted, no boundary crossed.
    table = _table(
        base_state_ids=torch.tensor([7, 9], dtype=torch.int32),
        num_reqs=2,
        spec_rows=torch.tensor([0, 1]),
        num_accepted=torch.tensor([3, 8]),
        num_computed_tokens=torch.tensor([10, 20]),
        block_table=_block_table(2),
    )
    pad = PAD_SLOT_ID
    assert table[0] == [pad, pad, 7, pad, pad, pad, pad, pad, 7]
    assert table[1] == [pad, pad, pad, pad, pad, pad, pad, 9, 9]


def test_boundary_before_the_last_position_gets_its_block():
    # 5 accepted tokens; the state after 2 of them reaches 2 * BS tokens.
    computed = 2 * BS + 3
    table = _table(
        base_state_ids=torch.tensor([42], dtype=torch.int32),
        num_reqs=1,
        spec_rows=torch.tensor([0]),
        num_accepted=torch.tensor([5]),
        num_computed_tokens=torch.tensor([computed]),
        block_table=_block_table(1),
    )
    # running = computed - 5 + 1 = 2 * BS - 1; boundary at position 1; block col 1.
    pad = PAD_SLOT_ID
    assert table[0] == [pad, 1, pad, pad, 42, pad, pad, pad, 42]


def test_boundary_at_the_last_position_is_left_to_the_align_copy():
    table = _table(
        base_state_ids=torch.tensor([42], dtype=torch.int32),
        num_reqs=1,
        spec_rows=torch.tensor([0]),
        num_accepted=torch.tensor([4]),
        num_computed_tokens=torch.tensor([3 * BS]),
        block_table=_block_table(1),
    )
    pad = PAD_SLOT_ID
    assert table[0] == [pad, pad, pad, 42, pad, pad, pad, pad, 42]


def test_graph_padding_rows_are_skipped():
    # A FULL graph captured for four requests replays two real ones.
    table = _table(
        base_state_ids=torch.tensor([5, 6, PAD_SLOT_ID, PAD_SLOT_ID]),
        num_reqs=4,
        spec_rows=torch.tensor([0, 1]),
        num_accepted=torch.tensor([1, 2]),
        num_computed_tokens=torch.tensor([30, 31]),
        block_table=_block_table(2),
    )
    assert table[2] == [PAD_SLOT_ID] * (SLOTS + 1)
    assert table[3] == [PAD_SLOT_ID] * (SLOTS + 1)
    assert table[0][0] == 5 and table[1][1] == 6


def test_rows_follow_the_speculative_requests_not_the_batch():
    # Batch rows 1 and 3 are speculative (a prefill sits between them).
    table = _table(
        base_state_ids=torch.tensor([11, 33], dtype=torch.int32),
        num_reqs=2,
        spec_rows=torch.tensor([1, 3]),
        num_accepted=torch.tensor([2, 6]),
        num_computed_tokens=torch.tensor([BS + 1, 2 * BS + 2]),
        block_table=_block_table(4),
    )
    pad = PAD_SLOT_ID
    # Row 0: running = BS, boundary at position 0 -> batch row 1, column 0.
    assert table[0] == [100, 11, pad, pad, pad, pad, pad, pad, 11]
    # Row 1: running = 2 * BS - 3, boundary at position 3 -> batch row 3, col 1.
    assert table[1] == [pad, pad, pad, 301, pad, 33, pad, pad, 33]


def test_state_table_pads_replayed_columns():
    spec = MambaSpec(
        block_size=4,
        shapes=((2,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=0,
        num_replayed_speculative_blocks=7,
    )
    block_table = torch.arange(24, dtype=torch.int32).view(2, 12)
    seq_lens = torch.tensor([5, 9])
    out = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")
    assert out.shape == (2, 8)
    assert out[:, 0].tolist() == [1, 14]
    assert (out[:, 1:] == PAD_SLOT_ID).all()


def test_state_table_without_replay_is_unchanged():
    spec = MambaSpec(
        block_size=4,
        shapes=((2,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=7,
    )
    block_table = torch.arange(24, dtype=torch.int32).view(2, 12)
    out = mamba_get_block_table_tensor(block_table, torch.tensor([5, 9]), spec, "align")
    assert out.tolist() == [list(range(1, 9)), list(range(14, 22))]


def test_replay_halves_the_blocks_a_request_holds():
    config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    common = dict(block_size=4, shapes=((256,),), dtypes=(torch.float32,))
    per_draft = MambaSpec(**common, num_speculative_blocks=7)
    replayed = MambaSpec(
        **common, num_speculative_blocks=0, num_replayed_speculative_blocks=7
    )
    page = per_draft.page_size_bytes
    assert per_draft.max_memory_usage_bytes(config) == 9 * page
    assert replayed.max_memory_usage_bytes(config) == 2 * page


def test_non_speculative_rows_read_the_running_block_with_replay(monkeypatch):
    # Exercise the accepted-column selection, not the legacy slot-zero override.
    _patch_env_flag(monkeypatch, "VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0", False)
    table = torch.tensor([[3, PAD_SLOT_ID, PAD_SLOT_ID], [4, PAD_SLOT_ID, PAD_SLOT_ID]])
    accepted = torch.tensor([1, 3])
    assert select_gdn_state_block_ids(table, accepted, 2, state_columns=1).tolist() == [
        3,
        4,
    ]
    per_draft = torch.tensor([[3, 5, 6], [4, 7, 8]])
    assert select_gdn_state_block_ids(per_draft, accepted, 2).tolist() == [3, 8]


@pytest.mark.parametrize("replay", [False, True])
def test_state_blocks_follow_the_resolved_flag(replay):
    spec = SimpleNamespace(
        gdn_spec_state_replay=replay, num_speculative_state_tokens=lambda: 7
    )
    blocks = SpeculativeConfig.num_speculative_state_blocks(spec)  # type: ignore[arg-type]
    assert blocks == (0 if replay else 7)


def _replay_config(**overrides):
    policy = SimpleNamespace(spec_state_replay=None, fused_gdn_verify=False)
    spec = SimpleNamespace(
        use_dflash=lambda: True, use_dflash_ddtree=lambda: False, sm70_dflash2=policy
    )
    config = SimpleNamespace(
        speculative_config=spec,
        model_config=SimpleNamespace(is_hybrid=True),
        use_v2_model_runner=True,
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    for key, value in overrides.items():
        target, _, attr = key.partition(".")
        setattr(getattr(config, target) if attr else config, attr or target, value)
    return config


@pytest.fixture
def sm70(monkeypatch):
    import vllm.config.vllm as config_module
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        config_module, "_any_participating_device_is_capability", lambda *a: True
    )
    _patch_env_flag(monkeypatch, "VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", False)
    _patch_env_flag(monkeypatch, "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV", True)
    return config_module._resolve_gdn_spec_state_replay


def test_replay_is_off_by_default_and_on_when_requested(sm70):
    assert sm70(_replay_config()) is False
    requested = SimpleNamespace(spec_state_replay=True, fused_gdn_verify=False)
    assert sm70(_replay_config(**{"speculative_config.sm70_dflash2": requested}))


@pytest.mark.parametrize(
    "override",
    [
        {
            "speculative_config.sm70_dflash2": SimpleNamespace(
                spec_state_replay=False, fused_gdn_verify=False
            )
        },
        {"cache_config.mamba_cache_mode": "none"},
        {"scheduler_config.async_scheduling": True},
        {"use_v2_model_runner": False},
        {"model_config.is_hybrid": False},
        {"parallel_config.pipeline_parallel_size": 2},
        {
            "speculative_config.sm70_dflash2": SimpleNamespace(
                spec_state_replay=None, fused_gdn_verify=True
            )
        },
    ],
)
def test_replay_stays_off_where_unsupported(sm70, override):
    assert sm70(_replay_config(**override)) is False


def test_replay_never_applies_to_other_methods(sm70):
    config = _replay_config()
    config.speculative_config.use_dflash = lambda: False
    assert sm70(config) is False
    config.speculative_config = None
    assert sm70(config) is False


class _GDNLayer:
    # The MambaBase attributes get_kv_cache_spec reads, with the TP4 GDN shapes.
    mamba_type = MambaAttentionBackendEnum.GDN_ATTN
    is_kv_cache_tp_replicated = False

    def get_state_shape(self):
        return ((3, 2560), (12, 128, 128))

    def get_state_dtype(self):
        return (torch.float16, torch.float32)


def _gdn_spec(sm70, method, num_tokens):
    """The GDN layer's KV cache spec after VllmConfig resolution, for one method."""
    config = _replay_config()
    if method is None:
        config.speculative_config = None
    else:
        spec = config.speculative_config
        # Replay requested (off by default): only the method decides.
        spec.sm70_dflash2.spec_state_replay = True
        spec.method = method
        spec.num_speculative_tokens = num_tokens
        spec.use_dflash = lambda: method == "dflash"
        spec.num_speculative_state_tokens = (
            lambda: SpeculativeConfig.num_speculative_state_tokens(spec)  # type: ignore[arg-type]
        )
        spec.num_speculative_state_blocks = (
            lambda: SpeculativeConfig.num_speculative_state_blocks(spec)  # type: ignore[arg-type]
        )
        spec.gdn_spec_state_replay = sm70(config)
    config.cache_config.mamba_block_size = BS
    config.cache_config.mamba_page_size_padded = None
    return MambaBase.get_kv_cache_spec(_GDNLayer(), config)  # type: ignore[arg-type]


def _legacy_gdn_spec(num_speculative_blocks):
    # What get_kv_cache_spec built before replay existed.
    layer = _GDNLayer()
    return MambaSpec(
        shapes=layer.get_state_shape(),
        dtypes=layer.get_state_dtype(),
        block_size=BS,
        page_size_padded=None,
        mamba_type=layer.mamba_type,
        tp_replicated=False,
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
    )


@pytest.mark.parametrize(("method", "num_tokens"), [("mtp", 4), ("mtp", 1), (None, 0)])
def test_mtp_and_no_speculation_keep_their_gdn_allocation(sm70, method, num_tokens):
    # Every other condition for replay holds; only the method differs from DFlash.
    spec = _gdn_spec(sm70, method, num_tokens)
    assert spec == _legacy_gdn_spec(num_tokens)
    assert spec.num_replayed_speculative_blocks == 0
    config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    page = spec.page_size_bytes
    assert spec.max_memory_usage_bytes(config) == (2 + num_tokens) * page
    # The state table keeps one real block per speculative position.
    block_table = torch.arange(2 * 12, dtype=torch.int32).view(2, 12) + 100
    seq_lens = torch.tensor([BS + 1, 3 * BS])
    out = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")
    legacy = mamba_get_block_table_tensor(
        block_table, seq_lens, _legacy_gdn_spec(num_tokens), "align"
    )
    assert torch.equal(out, legacy)
    assert (out != PAD_SLOT_ID).all()


def test_dflash_replay_is_the_only_change_to_gdn_allocation(sm70):
    spec = _gdn_spec(sm70, "dflash", 7)
    assert spec.num_speculative_blocks == 0
    assert spec.num_replayed_speculative_blocks == 7
    assert spec.page_size_bytes == _legacy_gdn_spec(7).page_size_bytes


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 20, 24, 40, 48, 64])
def test_replay_alone_widens_the_mixed_qkv_verifier_admission(
    monkeypatch, wide, num_tokens
):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    _patch_env_flag(monkeypatch, "VLLM_SM70_QWEN38_WIDE_M_GDN", wide)
    _patch_env_flag(monkeypatch, "VLLM_SM70_GDN_TAIL_COPY", False)
    admits = gdn.QwenGatedDeltaNetAttention._mixed_qkv_verify_admits
    # Replay off (MTP, no speculation, DFlash with replay off): the band as before.
    off = SimpleNamespace(spec_state_replay=False)
    assert admits(off, num_tokens) == gdn._sm70_mixed_qkv_verify_tokens_ok(num_tokens)  # type: ignore[arg-type]
    expected_band = 1 < num_tokens <= 16 or (wide and 2 <= num_tokens <= 48)
    assert gdn._sm70_mixed_qkv_verify_tokens_ok(num_tokens) == expected_band
    # Replay on: every verify batch, up to 8 requests x 8 tokens.
    assert admits(SimpleNamespace(spec_state_replay=True), num_tokens)  # type: ignore[arg-type]


def _qkvz_rows(num_tokens):
    # The q/k/v/z projection output; q/k/v is its row-strided leading slice.
    qkvz = torch.zeros(num_tokens, 2560 + 1536, dtype=torch.float16)
    return qkvz[:, :2560]


def _strided_layer(replay):
    return SimpleNamespace(
        spec_state_replay=replay,
        enable_sm70_fused_sigmoid_mixed_qkv=True,
        num_k_heads=16,
        num_v_heads=48,
        tp_size=4,
    )


@pytest.mark.parametrize("num_tokens", [8, 24, 40, 64])
def test_replay_verifier_reads_row_strided_qkv(num_tokens):
    # Batches outside the fused input projection hand the verifier a strided
    # view; without this the replay layer reached the split verifier and raised.
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    reads = gdn.QwenGatedDeltaNetAttention._replay_reads_strided_qkv
    qkv = _qkvz_rows(num_tokens)
    assert not qkv.is_contiguous()
    assert reads(_strided_layer(True), qkv)  # type: ignore[arg-type]
    # Replay off: no input is admitted by this path, so the original
    # contiguity and tail-copy conditions alone decide, as before.
    assert not reads(_strided_layer(False), qkv)  # type: ignore[arg-type]
    assert not reads(_strided_layer(False), qkv.contiguous())  # type: ignore[arg-type]


def test_replay_verifier_rejects_layouts_the_kernel_cannot_read():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    reads = gdn.QwenGatedDeltaNetAttention._replay_reads_strided_qkv
    layer = _strided_layer(True)
    qkv = _qkvz_rows(8)
    assert not reads(layer, qkv.float())  # type: ignore[arg-type]
    assert not reads(layer, qkv.t())  # type: ignore[arg-type]
    assert not reads(layer, None)  # type: ignore[arg-type]


def test_release_default_is_one_switch(sm70, monkeypatch):
    import vllm.config.vllm as config_module
    from vllm.config import sm70_dflash2

    # The release default lives in one constant, read through vllm.config.vllm.
    assert sm70_dflash2.SM70_DFLASH2_SPEC_STATE_REPLAY_DEFAULT is False
    assert config_module.SM70_DFLASH2_SPEC_STATE_REPLAY_DEFAULT is False
    assert sm70(_replay_config()) is False
    monkeypatch.setattr(config_module, "SM70_DFLASH2_SPEC_STATE_REPLAY_DEFAULT", True)
    # Default on: None resolves on, explicit False still turns replay off.
    assert sm70(_replay_config()) is True
    explicit = SimpleNamespace(spec_state_replay=False, fused_gdn_verify=False)
    assert (
        sm70(_replay_config(**{"speculative_config.sm70_dflash2": explicit})) is False
    )


_FUSED_TABLE_CODE = r"""
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.gdn_spec_replay import (
    build_replay_state_indices,
    build_replay_state_table,
)

gen = torch.Generator().manual_seed(0)
cases = 0
for block_size in (4, 7, 16, 1664):
    for num_columns in (5, 9):
        for _ in range(60):
            batch = int(torch.randint(1, 9, (1,), generator=gen))
            # Graph padding rows past the batch.
            num_reqs = batch + int(torch.randint(0, 3, (1,), generator=gen))
            spec_mask = torch.rand(batch, generator=gen) < 0.8
            spec_rows = torch.nonzero(spec_mask).reshape(-1)
            num_spec = spec_rows.numel()
            num_states = 16
            idx_mapping = torch.randperm(num_states, generator=gen)[:batch]
            idx_mapping = idx_mapping.to(torch.int32)
            num_sampled = torch.randint(0, num_columns, (batch,), generator=gen)
            num_sampled = num_sampled.to(torch.int32)
            computed = torch.randint(1, 6 * block_size, (num_states,), generator=gen)
            computed = computed.to(torch.int32)
            width = 6 * block_size // block_size + 2
            block_table = torch.randint(0, 10_000, (batch, width), generator=gen)
            block_table = block_table.to(torch.int32)
            # Strided base ids (the verifier table's column 0); padding is PAD.
            state_table = torch.randint(0, 10_000, (num_reqs, 3), generator=gen)
            state_table = state_table.to(torch.int32)
            state_table[num_spec:, 0] = PAD_SLOT_ID
            base = state_table[:, 0]
            req = idx_mapping[spec_rows].long()
            want = build_replay_state_indices(
                base_state_ids=base, num_reqs=num_reqs, num_columns=num_columns,
                spec_rows=spec_rows, num_accepted=num_sampled[spec_rows],
                num_computed_tokens=computed[req], block_table=block_table,
                block_size=block_size)
            got = build_replay_state_table(
                base_state_ids=base, num_reqs=num_reqs, num_columns=num_columns,
                spec_rows=spec_rows, idx_mapping=idx_mapping, num_sampled=num_sampled,
                num_computed_tokens=computed, block_table=block_table,
                block_size=block_size)
            assert torch.equal(got, want), (got, want, block_size, num_columns)
            cases += 1
print("FUSED_TABLE_EQUAL", cases)
"""


def test_fused_replay_table_equals_the_torch_reference():
    import os
    import subprocess
    import sys

    env = dict(os.environ, TRITON_INTERPRET="1")
    result = subprocess.run(
        [sys.executable, "-c", _FUSED_TABLE_CODE],
        env=env,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-2000:], result.stderr[-4000:])
    assert result.returncode == 0 and "FUSED_TABLE_EQUAL 480" in result.stdout


_PADDED_TABLE_CODE = r"""
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.gdn_spec_replay import (
    build_replay_state_indices,
    build_replay_state_table,
    fill_replay_graph_inputs,
)

gen = torch.Generator().manual_seed(1)
cases = 0
block_size = 16
for _ in range(200):
    num_reqs = int(torch.randint(1, 9, (1,), generator=gen))
    batch = int(torch.randint(1, num_reqs + 1, (1,), generator=gen))
    spec_rows = torch.arange(batch)  # a FULL step: every real row speculates
    idx_mapping = torch.randperm(16, generator=gen)[:batch].to(torch.int32)
    num_sampled = torch.randint(1, 9, (batch,), generator=gen).to(torch.int32)
    computed = torch.randint(1, 6 * block_size, (16,), generator=gen)
    computed = computed.to(torch.int32)
    block_table = torch.randint(0, 10_000, (num_reqs, 8), generator=gen)
    block_table = block_table.to(torch.int32)
    base = torch.randint(0, 10_000, (num_reqs,), generator=gen).to(torch.int32)
    base[batch:] = PAD_SLOT_ID
    want = build_replay_state_indices(
        base_state_ids=base, num_reqs=num_reqs, num_columns=9,
        spec_rows=spec_rows, num_accepted=num_sampled,
        num_computed_tokens=computed[idx_mapping.long()],
        block_table=block_table, block_size=block_size)
    # The graph's fixed-size buffers: spec rows padded with -1.
    rows_buf = torch.full((num_reqs,), 7, dtype=torch.int64)
    idx_buf = torch.zeros(num_reqs, dtype=torch.int32)
    sampled_buf = torch.ones(num_reqs, dtype=torch.int32)
    fill_replay_graph_inputs(rows_buf, idx_buf, sampled_buf, spec_rows=spec_rows,
                             idx_mapping=idx_mapping, num_sampled=num_sampled)
    assert rows_buf[batch:].eq(-1).all() and torch.equal(rows_buf[:batch], spec_rows)
    got = build_replay_state_table(
        base_state_ids=base, num_reqs=num_reqs, num_columns=9,
        spec_rows=rows_buf, idx_mapping=idx_buf, num_sampled=sampled_buf,
        num_computed_tokens=computed, block_table=block_table,
        block_size=block_size)
    assert torch.equal(got, want), (got, want)
    cases += 1
print("PADDED_TABLE_EQUAL", cases)
"""


def test_graph_padded_spec_rows_build_the_same_table():
    import os
    import subprocess
    import sys

    env = dict(os.environ, TRITON_INTERPRET="1")
    result = subprocess.run(
        [sys.executable, "-c", _PADDED_TABLE_CODE],
        env=env,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-2000:], result.stderr[-4000:])
    assert result.returncode == 0 and "PADDED_TABLE_EQUAL 200" in result.stdout


def _graph_step(full_graph):
    from vllm.v1.worker.gpu.gdn_spec_replay import (
        GDNSpecReplayLayer,
        GDNSpecReplayStep,
    )

    layer = SimpleNamespace(_replay_slots=8)
    base = torch.zeros(4, 9, dtype=torch.int32)[:, 0]
    return GDNSpecReplayStep(
        num_reqs=4,
        num_tokens=32,
        spec_rows=torch.arange(3),
        layers=[GDNSpecReplayLayer(layer, 0, base, torch.zeros(5, dtype=torch.int32))],
        block_tables=(torch.zeros(4, 8, dtype=torch.int32),),
        full_graph=full_graph,
    )


def test_replay_graphs_use_eager_replay_for_mixed_steps(monkeypatch):
    from vllm.v1.worker.gpu import gdn_spec_replay as m

    calls = []
    monkeypatch.setattr(m, "replay_gdn_spec_states", lambda step, **kw: calls.append(1))
    monkeypatch.setattr(
        m.GDNSpecReplayGraphs, "_capture", lambda *a: pytest.fail("captured")
    )
    graphs = m.GDNSpecReplayGraphs(enabled=True)
    kw = dict(
        idx_mapping=torch.zeros(3, dtype=torch.int32),
        num_sampled=torch.ones(3, dtype=torch.int32),
        num_computed_tokens=torch.zeros(8, dtype=torch.int32),
        block_size=16,
    )
    for _ in range(3):
        graphs.run(_graph_step(full_graph=False), **kw)
    assert calls == [1, 1, 1] and not graphs.graphs
    # Disabled: FULL steps stay eager too.
    off = m.GDNSpecReplayGraphs(enabled=False)
    for _ in range(3):
        off.run(_graph_step(full_graph=True), **kw)
    assert calls == [1] * 6 and not off.graphs


def test_replay_graphs_capture_on_the_second_full_step(monkeypatch):
    from vllm.v1.worker.gpu import gdn_spec_replay as m

    eager, replays = [], []
    monkeypatch.setattr(m, "replay_gdn_spec_states", lambda step, **kw: eager.append(1))
    fake_graph = SimpleNamespace(replay=lambda: replays.append(1))

    def capture(self, step, num_computed_tokens, block_size):
        n = step.num_reqs
        return m._ReplayGraph(
            fake_graph,
            torch.full((n,), -1, dtype=torch.int64),
            torch.zeros(n, dtype=torch.int32),
            torch.ones(n, dtype=torch.int32),
        )

    monkeypatch.setattr(m.GDNSpecReplayGraphs, "_capture", capture)
    graphs = m.GDNSpecReplayGraphs(enabled=True)
    step = _graph_step(full_graph=True)
    computed = torch.zeros(8, dtype=torch.int32)
    kw = dict(num_computed_tokens=computed, block_size=16)
    graphs.run(
        step,
        idx_mapping=torch.tensor([5, 6, 7], dtype=torch.int32),
        num_sampled=torch.tensor([1, 2, 3], dtype=torch.int32),
        **kw,
    )
    assert eager == [1] and replays == [] and not graphs.graphs
    graphs.run(
        step,
        idx_mapping=torch.tensor([5, 6, 7], dtype=torch.int32),
        num_sampled=torch.tensor([4, 2, 1], dtype=torch.int32),
        **kw,
    )
    assert eager == [1] and replays == [1] and len(graphs.graphs) == 1
    entry = next(iter(graphs.graphs.values()))
    assert entry.spec_rows.tolist() == [0, 1, 2, -1]
    assert entry.num_sampled.tolist()[:3] == [4, 2, 1]
    assert entry.idx_mapping.tolist()[:3] == [5, 6, 7]
    # A different tensor behind the step (another shape) is a new key.
    other = _graph_step(full_graph=True)
    graphs.run(
        other,
        idx_mapping=torch.zeros(3, dtype=torch.int32),
        num_sampled=torch.ones(3, dtype=torch.int32),
        **kw,
    )
    assert eager == [1, 1] and replays == [1]
