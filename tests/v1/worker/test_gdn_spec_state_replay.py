# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN speculative state replay: allocation, state tables and replay indices (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.attention.backends.gdn_attn import select_gdn_state_block_ids
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.gdn_spec_replay import build_replay_state_indices

pytestmark = pytest.mark.cpu_test

SLOTS = 8  # 1 + 7 speculative tokens: the verifier table has SLOTS + 1 columns
BS = 1664


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
    from vllm import envs

    # Exercise the accepted-column selection, not the legacy slot-zero override.
    monkeypatch.setattr(
        envs, "VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0", False, raising=False
    )
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
    from vllm import envs
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        config_module, "_any_participating_device_is_capability", lambda *a: True
    )
    monkeypatch.setattr(envs, "VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", False, raising=False)
    monkeypatch.setattr(envs, "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV", True, raising=False)
    return config_module._resolve_gdn_spec_state_replay


def test_replay_is_on_by_default_where_supported(sm70):
    assert sm70(_replay_config()) is True


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
