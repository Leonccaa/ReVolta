# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 warmup must reserve the same speculative KV tail as the scheduler."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryShape
from vllm.models.qwen4_exp.nvidia.host_kv.planning import history_groups
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.warmup import (
    _host_kv_warmup_profiles,
    _kernel_prefill_warmup_token_counts,
    _reserved_block_count,
    warmup_kernels,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
NUM_SPEC_TOKENS = 7


def test_kernel_prefill_warmup_profiles_are_capability_advertised() -> None:
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        max_model_len=64,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "plain": object(),
                "qsa_0": SimpleNamespace(
                    kernel_warmup_prefill_token_counts=(33, 257, True, -1)
                ),
                "qsa_1": SimpleNamespace(kernel_warmup_prefill_token_counts=(33,)),
            }
        ),
    )

    assert _kernel_prefill_warmup_token_counts(runner, 6) == (6, 33)


def test_kernel_prefill_warmup_runs_default_batch_and_extra_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector_states: list[bool] = []
    attention_spec = _full_attention_spec()
    runner = SimpleNamespace(
        num_speculative_steps=4,
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=128,
        ),
        max_model_len=64,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "profile": SimpleNamespace(kernel_warmup_prefill_token_counts=(33,))
            }
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=attention_spec)],
            num_blocks=64,
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=4),
        model_state=SimpleNamespace(max_encoder_len=0),
        is_pooling_model=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        kv_connector=SimpleNamespace(
            set_disabled=lambda disabled: connector_states.append(disabled)
        ),
    )
    executions: list[Any] = []
    samples: list[Any] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    warmup_kernels(runner, executions.append, samples.append)

    assert [output.total_num_scheduled_tokens for output in executions] == [
        24,
        20,
        0,
        33,
        5,
        0,
    ]
    assert connector_states == [True, False]
    assert len(samples) == 4


def _speculative_config(method: str) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    object.__setattr__(config, "method", method)
    object.__setattr__(config, "num_speculative_tokens", NUM_SPEC_TOKENS)
    object.__setattr__(config, "ddtree_disable_tree_verify", False)
    object.__setattr__(config, "ddtree_budget", 24)
    return config


class _Config:
    speculative_config: SpeculativeConfig | None = None
    diffusion_config = None
    num_speculative_tokens = VllmConfig.num_speculative_tokens
    num_lookahead_tokens = VllmConfig.num_lookahead_tokens


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("dflash", NUM_SPEC_TOKENS + 1),
        ("dflash_ddtree", 25),
        ("eagle3", NUM_SPEC_TOKENS),
        ("mtp", NUM_SPEC_TOKENS),
        ("dspark", NUM_SPEC_TOKENS),
        ("draft_model", NUM_SPEC_TOKENS),
        ("ngram", 0),
    ],
)
def test_num_lookahead_tokens_per_method(method: str, expected: int) -> None:
    config = _Config()
    config.speculative_config = _speculative_config(method)

    assert config.num_lookahead_tokens == expected


def test_num_lookahead_tokens_without_speculation() -> None:
    assert _Config().num_lookahead_tokens == 0


def _full_attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
    )


def _mamba_spec(mode: str) -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mode,
        num_speculative_blocks=NUM_SPEC_TOKENS,
    )


def _circular_spec() -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (_full_attention_spec(), 2),
        (_circular_spec(), 1),
        (
            UniformTypeKVCacheSpecs(
                block_size=4, kv_cache_specs={"compressor_state": _circular_spec()}
            ),
            1,
        ),
        (_mamba_spec("align"), 8),
        (_mamba_spec("none"), 9),
        (_mamba_spec("all"), 9),
    ],
)
def test_reserved_block_count_matches_speculative_tail(spec, expected) -> None:
    # 14 scheduled tokens plus DFlash's eight lookahead positions crosses the
    # second attention block. Align-mode Mamba excludes lookahead from its
    # token range but always retains seven speculative state blocks.
    assert (
        _reserved_block_count(
            14,
            spec,
            num_lookahead_tokens=NUM_SPEC_TOKENS + 1,
            max_model_len=MAX_MODEL_LEN,
            max_encoder_len=0,
        )
        == expected
    )


def test_warmup_numbers_block_ids_per_block_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long-context mode: the state and history spaces number their blocks
    independently, so warmup must keep each group's IDs inside its space and
    size the batch by the tighter space."""
    from vllm.v1.kv_cache_interface import KVBlockSegments

    attention_spec = _full_attention_spec()
    mamba_spec = _mamba_spec("align")
    # Space 0 (state) holds 2 requests' worst-case states, space 1 is roomy.
    # A decode step of the default 6-token prompt with 4 speculative tokens.
    per_request_states = _reserved_block_count(
        6 + 1 + 4,
        mamba_spec,
        num_lookahead_tokens=4,
        max_model_len=64,
        max_encoder_len=0,
    )
    segments = KVBlockSegments(
        names=("state", "history"),
        num_blocks=(2 * per_request_states + 1, 64),
        group_segments=(1, 0),
        primary=1,
    )
    runner = SimpleNamespace(
        num_speculative_steps=4,
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=128),
        max_model_len=64,
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=attention_spec),
                SimpleNamespace(kv_cache_spec=mamba_spec),
            ],
            num_blocks=64,
            block_segments=segments,
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=4),
        model_state=SimpleNamespace(max_encoder_len=0),
        is_pooling_model=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )
    executions: list[Any] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    warmup_kernels(runner, executions.append, lambda _: None)

    prefill = executions[0]
    assert len(prefill.scheduled_new_reqs) == 2  # limited by the state space
    ids: list[list[int]] = [[], []]
    for req in prefill.scheduled_new_reqs:
        for group_id, block_ids in enumerate(req.block_ids):
            ids[group_id].extend(block_ids)
    decode = executions[1].scheduled_cached_reqs
    for new in decode.new_block_ids:
        for group_id, block_ids in enumerate(new or ()):
            ids[group_id].extend(block_ids)
    for group_id, space in enumerate(segments.group_segments):
        assert ids[group_id]
        assert len(set(ids[group_id])) == len(ids[group_id])
        assert (
            min(ids[group_id]) > 0 and max(ids[group_id]) < (segments.num_blocks[space])
        )
    # Both spaces start at block 1: the IDs overlap across groups.
    assert set(ids[0]) & set(ids[1])


def _host_kv_shape(slots_per_block: int) -> HostHistoryShape:
    return HostHistoryShape(
        num_blocks=64,
        kernel_blocks_per_block=1,
        kernel_block_tokens=8,
        num_layers=1,
        kv_heads=1,
        head_dim=8,
        dtype=torch.float16,
        hot_slots=256,
        max_requests=4,
        selection_width=8,
        max_write_rows=256,
        dcp_world_size=2,
        interleave=1,
        decode_rows=5,
        slots_per_block=slots_per_block,
        staged_writethrough=True,
        prefill_gather="union",
    )


def _host_kv_runner(
    slots_per_block: int = 8, max_captured_tokens: int = 64
) -> tuple[Any, list[Any]]:
    """Warmup runner fixture for long-context host KV mode."""
    attention_spec = _full_attention_spec()
    store = SimpleNamespace(
        shape=_host_kv_shape(slots_per_block),
        host=torch.zeros(64, 4),
        hot_metrics=None,
    )
    runner = SimpleNamespace(
        num_speculative_steps=4,
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=256),
        max_model_len=256,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "qsa_0": SimpleNamespace(kernel_warmup_prefill_token_counts=(33,))
            }
        ),
        cudagraph_manager=SimpleNamespace(max_captured_tokens=max_captured_tokens),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=attention_spec, layer_names=["qsa.0"])
            ],
            num_blocks=64,
            block_segments=None,
            host_kv_layers=["qsa.0"],
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=0),
        model_state=SimpleNamespace(max_encoder_len=0),
        is_pooling_model=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        host_history=store,
    )
    return runner, []


def test_host_kv_warmup_profiles_start_at_first_eager_size() -> None:
    runner, _ = _host_kv_runner()
    # First size dispatch() cannot route to a captured graph (64), above
    # the advertised profile 33.
    assert _host_kv_warmup_profiles(runner, (6, 33)) == [(65, 1), (65, 2)]
    runner, _ = _host_kv_runner(max_captured_tokens=256)
    assert _host_kv_warmup_profiles(runner, (6, 33)) == []


def test_host_kv_warmup_runs_host_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every warmup iteration carries scheduler-side host KV metadata, so the
    step-head role kernels, the union plan and the staged write plan run
    before serving; eager profiles cover both staged write plan kinds."""
    runner, executions_holder = _host_kv_runner()
    executions = executions_holder

    def execute(output: Any) -> None:
        executions.append(output)

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    warmup_kernels(runner, execute, lambda _: None)

    # 4 x 6 prefill, 4 x 5 decode, cleanup, 33/1, 65/1 eager, 65/2 eager.
    assert [o.total_num_scheduled_tokens for o in executions] == [
        24,
        20,
        0,
        33,
        5,
        0,
        65,
        5,
        0,
        130,
        10,
        0,
    ]

    shape = runner.host_history.shape
    prefill = executions[6]  # eager single-request prefill
    step = prefill.host_kv
    req_ids = list(prefill.num_scheduled_tokens)
    assert len(req_ids) == 1
    rows, groups, kind = step.requests[req_ids[0]]
    assert (rows, kind) == (65, 1)
    assert groups == history_groups(65, shape.dcp_world_size, shape.interleave)
    claim = list(prefill.scheduled_new_reqs[0].block_ids[0])
    assert step.claim == claim
    assert step.bindings == [(hid, hid) for hid in claim]
    assert prefill.new_block_ids_to_zero == claim

    decode = executions[7]
    decode_step = decode.host_kv
    for req_id in decode.num_scheduled_tokens:
        rows, groups, kind = decode_step.requests[req_id]
        assert (rows, kind) == (5, 0)
        assert groups == history_groups(70, shape.dcp_world_size, shape.interleave)

    cleanup = executions[8]
    assert sorted(cleanup.host_kv.donate) == sorted(claim)
    assert cleanup.host_kv.requests == {}

    # The eager multi-request profile cannot use the single-request runs
    # plan: its step still claims and binds its history blocks.
    multi = executions[9]
    assert multi.host_kv.claim
    assert len(multi.host_kv.requests) == 2

    # Serving must still perform its own first-step reset.
    assert runner._host_history_live is False


def test_host_kv_warmup_static_pool_has_no_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, executions = _host_kv_runner(slots_per_block=0)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    warmup_kernels(runner, executions.append, lambda _: None)

    prefill = next(
        o for o in executions if o.host_kv is not None and o.scheduled_new_reqs
    )
    assert prefill.host_kv.claim == []
    assert prefill.host_kv.bindings  # pages are still bound for staged writes
    cleanup = next(o for o in executions if o.total_num_scheduled_tokens == 0)
    assert cleanup.host_kv is None


def test_warmup_without_host_kv_attaches_no_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention_spec = _full_attention_spec()
    runner = SimpleNamespace(
        num_speculative_steps=4,
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=128),
        max_model_len=64,
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=attention_spec, layer_names=[])
            ],
            num_blocks=64,
            block_segments=None,
            host_kv_layers=[],
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=4),
        model_state=SimpleNamespace(max_encoder_len=0),
        is_pooling_model=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )
    executions: list[Any] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    warmup_kernels(runner, executions.append, lambda _: None)

    assert all(output.host_kv is None for output in executions)
    assert all(output.new_block_ids_to_zero is None for output in executions)
