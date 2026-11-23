# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash verifies fewer drafts per request at high concurrency."""

import json
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.scheduler import SchedulerConfig
from vllm.config.speculative import (
    SpeculativeConfig,
    check_dflash_concurrency_drafts,
)
from vllm.config.vllm import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

pytestmark = pytest.mark.skip_global_cleanup


pytest_plugins = ("tests.v1.spec_decode.test_dflash2_native_lookup",)


@pytest.fixture
def make_config(native_allocation_config):
    model = native_allocation_config.model_config
    parallel = native_allocation_config.parallel_config
    draft = native_allocation_config.speculative_config.model

    def make(fields):
        return SpeculativeConfig(
            method="dflash",
            model=draft,
            num_speculative_tokens=7,
            target_model_config=model,
            target_parallel_config=parallel,
            **json.loads(fields),
        )

    return make


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ('{"5":4,"4":3}', [[1, 3, 7], [4, 4, 3], [5, 5, 4]], [7, 7, 7, 3, 4, 4, 4, 4]),
        ('{"4":3,"5":4}', [[1, 3, 7], [4, 4, 3], [5, 8, 4]], [7, 7, 7, 3, 4, 4, 4, 4]),
        ('{"4":3,"5":4}', [[1, 3, 7], [4, 4, 3], [5, 5, 4]], [7, 7, 7, 3]),
        ('{"3":3}', [[1, 2, 7], [3, 8, 3]], [7, 7, 3, 3, 3, 3, 3, 3]),
        ('{"8":4}', [[1, 7, 7], [8, 8, 4]], [7, 7, 7, 7, 7, 7, 7, 4]),
    ],
)
def test_alias_migration(make_config, old, new, expected):
    legacy = make_config('{"dflash_concurrency_drafts":' + old + "}")
    canonical = make_config(json.dumps({"num_speculative_tokens_per_batch_size": new}))
    assert legacy.dflash_concurrency_drafts is None
    for spec in (legacy, canonical):
        assert (
            build_dynamic_sd_schedule_lookup(
                spec.num_speculative_tokens_per_batch_size, len(expected), 7
            )[1:]
            == expected
        )


def test_alias_config_copy_and_transport(make_config, monkeypatch):
    warn = MagicMock()
    monkeypatch.setattr("vllm.config.speculative.logger.warning_once", warn)
    legacy = make_config('{"dflash_concurrency_drafts":{"4":3,"5":4}}')
    schedule = [(1, 3, 7), (4, 4, 3), (5, 5, 4)]
    warn.assert_called_once()
    assert "num_speculative_tokens_per_batch_size" in warn.call_args.args[0]
    rebuilt = SpeculativeConfig.from_worker_payload(
        json.loads(json.dumps(legacy.to_worker_payload())),
        target_model_config=legacy.target_model_config,
        target_parallel_config=legacy.target_parallel_config,
    )
    for spec in (
        legacy,
        legacy.__replace__(enforce_eager=True),
        ForkingPickler.loads(ForkingPickler.dumps(legacy)),
        rebuilt,
    ):
        assert spec.dflash_concurrency_drafts is None
        assert spec.num_speculative_tokens_per_batch_size == schedule


def test_empty_alias_preserves_fixed_width(make_config):
    fixed = make_config('{"dflash_concurrency_drafts":{}}')
    assert fixed.num_speculative_tokens == 7
    assert fixed.dflash_concurrency_drafts is None
    assert fixed.num_speculative_tokens_per_batch_size is None


def test_alias_conflict_even_with_empty_fields(make_config):
    with pytest.raises(ValueError, match="Set only one of dflash_concurrency_drafts"):
        make_config(
            '{"dflash_concurrency_drafts":{},"num_speculative_tokens_per_batch_size":[]}'
        )


@pytest.mark.parametrize(
    ("policy", "dflash", "ngram_assist", "ok"),
    [
        (None, False, False, True),
        ({5: 4}, True, False, True),
        ({5: 4}, False, False, False),  # not DFlash
        ({5: 4}, True, True, False),  # lookup owns the width
        ({1: 4}, True, False, False),  # a single request keeps the full width
        ({5: 7}, True, False, True),  # the full width, e.g. after a narrower step
        ({5: 8}, True, False, False),  # wider than the proposal
        ({5: 0}, True, False, False),
        ({5: True}, True, False, False),  # a bool is not a draft count
    ],
)
def test_policy_validation(policy, dflash, ngram_assist, ok) -> None:
    def check():
        check_dflash_concurrency_drafts(
            policy, dflash=dflash, ngram_assist=ngram_assist, num_speculative_tokens=7
        )

    if ok:
        check()
    else:
        with pytest.raises(ValueError, match="dflash_concurrency_drafts"):
            check()


def _manager(monkeypatch, schedule, tails=True, sizes=None):
    # A V100 worker without the long-context attention operator.
    monkeypatch.setattr(
        "vllm.v1.attention.ops.sm70_e4m3_long.long_attention_enabled", lambda: False
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils._worker_device_is_pre_ampere", lambda: True
    )
    monkeypatch.setenv("VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS", str(int(tails)))
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_global_graph_pool",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_pp_group",
        lambda: MagicMock(is_first_rank=True, is_last_rank=True),
    )
    sizes = sizes or [8, 16, 24, 32, 40, 48, 56, 64]
    config = SimpleNamespace(
        adaptive_dflash2_enabled=False,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(tensor_parallel_size=4, data_parallel_size=1),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=sizes,
            max_cudagraph_capture_size=max(sizes),
        ),
        speculative_config=SimpleNamespace(
            method="dflash",
            ngram_assist=False,
            num_speculative_tokens=7,
            num_speculative_tokens_per_batch_size=schedule,
        ),
    )
    manager = ModelCudaGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, decode_query_len=8
    )
    manager._graphs_captured = True
    return manager


@pytest.mark.parametrize("tails", [True, False])
def test_narrow_target_graphs_from_the_threshold_up(monkeypatch, tails) -> None:
    manager = _manager(monkeypatch, [(1, 4, 7), (5, 8, 4)], tails)
    assert manager.decode_query_lens[0] == 8 and 5 in manager.decode_query_lens
    for num_reqs in range(5, 9):
        assert num_reqs * 5 in manager._capture_sizes
        desc = manager.dispatch(num_reqs, num_reqs * 5, 5)
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.uniform_token_count == 5 and desc.num_reqs == num_reqs
    # The full width keeps every batch size; the narrow width stays above 4.
    assert manager.dispatch(8, 64, 8).cg_mode == CUDAGraphMode.FULL
    assert manager.dispatch(2, 16, 8).cg_mode == CUDAGraphMode.FULL
    assert manager.dispatch(2, 10, 5).uniform_token_count != 5
    # A single-request tail graph of the same length stays with the tail policy.
    assert (manager.dispatch(1, 5, 5).cg_mode == CUDAGraphMode.FULL) is tails


def test_no_policy_keeps_the_capture_plan(monkeypatch) -> None:
    manager = _manager(monkeypatch, None, tails=False)
    assert manager.decode_query_lens == (8,)
    assert not manager._dflash_width_reqs


@pytest.mark.parametrize(
    "fields",
    [
        '{"dflash_concurrency_drafts":{"4":3,"5":4}}',
        '{"num_speculative_tokens_per_batch_size":[[1,3,7],[4,4,3],[5,8,4]]}',
    ],
)
def test_ladder_widths_capture_only_their_request_counts(
    monkeypatch, make_config, fields
):
    schedule = make_config(fields).num_speculative_tokens_per_batch_size
    # {4: 3, 5: 4}: three drafts (q4) for four requests, four (q5) from five up,
    # all seven (q8) for one to three.
    manager = _manager(monkeypatch, schedule, tails=False)
    assert manager._dflash_width_reqs == {4: frozenset({4}), 5: frozenset(range(5, 9))}
    desc = manager.dispatch(4, 16, 4)
    assert desc.cg_mode == CUDAGraphMode.FULL and desc.uniform_token_count == 4
    for num_reqs in range(5, 9):
        desc = manager.dispatch(num_reqs, num_reqs * 5, 5)
        assert desc.cg_mode == CUDAGraphMode.FULL and desc.uniform_token_count == 5
    # q4 is not a graph shape for the counts that verify four drafts, nor q5 for four.
    assert manager.dispatch(6, 24, 4).uniform_token_count != 4
    assert manager.dispatch(4, 20, 5).uniform_token_count != 5
    assert manager.dispatch(3, 24, 8).cg_mode == CUDAGraphMode.FULL


def test_narrowing_policy_drops_unused_wide_shapes(monkeypatch) -> None:
    # {3: 5, 5: 3}: six-token rows only serve three and four requests now.
    manager = _manager(monkeypatch, [(1, 2, 7), (3, 4, 5), (5, 8, 3)], tails=False)
    assert manager._dflash_width_reqs == {
        6: frozenset({3, 4}),
        4: frozenset(range(5, 9)),
    }


def test_added_sizes_stay_within_the_largest_capture_size(monkeypatch) -> None:
    # Explicit capture sizes up to 32: 6-token widths for 6..8 requests (36..48
    # tokens) would leave the compiled decode range.
    manager = _manager(monkeypatch, [(1, 1, 7), (2, 8, 5)], sizes=[8, 16, 24, 32])
    assert max(manager._capture_sizes) == 32
    assert {12, 18, 24, 30} <= set(manager._capture_sizes)
    assert not {36, 42, 48} & set(manager._capture_sizes)


def test_schedule_captures_single_request_and_zero_draft_width(monkeypatch):
    manager = _manager(monkeypatch, [(1, 1, 3), (2, 8, 0)], tails=False)
    assert manager._dflash_width_reqs == {4: frozenset({1}), 1: frozenset(range(2, 9))}
    for c, q in ((1, 4), (4, 1)):
        desc = manager.dispatch(c, c * q, q)
        assert (desc.cg_mode, desc.num_reqs, desc.uniform_token_count) == (
            CUDAGraphMode.FULL,
            c,
            q,
        )


@pytest.mark.parametrize("async_scheduling", [None, True])
def test_schedule_defaults_to_sync_but_accepts_explicit_async(
    make_config, monkeypatch, async_scheduling
):
    # The CPU platform turns async off after the model/executor checks.
    monkeypatch.setattr(
        current_platform, "check_and_update_config", lambda config: None
    )
    spec = make_config('{"dflash_concurrency_drafts":{"4":3}}')
    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
        ),
        speculative_config=spec,
    )
    assert config.scheduler_config.async_scheduling is bool(async_scheduling)


def test_schedule_is_incompatible_with_legacy_ngram_assist(make_config):
    spec = make_config(
        '{"ngram_assist":true,"num_speculative_tokens_per_batch_size":[[1,8,3]]}'
    )
    with pytest.raises(ValueError, match="incompatible with ngram_assist"):
        VllmConfig(speculative_config=spec)


def test_schedule_leaves_the_compile_hash_alone(make_config):
    fixed = make_config("{}")
    dynamic = make_config('{"num_speculative_tokens_per_batch_size":[[1,8,3]]}')
    assert fixed.compute_hash() == dynamic.compute_hash()
