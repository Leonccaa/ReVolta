# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash verifies fewer drafts per request at high concurrency."""

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.scheduler import SchedulerConfig
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.config.speculative import (
    SpeculativeConfig,
    check_dflash_concurrency_drafts,
    dflash_drafts_for_requests,
)
from vllm.config.vllm import VllmConfig
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("policy", "num_reqs", "drafts"),
    [
        (None, 8, 7),
        ({5: 4}, 1, 7),
        ({5: 4}, 4, 7),
        ({5: 4}, 5, 4),
        ({5: 4}, 8, 4),
        ({3: 5, 5: 3}, 3, 5),
        ({3: 5, 5: 3}, 6, 3),
        # A ladder: each entry holds up to the next one, and may widen again.
        ({4: 3, 5: 4}, 3, 7),
        ({4: 3, 5: 4}, 4, 3),
        ({4: 3, 5: 4}, 5, 4),
        ({4: 3, 5: 4}, 8, 4),
        ({2: 4, 4: 7}, 3, 4),
        ({2: 4, 4: 7}, 5, 7),
    ],
)
def test_drafts_follow_the_running_request_count(policy, num_reqs, drafts) -> None:
    spec = SimpleNamespace(num_speculative_tokens=7, dflash_concurrency_drafts=policy)
    assert dflash_drafts_for_requests(spec, num_reqs) == drafts


def _min_semantics(policy, num_reqs, nst=7):
    """The first release: every threshold reached may only lower the width."""
    drafts = nst
    for min_reqs, width in (policy or {}).items():
        if num_reqs >= min_reqs:
            drafts = min(drafts, width)
    return drafts


def _monotone_policies(nst=7, max_reqs=8):
    thresholds = range(2, max_reqs + 1)
    for size in (1, 2, 3):
        for keys in itertools.combinations(thresholds, size):
            for widths in itertools.combinations_with_replacement(
                range(nst - 1, 0, -1), size
            ):
                yield dict(zip(keys, widths))  # widths never increase with the count


def test_ladder_matches_the_first_release_for_narrowing_policies() -> None:
    # Every policy the first release could express (widths that only shrink as
    # requests grow) resolves to the same width at every request count.
    checked = 0
    for policy in _monotone_policies():
        spec = SimpleNamespace(
            num_speculative_tokens=7, dflash_concurrency_drafts=policy
        )
        for num_reqs in range(0, 13):
            assert dflash_drafts_for_requests(spec, num_reqs) == _min_semantics(
                policy, num_reqs
            ), (policy, num_reqs)
            checked += 1
    assert checked > 10_000


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


def _manager(monkeypatch, policy, tails=True, sizes=None):
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
            dflash_concurrency_drafts=policy,
        ),
    )
    manager = ModelCudaGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, decode_query_len=8
    )
    manager._graphs_captured = True
    return manager


@pytest.mark.parametrize("tails", [True, False])
def test_narrow_target_graphs_from_the_threshold_up(monkeypatch, tails) -> None:
    manager = _manager(monkeypatch, {5: 4}, tails)
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


def test_ladder_widths_capture_only_their_request_counts(monkeypatch) -> None:
    # {4: 3, 5: 4}: three drafts (q4) for four requests, four (q5) from five up,
    # all seven (q8) for one to three.
    manager = _manager(monkeypatch, {4: 3, 5: 4}, tails=False)
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
    manager = _manager(monkeypatch, {3: 5, 5: 3}, tails=False)
    assert manager._dflash_width_reqs == {
        6: frozenset({3, 4}),
        4: frozenset(range(5, 9)),
    }


def test_added_sizes_stay_within_the_largest_capture_size(monkeypatch) -> None:
    # Explicit capture sizes up to 32: 6-token widths for 6..8 requests (36..48
    # tokens) would leave the compiled decode range.
    manager = _manager(monkeypatch, {2: 5}, sizes=[8, 16, 24, 32])
    assert max(manager._capture_sizes) == 32
    assert {12, 18, 24, 30} <= set(manager._capture_sizes)
    assert not {36, 42, 48} & set(manager._capture_sizes)


def _speculator(cls, policy, **attrs):
    spec = object.__new__(cls)
    spec.speculative_config = SimpleNamespace(
        num_speculative_tokens=7, dflash_concurrency_drafts=policy
    )
    spec.num_speculative_steps = 7
    spec._running_reqs = 0
    for key, value in attrs.items():
        setattr(spec, key, value)
    return spec


@pytest.mark.parametrize("cls", [DFlashSpeculator, DFlash2Speculator])
def test_both_dflash_speculators_follow_the_policy(cls) -> None:
    spec = _speculator(cls, {4: 3}, _lookup_enabled=False)
    for num_reqs, drafts in ((1, 7), (3, 7), (4, 3), (8, 3)):
        spec._prepare_proposal_runtime(
            SimpleNamespace(num_reqs=num_reqs), torch.zeros(1), torch.zeros(1)
        )
        assert spec.next_num_draft_tokens() == drafts


def test_without_policy_the_plain_speculator_keeps_the_full_proposal() -> None:
    # None: the draft-token handler keeps the proposal tensor width, as before.
    spec = _speculator(DFlashSpeculator, None)
    spec._prepare_proposal_runtime(
        SimpleNamespace(num_reqs=8), torch.zeros(1), torch.zeros(1)
    )
    assert spec.next_num_draft_tokens() is None


def _width_config(policy) -> SpeculativeConfig:
    config = object.__new__(SpeculativeConfig)
    config.method = "dflash"
    config.ngram_assist = False
    config.num_speculative_tokens = 7
    config.dflash_concurrency_drafts = policy
    config.sm70_dflash2 = Sm70DFlash2Config()
    config.ddtree_disable_tree_verify = False
    config.disable_padded_drafter_batch = False
    config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            dflash_config={"block_size": 8, "selector_top_k": 16}
        ),
        verify_with_parallel_config=lambda *_args: None,
        verify_with_model_config=lambda *_args: None,
    )
    config.verify_with_parallel_config = lambda *_args: None
    config.verify_with_model_config = lambda *_args: None
    return config


def _scheduler(async_scheduling):
    return SchedulerConfig(
        max_model_len=8192, is_encoder_decoder=False, async_scheduling=async_scheduling
    )


def test_policy_disables_async_scheduling_by_default() -> None:
    config = VllmConfig(
        scheduler_config=_scheduler(None), speculative_config=_width_config({4: 3})
    )
    assert config.scheduler_config.async_scheduling is False


def test_policy_rejects_explicit_async_scheduling() -> None:
    with pytest.raises(ValueError, match="ignores dflash_concurrency_drafts"):
        VllmConfig(
            scheduler_config=_scheduler(True),
            speculative_config=_width_config({4: 3}),
        )


def test_policy_leaves_the_compile_hash_alone() -> None:
    def config(policy):
        spec = object.__new__(SpeculativeConfig)
        spec.method = "dflash"
        spec.use_local_argmax_reduction = False
        spec.sm70_dflash2 = SimpleNamespace(resolved=False, explicit_fields=())
        spec.draft_model_config = None
        spec.ddtree_disable_tree_verify = False
        spec.dflash_concurrency_drafts = policy
        return spec

    assert config(None).compute_hash() == config({4: 3}).compute_hash()
