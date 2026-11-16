# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU equivalence and lifetime checks for optional MRV2 preparation reuse."""

import random
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu.gdn_spec_replay import replay_graph_key
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

COMMON = "REVOLTA_RESEARCH_PREGRAPH_GDN_COMMON_CACHE"
VIEWS = "REVOLTA_RESEARCH_PREGRAPH_REPLAY_VIEWS"


def _flag(monkeypatch, name, enabled):
    monkeypatch.delitem(vars(envs), name, raising=False)
    monkeypatch.setenv(name, "1" if enabled else "0")


def _equal_common(ref, actual):
    assert (ref is None) == (actual is None)
    if ref is None:
        return
    for field in fields(ref):
        a, b = getattr(ref, field.name), getattr(actual, field.name)
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape, field.name
            assert torch.equal(a, b), field.name
        else:
            assert a == b, field.name


def _common_args(qsl, drafts, gpu_offsets=None, k=7, legacy=False):
    return dict(
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
        query_start_loc=(
            torch.tensor(qsl, dtype=torch.int32) if gpu_offsets is None else gpu_offsets
        ),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        num_spec_state_tokens=k,
        legacy_mixed_decode_routing=legacy,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_common_cache_matches_builder_across_batch_transitions(legacy):
    rng = random.Random(119)
    cache = attn_utils.CommonGDNMetadataCache()
    # One persistent input buffer, just as in the runner. Slots, request IDs,
    # token IDs, accepted counts and state IDs are deliberately absent from
    # the cache: the original common builder does not consume them either.
    offsets = torch.empty(17, dtype=torch.int32)
    for _ in range(150):
        n, pad, k = rng.randint(1, 8), rng.randint(0, 3), rng.choice([3, 4, 7])
        lens = [k + 1] * n + [0] * pad
        qsl = [0] + list(np.cumsum(lens))
        drafts = [k] * n + [-1] * pad
        gpu_view = offsets[: len(qsl)]
        gpu_view.copy_(torch.tensor(qsl))
        args = _common_args(qsl, drafts, gpu_view, k, legacy)
        ref = attn_utils.compute_common_gdn_attn_metadata(**args)
        actual = cache.get(**args)
        _equal_common(ref, actual)
        assert cache.get(**args) is actual
        assert actual.spec_query_start_loc.data_ptr() == gpu_view.data_ptr()
        # A hit must not write either the live input or retained constants.
        assert gpu_view.tolist() == qsl
        _equal_common(ref, actual)


def test_common_cache_skips_work_on_hit_and_invalidates_every_dependency(monkeypatch):
    compute = attn_utils.compute_common_gdn_attn_metadata
    calls = []

    def tracked(**kwargs):
        calls.append(kwargs)
        return compute(**kwargs)

    monkeypatch.setattr(attn_utils, "compute_common_gdn_attn_metadata", tracked)
    cache = attn_utils.CommonGDNMetadataCache()
    offsets = torch.tensor([0, 8, 16], dtype=torch.int32)
    args = _common_args([0, 8, 16], [7, 7], offsets)
    first = cache.get(**args)
    # Distinct CPU tensors with identical values must still hit.
    assert cache.get(**_common_args([0, 8, 16], [7, 7], offsets)) is first
    assert len(calls) == 1
    variants = [
        _common_args([0, 4, 8], [3, 3], offsets, k=3),
        _common_args([0, 4, 8], [3, 3], offsets, k=4),
        _common_args([0, 4, 8], [3, 3], offsets, k=4, legacy=True),
        # Same storage and size but a different stride.
        _common_args([0, 4, 8], [3, 3], torch.empty(6, dtype=torch.int32)[::2]),
        # A replacement buffer with the same shape cannot reuse the old view.
        _common_args([0, 4, 8], [3, 3], offsets.clone()),
        _common_args([0, 4], [3], offsets[:2]),
        _common_args([0, 5], [3], offsets[:2]),
        _common_args([0, 5], [2], offsets[:2]),
    ]
    for changed in variants:
        changed["query_start_loc"].copy_(changed["query_start_loc_cpu"])
        result = cache.get(**changed)
        _equal_common(compute(**changed), result)
    assert len(calls) == 1 + len(variants)


@pytest.mark.parametrize("drafts", [[-1, -1], [0, 0], [7, -1], [-1, 7]])
def test_common_cache_never_retains_non_spec_or_mixed_batches(drafts):
    cache = attn_utils.CommonGDNMetadataCache()
    args = _common_args([0, 8, 16], drafts)
    _equal_common(
        attn_utils.compute_common_gdn_attn_metadata(**args), cache.get(**args)
    )
    assert cache._metadata is None
    assert cache._key is None


def _state():
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state.device = torch.device("cpu")
    state.max_model_len = 4096
    state.num_accepted_tokens_gpu = torch.arange(1, 9, dtype=torch.int32)
    state._use_common_gdn_metadata = True
    state._use_dflash2_fused_gdn_metadata = False
    state._use_mtp4_fused_gdn_metadata = False
    state._use_dflash2_grouped_smallq_metadata = False
    state._spec_state_replay = False
    state.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_state_tokens=lambda: 7)
    )
    return state


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])
@pytest.mark.parametrize("kind", ["ordinary", "spec", "prefill", "mixed", "capture"])
def test_prepare_attn_real_entry_with_flags_enabled(monkeypatch, mode, kind):
    # Execute the real model-state entry, including the ordinary non-spec
    # request with both switches on. No device kernels or model are mocked as
    # evidence of GPU/serving correctness: only the downstream backend is a
    # CPU recorder because this test exercises preparation and routing.
    state = _state()
    _flag(monkeypatch, COMMON, True)
    _flag(monkeypatch, VIEWS, True)
    _flag(monkeypatch, "VLLM_SM70_DFLASH2_GDN_SYNC_ASSERT", False)
    counts = np.array([8, 8], dtype=np.int32)
    drafts = np.array([7, 7], dtype=np.int32) if kind != "ordinary" else None
    prefill = np.array([kind in ("prefill", "mixed"), kind == "prefill"])
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=16,
        num_reqs_after_padding=2,
        num_tokens_after_padding=16,
        query_start_loc_np=np.array([0, 8, 16], dtype=np.int32),
        query_start_loc=torch.tensor([0, 8, 16], dtype=torch.int32),
        num_scheduled_tokens=counts,
        is_prefilling_np=prefill,
        idx_mapping=torch.tensor([4, 1]),
        num_draft_tokens_per_req=drafts,
        seq_lens=torch.tensor([108, 208]),
        seq_lens_cpu_upper_bound=torch.tensor([108, 208]),
        dcp_local_seq_lens=None,
        prefix_anchor_lens=None,
        is_dummy_batch=False,
    )

    def record(**kwargs):
        specific = kwargs["model_specific_attn_metadata"]
        return dict(
            common=specific.common_gdn_metadata, accepted=specific.num_accepted_tokens
        )

    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", record)
    results = []
    for enabled in [False, True, True]:
        _flag(monkeypatch, COMMON, enabled)
        results.append(
            state.prepare_attn(
                batch,
                mode,
                (),
                torch.empty((0, 16), dtype=torch.int32),
                [],
                SimpleNamespace(kv_cache_groups=[]),
                for_capture=kind == "capture",
            )
        )
    _equal_common(results[0]["common"], results[1]["common"])
    _equal_common(results[0]["common"], results[2]["common"])
    if kind != "capture":
        assert results[2]["accepted"].tolist() == [5, 2]
    if mode == CUDAGraphMode.FULL and kind == "spec":
        assert results[1]["common"] is results[2]["common"]
        batch.idx_mapping = torch.tensor([1, 4])
        state.num_accepted_tokens_gpu[1] = 7
        state.num_accepted_tokens_gpu[4] = 2
        reordered = state.prepare_attn(
            batch,
            mode,
            (),
            torch.empty((0, 16), dtype=torch.int32),
            [],
            SimpleNamespace(kv_cache_groups=[]),
        )
        assert reordered["common"] is results[2]["common"]
        assert reordered["accepted"].tolist() == [7, 2]
    else:
        assert getattr(state, "_pregraph_gdn_common_cache", None) is None or (
            state._pregraph_gdn_common_cache._metadata is None
        )


def _replay_fixture(n=4, groups=3, layers_per_group=12):
    state = _state()
    config_groups, metadata, context, tables = [], {}, {}, []
    for gid in range(groups):
        names = [f"layer.{gid}.{j}" for j in range(layers_per_group)]
        config_groups.append(SimpleNamespace(layer_names=names))
        table = torch.arange(n * 10, dtype=torch.int32).view(n, 10) + gid * 100
        tables.append(table)
        common = GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=2,
            num_spec_decode_tokens=16,
            num_actual_tokens=n * 8,
            spec_state_indices_tensor=table,
            spec_query_start_loc=torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32),
        )
        for name in names:
            context[name] = SimpleNamespace(spec_state_replay=True)
            metadata[name] = common
    state.vllm_config.compilation_config = SimpleNamespace(
        static_forward_context=context
    )
    return (
        state,
        metadata,
        SimpleNamespace(kv_cache_groups=config_groups),
        tuple(tables),
    )


@pytest.mark.parametrize("full", [True, False])
def test_replay_views_preserve_values_layout_order_and_live_updates(monkeypatch, full):
    state, metadata, config, tables = _replay_fixture()
    drafts = torch.tensor([7, 7, -1, -1], dtype=torch.int32)
    steps = []
    for enabled in [False, True]:
        _flag(monkeypatch, VIEWS, enabled)
        steps.append(
            state._make_spec_replay_step(
                metadata,
                config,
                tables,
                drafts,
                full_graph=full,
                num_reqs=4,
                num_tokens=32,
            )
        )
    ref, actual = steps
    assert ref.num_reqs == actual.num_reqs
    assert ref.num_tokens == actual.num_tokens
    assert ref.block_tables is actual.block_tables is tables
    assert torch.equal(ref.spec_rows, actual.spec_rows)
    assert len(ref.layers) == len(actual.layers) == 36
    computed = torch.tensor([1024] * 4, dtype=torch.int32)
    assert replay_graph_key(ref, computed, 1664) == replay_graph_key(
        actual, computed, 1664
    )
    for a, b in zip(ref.layers, actual.layers):
        assert a.layer is b.layer and a.group_id == b.group_id
        for name in ["base_state_ids", "cu_seqlens"]:
            av, bv = getattr(a, name), getattr(b, name)
            assert torch.equal(av, bv)
            assert av.data_ptr() == bv.data_ptr() and av.stride() == bv.stride()
    assert (actual.layers[0].base_state_ids is actual.layers[1].base_state_ids) == full
    assert actual.layers[0].base_state_ids is not actual.layers[12].base_state_ids
    # Live state contents can change; views must follow rather than snapshot.
    tables[0][0, 0] = 999
    assert ref.layers[0].base_state_ids[0] == actual.layers[0].base_state_ids[0] == 999
    # A new metadata object within the same group cannot reuse another view.
    first_name = config.kv_cache_groups[0].layer_names[0]
    metadata[first_name] = GDNAttentionMetadata(
        **{
            **vars(metadata[first_name]),
            "spec_state_indices_tensor": tables[0].clone(),
        }
    )
    changed = state._make_spec_replay_step(
        metadata,
        config,
        tables,
        drafts,
        full_graph=full,
        num_reqs=4,
        num_tokens=32,
    )
    assert (
        changed.layers[0].base_state_ids.data_ptr()
        != changed.layers[1].base_state_ids.data_ptr()
    )


def test_switches_default_off_and_do_not_change_compile_identity(monkeypatch):
    for name in [COMMON, VIEWS]:
        monkeypatch.delitem(vars(envs), name, raising=False)
        monkeypatch.delenv(name, raising=False)
        assert getattr(envs, name) is False
    off = envs.compile_factors()
    for name in [COMMON, VIEWS]:
        _flag(monkeypatch, name, True)
        assert getattr(envs, name) is True
    assert envs.compile_factors() == off
