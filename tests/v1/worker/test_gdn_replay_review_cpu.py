# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent replay review (#20): CPU contracts.

Written for the review of #21; the replay policy and the verifier admission
assertions follow what 0.1.3 settled (#21: replay=None yields to the packed
verifier under the qualified contract; with replay on the mixed-QKV verifier
admits every verify batch).
"""

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm.config.vllm as config_module
from vllm import envs
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.cpu.spec import mamba_state_slots

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _config(policy):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_dflash=lambda: True,
            use_dflash_ddtree=lambda: False,
            sm70_dflash2=policy,
        ),
        model_config=SimpleNamespace(is_hybrid=True, dtype=torch.float16),
        use_v2_model_runner=True,
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=8),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, tensor_parallel_size=4
        ),
    )


@pytest.fixture
def resolve(monkeypatch):
    resolver = getattr(config_module, "_resolve_gdn_spec_state_replay", None)
    if resolver is None:
        pytest.skip("Replay resolver is absent on the baseline")
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        config_module, "_any_participating_device_is_capability", lambda *a: True
    )
    monkeypatch.setattr(envs, "VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", False)
    monkeypatch.setattr(envs, "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV", True)
    return resolver


def test_qualified_default_yields_replay_to_the_packed_verifier(resolve, monkeypatch):
    # 0.1.3 policy (#21): under the qualified contract the packed verifier is the
    # default, and replay=None yields to it; an explicit replay=True turns the
    # automatic packed default off. Explicitly enabling both is rejected
    # (tests/config/test_gdn_replay_policy.py).
    for name in envs.environment_variables:
        if name.startswith("VLLM_SM70_DFLASH2_"):
            monkeypatch.delenv(name, raising=False)
    policy = Sm70DFlash2Config()
    # VllmConfig resolves this policy before resolving replay.
    policy.resolve(qualified=True)
    assert policy.fused_gdn_verify
    assert not resolve(_config(policy))
    assert policy.fused_gdn_verify

    explicit = Sm70DFlash2Config(spec_state_replay=True)
    explicit.resolve(qualified=True)
    assert resolve(_config(explicit))
    assert not explicit.fused_gdn_verify


@pytest.mark.parametrize("requests", [1, 2, 4, 8])
@pytest.mark.parametrize("wide", [False, True])
def test_enabled_replay_admits_the_whole_verify_batch(
    resolve, monkeypatch, requests, wide
):
    policy = Sm70DFlash2Config(fused_gdn_verify=False)
    monkeypatch.setattr(envs, "VLLM_SM70_QWEN38_WIDE_M_GDN", wide)
    monkeypatch.setattr(envs, "VLLM_SM70_GDN_TAIL_COPY", False)
    enabled = resolve(_config(policy))
    num_tokens = requests * 8
    layer = SimpleNamespace(
        spec_state_replay=enabled,
        enable_sm70_fused_sigmoid_mixed_qkv=True,
        num_k_heads=16,
        num_v_heads=48,
        tp_size=4,
    )
    cls = gdn.QwenGatedDeltaNetAttention
    # The layer-level admission _forward_core uses (#21): with replay on the
    # mixed-QKV verifier takes every verify batch, so the split verifier (which
    # raises under replay) is never reached; with replay off the token band is
    # the module predicate, unchanged.
    admitted = cls._mixed_qkv_verify_admits(layer, num_tokens)
    assert admitted == (enabled or gdn._sm70_mixed_qkv_verify_tokens_ok(num_tokens))
    if enabled:
        assert admitted, f"Replay is on but the verifier rejects {num_tokens} tokens"
        # Batches outside the fused input projection pass a row-strided view.
        qkvz = torch.zeros(num_tokens, 2560 + 1536, dtype=torch.float16)
        assert cls._replay_reads_strided_qkv(layer, qkvz[:, :2560])


def _spec(spec_tokens, replay=False):
    spec_config = SimpleNamespace(
        gdn_spec_state_replay=replay,
        num_speculative_state_tokens=lambda: spec_tokens,
    )
    allocate = getattr(SpeculativeConfig, "num_speculative_state_blocks", None)
    blocks = allocate(spec_config) if allocate is not None else spec_tokens
    kwargs = dict(
        block_size=16,
        shapes=((3, 8), (2, 4, 4)),
        dtypes=(torch.float16, torch.float32),
        mamba_cache_mode="align",
        num_speculative_blocks=blocks,
    )
    if "num_replayed_speculative_blocks" in MambaSpec.__dataclass_fields__:
        kwargs["num_replayed_speculative_blocks"] = spec_tokens - blocks
    return MambaSpec(**kwargs)


def test_replay_releases_the_seven_allocated_draft_blocks():
    # A positive feature regression: baseline fails, replay branch passes.
    assert _spec(7, replay=True).num_speculative_blocks == 0


@pytest.mark.parametrize("drafts,replay", [(0, False), (4, False), (7, True)])
def test_real_manager_admission_matches_allocated_blocks_and_preemption(drafts, replay):
    spec = _spec(drafts, replay)
    extra = spec.num_speculative_blocks
    pool = BlockPool(512, False, spec.block_size)
    manager = MambaManager(spec, pool, enable_caching=False, kv_cache_group_id=0)
    free_initial = pool.get_num_free_blocks()
    request = "allocation-review"
    for end in range(32, 321, 32):
        manager.new_step_starts()
        manager.remove_skipped_blocks(request, end - 32)
        before = pool.get_num_free_blocks()
        estimate = manager.get_num_blocks_to_allocate(request, end, [], end - 32, end)
        manager.allocate_new_blocks(request, end, end)
        allocated = before - pool.get_num_free_blocks()
        assert estimate == allocated
        assert sum(not b.is_null for b in manager.req_to_blocks[request]) <= 2 + extra
    manager.free(request)
    assert pool.get_num_free_blocks() == free_initial
    assert request not in manager.last_state_block_idx
    assert request not in manager._allocated_block_reqs
    # Recompute after preemption must be a fresh allocation.
    manager.allocate_new_blocks(request, 16, 16)
    assert sum(not b.is_null for b in manager.req_to_blocks[request]) == 1 + extra
    manager.free(request)
    assert pool.get_num_free_blocks() == free_initial


@pytest.mark.parametrize("reference", [32, 128, 524288])
@pytest.mark.parametrize("retention", [None, 0, 64])
def test_host_state_slot_budget_depends_on_retention_not_draft_blocks(
    reference, retention
):
    old, new = _spec(7), _spec(7, replay=True)
    assert mamba_state_slots(
        128, 16, 16, old, retention, reference
    ) == mamba_state_slots(128, 16, 16, new, retention, reference)


def test_replay_index_contract_exhaustive_near_boundaries():
    pytest.importorskip("vllm.v1.worker.gpu.gdn_spec_replay")
    from vllm.v1.worker.gpu.gdn_spec_replay import build_replay_state_indices

    # Real speculative rows need not be leading batch rows. Distinct base and
    # boundary IDs make an accidental alias visible without a CUDA kernel.
    block_table = torch.arange(40, dtype=torch.int32).reshape(4, 10) + 100
    for accepted in range(1, 9):
        for end in range(16 - 8, 16 + 9):
            out = build_replay_state_indices(
                base_state_ids=torch.tensor([3, 7, -1, -1]),
                num_reqs=4,
                num_columns=9,
                spec_rows=torch.tensor([1, 3]),
                num_accepted=torch.tensor([accepted, 1]),
                num_computed_tokens=torch.tensor([end, 25]),
                block_table=block_table,
                block_size=16,
            )
            expected = [-1] * 9
            expected[-1] = 3
            expected[accepted - 1] = 3
            start = end - accepted + 1
            for pos in range(accepted - 1):
                if (start + pos) % 16 == 0:
                    expected[pos] = int(block_table[1, (start + pos) // 16 - 1])
            assert out[0].tolist() == expected
            assert out[1, 0] == out[1, -1] == 7
            assert (out[2:] == -1).all()


def test_non_replay_forward_still_traces_the_changed_gdn_source(monkeypatch):
    from torch._dynamo.symbolic_convert import InliningInstructionTranslator

    traced = set()
    original = InliningInstructionTranslator.inline_call_

    def record(translator):
        traced.add(translator.f_code.co_filename)
        return original(translator)

    monkeypatch.setattr(InliningInstructionTranslator, "inline_call_", record)

    class MinimalGDN(torch.nn.Module):
        forward = gdn.QwenGatedDeltaNetAttention.forward
        maybe_sm70_qwen_gdn_full_forward = False

        def _forward_method(self, hidden_states, output):
            return hidden_states.sin()

    layer = MinimalGDN()
    torch._dynamo.reset()

    def call(x):
        return layer(x, None)

    torch._dynamo.export(call)(torch.ones(2, 4))
    filename = inspect.getsourcefile(gdn.QwenGatedDeltaNetAttention.forward)
    assert filename is not None
    assert filename in traced
    source = Path(filename).read_bytes()
    # Optional local evidence receipt; this is not a serving environment switch.
    if receipt := os.getenv("REVIEW_TRACE_RECEIPT"):
        Path(receipt).write_text(
            json.dumps(
                {
                    "traced_gdn": True,
                    "source_sha256": hashlib.sha256(source).hexdigest(),
                }
            )
            + "\n"
        )
