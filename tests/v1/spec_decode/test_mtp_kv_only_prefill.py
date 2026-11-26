# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for MTP incomplete-prefill routing and cache writes."""

import importlib
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.qwen4_exp.nvidia import indexer_qsa, qsa
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle


@pytest.mark.parametrize(
    (
        "enabled",
        "incomplete",
        "drafts",
        "mm",
        "dummy",
        "profile",
        "batch_dummy",
        "mode",
        "hit",
    ),
    [
        (False, [True, True], 0, False, False, False, False, CUDAGraphMode.NONE, False),
        (True, [True, True], 0, False, False, False, False, CUDAGraphMode.NONE, True),
        (True, [True, False], 0, False, False, False, False, CUDAGraphMode.NONE, False),
        (True, [True, True], 1, False, False, False, False, CUDAGraphMode.NONE, False),
        (True, [True, True], 0, True, False, False, False, CUDAGraphMode.NONE, False),
        (True, [True, True], 0, False, True, False, False, CUDAGraphMode.NONE, False),
        (True, [True, True], 0, False, False, True, False, CUDAGraphMode.NONE, False),
        (True, [True, True], 0, False, False, False, True, CUDAGraphMode.NONE, False),
        (True, [True, True], 0, False, False, False, False, CUDAGraphMode.FULL, False),
    ],
)
def test_propose_selects_kv_only(
    monkeypatch, enabled, incomplete, drafts, mm, dummy, profile, batch_dummy, mode, hit
):
    monkeypatch.setenv("VLLM_SM70_MTP_KV_ONLY_PREFILL", str(int(enabled)))
    monkeypatch.setattr(eagle, "prepare_eagle_inputs", Mock())
    descriptor = SimpleNamespace(cg_mode=mode, num_tokens=4)
    monkeypatch.setattr(
        eagle, "dispatch_cg_and_sync_dp", Mock(return_value=(descriptor, None))
    )
    spec = object.__new__(eagle.EagleSpeculator)
    spec.method = "mtp"
    spec.num_speculative_steps = 2
    spec.max_num_reqs = 2
    spec.dp_size = 1
    spec.dp_rank = 0
    spec.hidden_states = torch.zeros(4, 4)
    spec.temperature = torch.zeros(2)
    spec.seeds = torch.zeros(2, dtype=torch.int64)
    spec.idx_mapping = torch.zeros(2, dtype=torch.int32)
    spec.draft_logits = None
    spec.draft_tokens = torch.zeros(2, 2, dtype=torch.int64)
    spec.input_buffers = SimpleNamespace()
    spec.last_token_indices = torch.zeros(2, dtype=torch.int64)
    spec.current_draft_step = torch.tensor(0)
    spec.prefill_cudagraph_manager = SimpleNamespace(run_fullgraph=Mock())
    spec.run_kv_only_prefill = Mock(return_value=True)
    spec.prefill = Mock()
    spec._mtp_prefill_begin = Mock()
    spec._mtp_prefill_end = Mock()
    batch = SimpleNamespace(
        num_tokens_after_padding=4,
        num_tokens=4,
        num_reqs=2,
        num_scheduled_tokens=np.array([2, 2]),
        idx_mapping=torch.tensor([0, 1]),
        is_prefilling_np=np.array([True, True]),
        is_incomplete_prefilling_np=np.array(incomplete),
        num_draft_tokens=drafts,
        is_dummy_batch=batch_dummy,
    )
    mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = (
        ([], torch.zeros(4, dtype=torch.bool)) if mm else None
    )
    spec.propose(
        batch,
        {},
        {},
        torch.zeros(4, 4),
        None,
        torch.ones(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        dummy_run=dummy,
        is_profile=profile,
        mm_inputs=mm_inputs,
        num_speculative_tokens=0,
    )
    assert spec.run_kv_only_prefill.called == hit
    assert spec.prefill.called == (not hit and mode == CUDAGraphMode.NONE)
    assert spec.prefill_cudagraph_manager.run_fullgraph.called == (
        mode == CUDAGraphMode.FULL
    )
    assert spec._mtp_prefill_end.called == (not hit)


@pytest.fixture
def mtp_module(monkeypatch):
    # The CPU test environment lacks torchvision, which the unrelated full
    # model import needs. Exercise the real MTP methods with a small fake layer.
    name = "vllm.models.qwen4_exp.nvidia.model"
    stub: Any = types.ModuleType(name)
    stub._HC_WEIGHTS_MAPPER = WeightsMapper()
    stub._QWEN3_5_WEIGHTS_MAPPER = WeightsMapper()
    stub._QWEN4_EXP_IGNORED_MISSING_SUFFIXES = []
    stub.Qwen4ExpDecoderLayer = nn.Module
    stub.Qwen4ExpMixtureOfExperts = type("Qwen4ExpMixtureOfExperts", (), {})
    stub._make_qwen38_decode_compile_config = Mock()
    monkeypatch.setitem(sys.modules, name, stub)
    module_name = "vllm.models.qwen4_exp.nvidia.mtp"
    parent: Any = importlib.import_module("vllm.models.qwen4_exp.nvidia")
    previous = getattr(parent, "mtp", None)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)
    if previous is None:
        delattr(parent, "mtp")
    else:
        parent.mtp = previous


class Projection(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, x):
        return x @ self.weight.T, None


class Mix(nn.Module):
    def mix(self, hidden_states):
        return hidden_states, hidden_states.view(-1, 2, 4).sum(dim=1), None


class DraftLayer(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.layer_type = "full_attention"
        self.ple = None
        self.attn_hyper_connection = Mix()
        self.self_attn = attention

    def forward(self, hidden_states, positions, **kwargs):
        _, block_input, _ = self.attn_hyper_connection.mix(hidden_states)
        out = self.self_attn(positions, None, block_input)
        return hidden_states, out, None


class FinalMix(nn.Module):
    def combine_and_mix(self, hidden, output, injection):
        return hidden, output, injection


@pytest.mark.parametrize("host_draft", [False, True])
def test_draft_cache_writes_match_full_path(
    monkeypatch, mtp_module, default_vllm_config, host_draft
):
    torch.manual_seed(16)
    monkeypatch.setattr(mtp_module, "tensor_model_parallel_all_gather", lambda x: x)
    monkeypatch.setattr(qsa.current_platform, "opaque_attention_op", lambda: False)
    select = Mock(side_effect=lambda *a, **kw: a[8])
    monkeypatch.setattr(qsa_ops, "qsa_select_paged_tokens", select)
    state = SimpleNamespace()
    monkeypatch.setattr(qsa, "get_forward_context", lambda: state.context)
    monkeypatch.setattr(indexer_qsa, "get_forward_context", lambda: state.context)
    monkeypatch.setattr(eagle, "set_forward_context", lambda *a, **kw: nullcontext())

    # The CPU fake replaces only the fused writer/rotary op. It records raw
    # rows and complete pairs in the same slots supplied to the real indexer.
    raw_slots = [5, 1, 4, 2, 0]

    def fake_pre(
        q,
        raw,
        positions,
        cos,
        qw,
        kw,
        eps,
        qout,
        raw_cache,
        slots,
        table,
        starts,
        logical,
        compressed,
        cslots,
        work,
        **kwargs,
    ):
        qout.zero_()
        for row, position in enumerate(logical.tolist()):
            raw_cache[slots[row]] = raw[row]
            if cslots[row] >= 0:
                compressed[cslots[row]] = (
                    raw_cache[raw_slots[position - 1]] + raw[row]
                ) / 2

    monkeypatch.setattr(indexer_qsa, "qsa_pre_indexer", fake_pre)

    weights = [
        torch.randn(12, 4, dtype=torch.float16),
        torch.randn(6, 4, dtype=torch.float16),
    ]
    embeddings = torch.randn(8, 4, dtype=torch.float16)
    fc1 = torch.randn(4, 4, dtype=torch.float16)
    fc2 = torch.randn(4, 4, dtype=torch.float16)

    def make_draft():
        idx = object.__new__(indexer_qsa.QSAIndexer)
        nn.Module.__init__(idx)
        idx.index_qk_proj = Projection(weights[1])
        idx.index_n_heads = 2
        idx.index_kv_heads = 1
        idx.index_head_dim = 2
        idx.compress_ratio = 2
        idx.token_topk = 2
        idx.use_fused_pre_indexer = True
        idx.skip_topk = False
        idx.shared_key_scoring = False
        idx.fused_topk = False
        idx.q_layernorm = GemmaRMSNorm(2)
        idx.k_layernorm = GemmaRMSNorm(2)
        idx.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(1))
        idx.raw_key_cache = SimpleNamespace(
            prefix="raw",
            kv_cache=torch.zeros(6, 2, dtype=torch.float16),
            rope_position_cache=None,
        )
        idx.compressed_key_cache = SimpleNamespace(
            prefix="compressed", kv_cache=torch.zeros(3, 2, dtype=torch.float16)
        )

        attention = object.__new__(qsa.Qwen4ExpQSAAttention)
        nn.Module.__init__(attention)
        attention.indexer = idx
        attention.qkv_proj = Projection(weights[0])
        attention.o_proj = Projection(torch.eye(4, dtype=torch.float16))
        attention.layer_name = "main"
        attention.num_heads = 2
        attention.num_kv_heads = 1
        attention.head_dim = 2
        attention.host_history = None
        attention._qsa_kv_scales_finalized = True
        attention.kv_cache = torch.zeros(2, 6, 2, dtype=torch.float16)
        attention.topk_indices_buffer = torch.empty(
            4, idx.output_width, dtype=torch.int32
        )

        def split(tensor, positions):
            query, key, value, gate = tensor.split([4, 2, 2, 4], dim=-1)
            return query, key, value, gate

        attention._project_qkv_gate = split
        attention.written_slots = []

        def write(layer, key, value, cache, slots):
            count = len(slots)
            layer.written_slots.append(slots.clone())
            cache[0, slots] = key[:count, 0]
            cache[1, slots] = value[:count, 0]

        def attend(layer, query, key, value, cache, metadata, output, **kwargs):
            output.zero_()

        def attend_host(layer, host, query, metadata, output, **kwargs):
            output.zero_()

        attention.impl = SimpleNamespace(
            do_kv_cache_update=write, forward_qsa=attend, forward_qsa_host=attend_host
        )
        attention.cp_kv_cache_interleave_size = 1
        attention.kv_cache_dtype = "auto"
        attention._k_scale = torch.tensor(1.0)
        attention._v_scale = torch.tensor(1.0)
        attention.indexer.layer_id = 0
        if host_draft:
            attention.host_cache = torch.zeros_like(attention.kv_cache)

            def host_write(key, value, slots, dtype, k_scale, v_scale):
                write(attention, key, value, attention.host_cache, slots)

            attention.host_history = SimpleNamespace(write=host_write)

        draft = object.__new__(mtp_module.Qwen4ExpMultiTokenPredictor)
        nn.Module.__init__(draft)
        draft.num_mtp_layers = 1
        draft.hc_count = 2
        draft.hidden_size = 4
        draft.embed_tokens = nn.Embedding.from_pretrained(embeddings)
        draft.pre_fc_norm_embedding = GemmaRMSNorm(4)
        draft.pre_fc_norm_hidden = GemmaRMSNorm(8)
        draft.fc_embedding = nn.Linear(4, 4, bias=False, dtype=torch.float16)
        draft.fc_hidden = nn.Linear(4, 4, bias=False, dtype=torch.float16)
        draft.fc_embedding.weight.data.copy_(fc1)
        draft.fc_hidden.weight.data.copy_(fc2)
        draft.layers = nn.ModuleList([DraftLayer(attention)])
        draft.hyper_connection_mixer = FinalMix()
        return draft, attention

    full, full_attn = make_draft()
    fast, fast_attn = make_draft()
    spec = object.__new__(eagle.EagleSpeculator)
    spec.model = SimpleNamespace(forward_kv_only=fast.forward_kv_only)
    spec.vllm_config = SimpleNamespace()
    inputs = torch.tensor([0, 1, 2, 3, 4, 5])
    hidden = torch.randn(6, 8, dtype=torch.float16)
    for start, count, padded in [(0, 3, 4), (3, 2, 2)]:
        pos = torch.arange(start, start + padded)
        slots = torch.tensor(raw_slots[start : start + count])
        cslots = torch.tensor(
            [-1 if i % 2 == 0 else i // 2 for i in range(start, start + count)]
        )
        meta = SimpleNamespace(num_actual_tokens=count, slot_mapping=slots)
        side = SimpleNamespace(
            num_actual_tokens=count,
            slot_mapping=slots,
            block_table=None,
            query_start_loc=None,
            logical_positions=pos[:count],
            token_to_req=None,
            seq_lens=None,
        )
        comp = SimpleNamespace(
            num_actual_tokens=count,
            slot_mapping=cslots,
            k_work_metadata=None,
            block_table=None,
            token_to_req=None,
            logical_positions=pos[:count],
            seq_lens=None,
            query_start_loc_cpu=None,
            max_seq_len=5,
            seq_lens_cpu_upper_bound=None,
            index_dcp_world_size=1,
            index_dcp_rank=0,
            prefill_ranges=None,
        )
        state.context = SimpleNamespace(
            attn_metadata={"main": meta, "raw": side, "compressed": comp},
            no_compile_layers={"main": full_attn},
            is_uniform_decode=False,
        )
        full.forward(
            inputs[start : start + padded], pos, hidden[start : start + padded]
        )
        state.context.no_compile_layers["main"] = fast_attn
        spec.input_buffers = SimpleNamespace(
            input_ids=inputs[start : start + padded], positions=pos
        )
        spec.hidden_states = hidden[start : start + padded]
        assert spec.run_kv_only_prefill(padded, state.context.attn_metadata, {}, None)
        assert torch.equal(fast_attn.written_slots[-1], slots)
        assert torch.equal(full_attn.written_slots[-1], slots)
        assert len(fast_attn.written_slots[-1]) == count
        cache = "host_cache" if host_draft else "kv_cache"
        fast_main = getattr(fast_attn, cache)
        full_main = getattr(full_attn, cache)
        for actual, expected in [
            (fast_main, full_main),
            (
                fast_attn.indexer.raw_key_cache.kv_cache,
                full_attn.indexer.raw_key_cache.kv_cache,
            ),
            (
                fast_attn.indexer.compressed_key_cache.kv_cache,
                full_attn.indexer.compressed_key_cache.kv_cache,
            ),
        ]:
            assert (actual.float() - expected.float()).abs().max().item() == 0.0
    assert select.call_count == 2  # Full path only; no index scoring on shortcut.
    assert torch.count_nonzero(fast_main[:, raw_slots]) > 0
    assert torch.count_nonzero(fast_attn.indexer.raw_key_cache.kv_cache[raw_slots]) > 0
    assert torch.count_nonzero(fast_attn.indexer.compressed_key_cache.kv_cache[:2]) > 0
    assert torch.count_nonzero(fast_main[:, 3]) == 0
    if host_draft:
        assert torch.count_nonzero(fast_attn.kv_cache) == 0
    assert torch.count_nonzero(fast_attn.indexer.compressed_key_cache.kv_cache[2]) == 0
