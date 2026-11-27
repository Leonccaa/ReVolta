# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for MTP prefill routing, cache writes, and sampled rows."""

import importlib
import math
import os
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
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.nvidia import indexer_qsa, qsa
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.triton_utils import tl
from vllm.utils import torch_utils
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
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
        "path",
    ),
    [
        (
            False,
            [True, True],
            0,
            False,
            False,
            False,
            False,
            CUDAGraphMode.NONE,
            "full",
        ),
        (
            None,
            [True, True],
            0,
            False,
            False,
            False,
            False,
            CUDAGraphMode.NONE,
            "kv",
        ),
        (True, [True, True], 0, False, False, False, False, CUDAGraphMode.NONE, "kv"),
        (
            None,
            [True, False],
            0,
            False,
            False,
            False,
            False,
            CUDAGraphMode.NONE,
            "last",
        ),
        (True, [True, True], 1, False, False, False, False, CUDAGraphMode.NONE, "full"),
        (True, [True, True], 0, True, False, False, False, CUDAGraphMode.NONE, "full"),
        (True, [True, True], 0, False, True, False, False, CUDAGraphMode.NONE, "full"),
        (True, [True, True], 0, False, False, True, False, CUDAGraphMode.NONE, "full"),
        (True, [True, True], 0, False, False, False, True, CUDAGraphMode.NONE, "full"),
        (
            True,
            [True, True],
            0,
            False,
            False,
            False,
            False,
            CUDAGraphMode.FULL,
            "graph",
        ),
        (False, [False], 0, False, False, False, False, CUDAGraphMode.NONE, "full"),
        (True, [False], 0, False, False, False, False, CUDAGraphMode.PIECEWISE, "last"),
        (True, [True], 0, False, False, False, False, CUDAGraphMode.PIECEWISE, "full"),
        (True, [False], 0, False, False, False, False, CUDAGraphMode.FULL, "graph"),
    ],
)
def test_propose_selects_prefill_path(
    monkeypatch,
    enabled,
    incomplete,
    drafts,
    mm,
    dummy,
    profile,
    batch_dummy,
    mode,
    path,
):
    if enabled is None:
        monkeypatch.delenv("VLLM_SM70_MTP_KV_ONLY_PREFILL", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_MTP_KV_ONLY_PREFILL", str(int(enabled)))
    monkeypatch.setattr(eagle, "prepare_eagle_inputs", Mock())
    num_reqs = len(incomplete)
    num_tokens = 2 * num_reqs
    descriptor = SimpleNamespace(cg_mode=mode, num_tokens=num_tokens)
    monkeypatch.setattr(
        eagle, "dispatch_cg_and_sync_dp", Mock(return_value=(descriptor, None))
    )
    spec = object.__new__(eagle.EagleSpeculator)
    spec.method = "mtp"
    spec.model = SimpleNamespace(supports_last_row_prefill=True)
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
    spec.last_token_indices = torch.tensor([1, 3])
    spec.current_draft_step = torch.tensor(0)
    spec.prefill_cudagraph_manager = SimpleNamespace(run_fullgraph=Mock())
    spec.run_kv_only_prefill = Mock(return_value=True)
    spec.prefill = Mock()
    spec._mtp_prefill_begin = Mock()
    spec._mtp_prefill_end = Mock()
    batch = SimpleNamespace(
        num_tokens_after_padding=num_tokens,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        num_scheduled_tokens=np.full(num_reqs, 2),
        idx_mapping=torch.arange(num_reqs),
        is_prefilling_np=np.ones(num_reqs, dtype=bool),
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
        torch.zeros(num_tokens, 4),
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
    assert spec.run_kv_only_prefill.called == (path == "kv")
    assert spec.prefill.called == (path in ("last", "full"))
    if spec.prefill.called:
        actual = spec.prefill.call_args.kwargs["row_indices"]
        if path == "last":
            torch.testing.assert_close(actual, spec.last_token_indices[:num_reqs])
        else:
            assert actual is None
    assert spec.prefill_cudagraph_manager.run_fullgraph.called == (path == "graph")
    assert spec._mtp_prefill_end.called == (path in ("full", "graph"))


@pytest.fixture
def mtp_module(monkeypatch):
    # Stub the unused vision imports when running the text decoder without
    # torchvision. The predictor and decoder layer below are the real classes.
    stubs = {
        "qwen3_5": {
            "Qwen3_5ForConditionalGeneration": type(
                "Qwen3_5ForConditionalGeneration",
                (nn.Module,),
                {"packed_modules_mapping": {}},
            ),
        },
        "qwen3_vl": {
            "Qwen3_VisionTransformer": nn.Module,
            "Qwen3VLDummyInputsBuilder": object,
            "Qwen3VLMultiModalProcessor": object,
            "Qwen3VLProcessingInfo": object,
        },
    }
    for name, attrs in stubs.items():
        module_name = f"vllm.model_executor.models.{name}"
        stub = types.ModuleType(module_name)
        vars(stub).update(attrs)
        monkeypatch.setitem(sys.modules, module_name, stub)

    package_name = "vllm.models.qwen4_exp.nvidia"
    package = importlib.import_module(package_name)
    imported = []
    try:
        for name in ("model", "mtp"):
            module_name = f"{package_name}.{name}"
            if module_name not in sys.modules:
                imported.append(name)
            module = importlib.import_module(module_name)
        yield module
    finally:
        for name in reversed(imported):
            sys.modules.pop(f"{package_name}.{name}", None)
            if hasattr(package, name):
                delattr(package, name)


class Projection(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, x):
        return x @ self.weight.T, None


class Mix(nn.Module):
    def mix(self, hidden_states):
        streams = hidden_states.view(-1, 2, 4)
        return hidden_states, streams.sum(dim=1), streams[:, :, :1]

    def combine_and_mix(self, hidden_states, output, injection):
        hidden_states = hidden_states.view(-1, 2, 4)
        hidden_states = hidden_states + output.unsqueeze(1) * injection
        return self.mix(hidden_states.flatten(1))


def select_cpu(q, cache, table, requests, positions, lengths, topk, ratio, out, **kw):
    # This geometry selects one complete group and keeps the causal tail.
    assert topk == ratio
    out.fill_(-1)
    for row, (req, pos) in enumerate(zip(requests.tolist(), positions.tolist())):
        end = min(pos + 1, int(lengths[req]))
        groups = end // ratio
        selected: list[int] = []
        if groups:
            ids = torch.arange(groups)
            keys = cache[table[req, ids // 2], ids % 2, 0].float()
            scores = (q[row].float() @ keys.T).relu().sum(0)
            best = int(scores.argmax())
            selected.extend(range(best * ratio, (best + 1) * ratio))
        selected.extend(range(groups * ratio, end))
        out[row, : len(selected)] = torch.tensor(selected)
    return out


def sparse_cpu(query, key, value, selection, table, requests, out, output_gate, **kw):
    for row, req in enumerate(requests.tolist()):
        tokens = selection[row][selection[row] >= 0].long()
        page_size = key.shape[1]
        pages = table[req, tokens // page_size].long()
        k = key[pages, tokens % page_size, 0].float()
        v = value[pages, tokens % page_size, 0].float()
        scores = (query[row].float() @ k.T) / math.sqrt(query.shape[-1])
        out[row] = (scores.softmax(-1) @ v) * output_gate[row].view_as(
            query[row]
        ).sigmoid()
    return out


@pytest.mark.skipif(
    os.getenv("TRITON_INTERPRET") != "1", reason="Uses the CPU Triton interpreter"
)
@pytest.mark.parametrize("host_draft", [False, True], ids=["gpu", "host"])
@torch.inference_mode()
def test_last_rows_match_full_draft(
    monkeypatch, mtp_module, default_vllm_config, host_draft
):
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    torch.manual_seed(16)
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    masked_load = InterpreterBuilder.create_masked_load
    masked_store = InterpreterBuilder.create_masked_store

    # Triton's interpreter can represent boolean masks in integer arrays.
    def bool_mask(mask):
        return TensorHandle(mask.data.astype(np.bool_, copy=False), tl.int1)

    def load(self, ptrs, mask, other, cache_policy, is_volatile):
        return masked_load(
            self, ptrs, bool_mask(mask), other, cache_policy, is_volatile
        )

    def store(self, ptrs, value, mask, cache_policy):
        return masked_store(self, ptrs, value, bool_mask(mask), cache_policy)

    monkeypatch.setattr(InterpreterBuilder, "create_masked_load", load)
    monkeypatch.setattr(InterpreterBuilder, "create_masked_store", store)
    monkeypatch.setattr(mtp_module, "tensor_model_parallel_all_gather", lambda x: x)
    monkeypatch.setattr(qsa.current_platform, "opaque_attention_op", lambda: False)
    select = Mock(side_effect=select_cpu)
    monkeypatch.setattr(qsa_ops, "qsa_select_paged_tokens", select)
    sparse = Mock(side_effect=sparse_cpu)
    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", sparse)
    state: Any = SimpleNamespace()
    monkeypatch.setattr(qsa, "get_forward_context", lambda: state.context)
    monkeypatch.setattr(indexer_qsa, "get_forward_context", lambda: state.context)
    monkeypatch.setattr(eagle, "get_forward_context", lambda: state.context)
    context = Mock(return_value=nullcontext())
    monkeypatch.setattr(eagle, "set_forward_context", context)

    # A DCP2 target page owns 2 local slots. Each draft rank owns all 4.
    page_table = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=6),
    )
    monkeypatch.setattr(qsa, "build_qsa_metadata", qsa_cache._build_qsa_metadata_torch)
    monkeypatch.setattr(
        FlashAttentionMetadataBuilder, "__init__", lambda self, *args: None
    )

    def build_base(self, common_prefix_len, common, fast_build):
        return FlashAttentionMetadata(
            num_actual_tokens=common.num_actual_tokens,
            max_query_len=common.max_query_len,
            max_seq_len=common.max_seq_len,
            seq_lens=common.seq_lens,
            query_start_loc=common.query_start_loc,
            block_table=common.block_table_tensor,
            slot_mapping=common.slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )

    monkeypatch.setattr(FlashAttentionMetadataBuilder, "build", build_base)
    builder = qsa.Qwen4ExpQSAMetadataBuilder(
        None, ["mtp.attn"], config, torch.device("cpu")
    )
    builder.vllm_config = config
    builder.layer_names = ["mtp.attn"]
    builder.block_size = 4
    assert builder.replicated_draft

    qkv_weight = torch.randn(12, 4, dtype=torch.float16)
    index_weight = torch.randn(12, 4, dtype=torch.float16)
    o_weight = torch.randn(4, 4, dtype=torch.float16)
    embeddings = torch.randn(8, 4, dtype=torch.float16)
    fc1 = torch.randn(4, 4, dtype=torch.float16)
    fc2 = torch.randn(4, 4, dtype=torch.float16)
    dtype = torch.float16

    def make_draft():
        idx = object.__new__(indexer_qsa.QSAIndexer)
        nn.Module.__init__(idx)
        idx.index_qk_proj = Projection(index_weight)
        idx.index_n_heads = 2
        idx.index_kv_heads = 1
        idx.index_head_dim = 4
        idx.compress_ratio = 2
        idx.token_topk = 2
        idx.use_fused_pre_indexer = True
        idx.skip_topk = False
        idx.shared_key_scoring = False
        idx.fused_topk = False
        idx.q_layernorm = GemmaRMSNorm(4)
        idx.k_layernorm = GemmaRMSNorm(4)
        idx.rotary_emb = RotaryEmbedding(4, 2, 8, 10000, True, dtype)
        idx.raw_key_cache = SimpleNamespace(
            prefix="raw",
            kv_cache=torch.zeros(4, 4, 1, 4, dtype=dtype),
            rope_position_cache=None,
        )
        idx.compressed_key_cache = SimpleNamespace(
            prefix="compressed", kv_cache=torch.zeros(4, 2, 1, 4, dtype=dtype)
        )

        attention = object.__new__(qsa.Qwen4ExpQSAAttention)
        nn.Module.__init__(attention)
        attention.indexer = idx
        attention.qkv_proj = Projection(qkv_weight)
        attention.o_proj = Projection(o_weight)
        attention.layer_name = "mtp.attn"
        attention.num_heads = 2
        attention.num_kv_heads = 1
        attention.head_dim = 2
        attention.q_size, attention.kv_size = 4, 2
        attention.q_norm = GemmaRMSNorm(2)
        attention.k_norm = GemmaRMSNorm(2)
        attention.rotary_emb = RotaryEmbedding(
            2, 2, 8, 10000, True, dtype
        ).forward_native
        attention.host_history = None
        attention._qsa_kv_scales_finalized = True
        attention.qsa_dcp_sharded = False
        attention.qsa_draft_native_scales = None
        attention.kv_cache = torch.zeros(4, 2, 4, 1, 2, dtype=dtype)
        attention.topk_indices_buffer = torch.full(
            (6, idx.output_width), -1, dtype=torch.int32
        )

        def write(layer, key, value, cache, slots):
            pages, offsets = slots // 4, slots % 4
            cache[pages, 0, offsets] = key[: slots.numel()]
            cache[pages, 1, offsets] = value[: slots.numel()]

        impl = object.__new__(qsa.Qwen4ExpQSAFlashAttentionImpl)
        impl.do_kv_cache_update = write
        impl.kv_cache_dtype = "auto"
        impl.head_size = 2
        impl.dcp_world_size = 2
        impl.sliding_window, impl.alibi_slopes, impl.sinks = (-1, -1), None, None
        attention.impl = impl
        attention.kv_cache_dtype = "auto"
        attention._k_scale = torch.tensor(1.0)
        attention._v_scale = torch.tensor(1.0)
        attention._k_scale_float = attention._v_scale_float = 1.0
        attention.indexer.layer_id = 0
        if host_draft:
            shape = HostHistoryShape(
                num_blocks=4,
                kernel_blocks_per_block=1,
                kernel_block_tokens=4,
                num_layers=1,
                kv_heads=1,
                head_dim=2,
                dtype=dtype,
                hot_slots=24,
                max_requests=2,
                selection_width=idx.output_width,
                max_write_rows=6,
                row_groups=2,
                tail_groups=2,
                prefill_gather="passes",
            )
            store = HostHistoryStore(shape, ["mtp.attn"], device="cpu")
            store.bind_pages([(1, 1), (2, 2), (3, 3)])
            attention.host_history = store.layer("mtp.attn")

        layer_cls = mtp_module.Qwen4ExpDecoderLayer
        layer = object.__new__(layer_cls)
        nn.Module.__init__(layer)
        layer.layer_type = "full_attention"
        layer.ple = None
        layer.attn_hyper_connection = Mix()
        layer.self_attn = attention
        layer.mlp_hyper_connection = Mix()
        layer.mlp = nn.Linear(4, 4, bias=False, dtype=dtype)
        layer.mlp.weight.data.copy_(o_weight)

        predictor = object.__new__(mtp_module.Qwen4ExpMultiTokenPredictor)
        nn.Module.__init__(predictor)
        predictor.do_not_compile = True
        predictor.num_mtp_layers = 1
        predictor.hc_count = 2
        predictor.hidden_size = 4
        predictor.embed_tokens = nn.Embedding.from_pretrained(embeddings)
        predictor.pre_fc_norm_embedding = GemmaRMSNorm(4)
        predictor.pre_fc_norm_hidden = GemmaRMSNorm(8)
        predictor.fc_embedding = nn.Linear(4, 4, bias=False, dtype=dtype)
        predictor.fc_hidden = nn.Linear(4, 4, bias=False, dtype=dtype)
        predictor.fc_embedding.weight.data.copy_(fc1)
        predictor.fc_hidden.weight.data.copy_(fc2)
        predictor.layers = nn.ModuleList([layer])
        predictor.hyper_connection_mixer = Mix()
        draft = object.__new__(mtp_module.Qwen4ExpMTP)
        nn.Module.__init__(draft)
        draft.do_not_compile = True
        draft.model = predictor
        return draft, attention

    full, full_attn = make_draft()
    fast, fast_attn = make_draft()
    assert fast.supports_last_row_prefill
    spec = object.__new__(eagle.EagleSpeculator)
    spec.model = fast
    spec.vllm_config = SimpleNamespace()
    spec.idx_mapping = torch.tensor([0, 1])
    spec.current_draft_step = torch.tensor(0)
    spec.draft_logits = None
    spec.draft_tokens = torch.zeros(2, 2, dtype=torch.long)
    spec._sample_draft = Mock(side_effect=lambda hidden, *args: hidden.argmax(-1))

    def side_metadata(common, ratio, work=None):
        count = common.num_actual_tokens
        requests, pos, slots = qsa_cache._build_qsa_metadata_torch(
            common,
            torch.empty(count, dtype=torch.int32),
            torch.empty(count, dtype=torch.long),
            torch.empty(count, dtype=torch.long),
            storage_block_size=4 // ratio,
            compress_ratio=ratio,
            circular_buffer_size=4 if ratio == 1 else 0,
            k_work_metadata_buffer=work,
            ignore_common_slot_mask=True,
        )
        return qsa_cache.QSAForwardMetadata(
            block_table=common.block_table_tensor,
            slot_mapping=slots,
            seq_lens=common.seq_lens,
            query_start_loc=common.query_start_loc,
            token_to_req=requests,
            logical_positions=pos,
            k_work_metadata=work,
            num_actual_tokens=count,
            storage_block_size=4 // ratio,
            compress_ratio=ratio,
            query_start_loc_cpu=common.query_start_loc_cpu,
            max_seq_len=common.max_seq_len,
        )

    # Request 0 is split 3+2. Request 1 arrives as a single three-token prompt.
    for starts, lengths, padded in [([0, 3], [3], 4), ([0, 2, 5], [5, 3], 6)]:
        num_reqs, count = len(lengths), starts[-1]
        query_start = torch.tensor(starts, dtype=torch.int32)
        table = page_table[:num_reqs]
        common = CommonAttentionMetadata(
            num_reqs=num_reqs,
            num_actual_tokens=count,
            max_query_len=3,
            max_seq_len=max(lengths),
            query_start_loc=query_start,
            query_start_loc_cpu=query_start,
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            block_table_tensor=table,
            slot_mapping=torch.tensor([66, -1, 67, 70, -1][:count]),
        )
        main = builder.build(0, common)

        raw = side_metadata(common, 1)
        compressed = side_metadata(common, 2, torch.empty(3, 2, dtype=torch.int32))
        metadata = {"mtp.attn": main, "raw": raw, "compressed": compressed}
        state.context = SimpleNamespace(
            attn_metadata=metadata,
            no_compile_layers={"mtp.attn": full_attn},
            is_uniform_decode=False,
        )
        pos = torch.nn.functional.pad(raw.logical_positions, (0, padded - count))
        ids = torch.arange(padded)
        hidden = torch.randn(padded, 8, dtype=dtype)
        expected_sample, expected_feedback = full.model.forward(ids, pos, hidden)

        state.context.no_compile_layers["mtp.attn"] = fast_attn
        spec.input_buffers = SimpleNamespace(input_ids=ids, positions=pos.clone())
        spec.hidden_states = hidden.clone()
        if num_reqs == 1:
            assert spec.run_kv_only_prefill(padded, metadata, {}, None)
        else:
            rows = torch.tensor([1, 4])
            spec.last_token_indices = rows
            spec.prefill(
                num_reqs,
                padded,
                metadata,
                {},
                None,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                row_indices=rows,
            )
            assert (
                context.call_args.kwargs["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
            )
            sample, _, positions, _, _ = spec._sample_draft.call_args.args
            torch.testing.assert_close(
                sample, expected_sample[rows], rtol=1e-3, atol=2e-3
            )
            torch.testing.assert_close(
                spec.hidden_states[:num_reqs],
                expected_feedback[rows],
                rtol=1e-3,
                atol=2e-3,
            )
            torch.testing.assert_close(positions, pos[rows])
            torch.testing.assert_close(
                fast_attn.topk_indices_buffer[:num_reqs],
                full_attn.topk_indices_buffer[rows],
            )
            assert (
                select.call_args.args[0].shape[0]
                == sparse.call_args.args[0].shape[0]
                == num_reqs
            )

        def main_cache(attn):
            if host_draft:
                return attn.host_history.host_view[:, 0]
            return attn.kv_cache

        for actual, expected in [
            (main_cache(fast_attn)[1:], main_cache(full_attn)[1:]),
            (
                fast_attn.indexer.raw_key_cache.kv_cache,
                full_attn.indexer.raw_key_cache.kv_cache,
            ),
            (
                fast_attn.indexer.compressed_key_cache.kv_cache,
                full_attn.indexer.compressed_key_cache.kv_cache,
            ),
        ]:
            assert torch.count_nonzero(actual) > 0
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert select.call_count == 3  # Both full passes and the compact final pass.
    if host_draft:
        assert torch.count_nonzero(fast_attn.kv_cache) == 0

    # Compact the full path just as the speculator does before draft step 1.
    full.model.compact_topk_indices(rows)
    expected_selection = full_attn.topk_indices_buffer[:num_reqs].clone()
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        max_seq_len=6,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([6, 4], dtype=torch.int32),
        block_table_tensor=page_table,
        slot_mapping=torch.zeros(num_reqs, dtype=torch.long),
    )
    metadata = {
        "mtp.attn": builder.build(0, common),
        "raw": side_metadata(common, 1),
        "compressed": side_metadata(common, 2, torch.empty(2, 2, dtype=torch.int32)),
    }
    reader_name = "forward_qsa_host" if host_draft else "forward_qsa"
    reader = getattr(qsa.Qwen4ExpQSAFlashAttentionImpl, reader_name)
    selections = []

    def record_selection(self, layer, *args, **kwargs):
        selections.append(layer.topk_indices_buffer[:num_reqs].clone())
        return reader(self, layer, *args, **kwargs)

    monkeypatch.setattr(
        qsa.Qwen4ExpQSAFlashAttentionImpl, reader_name, record_selection
    )
    first_tokens = spec.draft_tokens[:, 0].clone()
    second_tokens = []
    for model, attention, feedback in (
        (full, full_attn, expected_feedback[rows]),
        (fast, fast_attn, spec.hidden_states[:num_reqs].clone()),
    ):
        spec.model = model
        spec.share_mtp_topk_indices = True
        spec.supports_mm_inputs = False
        spec.num_speculative_steps = 2
        spec.max_model_len = 8
        spec.hidden_states = feedback.clone()
        spec.input_buffers = SimpleNamespace(
            input_ids=torch.zeros(num_reqs, dtype=torch.long),
            positions=pos[rows].clone(),
            query_start_loc=torch.zeros(num_reqs + 1, dtype=torch.int32),
            seq_lens=torch.zeros(num_reqs, dtype=torch.int32),
        )
        eagle.prepare_eagle_decode(
            first_tokens,
            torch.tensor([5, 3]),
            torch.zeros(num_reqs, dtype=torch.int32),
            spec.input_buffers,
            spec.max_model_len,
            num_reqs,
        )
        state.context = SimpleNamespace(
            attn_metadata=metadata,
            no_compile_layers={"mtp.attn": attention},
            is_uniform_decode=False,
            additional_kwargs={},
        )
        spec.current_draft_step.fill_(1)
        spec._mtp_decode_begin()
        spec.generate_draft(num_reqs, num_reqs, metadata, {}, None)
        spec._mtp_decode_end()
        second_tokens.append(spec.draft_tokens[:, 1].clone())

    assert select.call_count == 3  # Neither decode step selects again.
    torch.testing.assert_close(selections[0], expected_selection)
    torch.testing.assert_close(selections[1], expected_selection)
    torch.testing.assert_close(second_tokens[1], second_tokens[0])
