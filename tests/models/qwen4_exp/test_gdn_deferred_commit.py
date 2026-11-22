# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.kernels.mamba.test_mrv2_mamba_align import _make_context
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.model_executor.layers.mamba.gdn.spec_replay import TMAX, SpecReplayState
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    select_gdn_state_block_ids,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.core.kv_cache_utils import MambaStateGather
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh
from vllm.v1.worker.utils import AttentionGroup

BLOCK = 832
LAYER = "model.language_model.layers.0.linear_attn"
H, HV, D = 4, 12, 128


def _i(x):
    return torch.tensor(x, dtype=torch.int32)


class _Verifier:
    """CPU implementation of the existing native operator's state contract."""

    def __init__(self):
        self.calls: list[tuple[list[int], list[int], list[int]]] = []
        self.buffer_ptrs: set[int] = set()

    def __call__(
        self,
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        out,
        cu,
        indices,
        nacc,
        h,
        hv,
        scale,
        cfg,
        rbuf,
        rn,
        ctr,
        width,
        rmap,
        ckt,
        cks,
    ):
        assert nacc is ctr is None
        assert (h, hv, scale, cfg) == (H, HV, D**-0.5, 1)
        assert width in (3, 6)
        self.calls.append((cu.tolist(), rn.tolist(), rmap.tolist()))
        self.buffer_ptrs.add(rbuf.data_ptr())
        payload = rbuf.view(2, -1, TMAX * ((H + HV) * D + 2 * HV))
        # The native ABI assumes unit stride for one-dimensional indices.
        slots = (
            indices.as_strided(indices.shape, (1,))
            if indices.ndim == 1
            else indices[:, 0]
        )
        for row, slot in enumerate(slots.tolist()):
            start, end = cu[row : row + 2].tolist()
            take = int(rn[row])
            if slot < 0 or (start == end and take == 0):
                continue
            req, side = divmod(int(rmap[row]), 2)
            s = state[slot, 0, 0, 0].item()
            s += payload[side ^ 1, req, :take].sum().item()
            if take:
                state[slot].fill_(s)
            for j, t in enumerate(range(start, end)):
                token = qkv[t, 0].item()
                payload[side, req, j] = token
                s += token
                out[t].fill_(s)
                if ckt is not None and int(ckt[row]) == j:
                    state[cks[row]].fill_(s)


def _ordinary(**args):
    state = args["initial_state"]
    q = args["q"].squeeze(0)
    cu = args["cu_seqlens"].tolist()
    indexed = "ssm_state_indices" in args
    slots = args["ssm_state_indices"].tolist() if indexed else list(range(len(cu) - 1))
    out = q.new_empty((q.shape[0], HV, D))
    for row, slot in enumerate(slots):
        total = state[slot, 0, 0, 0].item()
        for t in range(cu[row], cu[row + 1]):
            total += q[t, 0, 0].item()
            out[t].fill_(total)
        state[slot].fill_(total)
    return out.unsqueeze(0), state


class _Runner:
    def __init__(self, monkeypatch, width, max_rows=4):
        self.layer = layer = gdn.QwenGatedDeltaNetAttention.__new__(
            gdn.QwenGatedDeltaNetAttention
        )
        torch.nn.Module.__init__(layer)
        layer.prefix = LAYER
        layer.tp_size, layer.num_k_heads, layer.num_v_heads = 4, 16, 48
        layer.key_dim, layer.value_dim = 2048, 6144
        layer.head_k_dim = layer.head_v_dim = D
        layer.conv1d = NS(weight=torch.zeros(2560, 1, 4), bias=None)
        layer.activation = "silu"
        layer.A_log = torch.zeros(HV)
        layer.dt_bias = torch.zeros(HV, dtype=torch.float16)
        layer.spec_state_replay = True
        layer._replay_slots = width
        layer._native_replay_buffer = torch.empty(
            2 * max_rows * TMAX * ((H + HV) * D + 2 * HV)
        )
        layer.enable_packed_recurrent_decode = False
        layer.enable_sm70_fused_sigmoid_mixed_qkv = False
        layer.enable_sm70_dflash2_fused_qkv_pack = False
        layer.enable_sm70_legacy_prefill_prep = True
        layer.gdn_prefill_backend = "triton"
        layer.chunk_gated_delta_rule = _ordinary
        layer._can_use_dflash2_packed_gdn_verify = lambda **_: False

        self.temporal = torch.zeros((5, HV, D, D), dtype=torch.float32)
        self.temporal[1].fill_(10)
        self.temporal[3].fill_(20)
        self.conv = torch.zeros((5, width + 2, 2560), dtype=torch.float16)
        self.conv[1, :, 0] = torch.arange(width + 2)
        self.conv[3, :, 0] = 20 + torch.arange(width + 2)
        layer.kv_cache = (self.conv, self.temporal)
        self.spec = MambaSpec(
            shapes=(self.conv.shape[1:], self.temporal.shape[1:]),
            dtypes=(torch.float16, torch.float32),
            block_size=BLOCK,
            mamba_cache_mode="align",
            num_speculative_blocks=0,
            num_replayed_speculative_blocks=width - 1,
        )
        self.block_table = _i([[1, 2], [3, 4]])
        self.context = _make_context(
            self.conv,
            self.temporal,
            self.block_table,
            block_size=BLOCK,
            max_num_reqs=4,
        )
        self.worker = m = mh.MambaHybridModelState.__new__(mh.MambaHybridModelState)
        m._align_mode = m._spec_state_replay = True
        m.device = torch.device("cpu")
        m._mamba_spec = self.spec
        m._mamba_group_ids = [0]
        m._mamba_ctx = self.context
        m._ensure_align_ctx = lambda *_: self.context
        m._mamba_state_idx_gpu = _i([-2] * 4)
        m._mamba_token_bias_gpu = _i([0] * 4)
        m._mamba_src_col_gpu = _i([-1] * 4)
        m.num_accepted_tokens_gpu = _i([1] * 4)
        self.replay = SpecReplayState(max_rows, 1, m.device)
        m._native_replay = self.replay
        m._native_layers = ((0, layer),)
        layer.bind_deferred_replay(self.replay, 0)
        self.forward_context = NS(attn_metadata={}, no_compile_layers={LAYER: layer})
        self.native = _Verifier()
        monkeypatch.setattr(gdn.ops, "sm70_gdn_verify_out", self.native)
        monkeypatch.setattr(gdn, "get_forward_context", lambda: self.forward_context)
        monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)
        monkeypatch.setattr(gdn, "causal_conv1d_update", lambda x, *a, **k: x)
        monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda x, *a, **k: x)
        monkeypatch.setattr(gdn, "fused_gdn_gating", lambda a, b, *k: (b, b))
        monkeypatch.setattr(gdn, "fused_sigmoid_gating_delta_rule_update", _ordinary)

    def prepare(self, reqs, before, widths, drafts, prefills, *, scheduler_before=None):
        self.mapping = _i(reqs)
        computed = _i([0] * 4)
        computed[self.mapping.long()] = _i(before)
        batch = NS(
            idx_mapping=self.mapping,
            num_reqs=len(reqs),
            query_start_loc=_i([0, *np.cumsum(widths)]),
            num_computed_tokens_np=np.array(
                before if scheduler_before is None else scheduler_before, dtype=np.int32
            ),
            num_scheduled_tokens=np.array(widths, dtype=np.int32),
            num_draft_tokens_per_req=np.array(drafts, dtype=np.int32),
            is_prefilling_np=np.array(prefills, dtype=np.bool_),
        )
        self.worker.preprocess_state(batch, (self.block_table,), None, computed)
        self.ends = computed
        return mamba_get_block_table_tensor(
            self.block_table[: len(reqs)],
            _i(np.asarray(before) + widths),
            self.spec,
            "align",
        )

    def sample(self, accepted):
        self.ends[self.mapping.long()] += _i(accepted)
        self.worker.postprocess_state(self.mapping, _i(accepted), self.ends)

    def forward(self, metadata, values, *, qkv=None, output=None):
        size = len(values)
        if qkv is None:
            qkv = torch.empty((size, 2560), dtype=torch.float16)
        qkv.copy_(torch.tensor(values, dtype=torch.float16)[:, None])
        if output is None:
            output = torch.empty((size, HV, D), dtype=torch.float16)
        a = torch.zeros((size, HV), dtype=torch.float16)
        self.forward_context.attn_metadata = {LAYER: metadata}
        name = gdn._encode_layer_name(LAYER)
        operands = (qkv, a, a, output, *self.layer.kv_cache)
        empty = _i([])
        if metadata.spec_sequence_masks is not None:
            gdn.qwen_gdn_attention_core_standard_spec(
                *operands,
                metadata.non_spec_query_start_loc
                if metadata.non_spec_query_start_loc is not None
                else empty,
                metadata.non_spec_state_indices_tensor
                if metadata.non_spec_state_indices_tensor is not None
                else empty,
                metadata.spec_query_start_loc,
                metadata.spec_state_indices_tensor,
                metadata.spec_token_indx
                if metadata.spec_token_indx is not None
                else empty,
                metadata.non_spec_token_indx
                if metadata.non_spec_token_indx is not None
                else empty,
                metadata.spec_sequence_masks,
                empty,
                metadata.spec_state_slot_selectors,
                name,
            )
        else:
            gdn.qwen_gdn_attention_core_standard(*operands, empty, empty, name)
        return output[:, 0, 0].tolist()


def test_normal_mtp_dispatch_has_q4_and_q6_full_descriptors(monkeypatch):
    monkeypatch.setattr(
        cg, "get_pp_group", lambda: NS(is_first_rank=True, is_last_rank=True)
    )
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: None)
    config = NS(
        additional_config={},
        cache_config=NS(mamba_cache_mode="align"),
        scheduler_config=NS(max_num_seqs=8),
        parallel_config=NS(
            tensor_parallel_size=4, data_parallel_size=1, decode_context_parallel_size=2
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[4, 8, 12, 24, 48],
            max_cudagraph_capture_size=48,
        ),
        speculative_config=NS(
            method="mtp",
            num_speculative_tokens=5,
            parallel_drafting=False,
            num_speculative_state_tokens=lambda: 5,
            num_speculative_tokens_per_batch_size=[(1, 1, 3), (2, 2, 3), (3, 8, 5)],
        ),
        async_adaptive_mtp_enabled=False,
        adaptive_mtp_enabled=False,
    )
    graphs = cg.CudaGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, decode_query_len=6
    )
    graphs._graphs_captured = True
    r = _Runner(monkeypatch, 6, max_rows=8)
    worker = r.worker
    worker._native_replay = None
    worker._native_replay_requested = True
    worker._mamba_group_ids = []
    worker._mamba_spec = None
    worker.max_num_reqs = 8
    worker.max_model_len = BLOCK
    worker.vllm_config = config
    config.compilation_config.static_forward_context.update(
        {LAYER: r.layer, "ple": torch.nn.Identity()}
    )
    ple_spec = replace(
        r.spec,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        tp_replicated=True,
        num_replayed_speculative_blocks=0,
    )
    kv = KVCacheConfig(
        5, [], [KVCacheGroupSpec(["ple"], ple_spec), KVCacheGroupSpec([LAYER], r.spec)]
    )
    group = AttentionGroup(GDNAttentionBackend, [LAYER], r.spec, 1)
    group.create_metadata_builders(config, worker.device)
    inputs = InputBuffers(8, 48, worker.device)
    tables = BlockTables.__new__(BlockTables)
    tables.cp_size, tables.cp_rank, tables.cp_interleave = 2, 0, 1
    tables.input_block_tables = [torch.zeros(8, 2, dtype=torch.int32) for _ in range(2)]
    tables.slot_mappings = torch.empty((2, 48), dtype=torch.int64)
    address = None
    for rows, width in [(1, 4), (2, 4), (8, 6)]:
        desc = graphs.dispatch(rows, rows * width, width)
        assert desc in graphs._capture_descs[CUDAGraphMode.FULL]
        assert (desc.num_reqs, desc.num_tokens, desc.uniform_token_count) == (
            rows,
            rows * width,
            width,
        )
        captured = cg.prepare_inputs_to_capture(
            rows, rows * width, worker, inputs, tables, [[], [group]], kv
        )
        metadata = captured.attn_metadata[LAYER]
        assert metadata.num_spec_decodes == rows
        cu = list(range(0, (rows + 1) * width, width))
        assert metadata.spec_query_start_loc[: rows + 1].tolist() == cu
        assert (metadata.spec_query_start_loc[rows + 1 :] == rows * width).all()
        assert (metadata.spec_state_indices_tensor[:rows, 0] == 0).all()
        assert (metadata.spec_state_indices_tensor[rows:] < 0).all()
        assert (metadata.spec_state_indices_tensor[:rows, 1:] < 0).all()
        replay = worker._native_replay
        assert worker._native_layers == ((1, r.layer),)
        assert r.layer._native_spec_replay is replay
        assert (
            replay.spec_ckpt_slot[1].data_ptr()
            == r.layer._native_checkpoint_slots.data_ptr()
        )
        if address is None:
            address = replay.spec_map.data_ptr()
        else:
            assert address == replay.spec_map.data_ptr()
        assert (
            r.forward(metadata, [1] * (rows * width))
            == list(range(1, width + 1)) * rows
        )
        assert r.native.calls[-1] == (cu, [0] * rows, list(range(0, 2 * rows, 2)))


@pytest.mark.parametrize("q", [3, 6])
@pytest.mark.parametrize("prefill", [False, True])
def test_same_static_spec_forward_then_plain_read(monkeypatch, q, prefill):
    r = _Runner(monkeypatch, q)
    cu = _i([0, q, q])
    table = torch.full((2, q), -1, dtype=torch.int32)
    md = GDNAttentionMetadata(
        0,
        0,
        0,
        0,
        2,
        2 * q,
        2 * q,
        spec_query_start_loc=cu,
        spec_state_indices_tensor=table,
        spec_sequence_masks=torch.ones(2, dtype=torch.bool),
        spec_state_slot_selectors=_i([1, 1]),
        max_query_len=q,
    )
    qkv = torch.empty((2 * q, 2560), dtype=torch.float16)
    out = torch.empty((2 * q, HV, D), dtype=torch.float16)
    v = list(range(1, q + 1))
    stable = [
        x.data_ptr() for x in (cu, table, r.replay.spec_map, r.replay.spec_replay)
    ]
    for before, initial in [(40, 10), (42, 13)]:
        actual = r.prepare([2], [before], [q], [q - 1], [False])
        table[:1].copy_(actual)
        got = r.forward(md, v + [0] * q, qkv=qkv, output=out)
        assert got == (initial + np.cumsum(v)).tolist() + [0] * q
        r.sample([2 if before == 40 else 1])
    assert [
        x.data_ptr() for x in (cu, table, r.replay.spec_map, r.replay.spec_replay)
    ] == stable
    assert r.native.calls[:2] == [
        ([0, q, q], [0, 0], [5, 0]),
        ([0, q, q], [2, 0], [4, 0]),
    ]
    assert r.temporal[1, 0, 0, 0] == 13

    count = 2 if prefill else 1
    actual = r.prepare([2], [43], [count], [0], [prefill])
    md = GDNAttentionMetadata(
        int(prefill),
        count if prefill else 0,
        int(not prefill),
        int(not prefill),
        0,
        0,
        count,
        has_initial_state=torch.tensor([True]),
        non_spec_query_start_loc=_i([0, count]),
        non_spec_state_indices_tensor=actual[:, 0],
    )
    got = r.forward(md, [8] * count)
    assert got == [22, 30][:count]
    assert r.native.calls[-1] == ([0, 0], [1], [5])
    assert r.replay.verified[2] == 0
    assert len(r.native.buffer_ptrs) == 1
    r.sample([1])


@pytest.mark.parametrize("q", [3, 6])
@pytest.mark.parametrize("accept_all", [False, True])
def test_exact_end_alias_keeps_committed_or_publishes_once(monkeypatch, q, accept_all):
    r = _Runner(monkeypatch, q)
    table = r.prepare([2], [BLOCK - q], [q], [q - 1], [False])
    md = GDNAttentionMetadata(
        0,
        0,
        0,
        0,
        1,
        q,
        q,
        spec_query_start_loc=_i([0, q]),
        spec_state_indices_tensor=table,
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        spec_state_slot_selectors=_i([1]),
        max_query_len=q,
    )
    assert r.replay.spec_ckpt_token[0] == -1
    assert r.forward(md, [1] * q) == list(range(11, 11 + q))
    assert r.temporal[1, 0, 0, 0] == 10
    accepted = q if accept_all else 1
    r.sample([accepted])
    assert r.temporal[1, 0, 0, 0] == (10 + q if accept_all else 10)
    assert r.replay.verified[2].item() == int(not accept_all)
    # Align continues to move the conv history even when temporal is published
    # by the deferred path.
    assert r.conv[1, 0, 0] == (q - 1 if accept_all else 0)
    actual = r.prepare([2], [BLOCK - q + accepted], [1], [0], [False])
    ordinary = GDNAttentionMetadata(
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        non_spec_query_start_loc=_i([0, 1]),
        non_spec_state_indices_tensor=actual[:, 0],
    )
    assert r.forward(ordinary, [8]) == [10 + accepted + 8]
    assert r.native.calls[-1][1] == ([0] if accept_all else [1])


@pytest.mark.parametrize("q", [3, 6])
@pytest.mark.parametrize("is_exact", [False, True])
def test_async_checkpoint_uses_live_endpoint_after_unsettled_drafts(
    monkeypatch, q, is_exact
):
    r = _Runner(monkeypatch, q)
    # One outstanding proposal at depth two can lag by its entire draft width.
    r.worker._max_unsettled_draft_tokens = q - 1
    before = BLOCK - q - int(not is_exact)
    table = r.prepare(
        [2],
        [before],
        [q],
        [q - 1],
        [False],
        scheduler_before=[before + q - 1],
    )
    maps = r.replay._rows.data_ptr()
    assert table[0, 0] == 1 and r.replay.spec_ckpt_token[0] == -1
    plan = r.worker._end_checkpoints
    assert plan is not None and plan[1].tolist() == ([q] if is_exact else [0])
    assert plan[2][0].tolist() == [1]
    md = GDNAttentionMetadata(
        0,
        0,
        0,
        0,
        1,
        q,
        q,
        spec_query_start_loc=_i([0, q]),
        spec_state_indices_tensor=table,
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        spec_state_slot_selectors=_i([1]),
        max_query_len=q,
    )
    assert r.forward(md, [1] * q) == list(range(11, 11 + q))
    r.sample([q])
    assert r.temporal[1, 0, 0, 0].item() == (10 + q if is_exact else 10)
    assert r.replay.verified[2].item() == int(not is_exact)

    table = r.prepare(
        [2],
        [before + q],
        [1],
        [0],
        [False],
        scheduler_before=[before + q + q - 1],
    )
    md = GDNAttentionMetadata(
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        non_spec_query_start_loc=_i([0, 1]),
        non_spec_state_indices_tensor=table[:, 0],
    )
    assert r.forward(md, [2]) == [10 + q + 2]
    assert r.native.calls[-1][1] == ([0] if is_exact else [q])
    assert r.replay._rows.data_ptr() == maps


def test_group_checkpoint_rows_follow_requests_and_reset_reused_slots():
    replay = SpecReplayState(4, 2, torch.device("cpu"))
    tables = (_i([[11, 12], [13, 14]]), _i([[21, 22], [23, 24]]))
    pointers = torch.tensor([t.data_ptr() for t in tables], dtype=torch.int64)
    replay.prepare(
        _i([2, 0]),
        _i([0, 3, 9]),
        _i([826, 0, 830, 0]),
        np.array([2, 5]),
        np.array([False, False]),
        2,
        pointers,
        2,
        BLOCK,
    )
    assert replay.spec_ckpt_token.tolist() == [1, -1, -1, -1]
    assert replay.spec_ckpt_slot.tolist() == [[11, -1, -1, -1], [21, -1, -1, -1]]
    replay.record_accepted(_i([2, 0]), _i([4, 1, 2, 1]))
    replay.prepare(
        _i([0, 2]),
        _i([0, 1, 4]),
        _i([830, 0, 831, 0]),
        np.array([0, 2]),
        np.array([False, False]),
        2,
        pointers,
        2,
        BLOCK,
    )
    assert replay.spec_replay.tolist() == [2, 0, 0, 0]
    assert replay.spec_map[0] == 4 and replay.plain_replay[0] == 4
    replay.reset_request(2)
    replay.prepare(_i([2]), _i([0, 1]), _i([0] * 4), None, None, 1, pointers, 2, BLOCK)
    assert replay.plain_replay.tolist() == [0] * 4


def test_ragged_then_mixed_spec_and_prefill_on_different_runtime_rows(monkeypatch):
    r = _Runner(monkeypatch, 6)
    table = r.prepare([2, 0], [40, 60], [3, 6], [2, 5], [False, False])
    first = GDNAttentionMetadata(
        0,
        0,
        0,
        0,
        2,
        9,
        9,
        spec_query_start_loc=_i([0, 3, 9]),
        spec_state_indices_tensor=table,
        spec_sequence_masks=torch.ones(2, dtype=torch.bool),
        spec_state_slot_selectors=_i([1, 1]),
        max_query_len=6,
    )
    assert r.forward(first, [1] * 3 + [2] * 6) == [
        11,
        12,
        13,
        22,
        24,
        26,
        28,
        30,
        32,
    ]
    r.sample([1, 3])
    # Request zero changes to plain prefill; the next verification of request
    # two moves to batch row one. Its payload stays with the original slot.
    r.block_table[:] = r.block_table.flip(0)
    table = r.prepare([0, 2], [63, 41], [2, 3], [0, 2], [True, False])
    assert r.conv[3, 0, 0] == 22
    mixed = GDNAttentionMetadata(
        1,
        2,
        0,
        0,
        1,
        3,
        5,
        has_initial_state=torch.tensor([True]),
        non_spec_query_start_loc=_i([0, 2]),
        non_spec_state_indices_tensor=table[:1, 0],
        spec_query_start_loc=_i([0, 3]),
        spec_state_indices_tensor=table[1:],
        spec_sequence_masks=torch.tensor([False, True]),
        spec_token_indx=torch.tensor([2, 3, 4]),
        non_spec_token_indx=torch.tensor([0, 1]),
        spec_state_slot_selectors=_i([1]),
        max_query_len=3,
    )
    assert r.forward(mixed, [8] * 2 + [1] * 3) == [34, 42, 12, 13, 14]
    assert r.native.calls[-2:] == [([0, 0], [3], [0]), ([0, 3], [1], [4])]


@pytest.mark.parametrize("accepted", [1, 3])
def test_terminal_checkpoint_consumes_before_worker_removes_slot(monkeypatch, accepted):
    q = 6
    r = _Runner(monkeypatch, q)
    table = r.prepare([2], [BLOCK - 4], [q], [q - 1], [False])
    assert table[0, 0] == 2
    md = GDNAttentionMetadata(
        0,
        0,
        0,
        0,
        1,
        q,
        q,
        spec_query_start_loc=_i([0, q]),
        spec_state_indices_tensor=table,
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        spec_state_slot_selectors=_i([1]),
        max_query_len=q,
    )
    assert r.forward(md, [1] * q) == list(range(11, 11 + q))
    r.sample([accepted])
    assert r.temporal[2, 0, 0, 0] == 10
    assert r.conv[2, 0, 0] == 0
    req_slots = {"finished": 2}

    def remove(req_id):
        assert r.temporal[2, 0, 0, 0] == 10 + accepted
        assert r.conv[2, 0, 0] == accepted - 1
        del req_slots[req_id]

    runner = NS(
        model_state=r.worker,
        req_states=NS(req_id_to_index=req_slots),
        _remove_request=remove,
    )
    GPUModelRunner.finish_requests(
        runner,
        NS(
            mamba_state_gathers=[
                MambaStateGather(1, [1]),
                MambaStateGather(accepted - 1, [2], "finished"),
            ],
            finished_req_ids={"finished"},
            preempted_req_ids=set(),
            total_num_scheduled_tokens=0,
        ),
    )
    assert not req_slots
    assert r.native.calls[-1] == ([0, 0], [accepted], [4])


def test_empty_scheduler_output_has_no_state_gathers():
    output = SchedulerOutput.make_empty()
    empty = np.empty(0, dtype=np.int32)
    runner = NS(
        req_states=NS(
            num_computed_tokens_np=empty,
            prefill_len=NS(np=empty),
            num_computed_prefill_tokens=empty,
        ),
    )
    GPUModelRunner.finish_requests(runner, output)
    GPUModelRunner.update_requests(runner, output)


def test_plain_replay_commits_every_strided_state_slot(monkeypatch):
    r = _Runner(monkeypatch, 6)
    table = mamba_get_block_table_tensor(r.block_table, _i([42, 80]), r.spec, "align")
    indices = select_gdn_state_block_ids(table, None, num_spec=5, state_columns=1)
    payload = r.layer._native_replay_buffer.view(2, 4, -1)
    payload[0, :2, 0] = torch.tensor([2.0, 3.0])
    r.replay.plain_replay[:2] = 1
    r.replay.plain_map[:2] = _i([1, 3])

    r.layer._flush_plain_deferred_replay(r.temporal, indices)

    assert r.temporal[[1, 3], 0, 0, 0].tolist() == [12.0, 23.0]
