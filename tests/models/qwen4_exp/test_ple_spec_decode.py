# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm import envs
from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpPLEGroupedNorm,
    Qwen4ExpPLELayer,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.utils import AttentionGroup

DSD_CASES = [(7, 1, 8), (7, 2, 8), (7, 3, 8), (7, 4, 4), (7, 8, 5)]
MTP_CASES = [(4, 1, 5), (4, 2, 5), (4, 4, 4), (4, 8, 4), (4, 8, 5)]


class PleStep:
    """MRv2 metadata and the actual PLE convolution custom-op entry point."""

    def __init__(self, k, batch, q, device, state_dtype=torch.float16, hidden=10240):
        self.device, self.k, self.batch = device, k, batch
        self.rows = batch * q
        self.dtype = torch.float16
        self.layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
        nn.Module.__init__(self.layer)
        self.layer.prefix = "ple"
        self.layer.conv_state_len = 9
        self.layer.short_conv_dilation = 3
        self.layer.num_spec_tokens = k
        self.layer.conv1d = nn.Conv1d(
            hidden,
            hidden,
            4,
            dilation=3,
            groups=hidden,
            bias=False,
            dtype=self.dtype,
            device=device,
        )
        dims = (hidden, 9 + k) if is_conv_state_dim_first() else (9 + k, hidden)
        self.initial = torch.randn(
            2 * batch + 1, *dims, dtype=state_dtype, device=device
        )
        self.layer.kv_cache = (self.initial.clone(),)
        self.x = torch.randn(self.rows, hidden, dtype=self.dtype, device=device)

        spec = MambaSpec(
            block_size=16,
            shapes=(dims,),
            dtypes=(state_dtype,),
            mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
            mamba_cache_mode="align",
            num_speculative_blocks=0,
            tp_replicated=True,
        )
        cfg = NS(
            num_speculative_tokens=k,
            parallel_config=NS(decode_context_parallel_size=1),
            speculative_config=NS(num_speculative_tokens=k, parallel_drafting=False),
            compilation_config=NS(
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                max_cudagraph_capture_size=batch * (k + 1),
            ),
            scheduler_config=NS(max_num_seqs=batch),
            cache_config=NS(mamba_cache_mode="align"),
        )
        self.group = AttentionGroup(PleShortConvAttentionBackend, ["ple"], spec, 0)
        self.group.create_metadata_builders(cfg, torch.device(device))
        self.cache_cfg = KVCacheConfig(
            2 * batch + 1, [], [KVCacheGroupSpec(["ple"], spec)]
        )
        self.model_state = MambaHybridModelState.__new__(MambaHybridModelState)
        self.model_state._use_common_gdn_metadata = False
        self.model_state._use_dflash2_fused_gdn_metadata = False
        self.model_state._use_mtp4_fused_gdn_metadata = False
        self.model_state._use_dflash2_grouped_smallq_metadata = False
        self.model_state._spec_state_replay = False
        self.model_state.max_model_len = 64

    def metadata(self, lengths, accepted, *, capture=False, full=True, prefill=()):
        batch = self.batch
        active = len(lengths)
        # Reverse request order and alternate running blocks across a boundary.
        # A graph-padded request has no drafts, length, accepted prefix or slot.
        starts = np.cumsum([0, *lengths, *([0] * (batch - active))], dtype=np.int32)
        prefix = [30 + 2 * (i % 2) for i in range(active)]
        seq = torch.tensor(
            [n + length for n, length in zip(prefix, lengths)] + [0] * (batch - active),
            dtype=torch.int32,
        )
        ids = torch.arange(batch, 0, -1, dtype=torch.int32)
        block_table = torch.stack((ids, ids, ids + batch), dim=-1)
        block_table[active:].zero_()
        is_prefilling = np.array([i in prefill for i in range(active)], dtype=np.bool_)
        mode = CUDAGraphMode.FULL if full else CUDAGraphMode.NONE
        self.model_state.num_accepted_tokens_gpu = torch.tensor(
            accepted, dtype=torch.int32, device=self.device
        )
        batch_input = NS(
            num_reqs=active,
            num_reqs_after_padding=batch,
            num_tokens=int(starts[-1]),
            num_tokens_after_padding=self.rows,
            query_start_loc_np=starts if full else starts[: active + 1],
            query_start_loc=torch.from_numpy(starts).to(self.device),
            num_scheduled_tokens=np.array(lengths),
            is_prefilling_np=is_prefilling,
            num_draft_tokens_per_req=np.array(lengths) - 1,
            idx_mapping=torch.arange(active, device=self.device),
            seq_lens=seq.to(self.device),
            seq_lens_cpu_upper_bound=seq,
            dcp_local_seq_lens=None,
            is_dummy_batch=capture,
            prefix_anchor_lens=None,
        )
        return self.model_state.prepare_attn(
            batch_input,
            mode,
            (block_table.to(self.device),),
            torch.zeros((1, self.rows), dtype=torch.int64, device=self.device),
            [[self.group]],
            self.cache_cfg,
            for_capture=capture,
        )

    @contextmanager
    def context(self, metadata):
        with override_forward_context(
            ForwardContext({"ple": self.layer}, metadata, {})
        ):
            yield

    def run(self, metadata):
        output = torch.zeros_like(self.x)
        with self.context(metadata):
            torch.ops.vllm.qwen4_exp_ple_short_conv(self.x, output, "ple")
        return output

    def reset(self):
        self.layer.kv_cache[0].copy_(self.initial)

    def state(self):
        state = self.layer.kv_cache[0]
        return state if is_conv_state_dim_first() else state.transpose(1, 2)

    def graph(self, q):
        metadata = self.metadata([q] * self.batch, [q] * self.batch, capture=True)
        self.run(metadata)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.run(metadata)
        self.reset()
        return graph, output


def assert_bits(actual, expected):
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("k,batch,q", [*DSD_CASES, *MTP_CASES])
def test_mrv2_spec_capacity_and_accepted_state(k, batch, q):
    step = PleStep(k, batch, q, "cpu", torch.float32, hidden=4)
    lengths = [q] * batch
    accepted = [1 + (k if i % 2 else 2) for i in range(batch)]
    meta = step.metadata(lengths, accepted)
    assert meta["ple"].spec_query_len == k + 1
    assert meta["ple"].spec_query_start_loc.tolist() == list(range(0, step.rows + 1, q))
    assert meta["ple"].num_accepted_tokens.tolist() == accepted
    expected_state = step.state().clone()
    slots = meta["ple"].spec_state_indices_tensor.tolist()
    assert slots == [
        batch - i + (batch if 30 + 2 * (i % 2) + q > 32 else 0) for i in range(batch)
    ]
    hist = torch.stack(
        [
            torch.cat((expected_state[slot, :, n - 1 : n + 8].half(), x.T), dim=-1)
            for slot, n, x in zip(slots, accepted, step.x.split(q))
        ]
    )
    weight = step.layer.conv1d.weight
    expected = F.silu(F.conv1d(hist, weight, groups=4, dilation=3))
    expected = expected.transpose(1, 2).reshape(step.rows, 4)
    expected_state[slots, :, : 8 + q] = hist[:, :, 1:].float()
    assert_bits(step.run(meta), expected)
    assert_bits(step.state(), expected_state)


def test_mrv2_prefill_and_zero_draft_rows_keep_original_routing(monkeypatch):
    step = PleStep(7, 4, 4, "cpu", hidden=4)
    meta = step.metadata([4, 1, 3, 5], [3, 1, 1, 5], full=False, prefill=(2,))
    m = meta["ple"]
    assert (m.num_spec_decodes, m.num_decodes, m.num_prefills) == (2, 1, 1)
    original = step.layer._short_conv_dilated_spec_batched
    called = []

    def spec(*args, **kwargs):
        called.append(kwargs["allow_batched_native"])
        return original(*args, **kwargs)

    monkeypatch.setattr(step.layer, "_short_conv_dilated_spec_batched", spec)
    assert torch.isfinite(step.run(meta)[:13]).all()
    assert called == [False]


def gate_inputs(step):
    layer = step.layer
    layer.hidden_size, layer.hc_count = 2560, 4
    norms = [
        Qwen4ExpPLEGroupedNorm(10240, 1e-6, 2560, torch.float16).to(step.device)
        for _ in range(3)
    ]
    for norm in norms:
        norm.weight.copy_(torch.rand_like(norm.weight) - 0.5)
    layer.norm_key, layer.norm_query, layer.norm_conv = norms
    return (
        step.x,
        torch.randn_like(step.x),
        torch.randn(step.rows, 2560, device=step.device, dtype=torch.float16),
    )


@torch.inference_mode()
@pytest.mark.parametrize("enabled,decode", [(False, True), (True, True), (True, False)])
def test_gate_norm_compilation_selection(enabled, decode, monkeypatch):
    step = PleStep(7, 1, 4, "cpu")
    hidden, key, value = gate_inputs(step)
    step.layer.ple_embedding = lambda x, *_: x
    step.layer.key_proj = lambda _: (key, None)
    step.layer.value_proj = lambda _: (value, None)
    meta = step.metadata([4], [1], full=decode, prefill=() if decode else (0,))
    args = (hidden, torch.zeros(4, dtype=torch.long), None, None)
    seen: list[str] = []

    def backend(graph, _):
        seen.extend(str(node.target) for node in graph.graph.nodes)
        return graph.forward

    monkeypatch.setattr(envs, "VLLM_SM70_QWEN38_DUAL_COMPILE", True)
    monkeypatch.setattr(envs, "VLLM_SM70_PLE_GATE_NORM", False)
    with step.context(meta):
        expected = step.layer(*args)
    step.reset()
    monkeypatch.setattr(envs, "VLLM_SM70_PLE_GATE_NORM", enabled)
    with step.context(meta), sm70_decode_graph_compilation(decode):
        result = torch.compile(step.layer, backend=backend, fullgraph=True)(*args)
    assert any("vllm.qwen4_exp_ple_gate_norm" in op for op in seen) == (
        enabled and decode
    )
    assert_bits(result, expected)


def require_sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires an SM70 device")


@pytest.mark.parametrize("k,batch,q", [(7, 1, 8), (7, 4, 4), (7, 8, 5), (4, 8, 4)])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
@torch.inference_mode()
def test_mrv2_spec_native_graph_replay(k, batch, q, state_dtype, monkeypatch):
    require_sm70()
    torch.manual_seed(1521)
    step = PleStep(k, batch, q, "cuda", state_dtype)
    native = torch.ops._C.qwen38_ple_spec_sm70_v154_out
    calls = []

    def counted(*args):
        calls.append(1)
        native(*args)

    monkeypatch.setattr(torch.ops._C, "qwen38_ple_spec_sm70_v154_out", counted)
    monkeypatch.setattr(envs, "VLLM_SM70_MTP_PLE_CONV", False)
    ref_graph, ref = step.graph(q)
    monkeypatch.setattr(envs, "VLLM_SM70_MTP_PLE_CONV", True)
    fast_graph, fast = step.graph(q)
    assert len(calls) == 2  # warmup and capture
    states = [step.initial.clone(), step.initial.clone()]

    for cycle, lengths in enumerate(([q] * batch, [2] + [q] * (batch - 1), [q])):
        accepted = [k + 1 if (cycle + i) % 2 else 1 for i in range(len(lengths))]
        step.metadata(lengths, accepted)
        step.x.normal_()
        for graph, state in zip((ref_graph, fast_graph), states):
            step.layer.kv_cache[0].copy_(state)
            graph.replay()
            state.copy_(step.layer.kv_cache[0])
        valid = sum(lengths)
        assert_bits(fast[:valid], ref[:valid])
        assert_bits(states[1], states[0])
        assert torch.count_nonzero(fast[valid:]) == 0


@torch.inference_mode()
def test_mrv2_gate_norm_native_buffers(monkeypatch):
    require_sm70()
    step = PleStep(7, 4, 4, "cuda")
    args = gate_inputs(step)
    native = torch.ops._C.qwen38_ple_gate_norm_sm70_out
    calls = []

    def counted(*args):
        calls.append(1)
        native(*args)

    monkeypatch.setattr(torch.ops._C, "qwen38_ple_gate_norm_sm70_out", counted)
    with step.context(step.metadata([4] * 4, [1] * 4)):
        gated, normed = torch.ops.vllm.qwen4_exp_ple_gate_norm(*args, "ple")
    assert calls == [1]
    for actual, expected in zip((gated, normed), step.layer._gate_norm(*args)):
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
