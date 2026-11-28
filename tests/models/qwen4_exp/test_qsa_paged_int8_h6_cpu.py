# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.models.qwen4_exp.nvidia import qsa
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.v1.worker import workspace

from .test_qsa_paged_fp16_h12_cpu import (
    _forward_config,
    _h12_config,
    _planar_cache,
    h12_late_scale,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_int8_h6_admission(monkeypatch):
    monkeypatch.setattr(
        qsa, "current_platform", SimpleNamespace(is_device_capability=(70).__eq__)
    )

    def admitted(dtype="int8", sharded=False, geometry=(6, 1, 256), **kwargs):
        config = _h12_config(None, fp16=True, **{"int8": True, "draft": True, **kwargs})
        return qsa._sm70_qsa_paged_int8_h6_admitted(config, dtype, sharded, geometry)

    assert admitted()
    assert not admitted(int8=False)
    assert not admitted(draft=False)
    assert not admitted(sharded=True)
    assert not admitted(dtype="float16")
    assert not admitted(dtype="fp8_e4m3")
    assert not admitted(geometry=(12, 1, 256))
    monkeypatch.setattr(
        qsa, "current_platform", SimpleNamespace(is_device_capability=(75).__eq__)
    )
    assert not admitted()


@pytest.mark.parametrize(
    "enabled,phase,marker,rows,width,native",
    [
        (True, True, True, 64, 67, True),
        (True, True, True, 1, 1, True),
        (True, True, True, 1, 4096, True),
        (True, True, True, 65, 67, False),
        (True, True, True, 1, 4097, False),
        (False, True, True, 64, 67, False),
        (True, False, True, 64, 67, False),
        (True, True, False, 64, 67, False),
    ],
)
def test_int8_h6_dispatch(monkeypatch, enabled, phase, marker, rows, width, native):
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    _, carrier, view = _planar_cache()
    planes = (view.key, view.value, view.key_scale, view.value_scale)
    rng = torch.Generator().manual_seed(603)
    for plane in planes[:2]:
        plane.copy_(torch.randint(-127, 128, plane.shape, generator=rng))
    for plane in planes[2:]:
        plane.copy_(torch.rand(plane.shape, generator=rng) * 0.025 + 0.001)
    # Use a strided selection and an out-of-order page table.
    selection = torch.full((rows, width + 3), -1, dtype=torch.int32)[:, :width]
    table = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.int32)
    requests = torch.arange(rows, dtype=torch.int32) % 2
    positions = torch.full((rows,), 840, dtype=torch.int64)
    lengths = torch.tensor([853, 833], dtype=torch.int32)
    if width == 67:
        selection[0, :64] = torch.arange(811, 875)
        selection[0, 64:] = torch.tensor([11, 77, -1])
        selection[1, :2] = torch.tensor([11, 833])
        # Row 0 has future IDs. Row 1 has an ID at its sequence length
        # and an empty second split. All later rows are entirely empty.
    layer = SimpleNamespace(
        topk_indices_buffer=selection,
        qsa_dcp_sharded=False,
        qsa_draft_native_scales=None,
        sm70_qsa_paged_int8_h6=enabled,
        indexer=SimpleNamespace(output_width=width),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )
    qsa.Qwen4ExpQSAAttention.reserve_capture_workspace(layer)
    manager.lock()
    impl = object.__new__(qsa.Qwen4ExpQSAFlashAttentionImpl)
    impl.head_size, impl.kv_cache_dtype = 256, "int8"
    impl.alibi_slopes, impl.sinks, impl.sliding_window = None, None, (-1, -1)
    query = (torch.randn(rows, 6, 256, generator=rng) * 0.125).half()
    gate = torch.randn(rows, 6 * 256, generator=rng).half()
    output = torch.empty_like(query)
    calls = []

    def fallback(*args, **kwargs):
        calls.append("fallback")

    def h6(q, k, v, sk, sv, ids, blocks, reqs, pos, lens, out, g, partial, state):
        calls.append("h6")
        for actual, expected in zip((k, v, sk, sv), planes):
            assert actual.data_ptr() == expected.data_ptr()
            assert actual.stride() == expected.stride()
        assert (ids.data_ptr(), ids.stride(0)) == (selection.data_ptr(), width + 3)
        assert blocks is table and lens is lengths
        assert reqs.data_ptr() == requests.data_ptr()
        assert pos.data_ptr() == positions.data_ptr()
        assert g.shape == query.shape and g.data_ptr() == gate.data_ptr()
        assert partial.shape == (rows * 6 * math.ceil(width / 64), 256)
        assert state.shape == (partial.shape[0], 2)
        if width == 67:
            logical = ids.long().clamp_min(0)
            page = blocks[reqs.long()[:, None], logical // k.size(1)].long()
            offset = logical % k.size(1)
            codes_k, codes_v, scales_k, scales_v = (
                plane[page, offset, 0].float() for plane in (k, v, sk, sv)
            )
            valid = (ids >= 0) & (ids <= pos[:, None]) & (ids < lens[reqs.long(), None])
            result, lse = h12_late_scale(q, codes_k, codes_v, scales_k, scales_v, valid)
            assert lse[:2].isfinite().all() and lse[2:].isneginf().all()
            out.copy_(result.half().float() * g.float().sigmoid())

    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", fallback)
    monkeypatch.setattr(
        torch.ops.vllm_sm70_qsa_indexer, "paged_int8_h6", h6, raising=False
    )
    with (
        set_forward_context(None, _forward_config()),
        sm70_decode_graph_compilation(phase),
    ):
        get_forward_context().additional_kwargs["sm70_qsa_draft_decode"] = marker
        impl.forward_qsa(
            layer,
            query,
            None,
            None,
            carrier,
            SimpleNamespace(num_actual_tokens=rows, block_table=table),
            output,
            requests,
            output_gate=gate,
            query_positions=positions,
            sequence_lengths=lengths,
        )
    assert calls == (["h6"] if native else ["fallback"])
    if native and width == 67:
        for row, tokens in [(0, [*range(811, 841), 11, 77]), (1, [11])]:
            logical = torch.tensor(tokens)
            page = table[requests[row], logical // view.key.size(1)].long()
            offset = logical % view.key.size(1)
            k, v, sk, sv = (p[page, offset, 0].float() for p in planes)
            weights = (query[row].float() @ (k * sk[:, None]).T / 16).softmax(-1)
            attention = weights @ (v * sv[:, None])
            expected = (
                attention.half().float()
                * gate[row].view_as(attention).float().sigmoid()
            ).half()
            torch.testing.assert_close(output[row], expected, rtol=0, atol=0.001)
        assert not output[2:].count_nonzero()
