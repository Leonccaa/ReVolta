# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm import distributed
from vllm.config import CUDAGraphMode, KernelConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.models.qwen4_exp.nvidia import qsa
from vllm.models.qwen4_exp.nvidia.host_kv import history
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_dcp, qsa_paged_fp16_h12
from vllm.v1.attention.kv_cache_view import KVCacheView, bind_planar_kv_cache
from vllm.v1.attention.kv_codecs import FP16, INT8
from vllm.v1.worker import workspace
from vllm.v1.worker.gpu import cudagraph_utils

pytestmark = pytest.mark.skip_global_cleanup


def _forward_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_forward_context={},
            max_cudagraph_capture_size=0,
        ),
    )


def _planar_cache():
    # Padded pages with a second layer interleaved between successive pages.
    page_bytes = 832 * 520 + 16
    raw = torch.empty(3 * 2 * page_bytes, dtype=torch.int8)
    carrier = bind_planar_kv_cache(
        raw,
        num_blocks=3,
        scheduler_block_size=832,
        kernel_block_size=832,
        num_heads=1,
        head_size=256,
        page_size_bytes=page_bytes,
        member_index=1,
        member_count=2,
    )
    return raw, carrier, KVCacheView.bind(carrier, INT8, 256)


# Host main K/V admits only the INT8 instance (see the admission test).
@pytest.mark.parametrize(
    "dtype,host", [("float16", False), ("int8", False), ("int8", True)]
)
@pytest.mark.parametrize(
    "enabled,phase,rows,topk,native",
    [
        (True, True, 48, 2048, True),
        (False, True, 48, 2048, False),
        (True, False, 48, 2048, False),
        (True, True, 49, 2048, False),
        (True, True, 48, 8192, False),
    ],
)
def test_qsa_h12_dcp_reader(
    monkeypatch, dtype, host, enabled, phase, rows, topk, native
):
    layer = SimpleNamespace(
        sm70_qsa_paged_h12=enabled,
        qsa_dcp_sharded=True,
        sm70_qsa_device_history=False,
        dcp_local_indices_buffer=torch.empty(rows, topk + 3, dtype=torch.int32),
        dcp_partial_output_buffer=torch.empty(rows, 12, 256),
        dcp_partial_lse_buffer=torch.empty(rows, 12),
        cp_kv_cache_interleave_size=1,
        indexer=SimpleNamespace(token_topk=topk, compress_ratio=4),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )
    group = SimpleNamespace(
        all_gather=lambda q, dim: torch.cat((q, q), dim=dim),
        device_communicator=SimpleNamespace(sm70_dcp2=None),
    )
    monkeypatch.setattr(distributed, "get_dcp_group", lambda: group)
    monkeypatch.setattr(
        qsa_dcp, "qsa_localize_dcp_indices", lambda src, dst, **kwargs: dst.copy_(src)
    )
    monkeypatch.setattr(
        qsa_ops,
        "qsa_dcp_sparse_paged_attention_sm70_grouped_page4",
        lambda *args, **kwargs: 0,
    )
    layer.topk_indices_buffer = torch.zeros_like(layer.dcp_local_indices_buffer)
    query = torch.zeros(rows, 6, 256, dtype=torch.float16)
    output = torch.empty_like(query)
    if dtype == "int8":
        raw, cache, view = _planar_cache()
    else:
        cache = torch.empty(3, 2, 416, 1, 256, dtype=torch.float16)
        view = KVCacheView.bind(cache, FP16, 256)
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    requests = torch.zeros(rows, dtype=torch.int32)

    def reader(q, k, v, ids, block_table, token_to_req, out, lse, *, paged, **scales):
        assert native is paged
        assert q.shape == (rows, 12, 256)
        assert k.data_ptr() == view.key.data_ptr()
        assert k.stride() == view.key.stride()
        assert (
            v.data_ptr() == view.value.data_ptr() and v.stride() == view.value.stride()
        )
        assert block_table is table and token_to_req.shape == (rows,)
        assert ids.shape == (rows, topk // 2 + 2)
        assert ids.stride(0) == topk + 3
        assert out.dtype == lse.dtype == torch.float32
        if dtype == "int8":
            planes = (k, v, scales["k_scale_cache"], scales["v_scale_cache"])
            page_bytes = 832 * 520 + 16
            for plane, offset, token_bytes in zip(
                planes, (0, 832 * 256, 832 * 512, 832 * 516), (256, 256, 4, 4)
            ):
                assert plane.data_ptr() - raw.data_ptr() == page_bytes + offset
                assert plane.stride(0) * plane.element_size() == 2 * page_bytes
                assert plane.stride(1) * plane.element_size() == token_bytes
        else:
            assert not scales
        out.fill_(3)
        lse.fill_(-2)

    def combine(out, lse, dcp_group, *, is_lse_base_on_e):
        assert dcp_group is group and not is_lse_base_on_e
        assert torch.all(out == 3) and torch.all(lse == -2)
        return out[:, :6]

    def native_reader(q, k, v, ids, block_table, token_to_req, out, lse, **scales):
        reader(q, k, v, ids, block_table, token_to_req, out, lse, paged=True, **scales)

    def triton_reader(
        q,
        k,
        v,
        ids,
        block_table,
        token_to_req,
        out,
        *,
        lse,
        kv_cache_dtype,
        k_scale,
        v_scale,
        compacted_indices=False,
        **scales,
    ):
        # The host arm's Triton call is unchanged: hot IDs are not compacted.
        assert kv_cache_dtype == dtype and compacted_indices is not host
        reader(q, k, v, ids, block_table, token_to_req, out, lse, paged=False, **scales)

    monkeypatch.setattr(qsa, "qsa_paged_h12", native_reader)
    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", triton_reader)
    monkeypatch.setattr(
        qsa_ops, "_qsa_output_gate", lambda out, gate, *, merged: out.copy_(merged)
    )
    impl = object.__new__(qsa.Qwen4ExpQSAFlashAttentionImpl)
    impl.dcp_world_size, impl.dcp_rank = 2, 0
    impl.kv_cache_dtype, impl.dcp_combine = dtype, combine
    impl.head_size, impl.sliding_window = 256, (-1, -1)
    impl.alibi_slopes = impl.sinks = None
    with set_forward_context(None, _forward_config(), is_uniform_decode=phase):
        if not host:
            impl.forward_qsa(
                layer,
                query,
                query,
                query,
                cache,
                SimpleNamespace(num_actual_tokens=rows, block_table=table),
                output,
                token_to_req=requests,
                output_gate=torch.ones_like(query),
            )
        else:
            # The planar view stands in for the hot pool and the localized
            # selection for its remapped IDs (one remap pass, same columns).
            def one_pass(store, ids, reqs, blocks, lengths, attend, *, single_pass):
                assert single_pass
                attend(0, ids.shape[0], ids)

            monkeypatch.setattr(history, "for_each_pass", one_pass)
            shape = SimpleNamespace(
                decode_rows=1, prefill_gather="union", planar_scales=True
            )
            hot = SimpleNamespace(
                store=SimpleNamespace(shape=shape, hot_block_table=table),
                key_cache=view.key,
                value_cache=view.value,
                hot_cache=view,
                kernel_block_tokens=832,
            )
            impl.forward_qsa_host(
                layer,
                hot,
                query,
                SimpleNamespace(
                    num_actual_tokens=rows, block_table=None, max_query_len=1
                ),
                output,
                requests,
                torch.ones(1, dtype=torch.int32),
                output_gate=torch.ones_like(query),
            )
    assert torch.all(output == 3)


def _h12_config(placement, *, int8, fp16, v2=True, draft=False, dcp=2):
    return SimpleNamespace(
        kernel_config=SimpleNamespace(
            sm70_qsa_paged_int8_h12=int8,
            sm70_qsa_paged_fp16_h12=fp16,
            long_context_kv=SimpleNamespace(
                enabled=placement is not None, main_kv_placement=placement or "host"
            ),
        ),
        use_v2_model_runner=v2,
        is_speculative_draft=draft,
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        model_config=SimpleNamespace(dtype=torch.float16),
    )


@pytest.mark.parametrize("placement", [None, "gpu", "host"])
@pytest.mark.parametrize("dtype", ["int8", "float16", "auto", "fp8_e4m3"])
def test_qsa_h12_admission(monkeypatch, placement, dtype):
    monkeypatch.setattr(
        qsa, "current_platform", SimpleNamespace(is_device_capability=(70).__eq__)
    )

    def admitted(dtype=dtype, sharded=True, tp=4, geometry=(6, 1, 256), **kw):
        flags = {"int8": dtype == "int8", "fp16": dtype != "int8", **kw}
        config = _h12_config(placement, **flags)
        return qsa._sm70_qsa_paged_h12_admitted(config, dtype, sharded, tp, geometry)

    # GPU main K/V keeps K2's admission; host main K/V admits INT8 only.
    fp16 = dtype in ("float16", "auto")
    assert admitted() == (dtype == "int8" or (fp16 and placement != "host"))
    # Each instance has its own switch, default off.
    assert not admitted(int8=False, fp16=False)
    assert not admitted(int8=dtype != "int8", fp16=dtype == "int8")
    assert not any(
        admitted(**kw) for kw in ({"v2": False}, {"draft": True}, {"dcp": 1})
    )
    assert not admitted(sharded=False)
    assert not admitted(tp=2)
    assert not admitted(geometry=(12, 1, 256))
    monkeypatch.setattr(
        qsa, "current_platform", SimpleNamespace(is_device_capability=(75).__eq__)
    )
    assert not admitted()


def h12_late_scale(q, k, v, sk, sv, valid):
    """Python model of the INT8 H12 partials and their FP32 merge.

    ``k``/``v`` hold the integer codes of each selected column
    ``[rows, width, D]``, ``sk``/``sv`` their scales and ``valid`` the
    selected columns ``[rows, width]``. Returns FP32 attention and base-2 LSE.
    """
    width, splits = k.shape[1], -(-k.shape[1] // 64)

    # Mirror the 64-token partials and their FP32 numerator/max/sum merge.
    def tiles(x, fill):
        padded = torch.nn.functional.pad(x, (0, splits * 64 - width), value=fill)
        return padded.unflatten(-1, (splits, 64))

    logits = ((q.float() @ k.transpose(1, 2)) * sk[:, None, :]) * (
        math.log2(math.e) / math.sqrt(q.shape[-1])
    )
    scores = tiles(logits.masked_fill(~valid[:, None], -torch.inf), -torch.inf)
    split_max = scores.amax(-1)
    safe_max = torch.where(split_max.isfinite(), split_max, 0)
    probabilities = torch.exp2(scores - safe_max[..., None])
    numerators = torch.einsum(
        "mhst,mstd->mhsd",
        probabilities * tiles(sv, 0)[:, None],
        tiles(v.transpose(1, 2), 0).permute(0, 2, 3, 1),
    )
    maximum = split_max.amax(-1, keepdim=True)
    weights = torch.exp2(split_max - torch.where(maximum.isfinite(), maximum, 0))
    denominator = (probabilities.sum(-1) * weights).sum(-1)
    out = (numerators * weights[..., None]).sum(-2) / denominator.clamp_min(1)[
        ..., None
    ]
    return out, maximum.squeeze(-1) + denominator.log2()


def test_int8_h12_late_scale_split_merge():
    _, _, view = _planar_cache()
    rng = torch.Generator().manual_seed(603)
    view.key.random_(-127, 128, generator=rng)
    view.value.random_(-127, 128, generator=rng)
    view.key_scale.uniform_(0.002, 0.02, generator=rng)
    view.value_scale.uniform_(0.001, 0.01, generator=rng)
    q = torch.randn((3, 12, 256), generator=rng).half()
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    requests = torch.zeros(3, dtype=torch.int32)
    indices = torch.full((3, 2051), -1, dtype=torch.int32)[:, :1026]
    indices[0] = torch.arange(767, 1793)
    indices[1, :3] = torch.tensor([831, 832, 1664])
    positions = indices.clamp_min(0).long()
    pages, offsets = table[requests[:, None], positions // 832].long(), positions % 832
    k, v, sk, sv = (
        plane[pages, offsets, 0].float()
        for plane in (view.key, view.value, view.key_scale, view.value_scale)
    )
    valid = indices[:, None, :] >= 0
    dense_logits = (q.float() @ (k * sk[..., None]).transpose(1, 2)) / 16
    dense_logits = dense_logits.masked_fill(~valid, -torch.inf)
    dense_lse = torch.logsumexp(dense_logits, dim=-1) / math.log(2)
    dense_probs = torch.where(valid, dense_logits.softmax(-1), 0)
    dense_out = dense_probs @ (v * sv[..., None])
    h12_out, h12_lse = h12_late_scale(q, k, v, sk, sv, valid[:, 0])

    # Isolate the Triton PV rounding from the different split reduction order.
    weighted = dense_probs * sv[:, None]
    high = weighted.half().float()
    low = ((weighted - high) * 2048).half().float() / 2048
    triton_pv = high @ v + low @ v
    torch.testing.assert_close(h12_out, dense_out, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(h12_lse, dense_lse, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(triton_pv, dense_out, rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(h12_out[2]) == 0
    assert torch.isneginf(h12_lse[2]).all()


def test_qsa_h12_workspace_survives_capture_lock(monkeypatch):
    assert not KernelConfig().sm70_qsa_paged_fp16_h12
    default = KernelConfig()
    assert not default.sm70_qsa_paged_int8_h12
    assert (
        default.compute_hash()
        != KernelConfig(sm70_qsa_paged_int8_h12=True).compute_hash()
    )
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    layer = SimpleNamespace(
        qsa_dcp_sharded=True,
        dcp_partial_output_buffer=torch.empty(128, 12, 256),
        dcp_local_indices_buffer=torch.empty(128, 2051, dtype=torch.int32),
        sm70_qsa_paged_h12=True,
        sm70_qsa_device_history=False,
        indexer=SimpleNamespace(token_topk=2048, compress_ratio=4),
        cp_kv_cache_interleave_size=1,
        impl=SimpleNamespace(dcp_world_size=2),
    )
    qsa.Qwen4ExpQSAAttention.reserve_capture_workspace(layer)
    manager.lock()
    partial, state = qsa_paged_fp16_h12.qsa_paged_h12_workspace(48, 1026)
    assert partial.shape == (48, 12, 17, 256)
    assert state.shape == (48, 12, 17, 2)
    smaller, _ = qsa_paged_fp16_h12.qsa_paged_h12_workspace(6, 1026)
    assert smaller.data_ptr() == partial.data_ptr()


@pytest.mark.parametrize(
    "mode,uniform_token_count,expected",
    [
        (CUDAGraphMode.FULL, 6, True),
        (CUDAGraphMode.FULL, 1, True),
        (CUDAGraphMode.FULL, None, False),
        (CUDAGraphMode.PIECEWISE, None, False),
    ],
)
def test_qsa_h12_capture_phase(monkeypatch, mode, uniform_token_count, expected):
    graphs = object.__new__(cudagraph_utils.ModelCudaGraphManager)
    graphs.dp_size = 1
    graphs.is_first_pp_rank = graphs.is_last_pp_rank = True
    graphs.hidden_states = None
    graphs.vllm_config = _forward_config()
    desc = cudagraph_utils.BatchExecutionDescriptor(mode, 48, 8, uniform_token_count)

    def capture(self, create_forward_fn, progress_bar_desc):
        forward, _ = create_forward_fn(desc)
        forward(CUDAGraphMode.NONE)
        forward(mode)
        return {}

    seen = []
    verify = []

    def model(**kwargs):
        context = get_forward_context()
        seen.append(context.is_uniform_decode)
        verify.append(context.additional_kwargs.get("sm70_gdn_verify", False))
        return torch.empty(48, 1)

    monkeypatch.setattr(cudagraph_utils.CudaGraphManager, "capture", capture)
    monkeypatch.setattr(
        cudagraph_utils, "prepare_inputs_to_capture", lambda *a, **kw: (None, {})
    )
    graphs.capture(
        model,
        SimpleNamespace(prepare_dummy_inputs=lambda *a: {}),
        SimpleNamespace(input_ids=torch.zeros(48), positions=torch.arange(48)),
        None,
        None,
        [],
        None,
    )
    assert seen == [expected, expected]
    assert verify == [mode == CUDAGraphMode.FULL and uniform_token_count == 6] * 2
