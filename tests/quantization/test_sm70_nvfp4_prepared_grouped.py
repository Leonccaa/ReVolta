# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm import _custom_ops as ops
from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.config import CUDAGraphMode, DeviceConfig, VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe
from vllm.v1.worker.gpu import cudagraph_utils as graphs


def make_layer(tp):
    return NS(
        moe_config=NS(tp_size=tp, moe_parallel_config=NS(use_ep=False)),
        global_num_experts=512,
        swiglu_limit=None,
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=640 // tp,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_qwen38_raw_scale=False,
        w13_tm_weight=torch.empty(0),
        w13_tm_scales=torch.empty(0),
        w2_tm_weight=torch.empty(0),
        w2_tm_scales=torch.empty(0),
    )


@pytest.fixture
def method(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_GROUPED_DECODE", "1")
    envs.disable_envs_cache()
    monkeypatch.setattr(ops, "has_nvfp4_grouped_decode_sm70_v154", lambda: True)
    monkeypatch.setattr(sm70_ops, "has_nvfp4_grouped_decode_dispatch", lambda: True)
    monkeypatch.setattr(
        sm70_ops, "has_nvfp4_grouped_batch_reduce_dispatch", lambda: True
    )
    monkeypatch.setattr(
        torch.ops._moe_C,
        "moe_permute_sort_workspace_size",
        lambda *args: 16,
        raising=False,
    )
    return object.__new__(moe.ModelOptNvFp4SM70MoEMethod)


@pytest.mark.parametrize(
    ("tp", "m", "q", "split", "legacy"),
    [
        (1, 2, 1, 4, False),
        (2, 5, 5, 4, False),
        (1, 8, 8, 8, False),
        (2, 16, 8, 4, False),
        (1, 16, 4, 4, False),
        (2, 24, 8, 4, False),
        (1, 40, 5, 4, False),
        (4, 40, 5, 4, False),
        (4, 5, 5, 4, True),
        (4, 8, 1, 4, True),
        (4, 16, 1, 8, True),
        (4, 16, 4, 4, False),
    ],
)
def test_apply_prepared_grouped(method, monkeypatch, tp, m, q, split, legacy):
    layer = make_layer(tp)
    method._prepare_grouped_decode(layer)
    method._allocate_graph_safe_decode_buffers(layer)
    assert layer._nvfp4_grouped_rows.shape == (400, 8)
    assert layer._nvfp4_sm70_permuted_input.shape == (180, 2560)
    assert layer._nvfp4_sm70_intermediate.shape == (400, 640 // tp)
    routed_capacity = 180 if tp == 4 else 400
    assert layer._nvfp4_sm70_sorted_output.shape == (routed_capacity, 2560)
    assert layer._nvfp4_sm70_output.shape == (40, 2560)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **k: True)
    monkeypatch.setattr(method, "_get_buffers", Mock(side_effect=AssertionError))
    w13 = Mock()
    w2 = Mock()
    unused = Mock(side_effect=AssertionError)
    for module, suffix, use_batch in (
        (sm70_ops, "sm70_out", m == 5),
        (ops, "sm70_v154_out", tp == 4),
    ):
        selected = (module is sm70_ops) == legacy
        monkeypatch.setattr(
            module, f"nvfp4_grouped_w13_{suffix}", w13 if selected else unused
        )
        for batch, path in ((True, "batch_reduce_"), (False, "")):
            monkeypatch.setattr(
                module,
                f"nvfp4_grouped_w2_{path}{suffix}",
                w2 if selected and use_batch == batch else unused,
            )

    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    x = torch.empty(m, 2560, dtype=torch.float16)
    ids = torch.empty(m, 10, dtype=torch.int32)
    weights = torch.empty(m, 10, dtype=torch.float32)
    with set_forward_context({"qsa": NS(max_query_len=q)}, cfg):
        get_forward_context().additional_kwargs["sm70_nvfp4_decode_only"] = True
        out = method.apply(layer, x, weights, ids, None, None)
    a = w13.call_args.args
    b = w2.call_args.args
    assert w13.call_count == w2.call_count == 1
    assert a[-2:] == (split, False)
    assert a[0].shape == (m * 10, 640 // tp)
    assert a[0] is b[2]
    assert all(left is right for left, right in zip(a[5:9], b[6:10]))
    assert b[1].shape == (min(m * 10, routed_capacity), 2560)
    assert out is b[0] and out.shape == x.shape


@pytest.mark.parametrize("reason", ["flag", "raw", "limit", "ubatch", "shape"])
def test_loader_rejects_v154(method, monkeypatch, reason):
    layer = make_layer(1)
    if reason == "flag":
        monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_GROUPED_DECODE", "0")
    elif reason == "raw":
        layer.sm70_nvfp4_qwen38_raw_scale = True
    elif reason == "limit":
        layer.swiglu_limit = 7.0
    elif reason == "ubatch":
        monkeypatch.setattr(
            moe,
            "get_current_vllm_config_or_none",
            lambda: NS(parallel_config=NS(use_ubatching=True)),
        )
    elif reason == "shape":
        layer.sm70_nvfp4_intermediate_size = 320
    method._prepare_grouped_decode(layer)
    assert not layer.sm70_nvfp4_grouped_decode_v154
    assert not hasattr(layer, "_nvfp4_grouped_rows")


@pytest.mark.parametrize(
    "reason", ["m1", "m41", "x", "ids", "weights", "prefill", "no-context"]
)
def test_runtime_rejects_v154(method, reason):
    layer = make_layer(2)
    method._prepare_grouped_decode(layer)
    m = {"m1": 1, "m41": 41}.get(reason, 8)
    x = torch.empty(2560, m, dtype=torch.float16).T.contiguous()
    ids = torch.empty(m, 10, dtype=torch.int32)
    weights = torch.empty(m, 10, dtype=torch.float32)
    if reason == "x":
        x = torch.empty(2560, m, dtype=torch.float16).T
    elif reason == "ids":
        ids = ids.long()
    elif reason == "weights":
        weights = torch.empty(10, m, dtype=torch.float32).T
    if reason == "no-context":
        assert not moe._use_grouped_decode_v154(layer, x, ids, weights)
    else:
        cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
        with set_forward_context({}, cfg):
            if reason != "prefill":
                get_forward_context().additional_kwargs["sm70_nvfp4_decode_only"] = True
            assert not moe._use_grouped_decode_v154(layer, x, ids, weights)


@pytest.mark.parametrize(
    ("mode", "q"),
    [
        (CUDAGraphMode.FULL, 8),
        (CUDAGraphMode.FULL, None),
        (CUDAGraphMode.PIECEWISE, None),
    ],
)
def test_mrv2_capture_admission(method, monkeypatch, mode, q):
    desc = graphs.BatchExecutionDescriptor(mode, 8, None if q is None else 1, q)
    layer = make_layer(2)
    method._prepare_grouped_decode(layer)
    x = torch.empty(8, 2560, dtype=torch.float16)
    ids = torch.empty(8, 10, dtype=torch.int32)
    weights = torch.empty(8, 10, dtype=torch.float32)
    decisions = []

    def capture(self, create_forward_fn, progress_bar_desc):
        forward, _ = create_forward_fn(desc)
        forward(CUDAGraphMode.NONE)
        forward(mode)
        return {}

    monkeypatch.setattr(graphs.CudaGraphManager, "capture", capture)
    monkeypatch.setattr(graphs, "prepare_inputs_to_capture", lambda *a, **k: (None, {}))
    manager = graphs.ModelCudaGraphManager.__new__(graphs.ModelCudaGraphManager)
    manager.max_num_reqs = 8
    manager.dp_size = 1
    manager.is_first_pp_rank = manager.is_last_pp_rank = True
    manager.vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    manager.hidden_states, manager.aux_hidden_states = None, []

    def model(**inputs):
        decisions.append(moe._use_grouped_decode_v154(layer, x, ids, weights))
        return x

    manager.capture(
        model,
        NS(prepare_dummy_inputs=lambda *a: {}),
        NS(input_ids=ids[:, 0], positions=ids[:, 0]),
        None,
        None,
        [],
        None,
    )
    assert decisions == [q is not None] * 2


def test_prepared_v154_fake_ops():
    from torch._subclasses.fake_tensor import FakeTensorMode

    if not ops.has_nvfp4_grouped_decode_sm70_v154():
        pytest.skip("requires the native v154 schemas")
    with FakeTensorMode():
        x = torch.empty(40, 2560, dtype=torch.float16)
        mid = torch.empty(400, 640, dtype=torch.float16)
        routed = torch.empty(400, 2560, dtype=torch.float16)
        ids = torch.empty(400, dtype=torch.int32)
        plan = (
            torch.empty(400, 8, dtype=torch.int32),
            torch.empty_like(ids),
            torch.empty_like(ids),
            torch.empty(1, dtype=torch.int32),
        )
        ops.nvfp4_grouped_w13_sm70_v154_out(
            mid,
            x,
            torch.empty(512, 2560, 160, dtype=torch.int32),
            torch.empty(512, 160, 1280, dtype=torch.float16),
            ids,
            *plan,
            4,
            False,
        )
        for w2 in (
            ops.nvfp4_grouped_w2_sm70_v154_out,
            ops.nvfp4_grouped_w2_batch_reduce_sm70_v154_out,
        ):
            w2(
                x,
                routed,
                mid,
                torch.empty(512, 640, 320, dtype=torch.int32),
                torch.empty(512, 40, 2560, dtype=torch.float16),
                torch.empty(40, 10),
                *plan,
            )
