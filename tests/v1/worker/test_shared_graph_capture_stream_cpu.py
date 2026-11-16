# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for sharing one capture stream across serial graph managers.

The target runner and the Eagle/MTP draft managers capture one after another,
so they can share one capture stream (and its library workspaces) while keeping
one graph pool. These tests check the wiring only; the workspace saving
and bitwise replay need a GPU.
"""

import ast
import inspect
import textwrap
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import parallel_state as ps
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu import model_runner as mr
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle

_GLOBAL_POOL = object()  # stands in for the platform's global graph pool


class _FakeStream:
    created = 0

    def __init__(self, device=None):
        type(self).created += 1
        self.device = device


@pytest.fixture
def fake_streams(monkeypatch):
    _FakeStream.created = 0
    monkeypatch.setattr(cg.torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(ps.torch.cuda, "Stream", _FakeStream)
    return _FakeStream


def test_graph_capture_reuses_given_context(fake_streams, monkeypatch):
    seen = []

    class _Group:
        @contextmanager
        def graph_capture(self, context):
            seen.append(context)
            yield context

    monkeypatch.setattr(ps, "get_tp_group", lambda: _Group())
    monkeypatch.setattr(ps, "get_pp_group", lambda: _Group())
    context = ps.GraphCaptureContext(_FakeStream())
    created = fake_streams.created
    with ps.graph_capture("cpu", context=context) as got:
        assert got is context
    assert seen == [context, context]
    assert fake_streams.created == created

    with ps.graph_capture("cpu") as got:
        assert got is not context
    assert fake_streams.created == created + 1


def test_manager_capture_context_is_created_once(fake_streams):
    manager = SimpleNamespace(capture_context=None, device="cpu")
    first = cg.CudaGraphManager.get_capture_context(manager)
    second = cg.CudaGraphManager.get_capture_context(manager)
    assert first is second
    assert fake_streams.created == 1


def _make_speculator(monkeypatch, mode):
    class _Manager:
        def __init__(self, config, device, cudagraph_mode, *args, **kwargs):
            self.pool = _GLOBAL_POOL
            self.cudagraph_mode = cudagraph_mode
            self.capture_context = None
            self.device = device

        get_capture_context = cg.CudaGraphManager.get_capture_context

    monkeypatch.setattr(eagle, "PrefillEagleCudaGraphManager", _Manager)
    monkeypatch.setattr(eagle, "DecodeEagleCudaGraphManager", _Manager)
    speculator = eagle.EagleSpeculator.__new__(eagle.EagleSpeculator)
    speculator.device = "cpu"
    speculator.num_speculative_steps = 4
    speculator.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode)
    )
    return speculator


@pytest.mark.parametrize("external", [False, True])
def test_eagle_managers_share_one_context(fake_streams, monkeypatch, external):
    speculator = _make_speculator(monkeypatch, CUDAGraphMode.FULL_AND_PIECEWISE)
    context = ps.GraphCaptureContext(_FakeStream()) if external else None
    created = fake_streams.created
    speculator.init_cudagraph_manager(
        CUDAGraphMode.FULL_AND_PIECEWISE, capture_context=context
    )
    prefill = speculator.prefill_cudagraph_manager
    decode = speculator.decode_cudagraph_manager
    assert prefill.capture_context is decode.capture_context
    assert prefill.capture_context is not None
    if external:
        assert prefill.capture_context is context
        assert fake_streams.created == created
    else:
        assert fake_streams.created == created + 1
    # Draft prefill, draft decode and the target all use the global pool.
    assert decode.pool is prefill.pool is _GLOBAL_POOL


def test_eagle_without_graphs_leaves_context_unset(fake_streams, monkeypatch):
    speculator = _make_speculator(monkeypatch, CUDAGraphMode.NONE)
    speculator.init_cudagraph_manager(CUDAGraphMode.NONE, capture_context=None)
    assert speculator.prefill_cudagraph_manager.capture_context is None
    assert speculator.decode_cudagraph_manager.capture_context is None
    assert fake_streams.created == 0


def _source_tree(obj) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(obj)))


def test_full_capture_uses_the_context_stream():
    tree = _source_tree(cg.CudaGraphManager.capture)
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.cuda.graph"
    ]
    assert graph_calls
    for call in graph_calls:
        streams = [kw for kw in call.keywords if kw.arg == "stream"]
        assert streams, "torch.cuda.graph must capture on the shared stream"
        assert "get_capture_context()" in ast.unparse(streams[0].value)
    capture_ctx = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "graph_capture"
    ]
    assert capture_ctx and all(
        any(kw.arg == "context" for kw in c.keywords) for c in capture_ctx
    )


def test_runner_passes_target_context_to_eagle():
    source = inspect.getsource(mr)
    assert "capture_context=(" in source
    assert "self.cudagraph_manager.get_capture_context()" in source


def test_eagle_graph_managers_use_the_global_pool():
    """The Eagle/MTP managers no longer create or assign a dedicated pool; they
    inherit CudaGraphManager's global graph pool, shared with the target."""
    from vllm.v1.worker.gpu.spec_decode.eagle import cudagraph as eagle_cg

    for cls in (
        eagle_cg.PrefillEagleCudaGraphManager,
        eagle_cg.DecodeEagleCudaGraphManager,
    ):
        assert cls.__mro__[1] is cg.CudaGraphManager
        assert "__init__" not in vars(cls)
    for module in (eagle_cg, eagle):
        tree = _source_tree(module)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "graph_pool_handle", module.__name__
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assert not (
                        isinstance(target, ast.Attribute) and target.attr == "pool"
                    ), module.__name__
    init = _source_tree(cg.CudaGraphManager.__init__)
    assert "get_global_graph_pool" in ast.dump(init)
