# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for reclaiming target-loading temporaries before the draft load.

The amount reclaimed depends on the checkpoint and needs a GPU start-up to
measure; these tests check the order of operations and the call site.
"""

import ast
import inspect
import textwrap

from vllm.v1.worker.gpu import model_runner as mr


def test_reclaim_collects_then_empties_cache(monkeypatch):
    calls: list[str] = []
    allocated = iter([300, 100])

    def memory_allocated(*args, **kwargs) -> int:
        calls.append("allocated")
        return next(allocated)

    def memory_reserved(*args, **kwargs) -> int:
        calls.append("reserved")
        return 512

    monkeypatch.setattr(mr.torch.accelerator, "memory_allocated", memory_allocated)
    monkeypatch.setattr(mr.torch.accelerator, "memory_reserved", memory_reserved)
    monkeypatch.setattr(
        mr.torch.accelerator,
        "empty_cache",
        lambda *a, **k: calls.append("empty_cache"),
    )
    monkeypatch.setattr(mr.gc, "collect", lambda *a, **k: calls.append("collect"))

    assert mr._reclaim_before_draft_load() == (300, 100)
    assert calls == ["allocated", "collect", "empty_cache", "allocated", "reserved"]


def _load_model_tree() -> ast.AST:
    source = inspect.getsource(mr.GPUModelRunner.load_model)
    return ast.parse(textwrap.dedent(source))


def test_reclaim_runs_right_before_the_draft_load():
    tree = _load_model_tree()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "self.speculator is not None":
            continue
        calls = [
            ast.unparse(stmt.value.func)
            for stmt in node.body
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        ]
        if "self.speculator.load_model" not in calls:
            continue
        index = calls.index("self.speculator.load_model")
        assert index > 0 and calls[index - 1] == "_reclaim_before_draft_load"
        return
    raise AssertionError("speculator load block not found in load_model")


def test_reclaim_is_inside_the_load_profiler():
    # The reclaim must run inside DeviceMemoryProfiler so the measured model
    # memory reflects the freed temporaries.
    tree = _load_model_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and "DeviceMemoryProfiler" in ast.unparse(
            node.items[0].context_expr
        ):
            assert "_reclaim_before_draft_load()" in ast.unparse(node)
            return
    raise AssertionError("DeviceMemoryProfiler block not found in load_model")
