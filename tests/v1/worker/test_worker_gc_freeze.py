# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GPU worker freezes its startup heap after warmup and unfreezes it on
shutdown (the EngineCore process already does the same).

The worker module needs a GPU Triton driver to import, so the wiring is
checked on its source and the freeze itself on the shared helpers.
"""

import ast
import gc
from pathlib import Path

import pytest

import vllm
from vllm.utils.gc_utils import freeze_gc_heap

WORKER = Path(vllm.__file__).parent / "v1" / "worker" / "gpu_worker.py"


def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse(WORKER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Worker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"Worker.{name} not found")


def _calls(func: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute):
                names.append((node.lineno, target.attr))
            elif isinstance(target, ast.Name):
                names.append((node.lineno, target.id))
    return [name for _, name in sorted(names)]


@pytest.fixture(autouse=True)
def _restore_gc():
    gc.unfreeze()
    yield
    gc.unfreeze()


def test_warmup_freezes_after_last_warmup_step():
    calls = _calls(_method("compile_or_warm_up_model"))
    assert "_freeze_gc_after_warmup" in calls
    freeze = calls.index("_freeze_gc_after_warmup")
    assert calls.index("activate_triton_jit_monitor") < freeze
    # Nothing but building the return value comes after the freeze.
    assert set(calls[freeze + 1 :]) <= {"CompilationTimes"}


def test_freeze_helper_uses_shared_freeze_and_gc_debug():
    calls = _calls(_method("_freeze_gc_after_warmup"))
    assert calls == ["freeze_gc_heap", "maybe_attach_gc_debug_callback"]


def test_shutdown_unfreezes_first():
    calls = _calls(_method("shutdown"))
    assert calls[0] == "unfreeze"


def test_freeze_moves_startup_heap_out_of_collection():
    static = [object() for _ in range(1000)]
    assert gc.get_freeze_count() == 0
    freeze_gc_heap()
    assert gc.get_freeze_count() >= len(static)
    gc.unfreeze()
    assert gc.get_freeze_count() == 0
