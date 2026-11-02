# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the in-place resample count update (upstream vLLM #64990).

``_insert_resampled_kernel`` loads ``num_sampled`` and then stores
``num_sampled + 1`` to the same address without a barrier. With more than one
warp, a later warp can load the incremented value, branch differently from
warp 0, and leave the cross-warp argmax with unwritten partials. The launch
must use one warp.

These tests read the source and compile SM70 PTX offline. They do not run the
kernel; the race itself needs a GPU regression.
"""

import ast
from pathlib import Path

import pytest
import regex as re

import vllm

SOURCE = (
    Path(vllm.__file__).parent / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
)


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text())


def _kernels_updating_pointer_in_place(tree: ast.Module) -> dict[str, set[str]]:
    """Map each Triton kernel to the pointer args it both loads and stores."""
    result: dict[str, set[str]] = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        if not any(ast.unparse(d) == "triton.jit" for d in fn.decorator_list):
            continue
        loaded: set[str] = set()
        stored: set[str] = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = ast.unparse(node.func)
            if name not in ("tl.load", "tl.store"):
                continue
            ptrs = {
                n.id
                for n in ast.walk(node.args[0])
                if isinstance(n, ast.Name) and n.id.endswith("_ptr")
            }
            (loaded if name == "tl.load" else stored).update(ptrs)
        if loaded & stored:
            result[fn.name] = loaded & stored
    return result


def _launches(tree: ast.Module, kernel: str) -> list[ast.Call]:
    """Calls of the form ``kernel[grid](...)``."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and ast.unparse(node.func.value) == kernel
    ]


def _num_warps(call: ast.Call):
    for kw in call.keywords:
        if kw.arg == "num_warps":
            return ast.literal_eval(kw.value)
    return None


def test_insert_resampled_updates_count_in_place():
    in_place = _kernels_updating_pointer_in_place(_tree())
    assert in_place.get("_insert_resampled_kernel") == {"num_sampled_ptr"}


def test_insert_resampled_launch_uses_one_warp():
    launches = _launches(_tree(), "_insert_resampled_kernel")
    assert len(launches) == 1
    assert _num_warps(launches[0]) == 1


def test_every_in_place_kernel_launches_with_one_warp():
    tree = _tree()
    in_place = _kernels_updating_pointer_in_place(tree)
    assert in_place, "expected at least one in-place kernel"
    for kernel in in_place:
        launches = _launches(tree, kernel)
        assert launches, f"{kernel} has no launch in this module"
        for call in launches:
            assert _num_warps(call) == 1, (
                f"{kernel} loads and stores {sorted(in_place[kernel])} in place "
                f"and must launch with num_warps=1 (line {call.lineno})"
            )


_SIGNATURE = {
    "sampled_ptr": "*i64",
    "sampled_stride": "i64",
    "num_sampled_ptr": "*i32",
    "resampled_local_argmax_ptr": "*i64",
    "resampled_local_argmax_stride": "i64",
    "resampled_local_max_ptr": "*fp32",
    "resampled_local_max_stride": "i64",
    "resample_num_blocks": "i32",
    "cu_num_logits_ptr": "*i32",
    "expanded_idx_mapping_ptr": "*i32",
    "temp_ptr": "*fp32",
}
# Float arithmetic that could round. Comparisons (setp) and selects (selp)
# are exact, so they are allowed.
_FLOAT_ARITH = re.compile(
    r"^\s*(?:@!?%p\d+\s+)?"
    r"(?:add|sub|mul|mad|fma|div|rcp|sqrt|rsqrt|ex2|lg2|min|max|cvt)"
    r"(?:\.[a-z0-9]+)*\.f(?:16|32|64)\b",
    re.MULTILINE,
)


def _sm70_ptx(num_warps: int) -> str:
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _insert_resampled_kernel,
    )

    try:
        compiled = triton.compile(
            ASTSource(
                _insert_resampled_kernel,
                signature=_SIGNATURE,
                # cdiv(248320, 1024) = 243 blocks, padded to 256.
                constexprs={"PADDED_RESAMPLE_NUM_BLOCKS": 256},
            ),
            target=GPUTarget("cuda", 70, 32),
            options={"num_warps": num_warps},
        )
    except Exception as exc:  # toolchain missing on this host
        pytest.skip(f"offline SM70 compilation unavailable: {exc}")
    return compiled.asm["ptx"]


def _count_store_line(lines: list[str]) -> int:
    # The count update is the only 32-bit global store; the token store is
    # 64-bit.
    stores = [i for i, line in enumerate(lines) if "st.global.b32" in line]
    assert len(stores) == 1
    return stores[0]


def test_sm70_ptx_four_warps_store_count_before_any_barrier():
    """Document the hazard the fix removes."""
    ptx = _sm70_ptx(4)
    assert ".reqntid 128" in ptx
    lines = ptx.splitlines()
    store = _count_store_line(lines)
    barriers = [i for i, line in enumerate(lines) if "bar.sync" in line]
    assert barriers, "four warps need a cross-warp reduction barrier"
    # Every warp loads the count, one thread stores it, and no barrier orders
    # the other warps' loads before that store.
    assert min(barriers) > store
    assert any("ld.global.b32" in line for line in lines[:store])


def test_sm70_ptx_one_warp_has_no_cross_warp_step():
    ptx = _sm70_ptx(1)
    assert ".reqntid 32" in ptx
    assert "bar.sync" not in ptx
    assert "ld.shared" not in ptx and "st.shared" not in ptx
    _count_store_line(ptx.splitlines())


@pytest.mark.parametrize("num_warps", [1, 4])
def test_sm70_ptx_has_no_rounding_float_ops(num_warps):
    """Floats are only compared and selected, so the warp count cannot change
    any value the kernel writes."""
    ptx = _sm70_ptx(num_warps)
    assert _FLOAT_ARITH.findall(ptx) == []
    assert "setp" in ptx and "selp.f32" in ptx
