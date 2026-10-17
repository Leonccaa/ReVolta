# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the kernel's block-reduction expressions with CPU arrays.

This checks NaN normalization semantics, not Triton compilation or GPU safety.
The CUDA regression must also exercise the block-local argmax and memory loads.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vllm


@pytest.mark.parametrize(
    "kernel,value_name,index_name,block_size",
    [
        ("_rejection_kernel", "target_local_max", "max_target_block_idx", 8192),
        (
            "_insert_resampled_kernel",
            "resampled_local_max",
            "resampled_max_block_idx",
            1024,
        ),
    ],
)
@pytest.mark.parametrize("vocab_size", [248320, 8192, 8193])
@pytest.mark.parametrize("values", ["nan", "mixed", "finite", "infinity"])
def test_kernel_block_max_normalization(
    kernel, value_name, index_name, block_size, vocab_size, values
):
    path = (
        Path(vllm.__file__).parent
        / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
    )
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kernel
    )
    # Extract only normalization and block argmax from the actual kernel.
    statements: list[ast.stmt] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == value_name and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "tl.where":
                statements.append(node)
        elif target.id == index_name:
            statements.append(node)
    statements.sort(key=lambda n: n.lineno)
    count = (vocab_size + block_size - 1) // block_size
    padded = 1 << (count - 1).bit_length()
    original = np.full(padded, -np.inf)
    original[:count] = np.arange(count, dtype=float)
    if values == "nan":
        original[:count] = np.nan
    elif values == "mixed":
        original[0] = np.nan
    elif values == "infinity":
        original[0] = np.inf
    env = {
        value_name: original.copy(),
        "tl": SimpleNamespace(where=np.where, argmax=np.argmax),
    }
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), env)
    normalized = env[value_name]
    assert not np.isnan(normalized).any()
    assert 0 <= env[index_name] < count
    if values in ("finite", "infinity"):
        np.testing.assert_array_equal(normalized, original)
        assert env[index_name] == np.argmax(original)
