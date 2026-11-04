# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP router batch pack is only prepared for the [512, 2560] router."""

import ast
import inspect

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv


def test_flash_next_router_shape_is_admitted():
    assert gemv._router_batch_shape_ok("model.layers.0.mlp.gate", (512, 2560))


@pytest.mark.parametrize("shape", [(12, 2560), (8, 256), (512, 640), (256, 2560)])
def test_other_router_shapes_fall_back(shape):
    assert not gemv._router_batch_shape_ok("model.layers.0.mlp.gate", shape)


def test_flash_next_pack_is_unchanged():
    raw = torch.randint(-(2**15), 2**15, (512, 2560), dtype=torch.int16)
    packed = gemv._pack_router_batch_weight(raw.view(torch.float16))
    restored = packed.permute(0, 4, 3, 1, 2, 5).contiguous().view(512, 2560)
    assert torch.equal(restored.view(torch.int16), raw)


def test_router_batch_preparation_checks_the_shape():
    tree = ast.parse(inspect.getsource(gemv))
    prepares = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "child._sm70_mtp_prepare_router_batch"
    ]
    assert len(prepares) == 1
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "_router_batch_shape_ok(prefix, shape)" in ast.unparse(node.test)
        and prepares[0] in list(ast.walk(node))
    ]
    assert len(guards) == 1
