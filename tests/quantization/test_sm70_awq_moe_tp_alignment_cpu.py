# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for AWQ groups split by tensor-parallel MoE sharding.

Run with --confcutdir=tests/quantization -o addopts=''. Compile the real loader,
allocation and padding function bodies from source to avoid importing vLLM's
GPU runtime. No CUDA discovery, native ops or distributed services are used.
"""

import ast
from enum import Enum
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
AWQ_SOURCE = "vllm/model_executor/layers/quantization/awq_sm70_moe.py"
LOADER_SOURCE = "vllm/model_executor/layers/fused_moe/layer.py"
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
GROUP_SIZE = 32


def _source_functions(path, names, namespace):
    """Execute unchanged function bodies; only runtime imports are omitted."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        # Static helpers are called directly or bound explicitly below.
        node.decorator_list = []
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, *nodes],
        type_ignores=[],
    )
    exec(compile(module, str(ROOT / path), "exec"), namespace)


@pytest.fixture(scope="module")
def source():
    namespace = {"torch": torch, "Parameter": torch.nn.Parameter, "Enum": Enum}
    _source_functions(
        LOADER_SOURCE,
        (
            "FusedMoeWeightScaleSupported",
            "_get_hidden_dim",
            "_narrow_expert_data_for_padding",
            "_load_w2",
            "_load_model_weight_or_group_weight_scale",
        ),
        namespace,
    )
    # Attribute attachment does not affect allocation or the sharding helpers.
    namespace["set_weight_attrs"] = lambda weight, attrs: None
    _source_functions(
        AWQ_SOURCE,
        (
            "create_weights",
            "_round_up",
            "_pad_last_dim",
            "_pad_penultimate_dim",
            "_align_awq_output_dim",
            "_align_awq_input_dim",
        ),
        namespace,
    )
    return SimpleNamespace(**namespace)


def _pack_awq(values):
    lanes = values.long().reshape(*values.shape[:-1], -1, 8)
    packed = torch.zeros(lanes.shape[:-1], dtype=torch.int64)
    for nibble, column in enumerate(AWQ_ORDER):
        packed |= lanes[..., column] << (4 * nibble)
    return packed.to(torch.int32)


def _unpack_awq(packed):
    # Inverse permutation used by awq_sm70_prepare_impl for AWQ GEMM words.
    shifts = torch.tensor((0, 4, 1, 5, 2, 6, 3, 7)) * 4
    return ((packed.long().unsqueeze(-1) >> shifts) & 15).flatten(-2)


def _dequantize(qweight, scales, qzeros):
    codes = _unpack_awq(qweight)
    zeros = _unpack_awq(qzeros).half()
    group = torch.arange(qweight.shape[0]) // GROUP_SIZE
    # Native FP16 bias followed by one FP16 FMA rounding. Power-of-two fixture
    # scales make this equal to (q - z) * scale, so rounding cannot hide the bug.
    bias = (-zeros * scales).half()
    return (codes.float() * scales[group].float() + bias[group].float()).half()


def _load_down_proj(source, checkpoint, tp_size, rank):
    loader = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_size=tp_size)
        ),
        _get_hidden_dim=source._get_hidden_dim,
        _narrow_expert_data_for_padding=source._narrow_expert_data_for_padding,
    )
    loader._load_w2 = MethodType(source._load_w2, loader)
    qweight, scales, qzeros = checkpoint
    k = qweight.shape[0] // tp_size
    # Reproduce the pre-fix create_weights layout, including floor(K/group).
    # The fixed public allocation entry rejects this layout when k=176.
    local = (
        torch.empty((k, qweight.shape[1]), dtype=qweight.dtype),
        torch.empty((k // GROUP_SIZE, scales.shape[1]), dtype=scales.dtype),
        torch.empty((k // GROUP_SIZE, qzeros.shape[1]), dtype=qzeros.dtype),
    )
    for param, loaded in zip(local, checkpoint):
        source._load_model_weight_or_group_weight_scale(
            loader,
            expert_data=param,
            shard_dim=0,
            shard_id="w2",
            loaded_weight=loaded,
            tp_rank=rank,
        )
    return source._align_awq_input_dim(*local, GROUP_SIZE, GROUP_SIZE)[:3]


@pytest.mark.parametrize("k", [640, 704])
def test_down_proj_tp4_against_unsharded_reference(source, k):
    n = 32
    rows = torch.arange(k)[:, None]
    columns = torch.arange(n)[None, :]
    codes = (rows * 3 + columns * 5) % 16
    zeros = (torch.arange(k // GROUP_SIZE)[:, None] + columns) % 16
    scales = (torch.arange(1, k // GROUP_SIZE + 1)[:, None] / 1024).expand(-1, n).half()
    checkpoint = (_pack_awq(codes), scales, _pack_awq(zeros))
    # Independent reference uses the unpacked integers, not the loader/decoder.
    reference_weight = (codes - zeros[rows[:, 0] // GROUP_SIZE]) * scales[
        rows[:, 0] // GROUP_SIZE
    ].double()
    x = ((torch.arange(3 * k).reshape(3, k) % 7) - 3).double()
    reference = x @ reference_weight
    tp1_weight = _dequantize(*_load_down_proj(source, checkpoint, 1, 0))
    torch.testing.assert_close(tp1_weight.double(), reference_weight, rtol=0, atol=0)
    torch.testing.assert_close(x @ tp1_weight.double(), reference, rtol=0, atol=0)

    partials = []
    for rank in range(4):
        local = _load_down_proj(source, checkpoint, 4, rank)
        weight = _dequantize(*local).double()
        start = rank * (k // 4)
        local_x = torch.zeros((x.shape[0], weight.shape[0]), dtype=x.dtype)
        local_x[:, : k // 4] = x[:, start : start + k // 4]
        partials.append(local_x @ weight)
        if k == 704:
            # Only global metadata groups [5r, 5r+4] were loaded. Local group
            # 5 is zero-padded, even though its first 16 weight rows are real.
            torch.testing.assert_close(local[1][:5], scales[5 * rank : 5 * rank + 5])
            assert torch.count_nonzero(local[1][5]) == 0
            assert torch.count_nonzero(local[2][5]) == 0
            assert torch.count_nonzero(weight[160:]) == 0
    error = (sum(partials) - reference).abs().max().item()
    if k == 640:
        assert error == 0
    else:
        assert error > 1
        # Isolate the exact straddling group 5 (global rows 160..191).
        # Rank 1 receives group 5: correct for rows 176..191, but incorrectly
        # reuses it for rows 192..207, where checkpoint group 6 is required.
        for rank, row, matches in ((0, 160, False), (1, 0, True), (1, 16, False)):
            local = _load_down_proj(source, checkpoint, 4, rank)
            actual = _dequantize(*local)[row : row + 16].double()
            start = rank * 176 + row
            assert torch.equal(actual, reference_weight[start : start + 16]) == matches
    print(f"K={k}, TP1 max error=0, TP4 max error={error}")


def test_w13_tail_padding_also_changes_gate_up_boundary(source):
    # Legacy W13 packs [gate(176), up(176)] and pads only the end to 384.
    # SiLU then splits at 192, so even repairing W2 metadata alone is unsafe.
    k = 176
    gate = torch.arange(1, k + 1).half()
    up = gate + 1000
    scales = torch.cat((gate, up)).reshape(1, -1)
    qweight = torch.zeros((32, 2 * k // 8), dtype=torch.int32)
    qzeros = torch.zeros((1, 2 * k // 8), dtype=torch.int32)
    _, padded_scales, _, n = source._align_awq_output_dim(
        qweight, scales, qzeros, 8, 64
    )
    assert n == 384
    torch.testing.assert_close(padded_scales[0, 176:192], up[:16])
    torch.testing.assert_close(padded_scales[0, 192:352], up[16:])
    assert torch.count_nonzero(padded_scales[0, 352:]) == 0


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("k", [16, 176])
def test_create_weights_rejects_split_groups_before_allocation(
    source, monkeypatch, group_size, k
):
    layer = torch.nn.Module()
    method = SimpleNamespace(group_size=group_size, pack_factor=8)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("misaligned AWQ weights must be rejected before allocation")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    with pytest.raises(
        ValueError,
        match=rf"down_proj input size per TP shard.*"
        rf"intermediate_size_per_partition={k}, group_size={group_size}",
    ):
        source.create_weights(method, layer, 1, 256, k, torch.float16)
    assert not list(layer.parameters())


@pytest.mark.parametrize(
    "group_size,k", [(32, 160), (32, 192), (32, 352), (32, 704), (64, 320), (128, 640)]
)
def test_create_weights_preserves_aligned_layout(source, group_size, k):
    layer = torch.nn.Module()
    method = SimpleNamespace(group_size=group_size, pack_factor=8)
    source.create_weights(method, layer, 1, 256, k, torch.float16)
    assert layer.w2_qweight.shape == (1, k, 32)
    assert layer.w2_scales.shape == (1, k // group_size, 256)
    assert layer.w2_qzeros.shape == (1, k // group_size, 32)
    assert layer.w13_qweight.shape == (1, 256, 2 * k // 8)
    assert layer.w13_scales.shape == (1, 256 // group_size, 2 * k)
    assert layer.w13_qzeros.shape == (1, 256 // group_size, 2 * k // 8)
