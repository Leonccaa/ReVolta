# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the random MTP fixture builder, including source rejection."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.prepare_mtp_production_layout import BODY_FILE, INDEX_FILE, build

PREFIX = "mtp.layers.0.mlp.experts."
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def expert_tensors(fused):
    gate = torch.arange(64).reshape(2, 8, 4).to(torch.bfloat16)
    up = gate + 128
    down = (torch.arange(64).reshape(2, 4, 8) + 256).to(torch.bfloat16)
    if fused:
        return {
            PREFIX + "gate_up_proj": torch.cat((gate, up), dim=1),
            PREFIX + "down_proj": down,
        }
    return {
        f"{PREFIX}{expert}.{projection}.weight": tensor[expert].contiguous()
        for expert in range(2)
        for projection, tensor in zip(PROJECTIONS, (gate, up, down))
    }


def source_checkpoint(root, tensors, indexed=False, num_experts=2, nested=True):
    root.mkdir()
    config = {"num_experts": num_experts}
    (root / "config.json").write_text(
        json.dumps({"text_config": config} if nested else config)
    )
    save_file(tensors, str(root / "original.safetensors"))
    if indexed:
        (root / INDEX_FILE).write_text(
            json.dumps({"weight_map": dict.fromkeys(tensors, "original.safetensors")})
        )


def digests(root):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
    }


def assert_rejected(tmp_path, tensors, **config):
    source, destination = tmp_path / "source", tmp_path / "generated"
    source_checkpoint(source, tensors, **config)
    before = digests(source)
    with pytest.raises(ValueError):
        build(source, destination)
    assert not destination.exists()
    assert digests(source) == before


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_builder_preserves_complete_experts(tmp_path, fused, indexed, nested):
    source, destination = tmp_path / "source", tmp_path / "generated"
    source_checkpoint(source, expert_tensors(fused), indexed=indexed, nested=nested)
    before = digests(source)
    build(source, destination)
    actual = load_file(str(destination / BODY_FILE))
    expected = expert_tensors(True)
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert actual[name].dtype == torch.bfloat16
        assert torch.equal(actual[name], tensor)
    assert digests(source) == before


def test_builder_rejects_missing_fused_down_expert(tmp_path):
    """Review regression: [2,16,4]/[1,4,8] must fail before mkdir."""
    tensors = expert_tensors(True)
    name = PREFIX + "down_proj"
    tensors[name] = tensors[name][:1].clone()
    assert_rejected(tmp_path, tensors)


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
@pytest.mark.parametrize(
    "defect", ["missing", "too_few", "too_many", "rank", "geometry", "empty", "dtype"]
)
def test_builder_rejects_invalid_fused_projection(tmp_path, projection, defect):
    tensors = expert_tensors(True)
    name = PREFIX + projection
    tensor = tensors[name]
    if defect == "missing":
        del tensors[name]
    elif defect == "too_few":
        tensors[name] = tensor[:1].clone()
    elif defect == "too_many":
        tensors[name] = torch.cat((tensor, tensor[:1]))
    elif defect == "rank":
        tensors[name] = tensor[0].clone()
    elif defect == "geometry":
        tensors[name] = tensor[:, :-1].clone()
    elif defect == "empty":
        tensors[name] = tensor[:, :0].clone()
    else:
        tensors[name] = tensor.half()
    assert_rejected(tmp_path, tensors)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("expert", [0, 1])
@pytest.mark.parametrize("defect", ["missing", "shape", "rank", "dtype"])
def test_builder_rejects_invalid_individual_projection(
    tmp_path, projection, expert, defect
):
    tensors = expert_tensors(False)
    name = f"{PREFIX}{expert}.{projection}.weight"
    tensor = tensors[name]
    if defect == "missing":
        del tensors[name]
    elif defect == "shape":
        tensors[name] = tensor[:-1].clone()
    elif defect == "rank":
        tensors[name] = tensor.unsqueeze(0)
    else:
        # Integer inputs used to be silently promoted to BF16 by torch.stack.
        tensors[name] = tensor.to(torch.int32)
    assert_rejected(tmp_path, tensors)


def test_builder_rejects_complementary_gate_up_shapes(tmp_path):
    tensors = expert_tensors(False)
    for expert in range(2):
        tensors[f"{PREFIX}{expert}.gate_proj.weight"] = torch.zeros(
            7, 4, dtype=torch.bfloat16
        )
        tensors[f"{PREFIX}{expert}.up_proj.weight"] = torch.zeros(
            9, 4, dtype=torch.bfloat16
        )
    assert_rejected(tmp_path, tensors)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("num_experts", [1, 3, 0, -1, True, 2.0])
def test_builder_rejects_invalid_expert_count(tmp_path, fused, num_experts):
    assert_rejected(tmp_path, expert_tensors(fused), num_experts=num_experts)


@pytest.mark.parametrize("fused", [False, True])
def test_builder_rejects_mixed_expert_layout(tmp_path, fused):
    tensors = expert_tensors(fused)
    tensors.update(expert_tensors(not fused))
    assert_rejected(tmp_path, tensors)


def test_builder_rejects_empty_expert_geometry(tmp_path):
    tensors = {
        PREFIX + "gate_up_proj": torch.empty(2, 0, 4, dtype=torch.bfloat16),
        PREFIX + "down_proj": torch.empty(2, 4, 0, dtype=torch.bfloat16),
    }
    assert_rejected(tmp_path, tensors)
