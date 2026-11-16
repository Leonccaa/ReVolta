# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Directory-level MTP selection and real recursive expert loading on CPU."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from vllm.config import LoadConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.models.qwen4_exp.nvidia.mtp import (
    Qwen4ExpMTP,
    Qwen4ExpMultiTokenPredictor,
)

EXPERT_PREFIX = "mtp.layers.0.mlp.experts."
BODY_FILE = "model-00002-of-00002.safetensors"
SCALE_FILE = "model-bf16-kvscales-mtp.safetensors"
TARGET_NAME = "model.language_model.layers.0.mlp.gate.weight"


class CPUExperts(nn.Module):
    # Use the production mapping, recursive loader and TP slicing; only the
    # GPU/distributed constructor and global-to-local expert lookup are stubbed.
    load_weights = FusedMoE.load_weights
    weight_loader = FusedMoE.weight_loader
    _load_model_weight_or_group_weight_scale = (
        FusedMoE._load_model_weight_or_group_weight_scale
    )
    _load_w13 = FusedMoE._load_w13
    _load_w2 = FusedMoE._load_w2
    _get_hidden_dim = staticmethod(FusedMoE._get_hidden_dim)
    _narrow_expert_data_for_padding = staticmethod(
        FusedMoE._narrow_expert_data_for_padding
    )

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return expert_id

    def __init__(self, tp_size, rank, intermediate_size):
        super().__init__()
        self.layer_name = "mtp.layers.48.mlp.experts"
        self.quant_config = None
        self.quant_method = SimpleNamespace()
        self.tp_rank, self.tp_size = rank, tp_size
        self.moe_config = SimpleNamespace(
            is_act_and_mul=True,
            moe_parallel_config=SimpleNamespace(tp_size=tp_size),
        )
        width = intermediate_size // tp_size
        self.w13_weight = nn.Parameter(
            torch.full((2, 2 * width, 4), float("nan"), dtype=torch.float16),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.full((2, 4, width), float("nan"), dtype=torch.float16),
            requires_grad=False,
        )
        self.expert_mapping = FusedMoE.make_expert_params_mapping(
            self, "gate_proj", "down_proj", "up_proj", 2, include_fused=True
        )


def make_model(tp_size, rank, intermediate_size):
    model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
    nn.Module.__init__(model)
    predictor = Qwen4ExpMultiTokenPredictor.__new__(Qwen4ExpMultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.config = SimpleNamespace(num_experts=2)
    predictor.fp8_mtp_checkpoint_prefixes = set()
    predictor.fp8_mtp_tp_size = tp_size
    predictor._kv_cache_dtype = "float16"
    predictor.embed_tokens = nn.Embedding(3, 4, dtype=torch.float16)
    predictor.fc_embedding = nn.Linear(4, 4, bias=False, dtype=torch.float16)
    predictor.fc_hidden = nn.Linear(4, 4, bias=False, dtype=torch.float16)
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = CPUExperts(tp_size, rank, intermediate_size)
    layer.self_attn = nn.Module()
    for kind in ("k", "v"):
        layer.self_attn.register_parameter(
            f"{kind}_scale", nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        )
    predictor.layers = nn.ModuleList([layer])
    model.model = predictor
    model.lm_head = nn.Linear(4, 3, bias=False, dtype=torch.float16)
    return model


def write_checkpoint(root, layout, intermediate_size):
    # Distinct BF16 values in both halves catch cutting the fused 2I axis
    # contiguously instead of slicing gate and up independently.
    gate_up = (
        torch.arange(2 * 2 * intermediate_size * 4)
        .reshape(2, 2 * intermediate_size, 4)
        .to(torch.bfloat16)
    )
    down = (
        torch.arange(2 * 4 * intermediate_size).reshape(2, 4, intermediate_size) + 10000
    ).to(torch.bfloat16)
    body = {
        "model.language_model.embed_tokens.weight": torch.ones(3, 4),
        "lm_head.weight": torch.full((3, 4), 2.0),
        "mtp.fc_embedding.weight": torch.full((4, 4), 3.0),
        "mtp.fc_hidden.weight": torch.full((4, 4), 4.0),
    }
    if layout == "per_expert":
        for expert in range(2):
            for proj, tensor in (
                ("gate_proj", gate_up[expert, :intermediate_size]),
                ("up_proj", gate_up[expert, intermediate_size:]),
                ("down_proj", down[expert]),
            ):
                body[f"{EXPERT_PREFIX}{expert}.{proj}.weight"] = tensor.contiguous()
    else:
        body[f"{EXPERT_PREFIX}gate_up_proj"] = gate_up
        body[f"{EXPERT_PREFIX}down_proj"] = down
    scales = {
        f"mtp.layers.0.self_attn.{kind}_scale": torch.tensor(value)
        for kind, value in (("k", 0.25), ("v", 0.5))
    }
    shards = {
        "model-00001-of-00002.safetensors": {TARGET_NAME: torch.zeros(4, 4)},
        BODY_FILE: body,
        SCALE_FILE: scales,
    }
    if layout == "compact":
        shards.pop(BODY_FILE)
        shards.pop(SCALE_FILE)
        shards["model-bf16-full-mtp.safetensors"] = body | scales
    weight_map = {}
    for filename, tensors in shards.items():
        save_file(tensors, str(root / filename))
        weight_map.update(dict.fromkeys(tensors, filename))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return gate_up, down, weight_map


def make_loader():
    return DefaultModelLoader(
        LoadConfig(
            load_format="safetensors",
            use_tqdm_on_load=False,
            safetensors_load_strategy="lazy",
        )
    )


@pytest.mark.parametrize("layout", ["production", "compact", "per_expert"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("intermediate_size", [8, 640])
def test_checkpoint_selection_and_expert_loading(
    tmp_path, monkeypatch, layout, tp_size, intermediate_size
):
    gate_up, down, weight_map = write_checkpoint(tmp_path, layout, intermediate_size)
    loader = make_loader()
    _, files, _, _ = loader._prepare_weights(
        str(tmp_path), None, None, True, Qwen4ExpMTP.allow_patterns_overrides
    )
    assert {Path(f).name for f in files} == set(weight_map.values())

    reads = []

    class TrackedFile:
        def __init__(self, *args, **kwargs):
            self.file = safe_open(*args, **kwargs)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def keys(self):
            return self.file.keys()

        def get_tensor(self, name):
            reads.append(name)
            assert name != TARGET_NAME
            return self.file.get_tensor(name)

    monkeypatch.setattr(weight_utils, "safe_open", TrackedFile)
    all_w13, all_w2 = [], []
    for rank in range(tp_size):
        model = make_model(tp_size, rank, intermediate_size)
        weights = loader.get_all_weights(
            SimpleNamespace(model=str(tmp_path), revision=None), model
        )
        loaded = model.load_weights(weights)
        assert loaded == set(dict(model.named_parameters()))
        experts = model.model.layers[0].mlp.experts
        lo, hi = (
            rank * intermediate_size // tp_size,
            (rank + 1) * intermediate_size // tp_size,
        )
        expected = torch.cat(
            (
                gate_up[:, lo:hi],
                gate_up[:, intermediate_size + lo : intermediate_size + hi],
            ),
            dim=1,
        ).half()
        assert torch.equal(experts.w13_weight, expected)
        assert torch.equal(experts.w2_weight, down[:, :, lo:hi].half())
        assert torch.equal(model.model.embed_tokens.weight, torch.ones(3, 4).half())
        assert torch.equal(model.lm_head.weight, torch.full((3, 4), 2.0).half())
        assert torch.equal(
            model.model.fc_embedding.weight, torch.full((4, 4), 3.0).half()
        )
        assert torch.equal(model.model.fc_hidden.weight, torch.full((4, 4), 4.0).half())
        assert model.model.layers[0].self_attn.k_scale.item() == 0.25
        assert model.model.layers[0].self_attn.v_scale.item() == 0.5
        all_w13.append(experts.w13_weight.detach())
        all_w2.append(experts.w2_weight.detach())
    gates, ups = zip(*(w.chunk(2, dim=1) for w in all_w13))
    assert torch.equal(torch.cat((*gates, *ups), dim=1), gate_up.half())
    assert torch.equal(torch.cat(all_w2, dim=2), down.half())
    assert set(reads) == set(weight_map) - {TARGET_NAME}


@pytest.mark.parametrize("missing", ["body_shard", "down_tensor", "scale_only"])
def test_incomplete_mtp_checkpoint_is_rejected(tmp_path, missing):
    _, _, weight_map = write_checkpoint(tmp_path, "production", 8)
    if missing == "body_shard":
        (tmp_path / BODY_FILE).unlink()
    else:
        with safe_open(tmp_path / BODY_FILE, framework="pt") as f:
            tensors = {name: f.get_tensor(name) for name in f.keys()}  # noqa: SIM118
        removed = (
            set(tensors) if missing == "scale_only" else {f"{EXPERT_PREFIX}down_proj"}
        )
        save_file(
            {k: v for k, v in tensors.items() if k not in removed},
            str(tmp_path / BODY_FILE),
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        k: v for k, v in weight_map.items() if k not in removed
                    }
                }
            )
        )
    model = make_model(1, 0, 8)
    with pytest.raises(ValueError, match="routed-expert checkpoint weights.*w2_weight"):
        model.load_weights(
            make_loader().get_all_weights(
                SimpleNamespace(model=str(tmp_path), revision=None), model
            )
        )
