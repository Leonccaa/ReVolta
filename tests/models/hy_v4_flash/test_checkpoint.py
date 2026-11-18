# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.hy_v4_flash.configs import SPARSE_ATTENTION
from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashForCausalLM
from vllm.transformers_utils.config import get_config, uses_mrope
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def checkpoint_metadata():
    path = os.environ.get("HYV4_FLASH_CHECKPOINT_METADATA", "")
    if not path:
        pytest.skip(
            "set HYV4_FLASH_CHECKPOINT_METADATA to the checkpoint metadata directory"
        )
    return Path(path)


def test_checkpoint_config(checkpoint_metadata):
    config = get_config(checkpoint_metadata, trust_remote_code=False)
    assert config.get_text_config() is config
    assert config.architectures == ["HYV4FlashForCausalLM"]
    model_config = SimpleNamespace(model_impl="auto")
    assert ModelRegistry.resolve_model_cls(config.architectures, model_config)[0] is (
        HYV4FlashForCausalLM
    )
    arch = ModelArchConfigConvertorBase(config, config)
    assert arch.is_deepseek_mla() and arch.get_head_size() == 576
    assert not uses_mrope(config)
    assert config.num_experts == 512 and config.num_shared_experts == 1
    assert config.expert_hidden_dim == 1536 and config.route_norm
    assert config.router_scaling_factor == 3.998
    assert sum(kind == SPARSE_ATTENTION for kind in config.layer_types) == 12
    assert config.rope_for_layer(0)["rope_theta"] == 1e7
    assert config.rope_for_layer(2)["rope_theta"] == 1e4
    assert not hasattr(config, "vision_config")


def _params(names):
    result = nn.Module()
    for name in names.split():
        setattr(result, name, nn.Parameter(torch.empty(1)))
    return result


class IndexAttention(nn.Module):
    """Parameter names from the lane A / upstream attention contract."""

    def __init__(self, vllm_config, config, layer_idx, topk_indices_buffer, prefix):
        super().__init__()
        for name in (
            "kv_a_layernorm",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "linear_gate",
            "o_proj",
            "post_proj_norm",
            "q_a_layernorm",
            "q_a_proj",
            "q_b_proj",
        ):
            setattr(self, name, _params("weight"))
        self.learnable_sink_param = nn.Parameter(torch.empty(1))
        if config.layer_types[layer_idx] == SPARSE_ATTENTION:
            self.indexer = _params("intra_group_ape")
            for name in (
                "head_weight_projection",
                "key_gate_projection",
                "key_norm",
                "key_value_projection",
                "query_projection",
                "token_key_norm",
                "token_key_projection",
            ):
                setattr(self.indexer, name, _params("weight"))


class IndexEngram(nn.Module):
    """Weight-facing part of the lane C contract, without host allocations."""

    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        self.multi_head_embedding = SimpleNamespace(
            head_start=0, head_end=16, load_weight=self.load_table_weight
        )
        self.table_loads = []
        self.lookup_per_head_norms = nn.ModuleList(_params("weight") for _ in range(16))
        self.key_projs = nn.ModuleList(_params("weight bias") for _ in range(4))
        self.norm_key = nn.ModuleList(_params("weight") for _ in range(4))
        self.norm_query = nn.ModuleList(_params("weight") for _ in range(4))
        self.value_proj = _params("weight bias")
        self.unified_pre_gamma = nn.Parameter(torch.empty(1))
        self.unified_per_hc_scales = nn.ParameterList(
            nn.Parameter(torch.empty(1)) for _ in range(4)
        )

    def load_table_weight(self, name, value):
        self.table_loads.append(name)


class IndexHash:
    def __init__(self, config, dummy):
        self.layout = None
        self.maps = []

    def load_lookup(self, value, layer_id):
        self.maps.append(layer_id)


@torch.inference_mode()
def test_real_index_maps_exactly(checkpoint_metadata, cpu_vllm, monkeypatch):
    for name, stubs in (
        ("sm70.attention", {"HYV4FlashAttention": IndexAttention}),
        (
            "nvidia.engram",
            {"FlashEngram": IndexEngram, "FlashNgramHashState": IndexHash},
        ),
    ):
        path = f"vllm.models.hy_v4_flash.{name}"
        module = ModuleType(path)
        module.__dict__.update(stubs)
        monkeypatch.setitem(sys.modules, path, module)
    config = get_config(checkpoint_metadata, trust_remote_code=False)
    monkeypatch.setattr(HYV4FlashForCausalLM, "_load_engram_tables", lambda self: None)
    cpu_vllm.model_config = SimpleNamespace(hf_text_config=config, dtype=torch.float16)
    with torch.device("meta"):
        candidate = HYV4FlashForCausalLM(vllm_config=cpu_vllm)

    # Count individual visits as well as unique parameters, so a wrong alias,
    # double load, repeated shard or omitted checkpoint key cannot pass.
    counts: Counter[tuple[str, str | int | None, int | None]] = Counter()

    def record(name, param, value, dest=None, shard_id=None, expert_id=None):
        shard = dest if isinstance(dest, int) else shard_id
        counts[name, shard, expert_id] += 1

    params = dict(candidate.named_parameters())
    for name, param in params.items():
        param.weight_loader = partial(record, name)

    index = json.loads(
        (checkpoint_metadata / "model.safetensors.index.json").read_text()
    )
    keys = index["weight_map"]
    scalar = torch.empty(1, device="meta")

    def checkpoint_weights():
        for name in keys:
            if candidate.skip_checkpoint_weight(name):
                continue
            if name.endswith(".experts.gate_up_proj"):
                shape = (512, 3072, 1536)
            elif name.endswith(".experts.down_proj"):
                shape = (512, 1536, 1536)
            else:
                yield name, scalar.view_as(scalar)
                continue
            yield name, torch.empty(shape, device="meta")

    loaded = candidate.load_weights(checkpoint_weights())
    hash_state = candidate.model.engram_hash
    assert sorted(hash_state.maps) == [2, 12, 22, 32]
    for i in hash_state.maps:
        assert candidate.model.layers[i - 1].engram_layer.table_loads == []
    assert loaded == params.keys()
    expected: Counter[tuple[str, str | int | None, int | None]] = Counter()
    for name in params:
        if name.endswith(".w13_weight"):
            expected.update(
                (name, shard, expert) for expert in range(512) for shard in ("w1", "w3")
            )
        elif name.endswith(".w2_weight"):
            expected.update((name, "w2", expert) for expert in range(512))
        elif name.endswith(".gate_up_proj.weight"):
            expected.update((name, shard, None) for shard in (0, 1))
        else:
            expected[name, None, None] = 1
    assert counts == expected
    assert sum(not name.startswith("vit.") for name in keys) == 1523
