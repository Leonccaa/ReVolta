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

from vllm.config import ModelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_sm70_moe import AWQSM70MoEMethod
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.hy_v4_flash.configs import SPARSE_ATTENTION
from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashForCausalLM
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.transformers_utils.config import get_config, uses_mrope
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
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


def _construct_model(checkpoint_metadata, cpu_vllm, monkeypatch, tmp_path, awq):
    path = "vllm.models.hy_v4_flash.nvidia.engram"
    module = ModuleType(path)
    module.__dict__.update(FlashEngram=IndexEngram, FlashNgramHashState=IndexHash)
    monkeypatch.setitem(sys.modules, path, module)
    monkeypatch.setattr(HYV4FlashForCausalLM, "_load_engram_tables", lambda self: None)
    model_dir = checkpoint_metadata
    if awq:
        raw = json.loads((checkpoint_metadata / "config.json").read_text())
        raw["quantization_config"] = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 32,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": [
                "self_attn",
                "shared_experts",
                "routed_expert_",
                "engram_layer",
                ".mlp.gate",
                "layers.0.mlp",
                "lm_head",
                "dfly_layers",
                "vit",
            ],
        }
        model_dir = tmp_path
        (model_dir / "config.json").write_text(json.dumps(raw))
        # Exercise real SM70 AWQ dispatch without initializing CUDA.
        monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "turbomind")
        monkeypatch.setenv("VLLM_SM70_AWQ_MOE_DISABLE", "0")
        monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
        monkeypatch.setattr(
            current_platform, "get_device_capability", lambda: DeviceCapability(7, 0)
        )
        monkeypatch.setattr(
            current_platform, "has_device_capability", lambda c: c <= 70
        )

    cpu_vllm.model_config = ModelConfig(
        model=str(model_dir),
        dtype="half",
        max_model_len=8192,
        enforce_eager=True,
        skip_tokenizer_init=True,
    )
    cpu_vllm.cache_config.cache_dtype = "float16"
    cpu_vllm.scheduler_config.max_num_batched_tokens = 2048
    # CPU platform defaults to disabling hybrid KV; test the M1 grouping.
    cpu_vllm.scheduler_config.disable_hybrid_kv_cache_manager = False
    cpu_vllm.quant_config = VllmConfig.get_quantization_config(
        cpu_vllm.model_config, cpu_vllm.load_config
    )
    with torch.device("meta"), set_default_torch_dtype(torch.float16):
        candidate = HYV4FlashForCausalLM(vllm_config=cpu_vllm)
    assert len(candidate.model.layers) == 42
    assert candidate.lm_head.weight.dtype == torch.float32
    assert all(p.is_meta for p in candidate.parameters())
    if awq:
        quant = cpu_vllm.quant_config
        assert isinstance(quant, AWQConfig)
        assert (quant.weight_bits, quant.group_size, quant.zero_point) == (4, 32, True)
        for layer in candidate.model.layers[1:]:
            assert isinstance(layer.mlp.experts.quant_method, AWQSM70MoEMethod)
            assert layer.mlp.experts.w13_qweight.shape == (512, 1536, 384)
            assert layer.mlp.experts.w2_qweight.shape == (512, 1536, 192)
    return candidate


def _assert_cache_specs(candidate, config):
    owners = get_layers_from_vllm_config(config, AttentionLayerBase)
    specs = {name: layer.get_kv_cache_spec(config) for name, layer in owners.items()}
    assert len(specs) == 66
    assert Counter(type(s) for s in specs.values()) == {
        MLAAttentionSpec: 24,
        SlidingWindowMLASpec: 30,
        CircularBufferSpec: 12,
    }
    for layer in candidate.model.layers:
        attn = layer.self_attn
        spec = specs[attn.attn.prefix]
        assert (spec.block_size, spec.head_size, spec.dtype) == (
            256,
            576,
            torch.float16,
        )
        if attn.is_window_layer:
            assert isinstance(spec, SlidingWindowMLASpec) and spec.sliding_window == 512
        else:
            assert isinstance(spec, MLAAttentionSpec) and spec.compress_ratio == 1
            pool = specs[attn.indexer.k_cache.prefix]
            ring = specs[attn.indexer.tail_cache.prefix]
            assert (pool.block_size, pool.storage_block_size, pool.head_size) == (
                256,
                64,
                128,
            )
            assert pool.dtype == torch.float16 and ring.block_size == 4
    # Plan the startup allocation against a simulated per-rank KV budget.
    (cache,) = get_kv_cache_configs(config, [specs], [2 << 30])
    assert cache.num_blocks > 0
    assert Counter(
        name for group in cache.kv_cache_groups for name in group.layer_names
    ) == Counter(specs.keys())


@pytest.mark.parametrize("awq", [False, True], ids=["packed", "awq"])
@torch.inference_mode()
def test_real_index_maps_exactly(
    checkpoint_metadata, cpu_vllm, monkeypatch, tmp_path, awq
):
    candidate = _construct_model(
        checkpoint_metadata, cpu_vllm, monkeypatch, tmp_path, awq
    )
    _assert_cache_specs(candidate, cpu_vllm)

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
            prefix, _, leaf = name.rpartition(".")
            if prefix.endswith(".experts") and leaf in ("gate_up_proj", "down_proj"):
                gate_up = leaf == "gate_up_proj"
                if awq:
                    projections = (
                        ("gate_proj", "up_proj") if gate_up else ("down_proj",)
                    )
                    for expert in range(512):
                        for projection in projections:
                            for suffix in ("qweight", "qzeros", "scales"):
                                yield f"{prefix}.{expert}.{projection}.{suffix}", scalar
                else:
                    shape = (512, 3072 if gate_up else 1536, 1536)
                    yield name, torch.empty(shape, device="meta")
            else:
                yield name, scalar.view_as(scalar)

    loaded = candidate.load_weights(checkpoint_weights())
    hash_state = candidate.model.engram_hash
    assert sorted(hash_state.maps) == [2, 12, 22, 32]
    for i in hash_state.maps:
        assert candidate.model.layers[i - 1].engram_layer.table_loads == []
    assert loaded == params.keys()
    expected: Counter[tuple[str, str | int | None, int | None]] = Counter()
    suffixes = ("qweight", "qzeros", "scales") if awq else ("weight",)
    w13_suffixes = tuple(f".w13_{suffix}" for suffix in suffixes)
    w2_suffixes = tuple(f".w2_{suffix}" for suffix in suffixes)
    for name in params:
        if name.endswith(w13_suffixes):
            expected.update(
                (name, shard, expert) for expert in range(512) for shard in ("w1", "w3")
            )
        elif name.endswith(w2_suffixes):
            expected.update((name, "w2", expert) for expert in range(512))
        elif name.endswith(".gate_up_proj.weight"):
            expected.update((name, shard, None) for shard in (0, 1))
        else:
            expected[name, None, None] = 1
    assert counts == expected
    assert sum(not name.startswith("vit.") for name in keys) == 1523
