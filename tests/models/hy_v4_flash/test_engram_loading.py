# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import direct_io, weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.models.hy_v4_flash.nvidia import engram
from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashForCausalLM

pytestmark = pytest.mark.skip_global_cleanup


def make_model(tmp_path, monkeypatch):
    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: 2)
    cfg = SimpleNamespace(
        device_config=SimpleNamespace(device_type="cpu"),
        load_config=SimpleNamespace(load_format="safetensors"),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    candidate = HYV4FlashForCausalLM.__new__(HYV4FlashForCausalLM)
    nn.Module.__init__(candidate)
    candidate.model = nn.Module()
    candidate.model.engram_indices = {i: n for n, i in enumerate((1, 11, 21, 31))}
    candidate.model.layers = nn.ModuleList(nn.Module() for _ in range(32))
    candidate.model.layers[0].test = nn.Linear(1, 1, bias=False)
    candidate.lm_head = nn.Linear(1, 1, bias=False)
    candidate.checkpoint_model_config = SimpleNamespace(
        model=str(tmp_path), revision=None
    )
    candidate.checkpoint_cache_dir = None
    for i in candidate.model.engram_indices:
        layer = candidate.model.layers[i]
        layer.engram_layer = nn.Module()
        layer.engram_layer.multi_head_embedding = engram.FlashEmbedding(
            [5, 7, 11, 13], 4, cfg, f"layer{i}"
        )
    return candidate


def checkpoint(tmp_path, *, bf16=False):
    weight_map = {}
    for i in (1, 11, 21, 31):
        name = f"model.layers.{i}.engram_layer.multi_head_embedding.embedding.weight"
        shard = f"model-{i}.safetensors"
        values = torch.arange(144, dtype=torch.float32).reshape(36, 4) / 10
        values = values.to(torch.bfloat16 if bf16 else torch.float8_e4m3fn)
        tensors = {name: values}
        weight_map[name] = shard
        if not bf16:
            tensors[name + "_scale_inv"] = torch.arange(1, 37, dtype=torch.bfloat16)
            weight_map[name + "_scale_inv"] = shard
        if i == 1:
            tensors["lm_head.weight"] = torch.tensor([[2.0]])
            tensors["model.layers.0.test.weight"] = torch.tensor([[3.0]])
            weight_map["lm_head.weight"] = shard
            weight_map["model.layers.0.test.weight"] = shard
        save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return weight_map


@pytest.mark.parametrize("strategy", [None, "direct"])
def test_default_loader_skips_tables_and_reads_rank_rows(
    tmp_path, monkeypatch, strategy
):
    weight_map = checkpoint(tmp_path)
    candidate = make_model(tmp_path, monkeypatch)
    tensor_reads = []
    slices = []
    direct_yields = []

    class Slice:
        def __init__(self, key, wrapped):
            self.key, self.wrapped = key, wrapped

        def get_dtype(self):
            return self.wrapped.get_dtype()

        def __getitem__(self, item):
            slices.append((self.key, item))
            return self.wrapped[item]

    class Reader:
        def __init__(self, *args, **kwargs):
            self.wrapped = safe_open(*args, **kwargs)

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def keys(self):
            return self.wrapped.keys()

        def get_tensor(self, key):
            tensor_reads.append(key)
            return self.wrapped.get_tensor(key)

        def get_slice(self, key):
            return Slice(key, self.wrapped.get_slice(key))

    direct = weight_utils.direct_io_weights

    def recorded_direct(*args, **kwargs):
        for name, value in direct(*args, **kwargs):
            direct_yields.append(name)
            yield name, value

    monkeypatch.setattr(weight_utils, "safe_open", Reader)
    monkeypatch.setattr(direct_io, "safe_open", Reader)
    monkeypatch.setattr(engram, "safe_open", Reader)
    monkeypatch.setattr(weight_utils, "direct_io_weights", recorded_direct)
    load_config = LoadConfig(
        load_format="safetensors",
        safetensors_load_strategy=strategy,
        use_tqdm_on_load=False,
    )
    loader = DefaultModelLoader(load_config)
    loaded = candidate.load_weights(
        loader.get_all_weights(candidate.checkpoint_model_config, candidate)
    )
    assert loaded == {"lm_head.weight", "model.layers.0.test.weight"}
    assert candidate.lm_head.weight.item() == 2.0
    assert candidate.model.layers[0].test.weight.item() == 3.0
    if strategy == "direct":
        assert direct_yields == ["model.layers.0.test.weight"]
    table_names = {name for name in weight_map if "engram_layer" in name}
    assert table_names.isdisjoint(tensor_reads)
    assert table_names.isdisjoint(direct_yields)
    assert len(slices) == 8
    for key, item in slices:
        assert key in table_names
        expected = slice(12, 23)
        if key.endswith(".weight"):
            assert item == (expected, slice(None))
        else:
            assert item == expected
    for i in candidate.model.engram_indices:
        table = candidate.model.layers[i].engram_layer.multi_head_embedding
        assert table.vocab_start == 12 and table.vocab_end == 23
        assert table.storage.numel() == 11 * 6


def test_bf16_table_has_clear_error(tmp_path, monkeypatch):
    checkpoint(tmp_path, bf16=True)
    candidate = make_model(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="FP8 weights and BF16 row scales"):
        candidate._load_engram_tables()
    table = candidate.model.layers[1].engram_layer.multi_head_embedding
    name = "model.layers.1.engram_layer.multi_head_embedding.embedding.weight"
    with pytest.raises(ValueError, match="FP8 weights and BF16 row scales"):
        table.load_from_safetensors(str(tmp_path / "model-1.safetensors"), name)
    assert table.storage.numel() == 0
