# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise checkpoint coverage through the real default loader."""

import time
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vllm.config import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class AWQFixture(nn.Module):
    def __init__(self, experts=False):
        super().__init__()
        self.experts = experts
        names = (
            [f"{proj}_{kind}" for proj in ("w13", "w2") for kind in KINDS]
            if experts
            else KINDS
        )
        for name in names:
            param = nn.Parameter(torch.full((2, 2), -1.0), requires_grad=False)
            param.weight_loader = self.weight_loader
            self.register_parameter(name, param)

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return expert_id if expert_id in (0, 1) else -1

    def weight_loader(
        self,
        param,
        value,
        weight_name=None,
        shard_id=None,
        expert_id=None,
        return_success=False,
    ):
        if self.experts:
            local = self._map_global_expert_id_to_local_expert_id(expert_id)
            if local < 0:
                return False
            column = 1 if shard_id == "w3" else 0
            param.data[local, column].copy_(value.reshape(()))
        else:
            param.data.copy_(value)
        return True if return_success else None

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            if self.experts:
                expert, shard, kind = name.split(".")
                param_name = f"{'w2' if shard == 'w2' else 'w13'}_{kind}"
                # Real FusedMoE.load_weights calls the module loader directly.
                self.weight_loader(
                    getattr(self, param_name), value, name, shard, int(expert), True
                )
            else:
                param_name = name
                param = getattr(self, name)
                param.weight_loader(param, value)
            loaded.add(param_name)
        return loaded


KINDS = ["qweight", "qzeros", "scales"]


def checkpoint(experts):
    if experts:
        return {
            f"{expert}.{shard}.{kind}": torch.tensor(float(expert + 2))
            for expert in range(2)
            for shard in ("w1", "w2", "w3")
            for kind in KINDS
        }
    return {kind: torch.full((2, 2), float(i)) for i, kind in enumerate(KINDS)}


def load(model, weights, tmp_path, monkeypatch):
    # Separate files exercise validation after the complete iterator, rather
    # than requiring each contiguous module run to be complete.
    paths = []
    for i, (name, value) in enumerate(weights.items()):
        path = tmp_path / f"part-{i}.safetensors"
        save_file({name: value}, path)
        paths.append(path)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    monkeypatch.setattr(loader, "_init_ep_weight_filter", lambda _: None)
    monkeypatch.setattr(
        loader,
        "get_all_weights",
        lambda *_: (item for path in paths for item in load_file(path).items()),
    )
    loader.counter_before_loading_weights = time.perf_counter()
    loader.load_weights(model, SimpleNamespace(quantization="awq"))


@pytest.mark.parametrize("missing", KINDS)
@pytest.mark.parametrize("experts", [False, True])
def test_missing_awq_tensor_rejected(tmp_path, monkeypatch, missing, experts):
    weights = checkpoint(experts)
    del weights[f"1.w3.{missing}" if experts else missing]
    with pytest.raises(ValueError, match="AWQ.*coverage"):
        load(AWQFixture(experts), weights, tmp_path, monkeypatch)


def test_missing_entire_expert_rejected(tmp_path, monkeypatch):
    weights = {k: v for k, v in checkpoint(True).items() if not k.startswith("1.")}
    with pytest.raises(ValueError, match="AWQ.*coverage"):
        load(AWQFixture(True), weights, tmp_path, monkeypatch)


@pytest.mark.parametrize("experts", [False, True])
def test_complete_awq_preserves_tensors_and_loaders(tmp_path, monkeypatch, experts):
    model = AWQFixture(experts)
    params = dict(model.named_parameters())
    original_loader = model.weight_loader
    load(model, checkpoint(experts), tmp_path, monkeypatch)
    assert model.weight_loader == original_loader
    for name, param in model.named_parameters():
        assert param is params[name]
        assert param.weight_loader == original_loader
        if experts:
            columns = 1 if name.startswith("w2_") else 2
            assert param[:, :columns].tolist() == [[2.0] * columns, [3.0] * columns]
        else:
            assert torch.equal(param, checkpoint(False)[name])


@pytest.mark.parametrize("missing", [None, *KINDS])
def test_real_awq_parameters(tmp_path, monkeypatch, missing):
    import vllm.model_executor.parameter as parameter_module
    from vllm.model_executor.layers.quantization.awq import AWQConfig, AWQLinearMethod

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    model = AWQFixture()
    method = AWQLinearMethod(AWQConfig(4, 32, True))
    method.create_weights(
        model,
        input_size_per_partition=32,
        output_partition_sizes=[16],
        input_size=32,
        output_size=16,
        params_dtype=torch.float16,
        weight_loader=model.weight_loader,
    )
    weights = {
        name: torch.arange(param.numel()).reshape(param.shape).to(param.dtype)
        for name, param in model.named_parameters()
    }
    params = dict(model.named_parameters())
    if missing is not None:
        del weights[missing]
        with pytest.raises(ValueError, match="AWQ.*coverage"):
            load(model, weights, tmp_path, monkeypatch)
    else:
        load(model, weights, tmp_path, monkeypatch)
        for name, param in model.named_parameters():
            assert param is params[name]
            assert torch.equal(param, weights[name])
    for param in model.parameters():
        assert param.weight_loader == model.weight_loader


def test_load_exception_restores_callbacks(tmp_path, monkeypatch):
    model = AWQFixture(True)
    loader = model.weight_loader
    # Invalid shape exercises restoration on a real loader exception.
    weights = {"0.w1.qweight": torch.ones(3)}
    with pytest.raises(RuntimeError):
        load(model, weights, tmp_path, monkeypatch)
    assert model.weight_loader == loader
    assert "weight_loader" not in model.__dict__
    assert all(p.weight_loader == loader for p in model.parameters())


@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("missing", [False, True])
def test_real_moe_loader(tmp_path, monkeypatch, tp_rank, missing):
    from tests.quantization.test_sm70_awq_indexed_prefill import (
        _FakeRoutedExpertsLoader,
    )

    class Model(nn.Module, _FakeRoutedExpertsLoader):
        load_weights = AWQFixture.load_weights

        def __init__(self):
            nn.Module.__init__(self)
            _FakeRoutedExpertsLoader.__init__(self, tp_rank)
            self.experts = True
            self.quant_method.pack_factor = 8
            self.quant_method.group_size = 32
            self.quant_method.create_weights(
                self,
                num_experts=2,
                hidden_size=32,
                intermediate_size_per_partition=32,
                params_dtype=torch.float16,
                weight_loader=self.weight_loader,
            )

    model = Model()
    weights = {}
    for expert in range(2):
        for shard in ("w1", "w2", "w3"):
            for kind in KINDS:
                rows = 128 if shard == "w2" else 32
                cols = 32 if shard == "w2" else 128
                shape = (
                    rows if kind == "qweight" else rows // 32,
                    cols if kind == "scales" else cols // 8,
                )
                dtype = torch.float16 if kind == "scales" else torch.int32
                weights[f"{expert}.{shard}.{kind}"] = torch.full(
                    shape, 2 + expert, dtype=dtype
                )
    if missing:
        del weights["1.w3.scales"]
        with pytest.raises(ValueError, match="AWQ.*coverage"):
            load(model, weights, tmp_path, monkeypatch)
    else:
        load(model, weights, tmp_path, monkeypatch)
        for param in model.parameters():
            assert torch.all(param[0] == 2)
            assert torch.all(param[1] == 3)


def test_awq_also_checks_unquantized_weights(tmp_path, monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class Model(AWQFixture):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.weight = nn.Parameter(torch.empty(2, 2), requires_grad=False)
            self.attention.quant_method = UnquantizedLinearMethod()

    with pytest.raises(ValueError, match="attention.weight"):
        load(Model(), checkpoint(False), tmp_path, monkeypatch)
