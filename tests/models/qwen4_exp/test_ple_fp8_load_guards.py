# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from tests.models.qwen4_exp.test_ple import _make_fp8_ngram_embedding_for_load_test
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
)


@pytest.fixture(params=["device", "pinned"])
def module(request, monkeypatch):
    module = _make_fp8_ngram_embedding_for_load_test()
    if request.param == "pinned":
        layer = Qwen4ExpPinnedHostEmbedding.__new__(Qwen4ExpPinnedHostEmbedding)
        nn.Module.__init__(layer)
        for attr in ("org_vocab_size", "embedding_dim", "shard_indices"):
            setattr(layer, attr, getattr(module.ngram_embedding, attr))
        layer.weight = module.ngram_embedding.weight
        layer.weight_scale = module.ngram_embedding.weight_scale
        layer.ple_device_table = torch.full((1, 2), -1.0).to(torch.float8_e4m3fn)
        layer.ple_host_storage = torch.full((3, 2), -1.0).to(torch.float8_e4m3fn)
        layer._device_rows, layer._host_rows, layer._disk_rows = 1, 3, 0
        layer._checkpoint_shard_loaded = False
        layer._accelerator_weight_views = {}
        monkeypatch.setattr(layer, "materialize_tables", lambda: None)
        monkeypatch.setattr(layer, "get_accelerator_weight", lambda device: None)
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
        module.ngram_embedding = layer
    return module


def shard(index=0, dtype=torch.float8_e4m3fn):
    return (
        f"ngram_embedding.shard_{index}.weight",
        torch.arange(index * 8, index * 8 + 8).reshape(4, 2).to(dtype),
    )


SCALE = ("ngram_embedding.weight_scale", torch.tensor([0.25], dtype=torch.bfloat16))


def finalize(module):
    Qwen4ExpPLEFp8EmbeddingMethod().process_weights_after_loading(
        module.ngram_embedding
    )


@pytest.mark.parametrize(
    "weights",
    [[shard(0), SCALE], [shard(0), shard(0), SCALE], [SCALE], []],
    ids=["missing", "duplicate-masks-hole", "scale-only", "empty"],
)
def test_incomplete_checkpoint_is_rejected_only_after_loading(module, weights):
    module.load_weights(weights)
    with pytest.raises((ValueError, RuntimeError), match="PLE.*(rows|coverage)"):
        finalize(module)


def test_scale_must_come_from_checkpoint(module):
    module.load_weights([shard(0), shard(1)])
    with pytest.raises((ValueError, RuntimeError), match="weight_scale"):
        finalize(module)


@pytest.mark.parametrize("module", ["pinned"], indirect=True)
def test_pinned_finalizer_itself_rejects_incomplete_rows(module):
    module.load_weights([shard(0), SCALE])
    with pytest.raises(RuntimeError, match="row coverage"):
        module.ngram_embedding.prepare_accelerator_weight()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_wrong_resident_dtype_is_rejected(module, dtype):
    with pytest.raises(ValueError, match="dtype"):
        module.load_weights([shard(dtype=dtype)])


def test_complete_checkpoint_split_across_calls_preserves_bytes(module):
    module.load_weights([shard(1)])
    module.load_weights([shard(0), SCALE])
    finalize(module)
    layer = module.ngram_embedding
    actual = (
        torch.cat(
            [
                layer.ple_device_table.view(torch.uint8),
                layer.ple_host_storage.view(torch.uint8),
            ]
        )
        if isinstance(layer, Qwen4ExpPinnedHostEmbedding)
        else layer.weight.view(torch.uint8)
    )
    expected = (
        torch.arange(4, 12).reshape(4, 2).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    assert torch.equal(actual, expected)
    assert torch.equal(layer.weight_scale, SCALE[1])


def test_padding_is_not_a_checkpoint_hole(module):
    # Only global rows [2, 5) are valid, row 5 is padding on this rank.
    module.ngram_embedding.org_vocab_size = 5
    module.ngram_embedding.shard_indices.org_vocab_end_index = 5
    module.load_weights(
        [
            (
                "ngram_embedding.shard_0.weight",
                torch.ones(3, 2).to(torch.float8_e4m3fn),
            ),
            (
                "ngram_embedding.shard_1.weight",
                torch.ones(2, 2).to(torch.float8_e4m3fn),
            ),
            SCALE,
        ]
    )
    finalize(module)


def test_explicit_dummy_loading_can_initialize_private_tables(module):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.load_config.load_format = "dummy"
    with set_current_vllm_config(config):
        finalize(module)
