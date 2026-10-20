# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.nvidia.ple_layer as ple_module
from tests.models.qwen4_exp.test_ple import (
    _make_disk_ngram_embedding_for_load_test,
    _make_fp8_ngram_embedding_for_load_test,
)
from tests.utils import set_lazy_env
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
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


@pytest.fixture
def file_backed():
    # The PLE offload worker of the cascade or disk lane keeps the shards as
    # mapped checkpoint files. It runs with TP=1 and owns the whole table.
    module = _make_disk_ngram_embedding_for_load_test()
    module.ngram_embedding.shard_indices.org_vocab_start_index = 0
    module.ngram_embedding.shard_indices.org_vocab_end_index = 8
    return module


def mapped(tmp_path, weights):
    """Checkpoint tensors as the loader hands them out: mapped from a file."""
    path = str(tmp_path / "ple.safetensors")
    save_file(dict(weights), path)
    with safe_open(path, framework="pt") as checkpoint:
        return [(name, checkpoint.get_tensor(name)) for name, _ in weights]


def test_file_backed_complete_checkpoint_passes(file_backed, tmp_path):
    file_backed.load_weights(mapped(tmp_path, [shard(0), shard(1), SCALE]))
    finalize(file_backed)


def test_file_backed_missing_shard_is_still_rejected(file_backed, tmp_path):
    with pytest.raises(RuntimeError, match=r"did not load shards: \[1\]"):
        file_backed.load_weights(mapped(tmp_path, [shard(0), SCALE]))
    # The recorded shard does not cover for the missing one after loading.
    with pytest.raises(RuntimeError, match="row coverage.*starting at 4"):
        finalize(file_backed)


def test_file_backed_scale_must_come_from_checkpoint(file_backed, tmp_path):
    file_backed.load_weights(mapped(tmp_path, [shard(0), shard(1)]))
    with pytest.raises(RuntimeError, match="weight_scale"):
        finalize(file_backed)


def test_cascade_worker_table_passes_post_load_processing(monkeypatch, tmp_path):
    # The real worker-side table: meta VocabParallelEmbedding with the FP8
    # method, post-processed like the PLE offload worker does after loading.
    for name in ("VLLM_PLE_CPU_OFFLOAD", "VLLM_PLE_DISK_OFFLOAD"):
        set_lazy_env(monkeypatch, name, None)
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", "0")
    set_lazy_env(monkeypatch, "VLLM_PLE_DISK_OFFLOAD_NUM_THREADS", "1")
    for target in (embedding_module, parameter_module):
        monkeypatch.setattr(target, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(target, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ple_module, "is_offload_process", lambda: True)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.kernel_config.ple_disk_cascade_active = True
    text_config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=8,
        eos_token_id=2,
        vocab_size=64,
        split_ngram_parts=2,
        seed=None,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=128,
        ple_embedding_dtype="float8_e4m3fn",
        ple_offload_embedding=False,
    )
    with set_current_vllm_config(config):
        layer = Qwen4ExpNGramEmbedding(
            text_config,
            embedding_dim=256,
            ple_dense_layer_id=0,
            max_total_tokens=8,
            max_num_reqs=2,
            prefix="model.layers.2.ple.ple_embedding",
            layer_name="model.layers.2.ple",
            params_dtype=torch.float16,
        )
        try:
            assert layer._file_backed_shards
            embedding = layer.ngram_embedding
            assert embedding.weight.is_meta
            rows, width = embedding.org_vocab_size, embedding.embedding_dim
            shard_rows = (rows + 1) // 2
            weights = [
                (
                    f"ngram_embedding.shard_{index}.weight",
                    torch.zeros(min(shard_rows, rows - index * shard_rows), width).to(
                        torch.float8_e4m3fn
                    ),
                )
                for index in range(2)
            ]
            loaded = layer.load_weights(mapped(tmp_path, [*weights, SCALE]))
            assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
            process_weights_after_loading(
                layer, SimpleNamespace(quantization=None), torch.device("cpu")
            )
        finally:
            layer._disk_executor.shutdown()
