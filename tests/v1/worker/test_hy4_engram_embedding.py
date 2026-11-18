# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import vllm.models.hy_v4_flash.nvidia.engram as engram

pytestmark = pytest.mark.skip_global_cleanup


def config(device="cpu"):
    return SimpleNamespace(
        device_config=SimpleNamespace(device_type=device),
        load_config=SimpleNamespace(load_format="safetensors"),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def test_local_safetensors_reads_and_cpu_gather(tmp_path, monkeypatch):
    sizes = [5, 7, 11, 13]
    dim = 4
    torch.manual_seed(10)
    source = torch.randn(sum(sizes), dim, dtype=torch.bfloat16)
    scale = source.abs().amax(-1).clamp(min=1e-12) / 448
    codes = (source / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    name = "table.weight"
    path = tmp_path / "table.safetensors"
    save_file({name: codes, name + "_scale_inv": scale}, path)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: 2)
    accesses = []
    actual_open = engram.safe_open

    class Slice:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getitem__(self, item):
            accesses.append(item)
            return self.wrapped[item]

        def get_dtype(self):
            return self.wrapped.get_dtype()

    class Reader:
        def __init__(self, *args, **kwargs):
            self.wrapped = actual_open(*args, **kwargs)

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def get_slice(self, key):
            return Slice(self.wrapped.get_slice(key))

    monkeypatch.setattr(engram, "safe_open", Reader)
    table = engram.FlashEmbedding(sizes, dim, config(), "table")
    table.load_from_safetensors(str(path), name)
    assert (table.head_start, table.head_end) == (2, 3)
    assert (table.vocab_start, table.vocab_end) == (12, 23)
    assert table.storage.numel() == 11 * (dim + 2)
    assert accesses == [(slice(12, 23), slice(None)), slice(12, 23)]
    hashes = torch.tensor([[1, 6, 12, 24], [1, 6, 22, 24], [-1, -1, -1, -1]])
    actual = table(hashes)
    expected = torch.stack(
        (
            codes[12].float() * scale[12].float(),
            codes[22].float() * scale[22].float(),
            torch.zeros(dim),
        )
    )[:, None, :]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Exercise the 131072-row read boundary without allocating a full model table.
    accesses.clear()
    sizes = [131073] * 4
    big_path = tmp_path / "chunked.safetensors"
    big_codes = torch.zeros((sum(sizes), dim), dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    big_scales = torch.ones(sum(sizes), dtype=torch.bfloat16)
    save_file({name: big_codes, name + "_scale_inv": big_scales}, big_path)
    chunked = engram.FlashEmbedding(sizes, dim, config(), "chunked")
    chunked.load_from_safetensors(str(big_path), name)
    assert accesses == [
        (slice(262146, 393218), slice(None)),
        slice(262146, 393218),
        (slice(393218, 393219), slice(None)),
        slice(393218, 393219),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_gather_matches_cpu(tmp_path, monkeypatch):
    from vllm.forward_context import override_forward_context

    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: 0)
    torch.manual_seed(20)
    source = torch.randn(36, 4, dtype=torch.bfloat16)
    scale = source.abs().amax(-1).clamp(min=1e-12) / 448
    codes = (source / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    path = tmp_path / "table.safetensors"
    save_file({"table.weight": codes, "table.weight_scale_inv": scale}, path)
    cpu = engram.FlashEmbedding([5, 7, 11, 13], 4, config(), "cpu")
    cuda_config = config("cuda")
    gpu = engram.FlashEmbedding([5, 7, 11, 13], 4, cuda_config, "gpu")
    cpu.load_from_safetensors(str(path), "table.weight")
    gpu.load_from_safetensors(str(path), "table.weight")
    ids = torch.tensor([[1, 6, 12, 24], [-1, -1, -1, -1]], dtype=torch.int32)
    ctx = SimpleNamespace(
        no_compile_layers=cuda_config.compilation_config.static_forward_context
    )
    with override_forward_context(ctx):
        actual = gpu(ids.cuda()).cpu()
    torch.testing.assert_close(actual, cpu(ids), rtol=0, atol=0)
