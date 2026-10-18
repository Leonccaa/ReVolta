# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

from unittest.mock import patch

import pytest
import torch

ple = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ple_nvfp4")
from vllm.model_executor.layers.quantization.utils.fp4_local import (
    dequantize_nvfp4_cpu,
)


@pytest.mark.parametrize("rank", range(4))
def test_sharded_packed_rows_and_scale_loading(rank):
    import vllm.model_executor.layers.vocab_parallel_embedding as embeddings
    import vllm.model_executor.parameter as parameters
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.models.qwen4_exp.nvidia import ple_layer

    with (
        patch.object(embeddings, "get_tensor_model_parallel_rank", return_value=rank),
        patch.object(
            embeddings, "get_tensor_model_parallel_world_size", return_value=4
        ),
        patch.object(parameters, "get_tensor_model_parallel_rank", return_value=rank),
        patch.object(
            parameters, "get_tensor_model_parallel_world_size", return_value=4
        ),
        patch.object(ple_layer, "_pin_host_tensor_exact"),
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
    ):
        layer = ple.NVFP4PinnedPLEEmbedding(32, 32, torch.float16, 4, "ple.table")
        assert layer.storage is None
        assert layer.weight.device.type == "cpu" and layer.weight._vllm_keep_on_cpu
        assert layer.row_bytes == 18
        weight = torch.arange(32 * 16, dtype=torch.int64).byte().reshape(32, 16)
        raw = torch.full((32, 2), 56, dtype=torch.uint8)
        scale = raw.view(torch.float8_e4m3fn)
        for index in reversed(range(3)):
            lo, hi = index * 11, min((index + 1) * 11, 32)
            # Components intentionally arrive scale first, across TP boundaries.
            for field, tensor in (("weight_scale", scale), ("weight", weight)):
                layer.load_checkpoint_tensor(
                    f"ngram_embedding.shard_{index}.{field}", tensor[lo:hi], 3
                )
        layer.load_checkpoint_tensor(
            "ngram_embedding.weight_scale_2", torch.tensor(0.125), 3
        )
        layer.check_loaded(3)
        assert layer.storage is not None
        assert layer.storage.shape == (8, 18)
        assert torch.equal(layer.storage[:, :16], weight[rank * 8 : (rank + 1) * 8])
        assert torch.equal(layer.storage[:, 16:], raw[rank * 8 : (rank + 1) * 8])
        actual = dequantize_nvfp4_cpu(
            layer.storage[:, :16],
            layer.storage[:, 16:].contiguous().view(torch.float8_e4m3fn),
            layer.weight_scale_2,
        )
        expected = dequantize_nvfp4_cpu(weight, scale, torch.tensor(0.125))[
            rank * 8 : (rank + 1) * 8
        ]
        assert torch.equal(actual, expected)
        layer._loaded_parts.remove("global")
        with pytest.raises(ValueError, match="Incomplete"):
            layer.check_loaded(3)


def test_gpu_gather_and_dequant_with_triton_interpreter(monkeypatch):
    # Interpreter exercises both real Triton bodies on CPU, including row
    # order/duplicates, nibble signs, E4M3 subnormals and tail masks.
    import triton
    from triton.runtime.interpreter import InterpretedFunction

    rows, d = 5, 160
    weight = torch.arange(rows * d // 2, dtype=torch.int64).byte().reshape(rows, d // 2)
    scale = (torch.arange(rows * d // 16).reshape(rows, d // 16) % 127).byte()
    storage = torch.cat((weight, scale), -1).contiguous()
    ids = torch.tensor([4, 0, 4, 2], dtype=torch.int64)
    gathered = torch.empty((4, d // 2 + d // 16), dtype=torch.uint8)
    out = torch.empty((4, d), dtype=torch.float16)
    # Pointer-as-int UVA is wrapped by the interpreter for the gather call.
    gather = InterpretedFunction(ple._gather_rows.fn)
    dequant = InterpretedFunction(ple._dequant_rows.fn)
    gather[(4,)](
        storage,
        ids,
        gathered,
        storage.shape[1],
        triton.next_power_of_2(storage.shape[1]),
    )
    dequant[(4,)](gathered, torch.tensor(0.037), out, d, 256)
    expected = dequantize_nvfp4_cpu(
        weight[ids], scale[ids].view(torch.float8_e4m3fn), torch.tensor(0.037)
    )
    assert torch.equal(out, expected)


def test_process_offload_rejected_before_placeholder_can_skip_constructor():
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import ple_layer

    with (
        patch.object(ple_layer, "ple_offload_enabled", return_value=True),
        pytest.raises(NotImplementedError, match="resident pinned-host"),
    ):
        ple_layer.Qwen4ExpPLELayer(
            SimpleNamespace(ple_embedding_dtype="nvfp4"), SimpleNamespace()
        )
