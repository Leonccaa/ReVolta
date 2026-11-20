# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_gdn_norm_follows_parameter_device(monkeypatch, default_vllm_config, device):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear, ple_offload_layer
    from vllm.model_executor.layers.mamba.gdn import base
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as module

    def forbidden(*args, **kwargs):
        pytest.fail("CPU PLE discovery must not select or initialize a GPU")

    for source in (base, linear, parameter):
        monkeypatch.setattr(source, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(source, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ple_offload_layer, "_offload_worker_flag", True)
    monkeypatch.setattr(module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        module.current_platform, "current_device", forbidden, raising=False
    )
    monkeypatch.setattr(module.current_platform, "get_device_capability", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.accelerator, "current_device_index", forbidden)

    config = SimpleNamespace(
        hidden_size=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    with torch.device(device):
        attention = module.QwenGatedDeltaNetAttention(
            config, default_vllm_config, "layers.0"
        )

    assert attention.norm.weight.device == attention.dt_bias.device
    assert attention.norm.weight.device.type == device


def test_offload_worker_fla_import_does_not_probe_cuda():
    root = Path(__file__).resolve().parents[3]
    code = """
from types import SimpleNamespace

import torch
import vllm.platforms
from vllm.model_executor.layers.ple_offload_layer import mark_as_offload_worker
from vllm.platforms.cpu import CpuPlatform
from vllm.triton_utils import triton

vllm.platforms._current_platform = CpuPlatform()
mark_as_offload_worker()

def forbidden(*args, **kwargs):
    raise AssertionError("CPU PLE discovery probed CUDA")

torch.cuda.is_available = lambda: True
torch.cuda.get_device_capability = forbidden
if not hasattr(triton, "runtime"):
    triton.runtime = SimpleNamespace(driver=SimpleNamespace())
triton.runtime.driver.active = SimpleNamespace(get_current_target=forbidden)

from vllm.model_executor.layers.fla.ops import kda, utils

assert utils.get_available_device() == "cpu"
assert not utils.is_sm70()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
