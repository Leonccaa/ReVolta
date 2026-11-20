# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.layers import utils as layer_utils
from vllm.platforms import current_platform
from vllm.platforms.interface import PlatformEnum


@pytest.fixture
def tp1(monkeypatch):
    """Single-rank collectives are identity operations; no process group needed."""
    group = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        is_first_rank=True,
        is_last_rank=True,
        gather=lambda tensor, *args, **kwargs: tensor,
        all_gather=lambda tensor, *args, **kwargs: tensor,
        all_reduce=lambda tensor: tensor,
    )
    for name in ("_TP", "_DP", "_PCP", "_PP"):
        monkeypatch.setattr(parallel_state, name, group)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


@pytest.fixture
def cpu_vllm(tp1, monkeypatch):
    # The source checkout has no CPU extension or installed platform tag.
    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CPU)
    monkeypatch.setattr(
        layer_utils, "cpu_unquantized_gemm", layer_utils.default_unquantized_gemm
    )
    tp1.kernel_config.ir_op_priority.rms_norm = ["native"]
    tp1.kernel_config.ir_op_priority.fused_add_rms_norm = ["native"]
    return tp1
