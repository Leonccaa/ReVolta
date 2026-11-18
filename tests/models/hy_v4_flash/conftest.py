# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state


@pytest.fixture
def tp1(monkeypatch):
    """Single-rank collectives are identity operations; no process group needed."""
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
    )
    for name in ("_TP", "_DP", "_PCP", "_PP"):
        monkeypatch.setattr(parallel_state, name, group)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config
