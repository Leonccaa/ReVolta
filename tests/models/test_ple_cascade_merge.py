# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression adapted from 1CatAI/1Cat-vLLM#1234's resident-row case."""

import pytest
import torch

from tests.utils import set_lazy_env
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers import ple_offload_layer
from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding


class CascadeLayer(PleOffloadLayer):
    @classmethod
    def offload_keeps_local_tables(cls):
        return Qwen4ExpNGramEmbedding.offload_keeps_local_tables()

    def __init__(self):
        super().__init__()

    def forward_impl(self, hidden_states, input_ids):
        remote = self.wait_offloaded_output(hidden_states, input_ids.numel())
        return torch.where(input_ids[:, None] < 5, input_ids[:, None], remote)


@pytest.mark.parametrize("ids", [[4, 5, 6], [1, 2, 4]])
def test_cascade_merges_resident_rows_after_leaving_constructor_config(
    monkeypatch, ids
):
    for name in ("VLLM_PLE_CPU_OFFLOAD", "VLLM_PLE_DISK_OFFLOAD"):
        set_lazy_env(monkeypatch, name, None)
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", "0")
    monkeypatch.setattr(ple_offload_layer, "is_offload_process", lambda: False)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.kernel_config.ple_disk_cascade_active = True
    with set_current_vllm_config(config):
        layer = CascadeLayer()
    assert not layer.offload_keeps_local_tables()
    layer._is_cpu_offloaded = True
    for stale in (0, 99):
        worker = torch.tensor([[i if i >= 5 else stale] for i in ids])
        monkeypatch.setattr(
            layer, "wait_offloaded_output", lambda h, n, worker=worker: worker
        )
        assert torch.equal(
            layer(torch.zeros(len(ids)), torch.tensor(ids)), torch.tensor(ids)[:, None]
        )
