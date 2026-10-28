# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 PCIe-switch TP4 custom all-reduce mode (CPU-only checks).

Four V100s under one PCIe switch (nvidia-smi PIX/PXB) are treated as fully
connected for custom all-reduce, limited to small graph-captured payloads.
Every other topology must keep the upstream rule.
"""

import pytest
import torch

import vllm.platforms.cuda as cuda_platform
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.platforms.interface import DeviceCapability

PIX, PXB, PHB, NODE, SYS = 10, 20, 30, 40, 50
# Four-GPU PCIe-switch topology observed with nvidia-smi, 2026-10-15.
PCIE_SWITCH_TOPOLOGY = [
    [0, PIX, PXB, PXB],
    [PIX, 0, PXB, PXB],
    [PXB, PXB, 0, PXB],
    [PXB, PXB, PXB, 0],
]


class _FakeNvml:
    NVML_TOPOLOGY_MULTIPLE = PXB
    NVML_P2P_CAPS_INDEX_NVLINK = 4
    NVML_P2P_STATUS_OK = 0

    class NVMLError(Exception):
        pass

    def __init__(self, matrix, fail=False, nvlink=False):
        self.matrix = matrix
        self.fail = fail
        self.nvlink = nvlink
        self.queries = 0

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetTopologyCommonAncestor(self, a, b):
        self.queries += 1
        if self.fail:
            raise self.NVMLError()
        return self.matrix[a][b]

    def nvmlDeviceGetP2PStatus(self, a, b, capability):
        assert capability == self.NVML_P2P_CAPS_INDEX_NVLINK
        if self.fail:
            raise self.NVMLError()
        return self.NVML_P2P_STATUS_OK if self.nvlink else 1


def _with_level(level):
    matrix = [row[:] for row in PCIE_SWITCH_TOPOLOGY]
    matrix[1][3] = matrix[3][1] = level
    return matrix


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (PCIE_SWITCH_TOPOLOGY, True),
        (_with_level(PHB), False),
        (_with_level(NODE), False),
        (_with_level(SYS), False),
    ],
)
def test_is_pcie_switch_connected(monkeypatch, matrix, expected):
    monkeypatch.setattr(cuda_platform, "pynvml", _FakeNvml(matrix))
    assert (
        cuda_platform.NvmlCudaPlatform.is_pcie_switch_connected([0, 1, 2, 3])
        is expected
    )


def test_is_pcie_switch_connected_nvml_error(monkeypatch):
    monkeypatch.setattr(cuda_platform, "pynvml", _FakeNvml(PCIE_SWITCH_TOPOLOGY, True))
    assert not cuda_platform.NvmlCudaPlatform.is_pcie_switch_connected([0, 1])


def test_non_nvml_platform_never_reports_pcie_switch():
    assert not cuda_platform.CudaPlatformBase.is_pcie_switch_connected([0, 1, 2, 3])


def _mode(monkeypatch, *, mode=None, world_size=4, capability=(7, 0), switch=True):
    if mode is None:
        monkeypatch.delenv("VLLM_SM70_PCIE_SWITCH_CUSTOM_AR", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_PCIE_SWITCH_CUSTOM_AR", mode)
    calls: list[list[int]] = []

    def is_pcie_switch_connected(ids: list[int]) -> bool:
        calls.append(ids)
        return switch

    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        car.current_platform,
        "is_pcie_switch_connected",
        is_pcie_switch_connected,
        raising=False,
    )
    cap = None if capability is None else DeviceCapability(*capability)
    result = car._sm70_pcie_switch_tp4_mode(world_size, cap, [0, 1, 2, 3])
    return result, calls


def test_mode_auto_enables_on_sm70_tp4_pcie_switch(monkeypatch):
    result, calls = _mode(monkeypatch)
    assert result and calls == [[0, 1, 2, 3]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"switch": False},
        {"world_size": 2},
        {"world_size": 8},
        {"capability": (8, 0)},
        {"capability": (7, 5)},
        {"capability": None},
        {"mode": "0"},
    ],
)
def test_mode_keeps_upstream_rule_elsewhere(monkeypatch, kwargs):
    result, _ = _mode(monkeypatch, **kwargs)
    assert not result


def test_mode_force_skips_topology_query(monkeypatch):
    result, calls = _mode(monkeypatch, mode="1", switch=False)
    assert result and calls == []


def test_mode_force_still_requires_sm70_tp4(monkeypatch):
    result, _ = _mode(monkeypatch, mode="1", world_size=8)
    assert not result


def test_mode_rejects_unknown_value(monkeypatch):
    with pytest.raises(ValueError):
        _mode(monkeypatch, mode="yes")


def _communicator(
    pcie_switch, max_bytes=81920, graph_only=True, nvlink=False, push_registered=True
):
    communicator = object.__new__(CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = 4
    communicator.fully_connected = True
    communicator.tp8_hierarchical = False
    communicator.dispatch_max_size = 8 * 1024 * 1024
    # The SM70 TP4 NVLink payload table is a separate, mode-scoped gate; set it
    # explicitly so these fakes cannot inherit it by accident.
    communicator.sm70_nvlink_tp4_car_sizes = nvlink
    communicator.sm70_tp4_push_buffer_ptrs = [1, 2, 3, 4] if push_registered else None
    if pcie_switch:
        communicator.interconnect_mode = "pex_p2p"
        communicator.sm70_pcie_switch = True
        communicator.sm70_pcie_switch_max_bytes = max_bytes
        communicator.sm70_pcie_switch_graph_only = graph_only
    else:
        communicator.interconnect_mode = "nvlink" if nvlink else "no_p2p"
    return communicator


def _fp16(nbytes):
    return torch.empty(nbytes // 2, dtype=torch.float16)


def test_pcie_switch_payload_limit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    communicator = _communicator(True)
    assert communicator.should_custom_ar(_fp16(81920))
    assert not communicator.should_custom_ar(_fp16(81920 + 16))


def test_pcie_switch_graph_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not _communicator(True).should_custom_ar(_fp16(1024))
    assert _communicator(True, graph_only=False).should_custom_ar(_fp16(1024))


def test_pcie_switch_zero_limit_means_no_byte_cap(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _communicator(True, max_bytes=0).should_custom_ar(_fp16(1024 * 1024))


def test_instances_outside_the_two_switches_keep_the_upstream_rule(monkeypatch):
    # An instance with neither switch on (TP2/TP8/no_p2p, or any other resolved
    # mode) has no payload and no capture limit, exactly as before. The SM70 TP4
    # NVLink table is a third, separate gate and must be machine-checked off here.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    upstream = _communicator(False)
    assert not upstream.sm70_pcie_switch
    assert upstream.sm70_nvlink_tp4_car_sizes is False
    assert upstream.should_custom_ar(_fp16(1024 * 1024))


def test_nvlink_tp4_payload_table_replaces_the_upstream_byte_rule(monkeypatch):
    # The intent statement that changed: an SM70 TP4 NVLink instance is no
    # longer "no payload limit"; only the table's payloads stay on custom.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    nvlink = _communicator(False, nvlink=True)
    assert not nvlink.sm70_pcie_switch
    assert nvlink.sm70_nvlink_tp4_car_sizes is True
    assert nvlink.should_custom_ar(_fp16(81920))
    assert not nvlink.should_custom_ar(_fp16(1024 * 1024))


def test_env_defaults_match_validated_pcie_switch_settings(monkeypatch):
    # These values match the validated four-V100 PCIe-switch configuration.
    import vllm.envs as envs

    for name in (
        "VLLM_SM70_PCIE_SWITCH_CUSTOM_AR",
        "VLLM_SM70_PCIE_SWITCH_CAR_MAX_BYTES",
        "VLLM_SM70_PCIE_SWITCH_CAR_GRAPH_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert envs.environment_variables["VLLM_SM70_PCIE_SWITCH_CUSTOM_AR"]() == "auto"
    assert envs.environment_variables["VLLM_SM70_PCIE_SWITCH_CAR_MAX_BYTES"]() == 81920
    assert envs.environment_variables["VLLM_SM70_PCIE_SWITCH_CAR_GRAPH_ONLY"]() is True
