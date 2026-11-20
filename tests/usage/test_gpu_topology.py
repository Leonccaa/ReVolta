# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only topology probes, with NVML and CUDA identities replaced."""

import json
import sys
from types import SimpleNamespace

import pytest

from vllm.usage import gpu_topology


class InvalidLink(Exception):
    pass


class UnsupportedLink(Exception):
    pass


class FakeNvml:
    NVML_TOPOLOGY_SINGLE = 10
    NVML_TOPOLOGY_MULTIPLE = 20
    NVML_TOPOLOGY_HOSTBRIDGE = 30
    NVML_TOPOLOGY_NODE = 40
    NVML_TOPOLOGY_SYSTEM = 50
    NVML_P2P_CAPS_INDEX_NVLINK = 0
    NVML_P2P_STATUS_OK = 0
    NVML_NVLINK_MAX_LINKS = 4
    NVMLError_InvalidArgument = InvalidLink
    NVMLError_NotSupported = UnsupportedLink

    def __init__(self, links, levels=None):
        self.links = links
        self.levels = levels or {}
        self.p2p_queries = []
        self.topology_queries = []

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByUUID(self, uuid):
        return int(uuid.rsplit("-", 1)[1])

    def nvmlDeviceGetIndex(self, handle):
        return handle

    def nvmlDeviceGetPciInfo(self, handle):
        return SimpleNamespace(busId=f"physical-bus-{handle}")

    def nvmlDeviceGetTopologyCommonAncestor(self, a, b):
        self.topology_queries.append((a, b))
        return self.levels.get(tuple(sorted((a, b))), 10)

    def nvmlDeviceGetP2PStatus(self, a, b, capability):
        self.p2p_queries.append((a, b))
        return 0 if self.links.get(tuple(sorted((a, b))), 0) else 1

    def nvmlDeviceGetNvLinkState(self, handle, link):
        remotes = [
            peer
            for pair, count in self.links.items()
            if handle in pair
            for peer in pair
            if peer != handle
            for _ in range(count)
        ]
        if link >= len(remotes):
            raise InvalidLink()
        return True

    def nvmlDeviceGetNvLinkRemotePciInfo(self, handle, link):
        remotes = [
            peer
            for pair, count in self.links.items()
            if handle in pair
            for peer in pair
            if peer != handle
            for _ in range(count)
        ]
        return self.nvmlDeviceGetPciInfo(remotes[link])


@pytest.fixture
def fake_probe(monkeypatch):
    def setup(links, visible=(2, 0, 1), levels=None):
        nvml = FakeNvml(links, levels)
        cuda = SimpleNamespace(pynvml=nvml, _usage_topology_cache={})
        monkeypatch.setitem(sys.modules, "vllm.platforms.cuda", cuda)
        monkeypatch.setattr(
            gpu_topology.torch.cuda,
            "get_device_properties",
            lambda ordinal: SimpleNamespace(uuid=f"GPU-secret-{visible[ordinal]}"),
        )
        return nvml, cuda

    return setup


def test_all_nvlink_and_reordered_visibility(fake_probe):
    fake_probe({(0, 1): 2, (0, 2): 1, (1, 2): 2})
    raw = gpu_topology.collect_gpu_topology(3)
    pairs = json.loads(raw)
    assert [(p["gpus"], p["nvlink_count"]) for p in pairs] == [
        ([0, 1], 1),
        ([0, 2], 2),
        ([1, 2], 2),
    ]
    assert all(p["type"] == "NVLink" and p["path"] == "PIX" for p in pairs)
    assert "secret" not in raw and "bus" not in raw and "serial" not in raw


def test_partial_nvlink_and_pcie(fake_probe):
    fake_probe({(0, 2): 2}, levels={(0, 1): 30})
    pairs = json.loads(gpu_topology.collect_gpu_topology(3))
    assert [(p["type"], p["path"]) for p in pairs] == [
        ("NVLink", "PIX"),
        ("PCIe", "PIX"),
        ("PCIe", "PHB"),
    ]


@pytest.mark.parametrize("error", [InvalidLink, UnsupportedLink])
def test_pcie_only(fake_probe, monkeypatch, error):
    nvml, _ = fake_probe({}, levels={(0, 1): 20})

    def link_state(handle, link):
        raise error()

    monkeypatch.setattr(nvml, "nvmlDeviceGetNvLinkState", link_state)
    raw = gpu_topology.collect_gpu_topology(3)
    assert raw is not None
    assert json.loads(raw) == [
        {"gpus": [0, 1], "type": "PCIe", "path": "PIX"},
        {"gpus": [0, 2], "type": "PCIe", "path": "PIX"},
        {"gpus": [1, 2], "type": "PCIe", "path": "PXB"},
    ]
    assert "physical-bus" not in raw


def test_probe_failure_and_single_gpu(fake_probe, monkeypatch):
    nvml, _ = fake_probe({})
    monkeypatch.setattr(nvml, "nvmlInit", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert gpu_topology.collect_gpu_topology(1) is None
    assert json.loads(gpu_topology.collect_gpu_topology(2)) == [
        {"gpus": [0, 1], "type": "unknown"}
    ]


@pytest.mark.parametrize(
    ("failed_link", "error"),
    [(0, RuntimeError), (1, RuntimeError), (1, UnsupportedLink)],
)
def test_incomplete_link_query_does_not_claim_pcie(
    fake_probe, monkeypatch, failed_link, error
):
    nvml, _ = fake_probe({})

    def link_state(handle, link):
        if link == failed_link:
            raise error()
        return False

    monkeypatch.setattr(nvml, "nvmlDeviceGetNvLinkState", link_state)
    pair = json.loads(gpu_topology.collect_gpu_topology(2))[0]
    assert pair == {"gpus": [0, 1], "type": "unknown", "path": "PIX"}


def test_unseen_nvlink_peer_and_duplicate_device_stay_unknown(fake_probe):
    fake_probe({(2, 3): 1}, visible=(2, 0))
    pair = json.loads(gpu_topology.collect_gpu_topology(2))[0]
    assert pair["type"] == "unknown"
    fake_probe({}, visible=(2, 2))
    pair = json.loads(gpu_topology.collect_gpu_topology(2))[0]
    assert pair == {"gpus": [0, 1], "type": "unknown"}


def test_reuses_startup_pair_results(fake_probe):
    nvml, cuda = fake_probe({(0, 2): 1}, visible=(2, 0))
    cuda._usage_topology_cache[(0, 2)] = {"nvlink": True, "level": 20}
    pair = json.loads(gpu_topology.collect_gpu_topology(2))[0]
    assert pair["type"] == "NVLink" and pair["path"] == "PXB"
    assert nvml.p2p_queries == [] and nvml.topology_queries == []
