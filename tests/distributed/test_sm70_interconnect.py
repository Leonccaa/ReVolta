# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only topology, fresh IPC, and group-consensus regression tests."""

import os
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.platforms.cuda as cuda_platform
from vllm.distributed.device_communicators import all_reduce_utils as aru
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.platforms.interface import DeviceCapability

from .test_sm70_pcie_switch_custom_all_reduce import PCIE_SWITCH_TOPOLOGY, _FakeNvml


@pytest.fixture
def topology(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_INTERCONNECT", raising=False)
    monkeypatch.delenv("VLLM_SM70_PCIE_SWITCH_CUSTOM_AR", raising=False)
    monkeypatch.delenv("VLLM_SM70_PCIE_SWITCH_CAR_MAX_BYTES", raising=False)
    monkeypatch.delenv("VLLM_SM70_PCIE_SWITCH_CAR_GRAPH_ONLY", raising=False)
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", "1")
    monkeypatch.setenv("VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR", "0")
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(car.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        car.current_platform,
        "is_fully_connected",
        cuda_platform.NvmlCudaPlatform.is_fully_connected,
    )
    monkeypatch.setattr(
        car.current_platform,
        "is_pcie_switch_connected",
        cuda_platform.NvmlCudaPlatform.is_pcie_switch_connected,
        raising=False,
    )
    nvml = _FakeNvml(PCIE_SWITCH_TOPOLOGY)
    monkeypatch.setattr(cuda_platform, "pynvml", nvml)
    driver = Mock(return_value=True)
    fresh = Mock(return_value=True)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", driver)
    monkeypatch.setattr(car, "gpu_p2p_access_check_fresh", fresh, raising=False)

    def gather(output, value, group):
        # Model all ranks agreeing; only rank zero produces fresh IPC results.
        output[:] = (
            [value] * len(output)
            if value is not None
            else [True] + [None] * (len(output) - 1)
        )

    monkeypatch.setattr(car.dist, "all_gather_object", gather)
    return SimpleNamespace(nvml=nvml, driver=driver, fresh=fresh)


def resolve(*, rank=0, size=4, cap=(7, 0), physical=None, visible=None, disabled=False):
    return car._resolve_interconnect(
        "cpu-group",
        rank,
        size,
        DeviceCapability(*cap),
        list(range(size)) if physical is None else physical,
        list(range(size)) if visible is None else visible,
        disabled,
    )


def test_pex_auto_validates_all_edges_despite_skip_flag(topology):
    decision = resolve()
    assert decision.mode == "pex_p2p" and decision.fully_connected
    assert decision.max_bytes == 81920 and decision.graph_only
    assert topology.driver.call_args_list == [((0, 1),), ((0, 2),), ((0, 3),)]
    topology.fresh.assert_called_once_with([0, 1, 2, 3])
    assert decision.nvml == "NVLink=False; PIX/PXB=True"
    assert "fresh IPC all directed peers=passed" in decision.p2p


@pytest.mark.parametrize("peer", [1, 2, 3])
def test_driver_failure_on_any_peer_disables_whole_group(topology, peer):
    topology.driver.side_effect = lambda src, tgt: tgt != peer
    decision = resolve()
    assert decision.mode == "no_p2p" and not decision.fully_connected
    assert topology.driver.call_count == 3
    topology.fresh.assert_not_called()


def test_driver_true_but_real_ipc_false(topology):
    topology.fresh.return_value = False
    assert resolve().mode == "no_p2p"


def test_real_probe_error_falls_back(topology):
    topology.fresh.side_effect = RuntimeError("probe failed")
    assert resolve().mode == "no_p2p"


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_only_rank_zero_runs_fresh_probe(topology, rank):
    assert resolve(rank=rank).mode == "pex_p2p"
    topology.fresh.assert_not_called()
    assert topology.driver.call_count == 3


@pytest.mark.parametrize("size", [4, 8])
def test_nvlink_mesh_keeps_upstream_dispatch(topology, size):
    # At TP8, NVML P2P reachability remains authoritative. This does not
    # claim that all links have equal width or that a DGX-1 is a uniform mesh.
    topology.nvml.nvlink = True
    decision = resolve(size=size)
    assert decision.mode == "nvlink" and decision.fully_connected
    assert decision.max_bytes == 0 and not decision.graph_only
    assert topology.nvml.queries == 0
    topology.fresh.assert_not_called()


def test_tp8_mixed_cube_uses_upstream_nvml_result(topology, monkeypatch):
    # NV1/NV2 widths are not a new policy input. If one pair lacks NVLink
    # reachability, upstream rejects ordinary TP8 custom AR; PEX TP4 must
    # never be used to turn this into a synthetic fully-connected mesh.
    original = topology.nvml.nvmlDeviceGetP2PStatus
    topology.nvml.nvlink = True

    def status(src, tgt, capability):
        return 1 if (src, tgt) == (0, 7) else original(src, tgt, capability)

    monkeypatch.setattr(topology.nvml, "nvmlDeviceGetP2PStatus", status)
    assert resolve(size=8).mode == "no_p2p"
    topology.fresh.assert_not_called()


def test_tp8_opt_in_hierarchy_still_uses_required_peer_edges(topology, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR", "1")
    assert resolve(size=8).mode == "tp8_hierarchical"
    assert topology.driver.call_args_list == [
        ((0, 1),),
        ((0, 2),),
        ((0, 3),),
        ((0, 4),),
    ]
    topology.fresh.assert_not_called()


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("auto", "pex_p2p"),
        ("pex", "pex_p2p"),
        ("nccl", "no_p2p"),
        ("nvlink", "nvlink"),
    ],
)
@pytest.mark.parametrize("legacy", ["auto", "0", "1"])
def test_override_precedence(topology, monkeypatch, requested, expected, legacy):
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    monkeypatch.setenv("VLLM_SM70_PCIE_SWITCH_CUSTOM_AR", legacy)
    if requested == "auto" and legacy == "0":
        expected = "no_p2p"
    assert resolve().mode == expected


@pytest.mark.parametrize("requested", ["auto", "pex", "nccl", "nvlink"])
def test_cli_disable_wins_over_every_mode(topology, monkeypatch, requested):
    topology.nvml.nvlink = True
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    assert resolve(disabled=True).mode == "no_p2p"
    topology.driver.assert_not_called()
    topology.fresh.assert_not_called()


@pytest.mark.parametrize("requested", ["pex", "nvlink"])
def test_force_still_checks_driver(topology, monkeypatch, requested):
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    topology.driver.return_value = False
    assert resolve().mode == "no_p2p"


@pytest.mark.parametrize("requested", ["pex", "nvlink"])
def test_force_mode_allows_missing_nvml(topology, monkeypatch, requested):
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    monkeypatch.setattr(
        car.current_platform, "is_fully_connected", Mock(side_effect=RuntimeError)
    )
    assert resolve().mode == ("pex_p2p" if requested == "pex" else "nvlink")
    assert topology.driver.call_count == 3


def test_force_pex_bypasses_topology_but_not_fresh_probe(topology, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", "pex")
    topology.nvml.matrix = [[30] * 4 for _ in range(4)]
    decision = resolve()
    assert decision.mode == "pex_p2p"
    assert "PIX/PXB=bypassed" in decision.nvml
    topology.fresh.assert_called_once()


@pytest.mark.parametrize("requested", ["pex", "nccl", "nvlink"])
@pytest.mark.parametrize("size,cap", [(2, (7, 0)), (8, (7, 0)), (4, (8, 0))])
def test_new_override_is_scoped_to_sm70_tp4(
    topology, monkeypatch, requested, size, cap
):
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    expected = "upstream" if size == 2 else "no_p2p"
    assert resolve(size=size, cap=cap).mode == expected
    topology.fresh.assert_not_called()


@pytest.mark.parametrize("requested", ["", "pcie", "1", "yes"])
def test_bad_override_is_rejected(topology, monkeypatch, requested):
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    with pytest.raises(ValueError, match="VLLM_SM70_INTERCONNECT"):
        resolve()


@pytest.mark.parametrize("mismatch", ["mode", "graph_only", "max_bytes"])
def test_rank_policy_mismatch_uses_nccl(topology, monkeypatch, mismatch):
    def gather(output, value, group):
        overrides = {"mode": "nvlink", "graph_only": False, "max_bytes": 16384}
        output[:] = [value] * len(output)
        output[3] = replace(value, **{mismatch: overrides[mismatch]})

    monkeypatch.setattr(car.dist, "all_gather_object", gather)
    decision = resolve()
    assert decision.mode == "no_p2p"
    assert "rank policy mismatch" in decision.reason
    topology.driver.assert_not_called()
    topology.fresh.assert_not_called()


def test_rank_driver_disagreement_uses_nccl(topology, monkeypatch):
    def gather(output, value, group):
        output[:] = [value] * len(output)
        if isinstance(value, bool):
            output[3] = False

    monkeypatch.setattr(car.dist, "all_gather_object", gather)
    assert resolve().mode == "no_p2p"
    topology.fresh.assert_not_called()


def test_rank_devices_are_mapped_to_cuda_ordinals(topology):
    topology.nvml.matrix = [[20] * 8 for _ in range(8)]
    assert resolve(physical=[7, 3, 5, 1], visible=[1, 3, 5, 7]).mode == "pex_p2p"
    topology.fresh.assert_called_once_with([3, 1, 2, 0])
    assert topology.driver.call_args_list == [((3, 1),), ((3, 2),), ((3, 0),)]


def test_unmappable_rank_device_falls_back(topology):
    assert resolve(visible=[0, 1]).mode == "no_p2p"
    topology.fresh.assert_not_called()


@pytest.mark.parametrize("skip", ["0", "1"])
def test_can_p2p_checks_every_peer(topology, monkeypatch, skip):
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", skip)
    check = Mock(side_effect=lambda src, tgt: tgt != 3)
    monkeypatch.setattr(car, "gpu_p2p_access_check", check)
    topology.driver.side_effect = check
    assert not car._can_p2p(0, 4)
    assert check.call_count == 3


def test_fresh_probe_ignores_disk_and_memory_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(aru, "_gpu_p2p_access_cache", {"0->1": True})
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    (tmp_path / "gpu_p2p_access_cache_for_0,1.json").write_text('{"0->1":true}')
    calls = []

    class Process:
        returncode = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, payload, timeout):
            src, tgt, output = aru.pickle.loads(payload)
            calls.append((src, tgt))
            with open(output, "wb") as file:
                aru.pickle.dump([True, False], file)
            return b"", b""

    monkeypatch.setattr(aru.subprocess, "Popen", Process)
    assert not aru.gpu_p2p_access_check_fresh([0, 1])
    assert calls == [((0, 1), (1, 0))]
    assert aru._gpu_p2p_access_cache == {"0->1": True}


@pytest.mark.parametrize(
    "mode,disabled,expected",
    [
        ("pex_p2p", False, "pex_p2p"),
        ("nvlink", False, "nvlink"),
        ("pex_p2p", True, "no_p2p"),
        ("no_p2p", True, "no_p2p"),
    ],
)
def test_profile_query_returns_confirmed_mode(mode, disabled, expected):
    comm = SimpleNamespace(interconnect_mode=mode, disabled=disabled)
    assert car.get_sm70_interconnect_mode(comm) == expected
    assert car.get_sm70_interconnect_mode(None) == "no_p2p"


def _gloo_worker(rank, rendezvous, mismatch):
    # Forked CPU workers inherit fake NVML/driver; only the consensus uses real
    # Gloo collectives. A timeout makes rank divergence a bounded test failure.
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=15),
    )
    try:
        if rank == 3:
            if mismatch == "topology":
                car.current_platform.is_pcie_switch_connected = lambda ids: False
            elif mismatch == "invalid-env":
                os.environ["VLLM_SM70_INTERCONNECT"] = "typo"
            else:
                torch.cuda.can_device_access_peer = lambda src, tgt: False
        decision = car._resolve_interconnect(
            dist.group.WORLD,
            rank,
            4,
            DeviceCapability(7, 0),
            list(range(4)),
            list(range(4)),
        )
        assert decision.mode == "no_p2p"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", ["topology", "driver", "invalid-env"])
def test_real_gloo_group_falls_back_without_hanging(
    topology, monkeypatch, tmp_path, mismatch
):
    # Restore the actual collective, replaced by the fixture for other tests.
    monkeypatch.undo()
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", "auto")
    monkeypatch.setenv("VLLM_SM70_PCIE_SWITCH_CUSTOM_AR", "auto")
    monkeypatch.setenv("VLLM_SKIP_P2P_CHECK", "1")
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(car.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(car.current_platform, "is_fully_connected", lambda ids: False)
    monkeypatch.setattr(
        car.current_platform,
        "is_pcie_switch_connected",
        lambda ids: True,
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda src, tgt: True)
    monkeypatch.setattr(
        car,
        "gpu_p2p_access_check_fresh",
        Mock(side_effect=AssertionError("must not probe on mismatch")),
        raising=False,
    )
    mp.start_processes(
        _gloo_worker,
        args=(f"file://{tmp_path / 'gloo'}", mismatch),
        nprocs=4,
        start_method="fork",
    )


@pytest.mark.parametrize(
    "requested,enabled,pex",
    [
        ("auto", True, True),
        ("nvlink", True, False),
        ("nccl", False, False),
    ],
)
def test_constructor_uses_confirmed_policy_before_native_init(
    topology,
    monkeypatch,
    requested,
    enabled,
    pex,
):
    monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    monkeypatch.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE", "0")
    monkeypatch.setattr(car, "custom_ar", True)
    monkeypatch.setattr(car, "in_the_same_node_as", lambda *args, **kwargs: [True] * 4)
    monkeypatch.setattr(car.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        car.current_platform, "get_device_capability", lambda: DeviceCapability(7, 0)
    )
    monkeypatch.setattr(car.current_platform, "device_count", lambda: 4)
    monkeypatch.setattr(car.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(car.dist, "get_rank", lambda **kwargs: 0)
    monkeypatch.setattr(car.dist, "get_world_size", lambda **kwargs: 4)

    def gather_ids(output, value, group):
        for i, tensor in enumerate(output):
            tensor.fill_(i)

    monkeypatch.setattr(car.dist, "all_gather", gather_ids)
    allocate = Mock(return_value=[0] * 4)
    native = Mock(return_value=0)
    monkeypatch.setattr(car.CustomAllreduce, "create_shared_buffer", allocate)
    monkeypatch.setattr(car.ops, "meta_size", lambda: 0)
    monkeypatch.setattr(car.ops, "init_custom_ar", native)
    monkeypatch.setattr(car.ops, "register_buffer", Mock())
    original_empty = torch.empty
    monkeypatch.setattr(
        car.torch,
        "empty",
        lambda *args, **kwargs: original_empty(*args, **{**kwargs, "device": "cpu"}),
    )
    log = Mock()
    monkeypatch.setattr(car.logger, "info", log)
    comm = car.CustomAllreduce("cpu-group", device=0)
    assert comm.disabled is not enabled
    assert comm.sm70_pcie_switch is pex
    # The SM70 TP4 NVLink sum2 route is resolved for the NVLink mode only;
    # PCIe-switch and no_p2p groups keep the routes they already had.
    assert comm.sm70_nvlink_tp4 is (requested == "nvlink")
    assert (
        car.get_sm70_interconnect_mode(comm)
        == {
            "auto": "pex_p2p",
            "nvlink": "nvlink",
            "nccl": "no_p2p",
        }[requested]
    )
    if enabled:
        assert native.call_args.args[-1] is True
        assert allocate.call_count == 2
        if pex:
            assert comm.sm70_pcie_switch_max_bytes == 81920
            assert comm.sm70_pcie_switch_graph_only
    else:
        allocate.assert_not_called()
        native.assert_not_called()
    assert any("SM70 interconnect=" in call.args[0] for call in log.call_args_list)


def test_stalled_fresh_probe_terminates_its_worker_group(monkeypatch):
    process = Mock(pid=1234)
    process.communicate.side_effect = [
        aru.subprocess.TimeoutExpired("probe", 60),
        (b"", b""),
    ]
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=process)
    factory.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(aru.subprocess, "Popen", factory)
    kill = Mock()
    monkeypatch.setattr(aru.os, "killpg", kill)
    with pytest.raises(RuntimeError, match="timed out"):
        aru.gpu_p2p_access_check_fresh([0, 1])
    kill.assert_called_once_with(1234, aru.signal.SIGKILL)
    assert process.communicate.call_count == 2
