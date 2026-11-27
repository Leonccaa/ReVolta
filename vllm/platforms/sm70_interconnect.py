# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 interconnect policy shared by the frontend and workers."""

from dataclasses import dataclass

from vllm import envs
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


def _sm70_pcie_switch_tp4_mode(
    world_size: int,
    device_capability: DeviceCapability | None,
    physical_device_ids: list[int],
) -> bool:
    """Whether to treat four SM70 GPUs under one PCIe switch as fully connected.

    Upstream enables custom all-reduce for more than two GPUs only over an
    NVLink mesh. Four V100s whose every pair meets below a PCIe switch
    (nvidia-smi PIX/PXB) have working P2P, and there the SM70 TP4 push kernels
    beat NCCL for small graph-captured payloads. Other topologies, world sizes
    and architectures keep the upstream rule.
    """
    mode = envs.VLLM_SM70_PCIE_SWITCH_CUSTOM_AR
    if mode not in ("auto", "0", "1"):
        raise ValueError(
            f"VLLM_SM70_PCIE_SWITCH_CUSTOM_AR must be auto, 0 or 1, got {mode!r}"
        )
    if mode == "0":
        return False
    if (
        not current_platform.is_cuda()
        or world_size != 4
        or device_capability is None
        or (device_capability.major, device_capability.minor) != (7, 0)
    ):
        return False
    if mode == "1":
        return True
    return current_platform.is_pcie_switch_connected(physical_device_ids)


@dataclass(frozen=True)
class _InterconnectDecision:
    mode: str
    fully_connected: bool = False
    source: str = "upstream"
    nvml: str = "not queried"
    p2p: str = "not checked"
    reason: str = ""
    max_bytes: int = 0
    graph_only: bool = False
    hc_ll_requested: bool = False


def _select_interconnect(
    world_size: int,
    capability: DeviceCapability | None,
    physical_device_ids: list[int],
    disable_custom_all_reduce: bool,
) -> _InterconnectDecision:
    """Select a candidate policy; only _resolve_interconnect confirms P2P."""
    requested = envs.VLLM_SM70_INTERCONNECT
    if requested not in ("auto", "pex", "nccl", "nvlink"):
        raise ValueError(
            "VLLM_SM70_INTERCONNECT must be auto, pex, nccl or nvlink, "
            f"got {requested!r}"
        )
    scoped = (
        current_platform.is_cuda()
        and world_size == 4
        and capability == DeviceCapability(7, 0)
    )
    if scoped and (disable_custom_all_reduce or requested == "nccl"):
        return _InterconnectDecision(
            "no_p2p",
            source="disable-custom-all-reduce"
            if disable_custom_all_reduce
            else "env:nccl",
            reason="custom AR disabled by policy",
        )
    try:
        fully_connected = current_platform.is_fully_connected(physical_device_ids)
        nvml = f"NVLink={fully_connected}"
    except Exception:
        if not (scoped and requested in ("pex", "nvlink")):
            raise
        fully_connected = False
        nvml = "NVLink=query failed (explicit override)"
    source = f"env:{requested}" if scoped and requested != "auto" else "auto"
    if scoped and requested == "nvlink":
        # Explicit escape hatch for unavailable NVML. P2P is still mandatory.
        return _InterconnectDecision("nvlink", True, source, nvml)
    if fully_connected and not (scoped and requested == "pex"):
        return _InterconnectDecision("nvlink", True, source, nvml)
    if scoped:
        pex = requested == "pex" or _sm70_pcie_switch_tp4_mode(
            world_size, capability, physical_device_ids
        )
        if requested == "pex" or envs.VLLM_SM70_PCIE_SWITCH_CUSTOM_AR == "1":
            nvml += "; PIX/PXB=bypassed"
        elif envs.VLLM_SM70_PCIE_SWITCH_CUSTOM_AR == "0":
            nvml += "; PIX/PXB=disabled"
        else:
            nvml += f"; PIX/PXB={pex}"
        if pex:
            return _InterconnectDecision(
                "pex_p2p",
                True,
                source,
                nvml,
                max_bytes=envs.VLLM_SM70_PCIE_SWITCH_CAR_MAX_BYTES,
                graph_only=envs.VLLM_SM70_PCIE_SWITCH_CAR_GRAPH_ONLY,
            )
    if (
        world_size == 8
        and capability == DeviceCapability(7, 0)
        and envs.VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR
    ):
        return _InterconnectDecision("tp8_hierarchical", False, "upstream", nvml)
    if world_size <= 2:
        return _InterconnectDecision("upstream", False, "upstream", nvml)
    return _InterconnectDecision(
        "no_p2p",
        source=source,
        nvml=nvml,
        reason="upstream excludes more than two PCIe-only GPUs",
    )
