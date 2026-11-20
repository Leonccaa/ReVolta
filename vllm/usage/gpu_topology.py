# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort, anonymous snapshot of CUDA-visible GPU pairs."""

import contextlib
import json
from typing import Any

import torch


def _unknown_pairs(count: int) -> list[dict[str, Any]]:
    return [
        {"gpus": [i, j], "type": "unknown"}
        for i in range(count)
        for j in range(i + 1, count)
    ]


def _bus_id(info: Any) -> str:
    value = info.busId
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return value.rstrip("\x00").lower()


def collect_gpu_topology(count: int) -> str | None:
    """Return a flat JSON string; never export device identities or raw NVML data.

    The CUDA runtime UUID maps each logical ordinal to NVML, including reordered
    visibility and CUDA's default device ordering. An unmappable ordinal stays
    unknown rather than being guessed from an NVML index.
    """
    if count < 2:
        return None
    pairs = _unknown_pairs(count)
    try:
        from vllm.platforms.cuda import _usage_topology_cache, pynvml

        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - telemetry must not interrupt startup
        return json.dumps(pairs, separators=(",", ":"))

    try:
        handles: list[Any | None] = []
        indices: list[int | None] = []
        buses: list[str | None] = []
        for ordinal in range(count):
            try:
                uuid = str(torch.cuda.get_device_properties(ordinal).uuid)
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                index = int(pynvml.nvmlDeviceGetIndex(handle))
            except Exception:  # noqa: BLE001
                handles.append(None)
                indices.append(None)
                buses.append(None)
                continue
            handles.append(handle)
            indices.append(index)
            try:
                buses.append(_bus_id(pynvml.nvmlDeviceGetPciInfo(handle)))
            except Exception:  # noqa: BLE001
                buses.append(None)

        levels = {
            pynvml.NVML_TOPOLOGY_SINGLE: "PIX",
            pynvml.NVML_TOPOLOGY_MULTIPLE: "PXB",
            pynvml.NVML_TOPOLOGY_HOSTBRIDGE: "PHB",
            pynvml.NVML_TOPOLOGY_NODE: "NODE",
            pynvml.NVML_TOPOLOGY_SYSTEM: "SYS",
        }
        # Count direct links only when all queried slots and active remotes are
        # accounted for.
        for pair in pairs:
            i, j = pair["gpus"]
            left, right = handles[i], handles[j]
            if left is None or right is None:
                continue
            left_index, right_index = indices[i], indices[j]
            assert left_index is not None and right_index is not None
            if left_index == right_index:
                continue
            key = (min(left_index, right_index), max(left_index, right_index))
            cached = _usage_topology_cache.get(key, {})
            level = cached.get("level")
            if level is None:
                with contextlib.suppress(Exception):
                    level = pynvml.nvmlDeviceGetTopologyCommonAncestor(left, right)
            if level in levels:
                pair["path"] = levels[level]

            nvlink = cached.get("nvlink") is True
            if not nvlink:
                with contextlib.suppress(Exception):
                    nvlink = (
                        pynvml.nvmlDeviceGetP2PStatus(
                            left, right, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                        )
                        == pynvml.NVML_P2P_STATUS_OK
                    )

            count_links = 0
            complete = buses[j] is not None
            if complete:
                for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                    try:
                        active = pynvml.nvmlDeviceGetNvLinkState(left, link)
                    except pynvml.NVMLError_InvalidArgument:
                        break
                    except pynvml.NVMLError_NotSupported:
                        # Only slot zero indicates device-wide lack of NVLink.
                        complete = link == 0
                        break
                    except Exception:  # noqa: BLE001
                        complete = False
                        break
                    if active:
                        try:
                            remote = _bus_id(
                                pynvml.nvmlDeviceGetNvLinkRemotePciInfo(left, link)
                            )
                        except Exception:  # noqa: BLE001
                            complete = False
                            break
                        if remote == buses[j]:
                            count_links += 1
                        elif remote not in buses:
                            # An unseen GPU or switch may route to this peer.
                            complete = False
            if count_links:
                nvlink = True
            if nvlink:
                pair["type"] = "NVLink"
                if complete and count_links:
                    pair["nvlink_count"] = count_links
            elif complete and "path" in pair:
                pair["type"] = "PCIe"
    except Exception:  # noqa: BLE001 - retain any already verified pair data
        pass
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
    return json.dumps(pairs, separators=(",", ":"))
