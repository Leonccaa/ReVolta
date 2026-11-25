# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest

import vllm.models.qwen4_exp.common.ple as ple
from tests.utils import set_lazy_env
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPinnedHostEmbedding

GIB = 1 << 30


@pytest.mark.parametrize("controller", ["v2", "v1", "v1-combined"])
def test_cgroup_budget_includes_parent_and_excludes_inactive_file(
    monkeypatch, tmp_path, controller
):
    root = tmp_path / "cgroup"
    mount = tmp_path / "combined mount" if controller == "v1-combined" else root
    child = mount / "worker"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("")
    if controller == "v2":
        membership.write_text("0::/worker\n")
        limit_name, usage_name = "memory.max", "memory.current"
        stat_prefix = ""
    else:
        controllers = "memory,blkio" if controller == "v1-combined" else "memory"
        mount_root = "/containers/parent" if controller == "v1-combined" else "/"
        membership.write_text(f"5:{controllers}:{mount_root.rstrip('/')}/worker\n")
        escaped_mount = str(mount).replace(" ", "\\040")
        mountinfo.write_text(
            f"1 0 0:1 {mount_root} {escaped_mount} rw "
            f"- cgroup cgroup rw,{controllers}\n"
        )
        limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        stat_prefix = "total_"
    groups = [(mount, 128, 6), (child, 140, 2)]
    for directory, limit, current in groups:
        (directory / limit_name).write_text(str(limit * GIB))
        (directory / usage_name).write_text(str(current * GIB))
    assert ple.cgroup_host_memory(root, membership, mountinfo) == (128 * GIB, 122 * GIB)
    read_cgroup = ple.cgroup_host_memory
    monkeypatch.setattr(
        ple, "cgroup_host_memory", lambda: read_cgroup(root, membership, mountinfo)
    )
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 64 * GIB)
    assert ple.preflight_host_memory()[:2] == (128 * GIB, 64 * GIB)

    # The parent includes the child's 70 GiB file cache. Active pages remain
    # in the working set.
    for directory, _, current in groups:
        (directory / usage_name).write_text(str((current + 70) * GIB))
        (directory / "memory.stat").write_text(
            f"{stat_prefix}active_file {GIB}\n{stat_prefix}inactive_file {70 * GIB}\n"
        )
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 192 * GIB)
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", None)
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    cap = ple.check_ple_host_share(
        NS(ple_layer_ids=[1]), 4, cpu_kv_bytes=48 * GIB, prefer_host=True
    )
    assert cap == 16 * GIB + GIB // 2


@pytest.mark.parametrize(
    ("membership_line", "limit_name", "limit"),
    [
        ("0::/\n", "memory.max", "max"),
        ("0::/\n", "memory.max", "invalid"),
        ("0::/\n", "memory.max", None),
        ("5:memory:/\n", "memory.limit_in_bytes", str((1 << 63) - 4096)),
    ],
)
def test_unavailable_cgroup_limit_falls_back_to_meminfo(
    monkeypatch, tmp_path, membership_line, limit_name, limit
):
    if limit is not None:
        (tmp_path / limit_name).write_text(limit)
    membership = tmp_path / "membership"
    membership.write_text(membership_line)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 0 0:1 / {tmp_path} rw - cgroup cgroup rw,memory\n")
    assert ple.cgroup_host_memory(tmp_path, membership, mountinfo) == (None, None)
    monkeypatch.setattr(ple, "cgroup_host_memory", lambda: (None, None))
    monkeypatch.setattr(ple, "total_host_bytes", lambda: 256 * GIB)
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 200 * GIB)
    assert ple.preflight_host_memory() == (256 * GIB, 200 * GIB, "/proc/meminfo")


def test_cgroup_shortage_counts_cpu_kv_pool_once(monkeypatch):
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", "12")
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    monkeypatch.setattr(ple, "cgroup_host_memory", lambda: (128 * GIB, 100 * GIB))
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 400 * GIB)
    warnings = []
    monkeypatch.setattr(
        ple.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    ple.check_ple_host_share(NS(ple_layer_ids=[1]), 4, cpu_kv_bytes=48 * GIB)
    assert len(warnings) == 1
    message = warnings[0]
    assert "cgroup" in message and "48.0" in message and "CPU KV" in message
    assert "100.0" in message and "reserve 8.0" in message


def test_existing_preflight_cannot_pass_on_host_ram_above_container_limit(monkeypatch):
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", "32")
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    monkeypatch.setattr(ple, "total_host_bytes", lambda: 512 * GIB)
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 400 * GIB)
    monkeypatch.setattr(
        ple, "cgroup_host_memory", lambda: (128 * GIB, 120 * GIB), raising=False
    )
    warnings = []
    monkeypatch.setattr(
        ple.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    # Exercise the existing public call without any new keyword arguments.
    ple.check_ple_host_share(NS(ple_layer_ids=[1]), 4)
    assert len(warnings) == 1
    assert "cgroup" in warnings[0] and "128.0" in warnings[0]


@pytest.mark.parametrize(
    "connector,extra",
    [
        ("OffloadingConnector", {"cpu_bytes_to_use": 48 * GIB}),
        (
            "OffloadingConnector",
            {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 48 * GIB},
        ),
        ("SimpleCPUOffloadConnector", {"cpu_bytes_to_use": 48 * GIB}),
    ],
)
def test_resolved_cpu_pool_is_a_total_not_per_rank(connector, extra):
    config = NS(
        kv_transfer_config=NS(kv_connector=connector, kv_connector_extra_config=extra),
        cache_config=NS(kv_offloading_size=48),
    )
    assert ple.cpu_kv_offload_bytes(config) == 48 * GIB


@pytest.mark.parametrize("total,expected", [(16, 4), (30, 7.5), (32, 8), (196, 8)])
def test_default_host_reserve_is_capped(monkeypatch, total, expected):
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    assert ple.ple_host_reserve_bytes(total * GIB) == int(expected * GIB)
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", "49")
    assert ple.ple_host_reserve_bytes(total * GIB) == 49 * GIB


@pytest.mark.parametrize(
    "enabled,placement,expected",
    [(True, "host", 98), (False, "host", 48), (True, "gpu", 48)],
)
def test_unified_host_store_replaces_legacy_pool_once(enabled, placement, expected):
    config = NS(
        kernel_config=NS(
            long_context_kv=NS(
                enabled=enabled, host_store_gib=98, main_kv_placement=placement
            )
        ),
        kv_transfer_config=NS(
            kv_connector="OffloadingConnector",
            kv_connector_extra_config={"cpu_bytes_to_use": 48 * GIB},
        ),
        cache_config=NS(kv_offloading_size=48),
    )
    assert ple.cpu_kv_offload_bytes(config) == expected * GIB


@pytest.mark.parametrize("placement,expected", [("host", 100), ("gpu", 2)])
def test_legacy_main_host_pool_adds_to_cpu_pool(placement, expected):
    config = NS(
        kernel_config=NS(
            long_context_kv=NS(
                enabled=True, main_kv_placement=placement, host_pool_gib=98
            )
        ),
        kv_transfer_config=NS(
            kv_connector="OffloadingConnector",
            kv_connector_extra_config={"cpu_bytes_to_use": 2 * GIB},
        ),
    )
    assert ple.cpu_kv_offload_bytes(config) == expected * GIB


def test_production_sized_budget_logs_shared_98_gib_and_8_gib_reserve(monkeypatch):
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", "12")
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    monkeypatch.setattr(
        ple, "preflight_host_memory", lambda: (196 * GIB, 188 * GIB, "cgroup")
    )
    messages = []
    monkeypatch.setattr(
        ple.logger, "info", lambda msg, *args: messages.append(msg % args)
    )
    ple.check_ple_host_share(NS(ple_layer_ids=[1]), 4, cpu_kv_bytes=98 * GIB)
    assert len(messages) == 1
    assert "upper bound 48.0" in messages[0]
    assert "CPU KV pool 98.0 GiB total, reserve 8.0" in messages[0]


def test_explicit_partial_host_placement_warns_and_keeps_existing_split(monkeypatch):
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", "0.00000001")
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", "0")
    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.ple_layer.ple_cascade_configured", lambda: False
    )
    layer = Qwen4ExpPinnedHostEmbedding.__new__(Qwen4ExpPinnedHostEmbedding)
    layer.embedding_dim = 2
    layer._meta_weight_shape = (16, 2)
    warnings = []
    from vllm.models.qwen4_exp.nvidia import ple_layer

    monkeypatch.setattr(
        ple_layer.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    with set_current_vllm_config(VllmConfig()):
        placement = layer._plan_placement(None)
    assert (placement.vram_rows, placement.host_rows, placement.disk_rows) == (11, 5, 0)
    assert any("11" in msg and "device memory" in msg for msg in warnings)


def test_increasing_cap_does_not_increase_actual_resident_table():
    plans = [
        ple.plan_ple_placement(
            total_rows=80_000_384,
            row_bytes=160,
            host_budget_bytes=gib * GIB,
            vram_budget_bytes=None,
        )
        for gib in (12, 16)
    ]
    assert plans[0] == plans[1]
