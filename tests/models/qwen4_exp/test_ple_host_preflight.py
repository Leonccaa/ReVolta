# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest

import vllm.models.qwen4_exp.common.ple as ple
from tests.utils import set_lazy_env
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPinnedHostEmbedding

GIB = 1 << 30


def test_cgroup_limit_and_current_include_tighter_parent(tmp_path):
    root = tmp_path / "cgroup"
    child = root / "worker"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/worker\n")
    for directory, limit, current in [(root, 128, 32), (child, 200, 10)]:
        (directory / "memory.max").write_text(str(limit * GIB))
        (directory / "memory.current").write_text(str(current * GIB))
    assert ple.cgroup_host_memory(root, membership) == (128 * GIB, 96 * GIB)


@pytest.mark.parametrize("limit", ["max", "invalid", None])
def test_unavailable_cgroup_limit_falls_back_to_meminfo(monkeypatch, tmp_path, limit):
    if limit is not None:
        (tmp_path / "memory.max").write_text(limit)
    membership = tmp_path / "membership"
    membership.write_text("0::/\n")
    assert ple.cgroup_host_memory(tmp_path, membership) == (None, None)
    monkeypatch.setattr(ple, "cgroup_host_memory", lambda: (None, None))
    monkeypatch.setattr(ple, "total_host_bytes", lambda: 256 * GIB)
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 200 * GIB)
    assert ple.preflight_host_memory() == (256 * GIB, 200 * GIB, "/proc/meminfo")


def test_cgroup_shortage_counts_cpu_kv_pool_once(monkeypatch):
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_GIB", "12")
    set_lazy_env(monkeypatch, "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB", None)
    monkeypatch.setattr(ple, "cgroup_host_memory", lambda: (128 * GIB, 120 * GIB))
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 400 * GIB)
    warnings = []
    monkeypatch.setattr(
        ple.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    ple.check_ple_host_share(NS(ple_layer_ids=[1]), 4, cpu_kv_bytes=48 * GIB)
    assert len(warnings) == 1
    message = warnings[0]
    assert "cgroup" in message and "48.0" in message and "CPU KV" in message
    assert "120.0" in message and "32.0" in message


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
