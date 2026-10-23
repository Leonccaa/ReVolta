# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    get_explicit_cudagraph_memory_reserve,
    get_sm70_cudagraph_memory_reserve,
    warn_if_graphs_outgrew_reserve,
)

pytestmark = pytest.mark.cpu_test


def test_explicit_cudagraph_memory_reserve(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_V2_CUDAGRAPH_MEM_MIB", "1536.5")

    assert get_explicit_cudagraph_memory_reserve(CUDAGraphMode.FULL) == int(
        1536.5 * 1024 * 1024
    )
    assert get_explicit_cudagraph_memory_reserve(CUDAGraphMode.NONE) == 0


def test_explicit_cudagraph_memory_reserve_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_V2_CUDAGRAPH_MEM_MIB", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        get_explicit_cudagraph_memory_reserve(CUDAGraphMode.FULL)


def test_sm70_reserves_measured_peak_and_preserves_override(monkeypatch):
    monkeypatch.delenv("VLLM_V2_CUDAGRAPH_MEM_MIB", raising=False)
    assert get_sm70_cudagraph_memory_reserve(CUDAGraphMode.FULL, 123456) == 123456
    assert get_sm70_cudagraph_memory_reserve(CUDAGraphMode.NONE, 123456) == 0
    monkeypatch.setenv("VLLM_V2_CUDAGRAPH_MEM_MIB", "0")
    assert get_sm70_cudagraph_memory_reserve(CUDAGraphMode.FULL, 123456) == 0
    monkeypatch.setenv("VLLM_V2_CUDAGRAPH_MEM_MIB", "128")
    assert get_sm70_cudagraph_memory_reserve(CUDAGraphMode.FULL, 123456) == 128 * 2**20


MIB = 1 << 20
GIB = 1 << 30


@pytest.mark.parametrize(
    ("actual", "reserve", "suggested"),
    [
        # MTP4 on the AWQ TP4 pool: within 0.3% of the 700 MiB reserve.
        (int(700.4 * MIB), 700 * MIB, 0),
        # An overshoot within max(64 MiB, 5%) stays quiet.
        (760 * MIB, 700 * MIB, 0),
        (1560 * MIB, 1500 * MIB, 0),
        # DFlash2 against the MTP reserve: 1.37-1.39 GiB measured.
        (int(1.39 * GIB), 700 * MIB, 1500),
        (int(1.37 * GIB), 700 * MIB, 1500),
        (int(1.39 * GIB), 1500 * MIB, 0),
    ],
)
def test_warns_only_when_graphs_outgrow_the_reserve(
    monkeypatch, actual, reserve, suggested
):
    warnings = []
    monkeypatch.setattr(
        cudagraph_utils.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )
    assert warn_if_graphs_outgrew_reserve(actual, reserve) == suggested
    assert len(warnings) == (1 if suggested else 0)
    if suggested:
        assert f"VLLM_V2_CUDAGRAPH_MEM_MIB to at least {suggested}" in warnings[0]
