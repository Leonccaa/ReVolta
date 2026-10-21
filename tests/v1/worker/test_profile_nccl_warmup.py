# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Force c10d communicators before the first profiling forward (CPU mocks)."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker import gpu_worker


@pytest.mark.parametrize(
    "cuda,sizes,backends,shared,expected",
    [
        (False, [4, 2], ["nccl", "nccl"], False, []),
        (True, [1, 1], ["nccl", "nccl"], False, []),
        (True, [4, 2], ["gloo", "nccl"], False, [1]),
        (True, [4, 2], ["nccl", "nccl"], False, [0, 1]),
        (True, [4, 4], ["nccl", "nccl"], True, [0]),
    ],
)
def test_profile_collective_targets_c10d(
    monkeypatch, cuda, sizes, backends, shared, expected
):
    method = getattr(gpu_worker.Worker, "_warmup_profile_communicators", None)
    assert method is not None
    groups = [NS(world_size=size, device_group=object()) for size in sizes]
    if shared:
        groups[1] = groups[0]
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: groups[0])
    monkeypatch.setattr(gpu_worker, "get_dcp_group", lambda: groups[1], raising=False)
    monkeypatch.setattr(
        torch.distributed,
        "get_backend",
        lambda group: backends[
            next(i for i, g in enumerate(groups) if g.device_group is group)
        ],
    )
    real_zeros = torch.zeros
    allocations = []

    def zeros(*args, **kwargs):
        allocations.append(kwargs.pop("device"))
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    seen = []

    def reduce(tensor, *, group):
        assert tensor.numel() == 1 and tensor.dtype == torch.float32
        assert tensor.item() == 0
        seen.append(next(i for i, g in enumerate(groups) if g.device_group is group))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    sync = Mock()
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)
    method(NS(device=torch.device("cuda:0")))
    assert seen == expected
    assert allocations == ([torch.device("cuda:0")] if expected else [])
    assert sync.call_count == bool(expected)


def test_collectives_precede_first_profile_forward(monkeypatch):
    events = []

    class StopProfile(Exception):
        pass

    def forward():
        events.append("forward")
        raise StopProfile

    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.model_runner = NS(profile_run=forward)
    worker._warmup_profile_communicators = lambda: events.append("communicators")
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: False)
    with pytest.raises(StopProfile):
        worker.determine_available_memory()
    assert events == ["communicators", "forward"]
