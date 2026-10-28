# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 TP4 NVLink sum2 routing (CPU-only checks).

On an NVLink-connected group of four V100s the ordinary all-reduce keeps the
admission it had: custom below the 8 MiB dispatch ceiling, in eager mode and
inside CUDA graphs. Only sum2 changes. Inside a CUDA graph it keeps its fused
push kernel at the sizes that have one; every other call adds the two inputs
locally and takes that ordinary all-reduce, as under a PCIe switch. The generic
fused sum2 pull it no longer uses was slower than that route at every measured
size. PCIe-switch TP4, TP2, TP8 and no_p2p keep the routes they already had.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

from .test_sm70_pcie_sum2_route import FLAGS, _cuda, _custom

# The reference decode payload is [tokens, 2560] FP16, i.e. 5120 bytes per token.
TOKEN_BYTES = 2560 * torch.float16.itemsize
# Payloads with a fused sum2 push kernel (_sm70_tp4_push_sum2_size defaults):
# 5120, 20480, 25600, 40960 and 81920 bytes.
SUM2_PUSH_TOKENS = (1, 4, 5, 8, 16)
DISPATCH_MAX_BYTES = 8 * 1024 * 1024
# csrc/custom_all_reduce.cuh:73,99-100 (MTP4 verifies five rows per request) and
# csrc/custom_all_reduce.cuh:80-81 (DFlash2 verifies eight positions at once).
MTP4_ROWS_PER_REQUEST = 5
DFLASH2_ROWS_PER_REQUEST = 8


def _communicator(
    *,
    nvlink_tp4: bool = True,
    mode: str = "nvlink",
    world_size: int = 4,
    fully_connected: bool = True,
) -> CustomAllreduce:
    """A resolved communicator without running __init__ (as in the PCIe tests)."""
    communicator = object.__new__(CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.rank = 0
    communicator.interconnect_mode = mode
    communicator.world_size = world_size
    communicator.fully_connected = fully_connected
    communicator.tp8_hierarchical = False
    communicator.dispatch_max_size = DISPATCH_MAX_BYTES
    communicator.sm70_pcie_switch = False
    communicator.sm70_pcie_switch_max_bytes = 0
    communicator.sm70_pcie_switch_graph_only = True
    communicator.sm70_nvlink_tp4 = nvlink_tp4
    communicator.sm70_tp4_push_buffer_ptrs = [1, 2, 3, 4]
    return communicator


def _rows(tokens: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return torch.empty(tokens, 2560, dtype=dtype)


def _nvlink_custom(
    monkeypatch: pytest.MonkeyPatch, *, capturing: bool
) -> car.CustomAllreduce:
    """The sum2-route fakes, resolved as SM70 TP4 NVLink with push buffers."""
    communicator = _custom(monkeypatch, switch=False, capturing=capturing)
    communicator.interconnect_mode = "nvlink"
    communicator.sm70_nvlink_tp4 = True
    return communicator


def test_flag_defaults_off_for_instances_built_without_init() -> None:
    # Every mode other than SM70 TP4 NVLink, and the fakes other tests build
    # with object.__new__, read the class default and keep the old sum2 route.
    assert CustomAllreduce.sm70_nvlink_tp4 is False
    assert object.__new__(CustomAllreduce).sm70_nvlink_tp4 is False


def test_sum2_push_sizes_match_the_native_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in FLAGS:
        monkeypatch.delenv(key, raising=False)
    for tokens in range(1, 65):
        assert car._sm70_tp4_push_sum2_size(tokens * TOKEN_BYTES) is (
            tokens in SUM2_PUSH_TOKENS
        )


@pytest.mark.parametrize(
    "tokens", list(range(1, 65)) + [80, 128, 256, 1024, 1638, 1639]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("capturing", [False, True])
def test_nvlink_all_reduce_admission_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    dtype: torch.dtype,
    capturing: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    payload = _rows(tokens, dtype)
    nbytes = payload.numel() * payload.element_size()
    flagged = _communicator(nvlink_tp4=True).should_custom_ar(payload)
    plain = _communicator(nvlink_tp4=False).should_custom_ar(payload)
    # The flag does not touch admission: only the dispatch ceiling decides, as
    # before (1638 FP16 tokens are 8,386,560 bytes, just below 8 MiB).
    assert flagged is plain is (nbytes < DISPATCH_MAX_BYTES)


def test_nvlink_all_reduce_keeps_the_contiguity_and_alignment_checks() -> None:
    communicator = _communicator()
    strided = torch.empty(2, 2 * 2560, dtype=torch.float16)[:, :2560]
    assert not car.is_weak_contiguous(strided)
    assert communicator.should_custom_ar(strided) is False
    assert communicator.should_custom_ar(strided.contiguous()) is True
    # Not a multiple of 16 bytes.
    assert communicator.should_custom_ar(torch.empty(7, dtype=torch.float16)) is False


@pytest.mark.parametrize("tokens", list(range(1, 65)) + [256, 1024])
@pytest.mark.parametrize("capturing", [False, True])
def test_nvlink_all_reduce_route_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tokens: int, capturing: bool
) -> None:
    custom = _nvlink_custom(monkeypatch, capturing=capturing)
    custom.can_sm70_tp4_push_sum2 = Mock(side_effect=AssertionError("ordinary AR"))
    comm = _cuda(custom)
    payload = torch.ones(tokens, 2560, dtype=torch.float16)
    assert CudaCommunicator.all_reduce(comm, payload) is payload
    custom.custom_all_reduce.assert_called_once_with(payload)
    comm.pynccl_comm.all_reduce.assert_not_called()


@pytest.mark.parametrize("tokens", range(1, 65))
@pytest.mark.parametrize("capturing", [False, True])
def test_nvlink_sum2_uses_its_push_kernel_or_the_local_sum(
    monkeypatch: pytest.MonkeyPatch, tokens: int, capturing: bool
) -> None:
    custom = _nvlink_custom(monkeypatch, capturing=capturing)
    custom.custom_all_reduce_sum2 = Mock(wraps=custom.custom_all_reduce_sum2)
    comm = _cuda(custom)
    input_a = torch.full((tokens, 2560), 0.1, dtype=torch.float16)
    input_b = torch.full_like(input_a, 0.2)
    result = comm.all_reduce_sum2(input_a, input_b)
    if capturing and tokens in SUM2_PUSH_TOKENS:
        # The sizes with a fused sum2 push kernel keep it inside a graph.
        assert result is input_a
        custom.custom_all_reduce_sum2.assert_called_once_with(input_a, input_b)
        custom.all_reduce_sum2.assert_called_once_with(input_a, input_b)
        custom.custom_all_reduce.assert_not_called()
        comm.all_reduce.assert_not_called()
    else:
        # Everything else adds locally and takes the ordinary all-reduce, which
        # stays on custom (below the dispatch ceiling); the generic fused pull
        # is never called.
        custom.custom_all_reduce_sum2.assert_not_called()
        custom.all_reduce_sum2.assert_not_called()
        comm.all_reduce.assert_called_once()
        custom.custom_all_reduce.assert_called_once()
        assert torch.equal(
            result.view(torch.int16), (input_a + input_b).view(torch.int16)
        )
    comm.pynccl_comm.all_reduce.assert_not_called()


@pytest.mark.parametrize("tokens", [2, 3, 10, 20, 32, 64])
def test_without_the_flag_nvlink_sum2_keeps_the_generic_pull(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    # TP2, other architectures and every non-NVLink mode resolve the flag off
    # and keep the fused generic pull they had.
    custom = _nvlink_custom(monkeypatch, capturing=True)
    custom.sm70_nvlink_tp4 = False
    comm = _cuda(custom)
    input_a = torch.ones(tokens, 2560, dtype=torch.float16)
    assert comm.all_reduce_sum2(input_a, input_a) is input_a
    custom.all_reduce_sum2.assert_called_once_with(input_a, input_a)
    comm.all_reduce.assert_not_called()
    comm.pynccl_comm.all_reduce.assert_not_called()


def test_nvlink_sum2_other_dtypes_keep_the_generic_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = _nvlink_custom(monkeypatch, capturing=True)
    comm = _cuda(custom)
    input_a = torch.ones(3, 2560, dtype=torch.float32)
    assert comm.all_reduce_sum2(input_a, input_a) is input_a
    custom.all_reduce_sum2.assert_called_once_with(input_a, input_a)
    comm.all_reduce.assert_not_called()


def test_nvlink_sum2_without_push_buffers_takes_the_local_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = _nvlink_custom(monkeypatch, capturing=True)
    custom.sm70_tp4_push_buffer_ptrs = None
    comm = _cuda(custom)
    input_a = torch.ones(8, 2560, dtype=torch.float16)
    assert torch.equal(comm.all_reduce_sum2(input_a, input_a), input_a + input_a)
    custom.all_reduce_sum2.assert_not_called()
    custom.custom_all_reduce.assert_called_once()
    # Called directly, the fused entry point declines rather than pulling.
    assert custom.custom_all_reduce_sum2(input_a, input_a) is None


@pytest.mark.parametrize("requests", range(1, 9))
@pytest.mark.parametrize(
    "rows_per_request", [MTP4_ROWS_PER_REQUEST, DFLASH2_ROWS_PER_REQUEST]
)
def test_nvlink_step_sum2_routes_for_mtp4_and_dflash2(
    monkeypatch: pytest.MonkeyPatch, rows_per_request: int, requests: int
) -> None:
    """The per-step sum2 payloads the reference deployment actually issues."""
    tokens = rows_per_request * requests
    custom = _nvlink_custom(monkeypatch, capturing=True)
    comm = _cuda(custom)
    input_a = torch.ones(tokens, 2560, dtype=torch.float16)
    comm.all_reduce_sum2(input_a, input_a)
    # Fused: one MTP4 request (5 rows) and one or two DFlash2 requests (8 and 16
    # rows). Every larger batch now takes the local sum instead of the pull.
    fused = tokens in SUM2_PUSH_TOKENS
    assert fused is (tokens in (5, 8, 16))
    assert custom.all_reduce_sum2.called is fused
    assert comm.all_reduce.called is not fused


@pytest.mark.parametrize("tokens", [1, 2, 4, 5, 8, 10, 16, 17, 32, 64])
def test_sum2_route_is_the_same_on_every_rank(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    # The decision depends on payload bytes and capture state only, so no rank
    # can pick a different collective than its peers.
    decisions = []
    for rank in range(4):
        custom = _nvlink_custom(monkeypatch, capturing=True)
        custom.rank = rank
        decisions.append(custom.can_sm70_tp4_push_sum2(_rows(tokens)))
    assert len(set(decisions)) == 1
    assert decisions[0] is (tokens in SUM2_PUSH_TOKENS)
