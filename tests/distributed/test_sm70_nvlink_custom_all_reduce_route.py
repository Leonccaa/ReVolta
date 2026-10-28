# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 TP4 NVLink custom all-reduce payload routing (CPU-only checks).

On an NVLink-connected group of four V100s the custom kernels beat NCCL for
payloads up to 27 tokens of the reference FP16 [tokens, 2560] tensor (138240
bytes) and for the two larger payloads that have their own push kernel (163840 and
327680 bytes, i.e. 32 and 64 tokens). Every other admitted payload goes to NCCL,
in eager mode and inside CUDA graphs. sum2 keeps its fused push kernel at the sizes
that have one and otherwise takes the local sum plus that ordinary route, as under
a PCIe switch.
PCIe-switch TP4, TP2, TP8 and no_p2p keep the rules they already had; the
PCIe-switch ladder is re-checked byte for byte here.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

from .test_sm70_pcie_sum2_route import _cuda, _custom

# The reference decode payload is [tokens, 2560] FP16, i.e. 5120 bytes per token.
TOKEN_BYTES = 2560 * torch.float16.itemsize
# csrc/custom_all_reduce.cuh:75,89-96 sizes the SM70 TP4 push payloads as 8, 16
# and 32 rows of the 5120-element reference hidden size: 81920, 163840 and
# 327680 bytes, which are 16, 32 and 64 tokens of the reference tensor.
SMALL_MAX_BYTES = 81920
SMALL_MAX_TOKENS = SMALL_MAX_BYTES // TOKEN_BYTES
LARGE_PUSH_BYTES = (163840, 327680)
# The NVLink custom/NCCL crossover inside CUDA graphs: 27 reference tokens.
NVLINK_MAX_BYTES = 138240
NVLINK_MAX_TOKENS = NVLINK_MAX_BYTES // TOKEN_BYTES
# Payloads with a fused sum2 push kernel (_sm70_tp4_push_sum2_size defaults).
SUM2_PUSH_TOKENS = (1, 4, 5, 8, 16)
LARGE_PUSH_TOKENS = tuple(size // TOKEN_BYTES for size in LARGE_PUSH_BYTES)
CONCURRENCY = "VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY"
DISPATCH_MAX_BYTES = 8 * 1024 * 1024
# csrc/custom_all_reduce.cuh:73,99-100 (MTP4 verifies five rows per request) and
# csrc/custom_all_reduce.cuh:80-81 (DFlash2 verifies eight positions at once).
MTP4_ROWS_PER_REQUEST = 5
DFLASH2_ROWS_PER_REQUEST = 8


def _communicator(
    *,
    gate: bool = True,
    mode: str = "nvlink",
    push_registered: bool = True,
    pcie_switch: bool = False,
    max_bytes: int = SMALL_MAX_BYTES,
    graph_only: bool = True,
    world_size: int = 4,
    fully_connected: bool = True,
    tp8_hierarchical: bool = False,
    dispatch_max_size: int = DISPATCH_MAX_BYTES,
) -> CustomAllreduce:
    """A resolved communicator without running __init__ (as in the PCIe tests)."""
    communicator = object.__new__(CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.rank = 0
    communicator.interconnect_mode = mode
    communicator.world_size = world_size
    communicator.fully_connected = fully_connected
    communicator.tp8_hierarchical = tp8_hierarchical
    communicator.dispatch_max_size = dispatch_max_size
    communicator.sm70_pcie_switch = pcie_switch
    communicator.sm70_pcie_switch_max_bytes = max_bytes if pcie_switch else 0
    communicator.sm70_pcie_switch_graph_only = graph_only
    communicator.sm70_nvlink_tp4_car_sizes = gate
    communicator.sm70_tp4_push_buffer_ptrs = [1, 2, 3, 4] if push_registered else None
    return communicator


def _rows(tokens: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return torch.empty(tokens, 2560, dtype=dtype)


def _fp16(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes // 2, dtype=torch.float16)


def _table(tokens: int) -> bool:
    """The payload table under test, expressed in tokens of the reference row."""
    nbytes = tokens * TOKEN_BYTES
    return nbytes <= NVLINK_MAX_BYTES or nbytes in LARGE_PUSH_BYTES


def _nvlink_custom(
    monkeypatch: pytest.MonkeyPatch, *, capturing: bool
) -> car.CustomAllreduce:
    """The sum2-route fakes, resolved as SM70 TP4 NVLink with push buffers."""
    communicator = _custom(monkeypatch, switch=False, capturing=capturing)
    communicator.interconnect_mode = "nvlink"
    communicator.sm70_nvlink_tp4_car_sizes = True
    monkeypatch.setenv(CONCURRENCY, "1")
    return communicator


def test_push_payload_constants_match_the_native_sizes() -> None:
    # The Python mirror has to stay on the native byte sizes instead of a copy
    # that can drift: 81920 / 163840 / 327680 bytes are 16 / 32 / 64 tokens.
    assert TOKEN_BYTES == 5120
    assert SMALL_MAX_TOKENS == 16
    assert LARGE_PUSH_TOKENS == (32, 64)
    assert car._SM70_TP4_PUSH_ALLREDUCE_M8_BYTES == SMALL_MAX_BYTES
    assert NVLINK_MAX_TOKENS == 27
    assert car._SM70_TP4_NVLINK_CAR_MAX_BYTES == NVLINK_MAX_BYTES
    assert frozenset(LARGE_PUSH_BYTES) == car._SM70_TP4_PUSH_ALLREDUCE_LARGE_BYTES


@pytest.mark.parametrize("tokens", range(1, NVLINK_MAX_TOKENS + 1))
@pytest.mark.parametrize("capturing", [False, True])
def test_nvlink_admits_every_payload_up_to_the_crossover(
    monkeypatch: pytest.MonkeyPatch, tokens: int, capturing: bool
) -> None:
    # 5120 .. 138240 bytes stay on custom, in eager mode and inside a graph, and
    # without depending on the opt-in push-concurrency switch.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.delenv(CONCURRENCY, raising=False)
    communicator = _communicator()
    assert tokens * TOKEN_BYTES <= NVLINK_MAX_BYTES
    assert communicator.should_custom_ar(_rows(tokens)) is True


@pytest.mark.parametrize("tokens", LARGE_PUSH_TOKENS)
def test_nvlink_admits_the_two_larger_push_payloads_only_with_push(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setenv(CONCURRENCY, "1")
    payload = _rows(tokens)
    assert payload.numel() * payload.element_size() in LARGE_PUSH_BYTES
    assert _communicator().should_custom_ar(payload) is True
    # The native kernel admits these two sizes only when it reads exactly "1"
    # (csrc/custom_all_reduce.cuh:170-177), and vllm/envs.py publishes the
    # resolved default into os.environ before any launch, so any other value or
    # an absent value means there is no push kernel and the payload goes to NCCL.
    for value in ("0", "", "true", "01", None):
        if value is None:
            monkeypatch.delenv(CONCURRENCY, raising=False)
        else:
            monkeypatch.setenv(CONCURRENCY, value)
        assert _communicator().should_custom_ar(payload) is False
    monkeypatch.setenv(CONCURRENCY, "1")
    # Without a registered push buffer the switch alone is not enough.
    assert _communicator(push_registered=False).should_custom_ar(payload) is False


@pytest.mark.parametrize(
    "tokens",
    [
        tokens
        for tokens in range(NVLINK_MAX_TOKENS + 1, 65)
        if tokens not in LARGE_PUSH_TOKENS
    ]
    + [80, 128, 256, 1024, 1638],
)
@pytest.mark.parametrize("capturing", [False, True])
def test_nvlink_routes_every_other_payload_to_nccl(
    monkeypatch: pytest.MonkeyPatch, tokens: int, capturing: bool
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setenv(CONCURRENCY, "1")
    communicator = _communicator()
    # Every payload here is below the 8 MiB dispatch ceiling, so declining it is
    # the new table and not some pre-existing limit.
    assert tokens * TOKEN_BYTES < DISPATCH_MAX_BYTES
    assert communicator.should_custom_ar(_rows(tokens)) is False


def test_nvlink_off_grid_payloads_follow_the_same_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The rule is byte based, so payloads that are not whole reference rows use
    # the same boundaries.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setenv(CONCURRENCY, "1")
    communicator = _communicator()
    assert communicator.should_custom_ar(_fp16(NVLINK_MAX_BYTES)) is True
    assert communicator.should_custom_ar(_fp16(NVLINK_MAX_BYTES + 16)) is False
    for size in LARGE_PUSH_BYTES:
        assert communicator.should_custom_ar(_fp16(size)) is True
        assert communicator.should_custom_ar(_fp16(size + 16)) is False


def test_nvlink_predicates_ignore_non_fp16_and_non_contiguous_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setenv(CONCURRENCY, "1")
    communicator = _communicator()
    # FP32 and BF16 were and stay admissible dtypes; the payload size decides.
    # 20 tokens are 204800 bytes in FP32 (above the crossover, no push kernel:
    # NCCL) and 102400 in BF16 (below it: custom); 30 BF16 tokens are 153600.
    assert 20 * 2560 * 4 == 204800 and 20 * 2560 * 2 == 102400
    assert communicator.should_custom_ar(_rows(20, torch.float32)) is False
    assert communicator.should_custom_ar(_rows(20, torch.bfloat16)) is True
    assert communicator.should_custom_ar(_rows(30, torch.bfloat16)) is False
    # Small payloads keep the dtype policy they had.
    assert communicator.should_custom_ar(_rows(2, torch.float32)) is True
    assert communicator.should_custom_ar(_rows(16, torch.bfloat16)) is True
    # A strided view is still rejected by the existing contiguity check, even
    # when its byte count would otherwise be admitted.
    strided_small = torch.empty(2, 2 * 2560, dtype=torch.float16)[:, :2560]
    assert strided_small.numel() * strided_small.element_size() == 10240
    assert not car.is_weak_contiguous(strided_small)
    assert communicator.should_custom_ar(strided_small) is False
    assert communicator.should_custom_ar(strided_small.contiguous()) is True
    # A zero-length payload must not raise and is admitted as it was before.
    assert communicator.should_custom_ar(_rows(0)) is True


@pytest.mark.parametrize("tokens", list(range(1, 65)) + [80, 128, 256, 1024, 1638])
@pytest.mark.parametrize("capturing", [False, True])
def test_pcie_switch_route_is_byte_for_byte_unchanged(
    monkeypatch: pytest.MonkeyPatch, tokens: int, capturing: bool
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setenv(CONCURRENCY, "1")
    switch = _communicator(gate=False, mode="pex_p2p", pcie_switch=True)
    # Exactly the rule this tree asserted before the NVLink table existed.
    assert switch.should_custom_ar(_rows(tokens)) is (
        capturing and tokens * TOKEN_BYTES <= SMALL_MAX_BYTES
    )
    # With the capture-only restriction lifted, only the byte cap decides, and
    # the push-concurrency switch still does not widen the PCIe-switch ladder.
    unlimited = _communicator(
        gate=False, mode="pex_p2p", pcie_switch=True, graph_only=False
    )
    assert unlimited.should_custom_ar(_rows(tokens)) is (
        tokens * TOKEN_BYTES <= SMALL_MAX_BYTES
    )


def test_other_modes_keep_their_own_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv(CONCURRENCY, "1")
    # TP2 upstream: the ordinary size ceiling, no payload table.
    tp2 = _communicator(gate=False, mode="upstream", world_size=2)
    assert tp2.sm70_nvlink_tp4_car_sizes is False
    assert tp2.should_custom_ar(_fp16(1024 * 1024)) is True
    assert tp2.should_custom_ar(_fp16(DISPATCH_MAX_BYTES)) is False
    # An NVLink group that never resolves the SM70 TP4 table (TP2 width, or a
    # newer architecture) keeps exactly that rule.
    other_nvlink = _communicator(gate=False, mode="nvlink", world_size=2)
    assert other_nvlink.should_custom_ar(_fp16(1024 * 1024)) is True
    # TP8 hierarchical: only the two FP16 allowlist shapes.
    tp8 = _communicator(
        gate=False, mode="tp8_hierarchical", world_size=8, tp8_hierarchical=True
    )
    assert tp8.should_custom_ar(torch.empty(4096, dtype=torch.float16)) is True
    assert tp8.should_custom_ar(torch.empty(8 * 4096, dtype=torch.float16)) is True
    assert tp8.should_custom_ar(torch.empty(16 * 4096, dtype=torch.float16)) is False
    assert tp8.should_custom_ar(torch.empty(4096, dtype=torch.float32)) is False
    # no_p2p admits nothing at all.
    no_p2p = _communicator(gate=False, mode="no_p2p", fully_connected=False)
    assert no_p2p.should_custom_ar(_fp16(1024)) is False


@pytest.mark.parametrize("tokens", range(1, 65))
@pytest.mark.parametrize("capturing", [False, True])
def test_nvlink_all_reduce_and_sum2_routes_match_the_payload_table(
    monkeypatch: pytest.MonkeyPatch, tokens: int, capturing: bool
) -> None:
    expected_custom = _table(tokens)

    ar_comm = _cuda(_nvlink_custom(monkeypatch, capturing=capturing))
    payload = torch.full((tokens, 2560), 0.1, dtype=torch.float16)
    CudaCommunicator.all_reduce(ar_comm, payload)
    if expected_custom:
        ar_comm.ca_comm.custom_all_reduce.assert_called_once_with(payload)
        ar_comm.pynccl_comm.all_reduce.assert_not_called()
    else:
        ar_comm.ca_comm.custom_all_reduce.assert_not_called()
        ar_comm.pynccl_comm.all_reduce.assert_called_once_with(payload)

    custom = _nvlink_custom(monkeypatch, capturing=capturing)
    custom.custom_all_reduce_sum2 = Mock(wraps=custom.custom_all_reduce_sum2)
    comm = _cuda(custom)
    input_a = torch.full((tokens, 2560), 0.1, dtype=torch.float16)
    input_b = torch.full_like(input_a, 0.2)
    result = comm.all_reduce_sum2(input_a, input_b)
    if capturing and tokens in SUM2_PUSH_TOKENS:
        # The sizes with a fused sum2 push kernel keep it inside a graph.
        custom.custom_all_reduce_sum2.assert_called_once_with(input_a, input_b)
        custom.all_reduce_sum2.assert_called_once_with(input_a, input_b)
        custom.custom_all_reduce.assert_not_called()
        comm.all_reduce.assert_not_called()
        comm.pynccl_comm.all_reduce.assert_not_called()
    else:
        # Everything else (the generic fused pull measured slower than NCCL from
        # 20 tokens on, 48.1 against 27.5 us at 64) takes the local sum plus the
        # ordinary all-reduce, which then follows the payload table: custom up
        # to 27 tokens and at the 32 / 64-token push sizes, NCCL otherwise.
        custom.custom_all_reduce_sum2.assert_not_called()
        custom.all_reduce_sum2.assert_not_called()
        comm.all_reduce.assert_called_once()
        if expected_custom:
            custom.custom_all_reduce.assert_called_once()
            comm.pynccl_comm.all_reduce.assert_not_called()
        else:
            custom.custom_all_reduce.assert_not_called()
            comm.pynccl_comm.all_reduce.assert_called_once()
        assert torch.equal(
            result.view(torch.int16), (input_a + input_b).view(torch.int16)
        )


@pytest.mark.parametrize("requests", range(1, 9))
@pytest.mark.parametrize(
    "rows_per_request", [MTP4_ROWS_PER_REQUEST, DFLASH2_ROWS_PER_REQUEST]
)
def test_nvlink_step_payload_table_for_mtp4_and_dflash2(
    monkeypatch: pytest.MonkeyPatch, rows_per_request: int, requests: int
) -> None:
    """The per-step payloads the reference deployment actually collects."""
    tokens = rows_per_request * requests
    custom = _nvlink_custom(monkeypatch, capturing=True)
    comm = _cuda(custom)
    payload = torch.ones(tokens, 2560, dtype=torch.float16)
    CudaCommunicator.all_reduce(comm, payload)
    if _table(tokens):
        custom.custom_all_reduce.assert_called_once_with(payload)
        comm.pynccl_comm.all_reduce.assert_not_called()
    else:
        custom.custom_all_reduce.assert_not_called()
        comm.pynccl_comm.all_reduce.assert_called_once_with(payload)


def test_documented_verify_bands_keep_the_crossover_and_push_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MTP4 with 4..8 requests and DFlash2 with 3..8 requests give the payload set
    # {20,25,30,35,40} | {24,32,40,48,56,64} tokens; 20, 24 and 25 tokens are
    # below the crossover and 32 and 64 tokens (163840 and 327680 bytes) keep
    # their dedicated push kernel; the rest go to NCCL.
    mtp4 = {MTP4_ROWS_PER_REQUEST * requests for requests in range(4, 9)}
    dflash2 = {DFLASH2_ROWS_PER_REQUEST * requests for requests in range(3, 9)}
    assert mtp4 == {20, 25, 30, 35, 40}
    assert dflash2 == {24, 32, 40, 48, 56, 64}
    assert {tokens for tokens in mtp4 | dflash2 if _table(tokens)} == {
        20,
        24,
        25,
        32,
        64,
    }
    # A single request (5 MTP4 rows or 8 DFlash2 rows) stays on custom, so
    # one-request greedy output cannot change backend.
    for tokens in (MTP4_ROWS_PER_REQUEST, DFLASH2_ROWS_PER_REQUEST):
        assert _table(tokens)
        communicator = _nvlink_custom(monkeypatch, capturing=True)
        assert communicator.should_custom_ar(_rows(tokens)) is True


@pytest.mark.parametrize(
    "tokens", [1, 8, 16, 17, 20, 27, 28, 31, 32, 33, 40, 63, 64, 80, 1024, 1638]
)
def test_admission_is_independent_of_capture_state_and_rank(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    monkeypatch.setenv(CONCURRENCY, "1")
    communicators = []
    for rank in range(4):
        communicator = _communicator()
        communicator.rank = rank
        communicators.append(communicator)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    eager = [c.should_custom_ar(_rows(tokens)) for c in communicators]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    captured = [c.should_custom_ar(_rows(tokens)) for c in communicators]
    # The table depends on payload bytes only, so no rank can pick a different
    # backend than its peers, in eager mode or inside a CUDA graph, and the
    # decision is the one the table says.
    assert len(set(eager)) == 1
    assert eager == captured
    assert eager[0] is _table(tokens)


def test_route_is_decided_before_the_native_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setenv(CONCURRENCY, "1")
    native = Mock()
    monkeypatch.setattr(car.ops, "all_reduce", native)
    communicator = _communicator()
    # A payload routed to NCCL never reaches the custom entry point, so it
    # cannot half-start a custom collective before falling back.
    assert communicator.custom_all_reduce(_rows(30)) is None
    native.assert_not_called()


def test_vllm_envs_publishes_the_push_concurrency_default(tmp_path: Path) -> None:
    # The table reads the raw string the native kernel reads at launch
    # (csrc/custom_all_reduce.cuh:170-177). Importing vllm.envs must publish the
    # resolved default, so an unset variable means "1" on both sides and the
    # 32 / 64-token payloads keep their push kernel. Run from an empty directory
    # so the child imports the installed package, not a checkout in the cwd.
    env = {key: value for key, value in os.environ.items() if key != CONCURRENCY}
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, vllm.envs; print(os.environ.get(" + repr(CONCURRENCY) + "))",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip().splitlines()[-1] == "1"
