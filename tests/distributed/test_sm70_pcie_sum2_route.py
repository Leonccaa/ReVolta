# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 PCIe sum2 must avoid generic pull and retain exact native push gates."""

from itertools import product
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators import cuda_communicator as cuda_comm
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

FLAGS = (
    "VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH",
    "VLLM_SM70_QWEN38_BATCH_FASTPATH",
    "VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5",
    "VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1",
)


@pytest.mark.parametrize("values", list(product((None, "0", "1"), repeat=4)))
def test_native_push_size_contract_all_tokens(monkeypatch, values):
    for key, value in zip(FLAGS, values):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    batch, fast, mtp5, m1 = values
    admitted = set()
    if batch in (None, "1"):
        admitted.update((4, 8, 16))
        if fast == "1":
            admitted.add(2)
    if mtp5 in (None, "1"):
        admitted.add(5)
    if m1 in (None, "1"):
        admitted.add(1)
    for tokens in range(1, 65):
        assert car._sm70_tp4_push_sum2_size(tokens * 5120) is (tokens in admitted)


@pytest.mark.parametrize("value", ["", "01", "true", "2", "-1"])
def test_native_flags_use_exact_strings(monkeypatch, value):
    for key in FLAGS:
        monkeypatch.setenv(key, value)
    for tokens in range(1, 65):
        assert not car._sm70_tp4_push_sum2_size(tokens * 5120)


def test_fastpath_python_default_does_not_enable_native_m2(monkeypatch):
    import vllm.envs as envs

    for key in FLAGS:
        monkeypatch.delenv(key, raising=False)
    assert envs.environment_variables["VLLM_SM70_QWEN38_BATCH_FASTPATH"]()
    assert not car._sm70_tp4_push_sum2_size(10240)


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_ordinary_push_switches_do_not_widen_sum2(monkeypatch, value):
    for key in (
        "VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES",
        "VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY",
        "VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH_BLOCKS",
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    for key in FLAGS:
        monkeypatch.delenv(key, raising=False)
    for size in (0, 16, 8192, 163840, 327680):
        assert not car._sm70_tp4_push_sum2_size(size)
    assert car._sm70_tp4_push_sum2_size(81920)


def _custom(monkeypatch, *, switch=True, capturing=True):
    comm = object.__new__(car.CustomAllreduce)
    comm.disabled = False
    comm._ptr = 0
    comm.world_size = 4
    comm.fully_connected = True
    comm.tp8_hierarchical = False
    comm._IS_CAPTURING = capturing
    comm.sm70_pcie_switch = switch
    comm.sm70_pcie_switch_max_bytes = 81920
    comm.sm70_pcie_switch_graph_only = True
    comm.sm70_tp4_push_buffer_ptrs = [1, 2, 3, 4]
    comm.dispatch_max_size = 8 * 1024 * 1024
    comm.all_reduce_sum2 = Mock(side_effect=lambda a, b: a)
    comm.custom_all_reduce = Mock(side_effect=lambda inp: inp)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        cuda_comm, "should_nccl_symm_mem_allreduce", lambda *args: False
    )
    monkeypatch.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES", "1")
    for key in FLAGS:
        monkeypatch.delenv(key, raising=False)
    return comm


def _cuda(custom):
    comm = object.__new__(CudaCommunicator)
    comm.use_custom_allreduce = True
    comm.ca_comm = custom
    comm.pynccl_comm = Mock(disabled=False)
    comm.pynccl_comm.all_reduce.side_effect = lambda inp: inp
    comm.qr_comm = comm.fi_ar_comm = comm.symm_mem_comm = None
    comm.device_group = None
    comm.all_reduce = Mock(wraps=comm.all_reduce)
    return comm


@pytest.mark.parametrize("tokens", range(1, 65))
@pytest.mark.parametrize("capturing", [False, True])
def test_sum2_uses_complete_ordinary_route_or_sum2_push(monkeypatch, tokens, capturing):
    custom = _custom(monkeypatch, capturing=capturing)
    comm = _cuda(custom)
    inp = torch.full((tokens, 2560), 0.1, dtype=torch.float16)
    other = torch.full_like(inp, 0.2)
    result = comm.all_reduce_sum2(inp, other)
    if capturing and tokens in (1, 4, 5, 8, 16):
        assert result is inp
        custom.all_reduce_sum2.assert_called_once_with(inp, other)
        custom.custom_all_reduce.assert_not_called()
        comm.pynccl_comm.all_reduce.assert_not_called()
        comm.all_reduce.assert_not_called()
    else:
        custom.all_reduce_sum2.assert_not_called()
        comm.all_reduce.assert_called_once()
        if capturing and tokens <= 16:
            custom.custom_all_reduce.assert_called_once()
            comm.pynccl_comm.all_reduce.assert_not_called()
        else:
            custom.custom_all_reduce.assert_not_called()
            comm.pynccl_comm.all_reduce.assert_called_once()
        assert torch.equal(result.view(torch.int16), (inp + other).view(torch.int16))


@pytest.mark.parametrize("tokens", [1, 2, 4, 5, 8, 16])
def test_direct_custom_sum2_rejects_nonpush(monkeypatch, tokens):
    custom = _custom(monkeypatch)
    custom.sm70_tp4_push_buffer_ptrs = None
    inp = torch.empty(tokens, 2560, dtype=torch.float16)
    assert custom.custom_all_reduce_sum2(inp, inp) is None
    custom.all_reduce_sum2.assert_not_called()


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("disabled", True),
        ("sm70_pcie_switch", False),
        ("world_size", 2),
        ("fully_connected", False),
        ("sm70_tp4_push_buffer_ptrs", None),
        ("_IS_CAPTURING", False),
    ],
)
def test_push_predicate_requires_native_preconditions(monkeypatch, attribute, value):
    custom = _custom(monkeypatch)
    setattr(custom, attribute, value)
    assert not custom.can_sm70_tp4_push_sum2(torch.empty(4, 2560, dtype=torch.float16))


def test_push_predicate_requires_active_capture(monkeypatch):
    custom = _custom(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not custom.can_sm70_tp4_push_sum2(torch.empty(4, 2560, dtype=torch.float16))


def test_failed_push_registration_retries_ordinary_all_reduce(monkeypatch):
    custom = _custom(monkeypatch)
    custom.custom_all_reduce_sum2 = Mock(return_value=None)
    comm = _cuda(custom)
    inp = torch.ones(4, 2560, dtype=torch.float16)
    assert torch.equal(comm.all_reduce_sum2(inp, inp), inp + inp)
    custom.custom_all_reduce.assert_called_once()
    comm.pynccl_comm.all_reduce.assert_not_called()
    comm.all_reduce.assert_called_once()


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 17, 32, 64])
def test_ordinary_all_reduce_keeps_its_policy(monkeypatch, tokens):
    custom = _custom(monkeypatch)
    custom.custom_all_reduce = Mock(side_effect=lambda inp: inp)
    custom.can_sm70_tp4_push_sum2 = Mock(side_effect=AssertionError("ordinary AR"))
    comm = _cuda(custom)
    monkeypatch.setattr(
        cuda_comm, "should_nccl_symm_mem_allreduce", lambda *args: False
    )
    inp = torch.ones(tokens, 2560, dtype=torch.float16)
    assert CudaCommunicator.all_reduce(comm, inp) is inp
    if tokens <= 16:
        custom.custom_all_reduce.assert_called_once_with(inp)
        comm.pynccl_comm.all_reduce.assert_not_called()
    else:
        custom.custom_all_reduce.assert_not_called()
        comm.pynccl_comm.all_reduce.assert_called_once_with(inp)


@pytest.mark.parametrize("switch", [False, True])
def test_other_dtypes_keep_legacy_custom_sum2(monkeypatch, switch):
    custom = _custom(monkeypatch, switch=switch)
    comm = _cuda(custom)
    inp = torch.ones(3, 2560, dtype=torch.float32)
    assert comm.all_reduce_sum2(inp, inp) is inp
    custom.all_reduce_sum2.assert_called_once_with(inp, inp)
    comm.all_reduce.assert_not_called()
    comm.pynccl_comm.all_reduce.assert_not_called()


def test_non_switch_custom_sum2_keeps_generic_pull(monkeypatch):
    custom = _custom(monkeypatch, switch=False)
    comm = _cuda(custom)
    inp = torch.ones(3, 2560, dtype=torch.float16)
    assert comm.all_reduce_sum2(inp, inp) is inp
    custom.all_reduce_sum2.assert_called_once_with(inp, inp)
    comm.pynccl_comm.all_reduce.assert_not_called()


@pytest.mark.parametrize("disabled,returns_none", [(True, False), (False, True)])
def test_sum2_pynccl_fallback_retains_torch_fallback(
    monkeypatch, disabled, returns_none
):
    custom = _custom(monkeypatch, capturing=False)
    comm = _cuda(custom)
    comm.pynccl_comm.disabled = disabled
    if returns_none:
        comm.pynccl_comm.all_reduce.side_effect = lambda inp: None
    fallback = Mock()
    monkeypatch.setattr(torch.distributed, "all_reduce", fallback)
    inp = torch.ones(3, 2560, dtype=torch.float16)
    result = comm.all_reduce_sum2(inp, inp)
    fallback.assert_called_once()
    assert torch.equal(result, inp + inp)
    comm.all_reduce.assert_called_once()
