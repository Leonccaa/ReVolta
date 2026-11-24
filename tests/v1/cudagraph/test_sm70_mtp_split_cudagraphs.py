# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.platforms.interface import DeviceCapability
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, ModelCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import (
    DecodeEagleCudaGraphManager,
    PrefillEagleCudaGraphManager,
)


def _make_config(max_num_seqs: int, verifier_sizes: list[int]):
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens_per_batch_size=None,
        num_speculative_state_tokens=lambda: 4,
    )
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            data_parallel_size=1,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=verifier_sizes.copy(),
            max_cudagraph_capture_size=max(verifier_sizes),
        ),
        speculative_config=speculative_config,
        adaptive_mtp_enabled=False,
    )


@pytest.mark.parametrize("capability", [(7, 0), (7, 5)], ids=["volta", "turing"])
@pytest.mark.parametrize(
    ("max_num_seqs", "request_sizes", "verifier_sizes"),
    [
        (1, [1], [1, 2, 4, 5, 8, 9, 18]),
        (16, [1, 2, 4, 6, 8, 12, 16], [5, 10, 20, 30, 40, 60, 80]),
        (32, [1, 2, 4, 6, 8, 12, 16, 32], [5, 10, 20, 30, 40, 60, 80, 160]),
    ],
)
def test_split_managers_keep_exact_full_graph_shapes(
    monkeypatch,
    max_num_seqs: int,
    request_sizes: list[int],
    verifier_sizes: list[int],
    capability: tuple[int, int],
):
    monkeypatch.setenv("VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS", "1")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.is_cuda",
        lambda: True,
    )
    # The gate asks the worker's own device; answer for it only.
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_device_capability",
        lambda device_id=0: DeviceCapability(*capability),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.torch.accelerator.current_device_index",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_global_graph_pool",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_pp_group",
        lambda: MagicMock(is_first_rank=True, is_last_rank=True),
    )

    config = _make_config(max_num_seqs, verifier_sizes)
    shared_sizes = config.compilation_config.cudagraph_capture_sizes.copy()
    shared_max_size = config.compilation_config.max_cudagraph_capture_size
    target = CudaGraphManager(
        config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=5,
    )
    draft = CudaGraphManager(
        config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=1,
    )

    assert target._capture_sizes == verifier_sizes
    assert draft._capture_sizes == request_sizes
    assert config.compilation_config.cudagraph_capture_sizes == shared_sizes
    assert config.compilation_config.max_cudagraph_capture_size == shared_max_size
    assert [
        (desc.num_reqs, desc.num_tokens, desc.uniform_token_count)
        for desc in target._capture_descs[CUDAGraphMode.FULL]
    ] == [(c, c * 5, 5) for c in reversed(request_sizes)]

    target._graphs_captured = True
    draft._graphs_captured = True
    for batch_size in range(1, max(request_sizes) + 1):
        padded = next(c for c in request_sizes if c >= batch_size)
        desc = target.dispatch(batch_size, batch_size * 5, 5)
        assert (desc.cg_mode, desc.num_reqs, desc.num_tokens) == (
            CUDAGraphMode.FULL,
            padded,
            padded * 5,
        )
    for batch_size in request_sizes:
        target_desc = target.dispatch(batch_size, batch_size * 5, 5)
        draft_desc = draft.dispatch(batch_size, batch_size, 1)
        assert target_desc.cg_mode == CUDAGraphMode.FULL
        assert target_desc.num_tokens == batch_size * 5
        assert target_desc.num_reqs == batch_size
        assert draft_desc.cg_mode == CUDAGraphMode.FULL
        assert draft_desc.num_tokens == batch_size
        assert draft_desc.num_reqs == batch_size


def test_qpn8_draft_prefill_above_128_rows_is_eager(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS", "0")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_global_graph_pool",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_pp_group",
        lambda: MagicMock(is_first_rank=True, is_last_rank=True),
    )
    device = torch.device("cpu")
    mode = CUDAGraphMode.FULL_DECODE_ONLY
    config = _make_config(160, [40, 80, 120, 160])
    target = CudaGraphManager(config, device, mode, decode_query_len=5)
    prefill = PrefillEagleCudaGraphManager(
        config, device, mode, decode_query_len=5, max_capture_tokens=128
    )
    decode = DecodeEagleCudaGraphManager(
        config, device, mode, decode_query_len=1, max_capture_tokens=128
    )
    for manager in (target, prefill, decode):
        manager._graphs_captured = True

    assert target.dispatch(32, 160, 5).cg_mode == CUDAGraphMode.FULL
    assert prefill.dispatch(24, 120, 5).cg_mode == CUDAGraphMode.FULL
    assert prefill.dispatch(26, 130, 5).cg_mode == CUDAGraphMode.NONE
    assert decode.dispatch(120, 120, 1).cg_mode == CUDAGraphMode.FULL
    assert decode.dispatch(129, 129, 1).cg_mode == CUDAGraphMode.NONE
    assert config.compilation_config.cudagraph_capture_sizes == [40, 80, 120, 160]


def test_dflash_lookup_captures_b1_q8_and_q16(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.current_platform.get_global_graph_pool",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_pp_group",
        lambda: MagicMock(is_first_rank=True, is_last_rank=True),
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            data_parallel_size=1,
        ),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[16],
            max_cudagraph_capture_size=16,
        ),
        speculative_config=SimpleNamespace(
            method="dflash",
            ngram_assist=True,
            num_speculative_tokens=15,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    dflash_config={"block_size": 8, "selector_top_k": 16}
                )
            ),
        ),
    )

    manager = CudaGraphManager(
        config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=16,
    )

    assert manager.decode_query_lens == (8, 16)
    assert manager._capture_sizes == [8, 16]
    manager._graphs_captured = True
    q8 = manager.dispatch(1, 8, 8)
    q16 = manager.dispatch(1, 16, 16)
    assert q8.cg_mode == CUDAGraphMode.FULL
    assert q8.uniform_token_count == 8
    assert q16.cg_mode == CUDAGraphMode.FULL
    assert q16.uniform_token_count == 16


@pytest.mark.parametrize("split", [False, True])
def test_dynamic_mtp_full_graph_shapes(monkeypatch, split):
    from vllm.v1.worker.gpu import cudagraph_utils as cg

    monkeypatch.setattr(cg, "_use_split_sm70_mtp_cudagraphs", lambda cfg: split)
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(cg, "get_pp_group", lambda: MagicMock())
    monkeypatch.setattr(
        "vllm.v1.attention.ops.sm70_e4m3_long.long_attention_enabled",
        lambda: False,
    )
    config = _make_config(8, [5, 10, 20, 40])
    config.speculative_config.num_speculative_tokens_per_batch_size = [
        (1, 3, 4),
        (4, 7, 3),
        (8, 8, 0),
    ]
    mode = CUDAGraphMode.FULL_AND_PIECEWISE
    device = torch.device("cpu")
    target = ModelCudaGraphManager(config, device, mode, 5)
    first = PrefillEagleCudaGraphManager(config, device, mode, 5)
    decode = DecodeEagleCudaGraphManager(
        config, device, CUDAGraphMode.FULL_DECODE_ONLY, 1
    )
    assert first._capture_descs == target._capture_descs
    target._graphs_captured = first._graphs_captured = True
    for c, q, padded in [
        (1, 5, 1),
        (1, 4, 1),
        (3, 4, 4),
        (4, 4, 4),
        (4, 5, 4),
        (5, 4, 8),
        (6, 4, 8),
        (7, 4, 8),
        (8, 1, 8),
        (3, 1, 4),
    ]:
        desc = target.dispatch(c, c * q, q)
        assert desc in target._capture_descs[CUDAGraphMode.FULL]
        assert (desc.num_reqs, desc.num_tokens, desc.uniform_token_count) == (
            padded,
            padded * q,
            q,
        )
        assert first.dispatch(c, desc.num_tokens, q) == desc
    for manager in (target, first):
        assert manager.dispatch(3, 15, 5).num_tokens == 20
        assert manager.dispatch(4, 16, None).cg_mode == CUDAGraphMode.PIECEWISE
    assert {
        desc.uniform_token_count for desc in decode._capture_descs[CUDAGraphMode.FULL]
    } == {1}
    assert config.compilation_config.cudagraph_capture_sizes == [5, 10, 20, 40]
