# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Null rings must stay unwritable, including draft metadata rebuilds."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.qwen4_exp.common import qsa_cache
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu.spec_decode.eagle import speculator

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("physical_block", [-1, 0, 1, 7])
def test_circular_ring_rejects_null_block(physical_block):
    positions = torch.tensor([0, 3, 17, -1])
    slots = qsa_cache.circular_qsa_slot_mapping(
        torch.tensor([[physical_block]], dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        positions,
        compressor_state_size=16,
    )
    expected = (
        [physical_block * 16, physical_block * 16 + 3, physical_block * 16 + 1, -1]
        if physical_block > 0
        else [-1] * 4
    )
    assert slots.tolist() == expected


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="CPU kernel execution requires TRITON_INTERPRET=1",
)
def test_triton_ring_rejects_null_block(monkeypatch):
    monkeypatch.setattr(qsa_cache, "_metadata_launch_pdl", lambda: False)
    starts = torch.tensor([0, 3, 6], dtype=torch.int32)
    common = CommonAttentionMetadata(
        num_actual_tokens=6,
        num_reqs=2,
        max_query_len=3,
        max_seq_len=18,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([18, 18], dtype=torch.int32),
        slot_mapping=torch.arange(6, dtype=torch.int64),
        block_table_tensor=torch.tensor([[0], [7]], dtype=torch.int32),
    )
    _, positions, slots = qsa_cache.build_qsa_metadata_triton(
        common,
        torch.empty(6, dtype=torch.int32),
        torch.empty(6, dtype=torch.int64),
        torch.empty(6, dtype=torch.int64),
        storage_block_size=16,
        compress_ratio=1,
        circular_buffer_size=16,
    )
    assert positions.tolist() == [15, 16, 17, 15, 16, 17]
    assert slots.tolist() == [-1, -1, -1, 127, 112, 113]


@pytest.mark.parametrize("dummy", [False, True])
def test_draft_metadata_forwards_dummy_flag(monkeypatch, dummy):
    build = Mock(return_value={})
    monkeypatch.setattr(speculator, "build_attn_metadata", build)
    draft = SimpleNamespace(
        draft_attn_layer_names=["draft"],
        arange=torch.arange(3, dtype=torch.int32),
        block_tables=SimpleNamespace(
            input_block_tables=[torch.zeros((2, 1), dtype=torch.int32)],
            slot_mappings=torch.zeros((1, 2), dtype=torch.int64),
        ),
        attn_groups=[],
        input_buffers=SimpleNamespace(
            query_start_loc=torch.tensor([0, 1, 1], dtype=torch.int32),
            seq_lens=torch.tensor([1, 0], dtype=torch.int32),
        ),
        max_model_len=32,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    speculator.EagleSpeculator._build_draft_attn_metadata(
        draft, 1, 2, 2, is_dummy_batch=dummy
    )
    assert build.call_args.kwargs["is_dummy_batch"] is dummy


@pytest.mark.parametrize("dummy", [False, True])
def test_draft_steps_preserve_dummy_flag(monkeypatch, dummy):
    build = Mock(return_value={})
    monkeypatch.setattr(
        speculator, "build_slot_mappings_by_layer", Mock(return_value={})
    )
    draft = SimpleNamespace(
        num_speculative_steps=4,
        input_buffers=SimpleNamespace(
            positions=torch.zeros(2, dtype=torch.int64),
            query_start_loc=torch.tensor([0, 1, 1], dtype=torch.int32),
        ),
        idx_mapping=torch.tensor([0, 0], dtype=torch.int32),
        block_tables=SimpleNamespace(compute_slot_mappings=Mock()),
        kv_cache_config=SimpleNamespace(),
        _build_draft_attn_metadata=build,
        current_draft_step=torch.zeros(1, dtype=torch.int32),
        generate_draft=Mock(),
    )
    speculator.EagleSpeculator.multi_step_decode(
        draft,
        1,
        False,
        SimpleNamespace(num_reqs=2, num_tokens=2, cg_mode=CUDAGraphMode.NONE),
        None,
        is_dummy_batch=dummy,
    )
    assert build.call_count == 3
    assert all(call.kwargs["is_dummy_batch"] is dummy for call in build.call_args_list)
